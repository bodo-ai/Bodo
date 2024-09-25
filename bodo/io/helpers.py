# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains some IO related helpers.
"""

import os
import threading
import uuid
from typing import TYPE_CHECKING

import numba
import numpy as np
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    models,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.hiframes.datetime_date_ext import (
    datetime_date_array_type,
    datetime_date_type,
)
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.time_ext import TimeArrayType, TimeType
from bodo.hiframes.timestamptz_ext import ArrowTimestampTZType
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.mpi4py import MPI
from bodo.utils.typing import (
    BodoError,
    is_nullable_ignore_sentinels,
    raise_bodo_error,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from numba.core import ir


class PyArrowTableSchemaType(types.Opaque):
    """Type for pyarrow schema object passed to C++. It is just a Python object passed
    as a pointer to C++ (this is of type pyarrow.lib.Schema)
    """

    def __init__(self):
        super(PyArrowTableSchemaType, self).__init__(name="PyArrowTableSchemaType")


pyarrow_schema_type = PyArrowTableSchemaType()
types.pyarrow_schema_type = pyarrow_schema_type  # type: ignore
register_model(PyArrowTableSchemaType)(models.OpaqueModel)


@unbox(PyArrowTableSchemaType)
def unbox_pyarrow_schema_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(PyArrowTableSchemaType)
def box_pyarrow_schema_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


@typeof_impl.register(pa.lib.Schema)
def typeof_pyarrow_table_schema(val, c):
    return pyarrow_schema_type


@lower_constant(PyArrowTableSchemaType)
def lower_pyarrow_table_schema(context, builder, ty, pyval):
    # TODO: Cant lower metadata in schema because we can't hash metadata
    # (represented as a dictionary)
    # We are currently removing metadata in the necessary spots
    # See _gen_pq_reader_py in parquet_ext.py
    # To do so here, add the following line:
    # pyval = pyval.remove_metadata()
    pyapi = context.get_python_api(builder)
    return pyapi.unserialize(pyapi.serialize_object(pyval))


# Create an mpi4py reduction function.
def pa_schema_unify_reduction(schema_a_and_row_count, schema_b_and_row_count, unused):
    # Attempt to unify the schemas, but if any schema is associated with a row
    # count of 0, disregard it.
    schema_a, count_a = schema_a_and_row_count
    schema_b, count_b = schema_b_and_row_count
    if count_a == 0 and count_b > 0:
        return (schema_b, count_b)
    if count_a > 0 and count_b == 0:
        return (schema_a, count_a)
    return (pa.unify_schemas([schema_a, schema_b]), count_a + count_b)


pa_schema_unify_mpi_op = MPI.Op.Create(pa_schema_unify_reduction, commute=True)


# Read Arrow Int/Float columns as nullable array (IntegerArrayType/FloatingArrayType)
use_nullable_pd_arr = True

_pyarrow_numba_type_map = {
    # boolean
    pa.bool_(): types.bool_,
    # signed int types
    pa.int8(): types.int8,
    pa.int16(): types.int16,
    pa.int32(): types.int32,
    pa.int64(): types.int64,
    # unsigned int types
    pa.uint8(): types.uint8,
    pa.uint16(): types.uint16,
    pa.uint32(): types.uint32,
    pa.uint64(): types.uint64,
    # float types (TODO: float16?)
    pa.float32(): types.float32,
    pa.float64(): types.float64,
    # String
    pa.string(): string_type,
    # The difference between pa.string and pa.large_string
    # is the int offset type, which is 32bit for string
    # and 64bit for large_string.
    # We use int64 in Bodo for strings, so
    # we can map both to string_type
    pa.large_string(): string_type,
    # The difference between pa.binary and pa.large_binary
    # is the int offset type, which is 32bit for binary
    # and 64bit for large_binary.
    # We use int64 in Bodo for binary, so
    # we can map both to bytes_type
    pa.binary(): bytes_type,
    pa.large_binary(): bytes_type,
    # date
    pa.date32(): datetime_date_type,
    pa.date64(): types.NPDatetime("ns"),
    # time
    pa.time32("s"): TimeType(0),
    pa.time32("ms"): TimeType(3),
    pa.time64("us"): TimeType(6),
    pa.time64("ns"): TimeType(9),
    # all null column
    pa.null(): string_type,  # map it to string_type, handle differently at runtime
    # Timestamp information is computed in get_arrow_timestamp_type,
    # so we don't store it in this dictionary.
}


def get_arrow_timestamp_type(pa_ts_typ):
    """
    Function used to determine the the proper Bodo type for various
    Arrow timestamp types. This generates different types depending
    on Timestamp values.

    Returns:
        - Bodo type
        - Is the timestamp type supported. This is False if a timezone
          or frequency cannot currently be supported.
    """
    supported_units = ["ns", "us", "ms", "s"]
    if pa_ts_typ.unit not in supported_units:
        # Unsupported units get typed as numpy dt64 array but
        # marked not supported.
        return types.Array(bodo.datetime64ns, 1, "C"), False
    elif pa_ts_typ.tz is not None:
        # Timezones use the PandasDatetimeArrayType. Timezone information
        # is stored in the Pandas type.
        # List of timezones comes from:
        # https://arrow.readthedocs.io/en/latest/index.html
        # https://www.iana.org/time-zones
        tz_type = pa_ts_typ.to_pandas_dtype().tz
        tz_val = bodo.libs.pd_datetime_arr_ext.get_tz_type_info(tz_type)
        return bodo.DatetimeArrayType(tz_val), True
    else:
        # Without timezones Arrow ts arrays are converted to dt64 arrays.
        return types.Array(bodo.datetime64ns, 1, "C"), True


def _get_numba_typ_from_pa_typ(
    pa_typ: pa.Field,
    is_index: bool,
    nullable_from_metadata,
    category_info,
    str_as_dict: bool = False,
) -> tuple[types.ArrayCompatible, bool]:
    """
    Return Bodo array type from pyarrow Field (column type) and if the type is supported.
    If a type is not support but can be adequately typed, we return that it isn't supported
    and later in compilation we will check if dead code/column elimination has successfully
    removed the column.
    """

    if (
        pa.types.is_list(pa_typ.type)
        or pa.types.is_large_list(pa_typ.type)
        or pa.types.is_fixed_size_list(pa_typ.type)
    ):
        # nullable_from_metadata is only used for non-nested Int arrays
        arr_typ, supported = _get_numba_typ_from_pa_typ(
            pa_typ.type.value_field, is_index, nullable_from_metadata, category_info
        )
        return ArrayItemArrayType(arr_typ), supported

    if pa.types.is_map(pa_typ.type):
        key_type, key_sup = _get_numba_typ_from_pa_typ(
            pa_typ.type.key_field, is_index, nullable_from_metadata, category_info
        )
        value_type, value_sup = _get_numba_typ_from_pa_typ(
            pa_typ.type.item_field, is_index, nullable_from_metadata, category_info
        )
        return MapArrayType(key_type, value_type), key_sup and value_sup

    if isinstance(pa_typ.type, pa.StructType):
        child_types = []
        field_names = []
        supported = True
        for field in pa_typ.flatten():
            field_names.append(field.name.split(".")[-1])
            child_arr, child_supported = _get_numba_typ_from_pa_typ(
                field, is_index, nullable_from_metadata, category_info
            )
            child_types.append(child_arr)
            supported = supported and child_supported
        return StructArrayType(tuple(child_types), tuple(field_names)), supported

    # Decimal128Array type
    if isinstance(pa_typ.type, pa.Decimal128Type):
        return DecimalArrayType(pa_typ.type.precision, pa_typ.type.scale), True

    if str_as_dict:
        if pa_typ.type != pa.string() and pa_typ.type != pa.large_string():
            raise BodoError(f"Read as dictionary used for non-string column {pa_typ}")
        return dict_str_arr_type, True

    # Categorical data type
    # TODO: Use pa.types.is_dictionary? Same for other isinstances
    if isinstance(pa_typ.type, pa.DictionaryType):
        # NOTE: non-string categories seems not possible as of Arrow 4.0
        if (
            pa_typ.type.value_type != pa.string()
            and pa_typ.type.value_type != pa.large_string()
        ):  # pragma: no cover
            raise BodoError(
                f"Parquet Categorical data type should be string, not {pa_typ.type.value_type}"
            )
        # data type for storing codes
        int_type = _pyarrow_numba_type_map[pa_typ.type.index_type]
        cat_dtype = PDCategoricalDtype(
            category_info[pa_typ.name],
            bodo.string_type,
            pa_typ.type.ordered,
            int_type=int_type,
        )
        return CategoricalArrayType(cat_dtype), True

    if isinstance(pa_typ.type, pa.lib.TimestampType):
        return get_arrow_timestamp_type(pa_typ.type)
    elif isinstance(pa_typ.type, ArrowTimestampTZType):
        return bodo.timestamptz_array_type, True
    elif pa_typ.type in _pyarrow_numba_type_map:
        dtype = _pyarrow_numba_type_map[pa_typ.type]
        supported = True
    else:
        raise BodoError("Arrow data type {} not supported yet".format(pa_typ.type))

    if dtype == datetime_date_type:
        return datetime_date_array_type, supported

    if isinstance(dtype, TimeType):
        return TimeArrayType(dtype.precision), supported

    if dtype == bytes_type:
        return binary_array_type, supported

    arr_typ = string_array_type if dtype == string_type else types.Array(dtype, 1, "C")

    if dtype == types.bool_:
        arr_typ = boolean_array_type

    # Do what metadata says or use global defualt
    _use_nullable_pd_arr = (
        use_nullable_pd_arr
        if nullable_from_metadata is None
        else nullable_from_metadata
    )

    # TODO: support nullable int for indices
    if (
        _use_nullable_pd_arr
        and not is_index
        and isinstance(dtype, types.Integer)
        and pa_typ.nullable
    ):
        arr_typ = IntegerArrayType(dtype)

    if (
        _use_nullable_pd_arr
        and not is_index
        and isinstance(dtype, types.Float)
        and pa_typ.nullable
    ):
        arr_typ = FloatingArrayType(dtype)

    return arr_typ, supported


_numba_pyarrow_type_map = {
    types.bool_: pa.bool_(),
    # Signed Int Types
    types.int8: pa.int8(),
    types.int16: pa.int16(),
    types.int32: pa.int32(),
    types.int64: pa.int64(),
    # Unsigned Int Types
    types.uint8: pa.uint8(),
    types.uint16: pa.uint16(),
    types.uint32: pa.uint32(),
    types.uint64: pa.uint64(),
    # Float Types (TODO: float16?)
    types.float32: pa.float32(),
    types.float64: pa.float64(),
    # Date and Time
    types.NPDatetime("ns"): pa.date64(),
}


def is_nullable_arrow_out(numba_type: types.ArrayCompatible) -> bool:
    """
    Does this Array type produce an Arrow array with nulls when converted to C++
    This is more expansive than is_nullable since the original array may not have
    nulls but other values will be translated to nulls when converting to Arrow
    As of now, datetime arrays store NaTs instead of nulls, which are then
    translated to nulls in our Arrow conversion code
    """

    return (
        is_nullable_ignore_sentinels(numba_type)
        or isinstance(numba_type, bodo.DatetimeArrayType)
        or (
            isinstance(numba_type, types.Array)
            and numba_type.dtype == bodo.datetime64ns
        )
    )


def _numba_to_pyarrow_type(
    numba_type: types.ArrayCompatible,
    is_iceberg: bool = False,
    use_dict_arr: bool = False,
):
    """
    Convert Numba / Bodo Array Types to Equivalent PyArrow Type
    An additional flag `is_iceberg` is to handle the datetime type that must be
    converted to microseconds before writing to Iceberg tables.
    """
    from bodo.libs.array import TUPLE_ARRAY_SENTINEL

    if isinstance(numba_type, ArrayItemArrayType):
        # Set inner field name to 'element' so we can compare without worrying about
        # different names due to pyarrow ('item', 'element', 'field0', etc.)
        # Bodo List Arrays are always nullable (both the outer lists and inner elements)
        inner_elem = pa.field(
            "element",
            _numba_to_pyarrow_type(numba_type.dtype, is_iceberg, use_dict_arr)[0],
        )
        dtype = pa.large_list(inner_elem)

    elif isinstance(numba_type, StructArrayType):
        fields = []
        for name, inner_type in zip(numba_type.names, numba_type.data):
            pa_type, _ = _numba_to_pyarrow_type(inner_type, is_iceberg, use_dict_arr)
            # We set nullable as true here to match the schema
            # written to parquet files, which doesn't contain
            # nullability info (and hence defaults to nullable).
            # This should be changed when we implement [BE-3247].
            fields.append(pa.field(name, pa_type, True))
        dtype = pa.struct(fields)

    elif isinstance(numba_type, bodo.TupleArrayType):
        fields = []
        for i, inner_type in enumerate(numba_type.data):
            pa_type, _ = _numba_to_pyarrow_type(inner_type, is_iceberg, use_dict_arr)
            fields.append(pa.field(f"{TUPLE_ARRAY_SENTINEL}{i}", pa_type, True))
        dtype = pa.struct(fields)

    elif isinstance(numba_type, bodo.MapArrayType):
        key_type, _ = _numba_to_pyarrow_type(
            numba_type.key_arr_type, is_iceberg, use_dict_arr
        )
        item_type, _ = _numba_to_pyarrow_type(
            numba_type.value_arr_type, is_iceberg, use_dict_arr
        )
        dtype = pa.map_(key_type, item_type)

    elif isinstance(numba_type, DecimalArrayType):
        dtype = pa.decimal128(numba_type.precision, numba_type.scale)

    elif isinstance(numba_type, CategoricalArrayType):
        cat_dtype: PDCategoricalDtype = numba_type.dtype  # type: ignore
        dtype = pa.dictionary(
            _numba_to_pyarrow_type(cat_dtype.int_type, is_iceberg, use_dict_arr)[0],
            _numba_to_pyarrow_type(cat_dtype.elem_type, is_iceberg, use_dict_arr)[0],
            ordered=False if cat_dtype.ordered is None else cat_dtype.ordered,
        )

    elif numba_type == boolean_array_type:
        dtype = pa.bool_()
    elif use_dict_arr and numba_type == bodo.dict_str_arr_type:
        dtype = pa.dictionary(pa.int32(), pa.large_string())
    elif numba_type in (string_array_type, bodo.dict_str_arr_type):
        dtype = pa.large_string()
    elif numba_type == binary_array_type:
        dtype = pa.large_binary()
    elif numba_type == datetime_date_array_type:
        dtype = pa.date32()
    elif isinstance(numba_type, bodo.DatetimeArrayType) or (
        isinstance(numba_type, types.Array) and numba_type.dtype == bodo.datetime64ns
    ):
        # For Iceberg, all timestamp data needs to be written
        # as microseconds, so that's the type we
        # specify. We convert our nanoseconds to
        # microseconds during write.
        # See https://iceberg.apache.org/spec/#primitive-types,
        # https://iceberg.apache.org/spec/#parquet
        # We've also made the decision to always
        # write the `timestamptz` type when writing
        # Iceberg data, similar to Spark.
        # The underlying already is in UTC already
        # for timezone aware types, and for timezone
        # naive, it won't matter.
        tz = numba_type.tz if isinstance(numba_type, bodo.DatetimeArrayType) else None
        dtype = pa.timestamp("us", "UTC") if is_iceberg else pa.timestamp("ns", tz)

    # TODO: Figure out how to raise an error here for Iceberg (is_iceberg is set to True).
    elif isinstance(numba_type, types.Array) and numba_type.dtype == bodo.timedelta64ns:
        dtype = pa.duration("ns")
    elif (
        isinstance(
            numba_type,
            (types.Array, IntegerArrayType, FloatingArrayType, bodo.PrimitiveArrayType),
        )
        and numba_type.dtype in _numba_pyarrow_type_map
    ):
        dtype = _numba_pyarrow_type_map[numba_type.dtype]  # type: ignore
    elif isinstance(numba_type, bodo.TimeArrayType):
        if numba_type.precision == 0:
            dtype = pa.time32("s")
        elif numba_type.precision == 3:
            dtype = pa.time32("ms")
        elif numba_type.precision == 6:
            dtype = pa.time64("us")
        elif numba_type.precision == 9:
            dtype = pa.time64("ns")
    elif numba_type == bodo.null_array_type:
        dtype = pa.null()
    else:
        raise BodoError(
            f"Conversion from Bodo array type {numba_type} to PyArrow type not supported yet"
        )

    return dtype, is_nullable_arrow_out(numba_type)


def numba_to_pyarrow_schema(df: DataFrameType, is_iceberg: bool = False) -> pa.Schema:
    """Construct a PyArrow Schema from Bodo's DataFrame Type"""
    fields = []
    for name, col_type in zip(df.columns, df.data):
        try:
            pyarrow_type, nullable = _numba_to_pyarrow_type(col_type, is_iceberg)
        except BodoError as e:
            raise_bodo_error(e.msg, e.loc)

        fields.append(pa.field(name, pyarrow_type, nullable))
    return pa.schema(fields)


def pyarrow_type_to_numba(arrow_type):
    """Convert PyArrow data type to Bodo array Numba type

    Args:
        arrow_type (pyarrow.lib.DataType): PyArrow array type

    Returns:
        types.Type: Bodo array type
    """

    if (
        pa.types.is_large_list(arrow_type)
        or pa.types.is_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        return ArrayItemArrayType(pyarrow_type_to_numba(arrow_type.value_type))

    if pa.types.is_struct(arrow_type):
        data_arrays = []
        names = []
        for i in range(arrow_type.num_fields):
            field = arrow_type.field(i)
            data_arrays.append(pyarrow_type_to_numba(field.type))
            names.append(field.name)
        return StructArrayType(tuple(data_arrays), tuple(names))

    if pa.types.is_map(arrow_type):
        return MapArrayType(
            pyarrow_type_to_numba(arrow_type.key_type),
            pyarrow_type_to_numba(arrow_type.item_type),
        )

    if pa.types.is_integer(arrow_type):
        return IntegerArrayType(
            types.Integer.from_bitwidth(
                arrow_type.bit_width, pa.types.is_signed_integer(arrow_type)
            )
        )

    if pa.types.is_floating(arrow_type):
        if pa.types.is_float64(arrow_type):
            dtype = types.float64
        elif pa.types.is_float32(arrow_type):
            dtype = types.float32
        else:
            raise BodoError(f"Unsupported Arrow float type {arrow_type}")
        return FloatingArrayType(dtype)

    if pa.types.is_boolean(arrow_type):
        return boolean_array_type

    if pa.types.is_date32(arrow_type):
        return datetime_date_array_type

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return string_array_type

    if (
        pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_fixed_size_binary(arrow_type)
    ):
        return binary_array_type

    if (
        pa.types.is_dictionary(arrow_type)
        and (
            pa.types.is_string(arrow_type.value_type)
            or pa.types.is_large_string(arrow_type.value_type)
        )
        and pa.types.is_int32(arrow_type.index_type)
    ):
        return dict_str_arr_type

    if pa.types.is_decimal128(arrow_type):
        return DecimalArrayType(arrow_type.precision, arrow_type.scale)

    if pa.types.is_timestamp(arrow_type):
        return bodo.DatetimeArrayType(arrow_type.tz)

    if pa.types.is_null(arrow_type):
        return bodo.null_array_type

    if pa.types.is_time64(arrow_type):
        precision = 9 if arrow_type.unit == "ns" else 6
        return bodo.TimeArrayType(precision)

    if pa.types.is_time32(arrow_type):
        precision = 3 if arrow_type.unit == "ms" else 0
        return bodo.TimeArrayType(precision)

    raise BodoError(
        f"Conversion from PyArrow type {arrow_type} to Bodo array type not supported yet"
    )


# ---------------------------- Compilation Time Helpers ----------------------------- #
def map_cpp_to_py_table_column_idxs(
    col_names: list[str], out_used_cols: list[int]
) -> "npt.NDArray[np.int64]":
    """
    Compilation-time helper that maps the index / location of each
    column in the table type to its 'physical index' in the C++ table
    during IO. Columns that are pruned are assigned index -1.

    For Snowflake reads, it is generally true (but not enforced)
    that the order of columns in the C++ table is maintained in the
    table type. However, the table type will still retain typing info
    for pruned columns, which would not have any pointers for in C++
    """
    idx = []
    j = 0
    for i in range(len(col_names)):
        if j < len(out_used_cols) and i == out_used_cols[j]:
            idx.append(j)
            j += 1
        else:
            idx.append(-1)
    return np.array(idx, dtype=np.int64)


# ----------------------------- Snowflake Write Helpers ----------------------------- #


def update_env_vars(env_vars):  # pragma: no cover
    """Update the current environment variables with key-value pairs provided
    in a dictionary. Used in bodo.io.snowflake. "__none__" is used as a dummy
    value since Numba hates dictionaries with strings and NoneType's as values.

    Args
        env_vars (Dict(str, str or None)): A dictionary of environment variables to set.
            A value of "__none__" indicates a variable should be removed.

    Returns
        old_env_vars (Dict(str, str or None)): Previous value of any overwritten
            environment variables. A value of "__none__" indicates an environment
            variable was previously unset.
    """
    old_env_vars = {}
    for k, v in env_vars.items():
        if k in os.environ:
            old_env_vars[k] = os.environ[k]
        else:
            old_env_vars[k] = "__none__"

        if v == "__none__":
            del os.environ[k]
        else:
            os.environ[k] = v

    return old_env_vars


def update_file_contents(
    fname: str, contents: str, is_parallel=True
) -> str:  # pragma: no cover
    """
    Similar to update_env_vars, except here we will update the contents
    of a file and return the original contents if there are any.
    If the file didn't originally exist, we return "__none__",
    so that when the function is called back to restore the original
    contents, we can remove the file instead.
    We use "__none__" instead of `None` for type stability reasons when
    passing the output between JIT and regular Python.

    Args:
        fname (str): filename to update contents of
        contents (str): content to write to the file. In case this is
            "__none__", the file is removed.
        is_parallel (bool, optional): Whether or not the operation
            should be done in parallel. In case of is_parallel=True,
            the filesystem operations are only done on the first rank
            of every node, to avoid filesystem contention.
            Defaults to True.

    Returns:
        str: Original contents of the file. Returns "__none__"
            in case the file doesn't exist.
    """
    comm = MPI.COMM_WORLD

    old_content = None
    if (not is_parallel) or (comm.Get_rank() == 0):
        if os.path.exists(fname):
            # If the file does exist, get
            # its contents
            with open(fname, "r") as f:
                old_content = f.read()
    if is_parallel:
        old_content = comm.bcast(old_content)

    if old_content is None:
        # If the file didn't originally exist,
        # we use "__none__" as the identifier
        # so we can delete it later when
        # the function is called with
        # contents = "__none__"
        old_content = "__none__"

    # If parallel, choose the first rank on each node as the active
    # rank for performing filesystem operations. If not parallel,
    # all ranks are active ranks.
    active_rank = (
        (bodo.get_rank() in bodo.get_nodes_first_ranks()) if is_parallel else True
    )
    # As explained above, if  contents == "__none__",
    # then remove the file if it exists
    if contents == "__none__":
        if active_rank and os.path.exists(fname):
            os.remove(fname)
    else:
        # Else, restore the contents
        if active_rank:
            with open(fname, "w") as f:
                f.write(contents)
    if is_parallel:
        comm.Barrier()
    return old_content


@numba.njit
def uuid4_helper():  # pragma: no cover
    """Helper function that enters objmode and calls uuid4 from JIT

    Returns
        out (str): String output of `uuid4()`
    """
    with bodo.no_warning_objmode(out="unicode_type"):
        out = str(uuid.uuid4())
    return out


@numba.njit
def makedirs_helper(path, exist_ok=False):  # pragma: no cover
    """Helper function that enters objmode and calls os.makedirs from JIT
    This is intended to be called from all ranks at once within streaming
    Snowflake write, with a distinct `path` per rank.

    Args:
        path (str): Path of directory to create
        exist_ok (bool): If True, nothing happens if the directory already
            exists. If False, raise an exception if the directory exists.

    """
    with bodo.no_warning_objmode():
        os.makedirs(path, exist_ok=exist_ok)


class ExceptionPropagatingThread(threading.Thread):
    """A threading.Thread that propagates exceptions to the main thread.
    Derived from https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
    """

    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def get_table_iterator(rhs: "ir.Inst", func_ir: "ir.FunctionIR") -> str:
    """pattern match "table, is_last = read_arrow_next(reader)" and return the reader
    variable name.

    Args:
        rhs (ir.Inst): righthand side of assignemnt to table variable.
        func_ir (ir.FunctionIR): Function IR used for finding definitions and calls.

    Returns:
        str: Variable name of the ArrowReaderType var used in read_arrow_next

    Raises:
        GuardException if unable to find the pattern or varname
    """
    from numba.core import ir
    from numba.core.ir_utils import find_callname, get_definition, require

    from bodo.utils.utils import is_call

    # example IR:
    # $24load_global.0 = global(read_arrow_next:
    #       CPUDispatcher(<function read_arrow_next at 0x42f6543e20>))
    # $28call_function.2 = call $24load_global.0(arrow_iterator.18,
    #       func=$24load_global.0, args=[Var(arrow_iterator.18, arrow.py:22)], kws=(),
    #       vararg=None, varkwarg=None, target=None)
    # $30unpack_sequence.5 = exhaust_iter(value=$28call_function.2, count=2)
    # $table.167 = static_getitem(value=$30unpack_sequence.5, index=0, index_var=None,
    #       fn=<built-in function getitem>)

    require(isinstance(rhs, ir.Expr) and rhs.op == "static_getitem" and rhs.index == 0)
    exhaust_iter = get_definition(func_ir, rhs.value)
    require(isinstance(exhaust_iter, ir.Expr) and exhaust_iter.op == "exhaust_iter")
    tup_def = get_definition(func_ir, exhaust_iter.value)
    require(
        is_call(tup_def)
        and find_callname(func_ir, tup_def)
        == ("read_arrow_next", "bodo.io.arrow_reader")
    )
    return tup_def.args[0].name


def sync_and_reraise_error(
    err,
    _is_parallel=False,
    bcast_lowest_err: bool = True,
    default_generic_err_msg: str | None = None,
):  # pragma: no cover
    """
    If `err` is an Exception on any rank, raise an error on all ranks.
    If 'bcast_lowest_err' is True, we will broadcast the error from the
    "lowest" rank that has an error and raise it on all the ranks without
    their own error. If 'bcast_lowest_err' is False, we will raise a
    generic error on ranks without their own error. This is useful in
    cases where the error could be something that's not safe to broadcast
    (e.g. not pickle-able).
    This is a no-op if all ranks are exception-free.

    Args:
        err (Exception or None): Could be None or an exception
        _is_parallel (bool): Whether this is being called from many ranks
        bcast_lowest_err (bool): Whether to broadcast the error from the
            lowest rank. Only applicable in the _is_parallel case.
        default_generic_err_msg (str, optional): If bcast_lowest_err = False,
            this message will be used for the exception raised on
            ranks without their own error. Only applicable in the
            _is_parallel case.
    """
    comm = MPI.COMM_WORLD

    if _is_parallel:
        # If any rank raises an exception, re-raise that error on all non-failing
        # ranks to prevent deadlock on future MPI collective ops.
        # We use allreduce with MPI.MAXLOC to communicate the rank of the lowest
        # failing process, then broadcast the error backtrace across all ranks.
        err_on_this_rank = int(err is not None)
        err_on_any_rank, failing_rank = comm.allreduce(
            (err_on_this_rank, comm.Get_rank()), op=MPI.MAXLOC
        )
        if err_on_any_rank:
            if comm.Get_rank() == failing_rank:
                lowest_err = err
            else:
                lowest_err = None
            if bcast_lowest_err:
                lowest_err = comm.bcast(lowest_err, root=failing_rank)
            else:
                err_msg = (
                    default_generic_err_msg
                    if (default_generic_err_msg is not None)
                    else "Exception on some ranks. See other ranks for error."
                )
                lowest_err = Exception(err_msg)

            # Each rank that already has an error will re-raise their own error, and
            # any rank that doesn't have an error will re-raise the lowest rank's error.
            if err_on_this_rank:
                raise err
            else:
                raise lowest_err
    else:
        if err is not None:
            raise err


def _get_stream_writer_payload(
    context, builder, writer_typ, payload_type, writer
):  # pragma: no cover
    """Get payload struct proxy for a stream writer value (Snowflake or Iceberg)"""
    stream_writer = context.make_helper(builder, writer_typ, writer)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, stream_writer.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload, meminfo_data_ptr


def define_stream_writer_dtor(
    context, builder, payload_type, writer_payload_members
):  # pragma: no cover
    """
    Define destructor for stream writer type if not already defined
    (Snowflake or Iceberg stream writer)
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    # Ensure the name depends on the payload type because each streaming write may
    # have a different struct and we need to generate 1 dtor per struct.
    fn = cgutils.get_or_insert_function(
        mod, fnty, name=f".dtor.stream_writer.{payload_type}"
    )

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # Get payload struct
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    # Decref each payload field
    for attr, fe_type in writer_payload_members:
        if fe_type == bodo.libs.distributed_api.is_last_state_type:
            # Delete is_last sync state if writer has it (Parquet and Iceberg)
            c_fnty = lir.FunctionType(
                lir.VoidType(),
                [lir.IntType(8).as_pointer()],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, c_fnty, name="delete_is_last_state"
            )
            builder.call(fn_tp, [payload.is_last_state])
        context.nrt.decref(builder, fe_type, getattr(payload, attr))

    # Delete table builder state
    c_fnty = lir.FunctionType(
        lir.VoidType(),
        [lir.IntType(8).as_pointer()],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, c_fnty, name="delete_table_builder_state"
    )
    builder.call(fn_tp, [payload.batches])

    builder.ret_void()
    return fn


def stream_writer_alloc_codegen(
    context,
    builder,
    stream_writer_payload_type,
    stream_writer_type,
    stream_writer_payload_members,
):
    """Codegen for stream writer allocation intrinsics (Snowflake or Iceberg)"""
    # Create payload type
    payload_type = stream_writer_payload_type
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # Define dtor
    dtor_fn = define_stream_writer_dtor(
        context,
        builder,
        payload_type,
        stream_writer_payload_members,
    )

    # Create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # Alloc values in payload. Note: garbage values will be stored in all
    # fields until writer_setattr is called for the first time
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    builder.store(payload._getvalue(), meminfo_data_ptr)

    # Construct stream writer from payload
    stream_writer = context.make_helper(builder, stream_writer_type)
    stream_writer.meminfo = meminfo
    return stream_writer._getvalue()


def is_pyarrow_list_type(arrow_type):
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    )
