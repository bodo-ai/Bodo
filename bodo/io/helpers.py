# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains some IO related helpers.
"""

import os
import threading
import uuid

import numba
import pyarrow as pa
from mpi4py import MPI
from numba.core import types
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
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils import tracing
from bodo.utils.typing import BodoError, raise_bodo_error


class PyArrowTableSchemaType(types.Opaque):
    """Type for pyarrow schema object passed to C++. It is just a Python object passed
    as a pointer to C++ (this is of type pyarrow.lib.Schema)
    """

    def __init__(self):
        super(PyArrowTableSchemaType, self).__init__(name="PyArrowTableSchemaType")


pyarrow_table_schema_type = PyArrowTableSchemaType()
types.pyarrow_table_schema_type = pyarrow_table_schema_type  # type: ignore
register_model(PyArrowTableSchemaType)(models.OpaqueModel)


@unbox(PyArrowTableSchemaType)
def unbox_pyarrow_table_schema_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(PyArrowTableSchemaType)
def box_pyarrow_table_schema_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


@typeof_impl.register(pa.lib.Schema)
def typeof_pyarrow_table_schema(val, c):
    return pyarrow_table_schema_type


@lower_constant(PyArrowTableSchemaType)
def lower_pyarrow_table_schema(context, builder, ty, pyval):
    pyapi = context.get_python_api(builder)
    return pyapi.unserialize(pyapi.serialize_object(pyval))


def is_nullable(typ):
    return bodo.utils.utils.is_array_typ(typ, False) and (
        not isinstance(typ, types.Array)  # or is_dtype_nullable(typ.dtype)
        and not isinstance(typ, bodo.DatetimeArrayType)
    )


# Create an mpi4py reduction function.
def pa_schema_unify_reduction(schema_a, schema_b, unused):
    return pa.unify_schemas([schema_a, schema_b])


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
    pa.binary(): bytes_type,
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
        tz_val = bodo.libs.pd_datetime_arr_ext.get_pytz_type_info(tz_type)
        return bodo.DatetimeArrayType(tz_val), True
    else:
        # Without timezones Arrow ts arrays are converted to dt64 arrays.
        return types.Array(bodo.datetime64ns, 1, "C"), True


def _get_numba_typ_from_pa_typ(
    pa_typ: pa.Field,
    is_index,
    nullable_from_metadata,
    category_info,
    str_as_dict=False,
):
    """
    Return Bodo array type from pyarrow Field (column type) and if the type is supported.
    If a type is not support but can be adequately typed, we return that it isn't supported
    and later in compilation we will check if dead code/column elimination has successfully
    removed the column.
    """

    if isinstance(pa_typ.type, pa.ListType):
        # nullable_from_metadata is only used for non-nested Int arrays
        arr_typ, supported = _get_numba_typ_from_pa_typ(
            pa_typ.type.value_field, is_index, nullable_from_metadata, category_info
        )
        return ArrayItemArrayType(arr_typ), supported

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
        if pa_typ.type != pa.string():
            raise BodoError(f"Read as dictionary used for non-string column {pa_typ}")
        return dict_str_arr_type, True

    # Categorical data type
    # TODO: Use pa.types.is_dictionary? Same for other isinstances
    if isinstance(pa_typ.type, pa.DictionaryType):
        # NOTE: non-string categories seems not possible as of Arrow 4.0
        if pa_typ.type.value_type != pa.string():  # pragma: no cover
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
        arr_typ = boolean_array

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

    # TODO: uncomment once nullable float array is fully supported
    # if (
    #     _use_nullable_pd_arr
    #     and not is_index
    #     and isinstance(dtype, types.Float)
    #     and pa_typ.nullable
    # ):
    #     arr_typ = FloatingArrayType(dtype)

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
    # (TODO: time32, time64, ...)
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
        is_nullable(numba_type)
        or isinstance(numba_type, bodo.DatetimeArrayType)
        or (
            isinstance(numba_type, types.Array)
            and numba_type.dtype == bodo.datetime64ns
        )
    )


def _numba_to_pyarrow_type(numba_type: types.ArrayCompatible, is_iceberg: bool = False):
    """
    Convert Numba / Bodo Array Types to Equivalent PyArrow Type
    An additional flag `is_iceberg` is to handle the datetime type that must be
    converted to microseconds before writing to Iceberg tables.
    """
    if isinstance(numba_type, ArrayItemArrayType):
        # Set inner field name to 'element' so we can compare without worrying about
        # different names due to pyarrow ('item', 'element', 'field0', etc.)
        # Bodo List Arrays are always nullable (both the outer lists and inner elements)
        inner_elem = pa.field(
            "element", _numba_to_pyarrow_type(numba_type.dtype, is_iceberg)[0]
        )
        dtype = pa.list_(inner_elem)

    elif isinstance(numba_type, StructArrayType):
        fields = []
        for name, inner_type in zip(numba_type.names, numba_type.data):
            pa_type, _ = _numba_to_pyarrow_type(inner_type, is_iceberg)
            # We set nullable as true here to match the schema
            # written to parquet files, which doesn't contain
            # nullability info (and hence defaults to nullable).
            # This should be changed when we implement [BE-3247].
            fields.append(pa.field(name, pa_type, True))
        dtype = pa.struct(fields)

    elif isinstance(numba_type, DecimalArrayType):
        dtype = pa.decimal128(numba_type.precision, numba_type.scale)

    elif isinstance(numba_type, CategoricalArrayType):
        cat_dtype: PDCategoricalDtype = numba_type.dtype  # type: ignore
        dtype = pa.dictionary(
            _numba_to_pyarrow_type(cat_dtype.int_type, is_iceberg)[0],
            _numba_to_pyarrow_type(cat_dtype.elem_type, is_iceberg)[0],
            ordered=False if cat_dtype.ordered is None else cat_dtype.ordered,
        )

    elif numba_type == boolean_array:
        dtype = pa.bool_()
    elif numba_type in (string_array_type, bodo.dict_str_arr_type):
        dtype = pa.string()
    elif numba_type == binary_array_type:
        dtype = pa.binary()
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
        dtype = pa.timestamp("us", "UTC") if is_iceberg else pa.timestamp("ns", "UTC")

    elif (
        isinstance(numba_type, (types.Array, IntegerArrayType))
        and numba_type.dtype in _numba_pyarrow_type_map
    ):
        dtype = _numba_pyarrow_type_map[numba_type.dtype]  # type: ignore
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


# ----------------------------- Snowflake Write Helpers ----------------------------- #


def update_env_vars(env_vars):
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


def update_file_contents(fname: str, contents: str, is_parallel=True) -> str:
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
def uuid4_helper():
    """Helper function that enters objmode and calls uuid4 from JIT

    Returns
        out (str): String output of `uuid4()`
    """
    with numba.objmode(out="unicode_type"):
        out = str(uuid.uuid4())
    return out


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


# Register opaque type for bodo.io.helpers.ExceptionPropagatingThread so it
# can be shared between different sections of jitted code
class ExceptionPropagatingThreadType(types.Opaque):
    """Type for ExceptionPropagatingThread"""

    def __init__(self):
        super(ExceptionPropagatingThreadType, self).__init__(
            name="ExceptionPropagatingThreadType"
        )


exception_propagating_thread_type = ExceptionPropagatingThreadType()
types.exception_propagating_thread_type = exception_propagating_thread_type  # type: ignore
register_model(ExceptionPropagatingThreadType)(models.OpaqueModel)


@unbox(ExceptionPropagatingThreadType)
def unbox_exception_propagating_thread_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(ExceptionPropagatingThreadType)
def box_exception_propagating_thread_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


@typeof_impl.register(ExceptionPropagatingThread)
def typeof_exception_propagating_thread(val, c):
    return exception_propagating_thread_type


def join_all_threads(thread_list):
    """Given a list of threads, call `th.join()` on all threads in the list.

    Args
        thread_list (List(threading.Thread)): A list of threads to join
    """
    ev = tracing.Event("join_all_threads", is_parallel=True)

    comm = MPI.COMM_WORLD

    err = None
    try:
        for th in thread_list:
            if isinstance(th, threading.Thread):
                th.join()
    except Exception as e:
        err = e

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
        lowest_err = comm.bcast(lowest_err, root=failing_rank)

        # Each rank that already has an error will re-raise their own error, and
        # any rank that doesn't have an error will re-raise the lowest rank's error.
        if err_on_this_rank:
            raise err
        else:
            raise lowest_err

    ev.finalize()
