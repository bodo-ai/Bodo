"""
Boxing and unboxing support for DataFrame, Series, etc.
"""

import datetime
import decimal
import warnings
from enum import Enum

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.ir_utils import GuardException, guard
from numba.core.typing import signature
from numba.cpython.listobj import ListInstance
from numba.extending import NativeValue, box, intrinsic, typeof_impl, unbox
from numba.np.arrayobj import _getitem_array_single_int
from numba.typed.typeddict import Dict

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype
from bodo.hiframes.pd_dataframe_ext import (
    DataFramePayloadType,
    DataFrameType,
    check_runtime_cols_unsupported,
    construct_dataframe,
)
from bodo.hiframes.pd_index_ext import (
    BinaryIndexType,
    CategoricalIndexType,
    DatetimeIndexType,
    NumericIndexType,
    PeriodIndexType,
    RangeIndexType,
    StringIndexType,
    TimedeltaIndexType,
)
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.hiframes.time_ext import TimeArrayType, TimeType
from bodo.hiframes.timestamptz_ext import timestamptz_type
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.float_arr_ext import FloatDtype, FloatingArrayType
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.null_arr_ext import null_array_type
from bodo.libs.pd_datetime_arr_ext import PandasDatetimeTZDtype
from bodo.libs.str_arr_ext import string_array_type, string_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    dtype_to_array_type,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    raise_bodo_error,
    to_nullable_type,
)

# the number of dataframe columns above which we use table format in unboxing
TABLE_FORMAT_THRESHOLD = 0


# Unbox dataframe columns eagerly, which improves compilation time by disabling lazy
# unboxing calls (only for table format since non-table format is deprecated).
UNBOX_DATAFRAME_EAGERLY = True


# A flag to use dictionary-encode string arrays for all string arrays
# Used for testing purposes
_use_dict_str_type = False
# A flag for using StructArrays for dictionaries under n elements
# Modified for testing purposes, to force input argument type
struct_size_limit = 100


# Wrapper class around str to make typing treat str value as dictionary-encoded string
# array element not a regular string.
class DictStringSentinel(str):
    pass


def _set_bodo_meta_in_pandas():
    """
    Avoid pandas warnings for Bodo metadata setattr in boxing of Series/DataFrame.
    Has to run in import instead of somewhere in the compiler pipeline since user
    function may be loaded from cache.
    """
    if "_bodo_meta" not in pd.Series._metadata:
        pd.Series._metadata.append("_bodo_meta")

    if "_bodo_meta" not in pd.DataFrame._metadata:
        pd.DataFrame._metadata.append("_bodo_meta")


_set_bodo_meta_in_pandas()


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val: pd.DataFrame, c):
    from bodo.transforms.distributed_analysis import Distribution

    # convert "columns" from Index/MultiIndex to a tuple
    col_names = tuple(val.columns.to_list())
    col_types = get_hiframes_dtypes(val)
    if (
        len(val.index) == 0
        and val.index.dtype == np.dtype("O")
        and hasattr(val, "_bodo_meta")
        and val._bodo_meta is not None
        and "type_metadata" in val._bodo_meta
        and val._bodo_meta["type_metadata"] is not None
        # If the metadata hasn't updated but columns are added the information
        # is out of date and cannot be used.
        and len(val._bodo_meta["type_metadata"][1]) == len(val.columns)
        and val._bodo_meta["type_metadata"][0] is not None
    ):
        index_typ = _dtype_from_type_enum_list(val._bodo_meta["type_metadata"][0])
    else:
        index_typ = numba.typeof(val.index)

    # set distribution from Bodo metadata of df object if available
    # using REP as default to be safe in distributed analysis
    dist = (
        Distribution(val._bodo_meta["dist"])
        # check for None since df.copy() assigns None to DataFrame._metadata attributes
        # for some reason
        if hasattr(val, "_bodo_meta") and val._bodo_meta is not None
        else Distribution.REP
    )

    # TODO: enable table format by default
    use_table_format = len(col_types) >= TABLE_FORMAT_THRESHOLD
    return DataFrameType(
        col_types, index_typ, col_names, dist, is_table_format=use_table_format
    )


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_series(val: pd.Series, c):
    from bodo.transforms.distributed_analysis import Distribution

    dist = (
        Distribution(val._bodo_meta["dist"])
        if hasattr(val, "_bodo_meta") and val._bodo_meta is not None
        else Distribution.REP
    )
    if (
        len(val.index) == 0
        and val.index.dtype == np.dtype("O")
        and hasattr(val, "_bodo_meta")
        and val._bodo_meta is not None
        and "type_metadata" in val._bodo_meta
        and val._bodo_meta["type_metadata"] is not None
        and val._bodo_meta["type_metadata"][0] is not None
    ):  # pragma: no cover
        # pragma is needed here, as we will never enter this case without np > 1
        idx_typ = _dtype_from_type_enum_list(val._bodo_meta["type_metadata"][0])
    else:
        idx_typ = numba.typeof(val.index)

    arr_typ = _infer_series_arr_type(val)
    # use dictionary-encoded array if necessary for testing (_use_dict_str_type set)
    if _use_dict_str_type and arr_typ == string_array_type:
        arr_typ = bodo.types.dict_str_arr_type

    return SeriesType(
        arr_typ.dtype,
        data=arr_typ,
        index=idx_typ,
        name_typ=numba.typeof(val.name),
        dist=dist,
    )


@unbox(DataFrameType)
def unbox_dataframe(typ, val, c):
    """unbox dataframe to an empty DataFrame struct
    columns will be extracted later if necessary.
    """
    check_runtime_cols_unsupported(typ, "Unboxing")

    # unbox index
    # TODO: unbox index only if necessary
    ind_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, ind_obj).value
    c.pyapi.decref(ind_obj)

    # set data arrays as null due to lazy unboxing
    if typ.is_table_format:
        table = cgutils.create_struct_proxy(typ.table_type)(c.context, c.builder)

        # TODO(ehsan): do we need to incref/decref the parent object? JIT args will be
        # available while JIT function is running.
        # Numba's list object doesn't incref/decref its parent for example
        table.parent = val

        # create array list for each block (but don't unbox yet)
        for t, blk in typ.table_type.type_to_blk.items():
            n_arrs = c.context.get_constant(
                types.int64, len(typ.table_type.block_to_arr_ind[blk])
            )
            # not using allocate() since its exception causes calling convention error
            # NOTE: numba initializes the value data array to zero, which we use for
            # null check in table boxing
            _, out_arr_list = ListInstance.allocate_ex(
                c.context, c.builder, types.List(t), n_arrs
            )
            out_arr_list.size = n_arrs

            if UNBOX_DATAFRAME_EAGERLY:
                # lower array of array indices for block to use within the loop
                # using array since list doesn't have constant lowering
                arr_inds = c.context.make_constant_array(
                    c.builder,
                    types.Array(types.int64, 1, "C"),
                    np.array(typ.table_type.block_to_arr_ind[blk], dtype=np.int64),
                )
                arr_inds_struct = c.context.make_array(
                    types.Array(types.int64, 1, "C")
                )(c.context, c.builder, arr_inds)
                with cgutils.for_range(c.builder, n_arrs) as loop:
                    i = loop.index
                    # get array index in dataframe columns and unbox array
                    arr_ind = _getitem_array_single_int(
                        c.context,
                        c.builder,
                        types.int64,
                        types.Array(types.int64, 1, "C"),
                        arr_inds_struct,
                        i,
                    )
                    arr_obj = get_df_obj_column_codegen(
                        c.context,
                        c.builder,
                        c.pyapi,
                        val,
                        arr_ind,
                        t,
                    )
                    arr = c.pyapi.to_native_value(t, arr_obj).value
                    out_arr_list.inititem(i, arr, incref=False)
                    c.pyapi.decref(arr_obj)

            setattr(table, f"block_{blk}", out_arr_list.value)

        # Set the length of the table. This should be valid even
        # with 0 columns.
        n_obj = c.pyapi.call_method(val, "__len__", ())
        length = c.pyapi.long_as_longlong(n_obj)
        c.pyapi.decref(n_obj)

        table.len = length
        data_tup = c.context.make_tuple(
            c.builder, types.Tuple([typ.table_type]), [table._getvalue()]
        )
    else:
        data_nulls = [c.context.get_constant_null(t) for t in typ.data]
        data_tup = c.context.make_tuple(c.builder, types.Tuple(typ.data), data_nulls)

    dataframe_val = construct_dataframe(
        c.context, c.builder, typ, data_tup, index_val, val, None
    )

    return NativeValue(dataframe_val)


def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe"""

    # If the dataframe has typing metadata, pass the typing metadata for the given
    # column to _infer_series_arr_type
    if (
        hasattr(df, "_bodo_meta")
        and df._bodo_meta is not None
        and "type_metadata" in df._bodo_meta
        and df._bodo_meta["type_metadata"] is not None
        # If the metadata hasn't updated but columns are added the information
        # is out of date and cannot be used.
        and len(df._bodo_meta["type_metadata"][1]) == len(df.columns)
    ):
        column_typing_metadata = df._bodo_meta["type_metadata"][1]
    else:
        column_typing_metadata = [None] * len(df.columns)
    hi_typs = [
        _infer_series_arr_type(df.iloc[:, i], array_metadata=column_typing_metadata[i])
        for i in range(len(df.columns))
    ]
    # use dictionary-encoded array if necessary for testing (_use_dict_str_type set)
    hi_typs = [
        bodo.types.dict_str_arr_type
        if _use_dict_str_type and t == string_array_type
        else t
        for t in hi_typs
    ]
    return tuple(hi_typs)


# Modified/extended version of CtypeEnum found in utils. Needed as the base CtypeEnum was not sufficiently general.
# This is used for converting series dtypes to/from metadata.
class SeriesDtypeEnum(Enum):
    Int8 = 0
    UInt8 = 1
    Int32 = 2
    UInt32 = 3
    Int64 = 4
    UInt64 = 7
    Float32 = 5
    Float64 = 6
    Int16 = 8
    UInt16 = 9
    STRING = 10
    Bool = 11
    Decimal = 12
    Datime_Date = 13
    NP_Datetime64ns = 14
    NP_Timedelta64ns = 15
    Int128 = 16
    LIST = 18
    STRUCT = 19
    BINARY = 21
    ARRAY = 22
    PD_nullable_Int8 = 23
    PD_nullable_UInt8 = 24
    PD_nullable_Int16 = 25
    PD_nullable_UInt16 = 26
    PD_nullable_Int32 = 27
    PD_nullable_UInt32 = 28
    PD_nullable_Int64 = 29
    PD_nullable_UInt64 = 30
    PD_nullable_bool = 31
    CategoricalType = 32
    NoneType = 33
    Literal = 34
    IntegerArray = 35
    RangeIndexType = 36
    DatetimeIndexType = 37
    NumericIndexType = 38
    PeriodIndexType = 39
    IntervalIndexType = 40
    CategoricalIndexType = 41
    StringIndexType = 42
    BinaryIndexType = 43
    TimedeltaIndexType = 44
    LiteralType = 45
    PD_nullable_Float32 = 46
    PD_nullable_Float64 = 47
    FloatingArray = 48
    NullArray = 49
    PD_datetime_tz = 50
    Time = 51
    TimestampTZ = 52


# Map of types that can be mapped to a singular enum. Maps type -> enum
_one_to_one_type_to_enum_map: dict[types.Type, int] = {
    types.int8: SeriesDtypeEnum.Int8.value,
    types.uint8: SeriesDtypeEnum.UInt8.value,
    types.int32: SeriesDtypeEnum.Int32.value,
    types.uint32: SeriesDtypeEnum.UInt32.value,
    types.int64: SeriesDtypeEnum.Int64.value,
    types.uint64: SeriesDtypeEnum.UInt64.value,
    types.float32: SeriesDtypeEnum.Float32.value,
    types.float64: SeriesDtypeEnum.Float64.value,
    types.NPDatetime("ns"): SeriesDtypeEnum.NP_Datetime64ns.value,
    types.NPTimedelta("ns"): SeriesDtypeEnum.NP_Timedelta64ns.value,
    bodo.types.pd_timedelta_type: SeriesDtypeEnum.NP_Timedelta64ns.value,
    types.bool_: SeriesDtypeEnum.Bool.value,
    types.int16: SeriesDtypeEnum.Int16.value,
    types.uint16: SeriesDtypeEnum.UInt16.value,
    types.Integer("int128", 128): SeriesDtypeEnum.Int128.value,
    bodo.hiframes.datetime_date_ext.datetime_date_type: SeriesDtypeEnum.Datime_Date.value,
    IntDtype(types.int8): SeriesDtypeEnum.PD_nullable_Int8.value,
    IntDtype(types.uint8): SeriesDtypeEnum.PD_nullable_UInt8.value,
    IntDtype(types.int16): SeriesDtypeEnum.PD_nullable_Int16.value,
    IntDtype(types.uint16): SeriesDtypeEnum.PD_nullable_UInt16.value,
    IntDtype(types.int32): SeriesDtypeEnum.PD_nullable_Int32.value,
    IntDtype(types.uint32): SeriesDtypeEnum.PD_nullable_UInt32.value,
    IntDtype(types.int64): SeriesDtypeEnum.PD_nullable_Int64.value,
    IntDtype(types.uint64): SeriesDtypeEnum.PD_nullable_UInt64.value,
    FloatDtype(types.float32): SeriesDtypeEnum.PD_nullable_Float32.value,
    FloatDtype(types.float64): SeriesDtypeEnum.PD_nullable_Float64.value,
    bytes_type: SeriesDtypeEnum.BINARY.value,
    string_type: SeriesDtypeEnum.STRING.value,
    bodo.types.bool_: SeriesDtypeEnum.Bool.value,
    types.none: SeriesDtypeEnum.NoneType.value,
    null_array_type: SeriesDtypeEnum.NullArray.value,
    timestamptz_type: SeriesDtypeEnum.TimestampTZ.value,
}

# The reverse of the above map, Maps enum -> type
_one_to_one_enum_to_type_map: dict[int, types.Type] = {
    SeriesDtypeEnum.Int8.value: types.int8,
    SeriesDtypeEnum.UInt8.value: types.uint8,
    SeriesDtypeEnum.Int32.value: types.int32,
    SeriesDtypeEnum.UInt32.value: types.uint32,
    SeriesDtypeEnum.Int64.value: types.int64,
    SeriesDtypeEnum.UInt64.value: types.uint64,
    SeriesDtypeEnum.Float32.value: types.float32,
    SeriesDtypeEnum.Float64.value: types.float64,
    SeriesDtypeEnum.NP_Datetime64ns.value: types.NPDatetime("ns"),
    SeriesDtypeEnum.NP_Timedelta64ns.value: types.NPTimedelta("ns"),
    SeriesDtypeEnum.Int16.value: types.int16,
    SeriesDtypeEnum.UInt16.value: types.uint16,
    SeriesDtypeEnum.Int128.value: types.Integer("int128", 128),
    SeriesDtypeEnum.Datime_Date.value: bodo.hiframes.datetime_date_ext.datetime_date_type,
    SeriesDtypeEnum.PD_nullable_Int8.value: IntDtype(types.int8),
    SeriesDtypeEnum.PD_nullable_UInt8.value: IntDtype(types.uint8),
    SeriesDtypeEnum.PD_nullable_Int16.value: IntDtype(types.int16),
    SeriesDtypeEnum.PD_nullable_UInt16.value: IntDtype(types.uint16),
    SeriesDtypeEnum.PD_nullable_Int32.value: IntDtype(types.int32),
    SeriesDtypeEnum.PD_nullable_UInt32.value: IntDtype(types.uint32),
    SeriesDtypeEnum.PD_nullable_Int64.value: IntDtype(types.int64),
    SeriesDtypeEnum.PD_nullable_UInt64.value: IntDtype(types.uint64),
    SeriesDtypeEnum.PD_nullable_Float32.value: FloatDtype(types.float32),
    SeriesDtypeEnum.PD_nullable_Float64.value: FloatDtype(types.float64),
    SeriesDtypeEnum.BINARY.value: bytes_type,
    SeriesDtypeEnum.STRING.value: string_type,
    SeriesDtypeEnum.Bool.value: bodo.types.bool_,
    SeriesDtypeEnum.NoneType.value: types.none,
    SeriesDtypeEnum.NullArray.value: null_array_type,
    SeriesDtypeEnum.TimestampTZ.value: timestamptz_type,
}


def _dtype_from_type_enum_list(typ_enum_list):
    """Wrapper around _dtype_from_type_enum_list_recursor"""
    remaining, typ = _dtype_from_type_enum_list_recursor(typ_enum_list)
    if len(remaining) != 0:  # pragma: no cover
        raise_bodo_error(
            f"Unexpected Internal Error while converting typing metadata: Dtype list was not fully consumed.\n Input typ_enum_list: {typ_enum_list}.\nRemainder: {remaining}. Please file the error here: https://github.com/bodo-ai/Feedback"
        )
    return typ


def _dtype_from_type_enum_list_recursor(typ_enum_list):
    """
    Converts a list of type enums generated by _dtype_to_type_enum_list, and converts it
    back into a dtype.

    The general structure procedure as follows:

    The type enum list acts as a stack. At the beginning of each call to
    _dtype_from_type_enum_list_recursor, the function pops
    one or more enums from the typ_enum_list, and returns a tuple of the remaining
    typ_enum_list, and the dtype that was inferred.

    Example:

        [19, 3, "A", "B", "C", 11, 34, "hello", 19, 1, "D", 6]

    Recursor 0 pops from the top of the enum list, and see 19, which indicates a struct.
    The Recursor expects the next value on the stack to be the number of fields, so
    it pops from the enum list again, and sees three. The Recursor expects the next three
    arguments to be the three field names, so it pops "A", "B", "C" from the type enum
    list. Recursor 0 does a recursive call on the remaining enum list:

        [11, 34, "hello", 19, 1, "D", 6]

    Recursor 1 pops the value 11, which indicates a bool. Recursor 1 returns:

        ([34, "hello", 19, 1, "D", 6], bool)

    Recursor 0 now knows that the type of the struct's "A" field is bool. It generates another
    function call, using the reuduced list:

        [34, "hello", 19, 1, "D", 6]

    Recursor 2 pops the value 34, which is a the enum for literal. Therefore, it knows that
    the next value on the stack, "hello" is a literal. Recursor 2 returns:

        ([19, 1, "D", 6], "hello")

    Recursor 0 now knows that the type of the struct's "B" field is the literal string
    "hello" (if this isn't possible, pretend that it is for this demonstration).
    It generates another function call, using the reuduced list:

        [19, 1, "D", 6]

    Recursor 2 pops 19, which means we have a struct. As before, we pop the length, and the
    fieldname, then generate a recursive call to find the type of field "D":

        [6]

    Recursor 3 pops 6, maps the enum value 6 to the type Float64 and returns:
        [], Float64

    Recursor 2 takes the type and the reduced list, and returns

        [], StructType((Float64, ), ("D", ))

    Recursor 0 takes the information from Recursor 2, which indicates that it's C field is of
    type. Recursor 0 finally has all the type information for each of its three fields.
    It returns:

        [], StructType((Bool, "hello", StructType((Float64, ), ("D", ))) ("A", "B", "C"))

    """
    if len(typ_enum_list) == 0:  # pragma: no cover
        raise_bodo_error("Unable to infer dtype from empty typ_enum_list")
    elif typ_enum_list[0] in _one_to_one_enum_to_type_map:
        return (
            typ_enum_list[1:],
            _one_to_one_enum_to_type_map[typ_enum_list[0]],
        )
    elif typ_enum_list[0] == SeriesDtypeEnum.Time.value:
        precision: int = typ_enum_list[1]
        return (typ_enum_list[2:], TimeType(precision))
    elif typ_enum_list[0] == SeriesDtypeEnum.PD_datetime_tz.value:
        tz: str | None = typ_enum_list[1]
        return (typ_enum_list[2:], PandasDatetimeTZDtype(tz))
    # Integer array needs special handling, as integerArray.dtype does not return
    # a nullable integer type
    elif typ_enum_list[0] == SeriesDtypeEnum.IntegerArray.value:
        (remaining_typ_enum_list, typ) = _dtype_from_type_enum_list_recursor(
            typ_enum_list[1:]
        )
        return (remaining_typ_enum_list, IntegerArrayType(typ))
    # Float array needs special handling, as FloatingArray.dtype does not return
    # a nullable float type
    elif typ_enum_list[0] == SeriesDtypeEnum.FloatingArray.value:  # pragma: no cover
        (remaining_typ_enum_list, typ) = _dtype_from_type_enum_list_recursor(
            typ_enum_list[1:]
        )
        return (remaining_typ_enum_list, FloatingArrayType(typ))
    elif typ_enum_list[0] == SeriesDtypeEnum.ARRAY.value:
        (remaining_typ_enum_list, typ) = _dtype_from_type_enum_list_recursor(
            typ_enum_list[1:]
        )
        return (remaining_typ_enum_list, dtype_to_array_type(typ))
    elif typ_enum_list[0] == SeriesDtypeEnum.Decimal.value:
        precision = typ_enum_list[1]
        scale = typ_enum_list[2]
        return typ_enum_list[3:], Decimal128Type(precision, scale)
    elif typ_enum_list[0] == SeriesDtypeEnum.STRUCT.value:
        # For structs the expected structure is:
        # [STRUCT.value, num_fields, field_name_1, ... field_name_n, field_type_1, ... field_name_n,]
        num_fields = typ_enum_list[1]
        field_names = tuple(typ_enum_list[2 : 2 + num_fields])
        remainder = typ_enum_list[2 + num_fields :]
        field_typs = []
        for i in range(num_fields):
            remainder, cur_field_typ = _dtype_from_type_enum_list_recursor(remainder)
            field_typs.append(cur_field_typ)

        return remainder, StructType(tuple(field_typs), field_names)
    elif typ_enum_list[0] == SeriesDtypeEnum.Literal.value:
        # If we encounter LITERAL, we expect the next value to be a literal value.
        # This is generally used to pass things like struct names, which are a part of the type.
        if len(typ_enum_list) == 1:  # pragma: no cover
            raise_bodo_error(
                "Unexpected Internal Error while converting typing metadata: Encountered 'Literal' internal enum value with no value following it. Please file the error here: https://github.com/bodo-ai/Feedback"
            )
        lit_val = typ_enum_list[1]
        remainder = typ_enum_list[2:]
        return remainder, lit_val
    elif typ_enum_list[0] == SeriesDtypeEnum.LiteralType.value:
        if len(typ_enum_list) == 1:  # pragma: no cover
            raise_bodo_error(
                "Unexpected Internal Error while converting typing metadata: Encountered 'LiteralType' internal enum value with no value following it. Please file the error here: https://github.com/bodo-ai/Feedback"
            )
        lit_val = typ_enum_list[1]
        remainder = typ_enum_list[2:]
        return remainder, numba.types.literal(lit_val)
    elif typ_enum_list[0] == SeriesDtypeEnum.CategoricalType.value:
        # For CategoricalType the expected ordering is the same order as the constructor:
        # [CategoricalType.value, categories, elem_type, ordered, data, int_type]
        remainder, categories = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        remainder, elem_type = _dtype_from_type_enum_list_recursor(remainder)
        remainder, ordered = _dtype_from_type_enum_list_recursor(remainder)
        remainder, data = _dtype_from_type_enum_list_recursor(remainder)
        remainder, int_type = _dtype_from_type_enum_list_recursor(remainder)
        return remainder, PDCategoricalDtype(
            categories, elem_type, ordered, data, int_type
        )

    # For the index types, the arguments are stored in the same order
    # that they are passed to their constructor
    elif typ_enum_list[0] == SeriesDtypeEnum.DatetimeIndexType.value:
        # Constructor for DatetimeIndexType:
        # def __init__(self, name_typ=None)
        remainder, name_type = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        return remainder, DatetimeIndexType(name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.NumericIndexType.value:
        # Constructor for NumericIndexType
        # def __init__(self, dtype, name_typ=None, data=None)
        remainder, dtype = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        remainder, name_type = _dtype_from_type_enum_list_recursor(remainder)
        remainder, data = _dtype_from_type_enum_list_recursor(remainder)
        return remainder, NumericIndexType(dtype, name_type, data)
    elif typ_enum_list[0] == SeriesDtypeEnum.PeriodIndexType.value:
        # Constructor for PeriodIndexType
        # def __init__(self, freq, name_typ=None)
        remainder, freq = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        remainder, name_type = _dtype_from_type_enum_list_recursor(remainder)
        return remainder, PeriodIndexType(freq, name_type)
    # IntervalIndexType not supported due to coverage gaps, see BE-711
    # elif typ_enum_list[0] == SeriesDtypeEnum.IntervalIndexType.value:
    #     # Constructor for IntervalIndexType
    #     # def __init__(self, data, name_typ=None)
    #     remainder, data = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
    #     remainder, name_type = _dtype_from_type_enum_list_recursor(remainder)
    #     return remainder, IntervalIndexType(data, name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.CategoricalIndexType.value:
        # Constructor for CategoricalIndexType
        # def __init__(self, data, name_typ=None)
        remainder, data = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        remainder, name_type = _dtype_from_type_enum_list_recursor(remainder)
        return remainder, CategoricalIndexType(data, name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.RangeIndexType.value:
        # Constructor for RangeIndexType:
        # def __init__(self, name_typ)
        remainder, name_type = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        return remainder, RangeIndexType(name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.StringIndexType.value:
        # Constructor for StringIndexType:
        # def __init__(self, name_typ=None)
        remainder, name_type = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        return remainder, StringIndexType(name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.BinaryIndexType.value:
        # Constructor for BinaryIndexType:
        # def __init__(self, name_typ=None)
        remainder, name_type = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        return remainder, BinaryIndexType(name_type)
    elif typ_enum_list[0] == SeriesDtypeEnum.TimedeltaIndexType.value:
        # Constructor for TimedeltaIndexType:
        # def __init__(self, name_typ=None)
        remainder, name_type = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        return remainder, TimedeltaIndexType(name_type)

    # Previously, in _dtype_to_type_enum_list, if a type wasn't manually handled we
    # pickled it.
    # for example, if we added support for a new index type and fail to update
    # _dtype_to_type_enum_list, it would be converted to a pickled bytestring.
    # Currently, this does not occur. _dtype_to_type_enum_list will return None
    # If it encounters a type that is not explicitley handled.
    # elif isinstance(typ_enum_list[0], bytes):
    #     return typ_enum_list[1:], pickle.loads(typ_enum_list[0])

    else:  # pragma: no cover
        raise_bodo_error(
            f"Unexpected Internal Error while converting typing metadata: unable to infer dtype for type enum {typ_enum_list[0]}. Please file the error here: https://github.com/bodo-ai/Feedback"
        )


def _dtype_to_type_enum_list(typ):
    """wrapper around _dtype_to_type_enum_list_recursor"""
    return guard(_dtype_to_type_enum_list_recursor, typ)


def _dtype_to_type_enum_list_recursor(typ, upcast_numeric_index=True):
    """
    Recursively converts the dtype into a stack of nested enums/literal values.
    This dtype list will be appeneded to series/datframe metadata, so that we can infer the
    original dtype's of series with object dtype.

    For a complete example of the general process of converting to/from this stack, see
    _dtype_from_type_enum_list_recursor.
    """
    # handle common cases first
    if typ.__hash__ and typ in _one_to_one_type_to_enum_map:
        return [_one_to_one_type_to_enum_map[typ]]
    # manually handle the constant types
    # that we've verified to work ctx.get_constant_generic
    # in test_metadata/test_dtype_converter_literal_values

    # handle actually literal python values
    if isinstance(typ, (dict, int, list, tuple, str, bool, bytes, float)):
        return [SeriesDtypeEnum.Literal.value, typ]
    elif typ is None:
        return [SeriesDtypeEnum.Literal.value, typ]

    # handle literal types
    elif is_overload_constant_int(typ):
        const_val = get_overload_const_int(typ)
        if numba.types.maybe_literal(const_val) == typ:
            return [SeriesDtypeEnum.LiteralType.value, const_val]
    elif is_overload_constant_str(typ):
        const_val = get_overload_const_str(typ)
        if numba.types.maybe_literal(const_val) == typ:
            return [SeriesDtypeEnum.LiteralType.value, const_val]
    elif is_overload_constant_bool(typ):
        const_val = get_overload_const_bool(typ)
        if numba.types.maybe_literal(const_val) == typ:
            return [SeriesDtypeEnum.LiteralType.value, const_val]

    # TODO: handle the following literal types which need special handling
    # outside of just using numba.types.literal
    # tuples
    # lists
    # dicts
    # bytes
    # floats

    # integer arrays need special handling, as integerArray's dtype is not a nullable integer
    elif isinstance(typ, IntegerArrayType):
        return [SeriesDtypeEnum.IntegerArray.value] + _dtype_to_type_enum_list_recursor(
            typ.dtype
        )
    # floating arrays need special handling, as FloatingArray's dtype is not a nullable float
    elif isinstance(typ, FloatingArrayType):  # pragma: no cover
        return [
            SeriesDtypeEnum.FloatingArray.value
        ] + _dtype_to_type_enum_list_recursor(typ.dtype)
    elif bodo.utils.utils.is_array_typ(typ, False):
        return [SeriesDtypeEnum.ARRAY.value] + _dtype_to_type_enum_list_recursor(
            typ.dtype
        )
    elif isinstance(typ, PandasDatetimeTZDtype):
        return [SeriesDtypeEnum.PD_datetime_tz.value, typ.tz]
    elif isinstance(typ, TimeType):
        return [SeriesDtypeEnum.Time.value, typ.precision]
    # TODO: add Categorical, String
    elif isinstance(typ, StructType):
        # for struct include the type ID and number of fields
        types_list = [SeriesDtypeEnum.STRUCT.value, len(typ.names)]
        for name in typ.names:
            types_list.append(name)
        for field_typ in typ.data:
            types_list += _dtype_to_type_enum_list_recursor(field_typ)
        return types_list
    elif isinstance(typ, bodo.libs.decimal_arr_ext.Decimal128Type):
        return [SeriesDtypeEnum.Decimal.value, typ.precision, typ.scale]
    elif isinstance(typ, PDCategoricalDtype):
        # For CategoricalType the expected ordering is the same order as the constructor:
        # def __init__(self, categories, elem_type, ordered, data=None, int_type=None)
        categories_enum_list = _dtype_to_type_enum_list_recursor(typ.categories)
        elem_type_enum_list = _dtype_to_type_enum_list_recursor(typ.elem_type)
        ordered_enum_list = _dtype_to_type_enum_list_recursor(typ.ordered)
        data_enum_list = _dtype_to_type_enum_list_recursor(typ.data)
        int_type_enum_list = _dtype_to_type_enum_list_recursor(typ.int_type)
        return (
            [SeriesDtypeEnum.CategoricalType.value]
            + categories_enum_list
            + elem_type_enum_list
            + ordered_enum_list
            + data_enum_list
            + int_type_enum_list
        )

    # For the index types, we store the values in the same ordering as the constructor
    elif isinstance(typ, DatetimeIndexType):
        # Constructor for DatetimeIndexType:
        # def __init__(self, name_typ=None)
        return [
            SeriesDtypeEnum.DatetimeIndexType.value
        ] + _dtype_to_type_enum_list_recursor(typ.name_typ)
    elif isinstance(typ, NumericIndexType):
        # Constructor for NumericIndexType
        # def __init__(self, dtype, name_typ=None, data=None)

        # In the case that we're converting a dataframe index,
        # we need to upcast to 64 bit width in order to match pandas semantics
        if upcast_numeric_index:
            if isinstance(typ.dtype, types.Float):
                upcasted_dtype = types.float64
                if isinstance(typ.data, FloatingArrayType):  # pragma: no cover
                    upcasted_arr_typ = FloatingArrayType(upcasted_dtype)
                else:
                    upcasted_arr_typ = types.Array(upcasted_dtype, 1, "C")
            elif typ.dtype in {
                types.int8,
                types.int16,
                types.int32,
                types.int64,
            }:
                upcasted_dtype = types.int64
                if isinstance(typ.data, IntegerArrayType):
                    upcasted_arr_typ = IntegerArrayType(upcasted_dtype)
                else:
                    upcasted_arr_typ = types.Array(upcasted_dtype, 1, "C")
            elif typ.dtype in {
                types.uint8,
                types.uint16,
                types.uint32,
                types.uint64,
            }:
                upcasted_dtype = types.uint64
                if isinstance(typ.data, IntegerArrayType):
                    upcasted_arr_typ = IntegerArrayType(upcasted_dtype)
                else:
                    upcasted_arr_typ = types.Array(upcasted_dtype, 1, "C")
            elif typ.dtype == types.bool_:
                upcasted_dtype = typ.dtype
                upcasted_arr_typ = typ.data
            else:
                raise GuardException("Unable to convert type")

            return (
                [SeriesDtypeEnum.NumericIndexType.value]
                + _dtype_to_type_enum_list_recursor(upcasted_dtype)
                + _dtype_to_type_enum_list_recursor(typ.name_typ)
                + _dtype_to_type_enum_list_recursor(upcasted_arr_typ)
            )
        else:  # pragma: no cover
            # we currently never take this path, but there might be a valid
            # reason to in the future.
            return (
                [SeriesDtypeEnum.NumericIndexType.value]
                + _dtype_to_type_enum_list_recursor(typ.dtype)
                + _dtype_to_type_enum_list_recursor(typ.name_typ)
                + _dtype_to_type_enum_list_recursor(typ.data)
            )

    elif isinstance(typ, PeriodIndexType):
        # Constructor for PeriodIndexType
        # def __init__(self, freq, name_typ=None)
        return (
            [SeriesDtypeEnum.PeriodIndexType.value]
            + _dtype_to_type_enum_list_recursor(typ.freq)
            + _dtype_to_type_enum_list_recursor(typ.name_typ)
        )
    # IntervalIndexType not supported due to coverage gaps, see BE-711
    # elif isinstance(typ, IntervalIndexType):
    #     # Constructor for IntervalIndexType
    #     # def __init__(self, data, name_typ=None)
    #     return (
    #         [SeriesDtypeEnum.IntervalIndexType.value]
    #         + _dtype_to_type_enum_list_recursor(typ.data)
    #         + _dtype_to_type_enum_list_recursor(typ.name_typ)
    #     )
    elif isinstance(typ, CategoricalIndexType):
        # Constructor for CategoricalIndexType
        # def __init__(self, data, name_typ=None)
        return (
            [SeriesDtypeEnum.CategoricalIndexType.value]
            + _dtype_to_type_enum_list_recursor(typ.data)
            + _dtype_to_type_enum_list_recursor(typ.name_typ)
        )
    elif isinstance(typ, RangeIndexType):
        # Constructor for RangeIndexType:
        # def __init__(self, name_typ)
        return [
            SeriesDtypeEnum.RangeIndexType.value
        ] + _dtype_to_type_enum_list_recursor(typ.name_typ)
    elif isinstance(typ, StringIndexType):
        # Constructor for StringIndexType:
        # def __init__(self, name_typ=None)
        return [
            SeriesDtypeEnum.StringIndexType.value
        ] + _dtype_to_type_enum_list_recursor(typ.name_typ)
    elif isinstance(typ, BinaryIndexType):
        # Constructor for BinaryIndexType:
        # def __init__(self, name_typ=None)
        return [
            SeriesDtypeEnum.BinaryIndexType.value
        ] + _dtype_to_type_enum_list_recursor(typ.name_typ)
    elif isinstance(typ, TimedeltaIndexType):
        # Constructor for TimedeltaIndexType:
        # def __init__(self, name_typ=None)
        return [
            SeriesDtypeEnum.TimedeltaIndexType.value
        ] + _dtype_to_type_enum_list_recursor(typ.name_typ)
    else:
        # Previously,
        # If a type wasn't manually handled we, pickled it.
        # for example, if we add a support for a new index type and fail to update
        # this function, it would be converted to a pickled bytestring.
        # return [pickle.dumps(typ)]
        # as of now, we raise a guard exception, which is caught be the wrapping
        # _dtype_to_type_enum_list, and return None.
        raise GuardException("Unable to convert type")


def _is_wrapper_pd_arr(arr):
    """return True if 'arr' is a Pandas wrapper array around regular Numpy like NumpyExtensionArray"""

    # Pandas bug (as of 1.5): StringArray is a subclass of NumpyExtensionArray for some reason
    if isinstance(arr, pd.arrays.StringArray):
        return False

    return isinstance(
        arr, (pd.arrays.NumpyExtensionArray, pd.arrays.TimedeltaArray)
    ) or (isinstance(arr, pd.arrays.DatetimeArray) and arr.tz is None)


def unwrap_pd_arr(arr):
    """Unwrap Numpy array from the NumpyExtensionArray wrapper for unboxing purposes

    Args:
        arr (pd.Array): input array which could be NumpyExtensionArray

    Returns:
        pd.array or np.ndarray: numpy array or Pandas extension array
    """
    if _is_wrapper_pd_arr(arr):
        # call np.ascontiguousarray() on array since it may not be contiguous
        # the typing infrastructure assumes C-contiguous arrays
        # see test_df_multi_get_level() for an example of non-contiguous input
        return np.ascontiguousarray(arr._ndarray)

    return arr


def _fix_series_arr_type(pd_arr):
    """remove Pandas array wrappers like NumpyExtensionArray to make Series typing easier"""
    if _is_wrapper_pd_arr(pd_arr):
        return pd_arr._ndarray

    return pd_arr


def _infer_series_arr_type(S: pd.Series, array_metadata=None):
    """infer underlying array type for unboxing a Pandas Series object

    Args:
        S (pd.Series): input Pandas Series to unbox
        array_metadata (list(enum), optional): type metadata that Bodo stores during
        boxing to help with typing object arrays. Defaults to None.

    Raises:
        BodoError: cannot handle Pandas nullable float arrays
        BodoError: cannot handle tz-aware datetime with non-ns unit
        BodoError: other potential unsupported types

    Returns:
        types.Type: array type for Series data
    """

    if S.dtype == np.dtype("O"):
        # We check the metadata if the data is empty or all null
        if len(S.array) == 0 or S.isna().sum() == len(S):
            if array_metadata is not None:
                # If the metadata is passed by the dataframe, it is the type of the underlying array.

                # TODO: array metadata is going to return the type of the array, not the
                # type of the Series. This will return different types for null integer,
                # but for object series, I can't think of a situation in which the
                # dtypes would be different.
                return _dtype_from_type_enum_list(array_metadata)
            elif (
                hasattr(S, "_bodo_meta")
                and S._bodo_meta is not None
                and "type_metadata" in S._bodo_meta
                and S._bodo_meta["type_metadata"][1] is not None
            ):  # pragma: no cover
                # pragma is needed as we never enter this case with np > 1
                type_list = S._bodo_meta["type_metadata"][1]
                # If the Series itself has the typing metadata, it will be the original
                # dtype of the series
                # TODO: Update the encoded type metadata for array type instead of dtype.
                return dtype_to_array_type(_dtype_from_type_enum_list(type_list))

        return bodo.typeof(_fix_series_arr_type(S.array))

    # infer type of underlying data array
    try:
        arr_type = bodo.typeof(_fix_series_arr_type(S.array))

        if isinstance(arr_type, types.Array):
            dtype = arr_type.dtype

            # always unbox boolean Series using nullable boolean array instead of Numpy
            # because some processes may have nulls, leading to inconsistent data types
            if dtype == types.bool_:
                arr_type = bodo.types.boolean_array_type

            # Make sure datetime64 arrays are ns
            elif isinstance(dtype, types.NPDatetime):
                arr_type = bodo.types.DatetimeArrayType(None)

            # Make sure timedelta64 arrays are ns
            elif isinstance(dtype, types.NPTimedelta):
                arr_type = bodo.types.timedelta_array_type
            else:
                # We make all Series data arrays contiguous during unboxing to avoid type errors
                # see test_df_query_stringliteral_expr
                assert arr_type.ndim == 1, "invalid numpy array type in Series"
                arr_type = types.Array(arr_type.dtype, 1, "C")

        return arr_type
    except pa.lib.ArrowMemoryError:  # pragma: no cover
        # OOM
        raise
    except Exception:  # pragma: no cover
        raise BodoError(f"data type {S.dtype} for column {S.name} not supported yet")


def _get_use_df_parent_obj_flag(builder, context, pyapi, parent_obj, n_cols):
    """Returns a flag (LLVM value) that determines if the parent object of the DataFrame
    value should be used for boxing.
    """
    # the dataframe doesn't have a parent obj if the number of columns is unknown
    # (e.g. output of Bodo operations like pivot)
    if n_cols is None:
        return context.get_constant(types.bool_, False)

    # df unboxed from Python
    has_parent = cgutils.is_not_null(builder, parent_obj)
    # corner case (which should be avoided):
    # df parent could come from the data table used for df initialization.
    # The table may have changed (new columns added) using set_table_data() so parent
    # object may not usable since number of columns don't match anymore
    parent_ncols = cgutils.alloca_once_value(
        builder, context.get_constant(types.int64, 0)
    )
    with builder.if_then(has_parent):
        cols_obj = pyapi.object_getattr_string(parent_obj, "columns")
        n_obj = pyapi.call_method(cols_obj, "__len__", ())
        builder.store(pyapi.long_as_longlong(n_obj), parent_ncols)
        pyapi.decref(n_obj)
        pyapi.decref(cols_obj)

    use_parent_obj = builder.and_(
        has_parent,
        builder.icmp_unsigned(
            "==", builder.load(parent_ncols), context.get_constant(types.int64, n_cols)
        ),
    )
    return use_parent_obj


def _get_df_columns_obj(c, builder, context, pyapi, df_typ, dataframe_payload):
    """create the columns object for the boxed dataframe object"""

    # if columns are determined during runtime, column names are stored in payload
    if df_typ.has_runtime_cols:
        cols_arr_typ = df_typ.runtime_colname_typ
        context.nrt.incref(builder, cols_arr_typ, dataframe_payload.columns)
        return pyapi.from_native_value(
            cols_arr_typ, dataframe_payload.columns, c.env_manager
        )

    # avoid generating large tuples and lower a constant array if possible
    # (int and string types currently, TODO: support other types)
    if all(isinstance(c, str) for c in df_typ.columns):
        columns_vals = pd.array(df_typ.columns, "string")
    elif all(isinstance(c, int) for c in df_typ.columns):
        columns_vals = np.array(df_typ.columns, "int64")
    else:
        columns_vals = df_typ.columns

    columns_typ = numba.typeof(columns_vals)
    columns = context.get_constant_generic(builder, columns_typ, columns_vals)
    columns_obj = pyapi.from_native_value(columns_typ, columns, c.env_manager)

    # avoid ArrowStringArray for column names due to Pandas bug for df column getattr
    # see test_jit_inside_prange
    if columns_typ == bodo.types.string_array_type:
        prev_columns_obj = columns_obj
        columns_obj = pyapi.call_method(columns_obj, "to_numpy", ())
        pyapi.decref(prev_columns_obj)

    return columns_obj


def _create_initial_df_object(
    builder, context, pyapi, c, df_typ, obj, dataframe_payload, res, use_parent_obj
):
    """create the initial dataframe object to fill in boxing and store in 'res'"""
    # get initial dataframe object
    with c.builder.if_else(use_parent_obj) as (use_parent, otherwise):
        with use_parent:
            pyapi.incref(obj)
            # set parent dataframe column names to numbers for robust setting of columns
            # df.columns = np.arange(len(df.columns))
            mod_name = context.insert_const_string(c.builder.module, "numpy")
            class_obj = pyapi.import_module(mod_name)
            if df_typ.has_runtime_cols:
                # If we have columns determined at runtime, use_parent is always False
                num_cols = 0
            else:
                num_cols = len(df_typ.columns)
            n_cols_obj = pyapi.long_from_longlong(
                lir.Constant(lir.IntType(64), num_cols)
            )
            col_nums_arr_obj = pyapi.call_method(class_obj, "arange", (n_cols_obj,))
            pyapi.object_setattr_string(obj, "columns", col_nums_arr_obj)
            pyapi.decref(class_obj)
            pyapi.decref(col_nums_arr_obj)
            pyapi.decref(n_cols_obj)
        with otherwise:
            # df_obj = pd.DataFrame(index=index)
            context.nrt.incref(builder, df_typ.index, dataframe_payload.index)
            index_obj = c.pyapi.from_native_value(
                df_typ.index, dataframe_payload.index, c.env_manager
            )

            mod_name = context.insert_const_string(c.builder.module, "pandas")
            class_obj = pyapi.import_module(mod_name)
            df_obj = pyapi.call_method(
                class_obj, "DataFrame", (pyapi.borrow_none(), index_obj)
            )
            pyapi.decref(class_obj)
            pyapi.decref(index_obj)
            builder.store(df_obj, res)


@box(DataFrameType)
def box_dataframe(typ, val, c):
    """Boxes native dataframe value into Python dataframe object, required for function
    return, printing, object mode, etc.
    Works by boxing individual data arrays.
    """
    from bodo.hiframes.table import box_table

    context = c.context
    builder = c.builder
    pyapi = c.pyapi

    dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
        c.context, c.builder, typ, val
    )
    dataframe = cgutils.create_struct_proxy(typ)(context, builder, value=val)
    n_cols = len(typ.columns) if not typ.has_runtime_cols else None

    # see boxing of reflected list in Numba:
    # https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L561
    obj = dataframe.parent
    res = cgutils.alloca_once_value(builder, obj)
    use_parent_obj = _get_use_df_parent_obj_flag(builder, context, pyapi, obj, n_cols)
    _create_initial_df_object(
        builder, context, pyapi, c, typ, obj, dataframe_payload, res, use_parent_obj
    )

    # get data arrays and box them
    if typ.is_table_format:
        table_type = typ.table_type
        table = builder.extract_value(dataframe_payload.data, 0)
        context.nrt.incref(builder, table_type, table)
        # setting ensure_unboxed of box_table() to True if not using parent obj
        table_obj = box_table(table_type, table, c, builder.not_(use_parent_obj))

        with builder.if_else(use_parent_obj) as (then, orelse):
            with then:
                arrs_obj = pyapi.object_getattr_string(table_obj, "arrays")
                none_obj = c.pyapi.make_none()

                if n_cols is None:
                    n_obj = pyapi.call_method(arrs_obj, "__len__", ())
                    n_arrs = pyapi.long_as_longlong(n_obj)
                    pyapi.decref(n_obj)
                else:
                    n_arrs = context.get_constant(types.int64, n_cols)
                with cgutils.for_range(builder, n_arrs) as loop:
                    # df[i] = arr
                    i = loop.index
                    # PyList_GetItem returns borrowed reference (no need to decref)
                    col_arr_obj = pyapi.list_getitem(arrs_obj, i)
                    # box array if df doesn't have a parent, or column was unboxed in function,
                    # since changes in arrays like strings don't reflect back to parent object.
                    # Table boxing assigns None for null arrays
                    is_unboxed = c.builder.icmp_unsigned("!=", col_arr_obj, none_obj)

                    with builder.if_then(is_unboxed):
                        c_ind_obj = pyapi.long_from_longlong(i)
                        df_obj = builder.load(res)
                        pyapi.object_setitem(df_obj, c_ind_obj, col_arr_obj)
                        pyapi.decref(c_ind_obj)

                pyapi.decref(arrs_obj)
                pyapi.decref(none_obj)
            with orelse:
                # fast path for large number of columns
                # df = table.to_pandas(index)
                df_obj = builder.load(res)
                index_obj = pyapi.object_getattr_string(df_obj, "index")
                new_df_obj = c.pyapi.call_method(table_obj, "to_pandas", (index_obj,))
                builder.store(new_df_obj, res)
                pyapi.decref(df_obj)
                pyapi.decref(index_obj)

        pyapi.decref(table_obj)

    else:
        col_arrs = [
            builder.extract_value(dataframe_payload.data, i) for i in range(n_cols)
        ]
        arr_typs = typ.data

        none_obj = pyapi.borrow_none()
        slice_class_obj = pyapi.unserialize(pyapi.serialize_object(slice))
        slice_obj = pyapi.call_function_objargs(slice_class_obj, [none_obj])

        for i, arr, arr_typ in zip(range(n_cols), col_arrs, arr_typs):
            # box array if df doesn't have a parent, or column was unboxed in function,
            # since changes in arrays like strings don't reflect back to parent object
            arr_struct_ptr = cgutils.alloca_once_value(builder, arr)
            null_struct_ptr = cgutils.alloca_once_value(
                builder, context.get_constant_null(arr_typ)
            )

            is_unboxed = builder.not_(
                is_ll_eq(builder, arr_struct_ptr, null_struct_ptr)
            )
            box_array = builder.or_(
                builder.not_(use_parent_obj), builder.and_(use_parent_obj, is_unboxed)
            )

            with builder.if_then(box_array):
                # df.loc[:, i] = boxed_arr
                c_ind_obj = pyapi.long_from_longlong(
                    context.get_constant(types.int64, i)
                )

                context.nrt.incref(builder, arr_typ, arr)
                arr_obj = pyapi.from_native_value(arr_typ, arr, c.env_manager)
                df_obj = builder.load(res)
                df_loc_obj = pyapi.object_getattr_string(df_obj, "loc")
                slice_ind_tup_obj = pyapi.tuple_pack([slice_obj, c_ind_obj])
                pyapi.object_setitem(df_loc_obj, slice_ind_tup_obj, arr_obj)
                pyapi.decref(slice_ind_tup_obj)
                pyapi.decref(df_loc_obj)
                pyapi.decref(arr_obj)
                pyapi.decref(c_ind_obj)

        pyapi.decref(slice_obj)
        pyapi.decref(slice_class_obj)

    df_obj = builder.load(res)
    columns_obj = _get_df_columns_obj(
        c, builder, context, pyapi, typ, dataframe_payload
    )
    # set df columns separately to support repeated names and fix potential multi-index
    # issues, see test_dataframe.py::test_unbox_df_multi, test_box_repeated_names
    pyapi.object_setattr_string(df_obj, "columns", columns_obj)
    pyapi.decref(columns_obj)

    _set_bodo_meta_dataframe(c, df_obj, typ)

    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return df_obj


def get_df_obj_column_codegen(context, builder, pyapi, df_obj, col_ind, data_typ):
    # generate df.iloc[:,i] for parent dataframe object
    none_obj = pyapi.borrow_none()
    slice_class_obj = pyapi.unserialize(pyapi.serialize_object(slice))
    slice_obj = pyapi.call_function_objargs(slice_class_obj, [none_obj])
    col_ind_obj = pyapi.long_from_longlong(col_ind)
    slice_ind_tup_obj = pyapi.tuple_pack([slice_obj, col_ind_obj])

    df_iloc_obj = pyapi.object_getattr_string(df_obj, "iloc")
    series_obj = pyapi.object_getitem(df_iloc_obj, slice_ind_tup_obj)
    arr_obj_orig = pyapi.object_getattr_string(series_obj, "array")

    unwrap_pd_arr_obj = pyapi.unserialize(pyapi.serialize_object(unwrap_pd_arr))
    arr_obj = pyapi.call_function_objargs(unwrap_pd_arr_obj, [arr_obj_orig])

    pyapi.decref(arr_obj_orig)
    pyapi.decref(unwrap_pd_arr_obj)
    pyapi.decref(slice_class_obj)
    pyapi.decref(slice_obj)
    pyapi.decref(col_ind_obj)
    pyapi.decref(slice_ind_tup_obj)
    pyapi.decref(df_iloc_obj)
    pyapi.decref(series_obj)

    return arr_obj


@intrinsic(prefer_literal=True)
def unbox_dataframe_column(typingctx, df, i=None):
    assert isinstance(df, DataFrameType) and is_overload_constant_int(i)

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        c = numba.core.pythonapi._UnboxContext(context, builder, pyapi)

        df_typ = sig.args[0]
        col_ind = get_overload_const_int(sig.args[1])
        data_typ = df_typ.data[col_ind]
        # TODO: refcounts?

        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )

        # TODO: support column of tuples?
        arr_obj = get_df_obj_column_codegen(
            context, builder, pyapi, dataframe.parent, args[1], data_typ
        )
        native_val = _unbox_series_data(data_typ.dtype, data_typ, arr_obj, c)
        c.pyapi.decref(arr_obj)

        # assign array
        dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
            c.context, c.builder, df_typ, args[0]
        )
        if df_typ.is_table_format:
            table = cgutils.create_struct_proxy(df_typ.table_type)(
                c.context, c.builder, builder.extract_value(dataframe_payload.data, 0)
            )
            blk = df_typ.table_type.type_to_blk[data_typ]
            arr_list = getattr(table, f"block_{blk}")
            arr_list_inst = ListInstance(
                c.context, c.builder, types.List(data_typ), arr_list
            )
            offset = context.get_constant(
                types.int64, df_typ.table_type.block_offsets[col_ind]
            )
            arr_list_inst.inititem(offset, native_val.value, incref=False)
        else:
            dataframe_payload.data = builder.insert_value(
                dataframe_payload.data, native_val.value, col_ind
            )

        # store payload
        payload_type = DataFramePayloadType(df_typ)
        payload_ptr = context.nrt.meminfo_data(builder, dataframe.meminfo)
        ptrty = context.get_value_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        builder.store(dataframe_payload._getvalue(), payload_ptr)

    return signature(types.none, df, i), codegen


@numba.njit
def unbox_col_if_needed(df, i):  # pragma: no cover
    """
    Helper function used to call unbox_dataframe_column when
    necessary without inlining the function
    """
    if bodo.hiframes.pd_dataframe_ext.has_parent(
        df
    ) and bodo.hiframes.pd_dataframe_ext._column_needs_unboxing(df, i):
        bodo.hiframes.boxing.unbox_dataframe_column(df, i)


@unbox(SeriesType)
def unbox_series(typ, val, c):
    # use "array" attribute instead of "values" to handle ExtensionArrays like
    # DatetimeArray properly. Non-ExtensionArrays just use the NumpyExtensionArray
    # wrapper around Numpy
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.array.html#pandas.Series.array
    arr_obj_orig = c.pyapi.object_getattr_string(val, "array")

    unwrap_pd_arr_obj = c.pyapi.unserialize(c.pyapi.serialize_object(unwrap_pd_arr))
    arr_obj = c.pyapi.call_function_objargs(unwrap_pd_arr_obj, [arr_obj_orig])

    data_val = _unbox_series_data(typ.dtype, typ.data, arr_obj, c).value

    index_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, index_obj).value

    name_obj = c.pyapi.object_getattr_string(val, "name")
    name_val = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    series_val = bodo.hiframes.pd_series_ext.construct_series(
        c.context, c.builder, typ, data_val, index_val, name_val
    )
    # TODO: set parent pointer
    c.pyapi.decref(unwrap_pd_arr_obj)
    c.pyapi.decref(arr_obj_orig)
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(name_obj)
    return NativeValue(series_val)


def _unbox_series_data(dtype, data_typ, arr_obj, c):
    if data_typ == string_array_split_view_type:
        # XXX dummy unboxing to avoid errors in _get_dataframe_data()
        out_view = c.context.make_helper(c.builder, string_array_split_view_type)
        return NativeValue(out_view._getvalue())

    return c.pyapi.to_native_value(data_typ, arr_obj)


@box(HeterogeneousSeriesType)
@box(SeriesType)
def box_series(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)

    # TODO: handle parent
    series_payload = bodo.hiframes.pd_series_ext.get_series_payload(
        c.context, c.builder, typ, val
    )

    # box data/index/name
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, typ.data, series_payload.data)
    c.context.nrt.incref(c.builder, typ.index, series_payload.index)
    c.context.nrt.incref(c.builder, typ.name_typ, series_payload.name)
    arr_obj = c.pyapi.from_native_value(typ.data, series_payload.data, c.env_manager)
    index_obj = c.pyapi.from_native_value(
        typ.index, series_payload.index, c.env_manager
    )
    name_obj = c.pyapi.from_native_value(
        typ.name_typ, series_payload.name, c.env_manager
    )

    # call pd.Series()
    if isinstance(typ, HeterogeneousSeriesType) and isinstance(
        typ.data, bodo.types.NullableTupleType
    ):
        # Use object value to preserve NA values (i.e None)
        dtype = c.pyapi.unserialize(c.pyapi.serialize_object(object))
    else:
        dtype = c.pyapi.make_none()  # TODO: dtype
    res = c.pyapi.call_method(
        pd_class_obj, "Series", (arr_obj, index_obj, dtype, name_obj)
    )
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(name_obj)
    # Decref object if used.
    if isinstance(typ, HeterogeneousSeriesType) and isinstance(
        typ.data, bodo.types.NullableTupleType
    ):
        c.pyapi.decref(dtype)

    _set_bodo_meta_series(res, c, typ)

    c.pyapi.decref(pd_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


def type_enum_list_to_py_list_obj(pyapi, context, builder, env_manager, typ_list):
    """Helper function for the metadata functions. Takes a list of type enum values,
    and converts it to an llvm python list"""
    numba_typ_list = []
    for typ_enum_val in typ_list:
        if isinstance(typ_enum_val, int) and not isinstance(typ_enum_val, bool):
            cur_val_obj = pyapi.long_from_longlong(
                lir.Constant(lir.IntType(64), typ_enum_val)
            )
        else:
            # occasionally, we may need to output non enum types
            # as we have encountered literals that are part of the
            # type (for example, field names for struct types)
            typ_enum_typ = numba.typeof(typ_enum_val)

            enum_llvm_const = context.get_constant_generic(
                builder, typ_enum_typ, typ_enum_val
            )
            cur_val_obj = pyapi.from_native_value(
                typ_enum_typ, enum_llvm_const, env_manager
            )
        numba_typ_list.append(cur_val_obj)

    array_type_metadata_obj = pyapi.list_pack(numba_typ_list)
    for val in numba_typ_list:
        pyapi.decref(val)

    return array_type_metadata_obj


def _set_bodo_meta_dataframe(c, obj, typ):
    """set Bodo metadata in output so the next JIT call knows data distribution, and
    the datatypes of the arrays that make up the dataframe.
    e.g. df._bodo_meta = {"dist": 5, "type_metadata": [[*INT_ARRAY_ENUM_LIST*], ["STRING_ARRAY_ENUM_LSIT"]]}
    """
    pyapi = c.pyapi
    context = c.context
    builder = c.builder

    # Only provide typing information when DataFrames don't use table format.
    # TODO: support for dataframes with variable number of columns
    append_typing = not typ.has_runtime_cols

    dict_len = 2 if append_typing else 1

    meta_dict_obj = pyapi.dict_new(dict_len)

    # Set the distribution metadata (possible for all array types)
    # using the distribution number since easier to handle
    dist_val_obj = pyapi.long_from_longlong(
        lir.Constant(lir.IntType(64), typ.dist.value)
    )

    pyapi.dict_setitem_string(meta_dict_obj, "dist", dist_val_obj)

    pyapi.decref(dist_val_obj)

    if append_typing:
        # Setting meta for the array types contained within the dataframe and index type,
        # So that we can infer the dtypes if an empty dataframe is passed from bodo
        # to pandas, and then back to a bodo fn.

        index_typ_list = _dtype_to_type_enum_list(typ.index)

        if index_typ_list != None:
            index_type_metadata_obj = type_enum_list_to_py_list_obj(
                pyapi, context, builder, c.env_manager, index_typ_list
            )
        else:
            index_type_metadata_obj = pyapi.make_none()

        if typ.is_table_format:
            t = typ.table_type
            col_types_metadata_obj = pyapi.list_new(
                lir.Constant(lir.IntType(64), len(typ.data))
            )
            for blk, dtype in t.blk_to_type.items():
                # Determine the type for this block
                typ_list = _dtype_to_type_enum_list(dtype)
                if typ_list != None:
                    typ_list = type_enum_list_to_py_list_obj(
                        pyapi, context, builder, c.env_manager, typ_list
                    )
                else:
                    typ_list = pyapi.make_none()
                # Create a for loop to append to the list at runtime and minimize
                # the IR size.
                n_arrs = c.context.get_constant(
                    types.int64, len(t.block_to_arr_ind[blk])
                )
                arr_inds = c.context.make_constant_array(
                    c.builder,
                    types.Array(types.int64, 1, "C"),
                    # On windows np.array defaults to the np.int32 for integers.
                    # As a result, we manually specify int64 during the array
                    # creation to keep the lowered constant consistent with the
                    # expected type.
                    np.array(t.block_to_arr_ind[blk], dtype=np.int64),
                )
                arr_inds_struct = c.context.make_array(
                    types.Array(types.int64, 1, "C")
                )(c.context, c.builder, arr_inds)
                with cgutils.for_range(c.builder, n_arrs) as loop:
                    i = loop.index
                    # get offset into list
                    arr_ind = _getitem_array_single_int(
                        c.context,
                        c.builder,
                        types.int64,
                        types.Array(types.int64, 1, "C"),
                        arr_inds_struct,
                        i,
                    )
                    c.context.nrt.incref(builder, types.pyobject, typ_list)
                    pyapi.list_setitem(col_types_metadata_obj, arr_ind, typ_list)
                c.context.nrt.decref(builder, types.pyobject, typ_list)
        else:
            col_typs = []
            for dtype in typ.data:
                typ_list = _dtype_to_type_enum_list(dtype)
                if typ_list != None:
                    array_type_metadata_obj = type_enum_list_to_py_list_obj(
                        pyapi, context, builder, c.env_manager, typ_list
                    )
                else:
                    array_type_metadata_obj = pyapi.make_none()

                col_typs.append(array_type_metadata_obj)

            col_types_metadata_obj = pyapi.list_pack(col_typs)
            for val in col_typs:
                pyapi.decref(val)
        df_type_metadata_obj = pyapi.list_pack(
            [index_type_metadata_obj, col_types_metadata_obj]
        )

        pyapi.dict_setitem_string(meta_dict_obj, "type_metadata", df_type_metadata_obj)

    pyapi.object_setattr_string(obj, "_bodo_meta", meta_dict_obj)
    # Decref metadata object
    pyapi.decref(meta_dict_obj)


def get_series_dtype_handle_null_int_and_hetrogenous(series_typ):
    # Heterogeneous series are never distributed. Therefore, we should never need to use the typing metadata
    if isinstance(series_typ, HeterogeneousSeriesType):
        return None

    # when dealing with integer dtypes, the series class stores the non null int dtype
    # instead the nullable IntDtype to avoid errors (I'm not fully certain why, exactly).
    # Therefore, if we encounter a non null int dtype here, we need to confirm that it
    # actually is a non null int dtype
    if isinstance(series_typ.dtype, types.Number) and isinstance(
        series_typ.data, IntegerArrayType
    ):
        return IntDtype(series_typ.dtype)

    if isinstance(series_typ.dtype, types.Float) and isinstance(
        series_typ.data, FloatingArrayType
    ):  # pragma: no cover
        return FloatDtype(series_typ.dtype)

    return series_typ.dtype


def _set_bodo_meta_series(obj, c, typ):
    """set Bodo metadata in output so the next JIT call knows data distribution.
    Also in the case that the boxed series is going to be of object type,
    set the typing metadata, so that we infer the dtype if the series is empty.

    The series datatype is stored as a flattened list, see get_types for an explanation
    of how it is converted.

    e.g. df._bodo_meta = {"dist": 5 "}.
    """
    pyapi = c.pyapi
    context = c.context
    builder = c.builder

    meta_dict_obj = pyapi.dict_new(2)
    # using the distribution number since easier to handle
    dist_val_obj = pyapi.long_from_longlong(
        lir.Constant(lir.IntType(64), typ.dist.value)
    )

    # Setting meta for the series index type,
    index_typ_list = _dtype_to_type_enum_list(typ.index)

    if index_typ_list != None:
        index_type_metadata_obj = type_enum_list_to_py_list_obj(
            pyapi, context, builder, c.env_manager, index_typ_list
        )
    else:
        index_type_metadata_obj = pyapi.make_none()

    # Setting meta for the series dtype
    # handle hetrogenous series, and nullable integer Series.
    dtype = get_series_dtype_handle_null_int_and_hetrogenous(typ)

    # dtype == None if hetrogenous series
    if dtype != None:
        typ_list = _dtype_to_type_enum_list(dtype)
        if typ_list != None:
            colum_type_metadata_obj = type_enum_list_to_py_list_obj(
                pyapi, context, builder, c.env_manager, typ_list
            )
        else:
            colum_type_metadata_obj = pyapi.make_none()
    else:
        colum_type_metadata_obj = pyapi.make_none()

    type_metadata_obj = pyapi.list_pack(
        [index_type_metadata_obj, colum_type_metadata_obj]
    )
    pyapi.dict_setitem_string(meta_dict_obj, "type_metadata", type_metadata_obj)
    pyapi.decref(type_metadata_obj)

    pyapi.dict_setitem_string(meta_dict_obj, "dist", dist_val_obj)
    pyapi.object_setattr_string(obj, "_bodo_meta", meta_dict_obj)
    pyapi.decref(meta_dict_obj)
    pyapi.decref(dist_val_obj)


# --------------- typeof support for object arrays --------------------


# XXX: this is overwriting Numba's array type registration, make sure it is
# robust
# TODO: support other array types like datetime.date
@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    try:
        dtype = numba.np.numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        dtype = types.pyobject

    if dtype == types.pyobject:
        return _infer_ndarray_obj_dtype(val)

    layout = numba.np.numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)


def _infer_ndarray_obj_dtype(val):
    # strings only have object dtype, TODO: support fixed size np strings
    if not val.dtype == np.dtype("O"):  # pragma: no cover
        raise BodoError(f"Unsupported array dtype: {val.dtype}")

    # XXX assuming the whole array is strings if 1st val is string
    i = 0
    # skip NAs and empty lists/arrays (for array(item) array cases)
    # is_scalar call necessary since pd.isna() treats list of string as array
    while i < len(val) and (
        (_is_scalar_value(val[i]) and pd.isna(val[i]))
        or (not _is_scalar_value(val[i]) and len(val[i]) == 0)
    ):
        i += 1
    if i == len(val):
        # empty or all NA object arrays are assumed to be strings
        warnings.warn(
            BodoWarning(
                "Empty object array passed to Bodo, which causes ambiguity in typing. "
                "This can cause errors in parallel execution."
            )
        )
        return bodo.types.dict_str_arr_type if _use_dict_str_type else string_array_type

    first_val = val[i]
    # For compilation purposes we also impose a limit to the size
    # of the struct as very large structs cannot be efficiently compiled.
    if isinstance(first_val, DictStringSentinel):
        return bodo.types.dict_str_arr_type
    elif isinstance(first_val, str):
        return bodo.types.dict_str_arr_type if _use_dict_str_type else string_array_type
    elif isinstance(first_val, (bytes, bytearray)):
        return binary_array_type
    elif isinstance(first_val, (bool, np.bool_)):
        return bodo.libs.bool_arr_ext.boolean_array_type
    elif isinstance(
        first_val,
        (
            int,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return bodo.libs.int_arr_ext.IntegerArrayType(numba.typeof(first_val))
    elif isinstance(
        first_val,
        (
            float,
            np.float32,
            np.float64,
        ),
    ):  # pragma: no cover
        return bodo.libs.float_arr_ext.FloatingArrayType(numba.typeof(first_val))
    # assuming object arrays with dictionary values string keys are struct arrays, which
    # means all keys are string and match across dictionaries, and all values with same
    # key have same data type
    # TODO: distinguish between Struct and Map arrays properly
    elif (
        isinstance(first_val, (dict, Dict))
        and (len(first_val.keys()) <= struct_size_limit)
        and all(isinstance(k, str) for k in first_val.keys())
    ):
        field_names = tuple(first_val.keys())
        # TODO: handle None value in first_val elements
        data_types = tuple(_get_struct_value_arr_type(v) for v in first_val.values())
        return StructArrayType(data_types, field_names)
    elif isinstance(first_val, (dict, Dict)):
        key_arr_type = numba.typeof(_value_to_array(list(first_val.keys())))
        value_arr_type = numba.typeof(_value_to_array(list(first_val.values())))
        # TODO: handle 2D ndarray case
        return MapArrayType(key_arr_type, value_arr_type)
    elif isinstance(first_val, tuple):
        data_types = tuple(_get_struct_value_arr_type(v) for v in first_val)
        return TupleArrayType(data_types)
    if isinstance(
        first_val,
        (
            list,
            np.ndarray,
            pd.arrays.NumpyExtensionArray,
            pd.arrays.ArrowExtensionArray,
            pd.arrays.BooleanArray,
            pd.arrays.IntegerArray,
            pd.arrays.FloatingArray,
            pd.arrays.StringArray,
            pd.arrays.ArrowStringArray,
            pd.arrays.DatetimeArray,
        ),
    ):
        if isinstance(first_val, list):
            first_val = np.array(first_val, object)
        dtype = numba.typeof(first_val)
        dtype = to_nullable_type(dtype)
        if _use_dict_str_type and dtype == string_array_type:
            dtype = bodo.types.dict_str_arr_type
        return ArrayItemArrayType(dtype)
    if isinstance(first_val, pd.Timestamp):
        return bodo.types.DatetimeArrayType(first_val.tz)
    if isinstance(first_val, datetime.date):
        return datetime_date_array_type
    if isinstance(first_val, datetime.timedelta):
        return bodo.types.timedelta_array_type
    if isinstance(first_val, bodo.types.Time):
        return TimeArrayType(first_val.precision)
    if isinstance(first_val, datetime.time):
        return TimeArrayType(9)
    if isinstance(first_val, decimal.Decimal):
        # NOTE: converting decimal.Decimal objects to 38/18, same as Spark
        return DecimalArrayType(38, 18)
    if isinstance(first_val, pa.Decimal128Scalar):
        return DecimalArrayType(first_val.type.precision, first_val.type.scale)
    if isinstance(first_val, pd._libs.interval.Interval):
        left_dtype = numba.typeof(first_val.left)
        right_dtype = numba.typeof(first_val.right)
        if left_dtype != right_dtype:
            raise BodoError(
                "Bodo can only type Interval Arrays where the the left and right values are the same type"
            )
        arr = dtype_to_array_type(left_dtype, False)
        return bodo.libs.interval_arr_ext.IntervalArrayType(arr)
    if isinstance(first_val, bodo.types.TimestampTZ):
        return bodo.hiframes.timestamptz_ext.timestamptz_array_type

    raise BodoError(
        f"Unsupported object array with first value: {first_val}"
    )  # pragma: no cover


def _value_to_array(val):
    """convert list or dict value to object array for typing purposes"""
    assert isinstance(val, (list, dict, Dict))
    if isinstance(val, (dict, Dict)):
        if isinstance(val, Dict):
            val = dict(val)
        return np.array([val], np.object_)

    # add None to list to avoid Numpy's automatic conversion to 2D arrays
    val_infer = val.copy()
    val_infer.append(None)
    arr = np.array(val_infer, np.object_)
    return arr


def _get_struct_value_arr_type(v):
    """get data array type for a field value of a struct array"""
    if isinstance(v, (dict, Dict)):
        return numba.typeof(_value_to_array(v))

    if isinstance(v, list):
        return dtype_to_array_type(numba.typeof(_value_to_array(v)))

    if isinstance(v, DictStringSentinel):
        return bodo.types.dict_str_arr_type

    if _is_scalar_value(v) and pd.isna(v):
        # assume string array if first field value is NA
        # TODO: use other rows until non-NA is found
        warnings.warn(
            BodoWarning(
                "Field value in struct array is NA, which causes ambiguity in typing. "
                "This can cause errors in parallel execution."
            )
        )
        return string_array_type

    arr_typ = dtype_to_array_type(numba.typeof(v))
    # use nullable arrays since there could be None objects
    arr_typ = to_nullable_type(arr_typ)

    if _use_dict_str_type and arr_typ == string_array_type:
        arr_typ = bodo.types.dict_str_arr_type

    return arr_typ


def _is_scalar_value(val):
    """Return True if value is a Pandas scalar value or PyArrow decimal scalar.
    We use PyArrow scalars since Pandas uses decimal.Decimal which lose precision/scale.
    """
    return pd.api.types.is_scalar(val) or isinstance(val, pa.Decimal128Scalar)
