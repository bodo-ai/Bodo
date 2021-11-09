# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Boxing and unboxing support for DataFrame, Series, etc.
"""
import datetime
import decimal
import warnings
from enum import Enum

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.ir_utils import GuardException, guard
from numba.core.typing import signature
from numba.extending import NativeValue, box, intrinsic, typeof_impl, unbox
from numba.np import numpy_support
from numba.typed.typeddict import Dict

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_array_type
from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype
from bodo.hiframes.pd_dataframe_ext import (
    DataFramePayloadType,
    DataFrameType,
    construct_dataframe,
)
from bodo.hiframes.pd_index_ext import (
    BinaryIndexType,
    CategoricalIndexType,
    DatetimeIndexType,
    IntervalIndexType,
    NumericIndexType,
    PeriodIndexType,
    RangeIndexType,
    StringIndexType,
    TimedeltaIndexType,
)
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.libs import hstr_ext
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.int_arr_ext import (
    IntDtype,
    IntegerArrayType,
    typeof_pd_int_dtype,
)
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import string_array_type, string_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    dtype_to_array_type,
    get_overload_const_bool,
    get_overload_const_bytes,
    get_overload_const_float,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_constant_dict,
    is_overload_constant_bool,
    is_overload_constant_bytes,
    is_overload_constant_dict,
    is_overload_constant_float,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
    to_nullable_type,
)

ll.add_symbol("array_size", hstr_ext.array_size)
ll.add_symbol("array_getptr1", hstr_ext.array_getptr1)


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
def typeof_pd_dataframe(val, c):
    from bodo.transforms.distributed_analysis import Distribution

    # convert "columns" from Index/MultiIndex to a tuple
    col_names = tuple(val.columns.to_list())
    col_types = get_hiframes_dtypes(val)
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

    return DataFrameType(col_types, index_typ, col_names, dist)


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_series(val, c):
    from bodo.transforms.distributed_analysis import Distribution

    dist = (
        Distribution(val._bodo_meta["dist"])
        if hasattr(val, "_bodo_meta") and val._bodo_meta is not None
        else Distribution.REP
    )
    return SeriesType(
        _infer_series_dtype(val),
        index=numba.typeof(val.index),
        name_typ=numba.typeof(val.name),
        dist=dist,
    )


@unbox(DataFrameType)
def unbox_dataframe(typ, val, c):
    """unbox dataframe to an empty DataFrame struct
    columns will be extracted later if necessary.
    """
    n_cols = len(typ.columns)

    # set all columns as not unboxed
    zero = c.context.get_constant(types.int8, 0)
    unboxed_tup = c.context.make_tuple(
        c.builder, types.UniTuple(types.int8, n_cols + 1), [zero] * (n_cols + 1)
    )

    # unbox index
    # TODO: unbox index only if necessary
    ind_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, ind_obj).value
    c.pyapi.decref(ind_obj)

    # set data arrays as null due to lazy unboxing
    # TODO: does this work for array types?
    data_nulls = [c.context.get_constant_null(t) for t in typ.data]
    data_tup = c.context.make_tuple(c.builder, types.Tuple(typ.data), data_nulls)

    dataframe_val = construct_dataframe(
        c.context, c.builder, typ, data_tup, index_val, unboxed_tup, val
    )

    return NativeValue(dataframe_val)


def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe"""

    # If the dataframe has typing metadata, pass the typing metadata for the given
    # column to _infer_series_dtype
    if (
        hasattr(df, "_bodo_meta")
        and df._bodo_meta is not None
        and "type_metadata" in df._bodo_meta
        and df._bodo_meta["type_metadata"] is not None
        # If the metadata hasn't updated but columns are added the information
        # is out of date and cannot be used.
        and len(df._bodo_meta["type_metadata"]) == len(df.columns)
    ):
        typing_metadata = df._bodo_meta["type_metadata"]
    else:
        typing_metadata = [None] * len(df.columns)
    hi_typs = [
        dtype_to_array_type(
            _infer_series_dtype(df.iloc[:, i], array_metadata=typing_metadata[i])
        )
        for i in range(len(df.columns))
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


# Map of types that can be mapped to a singular enum. Maps type -> enum
_one_to_one_type_to_enum_map = {
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
    bytes_type: SeriesDtypeEnum.BINARY.value,
    string_type: SeriesDtypeEnum.STRING.value,
    bodo.bool_: SeriesDtypeEnum.Bool.value,
    types.NoneType: SeriesDtypeEnum.NoneType.value,
}

# The reverse of the above map, Maps enum -> type
_one_to_one_enum_to_type_map = {
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
    SeriesDtypeEnum.BINARY.value: bytes_type,
    SeriesDtypeEnum.STRING.value: string_type,
    SeriesDtypeEnum.Bool.value: bodo.bool_,
    SeriesDtypeEnum.NoneType.value: types.NoneType,
}


def _dtype_from_type_enum_list(typ_enum_list):
    """Wrapper around _dtype_from_type_enum_list_recursor"""
    remaining, typ = _dtype_from_type_enum_list_recursor(typ_enum_list)
    if len(remaining) != 0:
        raise_bodo_error(
            f"Unexpected Internal Error while converting typing metadata: Dtype list was not fully consumed.\n Input typ_enum_list: {typ_enum_list}.\nRemainder: {remaining}. Please file the error here: https://github.com/Bodo-inc/Feedback"
        )
    return typ


def _dtype_from_type_enum_list_recursor(typ_enum_list):
    """
    Converts a list of type enums generated by _dtype_to_type_enum_list, and converts it
    back into a dtype.

    The general structure procedure as follows:

    The type enum list acts as a stack. At the begining of each call to
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

    if len(typ_enum_list) == 0:
        raise_bodo_error("Unable to infer dtype from empty typ_enum_list")
    elif typ_enum_list[0] in _one_to_one_enum_to_type_map:
        return (
            typ_enum_list[1:],
            _one_to_one_enum_to_type_map[typ_enum_list[0]],
        )
    # Integer array needs special handling, as integerArray.dtype does not return
    # a nullable integer type
    elif typ_enum_list[0] == SeriesDtypeEnum.IntegerArray.value:
        (remaining_typ_enum_list, typ) = _dtype_from_type_enum_list_recursor(
            typ_enum_list[1:]
        )
        return (remaining_typ_enum_list, IntegerArrayType(typ))
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
        if len(typ_enum_list) == 1:
            raise_bodo_error(
                f"Unexpected Internal Error while converting typing metadata: Encountered 'Literal' internal enum value with no value following it. Please file the error here: https://github.com/Bodo-inc/Feedback"
            )
        lit_val = typ_enum_list[1]
        remainder = typ_enum_list[2:]
        return remainder, lit_val
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
    elif typ_enum_list[0] == SeriesDtypeEnum.IntervalIndexType.value:
        # Constructor for IntervalIndexType
        # def __init__(self, data, name_typ=None)
        remainder, data = _dtype_from_type_enum_list_recursor(typ_enum_list[1:])
        remainder, name_type = _dtype_from_type_enum_list_recursor(remainder)
        return remainder, IntervalIndexType(data, name_type)
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

    else:
        raise_bodo_error(
            f"Unexpected Internal Error while converting typing metadata: unable to infer dtype for type enum {typ_enum_list[0]}. Please file the error here: https://github.com/Bodo-inc/Feedback"
        )


def _dtype_to_type_enum_list(typ):
    return guard(_dtype_to_type_enum_list_recursor, typ)


def _dtype_to_type_enum_list_recursor(typ):
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
    elif is_overload_constant_dict(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_constant_dict(typ)]
    elif is_overload_constant_int(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_int(typ)]
    # is_overload_constant_list also handles constant tuples
    elif is_overload_constant_list(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_list(typ)]
    elif is_overload_constant_str(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_str(typ)]
    elif is_overload_constant_bool(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_bool(typ)]
    elif is_overload_constant_bytes(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_bytes(typ)]
    elif is_overload_constant_float(typ):
        return [SeriesDtypeEnum.Literal.value, get_overload_const_float(typ)]
    elif is_overload_none(typ):
        return [SeriesDtypeEnum.Literal.value, None]
    # integer arrays need special handling, as integerArray's dtype is not a nullable integer
    elif isinstance(typ, IntegerArrayType):
        return [SeriesDtypeEnum.IntegerArray.value] + _dtype_to_type_enum_list_recursor(
            typ.dtype
        )
    elif bodo.utils.utils.is_array_typ(typ, False):
        return [SeriesDtypeEnum.ARRAY.value] + _dtype_to_type_enum_list_recursor(
            typ.dtype
        )
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
    elif isinstance(typ, IntervalIndexType):
        # Constructor for IntervalIndexType
        # def __init__(self, data, name_typ=None)
        return (
            [SeriesDtypeEnum.IntervalIndexType.value]
            + _dtype_to_type_enum_list_recursor(typ.data)
            + _dtype_to_type_enum_list_recursor(typ.name_typ)
        )
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


def _infer_series_dtype(S, array_metadata=None):

    # TODO: use this if all the values in S.values are NULL as well
    if S.dtype == np.dtype("O"):
        if len(S.values) == 0:
            if (
                hasattr(S, "_bodo_meta")
                and S._bodo_meta is not None
                and "type_metadata" in S._bodo_meta
            ):
                type_list = S._bodo_meta["type_metadata"]
                # If the Series itself has the typing metadata, it will be the original
                # dtype of the series
                return _dtype_from_type_enum_list(type_list)
            elif array_metadata != None:
                # If the metadata is passed by the dataframe, it is the type of the underlying array.

                # TODO: array metadata is going to return the type of the array, not the
                # type of the Series. This will return different types for null integer,
                # but for object series, I can't think of a situation in which the
                # dtypes would be different.
                return _dtype_from_type_enum_list(array_metadata).dtype

        return numba.typeof(S.values).dtype

    # nullable int dtype
    if isinstance(S.dtype, pd.core.arrays.integer._IntegerDtype):
        return typeof_pd_int_dtype(S.dtype, None)
    elif isinstance(S.dtype, pd.CategoricalDtype):
        return bodo.typeof(S.dtype)
    elif isinstance(S.dtype, pd.StringDtype):
        return string_type
    elif isinstance(S.dtype, pd.BooleanDtype):
        return types.bool_

    if isinstance(S.dtype, pd.DatetimeTZDtype):
        raise BodoError("Timezone-aware datetime data type not supported yet")

    # regular numpy types
    try:
        return numpy_support.from_dtype(S.dtype)
    except:  # pragma: no cover
        raise BodoError(f"data type {S.dtype} for column {S.name} not supported yet")


@box(DataFrameType)
def box_dataframe(typ, val, c):
    """Boxes native dataframe value into Python dataframe object, required for function
    return, printing, object mode, etc.
    Works by boxing individual data arrays.
    """
    context = c.context
    builder = c.builder
    pyapi = c.pyapi

    dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
        c.context, c.builder, typ, val
    )
    dataframe = cgutils.create_struct_proxy(typ)(context, builder, value=val)

    # see boxing of reflected list in Numba:
    # https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L561
    obj = dataframe.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    # df unboxed from Python
    has_parent = cgutils.is_not_null(builder, obj)

    # get column names object
    # TODO: avoid generating large tuples and lower a constant Index if possible
    # (e.g. if homogeneous names)
    columns_typ = numba.typeof(typ.columns)
    columns = context.get_constant_generic(builder, columns_typ, typ.columns)
    context.nrt.incref(builder, columns_typ, columns)
    columns_obj = pyapi.from_native_value(columns_typ, columns, c.env_manager)

    with c.builder.if_else(has_parent) as (use_parent, otherwise):
        with use_parent:
            pyapi.incref(obj)
            # set parent dataframe column names to numbers for robust setting of columns
            # df.columns = np.arange(len(df.columns))
            mod_name = context.insert_const_string(c.builder.module, "numpy")
            class_obj = pyapi.import_module_noblock(mod_name)
            n_cols_obj = pyapi.long_from_longlong(
                lir.Constant(lir.IntType(64), len(typ.columns))
            )
            col_nums_arr_obj = pyapi.call_method(class_obj, "arange", (n_cols_obj,))
            pyapi.object_setattr_string(obj, "columns", col_nums_arr_obj)
            pyapi.decref(class_obj)
            pyapi.decref(col_nums_arr_obj)
            pyapi.decref(n_cols_obj)
        with otherwise:
            # df_obj = pd.DataFrame(index=index)
            context.nrt.incref(builder, typ.index, dataframe_payload.index)
            index_obj = c.pyapi.from_native_value(
                typ.index, dataframe_payload.index, c.env_manager
            )

            mod_name = context.insert_const_string(c.builder.module, "pandas")
            class_obj = pyapi.import_module_noblock(mod_name)
            df_obj = pyapi.call_method(
                class_obj, "DataFrame", (pyapi.borrow_none(), index_obj)
            )
            pyapi.decref(class_obj)
            pyapi.decref(index_obj)
            builder.store(df_obj, res)

    # get data arrays and box them
    n_cols = len(typ.columns)
    col_arrs = [builder.extract_value(dataframe_payload.data, i) for i in range(n_cols)]
    arr_typs = typ.data
    for i, arr, arr_typ in zip(range(n_cols), col_arrs, arr_typs):
        # box array if df doesn't have a parent, or column was unboxed in function,
        # since changes in arrays like strings don't reflect back to parent object
        unboxed = builder.extract_value(dataframe_payload.unboxed, i)
        is_unboxed = builder.icmp_unsigned("==", unboxed, lir.Constant(unboxed.type, 1))
        box_array = builder.or_(
            builder.not_(has_parent), builder.and_(has_parent, is_unboxed)
        )

        with builder.if_then(box_array):
            # df[i] = boxed_arr
            c_ind_obj = pyapi.long_from_longlong(context.get_constant(types.int64, i))

            context.nrt.incref(builder, arr_typ, arr)
            arr_obj = pyapi.from_native_value(arr_typ, arr, c.env_manager)
            df_obj = builder.load(res)
            pyapi.object_setitem(df_obj, c_ind_obj, arr_obj)
            pyapi.decref(arr_obj)
            pyapi.decref(c_ind_obj)

    df_obj = builder.load(res)
    # set df columns separately to support repeated names and fix potential multi-index
    # issues, see test_dataframe.py::test_unbox_df_multi, test_box_repeated_names
    pyapi.object_setattr_string(df_obj, "columns", columns_obj)
    pyapi.decref(columns_obj)

    _set_bodo_meta_dataframe(c, df_obj, typ)

    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return df_obj


@intrinsic
def unbox_dataframe_column(typingctx, df, i=None):
    assert isinstance(df, DataFrameType) and is_overload_constant_int(i)

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        c = numba.core.pythonapi._UnboxContext(context, builder, pyapi)
        gil_state = pyapi.gil_ensure()  # acquire GIL

        df_typ = sig.args[0]
        col_ind = get_overload_const_int(sig.args[1])
        data_typ = df_typ.data[col_ind]
        # TODO: refcounts?

        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        # generate df.iloc[:,i] for parent dataframe object
        none_obj = c.pyapi.borrow_none()
        slice_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(slice))
        slice_obj = c.pyapi.call_function_objargs(slice_class_obj, [none_obj])
        col_ind_obj = c.pyapi.long_from_longlong(args[1])
        slice_ind_tup_obj = c.pyapi.tuple_pack([slice_obj, col_ind_obj])

        df_iloc_obj = c.pyapi.object_getattr_string(dataframe.parent, "iloc")
        series_obj = c.pyapi.object_getitem(df_iloc_obj, slice_ind_tup_obj)
        arr_obj_orig = c.pyapi.object_getattr_string(series_obj, "values")

        if isinstance(data_typ, types.Array):
            # call np.ascontiguousarray() on array since it may not be contiguous
            # the typing infrastructure assumes C-contiguous arrays
            # see test_df_multi_get_level() for an example of non-contiguous input
            np_mod_name = c.context.insert_const_string(c.builder.module, "numpy")
            np_class_obj = c.pyapi.import_module_noblock(np_mod_name)
            arr_obj = c.pyapi.call_method(
                np_class_obj, "ascontiguousarray", (arr_obj_orig,)
            )
            c.pyapi.decref(arr_obj_orig)
            c.pyapi.decref(np_class_obj)
        else:
            arr_obj = arr_obj_orig

        # TODO: support column of tuples?
        native_val = _unbox_series_data(data_typ.dtype, data_typ, arr_obj, c)

        c.pyapi.decref(slice_class_obj)
        c.pyapi.decref(slice_obj)
        c.pyapi.decref(col_ind_obj)
        c.pyapi.decref(slice_ind_tup_obj)
        c.pyapi.decref(df_iloc_obj)
        c.pyapi.decref(series_obj)
        c.pyapi.decref(arr_obj)
        pyapi.gil_release(gil_state)  # release GIL

        # assign array and set unboxed flag
        dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
            c.context, c.builder, df_typ, args[0]
        )
        dataframe_payload.data = builder.insert_value(
            dataframe_payload.data, native_val.value, col_ind
        )
        dataframe_payload.unboxed = builder.insert_value(
            dataframe_payload.unboxed, context.get_constant(types.int8, 1), col_ind
        )

        # store payload
        payload_type = DataFramePayloadType(df_typ)
        payload_ptr = context.nrt.meminfo_data(builder, dataframe.meminfo)
        ptrty = context.get_value_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        builder.store(dataframe_payload._getvalue(), payload_ptr)

    return signature(types.none, df, i), codegen


@unbox(SeriesType)
def unbox_series(typ, val, c):
    arr_obj_orig = c.pyapi.object_getattr_string(val, "values")

    if isinstance(typ.data, types.Array):
        # make contiguous by calling np.ascontiguousarray()
        np_mod_name = c.context.insert_const_string(c.builder.module, "numpy")
        np_class_obj = c.pyapi.import_module_noblock(np_mod_name)
        arr_obj = c.pyapi.call_method(
            np_class_obj, "ascontiguousarray", (arr_obj_orig,)
        )
        c.pyapi.decref(arr_obj_orig)
        c.pyapi.decref(np_class_obj)
    else:
        arr_obj = arr_obj_orig

    data_val = _unbox_series_data(typ.dtype, typ.data, arr_obj, c).value

    index_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, index_obj).value

    name_obj = c.pyapi.object_getattr_string(val, "name")
    name_val = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    series_val = bodo.hiframes.pd_series_ext.construct_series(
        c.context, c.builder, typ, data_val, index_val, name_val
    )
    # TODO: set parent pointer
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
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

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
    dtype = c.pyapi.make_none()  # TODO: dtype
    res = c.pyapi.call_method(
        pd_class_obj, "Series", (arr_obj, index_obj, dtype, name_obj)
    )
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(name_obj)

    _set_bodo_meta_series(res, c, typ)

    c.pyapi.decref(pd_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


def _set_bodo_meta_dataframe(c, obj, typ):
    """set Bodo metadata in output so the next JIT call knows data distribution, and
    the datatypes of the arrays that make up the dataframe.
    e.g. df._bodo_meta = {"dist": 5, "type_metadata": [[*INT_ARRAY_ENUM_LIST*], ["STRING_ARRAY_ENUM_LSIT"]]}
    """
    pyapi = c.pyapi
    context = c.context
    builder = c.builder

    # Setting meta for the array types contained within the dataframe,
    # So that we can infer the dtypes if an empty dataframe is passed from bodo
    # to pandas, and then back to a bodo fn.
    col_typs = []
    for dtype in typ.data:
        numba_typ_list = []
        typ_list = _dtype_to_type_enum_list(dtype)
        if typ_list != None:
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
                        typ_enum_typ, enum_llvm_const, c.env_manager
                    )
                numba_typ_list.append(cur_val_obj)

            array_type_metadata_obj = pyapi.list_pack(numba_typ_list)
            for val in numba_typ_list:
                pyapi.decref(val)
            col_typs.append(array_type_metadata_obj)
        else:
            numba_none_typ = numba.typeof(None)
            none_llvm_const = context.get_constant_generic(
                builder, numba_none_typ, None
            )
            none_val_obj = pyapi.from_native_value(
                numba_none_typ, none_llvm_const, c.env_manager
            )
            array_type_metadata_obj = none_val_obj

    meta_dict_obj = pyapi.dict_new(2)

    df_type_metadata_obj = pyapi.list_pack(col_typs)

    for val in col_typs:
        pyapi.decref(val)

    # using the distribution number since easier to handle
    dist_val_obj = pyapi.long_from_longlong(
        lir.Constant(lir.IntType(64), typ.dist.value)
    )

    pyapi.dict_setitem_string(meta_dict_obj, "dist", dist_val_obj)
    pyapi.dict_setitem_string(meta_dict_obj, "type_metadata", df_type_metadata_obj)
    pyapi.object_setattr_string(obj, "_bodo_meta", meta_dict_obj)
    pyapi.decref(meta_dict_obj)
    pyapi.decref(dist_val_obj)


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

    # handle hetrogenous series, and nullable integer Series.
    dtype = get_series_dtype_handle_null_int_and_hetrogenous(typ)

    # dtype == None if hetrogenous series
    if dtype != None:
        numba_typ_list = []
        typ_list = _dtype_to_type_enum_list(dtype)
        if typ_list != None:
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
                        typ_enum_typ, enum_llvm_const, c.env_manager
                    )

                numba_typ_list.append(cur_val_obj)

            type_metadata_obj = pyapi.list_pack(numba_typ_list)
            for val in numba_typ_list:
                pyapi.decref(val)

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
        raise BodoError("Unsupported array dtype: {}".format(val.dtype))

    # XXX assuming the whole array is strings if 1st val is string
    i = 0
    # skip NAs and empty lists/arrays (for array(item) array cases)
    # is_scalar call necessary since pd.isna() treats list of string as array
    while i < len(val) and (
        (pd.api.types.is_scalar(val[i]) and pd.isna(val[i]))
        or (not pd.api.types.is_scalar(val[i]) and len(val[i]) == 0)
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
        return string_array_type

    first_val = val[i]
    if isinstance(first_val, str):
        return string_array_type
    elif isinstance(first_val, bytes):
        return binary_array_type
    elif isinstance(first_val, bool):
        return bodo.libs.bool_arr_ext.boolean_array
    elif isinstance(first_val, (int, np.int32, np.int64)):
        return bodo.libs.int_arr_ext.IntegerArrayType(numba.typeof(first_val))
    # assuming object arrays with dictionary values string keys are struct arrays, which
    # means all keys are string and match across dictionaries, and all values with same
    # key have same data type
    # TODO: distinguish between Struct and Map arrays properly
    elif isinstance(first_val, (dict, Dict)) and all(
        isinstance(k, str) for k in first_val.keys()
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
            pd.arrays.BooleanArray,
            pd.arrays.IntegerArray,
            pd.arrays.StringArray,
        ),
    ):
        # normalize list to array, 'np.object_' dtype to consider potential nulls
        if isinstance(first_val, list):
            first_val = _value_to_array(first_val)
        val_typ = numba.typeof(first_val)
        return ArrayItemArrayType(val_typ)
    if isinstance(first_val, datetime.date):
        return datetime_date_array_type
    if isinstance(first_val, datetime.timedelta):
        return datetime_timedelta_array_type
    if isinstance(first_val, decimal.Decimal):
        # NOTE: converting decimal.Decimal objects to 38/18, same as Spark
        return DecimalArrayType(38, 18)

    raise BodoError(
        "Unsupported object array with first value: {}".format(first_val)
    )  # pragma: no cover


def _value_to_array(val):
    """convert list or dict value to object array for typing purposes"""
    assert isinstance(val, (list, dict, Dict))
    if isinstance(val, (dict, Dict)):
        val = dict(val)
        return np.array([val], np.object_)

    # add None to list to avoid Numpy's automatic conversion to 2D arrays
    val_infer = val.copy()
    val_infer.append(None)
    arr = np.array(val_infer, np.object_)

    # assume float lists can be regular np.float64 arrays
    # TODO handle corener cases where None could be used as NA instead of np.nan
    if len(val) and isinstance(val[0], float):
        arr = np.array(val, np.float64)
    return arr


def _get_struct_value_arr_type(v):
    """get data array type for a field value of a struct array"""
    if isinstance(v, (dict, Dict)):
        return numba.typeof(_value_to_array(v))

    if isinstance(v, list):
        return dtype_to_array_type(numba.typeof(_value_to_array(v)))

    if pd.api.types.is_scalar(v) and pd.isna(v):
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
    # use nullable arrays for integer/bool values since there could be None objects
    if isinstance(v, (int, bool)):
        arr_typ = to_nullable_type(arr_typ)

    return arr_typ


# TODO: support array of strings
# @typeof_impl.register(np.ndarray)
# def typeof_np_string(val, c):
#     arr_typ = numba.core.typing.typeof._typeof_ndarray(val, c)
#     # match string dtype
#     if isinstance(arr_typ.dtype, (types.UnicodeCharSeq, types.CharSeq)):
#         return string_array_type
#     return arr_typ
