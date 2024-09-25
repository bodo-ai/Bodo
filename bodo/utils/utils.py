# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Collection of utility functions. Needs to be refactored in separate files.
"""
import functools
import hashlib
import inspect
import keyword
import re
import traceback
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, TypeGuard

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, types
from numba.core.imputils import lower_builtin, lower_constant
from numba.core.ir_utils import (
    find_callname,
    find_const,
    get_definition,
    guard,
    mk_unique_var,
    require,
)
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import intrinsic, overload
from numba.np.arrayobj import get_itemsize, make_array, populate_array
from numba.np.numpy_support import as_dtype

import bodo
from bodo.hiframes.pd_timestamp_ext import PandasTimestampType
from bodo.hiframes.timestamptz_ext import timestamptz_array_type, timestamptz_type
from bodo.libs.binary_arr_ext import bytes_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.null_arr_ext import null_dtype
from bodo.libs.pd_datetime_arr_ext import (
    DatetimeArrayType,
    PandasDatetimeTZDtype,
)
from bodo.libs.str_arr_ext import (
    num_total_chars,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.libs.str_ext import string_type
from bodo.mpi4py import MPI
from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.typing import (
    NOT_CONSTANT,
    BodoError,
    BodoWarning,
    MetaType,
    is_overload_none,
    is_str_arr_type,
)

int128_type = types.Integer("int128", 128)


# int values for types to pass to C code
# XXX: These are defined in _bodo_common.h and must match here
class CTypeEnum(Enum):
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
    Date = 13
    Time = 14
    Datetime = 15
    Timedelta = 16
    # NOTE: currently, only used for handling decimal array's data array for scatterv
    # since it handles the data array inside decimal array separately
    Int128 = 17
    LIST = 18
    STRUCT = 19
    BINARY = 20
    COMPLEX64 = 21
    COMPLEX128 = 22
    Map = 23
    TIMESTAMPTZ = 24  # Used to catch gaps based on dtype in C++


_numba_to_c_type_map = {
    types.int8: CTypeEnum.Int8.value,
    types.uint8: CTypeEnum.UInt8.value,
    types.int32: CTypeEnum.Int32.value,
    types.uint32: CTypeEnum.UInt32.value,
    types.int64: CTypeEnum.Int64.value,
    types.uint64: CTypeEnum.UInt64.value,
    types.float32: CTypeEnum.Float32.value,
    types.float64: CTypeEnum.Float64.value,
    types.NPDatetime("ns"): CTypeEnum.Datetime.value,
    types.NPTimedelta("ns"): CTypeEnum.Timedelta.value,
    types.bool_: CTypeEnum.Bool.value,
    types.int16: CTypeEnum.Int16.value,
    types.uint16: CTypeEnum.UInt16.value,
    int128_type: CTypeEnum.Int128.value,
    bodo.hiframes.datetime_date_ext.datetime_date_type: CTypeEnum.Date.value,
    # TODO: Timedelta arrays need to be supported
    # bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type: CTypeEnum.Timedelta.value,
    types.unicode_type: CTypeEnum.STRING.value,
    bodo.libs.binary_arr_ext.bytes_type: CTypeEnum.BINARY.value,
    # Null arrays are passed as nullable bool arrays to C++ currently.
    # TODO[BSE-433]: support null dtype in C++ directly
    # https://github.com/Bodo-inc/Bodo/blob/b9b38a8643d61a5038bcf4a3a5dff4f14040b76c/bodo/libs/_array.cpp#L165
    null_dtype: CTypeEnum.Bool.value,
    types.complex64: CTypeEnum.COMPLEX64.value,
    types.complex128: CTypeEnum.COMPLEX128.value,
    timestamptz_type: CTypeEnum.TIMESTAMPTZ.value,
}


# int values for array types to pass to C code
# XXX: These are defined in _bodo_common.h and must match here
class CArrayTypeEnum(Enum):
    NUMPY = 0
    STRING = 1
    NULLABLE_INT_BOOL = 2
    STRUCT = 3
    CATEGORICAL = 4
    ARRAY_ITEM = 5
    INTERVAL = 6
    DICT = 7  # dictionary-encoded string array
    MAP = 8
    TIMESTAMPTZ = 9


# silence Numba error messages for now
# TODO: customize through @bodo.jit
numba.core.errors.error_extras = {
    "unsupported_error": "",
    "typing": "",
    "reportable": "",
    "interpreter": "",
    "constant_inference": "",
}


np_alloc_callnames = ("empty", "zeros", "ones", "full")


# size threshold for throwing warning for const dictionary lowering (in slow path)
CONST_DICT_SLOW_WARN_THRESHOLD = 100


# size threshold for throwing warning for const list lowering
CONST_LIST_SLOW_WARN_THRESHOLD = 100000


def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


def get_constant(func_ir, var, default=NOT_CONSTANT):
    def_node = guard(get_definition, func_ir, var)
    if def_node is None:
        return default
    if isinstance(def_node, ir.Const):
        return def_node.value
    # call recursively if variable assignment
    if isinstance(def_node, ir.Var):
        return get_constant(func_ir, def_node, default)
    return default


def numba_to_c_type(t) -> int:  # pragma: no cover
    """
    Derive the enum value for the dtype of the array being passed to C++. Semi-structured array types (ArrayItemArrayType, StructArrayType, MapArrayType, TupleArrayType) are not supported. Please use numba_to_c_types instead.

    Args:
        t: Dtype that needs to be passed to C++.

    Returns:
        int: The value for the CTypeEnum value
    """
    if isinstance(t, bodo.libs.decimal_arr_ext.Decimal128Type):
        return CTypeEnum.Decimal.value
    elif isinstance(t, PandasDatetimeTZDtype):
        return CTypeEnum.Datetime.value
    elif isinstance(t, PandasTimestampType):
        return CTypeEnum.Int64.value
    elif isinstance(t, bodo.hiframes.time_ext.TimeType):
        return CTypeEnum.Time.value
    else:
        return _numba_to_c_type_map[t]


def numba_to_c_types(
    arr_types: Iterable[types.ArrayCompatible],
) -> np.ndarray:  # pragma: no cover
    """
    Derive the enum value for a list of array dtypes passed to C++.

    Args:
        arr_types: The list of array dtypes that needs to be passed to C++.

    Returns:
        List: The values for their CTypeEnum values
    """
    c_types = []
    for arr_type in arr_types:
        if isinstance(arr_type, (bodo.StructArrayType, bodo.TupleArrayType)):
            c_types.append(CTypeEnum.STRUCT.value)
            c_types.append(len(arr_type.data))
            c_types.extend(numba_to_c_types(arr_type.data))
        elif isinstance(arr_type, bodo.MapArrayType):
            c_types.append(CTypeEnum.Map.value)
            c_types.extend(
                numba_to_c_types((arr_type.key_arr_type, arr_type.value_arr_type))
            )
        elif isinstance(arr_type, bodo.ArrayItemArrayType):
            c_types.append(CTypeEnum.LIST.value)
            c_types.extend(numba_to_c_types((arr_type.dtype,)))
        elif isinstance(arr_type, bodo.DecimalArrayType):
            c_types.append(numba_to_c_type(arr_type.dtype))
            c_types.append(arr_type.dtype.precision)
            c_types.append(arr_type.dtype.scale)
        else:
            c_types.append(numba_to_c_type(arr_type.dtype))
    return np.array(c_types, dtype=np.int8)


def numba_to_c_array_type(arr_type: types.ArrayCompatible) -> int:  # pragma: no cover
    """
    Derive the enum value for the array being passed to C++. Semi-structured array types (ArrayItemArrayType, StructArrayType, MapArrayType, TupleArrayType) are not supported. Please use numba_to_c_array_types instead.

    Args:
        arr_type (types.ArrayCompatible): An array type that needs
        to be passed to C++.

    Returns:
        int: The value for the CArrayTypeEnum value
    """
    if isinstance(arr_type, types.Array):
        return CArrayTypeEnum.NUMPY.value
    elif arr_type == bodo.string_array_type or arr_type == bodo.binary_array_type:
        return CArrayTypeEnum.STRING.value
    elif arr_type in (
        bodo.null_array_type,
        bodo.datetime_date_array_type,
        bodo.boolean_array_type,
    ) or isinstance(
        arr_type,
        (
            bodo.IntegerArrayType,
            bodo.FloatingArrayType,
            bodo.TimeArrayType,
            bodo.DecimalArrayType,
            bodo.DatetimeArrayType,
        ),
    ):
        return CArrayTypeEnum.NULLABLE_INT_BOOL.value
    elif isinstance(arr_type, bodo.CategoricalArrayType):
        return CArrayTypeEnum.CATEGORICAL.value
    elif isinstance(arr_type, bodo.IntervalArrayType):
        return CArrayTypeEnum.INTERVAL.value
    elif arr_type == timestamptz_array_type:
        return CArrayTypeEnum.TIMESTAMPTZ.value
    elif arr_type == bodo.dict_str_arr_type:
        return CArrayTypeEnum.DICT.value
    else:
        raise BodoError(f"Unsupported Array Type '{arr_type}' in numba_to_c_array_type")


def numba_to_c_array_types(
    arr_types: Iterable[types.ArrayCompatible],
) -> np.ndarray:  # pragma: no cover
    """
    Derive the enum value for a list of array dtypes passed to C++.

    Args:
        arr_types (List[types.ArrayCompatible]): The list of array types that needs to be passed to C++.

    Returns:
        List: The values for their CArrayTypeEnum values
    """
    c_arr_types = []
    for arr_type in arr_types:
        if isinstance(arr_type, (bodo.StructArrayType, bodo.TupleArrayType)):
            c_arr_types.append(CArrayTypeEnum.STRUCT.value)
            c_arr_types.append(len(arr_type.data))
            c_arr_types.extend(numba_to_c_array_types(arr_type.data))
        elif isinstance(arr_type, bodo.MapArrayType):
            c_arr_types.append(CArrayTypeEnum.MAP.value)
            c_arr_types.extend(
                numba_to_c_array_types((arr_type.key_arr_type, arr_type.value_arr_type))
            )
        elif isinstance(arr_type, bodo.ArrayItemArrayType):
            c_arr_types.append(CArrayTypeEnum.ARRAY_ITEM.value)
            c_arr_types.extend(numba_to_c_array_types((arr_type.dtype,)))
        elif isinstance(arr_type, bodo.DecimalArrayType):
            c_arr_types.append(numba_to_c_array_type(arr_type))
            c_arr_types.append(arr_type.dtype.precision)
            c_arr_types.append(arr_type.dtype.scale)
        else:
            c_arr_types.append(numba_to_c_array_type(arr_type))
    return np.array(c_arr_types, dtype=np.int8)


def is_alloc_callname(func_name, mod_name):
    """
    return true if function represents an array creation call
    """
    return isinstance(mod_name, str) and (
        (mod_name == "numpy" and func_name in np_alloc_callnames)
        or (
            func_name == "empty_inferred"
            and mod_name in ("numba.extending", "numba.np.unsafe.ndarray")
        )
        or (
            func_name == "pre_alloc_string_array"
            and mod_name == "bodo.libs.str_arr_ext"
        )
        or (
            func_name == "pre_alloc_binary_array"
            and mod_name == "bodo.libs.binary_arr_ext"
        )
        or (
            func_name == "alloc_random_access_string_array"
            and mod_name == "bodo.libs.str_ext"
        )
        or (
            func_name == "pre_alloc_array_item_array"
            and mod_name == "bodo.libs.array_item_arr_ext"
        )
        or (
            func_name == "pre_alloc_struct_array"
            and mod_name == "bodo.libs.struct_arr_ext"
        )
        or (func_name == "pre_alloc_map_array" and mod_name == "bodo.libs.map_arr_ext")
        or (
            func_name == "pre_alloc_tuple_array"
            and mod_name == "bodo.libs.tuple_arr_ext"
        )
        or (func_name == "alloc_bool_array" and mod_name == "bodo.libs.bool_arr_ext")
        or (
            func_name == "alloc_false_bool_array"
            and mod_name == "bodo.libs.bool_arr_ext"
        )
        or (
            func_name == "alloc_true_bool_array"
            and mod_name == "bodo.libs.bool_arr_ext"
        )
        or (func_name == "alloc_int_array" and mod_name == "bodo.libs.int_arr_ext")
        or (func_name == "alloc_float_array" and mod_name == "bodo.libs.float_arr_ext")
        or (
            func_name == "alloc_datetime_date_array"
            and mod_name == "bodo.hiframes.datetime_date_ext"
        )
        or (
            func_name == "alloc_datetime_timedelta_array"
            and mod_name == "bodo.hiframes.datetime_timedelta_ext"
        )
        or (
            func_name == "alloc_decimal_array"
            and mod_name == "bodo.libs.decimal_arr_ext"
        )
        or (
            func_name == "alloc_categorical_array"
            and mod_name == "bodo.hiframes.pd_categorical_ext"
        )
        or (func_name == "gen_na_array" and mod_name == "bodo.libs.array_kernels")
        or (
            func_name == "alloc_pd_datetime_array"
            and mod_name == "bodo.libs.pd_datetime_arr_ext"
        )
        or (func_name == "alloc_time_array" and mod_name == "bodo.hiframes.time_ext")
        or (
            func_name == "alloc_timestamptz_array"
            and mod_name == "bodo.hiframes.timestamptz_ext"
        )
        or (func_name == "init_null_array" and mod_name == "bodo.libs.null_arr_ext")
        or (func_name == "full_type" and mod_name == "bodo.utils.utils")
    )


def find_build_tuple(func_ir, var, handle_const_tuple=False):
    """Check if a variable is constructed via build_tuple
    and return the sequence or raise GuardException otherwise.
    The output sequence can be a list/tuple of ir.Vars or a tuple of constant values.
    'handle_const_tuple=True' allows constant tuples to be returned. Otherwise,
    only a sequence of ir.Vars can be returned (which may be a requirement in the
    caller).
    """
    # variable or variable name
    require(isinstance(var, (ir.Var, str)))
    var_def = get_definition(func_ir, var)
    if isinstance(var_def, ir.Expr):
        require(var_def.op == "build_tuple")
        return var_def.items

    # Array analysis may convert tuples to an ir.Const expression:
    # https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/parfors/array_analysis.py#L2186
    require(handle_const_tuple)
    require(isinstance(var_def, ir.Const))
    require(isinstance(var_def.value, tuple))
    return var_def.value


# print function used for debugging that uses printf in C, instead of Numba's print that
# calls Python's print in object mode (which can fail sometimes)
def cprint(*s):  # pragma: no cover
    print(*s)


@infer_global(cprint)
class CprintInfer(AbstractTemplate):  # pragma: no cover
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *unliteral_all(args))


typ_to_format = {
    types.int32: "d",
    types.uint32: "u",
    types.int64: "lld",
    types.uint64: "llu",
    types.float32: "f",
    types.float64: "lf",
    types.voidptr: "s",
}


@lower_builtin(cprint, types.VarArg(types.Any))
def cprint_lower(context, builder, sig, args):  # pragma: no cover
    for i, val in enumerate(args):
        typ = sig.args[i]
        if isinstance(typ, types.ArrayCTypes):
            cgutils.printf(builder, "%p ", val)
            continue
        format_str = typ_to_format[typ]
        cgutils.printf(builder, "%{} ".format(format_str), val)
    cgutils.printf(builder, "\n")
    return context.get_dummy_value()


def is_whole_slice(typemap, func_ir, var, accept_stride=False):
    """return True if var can be determined to be a whole slice"""
    require(
        typemap[var.name] == types.slice2_type
        or (accept_stride and typemap[var.name] == types.slice3_type)
    )
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
    assert len(call_expr.args) == 2 or (accept_stride and len(call_expr.args) == 3)
    assert find_callname(func_ir, call_expr) == ("slice", "builtins")
    arg0_def = get_definition(func_ir, call_expr.args[0])
    arg1_def = get_definition(func_ir, call_expr.args[1])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
    require(isinstance(arg1_def, ir.Const) and arg1_def.value == None)
    return True


def is_slice_equiv_arr(arr_var, index_var, func_ir, equiv_set, accept_stride=False):
    """check whether 'index_var' is a slice equivalent to first dimension of 'arr_var'.
    Note: array analysis replaces some slices with 0:n form.
    """
    # index definition should be a slice() call
    index_def = get_definition(func_ir, index_var)
    require(find_callname(func_ir, index_def) == ("slice", "builtins"))
    require(len(index_def.args) in (2, 3))

    # start of slice should be 0
    require(find_const(func_ir, index_def.args[0]) in (0, None))

    # slice size should be the same as first dimension of array
    require(equiv_set.is_equiv(index_def.args[1], arr_var.name + "#0"))

    # check strides
    require(
        accept_stride
        or len(index_def.args) == 2
        or find_const(func_ir, index_def.args[2]) == 1
    )
    return True


# def is_const_slice(typemap, func_ir, var, accept_stride=False):
#     """ return True if var can be determined to be a constant size slice """
#     require(
#         typemap[var.name] == types.slice2_type
#         or (accept_stride and typemap[var.name] == types.slice3_type)
#     )
#     call_expr = get_definition(func_ir, var)
#     require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
#     assert len(call_expr.args) == 2 or (accept_stride and len(call_expr.args) == 3)
#     assert find_callname(func_ir, call_expr) == ("slice", "builtins")
#     arg0_def = get_definition(func_ir, call_expr.args[0])
#     require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
#     size_const = find_const(func_ir, call_expr.args[1])
#     require(isinstance(size_const, int))
#     return True


def get_slice_step(typemap, func_ir, var):
    require(typemap[var.name] == types.slice3_type)
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
    assert len(call_expr.args) == 3
    return call_expr.args[2]


def is_array_typ(
    var_typ, include_index_series=True
) -> TypeGuard[types.ArrayCompatible]:
    """return True if var_typ is an array type.
    include_index_series=True also includes Index and Series types (as "array-like").
    """

    # NOTE: make sure all Bodo arrays are here
    return isinstance(var_typ, bodo.libs.array_kernels.BODO_ARRAY_TYPE_CLASSES) or (
        include_index_series
        and (
            isinstance(
                var_typ,
                (
                    bodo.hiframes.pd_series_ext.SeriesType,
                    bodo.hiframes.pd_multi_index_ext.MultiIndexType,
                ),
            )
            or bodo.hiframes.pd_index_ext.is_pd_index_type(var_typ)
        )
    )


def is_np_array_typ(var_typ):
    return isinstance(var_typ, types.Array)


# TODO: fix tuple, dataframe distribution
def is_distributable_typ(var_typ):
    return (
        is_array_typ(var_typ)
        or isinstance(var_typ, bodo.hiframes.table.TableType)
        or isinstance(var_typ, bodo.hiframes.pd_dataframe_ext.DataFrameType)
        or (isinstance(var_typ, types.List) and is_distributable_typ(var_typ.dtype))
        or (
            isinstance(var_typ, types.DictType)
            # only dictionary values can be distributed since keys should be hashable
            and is_distributable_typ(var_typ.value_type)
        )
    )


def is_distributable_tuple_typ(var_typ):
    try:
        from bodosql.context_ext import BodoSQLContextType
    except ImportError:  # pragma: no cover
        # ignore if None.
        BodoSQLContextType = None
    return (
        (
            isinstance(var_typ, types.BaseTuple)
            and any(
                is_distributable_typ(t) or is_distributable_tuple_typ(t)
                for t in var_typ.types
            )
        )
        or (
            isinstance(var_typ, types.List)
            and is_distributable_tuple_typ(var_typ.dtype)
        )
        or (
            isinstance(var_typ, types.DictType)
            and is_distributable_tuple_typ(var_typ.value_type)
        )
        or (
            isinstance(var_typ, types.iterators.EnumerateType)
            and (
                is_distributable_typ(var_typ.yield_type[1])
                or is_distributable_tuple_typ(var_typ.yield_type[1])
            )
        )
        or (
            BodoSQLContextType is not None
            and isinstance(var_typ, BodoSQLContextType)
            and any([is_distributable_typ(df) for df in var_typ.dataframes])
        )
    )


@numba.generated_jit(nopython=True, cache=True)
def build_set_seen_na(A):
    """
    Function to build a set from A, omitting
    any NA values. This returns two values,
    the newly created set, and if any NA values
    were encountered. This separates avoids any
    NA values in the set, including NA, NaN,
    and NaT.
    """
    # TODO: Merge with build_set. These are currently
    # separate because this is only used by nunique and
    # build set is potentially used in many locations.

    # TODO: use more efficient hash table optimized for addition and
    # membership check
    # XXX using dict for now due to Numba's #4577
    def impl(A):  # pragma: no cover
        s = dict()
        seen_na = False
        for i in range(len(A)):
            if bodo.libs.array_kernels.isna(A, i):
                seen_na = True
                continue
            s[A[i]] = 0
        return s, seen_na

    return impl


def empty_like_type(n, arr):  # pragma: no cover
    return np.empty(n, arr.dtype)


@overload(empty_like_type, no_unliteral=True)
def empty_like_type_overload(n, arr):
    # categorical
    if isinstance(arr, bodo.hiframes.pd_categorical_ext.CategoricalArrayType):
        return lambda n, arr: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(
            n, arr.dtype
        )  # pragma: no cover

    if isinstance(arr, types.Array):
        return lambda n, arr: np.empty(n, arr.dtype)  # pragma: no cover

    if isinstance(arr, types.List) and arr.dtype == string_type:

        def empty_like_type_str_list(n, arr):  # pragma: no cover
            return [""] * n

        return empty_like_type_str_list

    if isinstance(arr, types.List) and arr.dtype == bytes_type:

        def empty_like_type_binary_list(n, arr):  # pragma: no cover
            return [b""] * n

        return empty_like_type_binary_list

    # nullable int arr
    if isinstance(arr, IntegerArrayType):
        _dtype = arr.dtype

        def empty_like_type_int_arr(n, arr):  # pragma: no cover
            return bodo.libs.int_arr_ext.alloc_int_array(n, _dtype)

        return empty_like_type_int_arr

    # nullable float arr
    if isinstance(arr, FloatingArrayType):  # pragma: no cover
        _dtype = arr.dtype

        def empty_like_type_float_arr(n, arr):  # pragma: no cover
            return bodo.libs.float_arr_ext.alloc_float_array(n, _dtype)

        return empty_like_type_float_arr

    if arr == boolean_array_type:

        def empty_like_type_bool_arr(n, arr):  # pragma: no cover
            return bodo.libs.bool_arr_ext.alloc_bool_array(n)

        return empty_like_type_bool_arr

    if arr == bodo.hiframes.datetime_date_ext.datetime_date_array_type:

        def empty_like_type_datetime_date_arr(n, arr):  # pragma: no cover
            return bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)

        return empty_like_type_datetime_date_arr

    if isinstance(arr, DatetimeArrayType):
        tz = arr.tz

        def empty_like_pandas_datetime_arr(n, arr):  # pragma: no cover
            return bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(n, tz)

        return empty_like_pandas_datetime_arr

    if isinstance(arr, bodo.hiframes.time_ext.TimeArrayType):
        precision = arr.precision

        def empty_like_type_time_arr(n, arr):
            return bodo.hiframes.time_ext.alloc_time_array(n, precision)

        return empty_like_type_time_arr

    if arr == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_array_type:

        def empty_like_type_datetime_timedelta_arr(n, arr):  # pragma: no cover
            return bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(
                n
            )

        return empty_like_type_datetime_timedelta_arr
    if isinstance(arr, bodo.libs.decimal_arr_ext.DecimalArrayType):
        precision = arr.precision
        scale = arr.scale

        def empty_like_type_decimal_arr(n, arr):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.alloc_decimal_array(n, precision, scale)

        return empty_like_type_decimal_arr

    if arr == timestamptz_array_type:

        def empty_like_timestamp_tz_arr(n, arr):  # pragma: no cover
            return bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(n)

        return empty_like_timestamp_tz_arr

    # string array buffer for join
    assert arr == string_array_type

    def empty_like_type_str_arr(n, arr):  # pragma: no cover
        # average character heuristic
        avg_chars = 20  # heuristic
        if len(arr) != 0:
            avg_chars = num_total_chars(arr) // len(arr)
        return pre_alloc_string_array(n, n * avg_chars)

    return empty_like_type_str_arr


# copied from numba.np.arrayobj (0.47), except the raising exception code is
# changed to just a print since unboxing call convention throws an error for exceptions
def _empty_nd_impl(context, builder, arrtype, shapes):  # pragma: no cover
    """Utility function used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """

    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    overflow = lir.Constant(lir.IntType(1), 0)
    for s in shapes:
        arrlen_mult = builder.smul_with_overflow(arrlen, s)
        arrlen = builder.extract_value(arrlen_mult, 0)
        overflow = builder.or_(overflow, builder.extract_value(arrlen_mult, 1))

    if arrtype.ndim == 0:
        strides = ()
    elif arrtype.layout == "C":
        strides = [itemsize]
        for dimension_size in reversed(shapes[1:]):
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(reversed(strides))
    elif arrtype.layout == "F":
        strides = [itemsize]
        for dimension_size in shapes[:-1]:
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(strides)
    else:
        raise NotImplementedError(
            "Don't know how to allocate array with layout '{0}'.".format(arrtype.layout)
        )

    # Check overflow, numpy also does this after checking order
    allocsize_mult = builder.smul_with_overflow(arrlen, itemsize)
    allocsize = builder.extract_value(allocsize_mult, 0)
    overflow = builder.or_(overflow, builder.extract_value(allocsize_mult, 1))

    with builder.if_then(overflow, likely=False):
        cgutils.printf(
            builder,
            (
                "array is too big; `arr.size * arr.dtype.itemsize` is larger than"
                " the maximum possible size."
            ),
        )

    dtype = arrtype.dtype
    align_val = context.get_preferred_array_alignment(dtype)
    align = context.get_constant(types.uint32, align_val)
    meminfo = context.nrt.meminfo_alloc_aligned(builder, size=allocsize, align=align)
    data = context.nrt.meminfo_data(builder, meminfo)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
    strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

    populate_array(
        ary,
        data=builder.bitcast(data, datatype.as_pointer()),
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=meminfo,
    )

    return ary


if bodo.numba_compat._check_numba_change:
    lines = inspect.getsource(numba.np.arrayobj._empty_nd_impl)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "009ebfa261e39c4d8b9fdcc956205d9ee03ad87feea6560ef5fc2ddc8551c70d"
    ):  # pragma: no cover
        warnings.warn("numba.np.arrayobj._empty_nd_impl has changed")


def alloc_arr_tup(n, arr_tup, init_vals=()):  # pragma: no cover
    arrs = []
    for in_arr in arr_tup:
        arrs.append(np.empty(n, in_arr.dtype))
    return tuple(arrs)


@overload(alloc_arr_tup, no_unliteral=True)
def alloc_arr_tup_overload(n, data, init_vals=()):
    count = data.count

    allocs = ",".join(["empty_like_type(n, data[{}])".format(i) for i in range(count)])

    if init_vals != ():
        # TODO check for numeric value
        allocs = ",".join(
            [
                "np.full(n, init_vals[{}], data[{}].dtype)".format(i, i)
                for i in range(count)
            ]
        )

    func_text = "def f(n, data, init_vals=()):\n"
    func_text += "  return ({}{})\n".format(
        allocs, "," if count == 1 else ""
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"empty_like_type": empty_like_type, "np": np}, loc_vars)
    alloc_impl = loc_vars["f"]
    return alloc_impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def tuple_to_scalar(n):
    """Convert to scalar if 1-tuple, otherwise return original value"""
    if isinstance(n, types.BaseTuple) and len(n.types) == 1:
        return lambda n: n[0]  # pragma: no cover
    return lambda n: n  # pragma: no cover


def create_categorical_type(categories, data, is_ordered):
    """Create categorical array with either pd.array or np.array
        based on data array type to ensure correct lowering happens when using
        dictionary-encoded arrays or numpy arrays.

    Args:
        categories (Any): unique values for categorical data
        data (Any): data type of cateogrical data
        is_ordered (bool): wether or not this categorical is ordered

    Returns:
        new_cats_arr (pd.CategoricalDtype) : return type of pd.CategoricalDtype
    """

    # For anything with variable bitwidth in Bodo, we need to perfrom explicite
    # cast to insure that the bitwidth is preserved. Currently, this is only the
    # following two types:
    # Int
    # Float
    if data == bodo.string_array_type or bodo.utils.typing.is_dtype_nullable(data):
        new_cats_arr = pd.CategoricalDtype(
            pd.array(categories), is_ordered
        ).categories.array

        # This path isn't currently taken, as we can't partiton a pq file by a nullable
        # value. However, we still include it in case this function is ever re-used for
        # a different purpose.
        if isinstance(data.dtype, types.Number):  # pragma: no cover
            # NOTE: When we implement nullable floating array, we will need to support
            # get_pandas_scalar_type_instance in order for this to work
            new_cats_arr = new_cats_arr.astype(data.get_pandas_scalar_type_instance)

    else:
        new_cats_arr = pd.CategoricalDtype(categories, is_ordered).categories.values
        if isinstance(data.dtype, types.Number):
            new_cats_arr = new_cats_arr.astype(as_dtype(data.dtype))

    return new_cats_arr


def alloc_type(n, t, s=None, dict_ref_arr=None):  # pragma: no cover
    pass


@overload(alloc_type)
def overload_alloc_type(n, t, s=None, dict_ref_arr=None):
    """Allocate an array with type 't'. 'n' is length of the array. 's' is a tuple for
    arrays with variable size elements (e.g. strings), providing the number of elements
    needed for allocation.
    """
    typ = t.instance_type if isinstance(t, types.TypeRef) else t

    # Dictionary-encoded arrays can be allocated if a reference array is provided to
    # reuse its dictionary
    if typ == bodo.dict_str_arr_type and not is_overload_none(dict_ref_arr):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.dict_arr_ext.init_dict_arr(
            dict_ref_arr._data,
            bodo.libs.int_arr_ext.alloc_int_array(n, np.int32),
            dict_ref_arr._has_global_dictionary,
            dict_ref_arr._has_unique_local_dictionary,
            dict_ref_arr._dict_id,
        )

    # NOTE: creating regular string array for dictionary-encoded strings to get existing
    # code that doesn't support dict arr to work
    if is_str_arr_type(typ):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.str_arr_ext.pre_alloc_string_array(
            n, s[0]
        )  # pragma: no cover

    if typ == bodo.null_array_type:
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.null_arr_ext.init_null_array(
            n
        )  # pragma: no cover

    if typ == bodo.binary_array_type:
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.binary_arr_ext.pre_alloc_binary_array(
            n, s[0]
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.array_item_arr_ext.ArrayItemArrayType):
        if not is_overload_none(dict_ref_arr):
            return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
                n, s, bodo.libs.array_item_arr_ext.get_data(dict_ref_arr)
            )  # pragma: no cover

        dtype = typ.dtype
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
            n, s, dtype
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.struct_arr_ext.StructArrayType):
        dtypes = typ.data
        names = typ.names
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.struct_arr_ext.pre_alloc_struct_array(
            n, s, dtypes, names, dict_ref_arr
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.map_arr_ext.MapArrayType):
        struct_typ = bodo.libs.struct_arr_ext.StructArrayType(
            (typ.key_arr_type, typ.value_arr_type), ("key", "value")
        )
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.map_arr_ext.pre_alloc_map_array(
            n, s, struct_typ, dict_ref_arr
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.tuple_arr_ext.TupleArrayType):
        dtypes = typ.data
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.tuple_arr_ext.pre_alloc_tuple_array(
            n, s, dtypes
        )  # pragma: no cover

    if isinstance(typ, bodo.hiframes.pd_categorical_ext.CategoricalArrayType):
        if isinstance(t, types.TypeRef):
            if typ.dtype.categories is None:
                # TODO: Fix error message if there are other usages?
                raise BodoError(
                    "UDFs or Groupbys that return Categorical values must have categories known at compile time."
                )
            # create the new categorical dtype inside the function instead of passing as
            # constant. This avoids constant lowered Index inside the dtype, which can
            # be slow since it cannot have a dictionary.
            # see https://github.com/Bodo-inc/Bodo/pull/3563
            is_ordered = typ.dtype.ordered
            int_type = typ.dtype.int_type
            new_cats_arr = create_categorical_type(
                typ.dtype.categories, typ.dtype.data.data, is_ordered
            )
            new_cats_tup = MetaType(tuple(new_cats_arr))
            return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(
                n,
                bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                    bodo.utils.conversion.index_from_array(new_cats_arr),
                    is_ordered,
                    int_type,
                    new_cats_tup,
                ),
            )  # pragma: no cover
        else:
            return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(
                n, t.dtype
            )  # pragma: no cover

    if typ.dtype == bodo.hiframes.datetime_date_ext.datetime_date_type:
        return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(
            n
        )  # pragma: no cover

    if isinstance(typ.dtype, bodo.hiframes.time_ext.TimeType):
        precision = typ.dtype.precision

        return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.time_ext.alloc_time_array(
            n, precision
        )  # pragma: no cover

    if typ.dtype == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
        return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(
            n
        )  # pragma: no cover

    if isinstance(typ, DecimalArrayType):
        precision = typ.dtype.precision
        scale = typ.dtype.scale
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.decimal_arr_ext.alloc_decimal_array(
            n, precision, scale
        )  # pragma: no cover

    if isinstance(typ, bodo.DatetimeArrayType) or isinstance(
        type,
        (
            PandasTimestampType,
            PandasDatetimeTZDtype,
        ),
    ):
        tz_literal = typ.tz
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
            n, tz_literal
        )  # pragma: no cover

    if isinstance(typ, bodo.hiframes.timestamptz_ext.TimestampTZArrayType):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(
            n
        )  # pragma: no cover

    dtype = numba.np.numpy_support.as_dtype(typ.dtype)

    # nullable int array
    if isinstance(typ, IntegerArrayType):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.int_arr_ext.alloc_int_array(
            n, dtype
        )  # pragma: no cover

    # primitive array
    if isinstance(typ, bodo.libs.primitive_arr_ext.PrimitiveArrayType):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.primitive_arr_ext.alloc_primitive_array(
            n, dtype
        )  # pragma: no cover

    # nullable float array
    if isinstance(typ, FloatingArrayType):
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.float_arr_ext.alloc_float_array(
            n, dtype
        )  # pragma: no cover

    # nullable bool array
    if typ == boolean_array_type:
        return lambda n, t, s=None, dict_ref_arr=None: bodo.libs.bool_arr_ext.alloc_bool_array(
            n
        )  # pragma: no cover

    return lambda n, t, s=None, dict_ref_arr=None: np.empty(
        n, dtype
    )  # pragma: no cover


def astype(A, t):  # pragma: no cover
    return A.astype(t.dtype)


@overload(astype, no_unliteral=True)
def overload_astype(A, t):
    """Convert array 'A' to type 't'"""
    typ = t.instance_type if isinstance(t, types.TypeRef) else t
    dtype = typ.dtype

    if A == typ:
        return lambda A, t: A  # pragma: no cover

    # numpy or nullable int/float array can convert to numpy directly
    if isinstance(A, (types.Array, IntegerArrayType, FloatingArrayType)) and isinstance(
        typ, types.Array
    ):
        return lambda A, t: A.astype(dtype)  # pragma: no cover

    # convert to nullable int
    if isinstance(typ, IntegerArrayType):
        return lambda A, t: bodo.libs.int_arr_ext.init_integer_array(
            A.astype(dtype),
            np.full((len(A) + 7) >> 3, 255, np.uint8),
        )  # pragma: no cover

    # convert to nullable float
    if isinstance(typ, FloatingArrayType):  # pragma: no cover
        return lambda A, t: bodo.libs.float_arr_ext.init_float_array(
            A.astype(dtype),
            np.full((len(A) + 7) >> 3, 255, np.uint8),
        )  # pragma: no cover

    # Convert dictionary array to regular string array. This path is used
    # by join when 1 key is a regular string array and the other is a
    # dictionary array.
    if A == bodo.libs.dict_arr_ext.dict_str_arr_type and typ == bodo.string_array_type:
        return lambda A, t: bodo.utils.typing.decode_if_dict_array(
            A
        )  # pragma: no cover

    raise BodoError(f"cannot convert array type {A} to {typ}")


def full_type(n, val, t):  # pragma: no cover
    return np.full(n, val, t.dtype)


@overload(full_type, no_unliteral=True)
def overload_full_type(n, val, t):
    typ = t.instance_type if isinstance(t, types.TypeRef) else t

    # numpy array
    if isinstance(typ, types.Array):
        dtype = numba.np.numpy_support.as_dtype(typ.dtype)
        return lambda n, val, t: np.full(n, val, dtype)  # pragma: no cover

    # nullable int array
    if isinstance(typ, IntegerArrayType):
        dtype = numba.np.numpy_support.as_dtype(typ.dtype)
        return lambda n, val, t: bodo.libs.int_arr_ext.init_integer_array(
            np.full(n, val, dtype),
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )  # pragma: no cover

    # nullable float array
    if isinstance(typ, FloatingArrayType):
        dtype = numba.np.numpy_support.as_dtype(typ.dtype)
        return lambda n, val, t: bodo.libs.float_arr_ext.init_float_array(
            np.full(n, val, dtype),
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )  # pragma: no cover

    # nullable bool array
    if typ == boolean_array_type:

        def impl(n, val, t):  # pragma: no cover
            length = tuple_to_scalar(n)
            if val:
                return bodo.libs.bool_arr_ext.alloc_true_bool_array(length)
            else:
                return bodo.libs.bool_arr_ext.alloc_false_bool_array(length)

        return impl

    # string array
    if typ == string_array_type:

        def impl_str(n, val, t):  # pragma: no cover
            n_chars = n * bodo.libs.str_arr_ext.get_utf8_size(val)
            A = pre_alloc_string_array(n, n_chars)
            for i in range(n):
                A[i] = val
            return A

        return impl_str

    # generic implementation
    def impl(n, val, t):  # pragma: no cover
        A = alloc_type(n, typ, (-1,))
        for i in range(n):
            A[i] = val
        return A

    return impl


@intrinsic
def is_null_pointer(typingctx, ptr_typ=None):
    """check whether the pointer type is NULL or not"""

    def codegen(context, builder, signature, args):
        (ptr,) = args
        null = context.get_constant_null(ptr_typ)
        return builder.icmp_unsigned("==", ptr, null)

    return types.bool_(ptr_typ), codegen


@intrinsic
def is_null_value(typingctx, val_typ=None):
    """check whether a value is NULL or not"""

    def codegen(context, builder, signature, args):
        (val,) = args
        arr_struct_ptr = cgutils.alloca_once_value(builder, val)
        null_struct_ptr = cgutils.alloca_once_value(
            builder, context.get_constant_null(val_typ)
        )
        return is_ll_eq(builder, arr_struct_ptr, null_struct_ptr)

    return types.bool_(val_typ), codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def tuple_list_to_array(A, data, elem_type):
    """
    Function used to keep list -> array transformation
    replicated.
    """
    elem_type = (
        elem_type.instance_type if isinstance(elem_type, types.TypeRef) else elem_type
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        A, "tuple_list_to_array()"
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        elem_type, "tuple_list_to_array()"
    )
    func_text = "def impl(A, data, elem_type):\n"
    func_text += "  for i, d in enumerate(data):\n"
    if elem_type == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type:
        func_text += "    A[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(d)\n"
    else:
        func_text += "    A[i] = d\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def object_length(c, obj):
    """
    len(obj)
    """
    pyobj_lltyp = c.context.get_argument_type(types.pyobject)
    fnty = lir.FunctionType(lir.IntType(64), [pyobj_lltyp])
    fn = cgutils.get_or_insert_function(c.builder.module, fnty, name="PyObject_Length")
    return c.builder.call(fn, (obj,))


@intrinsic
def incref(typingctx, data=None):
    """manual incref of data to workaround bugs. Should be avoided if possible."""

    def codegen(context, builder, signature, args):
        (data_val,) = args

        context.nrt.incref(builder, signature.args[0], data_val)

    return types.void(data), codegen


def gen_getitem(out_var, in_var, ind, calltypes, nodes):
    loc = out_var.loc
    getitem = ir.Expr.static_getitem(in_var, ind, None, loc)
    calltypes[getitem] = None
    nodes.append(ir.Assign(getitem, out_var, loc))


def is_static_getsetitem(node):
    return is_expr(node, "static_getitem") or isinstance(node, ir.StaticSetItem)


def get_getsetitem_index_var(node, typemap, nodes):
    # node is either getitem/static_getitem expr or Setitem/StaticSetitem
    index_var = node.index_var if is_static_getsetitem(node) else node.index
    # sometimes index_var is None, so fix it
    # TODO: get rid of static_getitem in general
    if index_var is None:
        # TODO: test this path
        assert is_static_getsetitem(node)
        # literal type is preferred for uniform/easier getitem index match
        try:
            index_typ = types.literal(node.index)
        except:
            index_typ = numba.typeof(node.index)
        index_var = ir.Var(
            node.value.scope, ir_utils.mk_unique_var("dummy_index"), node.loc
        )
        typemap[index_var.name] = index_typ
        # TODO: can every const index be ir.Const?
        nodes.append(ir.Assign(ir.Const(node.index, node.loc), index_var, node.loc))
    return index_var


# don't copy value since it can fail
# for example, deepcopy in get_parfor_reductions can fail for ObjModeLiftedWith const
import copy

ir.Const.__deepcopy__ = lambda self, memo: ir.Const(self.value, copy.deepcopy(self.loc))


def is_call_assign(stmt) -> TypeGuard[ir.Assign]:
    return (
        isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "call"
    )


def is_call(expr) -> TypeGuard[ir.Expr]:
    return isinstance(expr, ir.Expr) and expr.op == "call"


def is_var_assign(inst) -> TypeGuard[ir.Assign]:
    return isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var)


def is_assign(inst) -> TypeGuard[ir.Assign]:
    return isinstance(inst, ir.Assign)


def is_expr(val, op) -> TypeGuard[ir.Expr]:
    return isinstance(val, ir.Expr) and val.op == op


def sanitize_varname(varname):
    """convert variable name to be identifier compatible (e.g. remove whitespace)"""
    if isinstance(varname, (tuple, list)):
        varname = "_".join(sanitize_varname(v) for v in varname)
    varname = str(varname)
    new_name = re.sub(r"\W+", "_", varname)
    if not new_name or not new_name[0].isalpha():
        new_name = "_" + new_name
    if not new_name.isidentifier() or keyword.iskeyword(new_name):
        new_name = mk_unique_var("new_name").replace(".", "_")
    return new_name


def dump_node_list(node_list):  # pragma: no cover
    for n in node_list:
        print("   ", n)


def debug_prints():
    return numba.core.config.DEBUG_ARRAY_OPT == 1


# TODO: Move to Numba
@overload(reversed)
def list_reverse(A):
    """
    reversed(list)
    """
    if isinstance(A, types.List):

        def impl_reversed(A):
            A_len = len(A)
            for i in range(A_len):
                yield A[A_len - 1 - i]

        return impl_reversed


@numba.njit
def count_nonnan(a):  # pragma: no cover
    """
    Count number of non-NaN elements in an array
    """
    return np.count_nonzero(~np.isnan(a))


@numba.njit
def nanvar_ddof1(a):  # pragma: no cover
    """
    Simple implementation for np.nanvar(arr, ddof=1)
    """
    num_el = count_nonnan(a)
    if num_el <= 1:
        return np.nan
    return np.nanvar(a) * (num_el / (num_el - 1))


@numba.njit
def nanstd_ddof1(a):  # pragma: no cover
    """
    Simple implementation for np.nanstd(arr, ddof=1)
    """
    return np.sqrt(nanvar_ddof1(a))


def has_supported_h5py() -> bool:
    """returns True if supported versions of h5py and hdf5 are installed"""
    try:
        import h5py  # noqa

        from bodo.io import _hdf5  # noqa

        # TODO: make sure h5py/hdf5 supports parallel
    except ImportError:
        _has_h5py = False
    else:
        # NOTE: _hdf5 import fails if proper hdf5 version is not installed, but we
        # should check h5py as well since there may be an extra pip installation
        # see [BE-1382].
        # We only support 1.14
        _has_h5py = h5py.version.hdf5_version_tuple[1] == 14
    return _has_h5py


# Dummy function, that should be handled in untyped pass
# Used to handle aggregations that require more arguments than just the column
# and aggregation name.
def ExtendedNamedAgg():
    pass


def check_h5py():
    """raise error if h5py/hdf5 is not installed"""
    if not has_supported_h5py():
        raise BodoError("install 'h5py' package to enable hdf5 support")


def has_scipy():
    """returns True if scipy is installed"""
    try:
        import scipy  # noqa
    except ImportError:
        _has_scipy = False
    else:
        _has_scipy = True
    return _has_scipy


@intrinsic
def check_and_propagate_cpp_exception(typingctx):
    """
    Check if an error occured in C++ using the C Python API
    (PyErr_Occured). If it did, raise it in Python with
    the corresponding error message.
    """

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        err_flag = pyapi.err_occurred()
        error_occured = cgutils.is_not_null(builder, err_flag)

        with builder.if_then(error_occured):
            builder.ret(numba.core.callconv.RETCODE_EXC)

    return types.void(), codegen


def inlined_check_and_propagate_cpp_exception(context, builder):
    """
    Inlined version of the check_and_propagate_cpp_exception intrinsic
    defined above. Can be used in lower_builtin functions, etc.
    """
    pyapi = context.get_python_api(builder)
    err_flag = pyapi.err_occurred()
    error_occured = cgutils.is_not_null(builder, err_flag)

    with builder.if_then(error_occured):
        builder.ret(numba.core.callconv.RETCODE_EXC)


@numba.njit
def check_java_installation(fname):
    with bodo.no_warning_objmode():
        check_java_installation_(fname)


def check_java_installation_(fname):
    if not fname.startswith("hdfs://"):
        return
    import shutil

    if not shutil.which("java"):
        message = (
            "Java not found. Make sure openjdk is installed for hdfs."
            " openjdk can be installed by calling"
            " 'conda install 'openjdk>=9.0,<12' -c conda-forge'."
        )
        raise BodoError(message)


dt_err = """
        If you are trying to set NULL values for timedelta64 in regular Python, \n
        consider using np.timedelta64('nat') instead of None
        """


@lower_constant(types.List)
def lower_constant_list(context, builder, typ, pyval):
    """Support constant lowering of lists"""

    # Throw warning for large lists
    if len(pyval) > CONST_LIST_SLOW_WARN_THRESHOLD:  # pragma: no cover
        warnings.warn(
            BodoWarning(
                "Using large global lists can result in long compilation times. Please pass large lists as arguments to JIT functions or use arrays."
            )
        )

    value_consts = []
    for a in pyval:
        if bodo.typeof(a) != typ.dtype:
            raise BodoError(
                f"Values in list must have the same data type for type stability. Expected: {typ.dtype}, Actual: {bodo.typeof(a)}"
            )
        value_consts.append(context.get_constant_generic(builder, typ.dtype, a))

    size = context.get_constant_generic(builder, types.int64, len(pyval))
    dirty = context.get_constant_generic(builder, types.bool_, False)

    # create a constant payload with the same data model as ListPayload
    # "size", "allocated", "dirty", "data"
    # NOTE: payload and data are packed together in a single buffer
    parent_null = context.get_constant_null(types.pyobject)
    payload = lir.Constant.literal_struct([size, size, dirty] + value_consts)
    payload = cgutils.global_constant(builder, ".const.payload", payload).bitcast(
        cgutils.voidptr_t
    )

    # create a constant meminfo with the same data model as Numba
    minus_one = context.get_constant(types.int64, -1)
    null_ptr = context.get_constant_null(types.voidptr)
    meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, payload, minus_one]
    )
    meminfo = cgutils.global_constant(builder, ".const.meminfo", meminfo).bitcast(
        cgutils.voidptr_t
    )

    # create the list
    return lir.Constant.literal_struct([meminfo, parent_null])


@lower_constant(types.Set)
def lower_constant_set(context, builder, typ, pyval):
    """Support constant lowering of sets"""

    # reusing list constant lowering instead of creating a proper constant set due to
    # the complexities of set internals. This leads to potential memory leaks.
    # TODO [BE-2140]: create a proper constant set

    for a in pyval:
        if bodo.typeof(a) != typ.dtype:
            raise BodoError(
                f"Values in set must have the same data type for type stability. Expected: {typ.dtype}, Actual: {bodo.typeof(a)}"
            )

    list_typ = types.List(typ.dtype)
    list_const = context.get_constant_generic(builder, list_typ, list(pyval))

    set_val = context.compile_internal(
        builder,
        lambda l: set(l),
        # creating a new set type since 'typ' has the reflected flag
        types.Set(typ.dtype)(list_typ),
        [list_const],
    )  # pragma: no cover

    return set_val


def lower_const_dict_fast_path(context, builder, typ, pyval):
    """fast path for lowering a constant dictionary. It lowers key and value arrays
    and creates a dictionary from them.
    This approach allows faster compilation time for very large dictionaries.
    """
    from bodo.utils.typing import can_replace

    key_arr = pd.Series(pyval.keys()).values
    vals_arr = pd.Series(pyval.values()).values
    key_arr_type = bodo.typeof(key_arr)
    vals_arr_type = bodo.typeof(vals_arr)
    require(
        key_arr_type.dtype == typ.key_type
        or can_replace(typ.key_type, key_arr_type.dtype)
    )
    require(
        vals_arr_type.dtype == typ.value_type
        or can_replace(typ.value_type, vals_arr_type.dtype)
    )
    key_arr_const = context.get_constant_generic(builder, key_arr_type, key_arr)
    vals_arr_const = context.get_constant_generic(builder, vals_arr_type, vals_arr)

    def create_dict(keys, vals):  # pragma: no cover
        """create a dictionary from key and value arrays"""
        out = {}
        for k, v in zip(keys, vals):
            out[k] = v
        return out

    dict_val = context.compile_internal(
        builder,
        # TODO: replace when dict(zip()) works [BE-2113]
        # lambda keys, vals: dict(zip(keys, vals)),
        create_dict,
        typ(key_arr_type, vals_arr_type),
        [key_arr_const, vals_arr_const],
    )
    return dict_val


@lower_constant(types.DictType)
def lower_constant_dict(context, builder, typ, pyval):
    """Support constant lowering of dictionries.
    Has a fast path for dictionaries that their keys/values fit in arrays, and a slow
    path for the general case.
    Currently has memory leaks since Numba's dictionaries have malloc() calls in C
    [BE-2114]
    """
    # fast path for cases that fit in arrays
    try:
        return lower_const_dict_fast_path(context, builder, typ, pyval)
    except:
        pass

    # throw warning for large dicts in slow path since compilation can take long
    if len(pyval) > CONST_DICT_SLOW_WARN_THRESHOLD:  # pragma: no cover
        warnings.warn(
            BodoWarning(
                "Using large global dictionaries can result in long compilation times. Please pass large dictionaries as arguments to JIT functions."
            )
        )

    # slow path: create a dict and fill values individually
    key_type = typ.key_type
    val_type = typ.value_type

    def make_dict():  # pragma: no cover
        return numba.typed.Dict.empty(key_type, val_type)

    dict_val = context.compile_internal(
        builder,
        make_dict,
        typ(),
        [],
    )

    def set_dict_val(d, k, v):  # pragma: no cover
        d[k] = v

    for k, v in pyval.items():
        k_const = context.get_constant_generic(builder, key_type, k)
        v_const = context.get_constant_generic(builder, val_type, v)
        context.compile_internal(
            builder,
            set_dict_val,
            types.none(typ, key_type, val_type),
            [dict_val, k_const, v_const],
        )

    return dict_val


def synchronize_error(exception_str, error_message):
    """Syncrhonize error state across ranks

    Args:
        exception (Exception): exception, e.x. RuntimeError, ValueError
        error (string): error message, empty string means no error

    Raises:
        Exception: user supplied exception with custom error message
    """
    # TODO: Support pattern matching for more exceptions
    if exception_str == "ValueError":
        exception = ValueError
    else:
        exception = RuntimeError

    comm = MPI.COMM_WORLD
    # synchronize error state
    if comm.allreduce(error_message != "", op=MPI.LOR):
        for error_message in comm.allgather(error_message):
            if error_message:
                raise exception(error_message)


@numba.njit
def synchronize_error_njit(exception_str, error_message):
    """An njit wrapper around syncrhonize_error

    Args:
        exception_str (string): string representation of exception, e.x. 'RuntimeError', 'ValueError'
        error_message (string): error message, empty string means no error
    """
    with bodo.no_warning_objmode():
        synchronize_error(exception_str, error_message)


# Helper function that extracts the constant value from a tuple of constants or a constant
def get_const_or_build_tuple_of_consts(var):
    if is_expr(var, "build_tuple"):
        return tuple([item.value for item in var.items])
    elif isinstance(var, (ir.Global, ir.FreeVar, ir.Const)):
        return var.value
    else:
        raise BodoError(
            "Value of orderby should be a constant tuple or tuple of constants"
        )


# Helper function that adds a value to a dictionary of lists,
# creating the list if it doesn't exist, and appending if it does
def dict_add_multimap(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


@numba.njit(no_cpython_wrapper=True)
def set_wrapper(a):
    """wrapper around set() constructor to reduce compilation time.
    This makes sure set (e.g. of int array) is compiled once versus lower_builtin in
    Numba compiling it every time.
    """
    return set(a)


def run_rank0(func: Callable, bcast_result: bool = True, result_default=None):
    """
    Utility function decorator to run a function on just rank 0
    but re-raise any Exceptions safely on all ranks.
    NOTE: 'func' must be a simple python function that doesn't require
    any synchronization.
    e.g. Using a bodo.jit function might be unsafe in this situation.
    Similarly, a function that uses any MPI collective
    operation would be unsafe and could result in a hang.

    Args:
        func: Function to run.
        bcast_result (bool, optional): Whether the function should be
            broadcasted to all ranks. Defaults to True.
        result_default (optional): Default for result. This is only
            useful in the bcase_result=False case. Defaults to None.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        comm = MPI.COMM_WORLD
        result = result_default
        err = None
        # Run on rank 0 and catch any exceptions.
        if comm.Get_rank() == 0:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                err = e
        # Synchronize and re-raise any exception on all ranks.
        err = comm.bcast(err)
        if isinstance(err, Exception):
            raise err
        # Broadcast the result to all ranks.
        if bcast_result:
            result = comm.bcast(result)
        return result

    return inner


@dataclass
class AWSCredentials:
    access_key: str
    secret_key: str
    session_token: str | None = None
    region: str | None = None
