# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Collection of utility functions. Needs to be refactored in separate files.
"""
import hashlib
import inspect
import keyword
import re
import warnings
from enum import Enum

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

import bodo
from bodo.libs.binary_arr_ext import bytes_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import (
    num_total_chars,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.libs.str_ext import string_type
from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.typing import (
    NOT_CONSTANT,
    BodoError,
    BodoWarning,
    MetaType,
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
    Datetime = 14
    Timedelta = 15
    # NOTE: currently, only used for handling decimal array's data array for scatterv
    # since it handles the data array inside decimal array separately
    Int128 = 16
    # NOTE: 17 is used by LIST_STRING in bodo_common.h
    LIST = 18
    STRUCT = 19
    BINARY = 20


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
}


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


def numba_to_c_type(t):
    if isinstance(t, bodo.libs.decimal_arr_ext.Decimal128Type):
        return CTypeEnum.Decimal.value

    if t == bodo.hiframes.datetime_date_ext.datetime_date_type:
        return CTypeEnum.Date.value

    # TODO: Timedelta arrays need to be supported
    #    if t == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
    #        return CTypeEnum.Timedelta.value

    return _numba_to_c_type_map[t]


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
        or (func_name == "alloc_int_array" and mod_name == "bodo.libs.int_arr_ext")
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
    )


def find_build_tuple(func_ir, var):
    """Check if a variable is constructed via build_tuple
    and return the sequence or raise GuardException otherwise.
    """
    # variable or variable name
    require(isinstance(var, (ir.Var, str)))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    require(var_def.op == "build_tuple")
    return var_def.items


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


def is_array_typ(var_typ, include_index_series=True):
    """return True if var_typ is an array type.
    include_index_series=True also includes Index and Series types (as "array-like").
    """

    # NOTE: make sure all Bodo arrays are here
    return (
        is_np_array_typ(var_typ)
        or var_typ
        in (
            string_array_type,
            bodo.binary_array_type,
            bodo.dict_str_arr_type,
            bodo.hiframes.split_impl.string_array_split_view_type,
            bodo.hiframes.datetime_date_ext.datetime_date_array_type,
            bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_array_type,
            boolean_array,
            bodo.libs.str_ext.random_access_string_array,
        )
        or isinstance(
            var_typ,
            (
                IntegerArrayType,
                bodo.libs.decimal_arr_ext.DecimalArrayType,
                bodo.hiframes.pd_categorical_ext.CategoricalArrayType,
                bodo.libs.array_item_arr_ext.ArrayItemArrayType,
                bodo.libs.struct_arr_ext.StructArrayType,
                bodo.libs.interval_arr_ext.IntervalArrayType,
                bodo.libs.tuple_arr_ext.TupleArrayType,
                bodo.libs.map_arr_ext.MapArrayType,
                bodo.libs.csr_matrix_ext.CSRMatrixType,
                bodo.DatetimeArrayType,
            ),
        )
        or (
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

    if arr == boolean_array:

        def empty_like_type_bool_arr(n, arr):  # pragma: no cover
            return bodo.libs.bool_arr_ext.alloc_bool_array(n)

        return empty_like_type_bool_arr

    if arr == bodo.hiframes.datetime_date_ext.datetime_date_array_type:

        def empty_like_type_datetime_date_arr(n, arr):  # pragma: no cover
            return bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)

        return empty_like_type_datetime_date_arr

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
        != "b6a998927680caa35917a553c79704e9d813d8f1873d83a5f8513837c159fa29"
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


def alloc_type(n, t, s=None):  # pragma: no cover
    return np.empty(n, t.dtype)


@overload(alloc_type)
def overload_alloc_type(n, t, s=None):
    """Allocate an array with type 't'. 'n' is length of the array. 's' is a tuple for
    arrays with variable size elements (e.g. strings), providing the number of elements
    needed for allocation.
    """
    typ = t.instance_type if isinstance(t, types.TypeRef) else t
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(typ, "bodo.alloc_type()")

    # NOTE: creating regular string array for dictionary-encoded strings to get existing
    # code that doesn't support dict arr to work
    if is_str_arr_type(typ):
        return lambda n, t, s=None: bodo.libs.str_arr_ext.pre_alloc_string_array(
            n, s[0]
        )  # pragma: no cover

    if typ == bodo.binary_array_type:
        return lambda n, t, s=None: bodo.libs.binary_arr_ext.pre_alloc_binary_array(
            n, s[0]
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.array_item_arr_ext.ArrayItemArrayType):
        dtype = typ.dtype
        return lambda n, t, s=None: bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
            n, s, dtype
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.struct_arr_ext.StructArrayType):
        dtypes = typ.data
        names = typ.names
        return lambda n, t, s=None: bodo.libs.struct_arr_ext.pre_alloc_struct_array(
            n, s, dtypes, names
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.map_arr_ext.MapArrayType):
        struct_typ = bodo.libs.struct_arr_ext.StructArrayType(
            (typ.key_arr_type, typ.value_arr_type), ("key", "value")
        )
        return lambda n, t, s=None: bodo.libs.map_arr_ext.pre_alloc_map_array(
            n, s, struct_typ
        )  # pragma: no cover

    if isinstance(typ, bodo.libs.tuple_arr_ext.TupleArrayType):
        dtypes = typ.data
        return lambda n, t, s=None: bodo.libs.tuple_arr_ext.pre_alloc_tuple_array(
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
            new_cats_arr = pd.CategoricalDtype(
                typ.dtype.categories, is_ordered
            ).categories.values
            new_cats_tup = MetaType(tuple(new_cats_arr))
            return lambda n, t, s=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(
                n,
                bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                    bodo.utils.conversion.index_from_array(new_cats_arr),
                    is_ordered,
                    int_type,
                    new_cats_tup,
                ),
            )  # pragma: no cover
        else:
            return lambda n, t, s=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(
                n, t.dtype
            )  # pragma: no cover

    if typ.dtype == bodo.hiframes.datetime_date_ext.datetime_date_type:
        return lambda n, t, s=None: bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(
            n
        )  # pragma: no cover

    if typ.dtype == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
        return lambda n, t, s=None: bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(
            n
        )  # pragma: no cover

    if isinstance(typ, DecimalArrayType):
        precision = typ.dtype.precision
        scale = typ.dtype.scale
        return lambda n, t, s=None: bodo.libs.decimal_arr_ext.alloc_decimal_array(
            n, precision, scale
        )  # pragma: no cover

    dtype = numba.np.numpy_support.as_dtype(typ.dtype)

    # nullable int array
    if isinstance(typ, IntegerArrayType):
        return lambda n, t, s=None: bodo.libs.int_arr_ext.alloc_int_array(
            n, dtype
        )  # pragma: no cover

    # nullable bool array
    if typ == boolean_array:
        return lambda n, t, s=None: bodo.libs.bool_arr_ext.alloc_bool_array(
            n
        )  # pragma: no cover

    return lambda n, t, s=None: np.empty(n, dtype)  # pragma: no cover


def astype(A, t):  # pragma: no cover
    return A.astype(t.dtype)


@overload(astype, no_unliteral=True)
def overload_astype(A, t):
    """Convert array 'A' to type 't'"""
    typ = t.instance_type if isinstance(t, types.TypeRef) else t
    dtype = typ.dtype

    if A == typ:
        return lambda A, t: A  # pragma: no cover

    # numpy or nullable int array can convert to numpy directly
    if isinstance(A, (types.Array, IntegerArrayType)) and isinstance(typ, types.Array):
        return lambda A, t: A.astype(dtype)  # pragma: no cover

    # convert to nullable int
    if isinstance(typ, IntegerArrayType):
        return lambda A, t: bodo.libs.int_arr_ext.init_integer_array(
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

    # nullable bool array
    if typ == boolean_array:
        return lambda n, val, t: bodo.libs.bool_arr_ext.init_bool_array(
            np.full(n, val, np.bool_),
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )  # pragma: no cover

    # string array
    if typ == string_array_type:

        def impl_str(n, val, t):  # pragma: no cover
            n_chars = n * len(val)
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
    if elem_type == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type:
        func_text += "    A[i] = bodo.utils.conversion.unbox_if_timestamp(d)\n"
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


def is_call_assign(stmt):
    return (
        isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "call"
    )


def is_call(expr):
    return isinstance(expr, ir.Expr) and expr.op == "call"


def is_var_assign(inst):
    return isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var)


def is_assign(inst):
    return isinstance(inst, ir.Assign)


def is_expr(val, op):
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


def has_supported_h5py():
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
        # We support both 1.10 and 1.12
        _has_h5py = h5py.version.hdf5_version_tuple[1] in (10, 12)
    return _has_h5py


def check_h5py():
    """raise error if h5py/hdf5 is not installed"""
    if not has_supported_h5py():
        raise BodoError("install 'h5py' package to enable hdf5 support")


def has_pyarrow():
    """returns True if pyarrow is installed"""
    try:
        import pyarrow  # noqa
    except ImportError:
        _has_pyarrow = False
    else:
        _has_pyarrow = True
    return _has_pyarrow


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
    with numba.objmode():
        check_java_installation_(fname)


def check_java_installation_(fname):
    if not fname.startswith("hdfs://"):
        return
    import shutil

    if not shutil.which("java"):
        message = (
            "Java not found. Make sure openjdk is installed for hdfs."
            " openjdk can be installed by calling"
            " 'conda install openjdk=8 -c conda-forge'."
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
