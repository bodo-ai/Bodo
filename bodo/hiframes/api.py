# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Old code for some auxiliary functionality. Needs to be refactored.
"""
from collections import namedtuple
import pandas as pd
import numpy as np

import numba
from numba import ir
from numba.ir_utils import mk_unique_var
from numba import types, cgutils
import numba.array_analysis
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload, intrinsic
from numba.targets.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from numba.targets.arrayobj import _getitem_array1d
from numba.extending import register_model, models

import bodo
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type, is_str_arr_typ
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.list_str_arr_ext import list_string_array_type

from bodo.utils.utils import build_set
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    SeriesPayloadType,
    if_arr_to_series_type,
    if_series_to_array_type,
)
from bodo.hiframes.pd_index_ext import DatetimeIndexType, TimedeltaIndexType
from bodo.ir.sort import (
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
)
from bodo.ir.join import write_send_buff
from bodo.hiframes.split_impl import string_array_split_view_type

from numba.targets.arrayobj import make_array
from bodo.utils.utils import unliteral_all
import llvmlite.llvmpy.core as lc


def concat(arr_list):
    return pd.concat(arr_list)


@infer_global(concat)
class ConcatType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr_list = args[0]
        if isinstance(arr_list, types.UniTuple) and is_str_arr_typ(arr_list.dtype):
            ret_typ = string_array_type
        elif isinstance(arr_list, types.UniTuple) and (
            isinstance(arr_list.dtype, IntegerArrayType)
            or arr_list.dtype == boolean_array
        ):
            ret_typ = arr_list.dtype
            # TODO: support concat with different dtypes or with regular numpy
        else:
            # use typer of np.concatenate
            ret_typ = numba.typing.npydecl.NdConcatenate(self.context).generic()(
                arr_list
            )

        return signature(ret_typ, arr_list)


@lower_builtin(concat, types.Any)  # TODO: replace Any with types
def lower_concat(context, builder, sig, args):
    func = concat_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


# @overload(concat)
def concat_overload(arr_list):
    # all string input case
    # TODO: handle numerics to string casting case
    if isinstance(arr_list, types.UniTuple) and is_str_arr_typ(arr_list.dtype):

        def string_concat_impl(in_arrs):
            # preallocate the output
            num_strs = 0
            num_chars = 0
            for A in in_arrs:
                arr = A
                num_strs += len(arr)
                num_chars += bodo.libs.str_arr_ext.num_total_chars(arr)
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strs, num_chars)
            # copy data to output
            curr_str_ind = 0
            curr_chars_ind = 0
            for A in in_arrs:
                arr = A
                bodo.libs.str_arr_ext.set_string_array_range(
                    out_arr, arr, curr_str_ind, curr_chars_ind
                )
                curr_str_ind += len(arr)
                curr_chars_ind += bodo.libs.str_arr_ext.num_total_chars(arr)
            return out_arr

        return string_concat_impl

    if isinstance(arr_list, types.UniTuple) and isinstance(
        arr_list.dtype, IntegerArrayType
    ):
        return lambda arr_list: bodo.libs.int_arr_ext.init_integer_array(
            np.concatenate(bodo.libs.int_arr_ext.get_int_arr_data_tup(arr_list)),
            bodo.libs.int_arr_ext.concat_bitmap_tup(arr_list),
        )

    if isinstance(arr_list, types.UniTuple) and arr_list.dtype == boolean_array:
        # reusing int arr concat functions
        # TODO: test
        return lambda arr_list: bodo.libs.bool_arr_ext.init_bool_array(
            np.concatenate(bodo.libs.int_arr_ext.get_int_arr_data_tup(arr_list)),
            bodo.libs.int_arr_ext.concat_bitmap_tup(arr_list),
        )

    for typ in arr_list:
        if not isinstance(typ, types.Array):
            raise ValueError("concat supports only numerical and string arrays")
    # numerical input
    return lambda a: np.concatenate(a)


def nunique(A):  # pragma: no cover
    return len(set(A))


def nunique_parallel(A):  # pragma: no cover
    return len(set(A))


@infer_global(nunique)
@infer_global(nunique_parallel)
class NuniqueType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        # if arr == string_series_type:
        #     arr = string_array_type
        return signature(types.intp, arr)


@lower_builtin(nunique, types.Any)  # TODO: replace Any with types
def lower_nunique(context, builder, sig, args):
    func = nunique_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


# @overload(nunique)
def nunique_overload(arr_typ):
    if arr_typ == boolean_array:
        return lambda A: len(A.unique())
    # TODO: extend to other types like datetime?
    def nunique_seq(A):
        return len(build_set(A))

    return nunique_seq


@lower_builtin(nunique_parallel, types.Any)  # TODO: replace Any with types
def lower_nunique_parallel(context, builder, sig, args):
    func = nunique_overload_parallel(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


# @overload(nunique_parallel)
def nunique_overload_parallel(arr_typ):
    sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value

    def nunique_par(A):
        uniq_A = bodo.hiframes.api.unique_parallel(A)
        loc_nuniq = len(uniq_A)
        return bodo.libs.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))

    return nunique_par


def unique(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)


def unique_parallel(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)


@overload(unique)
def unique_overload(A):
    # TODO: extend to other types like datetime?
    def unique_seq(A):
        return bodo.utils.utils.unique(A)

    return unique_seq


@overload(unique_parallel)
def unique_overload_parallel(A):
    def unique_par(A):
        uniq_A = bodo.utils.utils.unique(A)
        key_arrs = (uniq_A,)
        n = len(uniq_A)
        node_ids = np.empty(n, np.int32)

        n_pes = bodo.libs.distributed_api.get_size()
        pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, (), n_pes, False)

        # calc send/recv counts
        for i in range(n):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            node_ids[i] = node_id
            update_shuffle_meta(pre_shuffle_meta, node_id, i, key_arrs, (), False)

        shuffle_meta = finalize_shuffle_meta(
            key_arrs, (), pre_shuffle_meta, n_pes, False
        )

        # write send buffers
        for i in range(n):
            node_id = node_ids[i]
            write_send_buff(shuffle_meta, node_id, i, key_arrs, ())
            # update last since it is reused in data
            shuffle_meta.tmp_offset[node_id] += 1

        # shuffle
        out_arr, = alltoallv_tup(key_arrs, shuffle_meta, ())

        return bodo.utils.utils.unique(out_arr)

    return unique_par


def alloc_shift(A):
    return np.empty_like(A)


@overload(alloc_shift)
def alloc_shift_overload(A):
    if isinstance(A.dtype, types.Integer):
        return lambda A: np.empty(len(A), np.float64)
    return lambda A: np.empty(len(A), A.dtype)


def shift_dtype(d):
    return d


@overload(shift_dtype)
def shift_dtype_overload(a):
    if isinstance(a.dtype, types.Integer):
        return lambda a: np.float64
    else:
        return lambda a: a


def isna(arr, i):
    return False


@overload(isna)
def isna_overload(arr, i):
    # String array
    if arr == string_array_type:
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(arr, i)

    # masked Integer array, boolean array
    if isinstance(arr, IntegerArrayType) or arr == boolean_array:
        return lambda arr, i: not bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            arr._null_bitmap, i
        )

    if arr == list_string_array_type:
        # reuse string array function
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(arr, i)

    if arr == string_array_split_view_type:
        return lambda arr, i: False

    # TODO: extend to other types
    assert isinstance(arr, types.Array)
    dtype = arr.dtype
    if isinstance(dtype, types.Float):
        return lambda arr, i: np.isnan(arr[i])

    # NaT for dt64
    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = dtype("NaT")
        # TODO: replace with np.isnat
        return lambda arr, i: arr[i] == nat

    # XXX integers don't have nans, extend to boolean
    return lambda arr, i: False


def argsort(A):
    return np.argsort(A)


@overload(argsort)
def overload_argsort(A):
    def impl(A):
        n = len(A)
        l_key_arrs = bodo.libs.str_arr_ext.to_string_list((A.copy(),))
        data = (np.arange(n),)
        bodo.libs.timsort.sort(l_key_arrs, 0, n, data)
        return data[0]

    return impl


def sort(arr, index_arr, ascending, inplace):
    return np.sort(arr), index_arr


@overload(sort)
def overload_sort(arr, index_arr, ascending, inplace):
    def impl(arr, index_arr, ascending, inplace):
        n = len(arr)
        key_arrs = (arr,)
        data = (index_arr,)
        if not inplace:
            key_arrs = (arr.copy(),)
            data = (index_arr.copy(),)

        l_key_arrs = bodo.libs.str_arr_ext.to_string_list(key_arrs)
        l_data = bodo.libs.str_arr_ext.to_string_list(data, True)
        bodo.libs.timsort.sort(l_key_arrs, 0, n, l_data)
        if not ascending:
            bodo.libs.timsort.reverseRange(l_key_arrs, 0, n, l_data)
        bodo.libs.str_arr_ext.cp_str_list_to_array(key_arrs, l_key_arrs)
        bodo.libs.str_arr_ext.cp_str_list_to_array(data, l_data, True)
        return key_arrs[0], data[0]

    return impl


# the same as fix_df_array but can be parallel
@numba.generated_jit(nopython=True)
def parallel_fix_df_array(c):  # pragma: no cover
    return lambda c: bodo.utils.conversion.coerce_to_array(c)


def fix_rolling_array(c):  # pragma: no cover
    return c


def df_isin(A, B):  # pragma: no cover
    return A


def df_isin_vals(A, B):  # pragma: no cover
    return A


@infer_global(df_isin)
@infer_global(df_isin_vals)
class DfIsinCol(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.Array(types.bool_, 1, "C"), *unliteral_all(args))


# dummy lowering functions
@lower_builtin(df_isin, types.Any, types.Any)
@lower_builtin(df_isin_vals, types.Any, types.Any)
def lower_dummy_isin(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


def to_numeric(A, dtype):
    return A


@infer_global(to_numeric)
class ToNumeric(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        dtype = args[1].dtype
        return signature(SeriesType(dtype), *unliteral_all(args))


def series_filter_bool(arr, bool_arr):
    return arr[bool_arr]


@infer_global(series_filter_bool)
class SeriesFilterBoolInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        ret = args[0]
        if isinstance(ret.dtype, types.Integer):
            ret = SeriesType(types.float64)
        return signature(ret, *args)


def set_df_col(df, cname, arr, inplace):
    df[cname] = arr


@infer_global(set_df_col)
class SetDfColInfer(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_dataframe_ext import DataFrameType

        assert not kws
        assert len(args) == 4
        assert isinstance(args[1], types.Literal)
        target = args[0]
        ind = args[1].literal_value
        val = args[2]
        ret = target

        if isinstance(target, DataFrameType):
            if isinstance(val, SeriesType):
                val = val.data
            if ind in target.columns:
                # set existing column, with possibly a new array type
                new_cols = target.columns
                col_id = target.columns.index(ind)
                new_typs = list(target.data)
                new_typs[col_id] = val
                new_typs = tuple(new_typs)
            else:
                # set a new column
                new_cols = target.columns + (ind,)
                new_typs = target.data + (val,)
            ret = DataFrameType(new_typs, target.index, new_cols, target.has_parent)

        return signature(ret, *args)


def get_series_data_tup(series_tup):
    return tuple(get_series_data(s) for s in series_tup)


@overload(get_series_data_tup)
def overload_get_series_data_tup(series_tup):
    n_series = len(series_tup.types)
    func_text = "def f(series_tup):\n"
    res = ",".join(
        "bodo.hiframes.api.get_series_data(series_tup[{}])".format(i)
        for i in range(n_series)
    )
    func_text += "  return ({}{})\n".format(res, "," if n_series == 1 else "")
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["f"]
    return impl


# convert tuple of Series to tuple of arrays statically (for append)
def series_tup_to_arr_tup(arrs):  # pragma: no cover
    return arrs


@infer_global(series_tup_to_arr_tup)
class SeriesTupleToArrTupleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        ret_typ = if_series_to_array_type(arr)
        return signature(ret_typ, arr)


def get_series_payload(context, builder, series_type, value):
    meminfo = cgutils.create_struct_proxy(series_type)(context, builder, value).meminfo
    payload_type = SeriesPayloadType(series_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_series_data(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, series_typ.data, series_payload.data)

    ret_typ = series_typ.data
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def update_series_data(typingctx, series_typ, arr_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        series_payload.data = args[1]
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[1], args[1])
        return

    sig = types.none(series_typ, arr_typ)
    return sig, codegen


@intrinsic
def update_series_index(typingctx, series_typ, arr_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        series_payload.index = args[1]
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[1], args[1])
        return

    sig = types.none(series_typ, arr_typ)
    return sig, codegen


@intrinsic
def _get_series_index(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, series_typ.index, series_payload.index
        )

    ret_typ = series_typ.index
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def _get_series_name(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        # TODO: is borrowing None reference ok here?
        return impl_ret_borrowed(
            context, builder, signature.return_type, series_payload.name
        )

    sig = signature(series_typ.name_typ, series_typ)
    return sig, codegen


# this function should be used for getting S._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_data(S):
    return lambda S: _get_series_data(S)


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_index(S):
    return lambda S: _get_series_index(S)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_name(S):
    return lambda S: _get_series_name(S)


@numba.generated_jit(nopython=True)
def get_index_data(S):
    return lambda S: S._data


@numba.generated_jit(nopython=True)
def get_index_name(S):
    return lambda S: S._name


# array analysis extension
def get_series_data_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


from numba.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_hiframes_api_get_series_data = get_series_data_equiv


def init_series_equiv(self, scope, equiv_set, args, kws):
    assert len(args) >= 1 and not kws
    # TODO: add shape for index
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_api_init_series = init_series_equiv


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_series(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    if len(args) > 1:  # has index
        numba.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


def alias_ext_init_integer_array(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 2
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.ir_utils.alias_func_extensions[
    ("init_series", "bodo.hiframes.api")
] = alias_ext_init_series
numba.ir_utils.alias_func_extensions[
    ("get_series_data", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_series_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_datetime_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_timedelta_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_numeric_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_string_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_index_data", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
# TODO: init_dataframe
numba.ir_utils.alias_func_extensions[
    ("init_integer_array", "bodo.libs.int_arr_ext")
] = alias_ext_init_integer_array
numba.ir_utils.alias_func_extensions[
    ("get_int_arr_data", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_int_arr_bitmap", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_bool_array", "bodo.libs.bool_arr_ext")
] = alias_ext_init_integer_array
numba.ir_utils.alias_func_extensions[
    ("get_bool_arr_data", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_bool_arr_bitmap", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func


def convert_tup_to_rec(val):
    return val


@infer_global(convert_tup_to_rec)
class ConvertTupRecType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.BaseTuple):
            np_dtype = np.dtype(",".join(str(t) for t in in_dtype.types), align=True)
            out_dtype = numba.numpy_support.from_dtype(np_dtype)

        return signature(out_dtype, in_dtype)


@lower_builtin(convert_tup_to_rec, types.Any)
def lower_convert_impl(context, builder, sig, args):
    val, = args
    in_typ = sig.args[0]
    rec_typ = sig.return_type

    if not isinstance(in_typ, types.BaseTuple):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    res = cgutils.alloca_once(builder, context.get_data_type(rec_typ))

    func_text = "def _set_rec(r, val):\n"
    for i in range(len(rec_typ.members)):
        func_text += "  r.f{} = val[{}]\n".format(i, i)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    set_rec = loc_vars["_set_rec"]

    context.compile_internal(builder, set_rec, types.void(rec_typ, in_typ), [res, val])
    return impl_ret_new_ref(context, builder, sig.return_type, res)


def convert_rec_to_tup(val):
    return val


@infer_global(convert_rec_to_tup)
class ConvertRecTupType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.Record):
            out_dtype = types.Tuple([m[1] for m in in_dtype.members])

        return signature(out_dtype, in_dtype)


@lower_builtin(convert_rec_to_tup, types.Any)
def lower_convert_rec_tup_impl(context, builder, sig, args):
    val, = args
    rec_typ = sig.args[0]
    tup_typ = sig.return_type

    if not isinstance(rec_typ, types.Record):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    n_fields = len(rec_typ.members)

    func_text = "def _rec_to_tup(r):\n"
    func_text += "  return ({},)\n".format(
        ", ".join("r.f{}".format(i) for i in range(n_fields))
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    _rec_to_tup = loc_vars["_rec_to_tup"]

    res = context.compile_internal(builder, _rec_to_tup, tup_typ(rec_typ), [val])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@infer_global(fix_rolling_array)
class FixDfRollingArrayType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        column = args[0]
        dtype = column.dtype
        ret_typ = column
        if dtype == types.boolean or isinstance(dtype, types.Integer):
            ret_typ = types.Array(types.float64, 1, "C")
        # TODO: add other types
        return signature(ret_typ, column)


@lower_builtin(fix_rolling_array, types.Any)  # TODO: replace Any with types
def lower_fix_rolling_array(context, builder, sig, args):
    func = fix_rolling_array_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


# @overload(fix_rolling_array)
def fix_rolling_array_overload(column):
    assert isinstance(column, types.Array)
    dtype = column.dtype
    # convert bool and integer to float64
    if dtype == types.boolean or isinstance(dtype, types.Integer):

        def fix_rolling_array_impl(column):  # pragma: no cover
            return column.astype(np.float64)

    else:

        def fix_rolling_array_impl(column):  # pragma: no cover
            return column

    return fix_rolling_array_impl


def construct_series(context, builder, series_type, data_val, index_val, name_val):
    # create payload struct and store values
    payload_type = SeriesPayloadType(series_type)
    series_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    series_payload.data = data_val
    series_payload.index = index_val
    series_payload.name = name_val

    # create meminfo and store payload
    payload_ll_type = context.get_data_type(payload_type)
    payload_size = context.get_abi_sizeof(payload_ll_type)
    meminfo = context.nrt.meminfo_alloc(
        builder, context.get_constant(types.uintp, payload_size)
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr, payload_ll_type.as_pointer())
    builder.store(series_payload._getvalue(), meminfo_data_ptr)

    # create Series struct
    series = cgutils.create_struct_proxy(series_type)(context, builder)
    series.meminfo = meminfo
    # Set parent to NULL
    series.parent = cgutils.get_null_value(series.parent.type)
    return series._getvalue()


@intrinsic
def init_series(typingctx, data, index=None, name=None):
    """Create a Series with provided data, index and name values.
    Used as a single constructor for Series and assigning its data, so that
    optimization passes can look for init_series() to see if underlying
    data has changed, and get the array variables from init_series() args if
    not changed.
    """

    index = types.none if index is None else index
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, index_val, name_val = args
        series_type = signature.return_type

        series_val = construct_series(
            context, builder, series_type, data_val, index_val, name_val
        )

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], index_val)
            context.nrt.incref(builder, signature.args[2], name_val)

        return series_val

    dtype = data.dtype
    # XXX pd.DataFrame() calls init_series for even Series since it's untyped
    data = if_series_to_array_type(data)
    ret_typ = SeriesType(dtype, data, index, name)
    sig = signature(ret_typ, data, index, name)
    return sig, codegen


@intrinsic
def init_datetime_index(typingctx, data, name=None):
    """Create a DatetimeIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create dt_index struct and store values
        dt_index = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dt_index.data = data_val
        dt_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return dt_index._getvalue()

    ret_typ = DatetimeIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@intrinsic
def init_timedelta_index(typingctx, data, name=None):
    """Create a TimedeltaIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create timedelta_index struct and store values
        timedelta_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        timedelta_index.data = data_val
        timedelta_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@overload(np.array)
def np_array_array_overload(A):
    if isinstance(A, types.Array):
        return lambda A: A

    if isinstance(A, types.containers.Set):
        # TODO: naive implementation, data from set can probably
        # be copied to array more efficienty
        dtype = A.dtype

        def f(A):
            n = len(A)
            arr = np.empty(n, dtype)
            i = 0
            for a in A:
                arr[i] = a
                i += 1
            return arr

        return f


# a dummy join function that will be replace in dataframe_pass
def join_dummy(left_df, right_df, left_on, right_on, how):
    return left_df


@infer_global(join_dummy)
class JoinTyper(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_dataframe_ext import DataFrameType
        from bodo.utils.typing import is_overload_str

        assert not kws
        left_df, right_df, left_on, right_on, how = args

        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_on.consts) & set(right_on.consts)
        comm_data = set(left_df.columns) & set(right_df.columns)
        add_suffix = comm_data - comm_keys

        columns = [(c + "_x" if c in add_suffix else c) for c in left_df.columns]
        # common keys are added only once so avoid adding them
        columns += [
            (c + "_y" if c in add_suffix else c)
            for c in right_df.columns
            if c not in comm_keys
        ]
        data = list(left_df.data)
        data += [
            right_df.data[right_df.columns.index(c)]
            for c in right_df.columns
            if c not in comm_keys
        ]

        # TODO: unify left/right indices if necessary (e.g. RangeIndex/Int64)
        index_typ = types.none
        left_index = "$_bodo_index_" in left_on.consts
        right_index = "$_bodo_index_" in right_on.consts
        if left_index and right_index and not is_overload_str(how, "asof"):
            index_typ = left_df.index
            if isinstance(index_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        elif right_index and is_overload_str(how, "left"):
            index_typ = left_df.index
        elif left_index and is_overload_str(how, "right"):
            index_typ = right_df.index

        out_df = DataFrameType(tuple(data), index_typ, tuple(columns))
        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


def drop_inplace(df):
    res = None
    return df, res


@overload(drop_inplace)
def drop_inplace_overload(
    df,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
):

    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    assert isinstance(df, DataFrameType)
    # TODO: support recovery when object is not df
    def _impl(
        df,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        new_df = bodo.hiframes.pd_dataframe_ext.drop_dummy(
            df, labels, axis, columns, inplace
        )
        return new_df, None

    return _impl


def sort_values_inplace(df):
    res = None
    return df, res


@overload(sort_values_inplace)
def sort_values_inplace_overload(
    df, by, axis=0, ascending=True, inplace=False, kind="quicksort", na_position="last"
):

    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    assert isinstance(df, DataFrameType)
    # TODO: support recovery when object is not df
    def _impl(
        df,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
    ):

        new_df = bodo.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, by, ascending, inplace
        )
        return new_df, None

    return _impl


class DataFrameTupleIterator(types.SimpleIteratorType):
    """
    Type class for itertuples of dataframes.
    """

    def __init__(self, col_names, arr_typs):
        self.array_types = arr_typs
        self.col_names = col_names
        name_args = [
            "{}={}".format(col_names[i], arr_typs[i]) for i in range(len(col_names))
        ]
        name = "itertuples({})".format(",".join(name_args))
        py_ntup = namedtuple("Pandas", col_names)
        yield_type = types.NamedTuple([_get_series_dtype(a) for a in arr_typs], py_ntup)
        super(DataFrameTupleIterator, self).__init__(name, yield_type)


def _get_series_dtype(arr_typ):
    # values of datetimeindex are extracted as Timestamp
    if arr_typ == types.Array(types.NPDatetime("ns"), 1, "C"):
        return pandas_timestamp_type
    return arr_typ.dtype


def get_itertuples():  # pragma: no cover
    pass


@infer_global(get_itertuples)
class TypeIterTuples(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) % 2 == 0, "name and column pairs expected"
        col_names = [a.literal_value for a in args[: len(args) // 2]]
        arr_types = [if_series_to_array_type(a) for a in args[len(args) // 2 :]]
        # XXX index handling, assuming implicit index
        assert "Index" not in col_names[0]
        col_names = ["Index"] + col_names
        arr_types = [types.Array(types.int64, 1, "C")] + arr_types
        iter_typ = DataFrameTupleIterator(col_names, arr_types)
        return signature(iter_typ, *args)


@register_model(DataFrameTupleIterator)
class DataFrameTupleIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        # XXX array_types[0] is implicit index
        members = [("index", types.EphemeralPointer(types.uintp))] + [
            ("array{}".format(i), arr) for i, arr in enumerate(fe_type.array_types[1:])
        ]
        super(DataFrameTupleIteratorModel, self).__init__(dmm, fe_type, members)

    def from_return(self, builder, value):
        # dummy to avoid lowering error for itertuples_overload
        # TODO: remove when overload_method can avoid lowering or avoid cpython
        # wrapper
        return value


@lower_builtin(get_itertuples, types.VarArg(types.Any))
def get_itertuples_impl(context, builder, sig, args):
    arrays = args[len(args) // 2 :]
    array_types = sig.args[len(sig.args) // 2 :]

    iterobj = context.make_helper(builder, sig.return_type)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr

    for i, arr in enumerate(arrays):
        setattr(iterobj, "array{}".format(i), arr)

    # Incref arrays
    if context.enable_nrt:
        for arr, arr_typ in zip(arrays, array_types):
            context.nrt.incref(builder, arr_typ, arr)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@lower_builtin("getiter", DataFrameTupleIterator)
def getiter_itertuples(context, builder, sig, args):
    # simply return the iterator
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# similar to iternext of ArrayIterator
@lower_builtin("iternext", DataFrameTupleIterator)
@iternext_impl(RefType.UNTRACKED)
def iternext_itertuples(context, builder, sig, args, result):
    # TODO: refcount issues?
    iterty, = sig.args
    it, = args

    # TODO: support string arrays
    iterobj = context.make_helper(builder, iterty, value=it)
    # first array type is implicit int index
    # use len() to support string arrays
    len_sig = signature(types.intp, iterty.array_types[1])
    nitems = context.compile_internal(
        builder, lambda a: len(a), len_sig, [iterobj.array0]
    )
    # ary = make_array(iterty.array_types[1])(context, builder, value=iterobj.array0)
    # nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        values = [index]  # XXX implicit int index
        for i, arr_typ in enumerate(iterty.array_types[1:]):
            arr_ptr = getattr(iterobj, "array{}".format(i))

            if arr_typ == types.Array(types.NPDatetime("ns"), 1, "C"):
                getitem_sig = signature(pandas_timestamp_type, arr_typ, types.intp)
                val = context.compile_internal(
                    builder,
                    lambda a, i: bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                        np.int64(a[i])
                    ),
                    getitem_sig,
                    [arr_ptr, index],
                )
            else:
                getitem_sig = signature(arr_typ.dtype, arr_typ, types.intp)
                val = context.compile_internal(
                    builder, lambda a, i: a[i], getitem_sig, [arr_ptr, index]
                )
            # arr = make_array(arr_typ)(context, builder, value=arr_ptr)
            # val = _getitem_array1d(context, builder, arr_typ, arr, index,
            #                      wraparound=False)
            values.append(val)

        value = context.make_tuple(builder, iterty.yield_type, values)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


# TODO: move this to array analysis
# the namedtuples created by get_itertuples-iternext-pair_first don't have
# shapes created in array analysis
# def _analyze_op_static_getitem(self, scope, equiv_set, expr):
#     var = expr.value
#     typ = self.typemap[var.name]
#     if not isinstance(typ, types.BaseTuple):
#         return self._index_to_shape(scope, equiv_set, expr.value, expr.index_var)
#     try:
#         shape = equiv_set._get_shape(var)
#         require(isinstance(expr.index, int) and expr.index < len(shape))
#         return shape[expr.index], []
#     except:
#         pass

#     return None

# numba.array_analysis.ArrayAnalysis._analyze_op_static_getitem = _analyze_op_static_getitem

# FIXME: fix array analysis for tuples in general
def _analyze_op_pair_first(self, scope, equiv_set, expr):
    # make dummy lhs since we don't have access to lhs
    typ = self.typemap[expr.value.name].first_type
    if not isinstance(typ, types.NamedTuple):
        return None
    lhs = ir.Var(scope, mk_unique_var("tuple_var"), expr.loc)
    self.typemap[lhs.name] = typ
    rhs = ir.Expr.pair_first(expr.value, expr.loc)
    lhs_assign = ir.Assign(rhs, lhs, expr.loc)
    # (shape, post) = self._gen_shape_call(equiv_set, lhs, typ.count, )
    var = lhs
    out = []
    size_vars = []
    ndims = typ.count
    for i in range(ndims):
        # get size: Asize0 = A_sh_attr[0]
        size_var = ir.Var(
            var.scope, mk_unique_var("{}_size{}".format(var.name, i)), var.loc
        )
        getitem = ir.Expr.static_getitem(lhs, i, None, var.loc)
        self.calltypes[getitem] = None
        out.append(ir.Assign(getitem, size_var, var.loc))
        self._define(equiv_set, size_var, types.intp, getitem)
        size_vars.append(size_var)
    shape = tuple(size_vars)
    return shape, [lhs_assign] + out


numba.array_analysis.ArrayAnalysis._analyze_op_pair_first = _analyze_op_pair_first
