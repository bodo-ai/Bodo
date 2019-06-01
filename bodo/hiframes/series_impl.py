"""
Implementation of Series attributes and methods using overload.
"""
import operator
import numpy as np
import pandas as pd
import numba
from numba import types
from numba.extending import overload, overload_attribute, overload_method
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_index_ext import is_pd_index_type
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false, is_overload_zero)


@overload_attribute(SeriesType, 'index')
def overload_series_index(s):
    # None should be range type
    if s.index == types.none:
        return lambda s: bodo.hiframes.pd_index_ext.init_range_index(
            0, len(bodo.hiframes.api.get_series_data(s)), 1, None)

    return lambda s: bodo.hiframes.api.get_series_index(s)


@overload_attribute(SeriesType, 'values')
def overload_series_values(s):
    return lambda s: bodo.hiframes.api.get_series_data(s)


@overload_attribute(SeriesType, 'dtype')
def overload_series_dtype(s):
    # TODO: check other dtypes like tuple, etc.
    if s.dtype == bodo.string_type:
        raise ValueError("Series.dtype not supported for string Series yet")

    return lambda s: bodo.hiframes.api.get_series_data(s).dtype


@overload_attribute(SeriesType, 'shape')
def overload_series_shape(s):
    return lambda s: (len(bodo.hiframes.api.get_series_data(s)),)


@overload_attribute(SeriesType, 'ndim')
def overload_series_ndim(s):
    return lambda s: 1


@overload_attribute(SeriesType, 'size')
def overload_series_size(s):
    return lambda s: len(bodo.hiframes.api.get_series_data(s))


@overload_attribute(SeriesType, 'T')
def overload_series_T(s):
    return lambda s: s


@overload_attribute(SeriesType, 'hasnans')
def overload_series_hasnans(s):
    return lambda s: s.isna().sum() != 0


@overload_attribute(SeriesType, 'empty')
def overload_series_empty(s):
    return lambda s: len(bodo.hiframes.api.get_series_data(s)) == 0


@overload_attribute(SeriesType, 'dtypes')
def overload_series_dtypes(s):
    return lambda s: s.dtype


@overload_attribute(SeriesType, 'name')
def overload_series_name(s):
    return lambda s: bodo.hiframes.api.get_series_name(s)


@overload_method(SeriesType, 'put')
def overload_series_put(S, indices, values):
    # TODO: non-numeric types like strings
    def impl(S, indices, values):
        bodo.hiframes.api.get_series_data(S)[indices] = values

    return impl


@overload(len)
def overload_series_len(S):
    if isinstance(S, SeriesType):
        return lambda S: len(bodo.hiframes.api.get_series_data(S))


# TODO: fix 'str' typing
# @overload_method(SeriesType, 'astype')
# def overload_series_astype(S, dtype):
#     # TODO: other data types like datetime, records/tuples
#     def impl(S, dtype):
#         arr = bodo.hiframes.api.get_series_data(S)
#         index = bodo.hiframes.api.get_series_index(S)
#         name = bodo.hiframes.api.get_series_name(S)
#         out_arr = arr.astype(dtype)

#         return bodo.hiframes.api.init_series(out_arr, index, name)

#     return impl
    # def _series_astype_str_impl(arr, index, name):
    #     n = len(arr)
    #     num_chars = 0
    #     # get total chars in new array
    #     for i in numba.parfor.internal_prange(n):
    #         s = arr[i]
    #         num_chars += len(str(s))  # TODO: check NA

    #     A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars)
    #     for i in numba.parfor.internal_prange(n):
    #         s = arr[i]
    #         A[i] = str(s)  # TODO: check NA

    #     return bodo.hiframes.api.init_series(A, index, name)
    # def impl(S):
    #     numba.parfor.init_prange()
    #     arr = bodo.hiframes.api.get_series_data(S)
    #     index = bodo.hiframes.api.get_series_index(S)
    #     name = bodo.hiframes.api.get_series_name(S)
    #     n = len(arr)
    #     out_arr = np.empty(n, np.bool_)
    #     for i in numba.parfor.internal_prange(n):
    #         out_arr[i] = bodo.hiframes.api.isna(arr, i)

    #     return bodo.hiframes.api.init_series(out_arr, index, name)

    # return impl


@overload_method(SeriesType, 'copy')
def overload_series_copy(S, deep=True):
    # TODO: test all Series data types
    # XXX specialized kinds until branch pruning is tested and working well
    if is_overload_true(deep):
        def impl1(S, deep=True):
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            return bodo.hiframes.api.init_series(arr.copy(), index, name)
        return impl1

    if is_overload_false(deep):
        def impl2(S, deep=True):
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            return bodo.hiframes.api.init_series(arr, index, name)
        return impl2

    def impl(S, deep=True):
        arr = bodo.hiframes.api.get_series_data(S)
        if deep:
            arr = arr.copy()
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(arr, index, name)

    return impl


@overload_method(SeriesType, 'to_list')
@overload_method(SeriesType, 'tolist')
def overload_series_to_list(S):
    # TODO: test all Series data types
    def impl(S):
        l = list()
        for i in range(len(S)):
            # using iat directly on S to box Timestamp/... properly
            l.append(S.iat[i])
        return l
    return impl


@overload_method(SeriesType, 'get_values')
def overload_series_get_values(S):
    def impl(S):
        return S.values
    return impl


@overload_method(SeriesType, 'isna')
@overload_method(SeriesType, 'isnull')
def overload_series_isna(S):
    # TODO: series that have different underlying data type than dtype
    # like records/tuples
    def impl(S):
        numba.parfor.init_prange()
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfor.internal_prange(n):
            out_arr[i] = bodo.hiframes.api.isna(arr, i)

        return bodo.hiframes.api.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, 'sum')
def overload_series_sum(S):
    # TODO: series that have different underlying data type than dtype
    # like records/tuples
    def impl(S):
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
        # TODO: fix output type
        s = 0
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s += A[i]

        return s

    return impl


@overload_method(SeriesType, 'prod')
def overload_series_prod(S):
    def impl(S):
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
        # TODO: fix output type
        s = 1
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s *= A[i]
        return s

    return impl


@overload_method(SeriesType, 'mean')
def overload_series_mean(S):
    def impl(S):  # pragma: no cover
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_series_data(S)
        count = 0
        s = 0
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s += A[i]
                count += 1

        res = bodo.hiframes.series_kernels._mean_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, 'var')
def overload_series_var(S):
    def impl(S):  # pragma: no cover
        m = S.mean()
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_series_data(S)
        s = 0
        count = 0
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s += (A[i] - m)**2
                count += 1

        res = bodo.hiframes.series_kernels._var_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, 'std')
def overload_series_std(S):
    def impl(S):  # pragma: no cover
        return S.var()**0.5

    return impl


@overload_method(SeriesType, 'cumsum')
def overload_series_cumsum(S):
    # TODO: support skipna
    def impl(S):  # pragma: no cover
        A = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(A.cumsum(), index, name)

    return impl


@overload_method(SeriesType, 'cumprod')
def overload_series_cumprod(S):
    # TODO: support skipna
    def impl(S):  # pragma: no cover
        A = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(A.cumprod(), index, name)

    return impl


@overload_method(SeriesType, 'rename')
def overload_series_rename(S, index=None):
    if not (index == bodo.string_type
            or isinstance(index, types.StringLiteral)):
        raise ValueError("Series.rename() 'index' can only be a string")

    # TODO: support index rename, kws
    def impl(S, index=None):  # pragma: no cover
        A = bodo.hiframes.api.get_series_data(S)
        s_index = bodo.hiframes.api.get_series_index(S)
        return bodo.hiframes.api.init_series(A, s_index, index)

    return impl


@overload_method(SeriesType, 'abs')
def overload_series_abs(S):
    # TODO: timedelta
    def impl(S):  # pragma: no cover
        A = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(np.abs(A), index, name)

    return impl


@overload_method(SeriesType, 'count')
def overload_series_count(S):
    # TODO: check 'level' argument
    def impl(S):  # pragma: no cover
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_series_data(S)
        count = 0
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                count += 1

        res = count
        return res

    return impl


@overload_method(SeriesType, 'corr')
def overload_series_corr(S, other, method='pearson', min_periods=None):
    if not is_overload_none(min_periods):
        raise ValueError("Series.corr(): 'min_periods' is not supported yet")

    if method != 'pearson':
        # TODO: check string constant value in Series pass?
        raise ValueError("Series.corr(): 'method' is not supported yet")

    def impl(S, other, method='pearson', min_periods=None):  # pragma: no cover
        n = S.count()
        # TODO: check lens
        ma = S.sum()
        mb = other.sum()
        # TODO: check aligned nans, (S.notna() != other.notna()).any()
        a = n * ((S*other).sum()) - ma * mb
        b1 = n * (S**2).sum() - ma**2
        b2 = n * (other**2).sum() - mb**2
        # TODO: np.clip
        # TODO: np.true_divide?
        return a / np.sqrt(b1*b2)

    return impl


@overload_method(SeriesType, 'cov')
def overload_series_cov(S, other, min_periods=None):
    if not is_overload_none(min_periods):
        raise ValueError("Series.cov(): 'min_periods' is not supported yet")

    # TODO: use online algorithm, e.g. StatFunctions.scala
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def impl(S, other, min_periods=None):  # pragma: no cover
        # TODO: check lens
        ma = S.mean()
        mb = other.mean()
        # TODO: check aligned nans, (S.notna() != other.notna()).any()
        return ((S-ma)*(other-mb)).sum()/(S.count()-1.0)

    return impl


@overload_method(SeriesType, 'min')
def overload_series_min(S, axis=None, skipna=None, level=None,
                                                            numeric_only=None):
    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise ValueError('Series.min(): axis argument not supported')

    # TODO: min/max of string dtype, etc.
    if S.dtype == types.NPDatetime('ns'):
        def impl_dt64(S, axis=None, skipna=None, level=None, numeric_only=None):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.api.get_series_data(S)
            s = numba.targets.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfor.internal_prange(len(in_arr)):
                if not bodo.hiframes.api.isna(in_arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                    s = min(s, val)
                    count += 1
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    def impl(S, axis=None, skipna=None, level=None, numeric_only=None):  # pragma: no cover
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.api.get_series_data(S)
        count = 0
        s = bodo.hiframes.series_kernels._get_type_max_value(in_arr.dtype)
        for i in numba.parfor.internal_prange(len(in_arr)):
            val = in_arr[i]
            if not bodo.hiframes.api.isna(in_arr, i):
                s = min(s, val)
                count += 1
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, 'max')
def overload_series_max(S, axis=None, skipna=None, level=None,
                                                            numeric_only=None):
    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise ValueError('Series.min(): axis argument not supported')

    if S.dtype == types.NPDatetime('ns'):
        def impl_dt64(S, axis=None, skipna=None, level=None, numeric_only=None):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.api.get_series_data(S)
            s = numba.targets.builtins.get_type_min_value(np.int64)
            count = 0
            for i in numba.parfor.internal_prange(len(in_arr)):
                if not bodo.hiframes.api.isna(in_arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                    s = max(s, val)
                    count += 1
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    def impl(S, axis=None, skipna=None, level=None, numeric_only=None):  # pragma: no cover
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.api.get_series_data(S)
        count = 0
        s = numba.targets.builtins.get_type_min_value(in_arr.dtype)
        for i in numba.parfor.internal_prange(len(in_arr)):
            val = in_arr[i]
            if not bodo.hiframes.api.isna(in_arr, i):
                s = max(s, val)
                count += 1
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, 'idxmin')
def overload_series_idxmin(S, axis=0, skipna=True):
    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise ValueError('Series.idxmin(): axis argument not supported')

    # TODO: other types like strings
    if S.dtype == types.none:
        return (lambda S, axis=0, skipna=True:
            bodo.hiframes.api.get_series_data(S).argmin())
    else:
        def impl(S, axis=0, skipna=True):
            i = bodo.hiframes.api.get_series_data(S).argmin()
            return bodo.hiframes.api.get_series_index(S)[i]
        return impl


@overload_method(SeriesType, 'idxmax')
def overload_series_idxmax(S, axis=0, skipna=True):
    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise ValueError('Series.idxmax(): axis argument not supported')

    # TODO: other types like strings
    if S.dtype == types.none:
        return (lambda S, axis=0, skipna=True:
            bodo.hiframes.api.get_series_data(S).argmax())
    else:
        def impl(S, axis=0, skipna=True):
            i = bodo.hiframes.api.get_series_data(S).argmax()
            return bodo.hiframes.api.get_series_index(S)[i]
        return impl


############################ binary operators #############################


def create_explicit_binary_op_overload(op):
    def overload_series_explicit_binary_op(
                                S, other, level=None, fill_value=None, axis=0):
        if not is_overload_none(level):
            raise ValueError("level argument not supported")

        if not is_overload_zero(axis):
            raise ValueError("axis argument not supported")

        # TODO: string array, datetimeindex/timedeltaindex
        if not isinstance(S.dtype, types.Number):
            raise TypeError("only numeric values supported")

        typing_context = numba.targets.registry.cpu_target.typing_context
        # scalar case
        if isinstance(other, types.Number):
            args = (types.Array(S.dtype, 1, 'C'), other)
            ret_dtype = typing_context.resolve_function_type(
                op, args, ()).return_type.dtype
            def impl_scalar(S, other, level=None, fill_value=None, axis=0):
                numba.parfor.init_prange()
                arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                # other could be tuple, list, array, Index, or Series
                n = len(arr)
                out_arr = np.empty(n, ret_dtype)
                for i in numba.parfor.internal_prange(n):
                    left_nan = bodo.hiframes.api.isna(arr, i)
                    if left_nan:
                        if fill_value is None:
                            bodo.ir.join.setitem_arr_nan(out_arr, i)
                        else:
                            out_arr[i] = op(fill_value, other)
                    else:
                        out_arr[i] = op(arr[i], other)

                return bodo.hiframes.api.init_series(out_arr, index, name)

            return impl_scalar

        args = (types.Array(S.dtype, 1, 'C'), types.Array(other.dtype, 1, 'C'))
        ret_dtype = typing_context.resolve_function_type(
            op, args, ()).return_type.dtype
        def impl(S, other, level=None, fill_value=None, axis=0):
            numba.parfor.init_prange()
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            # other could be tuple, list, array, Index, or Series
            other_arr = bodo.utils.conversion.coerce_to_array(other)
            n = len(arr)
            out_arr = np.empty(n, ret_dtype)
            for i in numba.parfor.internal_prange(n):
                left_nan = bodo.hiframes.api.isna(arr, i)
                right_nan = bodo.hiframes.api.isna(other_arr, i)
                if left_nan and right_nan:
                    bodo.ir.join.setitem_arr_nan(out_arr, i)
                elif left_nan:
                    if fill_value is None:
                        bodo.ir.join.setitem_arr_nan(out_arr, i)
                    else:
                        out_arr[i] = op(fill_value, other_arr[i])
                elif right_nan:
                    if fill_value is None:
                        bodo.ir.join.setitem_arr_nan(out_arr, i)
                    else:
                        out_arr[i] = op(arr[i], fill_value)
                else:
                    out_arr[i] = op(arr[i], other_arr[i])

            return bodo.hiframes.api.init_series(out_arr, index, name)

        return impl

    return overload_series_explicit_binary_op


# TODO: radd, ...
explicit_binop_funcs = {
    operator.add: 'add',
    operator.sub: 'sub',
    operator.mul: 'mul',
    operator.truediv: 'div',
    operator.truediv: 'truediv',
    operator.floordiv: 'floordiv',
    operator.mod: 'mod',
    operator.pow: 'pow',
    operator.lt: 'lt',
    operator.gt: 'gt',
    operator.le: 'le',
    operator.ge: 'ge',
    operator.ne: 'ne',
    operator.eq: 'eq',
}


def _install_explicit_binary_ops():
    for op, name in explicit_binop_funcs.items():
        overload_impl = create_explicit_binary_op_overload(op)
        overload_method(SeriesType, name)(overload_impl)


_install_explicit_binary_ops()


####################### binary operators ###############################


def create_binary_op_overload(op):
    def overload_series_binary_op(S, other):
        if isinstance(S, SeriesType):
            def impl(S, other):
                arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(
                    other)
                out_arr = op(arr, other_arr)
                return bodo.hiframes.api.init_series(out_arr, index, name)

            return impl

        # right arg is Series
        if isinstance(other, SeriesType):
            def impl2(S, other):
                arr = bodo.hiframes.api.get_series_data(other)
                index = bodo.hiframes.api.get_series_index(other)
                name = bodo.hiframes.api.get_series_name(other)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(
                    S)
                out_arr = op(other_arr, arr)
                return bodo.hiframes.api.init_series(out_arr, index, name)

            return impl2

    return overload_series_binary_op


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        overload_impl = create_binary_op_overload(op)
        overload(op)(overload_impl)


_install_binary_ops()


####################### binary inplace operators #############################


def create_inplace_binary_op_overload(op):
    def overload_series_inplace_binary_op(S, other):
        if isinstance(S, SeriesType) or isinstance(other, SeriesType):
            op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
            # TODO: use op directly when Numba's #4131 is resolved
            func_text = "def impl(S, other):\n"
            func_text += "  arr = bodo.utils.conversion.get_array_if_series_or_index(S)\n"
            func_text += "  other_arr = bodo.utils.conversion.get_array_if_series_or_index(other)\n"
            func_text += "  arr {} other_arr\n".format(op_str)
            func_text += "  return S\n"
            loc_vars = {}
            exec(func_text, {'bodo': bodo}, loc_vars)
            impl = loc_vars['impl']
            return impl

    return overload_series_inplace_binary_op


def _install_inplace_binary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
        overload_impl = create_inplace_binary_op_overload(op)
        overload(op)(overload_impl)


_install_inplace_binary_ops()


########################## unary operators ###############################


def create_unary_op_overload(op):
    def overload_series_unary_op(S):
        if isinstance(S, SeriesType):
            def impl(S):
                arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                out_arr = op(arr)
                return bodo.hiframes.api.init_series(out_arr, index, name)

            return impl

    return overload_series_unary_op


def _install_unary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in bodo.hiframes.pd_series_ext.series_unary_ops:
        overload_impl = create_unary_op_overload(op)
        overload(op)(overload_impl)


_install_unary_ops()


####################### numpy ufuncs #########################


# XXX: overloading ufuncs doesn't work (Numba's #4133)
# TODO: use this version when issue is resolved
# def create_ufunc_overload(ufunc):
#     if ufunc.nin == 1:
#         def overload_series_ufunc_nin_1(S):
#             if isinstance(S, SeriesType):
#                 def impl(S):
#                     arr = bodo.hiframes.api.get_series_data(S)
#                     index = bodo.hiframes.api.get_series_index(S)
#                     name = bodo.hiframes.api.get_series_name(S)
#                     out_arr = ufunc(arr)
#                     return bodo.hiframes.api.init_series(out_arr, index, name)
#                 return impl
#         return overload_series_ufunc_nin_1
#     elif ufunc.nin == 2:
#         def overload_series_ufunc_nin_2(S1, S2):
#             if isinstance(S1, SeriesType):
#                 def impl(S1, S2):
#                     arr = bodo.hiframes.api.get_series_data(S1)
#                     index = bodo.hiframes.api.get_series_index(S1)
#                     name = bodo.hiframes.api.get_series_name(S1)
#                     other_arr = bodo.utils.conversion.get_array_if_series_or_index(S2)
#                     out_arr = ufunc(arr, other_arr)
#                     return bodo.hiframes.api.init_series(out_arr, index, name)
#                 return impl
#             elif isinstance(S2, SeriesType):
#                 def impl(S1, S2):
#                     arr = bodo.utils.conversion.get_array_if_series_or_index(S1)
#                     other_arr = bodo.hiframes.api.get_series_data(S2)
#                     index = bodo.hiframes.api.get_series_index(S2)
#                     name = bodo.hiframes.api.get_series_name(S2)
#                     out_arr = ufunc(arr, other_arr)
#                     return bodo.hiframes.api.init_series(out_arr, index, name)
#                 return impl
#         return overload_series_ufunc_nin_2
#     else:
#         raise RuntimeError(
#             "Don't know how to register ufuncs from ufunc_db with arity > 2")


# def _install_np_ufuncs():
#     import numba.targets.ufunc_db
#     for ufunc in numba.targets.ufunc_db.get_ufuncs():
#         overload_impl = create_ufunc_overload(ufunc)
#         overload(ufunc)(overload_impl)


# _install_np_ufuncs()
