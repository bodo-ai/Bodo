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
    is_overload_false, is_overload_zero, is_overload_str)


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
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
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
    if isinstance(S.dtype, types.Integer):
        retty = types.intp
    else:
        retty = S.dtype
    zero = retty(0)
    def impl(S):
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
        s = zero
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s += A[i]

        return s

    return impl


@overload_method(SeriesType, 'prod')
def overload_series_prod(S):
    init = S.dtype(1)
    def impl(S):
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
        s = init
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s *= A[i]
        return s

    return impl


@overload_method(SeriesType, 'mean')
def overload_series_mean(S):
    # see core/nanops.py/nanmean() for output types
    # TODO: more accurate port of dtypes from pandas
    sum_dtype = types.float64
    count_dtype = types.float64
    if S.dtype == types.float32:
        sum_dtype = types.float32
        count_dtype = types.float32

    sum_init = sum_dtype(0)
    count_init = count_dtype(0)
    count_1 = count_dtype(1)

    def impl(S):  # pragma: no cover
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
        count = count_init
        s = sum_init
        for i in numba.parfor.internal_prange(len(A)):
            if not bodo.hiframes.api.isna(A, i):
                s += A[i]
                count += count_1

        res = bodo.hiframes.series_kernels._mean_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, 'var')
def overload_series_var(S):
    def impl(S):  # pragma: no cover
        m = S.mean()
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
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
        A = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
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

    if not is_overload_str(method, 'pearson'):
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
            in_arr = bodo.hiframes.api.get_series_data(S)
            numba.parfor.init_prange()
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
        in_arr = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
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
            in_arr = bodo.hiframes.api.get_series_data(S)
            numba.parfor.init_prange()
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
        in_arr = bodo.hiframes.api.get_series_data(S)
        numba.parfor.init_prange()
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
    if not is_overload_zero(axis):
        raise ValueError('Series.idxmin(): axis argument not supported')

    # TODO: other types like strings
    if S.dtype == types.none:
        return (lambda S, axis=0, skipna=True:
            bodo.hiframes.api.get_series_data(S).argmin())
    else:
        def impl(S, axis=0, skipna=True):
            i = bodo.hiframes.api.get_series_data(S).argmin()
            index = bodo.hiframes.api.get_series_index(S)
            index_t = bodo.utils.conversion.fix_none_index(index, len(S))
            return index_t[i]
        return impl


@overload_method(SeriesType, 'idxmax')
def overload_series_idxmax(S, axis=0, skipna=True):
    if not is_overload_zero(axis):
        raise ValueError('Series.idxmax(): axis argument not supported')

    # TODO: other types like strings
    if S.dtype == types.none:
        return (lambda S, axis=0, skipna=True:
            bodo.hiframes.api.get_series_data(S).argmax())
    else:
        def impl(S, axis=0, skipna=True):
            i = bodo.hiframes.api.get_series_data(S).argmax()
            index = bodo.hiframes.api.get_series_index(S)
            index_t = bodo.utils.conversion.fix_none_index(index, len(S))
            return index_t[i]
        return impl


@overload_method(SeriesType, 'median')
def overload_series_median(S, axis=None, skipna=None, level=None,
                                                            numeric_only=None):
    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise ValueError('Series.median(): axis argument not supported')

    # TODO: support NA
    return (lambda S, axis=None, skipna=None, level=None, numeric_only=None:
        bodo.libs.array_kernels.median(bodo.hiframes.api.get_series_data(S)))


@overload_method(SeriesType, 'head')
def overload_series_head(S, n=5):

    def impl(S, n=5):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(arr[:n], index_t[:n], name)

    return impl


@overload_method(SeriesType, 'tail')
def overload_series_tail(S, n=5):

    def impl(S, n=5):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(arr[-n:], index_t[-n:], name)

    return impl


@overload_method(SeriesType, 'nlargest')
def overload_series_nlargest(S, n=5, keep='first'):
    # TODO: cache implementation
    # TODO: strings, categoricals
    # TODO: support and test keep semantics
    def impl(S, n=5, keep='first'):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        index_arr = bodo.utils.conversion.coerce_to_ndarray(index_t)
        name = bodo.hiframes.api.get_series_name(S)
        out_arr, out_ind_arr = bodo.libs.array_kernels.nlargest(
            arr, index_arr, n, True, bodo.hiframes.series_kernels.gt_f)
        out_index = bodo.utils.conversion.convert_to_index(out_ind_arr)
        return bodo.hiframes.api.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, 'nsmallest')
def overload_series_nsmallest(S, n=5, keep='first'):
    # TODO: cache implementation
    def impl(S, n=5, keep='first'):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        index_arr = bodo.utils.conversion.coerce_to_ndarray(index_t)
        name = bodo.hiframes.api.get_series_name(S)
        out_arr, out_ind_arr = bodo.libs.array_kernels.nlargest(
            arr, index_arr, n, False, bodo.hiframes.series_kernels.lt_f)
        out_index = bodo.utils.conversion.convert_to_index(out_ind_arr)
        return bodo.hiframes.api.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, 'notna')
def overload_series_notna(S):
    # TODO: make sure this is fused and optimized properly
    return lambda S: S.isna() == False


@overload_method(SeriesType, 'astype')
def overload_series_astype(S, dtype, copy=True, errors='raise'):
    if isinstance(dtype, types.Function) and dtype.key[0] == str:
        def impl_str(S, dtype, copy=True, errors='raise'):
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            # XXX: init_prange() after get data calls to have array available
            # for length calculation before 1D_Var parfor
            numba.parfor.init_prange()
            n = len(arr)
            num_chars = 0
            # get total chars in new array
            for i in numba.parfor.internal_prange(n):
                s = arr[i]
                # TODO: check NA
                num_chars += bodo.libs.str_arr_ext.get_utf8_size(str(s))
            A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars)
            for j in numba.parfor.internal_prange(n):
                s = arr[j]
                A[j] = str(s)  # TODO: check NA

            return bodo.hiframes.api.init_series(A, index, name)

        return impl_str

    # TODO: other data types like datetime, records/tuples
    def impl(S, dtype, copy=True, errors='raise'):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        out_arr = arr.astype(dtype)
        return bodo.hiframes.api.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, 'take')
def overload_series_take(S, indices, axis=0, convert=None, is_copy=True):
    # TODO: categorical, etc.
    def impl(S, indices, axis=0, convert=None, is_copy=True):
        indices_t = bodo.utils.conversion.coerce_to_ndarray(indices)
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        name = bodo.hiframes.api.get_series_name(S)
        return bodo.hiframes.api.init_series(
            arr[indices_t], index_t[indices_t], name)
    return impl


@overload_method(SeriesType, 'argsort')
def overload_series_argsort(S, axis=0, kind='quicksort', order=None):
    # TODO: categorical, etc.
    # TODO: optimize the if path of known to be no NaNs (e.g. after fillna)
    def impl(S, axis=0, kind='quicksort', order=None):
        arr = bodo.hiframes.api.get_series_data(S)
        n = len(arr)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        mask = S.notna().values
        if not mask.all():
            out_arr = np.full(n, -1, np.int64)
            out_arr[mask] = bodo.hiframes.api.argsort(arr[mask])
        else:
            out_arr = bodo.hiframes.api.argsort(arr)
        return bodo.hiframes.api.init_series(
            out_arr, index, name)
    return impl


@overload_method(SeriesType, 'sort_values')
def overload_series_sort_values(S, axis=0, ascending=True, inplace=False,
                                         kind='quicksort', na_position='last'):
    def impl(S, axis=0, ascending=True, inplace=False, kind='quicksort',
                                                           na_position='last'):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        index_t = bodo.utils.conversion.fix_none_index(index, len(arr))
        index_arr = bodo.utils.conversion.coerce_to_array(index_t)
        name = bodo.hiframes.api.get_series_name(S)

        out_arr, out_ind_arr = bodo.hiframes.api.sort(
            arr, index_arr, ascending, inplace)

        out_index = bodo.utils.conversion.convert_to_index(out_ind_arr)
        return bodo.hiframes.api.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, 'append')
def overload_series_append(S, to_append, ignore_index=False,
                                                       verify_integrity=False):

    if is_overload_true(ignore_index):
        if isinstance(to_append, (types.BaseTuple, types.List)):
            def impl_multi_noindex(S, to_append, ignore_index=False,
                                                       verify_integrity=False):
                arr = bodo.hiframes.api.get_series_data(S)
                tup_other = bodo.utils.typing.to_const_tuple(to_append)
                other_arrs = bodo.hiframes.api.get_series_data_tup(tup_other)
                all_arrs = bodo.utils.typing.to_const_tuple(
                    (arr,) + other_arrs)
                out_arr = bodo.hiframes.api.concat(all_arrs)
                return bodo.hiframes.api.init_series(out_arr)
            return impl_multi_noindex

        def impl_single_noindex(S, to_append, ignore_index=False,
                                                       verify_integrity=False):
            arr = bodo.hiframes.api.get_series_data(S)
            other = bodo.hiframes.api.get_series_data(to_append)
            out_arr = bodo.hiframes.api.concat((arr, other))
            return bodo.hiframes.api.init_series(out_arr)
        return impl_single_noindex

    # TODO: other containers of Series
    if isinstance(to_append, (types.BaseTuple, types.List)):
        def impl(S, to_append, ignore_index=False, verify_integrity=False):
            arr = bodo.hiframes.api.get_series_data(S)
            index_arr = bodo.utils.conversion.extract_index_array(S)

            tup_other = bodo.utils.typing.to_const_tuple(to_append)
            other_arrs = bodo.hiframes.api.get_series_data_tup(tup_other)
            other_inds = bodo.utils.conversion.extract_index_array_tup(
                tup_other)
            # TODO: use regular list when tuple is not required
            # all_arrs = [arr]
            # all_inds = [index_arr]
            # for A in to_append:
            #     all_arrs.append(bodo.hiframes.api.get_series_data(A))
            #     all_inds.append(bodo.utils.conversion.extract_index_array(A))


            all_arrs = bodo.utils.typing.to_const_tuple((arr,) + other_arrs)
            all_inds = bodo.utils.typing.to_const_tuple(
                (index_arr,) + other_inds)
            out_arr = bodo.hiframes.api.concat(all_arrs)
            out_index = bodo.hiframes.api.concat(all_inds)
            return bodo.hiframes.api.init_series(out_arr, out_index)

        return impl

    def impl_single(S, to_append, ignore_index=False, verify_integrity=False):
        arr = bodo.hiframes.api.get_series_data(S)
        index_arr = bodo.utils.conversion.extract_index_array(S)
        # name = bodo.hiframes.api.get_series_name(S)

        other = bodo.hiframes.api.get_series_data(to_append)
        other_index = bodo.utils.conversion.extract_index_array(to_append)

        out_arr = bodo.hiframes.api.concat((arr, other))
        out_index = bodo.hiframes.api.concat((index_arr, other_index))
        return bodo.hiframes.api.init_series(out_arr, out_index)

    return impl_single


@overload_method(SeriesType, 'quantile')
def overload_series_quantile(S, q=0.5, interpolation='linear'):
    # TODO: datetime support
    def impl(S, q=0.5, interpolation='linear'):
        arr = bodo.hiframes.api.get_series_data(S)
        return bodo.libs.array_kernels.quantile(arr, q)

    return impl


@overload_method(SeriesType, 'nunique')
def overload_series_nunique(S, dropna=True):
    # TODO: refactor, support NA, dt64
    def impl(S, dropna=True):
        arr = bodo.hiframes.api.get_series_data(S)
        return bodo.hiframes.api.nunique(arr)

    return impl


@overload_method(SeriesType, 'unique')
def overload_series_unique(S):
    # TODO: refactor, support dt64
    def impl(S):
        arr = bodo.hiframes.api.get_series_data(S)
        return bodo.hiframes.api.unique(arr)

    return impl


@overload_method(SeriesType, 'describe')
def overload_series_describe(S, percentiles=None, include=None, exclude=None):
    # TODO: support categorical, dt64, ...
    def impl(S, percentiles=None, include=None, exclude=None):
        name = bodo.hiframes.api.get_series_name(S)
        a_count = np.float64(S.count())
        a_min = np.float64(S.min())
        a_max = np.float64(S.max())
        a_mean = np.float64(S.mean())
        a_std = np.float64(S.std())
        q25 = S.quantile(.25)
        q50 = S.quantile(.5)
        q75 = S.quantile(.75)
        return bodo.hiframes.api.init_series(
            np.array([a_count, a_mean, a_std, a_min, q25, q50, q75, a_max]),
            bodo.utils.conversion.convert_to_index(
                ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']),
            name)

    return impl


@overload_method(SeriesType, 'fillna')
def overload_series_fillna(S, value=None, method=None, axis=None,
                                     inplace=False, limit=None, downcast=None):
    #
    if is_overload_true(inplace):
        if S.dtype == bodo.string_type:
            # optimization: just set null bit if fill is empty
            if (value == types.StringLiteral("")
                        or getattr(value, 'value', "_") == ""):
                return (lambda S, value=None, method=None, axis=None,
                    inplace=False, limit=None, downcast=None:
                    bodo.libs.str_arr_ext.set_null_bits(
                        bodo.hiframes.api.get_series_data(S)))
            # Since string arrays can't be changed, we have to create a new
            # array and update the same Series variable
            # TODO: handle string array reflection
            # TODO: handle init_series() optimization guard for mutability
            def str_fillna_inplace_impl(S, value=None, method=None, axis=None,
                                     inplace=False, limit=None, downcast=None):
                in_arr = bodo.hiframes.api.get_series_data(S)
                n = len(in_arr)
                num_chars = 0
                # get total chars in new array
                for i in numba.parfor.internal_prange(n):
                    s = in_arr[i]
                    if bodo.hiframes.api.isna(in_arr, i):
                        l = bodo.libs.str_arr_ext.get_utf8_size(value)
                    else:
                        l = bodo.libs.str_arr_ext.get_utf8_size(s)
                    num_chars += l
                out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(
                    n, num_chars)
                for j in numba.parfor.internal_prange(n):
                    s = in_arr[j]
                    if bodo.hiframes.api.isna(in_arr, j):
                        s = value
                    out_arr[j] = s
                bodo.hiframes.api.update_series_data(S, out_arr)
                return
            return str_fillna_inplace_impl
        else:
            def fillna_inplace_impl(S, value=None, method=None, axis=None,
                                     inplace=False, limit=None, downcast=None):  # pragma: no cover
                in_arr = bodo.hiframes.api.get_series_data(S)
                for i in numba.parfor.internal_prange(len(in_arr)):
                    s = in_arr[i]
                    if bodo.hiframes.api.isna(in_arr, i):
                        s = value
                    in_arr[i] = s
            return fillna_inplace_impl
    else:
        if S.dtype == bodo.string_type:
            def str_fillna_alloc_impl(S, value=None, method=None, axis=None,
                                     inplace=False, limit=None, downcast=None):  # pragma: no cover
                in_arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                n = len(in_arr)
                num_chars = 0
                # get total chars in new array
                for i in numba.parfor.internal_prange(n):
                    s = in_arr[i]
                    # TODO: fix dist reduce when "num_chars += ..." is in
                    # both branches
                    if bodo.hiframes.api.isna(in_arr, i):
                        l = bodo.libs.str_arr_ext.get_utf8_size(value)
                    else:
                        l = bodo.libs.str_arr_ext.get_utf8_size(s)
                    num_chars += l
                out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(
                    n, num_chars)
                # TODO: fix SSA for loop variables
                for j in numba.parfor.internal_prange(n):
                    s = in_arr[j]
                    if bodo.hiframes.api.isna(in_arr, j):
                        s = value
                    out_arr[j] = s
                return bodo.hiframes.api.init_series(out_arr, index, name)
            return str_fillna_alloc_impl
        else:
            def fillna_impl(S, value=None, method=None, axis=None,
                                     inplace=False, limit=None, downcast=None):  # pragma: no cover
                in_arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                n = len(in_arr)
                out_arr = np.empty(n, in_arr.dtype)
                for i in numba.parfor.internal_prange(n):
                    s = in_arr[i]
                    if bodo.hiframes.api.isna(in_arr, i):
                        s = value
                    out_arr[i] = s
                return bodo.hiframes.api.init_series(out_arr, index, name)
            return fillna_impl


@overload_method(SeriesType, 'dropna')
def overload_series_dropna(S, axis=0, inplace=False):
    # TODO: fix inplace index output
    if is_overload_true(inplace):
        if S.dtype == bodo.string_type:
            def dropna_str_inplace_impl(S, axis=0, inplace=False):
                in_arr = bodo.hiframes.api.get_series_data(S)
                mask = S.notna().values
                index_arr = bodo.utils.conversion.extract_index_array(S)
                out_index = bodo.utils.conversion.convert_to_index(
                    index_arr[mask])
                out_arr = bodo.hiframes.series_kernels._series_dropna_str_alloc_impl_inner(in_arr)
                bodo.hiframes.api.update_series_data(S, out_arr)
                out_index_t = bodo.utils.conversion.force_convert_index(
                    out_index, bodo.hiframes.api.get_series_index(S))
                bodo.hiframes.api.update_series_index(S, out_index_t)
                return
            return dropna_str_inplace_impl
        else:
            def dropna_inplace_impl(S, axis=0, inplace=False):  # pragma: no cover
                in_arr = bodo.hiframes.api.get_series_data(S)
                index_arr = bodo.utils.conversion.extract_index_array(S)
                mask = S.notna().values
                out_index = bodo.utils.conversion.convert_to_index(
                    index_arr[mask])
                out_arr = in_arr[mask]
                bodo.hiframes.api.update_series_data(S, out_arr)
                out_index_t = bodo.utils.conversion.force_convert_index(
                    out_index, bodo.hiframes.api.get_series_index(S))
                bodo.hiframes.api.update_series_index(S, out_index_t)
                return
            return dropna_inplace_impl
    else:
        if S.dtype == bodo.string_type:
            def dropna_str_impl(S, axis=0, inplace=False):
                in_arr = bodo.hiframes.api.get_series_data(S)
                name = bodo.hiframes.api.get_series_name(S)
                mask = S.notna().values
                index_arr = bodo.utils.conversion.extract_index_array(S)
                out_index = bodo.utils.conversion.convert_to_index(
                    index_arr[mask])
                out_arr = bodo.hiframes.series_kernels._series_dropna_str_alloc_impl_inner(in_arr)
                return bodo.hiframes.api.init_series(out_arr, out_index, name)
            return dropna_str_impl
        else:
            def dropna_impl(S, axis=0, inplace=False):  # pragma: no cover
                in_arr = bodo.hiframes.api.get_series_data(S)
                name = bodo.hiframes.api.get_series_name(S)
                index_arr = bodo.utils.conversion.extract_index_array(S)
                mask = S.notna().values
                out_index = bodo.utils.conversion.convert_to_index(
                    index_arr[mask])
                out_arr = in_arr[mask]
                return bodo.hiframes.api.init_series(out_arr, out_index, name)
            return dropna_impl


@overload_method(SeriesType, 'shift')
def overload_series_shift(S, periods=1, freq=None, axis=0, fill_value=None):
    # TODO: handle dt64, strings
    def impl(S, periods=1, freq=None, axis=0, fill_value=None):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        out_arr = bodo.hiframes.rolling.shift(arr, periods, False)
        return bodo.hiframes.api.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, 'pct_change')
def overload_series_pct_change(S, periods=1, fill_method='pad', limit=None,
                                                                    freq=None):
    # TODO: handle dt64, strings
    def impl(S, periods=1, fill_method='pad', limit=None, freq=None):
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        out_arr = bodo.hiframes.rolling.pct_change(arr, periods, False)
        return bodo.hiframes.api.init_series(out_arr, index, name)

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
                arr = bodo.hiframes.api.get_series_data(S)
                index = bodo.hiframes.api.get_series_index(S)
                name = bodo.hiframes.api.get_series_name(S)
                numba.parfor.init_prange()
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
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            # other could be tuple, list, array, Index, or Series
            other_arr = bodo.utils.conversion.coerce_to_array(other)
            numba.parfor.init_prange()
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
    # install unary operators: ~, -, +
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
