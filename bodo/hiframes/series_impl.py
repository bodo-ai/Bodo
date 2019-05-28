"""
Implementation of Series attributes and methods.
"""
import numpy as np
import pandas as pd
import numba
from numba import types
from numba.extending import overload, overload_attribute, overload_method
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false)


@overload_attribute(SeriesType, 'index')
def overload_series_index(s):
    # None should be range type
    if s.index == types.none:
        return lambda s: bodo.hiframes.pd_index_ext.init_range_index(
            0, len(bodo.hiframes.api.get_series_data(s)), 1)

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
