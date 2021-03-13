# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implementation of Series attributes and methods using overload.
"""
import operator

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import overload, overload_attribute, overload_method

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArray,
    PDCategoricalDtype,
)
from bodo.hiframes.pd_offsets_ext import is_offsets_type
from bodo.hiframes.pd_series_ext import (
    HeterogeneousSeriesType,
    SeriesType,
    if_series_to_array_type,
)
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.utils.transform import gen_const_tup, is_var_size_item_array_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_literal_value,
    get_overload_const_int,
    get_overload_const_str,
    is_common_scalar_dtype,
    is_iterable_type,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_nan,
    is_overload_constant_str,
    is_overload_false,
    is_overload_int,
    is_overload_none,
    is_overload_true,
    is_overload_zero,
    is_scalar_type,
    raise_bodo_error,
)


@overload_attribute(SeriesType, "index", inline="always")
def overload_series_index(s):
    return lambda s: bodo.hiframes.pd_series_ext.get_series_index(s)


@overload_attribute(SeriesType, "values", inline="always")
def overload_series_values(s):
    return lambda s: bodo.hiframes.pd_series_ext.get_series_data(s)


@overload_attribute(SeriesType, "dtype", inline="always")
def overload_series_dtype(s):
    # TODO: check other dtypes like tuple, etc.
    if s.dtype == bodo.string_type:
        raise BodoError("Series.dtype not supported for string Series yet")

    return lambda s: bodo.hiframes.pd_series_ext.get_series_data(s).dtype


@overload_attribute(SeriesType, "shape")
def overload_series_shape(s):
    return lambda s: (len(bodo.hiframes.pd_series_ext.get_series_data(s)),)


@overload_attribute(SeriesType, "ndim", inline="always")
def overload_series_ndim(s):
    return lambda s: 1


@overload_attribute(SeriesType, "size")
def overload_series_size(s):
    return lambda s: len(bodo.hiframes.pd_series_ext.get_series_data(s))


@overload_attribute(SeriesType, "T", inline="always")
def overload_series_T(s):
    return lambda s: s


@overload_attribute(SeriesType, "hasnans", inline="always")
def overload_series_hasnans(s):
    return lambda s: s.isna().sum() != 0


@overload_attribute(SeriesType, "empty")
def overload_series_empty(s):
    return lambda s: len(bodo.hiframes.pd_series_ext.get_series_data(s)) == 0


@overload_attribute(SeriesType, "dtypes", inline="always")
def overload_series_dtypes(s):
    return lambda s: s.dtype


@overload_attribute(HeterogeneousSeriesType, "name", inline="always")
@overload_attribute(SeriesType, "name", inline="always")
def overload_series_name(s):
    return lambda s: bodo.hiframes.pd_series_ext.get_series_name(s)


@overload(len, no_unliteral=True)
def overload_series_len(S):
    if isinstance(S, SeriesType):
        return lambda S: len(bodo.hiframes.pd_series_ext.get_series_data(S))


@overload_method(SeriesType, "copy", inline="always", no_unliteral=True)
def overload_series_copy(S, deep=True):
    # TODO: test all Series data types
    # XXX specialized kinds until branch pruning is tested and working well
    if is_overload_true(deep):

        def impl1(S, deep=True):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            return bodo.hiframes.pd_series_ext.init_series(arr.copy(), index, name)

        return impl1

    if is_overload_false(deep):

        def impl2(S, deep=True):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

        return impl2

    def impl(S, deep=True):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        if deep:
            arr = arr.copy()
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

    return impl


@overload_method(SeriesType, "to_list", no_unliteral=True)
@overload_method(SeriesType, "tolist", no_unliteral=True)
def overload_series_to_list(S):
    # TODO: test all Series data types
    def impl(S):  # pragma: no cover
        l = list()
        for i in range(len(S)):
            # using iat directly on S to box Timestamp/... properly
            l.append(S.iat[i])
        return l

    return impl


@overload_method(SeriesType, "to_numpy", inline="always", no_unliteral=True)
def overload_series_to_numpy(S, dtype=None, copy=False, na_value=None):

    unsupported_args = dict(dtype=dtype, copy=copy, na_value=na_value)
    arg_defaults = dict(dtype=None, copy=False, na_value=None)
    check_unsupported_args("Series.to_numpy", unsupported_args, arg_defaults)

    def impl(S, dtype=None, copy=False, na_value=None):  # pragma: no cover
        return S.values

    return impl


@overload_method(SeriesType, "reset_index", inline="always", no_unliteral=True)
def overload_series_reset_index(S, level=None, drop=False, name=None, inplace=False):
    """overload for Series.reset_index(). Note that it requires the series'
    name and index name to be literal values, and so will only currently
    work in very specific cases where these are known at compile time
    (e.g. groupby("A")["B"].sum().reset_index())"""

    unsupported_args = dict(name=name, inplace=inplace)
    arg_defaults = dict(name=None, inplace=False)
    check_unsupported_args("Series.reset_index", unsupported_args, arg_defaults)

    # we only support dropping all levels currently
    if not bodo.hiframes.pd_dataframe_ext._is_all_levels(S, level):  # pragma: no cover
        raise_bodo_error(
            "Series.reset_index(): only dropping all index levels supported"
        )

    # make sure 'drop' is a constant bool
    if not is_overload_constant_bool(drop):  # pragma: no cover
        raise_bodo_error(
            "Series.reset_index(): 'drop' parameter should be a constant boolean value"
        )

    if is_overload_true(drop):

        def impl_drop(
            S, level=None, drop=False, name=None, inplace=False
        ):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_index_ext.init_range_index(0, len(arr), 1, None)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

        return impl_drop

    def get_name_literal(name_typ, is_index=False, series_name=None):
        """return literal value or throw error in non-literal type"""
        # if Series name is None, Pandas uses 0.
        # if Index name is None, Pandas uses "index".
        if is_overload_none(name_typ):
            if is_index:
                return "index" if series_name != "index" else "level_0"
            return 0

        if is_literal_type(name_typ):
            return get_literal_value(name_typ)
        else:
            raise BodoError(
                "Series.reset_index() not supported for non-literal series names"
            )

    # TODO: [BE-100] Support name argument with a constant string.
    series_name = get_name_literal(S.name_typ)
    index_name = get_name_literal(S.index.name_typ, True, series_name)
    columns = [
        index_name,
        series_name,
    ]

    func_text = "def _impl(S, level=None, drop=False, name=None, inplace=False):\n"
    func_text += "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "    df_index = bodo.hiframes.pd_index_ext.init_range_index(0, len(S), 1, None)\n"
    func_text += "    col_var = {}\n".format(gen_const_tup(columns))
    func_text += "    return bodo.hiframes.pd_dataframe_ext.init_dataframe((index, arr), df_index, col_var)\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
        },
        loc_vars,
    )
    _impl = loc_vars["_impl"]
    return _impl


@overload_method(SeriesType, "isna", inline="always", no_unliteral=True)
@overload_method(SeriesType, "isnull", inline="always", no_unliteral=True)
def overload_series_isna(S):
    # TODO: series that have different underlying data type than dtype
    # like records/tuples
    def impl(S):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = bodo.libs.array_kernels.isna(arr, i)

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "round", inline="always", no_unliteral=True)
def overload_series_round(S, decimals=0):
    def impl(S, decimals=0):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = np.empty(n, arr.dtype)
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = np.round(arr[i], decimals)

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "sum", inline="always", no_unliteral=True)
def overload_series_sum(
    S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.sum", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.sum(): axis argument not supported")

    # TODO: series that have different underlying data type than dtype
    # like records/tuples
    if isinstance(S.dtype, types.Integer):
        retty = types.intp
    elif isinstance(S.dtype, types.Boolean):
        retty = np.int64
    else:
        retty = S.dtype
    val_zero = retty(0)

    # For integer array we cannot handle the missing values because
    # we cannot mix np.nan with integers
    if isinstance(S.dtype, types.Float) and (
        not is_overload_true(skipna) or not is_overload_zero(min_count)
    ):

        def impl(
            S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
        ):  # pragma: no cover
            A = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = val_zero
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(A)):
                val = val_zero
                count_val = 0
                if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                    val = A[i]
                    count_val = 1
                s += val
                count += count_val
            res = bodo.hiframes.series_kernels._var_handle_mincount(s, count, min_count)
            return res

    else:

        def impl(
            S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
        ):  # pragma: no cover
            A = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = val_zero
            for i in numba.parfors.parfor.internal_prange(len(A)):
                val = val_zero
                if not bodo.libs.array_kernels.isna(A, i):
                    val = A[i]
                s += val
            return s

    return impl


@overload_method(SeriesType, "prod", inline="always", no_unliteral=True)
@overload_method(SeriesType, "product", inline="always", no_unliteral=True)
def overload_series_prod(
    S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.product", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.product(): axis argument not supported")

    val_one = S.dtype(1)

    # For integer array we cannot handle the missing values because
    # we cannot mix np.nan with integers
    if isinstance(S.dtype, types.Float):

        def impl(
            S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
        ):  # pragma: no cover
            A = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = val_one
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(A)):
                val = val_one
                count_val = 0
                if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                    val = A[i]
                    count_val = 1
                count += count_val
                s *= val
            res = bodo.hiframes.series_kernels._var_handle_mincount(s, count, min_count)
            return res

    else:

        def impl(
            S, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
        ):  # pragma: no cover
            A = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = val_one
            for i in numba.parfors.parfor.internal_prange(len(A)):
                val = val_one
                if not bodo.libs.array_kernels.isna(A, i):
                    val = A[i]
                s *= val
            return s

    return impl


@overload_method(SeriesType, "any", inline="always", no_unliteral=True)
def overload_series_any(S, axis=0, bool_only=None, skipna=True, level=None):

    unsupported_args = dict(axis=axis, bool_only=bool_only, skipna=skipna, level=level)
    arg_defaults = dict(axis=0, bool_only=None, skipna=True, level=None)
    check_unsupported_args("Series.any", unsupported_args, arg_defaults)

    def impl(S, axis=0, bool_only=None, skipna=True, level=None):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0
            if not bodo.libs.array_kernels.isna(A, i):
                val = int(A[i])
            count += val
        return count != 0

    return impl


@overload_method(SeriesType, "equals", inline="always", no_unliteral=True)
def overload_series_any(S, other):
    if not isinstance(other, SeriesType):
        raise BodoError("Series.equals() 'other' must be a Series")

    # Bodo Limitation. Compilation fails with ArrayItemArrayType because A1[i] != A2[i]
    # doesn't work properly
    # TODO: [BE-109] Support ArrayItemArrayType
    if isinstance(S.data, bodo.ArrayItemArrayType):
        raise BodoError(
            "Series.equals() not supported for Series where each element is an array or list"
        )

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.equals.html#pandas.Series.equals
    # From the docs: "DataFrames df and different_data_type have different types for the same values for their
    # elements, and will return False even though their column labels are the same values and types"
    # This check ensures the types are exactly the same (even int32 and int64 returns False)

    # We match this behavior by checking that both series have the "same" types at compile time,
    # and returning False if not.
    # TODO: [BE-132] Check that the index and name values are equal
    if S.data != other.data:
        return lambda S, other: False  # pragma: no cover

    def impl(S, other):  # pragma: no cover
        A1 = bodo.hiframes.pd_series_ext.get_series_data(S)
        A2 = bodo.hiframes.pd_series_ext.get_series_data(other)
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A1)):
            val = 0
            test1 = bodo.libs.array_kernels.isna(A1, i)
            test2 = bodo.libs.array_kernels.isna(A2, i)
            # Direct comparison "if test1 != test2" does not compile for numba
            if (test1 and not test2) or (not test1 and test2):
                val = 1
            else:
                if not test1:
                    if A1[i] != A2[i]:
                        val = 1
            count += val
        return count == 0

    return impl


@overload_method(SeriesType, "all", inline="always", no_unliteral=True)
def overload_series_all(S, axis=0, bool_only=None, skipna=True, level=None):

    unsupported_args = dict(axis=axis, bool_only=bool_only, skipna=skipna, level=level)
    arg_defaults = dict(axis=0, bool_only=None, skipna=True, level=None)
    check_unsupported_args("Series.all", unsupported_args, arg_defaults)

    def impl(S, axis=0, bool_only=None, skipna=True, level=None):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0
            if not bodo.libs.array_kernels.isna(A, i):
                val = int(not A[i])
            count += val
        return count == 0

    return impl


@overload_method(SeriesType, "mad", inline="always", no_unliteral=True)
def overload_series_mad(S, axis=None, skipna=True, level=None):

    unsupported_args = dict(axis=axis, level=level)
    arg_defaults = dict(axis=None, level=None)
    check_unsupported_args("Series.mad", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.mad(): axis argument not supported")

    # see core/nanops.py/nanmean() for output types
    # TODO: more accurate port of dtypes from pandas
    sum_dtype = types.float64
    count_dtype = types.float64
    if S.dtype == types.float32:
        sum_dtype = types.float32
        count_dtype = types.float32

    val_0 = sum_dtype(0)
    count_0 = count_dtype(0)
    count_1 = count_dtype(1)

    def impl(S, axis=None, skipna=True, level=None):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        # First computing the mean
        s_mean = val_0
        count = count_0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = val_0
            count_val = count_0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = A[i]
                count_val = count_1
            s_mean += val
            count += count_val

        res_mean = bodo.hiframes.series_kernels._mean_handle_nan(s_mean, count)
        # Second computing the mad
        s_mad = val_0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = val_0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = abs(A[i] - res_mean)
            s_mad += val

        res_mad = bodo.hiframes.series_kernels._mean_handle_nan(s_mad, count)
        return res_mad

    return impl


@overload_method(SeriesType, "mean", inline="always", no_unliteral=True)
def overload_series_mean(S, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("Series.mean", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.mean(): axis argument not supported")
    # see core/nanops.py/nanmean() for output types
    # TODO: more accurate port of dtypes from pandas
    sum_dtype = types.float64
    count_dtype = types.float64
    if S.dtype == types.float32:
        sum_dtype = types.float32
        count_dtype = types.float32

    val_0 = sum_dtype(0)
    count_0 = count_dtype(0)
    count_1 = count_dtype(1)

    def impl(
        S, axis=None, skipna=None, level=None, numeric_only=None
    ):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        s = val_0
        count = count_0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = val_0
            count_val = count_0
            if not bodo.libs.array_kernels.isna(A, i):
                val = A[i]
                count_val = count_1
            s += val
            count += count_val

        res = bodo.hiframes.series_kernels._mean_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, "sem", inline="always", no_unliteral=True)
def overload_series_sem(
    S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.sem", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.sem(): axis argument not supported")

    def impl(
        S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
    ):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        first_moment = 0
        second_moment = 0
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0
            count_val = 0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = A[i]
                count_val = 1
            first_moment += val
            second_moment += val * val
            count += count_val

        s = second_moment - first_moment * first_moment / count
        res = bodo.hiframes.series_kernels._handle_nan_count_ddof(s, count, ddof)
        res_out = (res / count) ** 0.5
        return res_out

    return impl


# Formula for Kurtosis is available at
# https://en.wikipedia.org/wiki/Kurtosis
# Precise formula taken from ./pandas/core/nanops.py [nankurt]
@overload_method(SeriesType, "kurt", inline="always", no_unliteral=True)
@overload_method(SeriesType, "kurtosis", inline="always", no_unliteral=True)
def overload_series_kurt(S, axis=0, skipna=True, level=None, numeric_only=None):

    unsupported_args = dict(axis=axis, level=level, numeric_only=numeric_only)
    arg_defaults = dict(axis=0, level=None, numeric_only=None)
    check_unsupported_args("Series.kurtosis", unsupported_args, arg_defaults)

    def impl(S, axis=0, skipna=True, level=None, numeric_only=None):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        first_moment = 0.0
        second_moment = 0.0
        third_moment = 0.0
        fourth_moment = 0.0
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0.0
            count_val = 0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = np.float64(A[i])
                count_val = 1
            first_moment += val
            second_moment += val ** 2
            third_moment += val ** 3
            fourth_moment += val ** 4
            count += count_val
        res = bodo.hiframes.series_kernels.compute_kurt(
            first_moment, second_moment, third_moment, fourth_moment, count
        )
        return res

    return impl


# Formula for skewness is available at
# https://en.wikipedia.org/wiki/Skewness
# Precise formula taken from ./pandas/core/nanops.py [nanskew]
@overload_method(SeriesType, "skew", inline="always", no_unliteral=True)
def overload_series_skew(S, axis=None, skipna=True, level=None, numeric_only=None):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.skew", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.skew(): axis argument not supported")

    def impl(
        S, axis=None, skipna=True, level=None, numeric_only=None
    ):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        first_moment = 0.0
        second_moment = 0.0
        third_moment = 0.0
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0.0
            count_val = 0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = np.float64(A[i])
                count_val = 1
            first_moment += val
            second_moment += val ** 2
            third_moment += val ** 3
            count += count_val
        res = bodo.hiframes.series_kernels.compute_skew(
            first_moment, second_moment, third_moment, count
        )
        return res

    return impl


@overload_method(SeriesType, "var", inline="always", no_unliteral=True)
def overload_series_var(
    S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.var", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.var(): axis argument not supported")

    def impl(
        S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
    ):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        first_moment = 0
        second_moment = 0
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            val = 0
            count_val = 0
            if not bodo.libs.array_kernels.isna(A, i) or not skipna:
                val = A[i]
                count_val = 1
            first_moment += val
            second_moment += val * val
            count += count_val

        s = second_moment - first_moment * first_moment / count
        res = bodo.hiframes.series_kernels._handle_nan_count_ddof(s, count, ddof)
        return res

    return impl


@overload_method(SeriesType, "std", inline="always", no_unliteral=True)
def overload_series_std(
    S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
):

    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.std", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.std(): axis argument not supported")

    def impl(
        S, axis=None, skipna=True, level=None, ddof=1, numeric_only=None
    ):  # pragma: no cover
        return S.var(skipna=skipna, ddof=ddof) ** 0.5

    return impl


@overload_method(SeriesType, "dot", inline="always", no_unliteral=True)
def overload_series_var(S, other):
    def impl(S, other):  # pragma: no cover
        A1 = bodo.hiframes.pd_series_ext.get_series_data(S)
        A2 = bodo.hiframes.pd_series_ext.get_series_data(other)
        numba.parfors.parfor.init_prange()
        e_dot = 0
        for i in numba.parfors.parfor.internal_prange(len(A1)):
            val1 = A1[i]
            val2 = A2[i]
            e_dot += val1 * val2

        return e_dot

    return impl


@overload_method(SeriesType, "cumsum", inline="always", no_unliteral=True)
def overload_series_cumsum(S, axis=0, skipna=True):

    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.cumsum", unsupported_args, arg_defaults)

    # TODO: support skipna
    def impl(S, axis=0, skipna=True):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(A.cumsum(), index, name)

    return impl


@overload_method(SeriesType, "cumprod", inline="always", no_unliteral=True)
def overload_series_cumprod(S, axis=0, skipna=True):

    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.cumprod", unsupported_args, arg_defaults)

    # TODO: support skipna
    def impl(S, axis=0, skipna=True):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(A.cumprod(), index, name)

    return impl


@overload_method(SeriesType, "cummin", inline="always", no_unliteral=True)
def overload_series_cummin(S, axis=0, skipna=True):
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.cummax", unsupported_args, arg_defaults)

    # TODO: support skipna
    #
    # The big difference between cumsum/cumprod and cummin/cummax
    # is that cumsum/cumprod are part of numpy and implemented in NUMBA,
    # but cummin/cummax are not implemented in numpy and therefore not in numba.
    # Thus for cummin/cummax we need to roll out our own implementation.
    # We cannot use parfor as it is not easily parallelizable and thus requires a
    # hand crafted parallelization (see dist_cummin/dist_cummax)
    def impl(S, axis=0, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(
            bodo.libs.array_kernels.cummin(arr), index, name
        )

    return impl


@overload_method(SeriesType, "cummax", inline="always", no_unliteral=True)
def overload_series_cummax(S, axis=0, skipna=True):

    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.cummax", unsupported_args, arg_defaults)

    # Remarks for cummin applies here.
    # TODO: support skipna
    def impl(S, axis=0, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(
            bodo.libs.array_kernels.cummax(arr), index, name
        )

    return impl


@overload_method(SeriesType, "rename", inline="always", no_unliteral=True)
def overload_series_rename(
    S, index=None, axis=None, copy=True, inplace=False, level=None, errors="ignore"
):
    # TODO: Pandas has * after index, so only index should be able to be provided
    # without kwargs.

    if not (index == bodo.string_type or isinstance(index, types.StringLiteral)):
        raise BodoError("Series.rename() 'index' can only be a string")

    unsupported_args = dict(copy=copy, inplace=inplace, level=level, errors=errors)
    arg_defaults = dict(copy=True, inplace=False, level=None, errors="ignore")
    check_unsupported_args("Series.rename", unsupported_args, arg_defaults)

    # Pandas ignores axis value entirely (in both implementation and documented)
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rename.html
    # We can match Pandas and just ignore it.

    # TODO: support index rename, kws
    def impl(
        S, index=None, axis=None, copy=True, inplace=False, level=None, errors="ignore"
    ):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        s_index = bodo.hiframes.pd_series_ext.get_series_index(S)
        return bodo.hiframes.pd_series_ext.init_series(A, s_index, index)

    return impl


@overload_method(SeriesType, "abs", inline="always", no_unliteral=True)
def overload_series_abs(S):
    # TODO: timedelta
    def impl(S):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(np.abs(A), index, name)

    return impl


@overload_method(SeriesType, "count", no_unliteral=True)
def overload_series_count(S, level=None):

    unsupported_args = dict(level=level)
    arg_defaults = dict(level=None)
    check_unsupported_args("Series.count", unsupported_args, arg_defaults)

    def impl(S, level=None):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            count_val = 0
            if not bodo.libs.array_kernels.isna(A, i):
                count_val = 1
            count += count_val

        res = count
        return res

    return impl


@overload_method(SeriesType, "corr", inline="always", no_unliteral=True)
def overload_series_corr(S, other, method="pearson", min_periods=None):

    unsupported_args = dict(method=method, min_periods=min_periods)
    arg_defaults = dict(method="pearson", min_periods=None)
    check_unsupported_args("Series.corr", unsupported_args, arg_defaults)

    def impl(S, other, method="pearson", min_periods=None):  # pragma: no cover
        n = S.count()
        # TODO: check lens
        ma = S.sum()
        mb = other.sum()
        # TODO: check aligned nans, (S.notna() != other.notna()).any()
        a = n * ((S * other).sum()) - ma * mb
        b1 = n * (S ** 2).sum() - ma ** 2
        b2 = n * (other ** 2).sum() - mb ** 2
        # TODO: np.clip
        # TODO: np.true_divide?
        return a / np.sqrt(b1 * b2)

    return impl


@overload_method(SeriesType, "cov", inline="always", no_unliteral=True)
def overload_series_cov(S, other, min_periods=None, ddof=1):

    unsupported_args = dict(min_periods=min_periods)
    arg_defaults = dict(min_periods=None)
    check_unsupported_args("Series.cov", unsupported_args, arg_defaults)

    # TODO: use online algorithm, e.g. StatFunctions.scala
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def impl(S, other, min_periods=None, ddof=1):  # pragma: no cover
        # TODO: Handle different lens (fails due to array analysis)
        # https://github.com/pandas-dev/pandas/blob/b58e2b86861fe248008d1f140752d1a558cd6516/pandas/core/nanops.py#L1493
        ma = S.mean()
        mb = other.mean()
        total = ((S - ma) * (other - mb)).sum()
        N = np.float64(S.count() - ddof)
        nonzero_len = S.count() * other.count()
        return _series_cov_helper(total, N, nonzero_len)

    return impl


def _series_cov_helper(sum_val, N, nonzero_len):  # pragma: no cover
    # Dummy function to overload
    return


@overload(_series_cov_helper, no_unliteral=True)
def _overload_series_cov_helper(sum_val, N, nonzero_len):
    def impl(sum_val, N, nonzero_len):  # pragma: no cover
        if not nonzero_len:
            # https://github.com/pandas-dev/pandas/blob/v1.2.1/pandas/core/series.py#L2347
            return np.nan
        if N <= 0.0:
            # Division should be handled by np.true_divide in the future, but
            # this seems to produce a bus error.
            # https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/function_base.py#L2469
            sign = np.sign(sum_val)
            return np.inf * sign
        return sum_val / N

    return impl


@overload_method(SeriesType, "min", inline="always", no_unliteral=True)
def overload_series_min(S, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("Series.min", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise BodoError("Series.min(): axis argument not supported")

    # TODO: min/max of string dtype, etc.
    if S.dtype == types.NPDatetime("ns"):

        def impl_dt64(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(in_arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(in_arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                    count_val = 1
                s = min(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    # categorical case
    if isinstance(S.dtype, PDCategoricalDtype):
        if not S.dtype.ordered:  # pragma: no cover
            raise BodoError("Series.min(): only ordered categoricals are possible")

        def impl_cat(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(in_arr)
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(codes)):
                v = codes[i]
                if v == -1:
                    continue
                s = min(s, v)
                count += 1

            res = bodo.hiframes.series_kernels._box_cat_val(s, in_arr.dtype, count)
            return res

        return impl_cat

    # TODO: Setup datetime_date_array.dtype() so we can reuse impl
    if S.dtype == datetime_date_type:

        def impl_date(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_date_max_value()
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(in_arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(in_arr, i):
                    val = in_arr[i]
                    count_val = 1
                s = min(s, val)
                count += count_val
            res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
            return res

        return impl_date

    def impl(
        S, axis=None, skipna=None, level=None, numeric_only=None
    ):  # pragma: no cover
        in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_max_value(in_arr.dtype)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
            val = s
            count_val = 0
            if not bodo.libs.array_kernels.isna(in_arr, i):
                val = in_arr[i]
                count_val = 1
            s = min(s, val)
            count += count_val
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


# inlining manually instead of inline="always" since Numba's max overload for iterables
# causes confusion for the inliner. TODO(ehsan): fix Numba's bug
@overload(max, no_unliteral=True)
def overload_series_builtins_max(S):
    if isinstance(S, SeriesType):

        def impl(S):
            return S.max()

        return impl


@overload(min, no_unliteral=True)
def overload_series_builtins_min(S):
    if isinstance(S, SeriesType):

        def impl(S):
            return S.min()

        return impl


@overload(sum, no_unliteral=True)
def overload_series_builtins_sum(S):
    if isinstance(S, SeriesType):

        def impl(S):
            return S.sum()

        return impl


@overload(np.prod, inline="always", no_unliteral=True)
def overload_series_prod(S):
    if isinstance(S, SeriesType):

        def impl(S):  # pragma: no cover
            return S.prod()

        return impl


@overload_method(SeriesType, "max", inline="always", no_unliteral=True)
def overload_series_max(S, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("Series.max", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.max(): axis argument not supported")

    # datetime case
    if S.dtype == types.NPDatetime("ns"):

        def impl_dt64(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_min_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(in_arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(in_arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                    count_val = 1
                s = max(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    # categorical case
    if isinstance(S.dtype, PDCategoricalDtype):
        if not S.dtype.ordered:  # pragma: no cover
            raise BodoError("Series.max(): only ordered categoricals are possible")

        def impl_cat(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(in_arr)
            numba.parfors.parfor.init_prange()
            s = -1
            # keeping track of NAs is not necessary for max since all valid codes are
            # greater than -1
            for i in numba.parfors.parfor.internal_prange(len(codes)):
                s = max(s, codes[i])

            res = bodo.hiframes.series_kernels._box_cat_val(s, in_arr.dtype, 1)
            return res

        return impl_cat

    # TODO: Setup datetime_date_array.dtype() so we can reuse impl
    if S.dtype == datetime_date_type:

        def impl_date(
            S, axis=None, skipna=None, level=None, numeric_only=None
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_date_min_value()
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(in_arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(in_arr, i):
                    val = in_arr[i]
                    count_val = 1
                s = max(s, val)
                count += count_val
            res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
            return res

        return impl_date

    def impl(
        S, axis=None, skipna=None, level=None, numeric_only=None
    ):  # pragma: no cover
        in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_min_value(in_arr.dtype)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
            val = s
            count_val = 0
            if not bodo.libs.array_kernels.isna(in_arr, i):
                val = in_arr[i]
                count_val = 1
            s = max(s, val)
            count += count_val
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


@overload_method(SeriesType, "idxmin", inline="always", no_unliteral=True)
def overload_series_idxmin(S, axis=0, skipna=True):

    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.idxmin", unsupported_args, arg_defaults)

    # TODO: Make sure we handle the issue with numpy library leading to argmin
    # https://github.com/pandas-dev/pandas/blob/7d32926db8f7541c356066dcadabf854487738de/pandas/compat/numpy/function.py#L91

    # Pandas restrictions:
    # Only supported for numeric types with numpy arrays
    # - int, floats, bool, dt64, td64. (maybe complex)
    # Bodo restrictions:
    # - td64 (TODO: support td64)
    # - Pandas cannot support BooleanArray,
    # so we may not support bool if we map it to BooleanArray
    if not (
        S.dtype == types.none
        or (
            bodo.utils.utils.is_np_array_typ(S.data)
            and (
                S.dtype == bodo.datetime64ns
                or isinstance(S.dtype, (types.Number, types.Boolean))
            )
        )
    ):
        raise BodoError(
            f"Series.idxmin() only supported for non-nullable numeric array types. Array type: {S.data} not supported."
        )
    if S.dtype == types.none:
        return (
            lambda S, axis=0, skipna=True: bodo.hiframes.pd_series_ext.get_series_data(
                S
            ).argmin()
        )
    else:
        # TODO: Make sure -1 is replaced with np.nan
        def impl(S, axis=0, skipna=True):  # pragma: no cover
            i = bodo.hiframes.pd_series_ext.get_series_data(S).argmin()
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            return index[i]

        return impl


@overload_method(SeriesType, "idxmax", inline="always", no_unliteral=True)
def overload_series_idxmax(S, axis=0, skipna=True):

    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("Series.idxmax", unsupported_args, arg_defaults)

    # TODO: Make sure we handle the issue with numpy library leading to argmax
    # https://github.com/pandas-dev/pandas/blob/7d32926db8f7541c356066dcadabf854487738de/pandas/compat/numpy/function.py#L103

    # Pandas restrictions:
    # Only supported for numeric types with numpy arrays
    # - int, floats, bool, dt64, td64. (maybe complex)
    # Bodo restrictions:
    # - td64 (TODO: support td64)
    # - Pandas cannot support BooleanArray,
    # so we may not support bool if we map it to BooleanArray
    if not (
        S.dtype == types.none
        or (
            bodo.utils.utils.is_np_array_typ(S.data)
            and (
                S.dtype == bodo.datetime64ns
                or isinstance(S.dtype, (types.Number, types.Boolean))
            )
        )
    ):
        raise BodoError(
            f"Series.idxmax() only supported for non-nullable numeric array types. Array type: {S.data} not supported."
        )
    # TODO: other types like strings
    if S.dtype == types.none:
        return (
            lambda S, axis=0, skipna=True: bodo.hiframes.pd_series_ext.get_series_data(
                S
            ).argmax()
        )

    else:
        # TODO: Make sure -1 is replaced with np.nan
        def impl(S, axis=0, skipna=True):  # pragma: no cover
            i = bodo.hiframes.pd_series_ext.get_series_data(S).argmax()
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            return index[i]

        return impl


@overload_attribute(SeriesType, "is_monotonic", inline="always")
@overload_attribute(SeriesType, "is_monotonic_increasing", inline="always")
def overload_series_is_monotonic_increasing(S):
    return lambda S: bodo.libs.array_kernels.series_monotonicity(
        bodo.hiframes.pd_series_ext.get_series_data(S), 1
    )


@overload_attribute(SeriesType, "is_monotonic_decreasing", inline="always")
def overload_series_is_monotonic_decreasing(S):
    return lambda S: bodo.libs.array_kernels.series_monotonicity(
        bodo.hiframes.pd_series_ext.get_series_data(S), 2
    )


@overload_method(SeriesType, "autocorr", inline="always", no_unliteral=True)
def overload_series_autocorr(S, lag=1):
    return lambda S, lag=1: bodo.libs.array_kernels.autocorr(
        bodo.hiframes.pd_series_ext.get_series_data(S), lag
    )


@overload_method(SeriesType, "median", inline="always", no_unliteral=True)
def overload_series_median(S, axis=None, skipna=True, level=None, numeric_only=None):
    unsupported_args = dict(level=level, numeric_only=numeric_only)
    arg_defaults = dict(level=None, numeric_only=None)
    check_unsupported_args("Series.median", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError("Series.median(): axis argument not supported")

    return lambda S, axis=None, skipna=True, level=None, numeric_only=None: bodo.libs.array_kernels.median(
        bodo.hiframes.pd_series_ext.get_series_data(S), skipna
    )


@overload_method(SeriesType, "head", inline="always", no_unliteral=True)
def overload_series_head(S, n=5):
    # n must be an integer for indexing.
    if not is_overload_int(n):
        raise BodoError("Series.head(): 'n' must be an Integer")

    def impl(S, n=5):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        new_data = bodo.allgatherv(arr[:n], False)
        new_index = bodo.allgatherv(index[:n], False)
        return bodo.hiframes.pd_series_ext.init_series(new_data, new_index, name)

    return impl


@overload_method(SeriesType, "tail", inline="always", no_unliteral=True)
def overload_series_tail(S, n=5):
    # n must be an integer for indexing.
    if not is_overload_int(n):
        raise BodoError("Series.tail(): 'n' must be an Integer")

    def impl(S, n=5):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        new_data = bodo.allgatherv(arr[-n:], False)
        new_index = bodo.allgatherv(index[-n:], False)
        return bodo.hiframes.pd_series_ext.init_series(new_data, new_index, name)

    return impl


@overload_method(SeriesType, "nlargest", inline="always", no_unliteral=True)
def overload_series_nlargest(S, n=5, keep="first"):
    # TODO: cache implementation
    # TODO: strings, categoricals
    # TODO: support and test keep semantics
    unsupported_args = dict(keep=keep)
    arg_defaults = dict(keep="first")
    check_unsupported_args("Series.nlargest", unsupported_args, arg_defaults)

    def impl(S, n=5, keep="first"):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        index_arr = bodo.utils.conversion.coerce_to_ndarray(index)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr, out_ind_arr = bodo.libs.array_kernels.nlargest(
            arr, index_arr, n, True, bodo.hiframes.series_kernels.gt_f
        )
        out_index = bodo.utils.conversion.convert_to_index(out_ind_arr)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, "nsmallest", inline="always", no_unliteral=True)
def overload_series_nsmallest(S, n=5, keep="first"):
    # TODO: cache implementation

    unsupported_args = dict(keep=keep)
    arg_defaults = dict(keep="first")
    check_unsupported_args("Series.nsmallest", unsupported_args, arg_defaults)

    def impl(S, n=5, keep="first"):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        index_arr = bodo.utils.conversion.coerce_to_ndarray(index)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr, out_ind_arr = bodo.libs.array_kernels.nlargest(
            arr, index_arr, n, False, bodo.hiframes.series_kernels.lt_f
        )
        out_index = bodo.utils.conversion.convert_to_index(out_ind_arr)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, "notnull", inline="always", no_unliteral=True)
@overload_method(SeriesType, "notna", inline="always", no_unliteral=True)
def overload_series_notna(S):
    # TODO: make sure this is fused and optimized properly
    return lambda S: S.isna() == False


@overload_method(SeriesType, "astype", no_unliteral=True)
def overload_series_astype(S, dtype, copy=True, errors="raise"):

    unsupported_args = dict(errors=errors)
    arg_defaults = dict(errors="raise")
    check_unsupported_args("Series.astype", unsupported_args, arg_defaults)

    # TODO: other data types like datetime, records/tuples
    def impl(S, dtype, copy=True, errors="raise"):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.utils.conversion.fix_arr_dtype(arr, dtype, copy)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "take", inline="always", no_unliteral=True)
def overload_series_take(S, indices, axis=0, is_copy=True):
    # TODO: Pandas accepts but ignores additional kwargs from Numpy
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.take.html

    unsupported_args = dict(axis=axis, is_copy=is_copy)
    arg_defaults = dict(axis=0, is_copy=True)
    check_unsupported_args("Series.take", unsupported_args, arg_defaults)

    # Pandas requirement: Indices must be array like with integers
    if not (is_iterable_type(indices) and isinstance(indices.dtype, types.Integer)):
        # TODO: Ensure is_iterable_type is consistent with valid inputs
        # to coerce_to_ndarray
        raise BodoError(
            f"Series.take() 'indices' must be an array-like and contain integers. Found type {indices}."
        )

    def impl(S, indices, axis=0, is_copy=True):  # pragma: no cover
        indices_t = bodo.utils.conversion.coerce_to_ndarray(indices)
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        return bodo.hiframes.pd_series_ext.init_series(
            arr[indices_t], index[indices_t], name
        )

    return impl


@overload_method(SeriesType, "argsort", inline="always", no_unliteral=True)
def overload_series_argsort(S, axis=0, kind="quicksort", order=None):
    # TODO: categorical, etc.
    # TODO: optimize the if path of known to be no NaNs (e.g. after fillna)

    unsupported_args = dict(axis=axis, kind=kind, order=order)
    arg_defaults = dict(axis=0, kind="quicksort", order=None)
    check_unsupported_args("Series.argsort", unsupported_args, arg_defaults)

    def impl(S, axis=0, kind="quicksort", order=None):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        n = len(arr)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        mask = S.notna().values
        if not mask.all():
            out_arr = np.full(n, -1, np.int64)
            out_arr[mask] = argsort(arr[mask])
        else:
            out_arr = argsort(arr)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "sort_values", inline="always", no_unliteral=True)
def overload_series_sort_values(
    S,
    axis=0,
    ascending=True,
    inplace=False,
    kind="quicksort",
    na_position="last",
    ignore_index=False,
    key=None,
):
    unsupported_args = dict(
        axis=axis,
        inplace=inplace,
        kind=kind,
        na_position=na_position,
        ignore_index=ignore_index,
        key=key,
    )
    arg_defaults = dict(
        axis=0,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    )
    check_unsupported_args("Series.sort_values", unsupported_args, arg_defaults)

    def impl(
        S,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe((arr,), index, ("A",))
        sorted_df = df.sort_values(
            ["A"], ascending=ascending, inplace=inplace, na_position=na_position
        )
        out_arr = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(sorted_df, 0)
        out_index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(sorted_df)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

    return impl


@overload_method(SeriesType, "value_counts", inline="always", no_unliteral=True)
def overload_series_value_counts(
    S,
    normalize=False,
    sort=True,
    ascending=False,
    bins=None,
    dropna=True,
):
    unsupported_args = dict(normalize=normalize, sort=sort, bins=bins, dropna=dropna)
    arg_defaults = dict(normalize=False, sort=True, bins=None, dropna=True)
    check_unsupported_args("Series.value_counts", unsupported_args, arg_defaults)

    # reusing aggregate/count
    # TODO(ehsan): write optimized implementation
    def impl(
        S,
        normalize=False,
        sort=True,
        ascending=False,
        bins=None,
        dropna=True,
    ):  # pragma: no cover
        # create a dummy dataframe to use groupby/count and sort_values
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        dummy_index = bodo.hiframes.pd_index_ext.init_range_index(0, len(arr), 1, None)
        in_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (arr, arr), dummy_index, ("A", "B")
        )
        count_df = in_df.groupby("A").count().sort_values("B", ascending=ascending)
        # create the output Series and remove "A"/"B" labels from index/column
        ind_arr = bodo.utils.conversion.coerce_to_array(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_index(count_df)
        )
        index = bodo.utils.conversion.index_from_array(ind_arr)
        count_arr = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(count_df, 0)
        return bodo.hiframes.pd_series_ext.init_series(count_arr, index, name)

    return impl


@overload_method(SeriesType, "groupby", inline="always", no_unliteral=True)
def overload_series_groupby(
    S,
    by=None,
    axis=0,
    level=None,
    as_index=True,
    sort=True,
    group_keys=True,
    squeeze=False,
    observed=False,
    dropna=True,
):
    unsupported_args = dict(
        axis=axis,
        group_keys=group_keys,
        squeeze=squeeze,
        observed=observed,
        dropna=dropna,
    )
    arg_defaults = dict(
        axis=0, group_keys=True, squeeze=False, observed=False, dropna=True
    )
    check_unsupported_args("Series.groupby", unsupported_args, arg_defaults)

    if not is_overload_true(as_index):  # pragma: no cover
        raise BodoError("as_index=False only valid with DataFrame")

    if is_overload_none(by) and is_overload_none(level):  # pragma: no cover
        raise BodoError("You have to supply one of 'by' and 'level'")

    if not is_overload_none(level):
        # NOTE: pandas seems to ignore the 'by' argument if level is provided

        # TODO: support MultiIndex case
        if not (
            is_overload_constant_int(level) and get_overload_const_int(level) == 0
        ) or isinstance(
            S.index, bodo.hiframes.pd_multi_index_ext.MultiIndexType
        ):  # pragma: no cover
            raise BodoError(
                "Series.groupby(): MultiIndex case or 'level' other than 0 not supported yet"
            )

        def impl_index(
            S,
            by=None,
            axis=0,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze=False,
            observed=False,
            dropna=True,
        ):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            keys = bodo.utils.conversion.coerce_to_array(index)
            # reuse DataFrame.groupby
            # Pandas assigns name=None to output Series/index, but not straightforward here.
            # we use empty/single-space to simplify
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (keys, arr), index, (" ", "")
            )
            return df.groupby(" ")[""]

        return impl_index

    if not is_overload_none(level):  # pragma: no cover
        raise BodoError(
            "Series.groupby(): 'level' argument should be None if 'by' is not None"
        )

    def impl(
        S,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True,
    ):  # pragma: no cover
        keys = bodo.utils.conversion.coerce_to_array(by)
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        # reuse DataFrame.groupby
        # Pandas assigns name=None to output Series/index, but not straightforward here.
        # we use empty/single-space to simplify
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (keys, arr), index, (" ", "")
        )
        return df.groupby(" ")[""]

    return impl


@overload_method(SeriesType, "append", inline="always", no_unliteral=True)
def overload_series_append(S, to_append, ignore_index=False, verify_integrity=False):
    unsupported_args = dict(verify_integrity=verify_integrity)
    arg_defaults = dict(verify_integrity=False)
    check_unsupported_args("Series.append", unsupported_args, arg_defaults)

    # call pd.concat()
    # single Series case
    if isinstance(to_append, SeriesType):
        return (
            lambda S, to_append, ignore_index=False, verify_integrity=False: pd.concat(
                (S, to_append),
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
            )
        )  # pragma: no cover

    # tuple case
    if isinstance(to_append, types.BaseTuple):
        return (
            lambda S, to_append, ignore_index=False, verify_integrity=False: pd.concat(
                (S,) + to_append,
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
            )
        )  # pragma: no cover

    # list/other cases
    return lambda S, to_append, ignore_index=False, verify_integrity=False: pd.concat(
        [S] + to_append, ignore_index=ignore_index, verify_integrity=verify_integrity
    )  # pragma: no cover


@overload_method(SeriesType, "isin", inline="always", no_unliteral=True)
def overload_series_isin(S, values):
    # if input is Series or array, special implementation is necessary since it may
    # require hash-based shuffling of both inputs for parallelization
    if bodo.utils.utils.is_array_typ(values):

        def impl_arr(S, values):  # pragma: no cover
            values_arr = bodo.utils.conversion.coerce_to_array(values)
            A = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            numba.parfors.parfor.init_prange()
            n = len(A)
            out_arr = np.empty(n, np.bool_)
            bodo.libs.array.array_isin(out_arr, A, values_arr, False)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl_arr

    # 'values' should be a set or list, TODO: support other list-likes such as Array
    if not isinstance(values, (types.Set, types.List)):
        raise BodoError("Series.isin(): 'values' parameter should be a set or a list")

    # TODO: use hash table for 'values' for faster check similar to Pandas
    def impl(S, values):  # pragma: no cover
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(A)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            # TODO: avoid Timestamp conversion for date comparisons if possible
            out_arr[i] = bodo.utils.conversion.box_if_dt64(A[i]) in values

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "quantile", inline="always", no_unliteral=True)
def overload_series_quantile(S, q=0.5, interpolation="linear"):

    unsupported_args = dict(interpolation=interpolation)
    arg_defaults = dict(interpolation="linear")
    check_unsupported_args("Series.quantile", unsupported_args, arg_defaults)

    # TODO: datetime support
    def impl(S, q=0.5, interpolation="linear"):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        return bodo.libs.array_kernels.quantile(arr, q)

    return impl


@overload_method(SeriesType, "nunique", inline="always", no_unliteral=True)
def overload_series_nunique(S, dropna=True):
    # TODO: refactor, support NA, dt64
    def impl(S, dropna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        return bodo.libs.array_kernels.nunique(arr)

    return impl


@overload_method(SeriesType, "unique", inline="always", no_unliteral=True)
def overload_series_unique(S):
    # TODO: refactor, support dt64
    def impl(S):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        return bodo.libs.array_kernels.unique(arr)

    return impl


@overload_method(SeriesType, "describe", inline="always", no_unliteral=True)
def overload_series_describe(
    S, percentiles=None, include=None, exclude=None, datetime_is_numeric=False
):

    unsupported_args = dict(
        percentiles=percentiles,
        include=include,
        exclude=exclude,
        datetime_is_numeric=datetime_is_numeric,
    )
    arg_defaults = dict(
        percentiles=None, include=None, exclude=None, datetime_is_numeric=False
    )
    check_unsupported_args("Series.describe", unsupported_args, arg_defaults)

    # TODO: support categorical, dt64, ...
    def impl(
        S, percentiles=None, include=None, exclude=None, datetime_is_numeric=False
    ):  # pragma: no cover
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        a_count = np.float64(S.count())
        a_min = np.float64(S.min())
        a_max = np.float64(S.max())
        a_mean = np.float64(S.mean())
        a_std = np.float64(S.std())
        q25 = S.quantile(0.25)
        q50 = S.quantile(0.5)
        q75 = S.quantile(0.75)
        return bodo.hiframes.pd_series_ext.init_series(
            np.array([a_count, a_mean, a_std, a_min, q25, q50, q75, a_max]),
            bodo.utils.conversion.convert_to_index(
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ),
            name,
        )

    return impl


def str_fillna_inplace_series_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    fill_arr = bodo.hiframes.pd_series_ext.get_series_data(value)
    n = len(in_arr)
    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
    for j in numba.parfors.parfor.internal_prange(n):
        s = in_arr[j]
        if bodo.libs.array_kernels.isna(in_arr, j) and not bodo.libs.array_kernels.isna(
            fill_arr, j
        ):
            s = fill_arr[j]
        out_arr[j] = s
    bodo.libs.str_arr_ext.move_str_arr_payload(in_arr, out_arr)
    return


# Since string arrays can't be changed, we have to create a new
# array and update the same Series variable
# TODO: handle string array reflection
# TODO: handle init_series() optimization guard for mutability
def str_fillna_inplace_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    val_len = bodo.libs.str_arr_ext.get_utf8_size(value)
    n = len(in_arr)
    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
    for j in numba.parfors.parfor.internal_prange(n):
        s = in_arr[j]
        if bodo.libs.array_kernels.isna(in_arr, j):
            s = value
        out_arr[j] = s
    bodo.libs.str_arr_ext.move_str_arr_payload(in_arr, out_arr)
    return


def fillna_inplace_series_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    fill_arr = bodo.hiframes.pd_series_ext.get_series_data(value)
    for i in numba.parfors.parfor.internal_prange(len(in_arr)):
        s = in_arr[i]
        if bodo.libs.array_kernels.isna(in_arr, i) and not bodo.libs.array_kernels.isna(
            fill_arr, i
        ):
            s = fill_arr[i]
        in_arr[i] = s


def fillna_inplace_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    for i in numba.parfors.parfor.internal_prange(len(in_arr)):
        s = in_arr[i]
        if bodo.libs.array_kernels.isna(in_arr, i):
            s = value
        in_arr[i] = s


def str_fillna_alloc_series_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    index = bodo.hiframes.pd_series_ext.get_series_index(S)
    name = bodo.hiframes.pd_series_ext.get_series_name(S)
    fill_arr = bodo.hiframes.pd_series_ext.get_series_data(value)
    n = len(in_arr)
    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
    # TODO: fix SSA for loop variables
    for j in numba.parfors.parfor.internal_prange(n):
        s = in_arr[j]
        if bodo.libs.array_kernels.isna(in_arr, j) and not bodo.libs.array_kernels.isna(
            fill_arr, j
        ):
            s = fill_arr[j]
        out_arr[j] = s
        if bodo.libs.array_kernels.isna(in_arr, j) and bodo.libs.array_kernels.isna(
            fill_arr, j
        ):
            bodo.libs.array_kernels.setna(out_arr, j)
    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)


# XXX: assuming indices are equivalent and alignment is not needed
def fillna_series_impl(
    S,
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
):  # pragma: no cover
    in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
    index = bodo.hiframes.pd_series_ext.get_series_index(S)
    name = bodo.hiframes.pd_series_ext.get_series_name(S)
    fill_arr = bodo.hiframes.pd_series_ext.get_series_data(value)
    n = len(in_arr)
    out_arr = bodo.utils.utils.alloc_type(n, in_arr.dtype, (-1,))
    for i in numba.parfors.parfor.internal_prange(n):
        s = in_arr[i]
        if bodo.libs.array_kernels.isna(in_arr, i) and not bodo.libs.array_kernels.isna(
            fill_arr, i
        ):
            s = fill_arr[i]
        out_arr[i] = s
    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)


@overload_method(SeriesType, "fillna", no_unliteral=True)
def overload_series_fillna(
    S, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
):
    unsupported_args = dict(method=method, limit=limit, downcast=downcast)
    arg_defaults = dict(method=None, limit=None, downcast=None)
    check_unsupported_args("Series.series_fillna", unsupported_args, arg_defaults)

    if not (is_overload_none(axis) or is_overload_zero(axis)):
        raise BodoError("Series.min(): axis argument not supported")

    # Pandas doesn't support fillna for non-scalar values as of 1.1.0
    # TODO(ehsan): revisit when supported in Pandas
    if is_var_size_item_array_type(S.data) and not S.dtype == bodo.string_type:
        raise BodoError(
            f"Series.fillna() with inplace=True not supported for {S.dtype} values yet"
        )

    if is_overload_true(inplace):
        if S.dtype == bodo.string_type:
            # optimization: just set null bit if fill is empty
            if is_overload_constant_str(value) and get_overload_const_str(value) == "":
                return lambda S, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None: bodo.libs.str_arr_ext.set_null_bits_to_value(
                    bodo.hiframes.pd_series_ext.get_series_data(S), -1
                )

            # value is a Series
            if isinstance(value, SeriesType):
                return str_fillna_inplace_series_impl

            return str_fillna_inplace_impl
        else:
            # value is a Series
            if isinstance(value, SeriesType):
                return fillna_inplace_series_impl

            return fillna_inplace_impl
    else:  # not inplace
        # value is a Series
        _dtype = S.data
        if isinstance(value, SeriesType):

            def fillna_series_impl(
                S,
                value=None,
                method=None,
                axis=None,
                inplace=False,
                limit=None,
                downcast=None,
            ):  # pragma: no cover
                in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                fill_arr = bodo.hiframes.pd_series_ext.get_series_data(value)
                n = len(in_arr)
                out_arr = bodo.utils.utils.alloc_type(n, _dtype, (-1,))
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        in_arr, i
                    ) and bodo.libs.array_kernels.isna(fill_arr, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue
                    if bodo.libs.array_kernels.isna(in_arr, i):
                        out_arr[i] = fill_arr[i]
                        continue
                    out_arr[i] = in_arr[i]
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return fillna_series_impl

        def fillna_impl(
            S,
            value=None,
            method=None,
            axis=None,
            inplace=False,
            limit=None,
            downcast=None,
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            n = len(in_arr)
            out_arr = bodo.utils.utils.alloc_type(n, _dtype, (-1,))
            for i in numba.parfors.parfor.internal_prange(n):
                s = in_arr[i]
                if bodo.libs.array_kernels.isna(in_arr, i):
                    s = value
                out_arr[i] = s
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return fillna_impl


@overload_method(SeriesType, "replace", inline="always", no_unliteral=True)
def overload_series_replace(
    S,
    to_replace=None,
    value=None,
    inplace=False,
    limit=None,
    regex=False,
    method="pad",
):

    unsupported_args = dict(inplace=inplace, limit=limit, regex=regex, method=method)
    merge_defaults = dict(inplace=False, limit=None, regex=False, method="pad")
    check_unsupported_args("Series.replace", unsupported_args, merge_defaults)

    ret_dtype = S.data
    if isinstance(ret_dtype, CategoricalArray):

        def cat_impl(
            S,
            to_replace=None,
            value=None,
            inplace=False,
            limit=None,
            regex=False,
            method="pad",
        ):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            return bodo.hiframes.pd_series_ext.init_series(
                in_arr.replace(to_replace, value), index, name
            )

        return cat_impl

    def impl(
        S,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):  # pragma: no cover

        in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        n = len(in_arr)
        out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, (-1,))
        replace_dict = build_replace_dict(to_replace, value)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(in_arr, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            s = in_arr[i]
            if s in replace_dict:
                s = replace_dict[s]
            out_arr[i] = s
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


# Helper function for creating the dictionary map[replace -> new value]
# For various data types.
def build_replace_dict(to_replace, value):
    # Dummy function used for overload
    pass


@overload(build_replace_dict)
def _build_replace_dict(to_replace, value):
    # TODO: replace with something that captures all scalars
    is_scalar_replace = isinstance(
        to_replace, (types.Number, Decimal128Type)
    ) or to_replace in [bodo.string_type, types.boolean]
    is_iterable_replace = is_iterable_type(to_replace)

    is_scalar_value = isinstance(value, (types.Number, Decimal128Type)) or value in [
        bodo.string_type,
        types.boolean,
    ]
    is_iterable_value = is_iterable_type(value)

    # Scalar, Scalar case
    if is_scalar_replace and is_scalar_value:

        def impl(to_replace, value):  # pragma: no cover
            replace_dict = {}
            replace_dict[to_replace] = value
            return replace_dict

        return impl

    # List, Scalar case
    if is_iterable_replace and is_scalar_value:

        def impl(to_replace, value):  # pragma: no cover
            replace_dict = {}
            for r in to_replace:
                replace_dict[r] = value
            return replace_dict

        return impl

    # List, List case
    if is_iterable_replace and is_iterable_value:

        def impl(to_replace, value):  # pragma: no cover
            replace_dict = {}
            assert len(to_replace) == len(
                value
            ), "To_replace and value lengths must be the same"
            for i in range(len(to_replace)):
                replace_dict[to_replace[i]] = value[i]
            return replace_dict

        return impl

    # Dictionary, None case
    # TODO(Nick): Add a check to ensure value type can be converted
    # to key type
    if isinstance(to_replace, numba.types.DictType) and is_overload_none(value):
        return lambda to_replace, value: to_replace  # pragma: no cover

    raise BodoError(
        "Series.replace(): Not supported for types to_replace={} and value={}".format(
            to_replace, value
        )
    )
    # List, List case


@overload_method(SeriesType, "explode", inline="always", no_unliteral=True)
def overload_series_explode(S, ignore_index=False):

    unsupported_args = dict(ignore_index=ignore_index)
    merge_defaults = dict(ignore_index=False)
    check_unsupported_args("Series.explode", unsupported_args, merge_defaults)

    if not isinstance(S.data, ArrayItemArrayType):
        # pandas copies input if not iterable
        return lambda S, ignore_index=False: S.copy()  # pragma: no cover

    def impl(S, ignore_index=False):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index_arr = bodo.utils.conversion.index_to_array(index)
        out_arr, out_index_arr = bodo.libs.array_kernels.explode(arr, index_arr)
        out_index = bodo.utils.conversion.index_from_array(out_index_arr)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

    return impl


@overload(np.digitize, inline="always", no_unliteral=True)
def overload_series_np_digitize(x, bins, right=False):
    # np.digitize() just uses underlying Series array and returns an output array
    if isinstance(x, SeriesType):

        def impl(x, bins, right=False):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(x)
            return np.digitize(arr, bins, right)

        return impl


@overload(np.argmax, inline="always", no_unliteral=True)
def argmax_overload(a, axis=None, out=None):
    if (
        isinstance(a, types.Array)
        and is_overload_constant_int(axis)
        and get_overload_const_int(axis) == 1
    ):

        def impl(a, axis=None, out=None):  # pragma: no cover
            argmax_arr = np.empty(len(a), a.dtype)
            numba.parfors.parfor.init_prange()
            n = len(a)
            for i in numba.parfors.parfor.internal_prange(n):
                argmax_arr[i] = np.argmax(a[i])
            return argmax_arr

        return impl


@overload(np.argmin, inline="always", no_unliteral=True)
def argmin_overload(a, axis=None, out=None):
    if (
        isinstance(a, types.Array)
        and is_overload_constant_int(axis)
        and get_overload_const_int(axis) == 1
    ):

        def impl(a, axis=None, out=None):  # pragma: no cover
            argmin_arr = np.empty(len(a), a.dtype)
            numba.parfors.parfor.init_prange()
            n = len(a)
            for i in numba.parfors.parfor.internal_prange(n):
                argmin_arr[i] = np.argmin(a[i])
            return argmin_arr

        return impl


@overload(np.dot, inline="always", no_unliteral=True)
@overload(operator.matmul, inline="always", no_unliteral=True)
def overload_series_np_dot(a, b, out=None):

    if (
        isinstance(a, SeriesType) or isinstance(b, SeriesType)
    ) and not is_overload_none(out):
        raise BodoError("np.dot(): 'out' parameter not supported yet")

    # just call np.dot on underlying arrays
    if isinstance(a, SeriesType):

        def impl(a, b, out=None):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(a)
            return np.dot(arr, b)

        return impl

    if isinstance(b, SeriesType):

        def impl(a, b, out=None):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(b)
            return np.dot(a, arr)

        return impl


@overload_method(SeriesType, "dropna", inline="always", no_unliteral=True)
def overload_series_dropna(S, axis=0, inplace=False):

    unsupported_args = dict(axis=axis, inplace=inplace)
    merge_defaults = dict(axis=0, inplace=False)
    check_unsupported_args("Series.dropna", unsupported_args, merge_defaults)

    if S.dtype == bodo.string_type:

        def dropna_str_impl(S, axis=0, inplace=False):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            mask = S.notna().values
            index_arr = bodo.utils.conversion.extract_index_array(S)
            out_index = bodo.utils.conversion.convert_to_index(index_arr[mask])
            out_arr = bodo.hiframes.series_kernels._series_dropna_str_alloc_impl_inner(
                in_arr
            )
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return dropna_str_impl
    else:

        def dropna_impl(S, axis=0, inplace=False):  # pragma: no cover
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            index_arr = bodo.utils.conversion.extract_index_array(S)
            mask = S.notna().values
            out_index = bodo.utils.conversion.convert_to_index(index_arr[mask])
            out_arr = in_arr[mask]
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return dropna_impl


@overload_method(SeriesType, "shift", inline="always", no_unliteral=True)
def overload_series_shift(S, periods=1, freq=None, axis=0, fill_value=None):
    # TODO: handle strings

    unsupported_args = dict(freq=freq, axis=axis, fill_value=fill_value)
    arg_defaults = dict(freq=None, axis=0, fill_value=None)
    check_unsupported_args("Series.shift", unsupported_args, arg_defaults)

    # Bodo specific limitations for supported types
    # Currently only float (not nullable), int (not nullable), and dt64 are supported
    if not (
        isinstance(S.data, types.Array)
        and (
            isinstance(S.data.dtype, (types.Number))
            or S.data.dtype == bodo.datetime64ns
        )
    ):
        # TODO: Link to supported Series input types.
        raise BodoError(
            f"Series.shift() Series input type {S.data.dtype} not supported."
        )

    # Ensure period is int
    if not is_overload_int(periods):
        raise BodoError("Series.shift(): 'periods' input must be an integer.")

    def impl(S, periods=1, freq=None, axis=0, fill_value=None):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.hiframes.rolling.shift(arr, periods, False)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "pct_change", inline="always", no_unliteral=True)
def overload_series_pct_change(S, periods=1, fill_method="pad", limit=None, freq=None):

    unsupported_args = dict(fill_method=fill_method, limit=limit, freq=freq)
    arg_defaults = dict(fill_method="pad", limit=None, freq=None)
    check_unsupported_args("Series.pct_change", unsupported_args, arg_defaults)

    # TODO: handle dt64, strings
    def impl(
        S, periods=1, fill_method="pad", limit=None, freq=None
    ):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.hiframes.rolling.pct_change(arr, periods, False)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "where", inline="always", no_unliteral=True)
def overload_series_where(
    S,
    cond,
    other=np.nan,
    inplace=False,
    axis=None,
    level=None,
    errors="raise",
    try_cast=False,
):
    """Overload Series.where. It replaces element with other if cond is False.
    It's the opposite of Series.mask
    """

    # Validate the inputs
    _validate_arguments_mask_where(
        "Series.where",
        S,
        cond,
        other,
        inplace,
        axis,
        level,
        errors,
        try_cast,
    )

    # TODO: handle other cases
    def impl(
        S,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.hiframes.series_impl.where_impl(cond, arr, other)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "mask", inline="always", no_unliteral=True)
def overload_series_mask(
    S,
    cond,
    other=np.nan,
    inplace=False,
    axis=None,
    level=None,
    errors="raise",
    try_cast=False,
):
    """Overload Series.mask. It replaces element with other if cond is True.
    It's the opposite of Series.where
    """

    # Validate the inputs
    _validate_arguments_mask_where(
        "Series.mask",
        S,
        cond,
        other,
        inplace,
        axis,
        level,
        errors,
        try_cast,
    )

    # TODO: handle other cases
    def impl(
        S,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):  # pragma: no cover
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.hiframes.series_impl.where_impl(~cond, arr, other)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


def _validate_arguments_mask_where(
    func_name,
    S,
    cond,
    other,
    inplace,
    axis,
    level,
    errors,
    try_cast,
):
    """Helper function to perform the necessary error checking for
    Series.where() and Series.mask()."""
    unsupported_args = dict(
        inplace=inplace, level=level, errors=errors, try_cast=try_cast
    )
    arg_defaults = dict(inplace=False, level=None, errors="raise", try_cast=False)
    check_unsupported_args(f"{func_name}", unsupported_args, arg_defaults)
    if not (is_overload_none(axis) or is_overload_zero(axis)):  # pragma: no cover
        raise BodoError(f"{func_name}(): axis argument not supported")

    # Bodo Limitation. Where/Mask is only supported for string arrays, categorical + scalar, and numpy arrays
    # Nullable int/bool arrays can be used, but they may have the wrong type or
    # drop NaN values.
    if not (
        isinstance(S.data, types.Array)
        or (
            bodo.utils.utils.is_array_typ(S.data, False) and S.dtype == bodo.string_type
        )
        # TODO: Support categorical of Timestamp/Timedelta
        or (
            isinstance(S.data, bodo.CategoricalArray)
            and S.dtype.elem_type
            not in [
                bodo.datetime64ns,
                bodo.timedelta64ns,
                bodo.pandas_timestamp_type,
                bodo.pd_timedelta_type,
            ]
        )
    ):
        raise BodoError(
            f"{func_name}() Series data with type {S.data} not yet supported"
        )

    # TODO: Support multidimensional arrays for cond + Dataframes
    # Check that cond is a supported array of booleans
    if not (
        isinstance(cond, (SeriesType, types.Array, BooleanArrayType))
        and cond.ndim == 1
        and cond.dtype == types.bool_
    ):
        raise BodoError(
            f"{func_name}() 'cond' argument must be a Series or 1-dim array of booleans"
        )

    # Bodo Restriction: Strict typing limits the type of 'other'
    # Check that other is an accepted value and that its type matches.
    # It can either be:
    # - a scalar of the "same" type as S (or np.nan)
    # - a Series or 1-dim Numpy array with the "same" type as S

    # Bodo Limitation. Where is only supported for string arrays and numpy arrays
    # Nullable int/bool arrays can be used, but they may have the wrong type or
    # drop NaN values.
    val_is_nan = is_overload_constant_nan(other)
    if not (
        # Handle actual np.nan value if other is omitted
        val_is_nan
        or is_scalar_type(other)
        or (isinstance(other, types.Array) and other.ndim == 1)
        or (
            isinstance(other, SeriesType)
            and (isinstance(S.data, types.Array) or S.dtype == bodo.string_type)
        )
    ):
        raise BodoError(
            f"{func_name}() 'other' must be a scalar, series, 1-dim numpy array or StringArray with a matching type for Series."
        )

    # Check that the types match
    if isinstance(S.dtype, bodo.PDCategoricalDtype):
        s_typ = S.dtype.elem_type
    else:
        s_typ = S.dtype

    if is_iterable_type(other):
        other_typ = other.dtype
    elif val_is_nan:
        other_typ = types.float64
    else:
        other_typ = types.unliteral(other)

    if not (is_common_scalar_dtype([s_typ, other_typ])):
        raise BodoError(f"{func_name}() series and 'other' must share a common type.")


############################ binary operators #############################


def create_explicit_binary_op_overload(op):
    def overload_series_explicit_binary_op(
        S, other, level=None, fill_value=None, axis=0
    ):
        unsupported_args = dict(level=level, axis=axis)
        arg_defaults = dict(level=None, axis=0)
        check_unsupported_args(
            "series.{}".format(op.__name__), unsupported_args, arg_defaults
        )

        is_str_scalar_other = other == string_type or is_overload_constant_str(other)
        is_str_iterable_other = is_iterable_type(other) and other.dtype == string_type
        is_legal_string_type = S.dtype == string_type and (
            (op == operator.add and (is_str_scalar_other or is_str_iterable_other))
            or (op == operator.mul and isinstance(other, types.Integer))
        )

        # TODO: Add pd.Timedelta
        is_series_timedelta = S.dtype == bodo.timedelta64ns
        is_series_datetime = S.dtype == bodo.datetime64ns
        is_other_timedelta_iter = is_iterable_type(other) and (
            other.dtype == datetime_timedelta_type or other.dtype == bodo.timedelta64ns
        )
        is_other_datetime_iter = is_iterable_type(other) and (
            other.dtype == datetime_datetime_type
            or other.dtype == pandas_timestamp_type
            or other.dtype == bodo.datetime64ns
        )

        is_legal_timedelta = (
            is_series_timedelta and (is_other_timedelta_iter or is_other_datetime_iter)
        ) or (is_series_datetime and is_other_timedelta_iter)
        is_legal_timedelta = is_legal_timedelta and op == operator.add

        # TODO: string array, datetimeindex/timedeltaindex
        if not (
            isinstance(S.dtype, types.Number)
            or is_legal_string_type
            or is_legal_timedelta
        ):  # pragma: no cover
            raise TypeError("Unsupported types for Series.{}".format(op.__name__))

        typing_context = numba.core.registry.cpu_target.typing_context
        # scalar case
        if isinstance(other, types.Number) or is_str_scalar_other:
            args = (S.data, other)
            ret_dtype = typing_context.resolve_function_type(op, args, {}).return_type
            # Pandas 1.0 returns nullable bool array for nullable int array
            if isinstance(S.data, IntegerArrayType) and ret_dtype == types.Array(
                types.bool_, 1, "C"
            ):
                ret_dtype = boolean_array

            def impl_scalar(
                S, other, level=None, fill_value=None, axis=0
            ):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                numba.parfors.parfor.init_prange()
                # other could be tuple, list, array, Index, or Series
                n = len(arr)
                out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, (-1,))
                for i in numba.parfors.parfor.internal_prange(n):
                    left_nan = bodo.libs.array_kernels.isna(arr, i)
                    if left_nan:
                        if fill_value is None:
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(fill_value, other)
                    else:
                        out_arr[i] = op(arr[i], other)

                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl_scalar

        args = (S.data, types.Array(other.dtype, 1, "C"))
        ret_dtype = typing_context.resolve_function_type(op, args, {}).return_type
        # Pandas 1.0 returns nullable bool array for nullable int array
        if isinstance(S.data, IntegerArrayType) and ret_dtype == types.Array(
            types.bool_, 1, "C"
        ):
            ret_dtype = boolean_array

        # Addition of NPDatetime("ns") and NPTimedelta("ns") seems to give
        # an incorrect ret_dtype for our always ns requirement,
        # so it fails in setitem
        if ret_dtype == types.Array(types.NPDatetime(""), 1, "C"):
            ret_dtype = types.Array(types.NPDatetime("ns"), 1, "C")

        def impl(S, other, level=None, fill_value=None, axis=0):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            # other could be tuple, list, array, Index, or Series
            other_arr = bodo.utils.conversion.coerce_to_array(other)
            numba.parfors.parfor.init_prange()
            n = len(arr)
            out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, (-1,))
            for i in numba.parfors.parfor.internal_prange(n):
                left_nan = bodo.libs.array_kernels.isna(arr, i)
                right_nan = bodo.libs.array_kernels.isna(other_arr, i)
                if left_nan and right_nan:
                    bodo.libs.array_kernels.setna(out_arr, i)
                elif left_nan:
                    if fill_value is None:
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(fill_value, other_arr[i])
                elif right_nan:
                    if fill_value is None:
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(arr[i], fill_value)
                else:
                    out_arr[i] = op(arr[i], other_arr[i])

            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl

    return overload_series_explicit_binary_op


# identical to the above overloads, except inputs to op() functions are reversed to
# support radd/rpow/...
# TODO: avoid code duplication
def create_explicit_binary_reverse_op_overload(op):
    def overload_series_explicit_binary_reverse_op(
        S, other, level=None, fill_value=None, axis=0
    ):
        if not is_overload_none(level):
            raise BodoError("level argument not supported")

        if not is_overload_zero(axis):
            raise BodoError("axis argument not supported")

        # TODO: string array, datetimeindex/timedeltaindex
        if not isinstance(S.dtype, types.Number):
            raise TypeError("only numeric values supported")

        typing_context = numba.core.registry.cpu_target.typing_context
        # scalar case
        if isinstance(other, types.Number):
            args = (other, S.data)
            ret_dtype = typing_context.resolve_function_type(op, args, {}).return_type
            # Pandas 1.0 returns nullable bool array for nullable int array
            if isinstance(S.data, IntegerArrayType) and ret_dtype == types.Array(
                types.bool_, 1, "C"
            ):
                ret_dtype = boolean_array

            def impl_scalar(
                S, other, level=None, fill_value=None, axis=0
            ):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                numba.parfors.parfor.init_prange()
                # other could be tuple, list, array, Index, or Series
                n = len(arr)
                out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, None)
                for i in numba.parfors.parfor.internal_prange(n):
                    left_nan = bodo.libs.array_kernels.isna(arr, i)
                    if left_nan:
                        if fill_value is None:
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(other, fill_value)
                    else:
                        out_arr[i] = op(other, arr[i])

                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl_scalar

        args = (types.Array(other.dtype, 1, "C"), S.data)
        ret_dtype = typing_context.resolve_function_type(op, args, {}).return_type
        # Pandas 1.0 returns nullable bool array for nullable int array
        if isinstance(S.data, IntegerArrayType) and ret_dtype == types.Array(
            types.bool_, 1, "C"
        ):
            ret_dtype = boolean_array

        def impl(S, other, level=None, fill_value=None, axis=0):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            # other could be tuple, list, array, Index, or Series
            other_arr = bodo.hiframes.pd_series_ext.get_series_data(other)
            numba.parfors.parfor.init_prange()
            n = len(arr)
            out_arr = bodo.utils.utils.alloc_type(n, ret_dtype, None)
            for i in numba.parfors.parfor.internal_prange(n):
                left_nan = bodo.libs.array_kernels.isna(arr, i)
                right_nan = bodo.libs.array_kernels.isna(other_arr, i)
                out_arr[i] = op(other_arr[i], arr[i])
                if left_nan and right_nan:
                    bodo.libs.array_kernels.setna(out_arr, i)
                elif left_nan:
                    if fill_value is None:
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(other_arr[i], fill_value)
                elif right_nan:
                    if fill_value is None:
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(fill_value, arr[i])
                else:
                    out_arr[i] = op(other_arr[i], arr[i])

            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl

    return overload_series_explicit_binary_reverse_op


explicit_binop_funcs_two_ways = {
    operator.add: {"add"},
    operator.sub: {"sub"},
    operator.mul: {"mul"},
    operator.truediv: {"div", "truediv"},
    operator.floordiv: {"floordiv"},
    operator.mod: {"mod"},
    operator.pow: {"pow"},
}

explicit_binop_funcs_single = {
    operator.lt: "lt",
    operator.gt: "gt",
    operator.le: "le",
    operator.ge: "ge",
    operator.ne: "ne",
    operator.eq: "eq",
}
explicit_binop_funcs = set()


def _install_explicit_binary_ops():
    for op, list_name in explicit_binop_funcs_two_ways.items():
        for name in list_name:
            overload_impl = create_explicit_binary_op_overload(op)
            overload_reverse_impl = create_explicit_binary_reverse_op_overload(op)
            r_name = "r" + name
            overload_method(SeriesType, name, no_unliteral=True)(overload_impl)
            overload_method(SeriesType, r_name, no_unliteral=True)(
                overload_reverse_impl
            )
            explicit_binop_funcs.add(name)
    for op, name in explicit_binop_funcs_single.items():
        overload_impl = create_explicit_binary_op_overload(op)
        overload_method(SeriesType, name, no_unliteral=True)(overload_impl)
        explicit_binop_funcs.add(name)


_install_explicit_binary_ops()


####################### binary operators ###############################


def create_binary_op_overload(op):
    def overload_series_binary_op(S, other):
        # sub for dt64 arrays fails in Numba, so we use our own function instead
        # TODO: fix it in Numba
        if (
            isinstance(S, SeriesType)
            and isinstance(other, SeriesType)
            and S.dtype == types.NPDatetime("ns")
            and other.dtype == types.NPDatetime("ns")
            and op == operator.sub
        ):

            def impl_dt64(S, other):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(other)
                out_arr = dt64_arr_sub(arr, other_arr)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl_dt64

        # Handle Offsets separation because addition is not defined on the array or scalar datetime64
        if (
            isinstance(S, SeriesType)
            and S.dtype == types.NPDatetime("ns")
            and is_offsets_type(other)
        ):

            def impl_offsets(S, other):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                numba.parfors.parfor.init_prange()
                n = len(S)
                out_arr = np.empty(n, np.dtype("datetime64[ns]"))
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(arr, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    timestamp_val = (
                        bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                            arr[i]
                        )
                    )
                    new_timestamp = op(timestamp_val, other)
                    out_arr[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        new_timestamp.value
                    )
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl_offsets

        if (
            op == operator.add
            and is_offsets_type(S)
            and isinstance(other, SeriesType)
            and other.dtype == types.NPDatetime("ns")
        ):

            def impl(S, other):  # pragma: no cover
                return op(other, S)

            return impl

        # left arg is Series
        if isinstance(S, SeriesType):

            def impl2(S, other):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(other)
                out_arr = op(arr, other_arr)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl2

        # right arg is Series
        if isinstance(other, SeriesType):

            def impl2(S, other):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(other)
                index = bodo.hiframes.pd_series_ext.get_series_index(other)
                name = bodo.hiframes.pd_series_ext.get_series_name(other)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(S)
                out_arr = op(other_arr, arr)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl2

    return overload_series_binary_op


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        overload_impl = create_binary_op_overload(op)
        # NOTE: cannot use inline="always". See test_pd_categorical
        overload(op, no_unliteral=True)(overload_impl)


_install_binary_ops()


# sub for dt64 arrays since it fails in Numba
def dt64_arr_sub(arg1, arg2):  # pragma: no cover
    return arg1 - arg2


@overload(dt64_arr_sub, no_unliteral=True)
def overload_dt64_arr_sub(arg1, arg2):
    assert arg1 == types.Array(types.NPDatetime("ns"), 1, "C") and arg2 == types.Array(
        types.NPDatetime("ns"), 1, "C"
    )
    td64_dtype = np.dtype("timedelta64[ns]")

    def impl(arg1, arg2):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        n = len(arg1)
        S = np.empty(n, td64_dtype)
        for i in numba.parfors.parfor.internal_prange(n):
            S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arg1[i])
                - bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arg2[i])
            )
        return S

    return impl


####################### binary inplace operators #############################


def create_inplace_binary_op_overload(op):
    def overload_series_inplace_binary_op(S, other):
        if isinstance(S, SeriesType) or isinstance(other, SeriesType):

            def impl(S, other):  # pragma: no cover
                arr = bodo.utils.conversion.get_array_if_series_or_index(S)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(other)
                op(arr, other_arr)
                return S

            return impl

    return overload_series_inplace_binary_op


def _install_inplace_binary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
        overload_impl = create_inplace_binary_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_inplace_binary_ops()


########################## unary operators ###############################


def create_unary_op_overload(op):
    def overload_series_unary_op(S):
        if isinstance(S, SeriesType):

            def impl(S):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                out_arr = op(arr)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

    return overload_series_unary_op


def _install_unary_ops():
    # install unary operators: ~, -, +
    for op in bodo.hiframes.pd_series_ext.series_unary_ops:
        overload_impl = create_unary_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_unary_ops()


####################### numpy ufuncs #########################


def create_ufunc_overload(ufunc):
    if ufunc.nin == 1:

        def overload_series_ufunc_nin_1(S):
            if isinstance(S, SeriesType):

                def impl(S):  # pragma: no cover
                    arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                    index = bodo.hiframes.pd_series_ext.get_series_index(S)
                    name = bodo.hiframes.pd_series_ext.get_series_name(S)
                    out_arr = ufunc(arr)
                    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

                return impl

        return overload_series_ufunc_nin_1
    elif ufunc.nin == 2:

        def overload_series_ufunc_nin_2(S1, S2):
            if isinstance(S1, SeriesType):

                def impl(S1, S2):  # pragma: no cover
                    arr = bodo.hiframes.pd_series_ext.get_series_data(S1)
                    index = bodo.hiframes.pd_series_ext.get_series_index(S1)
                    name = bodo.hiframes.pd_series_ext.get_series_name(S1)
                    other_arr = bodo.utils.conversion.get_array_if_series_or_index(S2)
                    out_arr = ufunc(arr, other_arr)
                    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

                return impl
            elif isinstance(S2, SeriesType):

                def impl(S1, S2):  # pragma: no cover
                    arr = bodo.utils.conversion.get_array_if_series_or_index(S1)
                    other_arr = bodo.hiframes.pd_series_ext.get_series_data(S2)
                    index = bodo.hiframes.pd_series_ext.get_series_index(S2)
                    name = bodo.hiframes.pd_series_ext.get_series_name(S2)
                    out_arr = ufunc(arr, other_arr)
                    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

                return impl

        return overload_series_ufunc_nin_2
    else:
        raise RuntimeError(
            "Don't know how to register ufuncs from ufunc_db with arity > 2"
        )


def _install_np_ufuncs():
    import numba.np.ufunc_db

    for ufunc in numba.np.ufunc_db.get_ufuncs():
        overload_impl = create_ufunc_overload(ufunc)
        overload(ufunc, no_unliteral=True)(overload_impl)


_install_np_ufuncs()


def argsort(A):  # pragma: no cover
    return np.argsort(A)


@overload(argsort, no_unliteral=True)
def overload_argsort(A):
    def impl(A):
        n = len(A)
        l_key_arrs = bodo.libs.str_arr_ext.to_string_list((A.copy(),))
        data = (np.arange(n),)
        bodo.libs.timsort.sort(l_key_arrs, 0, n, data)
        return data[0]

    return impl


@overload(pd.to_numeric, inline="always", no_unliteral=True)
def overload_to_numeric(arg_a, errors="raise", downcast=None):
    """pd.to_numeric() converts input to a numeric type determined dynamically, but we
    use the 'downcast' as type annotation (instead of downcasting the dynamic type).
    """
    # TODO: change 'arg_a' to 'arg' when inliner can handle it

    # check 'downcast' argument
    if not is_overload_none(downcast) and not (
        is_overload_constant_str(downcast)
        and get_overload_const_str(downcast)
        in ("integer", "signed", "unsigned", "float")
    ):  # pragma: no cover
        raise BodoError(
            "pd.to_numeric(): invalid downcasting method provided {}".format(downcast)
        )

    # find output dtype
    out_dtype = types.float64
    if not is_overload_none(downcast):
        downcast_str = get_overload_const_str(downcast)
        if downcast_str in ("integer", "signed"):
            out_dtype = types.int64
        elif downcast_str == "unsigned":
            out_dtype = types.uint64
        else:
            assert downcast_str == "float"

    # just return numeric array
    # TODO: handle dt64/td64 to int64 conversion
    if isinstance(arg_a, (types.Array, IntegerArrayType)):
        return lambda arg_a, errors="raise", downcast=None: arg_a.astype(out_dtype)

    # Series case
    if isinstance(arg_a, SeriesType):  # pragma: no cover

        def impl_series(arg_a, errors="raise", downcast=None):
            in_arr = bodo.hiframes.pd_series_ext.get_series_data(arg_a)
            index = bodo.hiframes.pd_series_ext.get_series_index(arg_a)
            name = bodo.hiframes.pd_series_ext.get_series_name(arg_a)
            out_arr = pd.to_numeric(in_arr, errors, downcast)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl_series

    # string array case
    # TODO: support tuple, list, scalar
    if arg_a != string_array_type:
        raise BodoError("pd.to_numeric(): invalid argument type {}".format(arg_a))

    if out_dtype == types.float64:

        def to_numeric_float_impl(
            arg_a, errors="raise", downcast=None
        ):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(arg_a)
            B = np.empty(n, np.float64)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(arg_a, i):
                    bodo.libs.array_kernels.setna(B, i)
                else:
                    bodo.libs.str_arr_ext.str_arr_item_to_numeric(B, i, arg_a, i)

            return B

        return to_numeric_float_impl
    else:

        def to_numeric_int_impl(
            arg_a, errors="raise", downcast=None
        ):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(arg_a)
            B = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(arg_a, i):
                    bodo.libs.array_kernels.setna(B, i)
                else:
                    bodo.libs.str_arr_ext.str_arr_item_to_numeric(B, i, arg_a, i)

            return B

        return to_numeric_int_impl


def series_filter_bool(arr, bool_arr):  # pragma: no cover
    return arr[bool_arr]


@infer_global(series_filter_bool)
class SeriesFilterBoolInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        ret = if_series_to_array_type(args[0])
        if isinstance(ret, types.Array) and isinstance(ret.dtype, types.Integer):
            ret = types.Array(types.float64, 1, "C")
        return ret(*args)


def where_impl_one_arg(c):  # pragma: no cover
    return np.where(c)


@overload(where_impl_one_arg, no_unliteral=True)
def overload_where_unsupported_one_arg(condition):
    if isinstance(condition, SeriesType) or bodo.utils.utils.is_array_typ(
        condition, False
    ):
        return lambda condition: np.where(condition)


@overload(where_impl_one_arg, inline="always", no_unliteral=True)
@overload(np.where, inline="always", no_unliteral=True)
def overload_np_where_one_arg(condition):
    if isinstance(condition, SeriesType):

        def impl_series(condition):  # pragma: no cover
            condition = bodo.hiframes.pd_series_ext.get_series_data(condition)
            return bodo.libs.array_kernels.nonzero(condition)

        return impl_series
    elif bodo.utils.utils.is_array_typ(condition, False):

        def impl(condition):  # pragma: no cover
            return bodo.libs.array_kernels.nonzero(condition)

        return impl


def where_impl(c, x, y):
    return np.where(c, x, y)


@overload(where_impl, no_unliteral=True)
def overload_where_unsupported(condition, x, y):
    if (
        not isinstance(condition, (SeriesType, types.Array, BooleanArrayType))
        or condition.ndim != 1
    ):
        return lambda condition, x, y: np.where(condition, x, y)  # pragma: no cover


@overload(where_impl, no_unliteral=True)
@overload(np.where, no_unliteral=True)
def overload_np_where(condition, x, y):
    """implement parallelizable np.where() for Series and 1D arrays"""
    # this overload only supports 1D arrays
    if (
        not isinstance(condition, (SeriesType, types.Array, BooleanArrayType))
        or condition.ndim != 1
    ):
        return

    assert condition.dtype == types.bool_, "invalid condition dtype"

    is_x_arr = bodo.utils.utils.is_array_typ(x, True)
    is_y_arr = bodo.utils.utils.is_array_typ(y, True)

    func_text = "def _impl(condition, x, y):\n"
    # get array data of Series inputs
    if isinstance(condition, SeriesType):
        func_text += (
            "  condition = bodo.hiframes.pd_series_ext.get_series_data(condition)\n"
        )
    if is_x_arr and not bodo.utils.utils.is_array_typ(x, False):
        func_text += "  x = bodo.utils.conversion.coerce_to_array(x)\n"
    if is_y_arr and not bodo.utils.utils.is_array_typ(y, False):
        func_text += "  y = bodo.utils.conversion.coerce_to_array(y)\n"
    func_text += "  n = len(condition)\n"

    x_dtype = x.dtype if is_x_arr else types.unliteral(x)
    y_dtype = y.dtype if is_y_arr else types.unliteral(y)

    if x_dtype == y_dtype:
        out_dtype = bodo.hiframes.pd_series_ext._get_series_array_type(x_dtype)
    # output is string if any input is string
    elif x_dtype == string_type or y_dtype == string_type:
        out_dtype = bodo.string_array_type
    # TODO: Support 2 categorical arrays
    # If the dtype is categorical, we need to use an actual
    # dtype from runtime.
    elif isinstance(x_dtype, bodo.PDCategoricalDtype):
        out_dtype = None
    # Support conversion between Timestamp/dt64 and Timedelta/td64.
    elif x_dtype in [bodo.timedelta64ns, bodo.datetime64ns]:
        out_dtype = types.Array(x_dtype, 1, "C")
    elif y_dtype in [bodo.timedelta64ns, bodo.datetime64ns]:
        out_dtype = types.Array(y_dtype, 1, "C")
    else:
        # similar to np.where typer of Numba
        out_dtype = numba.from_dtype(
            np.promote_types(
                numba.np.numpy_support.as_dtype(x_dtype),
                numba.np.numpy_support.as_dtype(y_dtype),
            )
        )
        out_dtype = types.Array(out_dtype, 1, "C")
    # If x_dtype is Categorical is_x_arr must be a true
    # (Categorical Array or Series)
    if isinstance(x_dtype, bodo.PDCategoricalDtype):
        arr_typ_ref = "x"
    else:
        arr_typ_ref = "out_dtype"
    func_text += f"  out_arr = bodo.utils.utils.alloc_type(n, {arr_typ_ref}, (-1,))\n"
    # Optimization for Categorical data that only transfers the codes directly.
    # This works because we know the input and output categories match.
    # If x_dtype is Categorical is_x_arr must be a true
    # (Categorical Array or Series)
    if isinstance(x_dtype, bodo.PDCategoricalDtype):
        func_text += "  out_codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(out_arr)\n"
        func_text += "  x_codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(x)\n"
    func_text += "  for j in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "    if condition[j]:\n"
    if is_x_arr:
        func_text += "      if bodo.libs.array_kernels.isna(x, j):\n"
        func_text += "        setna(out_arr, j)\n"
        func_text += "        continue\n"
    # Optimization for Categorical data that only transfers the codes directly.
    # This works because we know the input and output categories match.
    # If x_dtype is Categorical is_x_arr must be a true
    # (Categorical Array or Series)
    if isinstance(x_dtype, bodo.PDCategoricalDtype):
        func_text += "      out_codes[j] = x_codes[j]\n"
    else:
        func_text += (
            "      out_arr[j] = bodo.utils.conversion.unbox_if_timestamp({})\n".format(
                "x[j]" if is_x_arr else "x"
            )
        )
    func_text += "    else:\n"
    if is_y_arr:
        func_text += "      if bodo.libs.array_kernels.isna(y, j):\n"
        func_text += "        setna(out_arr, j)\n"
        func_text += "        continue\n"
    func_text += (
        "      out_arr[j] = bodo.utils.conversion.unbox_if_timestamp({})\n".format(
            "y[j]" if is_y_arr else "y"
        )
    )
    func_text += "  return out_arr\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "setna": bodo.libs.array_kernels.setna,
            "np": np,
            "out_dtype": out_dtype,
        },
        loc_vars,
    )
    _impl = loc_vars["_impl"]
    return _impl


@overload_method(SeriesType, "drop_duplicates", inline="always", no_unliteral=True)
def overload_series_drop_duplicates(S, subset=None, keep="first", inplace=False):
    # TODO: support inplace
    if not is_overload_none(subset):
        raise BodoError("drop_duplicates() subset argument not supported yet")

    if not is_overload_false(inplace):
        raise BodoError("drop_duplicates() inplace argument not supported yet")

    # XXX: can't reuse duplicated() here since it shuffles data and chunks
    # may not match

    def impl(S, subset=None, keep="first", inplace=False):  # pragma: no cover
        data_0 = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.utils.conversion.index_to_array(
            bodo.hiframes.pd_series_ext.get_series_index(S)
        )
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        (data_0,), index_arr = bodo.libs.array_kernels.drop_duplicates((data_0,), index)
        index = bodo.utils.conversion.index_from_array(index_arr)
        return bodo.hiframes.pd_series_ext.init_series(data_0, index, name)

    return impl


@overload_method(SeriesType, "between", inline="always", no_unliteral=True)
def overload_series_between(S, left, right, inclusive=True):
    def impl(S, left, right, inclusive=True):  # pragma: no cover
        # get series data
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            val = bodo.utils.conversion.box_if_dt64(arr[i])
            if inclusive:
                out_arr[i] = val <= right and val >= left
            else:
                out_arr[i] = val < right and val > left

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesType, "repeat", inline="always", no_unliteral=True)
def overload_series_repeat(S, repeats, axis=None):

    unsupported_args = dict(axis=axis)
    arg_defaults = dict(axis=None)
    check_unsupported_args("Series.repeat", unsupported_args, arg_defaults)

    # repeats can be int or array of int
    if not (
        isinstance(repeats, types.Integer)
        or (is_iterable_type(repeats) and isinstance(repeats.dtype, types.Integer))
    ):  # pragma: no cover
        raise BodoError(
            "Series.repeat(): 'repeats' should be an integer or array of integers"
        )

    # int case
    if isinstance(repeats, types.Integer):

        def impl_int(S, repeats, axis=None):  # pragma: no cover
            # get series data
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            index_arr = bodo.utils.conversion.index_to_array(index)

            out_arr = bodo.libs.array_kernels.repeat_kernel(arr, repeats)
            out_index_arr = bodo.libs.array_kernels.repeat_kernel(index_arr, repeats)
            out_index = bodo.utils.conversion.index_from_array(out_index_arr)

            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return impl_int

    # array case
    # TODO(ehsan): refactor to avoid code duplication (only diff is coerce_to_array)
    def impl_arr(S, repeats, axis=None):  # pragma: no cover
        # get series data
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index_arr = bodo.utils.conversion.index_to_array(index)
        repeats = bodo.utils.conversion.coerce_to_array(repeats)

        out_arr = bodo.libs.array_kernels.repeat_kernel(arr, repeats)
        out_index_arr = bodo.libs.array_kernels.repeat_kernel(index_arr, repeats)
        out_index = bodo.utils.conversion.index_from_array(out_index_arr)

        return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

    return impl_arr


@overload_method(SeriesType, "to_dict", inline="always", no_unliteral=True)
def overload_to_dict(S, into=None):
    """ Support Series.to_dict(). """

    def impl(S, into=None):  # pragma: no cover
        # default case, use a regular dict:
        data = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.utils.conversion.index_to_array(
            bodo.hiframes.pd_series_ext.get_series_index(S)
        )
        n = len(data)
        dico = {}
        for i in range(n):
            val = bodo.utils.conversion.box_if_dt64(data[i])
            dico[index[i]] = val
        return dico

        # TODO: support other types of dictionaries for the 'into' arg

    return impl
