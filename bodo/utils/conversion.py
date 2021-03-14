# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Utility functions for conversion of data such as list to array.
Need to be inlined for better optimization.
"""
import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.utils.indexing import add_nested_counts, init_nested_counts
from bodo.utils.typing import (
    BodoError,
    get_overload_const_list,
    get_overload_const_str,
    is_heterogeneous_tuple_type,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    is_overload_true,
)

NS_DTYPE = np.dtype("M8[ns]")  # similar pandas/_libs/tslibs/conversion.pyx
TD_DTYPE = np.dtype("m8[ns]")


# TODO: use generated_jit with IR inlining
def coerce_to_ndarray(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):  # pragma: no cover
    return data


@overload(coerce_to_ndarray, no_unliteral=True)
def overload_coerce_to_ndarray(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):
    # TODO: other cases handled by this function in Pandas like scalar
    """
    Coerces data to ndarray. Data should be numeric.
    """
    from bodo.hiframes.pd_index_ext import (
        DatetimeIndexType,
        NumericIndexType,
        RangeIndexType,
        TimedeltaIndexType,
    )
    from bodo.hiframes.pd_series_ext import SeriesType

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    data = types.unliteral(data)

    # TODO: handle NAs?
    # nullable int array
    if isinstance(
        data, bodo.libs.int_arr_ext.IntegerArrayType
    ) and not is_overload_none(use_nullable_array):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.int_arr_ext.get_int_arr_data(
            data
        )  # pragma: no cover

    # nullable boolean array
    if data == bodo.libs.bool_arr_ext.boolean_array and not is_overload_none(
        use_nullable_array
    ):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.bool_arr_ext.get_bool_arr_data(
            data
        )  # pragma: no cover

    # numpy array
    if isinstance(data, types.Array):
        if not is_overload_none(use_nullable_array) and isinstance(
            data.dtype, (types.Boolean, types.Integer)
        ):
            if data.dtype == types.bool_:
                if data.layout != "C":
                    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.bool_arr_ext.init_bool_array(
                        np.ascontiguousarray(data),
                        np.full((len(data) + 7) >> 3, 255, np.uint8),
                    )  # pragma: no cover
                else:
                    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.bool_arr_ext.init_bool_array(
                        data, np.full((len(data) + 7) >> 3, 255, np.uint8)
                    )  # pragma: no cover
            else:  # Integer case
                if data.layout != "C":
                    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.int_arr_ext.init_integer_array(
                        np.ascontiguousarray(data),
                        np.full((len(data) + 7) >> 3, 255, np.uint8),
                    )  # pragma: no cover
                else:
                    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.int_arr_ext.init_integer_array(
                        data, np.full((len(data) + 7) >> 3, 255, np.uint8)
                    )  # pragma: no cover
        if data.layout != "C":
            return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: np.ascontiguousarray(
                data
            )  # pragma: no cover
        return (
            lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
        )  # pragma: no cover

    # list/UniTuple
    if isinstance(data, (types.List, types.UniTuple)):

        if not is_overload_none(use_nullable_array) and isinstance(
            data.dtype, (types.Boolean, types.Integer)
        ):
            if data.dtype == types.bool_:
                return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.bool_arr_ext.init_bool_array(
                    np.asarray(data), np.full((len(data) + 7) >> 3, 255, np.uint8)
                )  # pragma: no cover
            else:  # Integer case
                return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.int_arr_ext.init_integer_array(
                    np.asarray(data), np.full((len(data) + 7) >> 3, 255, np.uint8)
                )  # pragma: no cover
        # convert Timestamp() back to dt64
        if data.dtype == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:

            def impl(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                vals = []
                for d in data:
                    vals.append(bodo.hiframes.pd_timestamp_ext.integer_to_dt64(d.value))
                return np.asarray(vals)

            return impl

        if isinstance(data.dtype, Decimal128Type):
            precision = data.dtype.precision
            scale = data.dtype.scale

            def impl(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = len(data)
                A = bodo.libs.decimal_arr_ext.alloc_decimal_array(n, precision, scale)
                for i, d in enumerate(data):
                    A._data[i] = bodo.libs.decimal_arr_ext.decimal128type_to_int128(d)
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

                return A

            return impl

        if data.dtype == bodo.hiframes.datetime_date_ext.datetime_date_type:

            def impl(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = len(data)
                A = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
                for i, d in enumerate(data):
                    A[i] = d
                return A

            return impl

        if data.dtype == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:

            def impl(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = len(data)
                A = bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(
                    n
                )
                for i, d in enumerate(data):
                    A[i] = d
                return A

            return impl

        if not is_overload_none(use_nullable_array) and data.dtype == types.bool_:
            return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.bool_arr_ext.init_bool_array(
                np.asarray(data), np.full((len(data) + 7) >> 3, 255, np.uint8)
            )
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: np.asarray(
            data
        )  # pragma: no cover

    # series
    if isinstance(data, SeriesType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )  # pragma: no cover

    # index types
    if isinstance(data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_index_ext.get_index_data(
            data
        )  # pragma: no cover

    # RangeIndex
    if isinstance(data, RangeIndexType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: np.arange(
            data._start, data._stop, data._step
        )  # pragma: no cover

    # types.RangeType
    if isinstance(data, types.RangeType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: np.arange(
            data.start, data.stop, data.step
        )  # pragma: no cover

    # convert scalar to ndarray
    # TODO: make sure scalar is a Numpy dtype

    if not is_overload_none(scalar_to_arr_len):

        if isinstance(data, Decimal128Type):
            precision = data.precision
            scale = data.scale

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = bodo.libs.decimal_arr_ext.alloc_decimal_array(n, precision, scale)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = data
                return A

            return impl_ts

        if data == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type:
            dt64_dtype = np.dtype("datetime64[ns]")

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = np.empty(n, dt64_dtype)
                v = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(data)
                v_ret = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(v)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = v_ret
                return A

            return impl_ts

        if data == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
            timedelta64_dtype = np.dtype("timedelta64[ns]")

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = np.empty(n, timedelta64_dtype)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    data
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = td64
                return A

            return impl_ts

        if data == bodo.hiframes.datetime_date_ext.datetime_date_type:

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = data
                return A

            return impl_ts

        # Timestamp values are stored as dt64 arrays
        if data == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:
            dt64_dtype = np.dtype("datetime64[ns]")

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = np.empty(scalar_to_arr_len, dt64_dtype)
                v = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(data.value)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = v
                return A

            return impl_ts

        dtype = types.unliteral(data)

        if not is_overload_none(use_nullable_array) and isinstance(
            dtype, types.Integer
        ):

            def impl_null_integer(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = scalar_to_arr_len
                out_arr = bodo.libs.int_arr_ext.alloc_int_array(n, dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    out_arr[i] = data
                return out_arr

            return impl_null_integer

        if not is_overload_none(use_nullable_array) and dtype == types.bool_:

            def impl_null_bool(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = scalar_to_arr_len
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    out_arr[i] = data
                return out_arr

            return impl_null_bool

        def impl_num(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            # TODO: parallelize np.full in PA
            # return np.full(scalar_to_arr_len, data)
            numba.parfors.parfor.init_prange()
            n = scalar_to_arr_len
            out_arr = np.empty(n, dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = data
            return out_arr

        return impl_num

    # data is already an array
    if bodo.utils.utils.is_array_typ(data, False):
        return (
            lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
        )  # pragma: no cover

    if is_overload_true(error_on_nonarray):
        raise BodoError("cannot coerce {} to array".format(data))

    return (
        lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
    )  # pragma: no cover


# TODO: use generated_jit with IR inlining
def coerce_to_array(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):  # pragma: no cover
    return data


@overload(coerce_to_array, no_unliteral=True)
def overload_coerce_to_array(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):
    """
    convert data to Bodo arrays.
    use_nullable_array=True returns nullable boolean/int arrays instead of Numpy arrays.
    """
    # TODO: support other arrays like list(str), datetime.date ...
    from bodo.hiframes.pd_index_ext import StringIndexType
    from bodo.hiframes.pd_series_ext import SeriesType

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    data = types.unliteral(data)

    # series
    if isinstance(data, SeriesType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )  # pragma: no cover

    # string Index
    if isinstance(data, StringIndexType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_index_ext.get_index_data(
            data
        )  # pragma: no cover

    # string array
    if data == bodo.string_array_type:
        return (
            lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
        )  # pragma: no cover

    # string list
    if isinstance(data, types.List) and data.dtype == bodo.string_type:
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.str_arr_ext.str_arr_from_sequence(
            data
        )  # pragma: no cover

    # string tuple
    if (
        isinstance(data, types.UniTuple)
        and isinstance(data.dtype, (types.UnicodeType, types.StringLiteral))
    ) or (
        isinstance(data, types.BaseTuple)
        and all(isinstance(t, types.StringLiteral) for t in data.types)
    ):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.str_arr_ext.str_arr_from_sequence(
            data
        )  # pragma: no cover

    if data in (
        bodo.libs.bool_arr_ext.boolean_array,
        bodo.hiframes.datetime_date_ext.datetime_date_array_type,
        bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_array_type,
        bodo.hiframes.split_impl.string_array_split_view_type,
    ) or isinstance(
        data,
        (
            bodo.libs.int_arr_ext.IntegerArrayType,
            DecimalArrayType,
            bodo.libs.tuple_arr_ext.TupleArrayType,
            bodo.libs.struct_arr_ext.StructArrayType,
            bodo.hiframes.pd_categorical_ext.CategoricalArray,
            bodo.libs.csr_matrix_ext.CSRMatrixType,
        ),
    ):
        return (
            lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
        )  # pragma: no cover

    # list/tuple of tuples
    if isinstance(data, (types.List, types.UniTuple)) and isinstance(
        data.dtype, types.BaseTuple
    ):
        # TODO: support variable length data (e.g strings) in tuples
        data_types = tuple(
            bodo.hiframes.pd_series_ext._get_series_array_type(t)
            for t in data.dtype.types
        )

        def impl_tuple_list(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = len(data)
            arr = bodo.libs.tuple_arr_ext.pre_alloc_tuple_array(n, (-1,), data_types)
            for i in range(n):
                arr[i] = data[i]
            return arr

        return impl_tuple_list

    # list(list/array) to array(array)
    if isinstance(data, types.List) and (
        bodo.utils.utils.is_array_typ(data.dtype, False)
        or isinstance(data.dtype, types.List)
    ):
        data_arr_type = bodo.hiframes.pd_series_ext._get_series_array_type(
            data.dtype.dtype
        )

        def impl_array_item_arr(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = len(data)
            nested_counts = init_nested_counts(data_arr_type)
            for i in range(n):
                arr_item = bodo.utils.conversion.coerce_to_array(
                    data[i], use_nullable_array=True
                )
                nested_counts = add_nested_counts(nested_counts, arr_item)

            out_arr = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
                n, nested_counts, data_arr_type
            )
            out_null_bitmap = bodo.libs.array_item_arr_ext.get_null_bitmap(out_arr)

            # write output
            for ii in range(n):
                arr_item = bodo.utils.conversion.coerce_to_array(
                    data[ii], use_nullable_array=True
                )
                out_arr[ii] = arr_item
                # set NA
                bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, ii, 1)

            return out_arr

        return impl_array_item_arr

    # string scalars to array
    if not is_overload_none(scalar_to_arr_len) and isinstance(
        data, (types.UnicodeType, types.StringLiteral)
    ):

        def impl_str(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = scalar_to_arr_len
            n_chars = n * bodo.libs.str_arr_ext.get_utf8_size(data)
            A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n_chars)
            for i in numba.parfors.parfor.internal_prange(n):
                A[i] = data
            return A

        return impl_str

    # Convert list of Timestamps to dt64 array
    if isinstance(data, types.List) and data.dtype == bodo.pandas_timestamp_type:

        def impl_list_timestamp(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = len(data)
            A = np.empty(n, np.dtype("datetime64[ns]"))
            for i in range(n):
                A[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(data[i].value)
            return A

        return impl_list_timestamp

    # Convert list of Timedeltas to td64 array
    if isinstance(data, types.List) and data.dtype == bodo.pd_timedelta_type:

        def impl_list_timedelta(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = len(data)
            A = np.empty(n, np.dtype("timedelta64[ns]"))
            for i in range(n):
                A[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    data[i].value
                )
            return A

        return impl_list_timedelta

    # assuming can be ndarray
    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.utils.conversion.coerce_to_ndarray(
        data, error_on_nonarray, use_nullable_array, scalar_to_arr_len
    )  # pragma: no cover


def _is_str_dtype(dtype):
    """return True if 'dtype' specifies a string data type."""
    return (isinstance(dtype, types.Function) and dtype.key[0] == str) or (
        is_overload_constant_str(dtype) and get_overload_const_str(dtype) == "str"
    )


# TODO: use generated_jit with IR inlining
def fix_arr_dtype(data, new_dtype, copy=None, nan_to_str=True):  # pragma: no cover
    return data


@overload(fix_arr_dtype, no_unliteral=True)
def overload_fix_arr_dtype(data, new_dtype, copy=None, nan_to_str=True):
    """convert data to new_dtype, copy if copy parameter is not None.
    'nan_to_str' specifies string conversion for np.nan values: write as 'nan'
    or actual NA (Pandas has inconsistent behavior in APIs)"""
    # TODO: support copy=True and copy=False when literals are passed reliably
    do_copy = not is_overload_none(copy)

    if is_overload_none(new_dtype):
        if do_copy:
            return (
                lambda data, new_dtype, copy=None, nan_to_str=True: data.copy()
            )  # pragma: no cover
        return (
            lambda data, new_dtype, copy=None, nan_to_str=True: data
        )  # pragma: no cover

    # convert to string
    if _is_str_dtype(new_dtype):

        # special optimized case for int to string conversion, uses inplace write to
        # string array to avoid extra allocation
        if isinstance(data.dtype, types.Integer):

            def impl_int_str(
                data, new_dtype, copy=None, nan_to_str=True
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(data)
                A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
                for j in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, j):
                        if nan_to_str:
                            bodo.libs.str_arr_ext.str_arr_setitem_NA_str(A, j)
                        else:
                            bodo.libs.array_kernels.setna(A, j)
                    else:
                        bodo.libs.str_arr_ext.str_arr_setitem_int_to_str(A, j, data[j])

                return A

            return impl_int_str

        def impl_str(data, new_dtype, copy=None, nan_to_str=True):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(data)
            A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
            for j in numba.parfors.parfor.internal_prange(n):

                if bodo.libs.array_kernels.isna(data, j):
                    if nan_to_str:
                        A[j] = "nan"
                    else:
                        bodo.libs.array_kernels.setna(A, j)
                    continue

                A[j] = str(data[j])

            return A

        return impl_str

    # convert to Categorical with predefined CategoricalDtype
    if isinstance(new_dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):

        def impl_cat_dtype(
            data, new_dtype, copy=None, nan_to_str=True
        ):  # pragma: no cover
            n = len(data)
            numba.parfors.parfor.init_prange()
            label_dict = (
                bodo.hiframes.pd_categorical_ext.get_label_dict_from_categories(
                    new_dtype.categories.values
                )
            )
            A = bodo.hiframes.pd_categorical_ext.alloc_categorical_array(n, new_dtype)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(A, i)
                    continue
                val = data[i]
                if val not in label_dict:
                    bodo.libs.array_kernels.setna(A, i)
                    continue
                codes[i] = label_dict[val]
            return A

        return impl_cat_dtype

    if (
        is_overload_constant_str(new_dtype)
        and get_overload_const_str(new_dtype) == "category"
    ):
        # find categorical dtype from data first and reuse the explicit dtype impl
        def impl_category(
            data, new_dtype, copy=None, nan_to_str=True
        ):  # pragma: no cover
            # find categories in data
            cats = bodo.libs.array_kernels.unique(data)
            # make sure categories are replicated since dtype is replicated
            cats = bodo.allgatherv(cats, False)
            # sort categories to match Pandas
            # TODO(ehsan): refactor to avoid long compilation time (too much inlining)
            cats = pd.Series(cats).dropna().sort_values().values
            cat_dtype = bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                bodo.utils.conversion.index_from_array(cats, None), False
            )
            return bodo.utils.conversion.fix_arr_dtype(data, cat_dtype, copy)

        return impl_category

    nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)

    # nullable int array case
    if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
        _dtype = nb_dtype.dtype
        if isinstance(data.dtype, types.Float):

            def impl_float(
                data, new_dtype, copy=None, nan_to_str=True
            ):  # pragma: no cover
                n = len(data)
                numba.parfors.parfor.init_prange()
                B = bodo.libs.int_arr_ext.alloc_int_array(n, _dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, i):
                        bodo.libs.array_kernels.setna(B, i)
                    else:
                        B[i] = int(data[i])
                        # no need for setting null bit since done by int arr's setitem
                return B

            return impl_float
        else:

            def impl(data, new_dtype, copy=None, nan_to_str=True):  # pragma: no cover
                n = len(data)
                n_bytes = (n + 7) >> 3
                bitmap = np.empty(n_bytes, np.uint8)
                for i in numba.parfors.parfor.internal_prange(n):
                    # TODO: use simple set_bit
                    bodo.libs.int_arr_ext.set_bit_to_arr(bitmap, i, 1)
                return bodo.libs.int_arr_ext.init_integer_array(
                    data.astype(_dtype), bitmap
                )

            return impl

    # Array case
    if do_copy or data.dtype != nb_dtype:
        return lambda data, new_dtype, copy=None, nan_to_str=True: data.astype(
            nb_dtype
        )  # pragma: no cover

    return lambda data, new_dtype, copy=None, nan_to_str=True: data  # pragma: no cover


def dtype_to_array_type(dtype):
    return bodo.hiframes.pd_series_ext._get_series_array_type(
        bodo.utils.typing.parse_dtype(dtype)
    )


@overload(dtype_to_array_type)
def overload_dtype_to_array_type(dtype):
    """parse dtype and return corresponding array type TypeRef"""
    arr_type = bodo.hiframes.pd_series_ext._get_series_array_type(
        bodo.utils.typing.parse_dtype(dtype)
    )
    return lambda dtype: arr_type


@numba.jit
def flatten_array(A):  # pragma: no cover
    flat_list = []
    n = len(A)
    for i in range(n):
        l = A[i]
        for s in l:
            flat_list.append(s)

    return bodo.utils.conversion.coerce_to_array(flat_list)


# TODO: use generated_jit with IR inlining
def parse_datetimes_from_strings(data):  # pragma: no cover
    return data


@overload(parse_datetimes_from_strings, no_unliteral=True)
def overload_parse_datetimes_from_strings(data):
    assert data == bodo.string_array_type

    def parse_impl(data):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        n = len(data)
        S = np.empty(n, bodo.utils.conversion.NS_DTYPE)
        for i in numba.parfors.parfor.internal_prange(n):
            S[i] = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(data[i])
        return S

    return parse_impl


# TODO: use generated_jit with IR inlining
def convert_to_dt64ns(data):  # pragma: no cover
    return data


@overload(convert_to_dt64ns, no_unliteral=True)
def overload_convert_to_dt64ns(data):
    """Converts data formats like int64 and arrays of strings to dt64ns"""
    # see pd.core.arrays.datetimes.sequence_to_dt64ns for constructor types
    # TODO: support dayfirst, yearfirst, tz
    if data == bodo.hiframes.datetime_date_ext.datetime_date_array_type:
        return (
            lambda data: bodo.hiframes.pd_timestamp_ext.datetime_date_arr_to_dt64_arr(
                data
            )  # pragma: no cover
        )

    if data == types.Array(types.int64, 1, "C"):
        return lambda data: data.view(
            bodo.utils.conversion.NS_DTYPE
        )  # pragma: no cover

    if data == types.Array(types.NPDatetime("ns"), 1, "C"):
        return lambda data: data  # pragma: no cover

    if data == bodo.string_array_type:
        return lambda data: bodo.utils.conversion.parse_datetimes_from_strings(
            data
        )  # pragma: no cover

    raise TypeError("invalid data type {} for dt64 conversion".format(data))


# TODO: use generated_jit with IR inlining
def convert_to_td64ns(data):  # pragma: no cover
    return data


@overload(convert_to_td64ns, no_unliteral=True)
def overload_convert_to_td64ns(data):
    """Converts data formats like int64 to timedelta64ns"""
    # TODO: array of strings
    # see pd.core.arrays.timedeltas.sequence_to_td64ns for constructor types
    # TODO: support datetime.timedelta
    if data == types.Array(types.int64, 1, "C"):
        return lambda data: data.view(bodo.utils.conversion.TD_DTYPE)

    if data == types.Array(types.NPTimedelta("ns"), 1, "C"):
        return lambda data: data

    if data == bodo.string_array_type:
        # TODO: support
        raise BodoError("conversion to timedelta from string not supported yet")

    raise TypeError("invalid data type {} for dt64 conversion".format(data))


def convert_to_index(data, name=None):  # pragma: no cover
    return data


@overload(convert_to_index, no_unliteral=True)
def overload_convert_to_index(data, name=None):
    """
    convert data to Index object if necessary.
    """
    from bodo.hiframes.pd_index_ext import (
        DatetimeIndexType,
        NumericIndexType,
        RangeIndexType,
        StringIndexType,
        TimedeltaIndexType,
    )

    # already Index
    if isinstance(
        data,
        (
            RangeIndexType,
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            StringIndexType,
            types.NoneType,
        ),
    ):
        return lambda data, name=None: data

    def impl(data, name=None):  # pragma: no cover
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        return bodo.utils.conversion.index_from_array(data_arr, name)

    return impl


def force_convert_index(I1, I2):  # pragma: no cover
    return I2


@overload(force_convert_index, no_unliteral=True)
def overload_force_convert_index(I1, I2):
    """
    Convert I1 to type of I2, with possible loss of data. TODO: remove this
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType

    if isinstance(I2, RangeIndexType):
        return lambda I1, I2: pd.RangeIndex(len(I1._data))

    return lambda I1, I2: I1


def index_from_array(data, name=None):  # pragma: no cover
    return data


@overload(index_from_array, no_unliteral=True)
def overload_index_from_array(data, name=None):
    """
    convert data array to Index object.
    """
    if data == bodo.string_array_type:
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_string_index(
            data, name
        )  # pragma: no cover

    assert (
        isinstance(data, (types.Array, bodo.libs.int_arr_ext.IntegerArrayType))
        or data == bodo.hiframes.datetime_date_ext.datetime_date_array_type
    )

    if (
        data == bodo.hiframes.datetime_date_ext.datetime_date_array_type
        or data.dtype == types.NPDatetime("ns")
    ):
        return lambda data, name=None: pd.DatetimeIndex(
            data, name=name
        )  # pragma: no cover

    if data.dtype == types.NPTimedelta("ns"):
        return lambda data, name=None: pd.TimedeltaIndex(
            data, name=name
        )  # pragma: no cover

    if isinstance(data.dtype, types.Integer):
        if not data.dtype.signed:
            return lambda data, name=None: pd.UInt64Index(
                data, name=name
            )  # pragma: no cover
        else:
            return lambda data, name=None: pd.Int64Index(
                data, name=name
            )  # pragma: no cover

    if isinstance(data.dtype, types.Float):
        return lambda data, name=None: pd.Float64Index(
            data, name=name
        )  # pragma: no cover

    # TODO: timedelta, period
    raise TypeError("invalid index type {}".format(data))


def index_to_array(data):  # pragma: no cover
    return data


@overload(index_to_array, no_unliteral=True)
def overload_index_to_array(I):
    """
    convert Index object to data array.
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType

    if isinstance(I, RangeIndexType):
        return lambda I: np.arange(I._start, I._stop, I._step)  # pragma: no cover

    # other indices have data
    return lambda I: bodo.hiframes.pd_index_ext.get_index_data(I)  # pragma: no cover


def false_if_none(val):  # pragma: no cover
    return False if val is None else val


@overload(false_if_none, no_unliteral=True)
def overload_false_if_none(val):
    """Return False if 'val' is None, otherwise same value"""

    if is_overload_none(val):
        return lambda val: False  # pragma: no cover

    return lambda val: val  # pragma: no cover


def extract_name_if_none(data, name):  # pragma: no cover
    return name


@overload(extract_name_if_none, no_unliteral=True)
def overload_extract_name_if_none(data, name):
    """Extract name if `data` is has name (Series/Index) and `name` is None"""
    from bodo.hiframes.pd_index_ext import (
        DatetimeIndexType,
        NumericIndexType,
        PeriodIndexType,
        TimedeltaIndexType,
    )
    from bodo.hiframes.pd_series_ext import SeriesType

    if not is_overload_none(name):
        return lambda data, name: name  # pragma: no cover

    # Index type, TODO: other indices like Range?
    if isinstance(
        data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType, PeriodIndexType)
    ):
        return lambda data, name: bodo.hiframes.pd_index_ext.get_index_name(
            data
        )  # pragma: no cover

    if isinstance(data, SeriesType):
        return lambda data, name: bodo.hiframes.pd_series_ext.get_series_name(
            data
        )  # pragma: no cover

    return lambda data, name: name  # pragma: no cover


def extract_index_if_none(data, index):  # pragma: no cover
    return index


@overload(extract_index_if_none, no_unliteral=True)
def overload_extract_index_if_none(data, index):
    """Extract index if `data` is Series and `index` is None"""
    from bodo.hiframes.pd_series_ext import SeriesType

    if not is_overload_none(index):
        return lambda data, index: index  # pragma: no cover

    if isinstance(data, SeriesType):
        return lambda data, index: bodo.hiframes.pd_series_ext.get_series_index(
            data
        )  # pragma: no cover

    return lambda data, index: bodo.hiframes.pd_index_ext.init_range_index(
        0, len(data), 1, None  # pragma: no cover
    )


def box_if_dt64(val):  # pragma: no cover
    return val


@overload(box_if_dt64, no_unliteral=True)
def overload_box_if_dt64(val):
    """If 'val' is dt64, box it to Timestamp otherwise just return 'val'"""
    if val == types.NPDatetime("ns"):
        return (
            lambda val: bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                val
            )
        )  # pragma: no cover

    if val == types.NPTimedelta("ns"):
        return lambda val: bodo.hiframes.pd_timestamp_ext.convert_numpy_timedelta64_to_pd_timedelta(
            val
        )  # pragma: no cover

    return lambda val: val  # pragma: no cover


def unbox_if_timestamp(val):  # pragma: no cover
    return val


@overload(unbox_if_timestamp, no_unliteral=True)
def overload_unbox_if_timestamp(val):
    """If 'val' is Timestamp, "unbox" it to dt64 otherwise just return 'val'"""
    # unbox Timestamp to dt64
    if val == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:
        return lambda val: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            val.value
        )  # pragma: no cover

    # unbox Timedelta to timedelta64
    if val == bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type:
        return lambda val: bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            val.value
        )  # pragma: no cover

    # Optional(timestamp)
    if val == types.Optional(bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type):

        def impl_optional(val):  # pragma: no cover
            if val is None:
                out = None
            else:
                out = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                    bodo.utils.indexing.unoptional(val).value
                )
            return out

        return impl_optional

    # Optional(Timedelta)
    if val == types.Optional(bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type):

        def impl_optional_td(val):  # pragma: no cover
            if val is None:
                out = None
            else:
                out = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    bodo.utils.indexing.unoptional(val).value
                )
            return out

        return impl_optional_td

    return lambda val: val  # pragma: no cover


def to_tuple(val):  # pragma: no cover
    return val


@overload(to_tuple, no_unliteral=True)
def overload_to_tuple(val):
    """convert tuple-like 'val' (e.g. constant list) to a tuple"""
    if not isinstance(val, types.BaseTuple) and is_overload_constant_list(val):
        # LiteralList values may be non-constant
        n_values = len(
            val.types
            if isinstance(val, types.LiteralList)
            else get_overload_const_list(val)
        )
        func_text = "def f(val):\n"
        res = ",".join(f"val[{i}]" for i in range(n_values))
        func_text += f"  return ({res},)\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        impl = loc_vars["f"]
        return impl

    assert isinstance(val, types.BaseTuple), "tuple type expected"
    return lambda val: val  # pragma: no cover


def get_array_if_series_or_index(data):  # pragma: no cover
    return data


@overload(get_array_if_series_or_index, no_unliteral=True)
def overload_get_array_if_series_or_index(data):
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(data, SeriesType):
        return lambda data: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )  # pragma: no cover

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        return lambda data: bodo.utils.conversion.coerce_to_array(
            data
        )  # pragma: no cover

    if isinstance(data, bodo.hiframes.pd_index_ext.HeterogeneousIndexType):
        # handle as regular array data if not actually heterogeneous
        if not is_heterogeneous_tuple_type(data.data):

            def impl(data):  # pragma: no cover
                in_data = bodo.hiframes.pd_index_ext.get_index_data(data)
                return bodo.utils.conversion.coerce_to_array(in_data)

            return impl

        # just pass the data and let downstream handle possible errors
        def impl(data):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.get_index_data(data)

        return impl

    return lambda data: data  # pragma: no cover


def extract_index_array(A):  # pragma: no cover
    return np.arange(len(A))


@overload(extract_index_array, no_unliteral=True)
def overload_extract_index_array(A):
    """Returns an index array for Series or array.
    if Series, return it's index array. Otherwise, create an index array.
    """
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(A, SeriesType):

        def impl(A):  # pragma: no cover
            index = bodo.hiframes.pd_series_ext.get_series_index(A)
            index_arr = bodo.utils.conversion.coerce_to_array(index)
            return index_arr

        return impl

    return lambda A: np.arange(len(A))


# return the NA value for array type (dtypes that support sentinel NA)
def get_NA_val_for_arr(arr):  # pragma: no cover
    return np.nan


@overload(get_NA_val_for_arr, no_unliteral=True)
def overload_get_NA_val_for_arr(arr):
    if isinstance(arr.dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = arr.dtype("NaT")
        return lambda arr: nat  # pragma: no cover

    if isinstance(arr.dtype, types.Float):
        return lambda arr: np.nan  # pragma: no cover

    # TODO: other types?
    raise BodoError(
        "Array {} does not support sentinel NA".format(arr)
    )  # pragma: no cover


def ensure_contig_if_np(arr):  # pragma: no cover
    return np.ascontiguousarray(arr)


@overload(ensure_contig_if_np, no_unliteral=True)
def overload_ensure_contig_if_np(arr):
    """make sure array 'arr' is contiguous in memory if it is a numpy array.
    Other arrays are always contiguous.
    """
    if isinstance(arr, types.Array):
        return lambda arr: np.ascontiguousarray(arr)  # pragma: no cover

    return lambda arr: arr  # pragma: no cover


def struct_if_heter_dict(values, names):  # pragma: no cover
    return {k: v for k, v in zip(names, values)}


@overload(struct_if_heter_dict, no_unliteral=True)
def overload_struct_if_heter_dict(values, names):
    """returns a struct with fields names 'names' and data 'values' if value types are
    heterogeneous, otherwise a regular dict.
    """

    if not types.is_homogeneous(*values.types):
        return lambda values, names: bodo.libs.struct_arr_ext.init_struct(
            values, names
        )  # pragma: no cover

    n_fields = len(values.types)
    func_text = "def f(values, names):\n"
    res = ",".join(
        "'{}': values[{}]".format(get_overload_const_str(names.types[i]), i)
        for i in range(n_fields)
    )
    func_text += "  return {{{}}}\n".format(res)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["f"]
    return impl


# def to_bool_array_if_np_bool(A):
#     return A


# @overload(to_bool_array_if_np_bool, no_unliteral=True)
# def overload_to_bool_array_if_np_bool(A):
#     """Returns a nullable BooleanArray if input is bool ndarray. Otherwise,
#     just returns the input.
#     """
#     if A == types.Array(types.bool_, 1, 'C'):
#         return lambda A: bodo.libs.bool_arr_ext.init_bool_array(
#                          A, np.full((len(A) + 7) >> 3, 255, np.uint8))

#     return lambda A: A
