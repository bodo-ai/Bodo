# Copyright (C) 2022 Bodo Inc. All rights reserved.
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
from bodo.hiframes.time_ext import TimeArrayType, cast_time_to_int
from bodo.libs.binary_arr_ext import bytes_type
from bodo.libs.bool_arr_ext import boolean_dtype
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.nullable_tuple_ext import NullableTupleType
from bodo.utils.indexing import add_nested_counts, init_nested_counts
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    get_overload_const_list,
    get_overload_const_str,
    is_heterogeneous_tuple_type,
    is_np_arr_typ,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    is_overload_true,
    is_str_arr_type,
    to_nullable_type,
    unwrap_typeref,
)

NS_DTYPE = np.dtype("M8[ns]")  # similar pandas/_libs/tslibs/conversion.pyx
TD_DTYPE = np.dtype("m8[ns]")


# TODO: use generated_jit with IR inlining
def coerce_to_ndarray(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):  # pragma: no cover
    return data


@overload(coerce_to_ndarray)
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

    if isinstance(data, types.Optional) and bodo.utils.typing.is_scalar_type(data.type):
        # If we have an optional scalar create a nullable array
        data = data.type
        use_nullable_array = True

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

        # If we have an optional type, extract the underlying type
        elem_type = data.dtype
        if isinstance(elem_type, types.Optional):
            elem_type = elem_type.type
            # If we have a scalar we need to use a nullable array
            if bodo.utils.typing.is_scalar_type(elem_type):
                use_nullable_array = True

        if isinstance(
            elem_type, (types.Boolean, types.Integer, Decimal128Type)
        ) or elem_type in [
            bodo.hiframes.pd_timestamp_ext.pd_timestamp_type,
            bodo.hiframes.datetime_date_ext.datetime_date_type,
            bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type,
        ]:
            arr_typ = dtype_to_array_type(elem_type)
            if not is_overload_none(use_nullable_array):
                arr_typ = to_nullable_type(arr_typ)

            def impl(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = len(data)
                A = bodo.utils.utils.alloc_type(n, arr_typ, (-1,))
                bodo.utils.utils.tuple_list_to_array(A, data, elem_type)
                return A

            return impl

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

        if isinstance(data, bodo.hiframes.time_ext.TimeType):

            precision = data.precision

            def impl_ts(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = bodo.hiframes.time_ext.alloc_time_array(n, precision)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = data
                return A

            return impl_ts

        # Timestamp values are stored as dt64 arrays
        if data == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type:
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

    # Tuple of numerics can be converted to Numpy array
    if isinstance(data, types.BaseTuple) and all(
        isinstance(t, (types.Float, types.Integer)) for t in data.types
    ):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: np.array(
            data
        )  # pragma: no cover

    # data is already an array
    if bodo.utils.utils.is_array_typ(data, False):
        return (
            lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
        )  # pragma: no cover

    if is_overload_true(error_on_nonarray):
        raise BodoError(f"cannot coerce {data} to array")

    return (
        lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: data
    )  # pragma: no cover


def coerce_scalar_to_array(scalar, length, arr_type):  # pragma: no cover
    pass


@overload(coerce_scalar_to_array)
def overload_coerce_scalar_to_array(scalar, length, arr_type):
    """
    Converts the given scalar to an array with the given length.
    If the scalar is None or optional then we generate the result
    as all NA with the given array type. If the value is optional
    we also convert the array to a nullable type.
    """
    # The array type always needs to be nullable for the gen_na_array case.
    _arr_typ = to_nullable_type(unwrap_typeref(arr_type))
    if scalar == types.none:
        # If the scalar is None we generate an array of all NA
        def impl(scalar, length, arr_type):  # pragma: no cover
            return bodo.libs.array_kernels.gen_na_array(length, _arr_typ, True)

    elif isinstance(scalar, types.Optional):

        def impl(scalar, length, arr_type):  # pragma: no cover
            if scalar is None:
                return bodo.libs.array_kernels.gen_na_array(length, _arr_typ, True)
            else:
                # If the data may be null both paths must produce the nullable array type.
                return bodo.utils.conversion.coerce_to_array(
                    bodo.utils.indexing.unoptional(scalar), True, True, length
                )

    else:

        def impl(scalar, length, arr_type):  # pragma: no cover
            return bodo.utils.conversion.coerce_to_array(scalar, True, None, length)

    return impl


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
    from bodo.hiframes.pd_index_ext import (
        BinaryIndexType,
        CategoricalIndexType,
        StringIndexType,
    )
    from bodo.hiframes.pd_series_ext import SeriesType

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    data = types.unliteral(data)
    if isinstance(data, types.Optional) and bodo.utils.typing.is_scalar_type(data.type):
        # If we have an optional scalar create a nullable array
        data = data.type
        use_nullable_array = True

    # series
    if isinstance(data, SeriesType):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )  # pragma: no cover

    # string/binary/categorical Index
    if isinstance(data, (StringIndexType, BinaryIndexType, CategoricalIndexType)):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.hiframes.pd_index_ext.get_index_data(
            data
        )  # pragma: no cover

    # string/binary list
    if isinstance(data, types.List) and data.dtype in (
        bodo.string_type,
        bodo.bytes_type,
    ):
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.str_arr_ext.str_arr_from_sequence(
            data
        )  # pragma: no cover

    # Empty Tuple
    # TODO: Remove once we can iterate with an empty tuple (next condition will capture this case)
    # Related Task: https://bodo.atlassian.net/browse/BE-1936
    if isinstance(data, types.BaseTuple) and data.count == 0:
        return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.libs.str_arr_ext.empty_str_arr(
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
        bodo.string_array_type,
        bodo.dict_str_arr_type,
        bodo.binary_array_type,
        bodo.libs.bool_arr_ext.boolean_array,
        bodo.hiframes.datetime_date_ext.datetime_date_array_type,
        bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_array_type,
        bodo.hiframes.split_impl.string_array_split_view_type,
    ) or isinstance(
        data,
        (
            bodo.libs.int_arr_ext.IntegerArrayType,
            DecimalArrayType,
            bodo.libs.interval_arr_ext.IntervalArrayType,
            bodo.libs.tuple_arr_ext.TupleArrayType,
            bodo.libs.struct_arr_ext.StructArrayType,
            bodo.hiframes.pd_categorical_ext.CategoricalArrayType,
            bodo.libs.csr_matrix_ext.CSRMatrixType,
            bodo.DatetimeArrayType,
            TimeArrayType,
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
        data_types = tuple(dtype_to_array_type(t) for t in data.dtype.types)

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
        data_arr_type = dtype_to_array_type(data.dtype.dtype)

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

    # string scalars to array. Since we know the scalar is repeated
    # for every value we opt to make the output array dictionary
    # encoded.
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
            # Use str_arr_from_sequence to force rep/avoid equiv_set
            dict_arr = bodo.libs.str_arr_ext.str_arr_from_sequence([data])
            indices = bodo.libs.int_arr_ext.alloc_int_array(n, np.int32)
            numba.parfors.parfor.init_prange()
            for i in numba.parfors.parfor.internal_prange(n):
                indices[i] = 0
            A = bodo.libs.dict_arr_ext.init_dict_arr(dict_arr, indices, True)
            return A

        return impl_str

    # Convert list of Timestamps to dt64 array
    if isinstance(data, types.List) and isinstance(
        data.dtype, bodo.hiframes.pd_timestamp_ext.PandasTimestampType
    ):

        # Currently only support tz naive. Need an allocation function.
        bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
            data, "coerce_to_array()"
        )

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

    # Timestamp with a timezone
    if (
        isinstance(data, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
        and data.tz is not None
    ):
        tz_literal = data.tz

        def impl_timestamp_tz_aware(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            A = np.empty(scalar_to_arr_len, "datetime64[ns]")
            dt64_val = data.to_datetime64()
            for i in numba.parfors.parfor.internal_prange(scalar_to_arr_len):
                A[i] = dt64_val
            return bodo.libs.pd_datetime_arr_ext.init_pandas_datetime_array(
                A, tz_literal
            )

        return impl_timestamp_tz_aware

    # Timestamp/Timedelta scalars to array
    if not is_overload_none(scalar_to_arr_len) and data in [
        bodo.pd_timestamp_type,
        bodo.pd_timedelta_type,
    ]:
        _dtype = (
            "datetime64[ns]" if data == bodo.pd_timestamp_type else "timedelta64[ns]"
        )

        def impl_timestamp(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
        ):  # pragma: no cover
            n = scalar_to_arr_len
            # NOTE: not using n to calculate n_chars since distributed pass will use
            # the global value of n and cannot replace it with the local version
            A = np.empty(n, _dtype)
            data = bodo.utils.conversion.unbox_if_timestamp(data)
            for i in numba.parfors.parfor.internal_prange(n):
                A[i] = data
            return A

        return impl_timestamp

    # assuming can be ndarray
    return lambda data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None: bodo.utils.conversion.coerce_to_ndarray(
        data, error_on_nonarray, use_nullable_array, scalar_to_arr_len
    )  # pragma: no cover


def _is_str_dtype(dtype):
    """return True if 'dtype' specifies a string data type."""
    return (
        isinstance(dtype, bodo.libs.str_arr_ext.StringDtype)
        or (isinstance(dtype, types.Function) and dtype.key[0] == str)
        or (is_overload_constant_str(dtype) and get_overload_const_str(dtype) == "str")
        or (
            isinstance(dtype, types.TypeRef)
            and dtype.instance_type == types.unicode_type
        )
    )


# TODO: use generated_jit with IR inlining
def fix_arr_dtype(
    data, new_dtype, copy=None, nan_to_str=True, from_series=False
):  # pragma: no cover
    pass


@overload(fix_arr_dtype, no_unliteral=True)
def overload_fix_arr_dtype(
    data, new_dtype, copy=None, nan_to_str=True, from_series=False
):
    """convert data to new_dtype, copy if copy parameter is not None.
    'nan_to_str' specifies string conversion for NA values: write as '<NA>'
    or actual NA (Pandas has inconsistent behavior in APIs).

    'from_series' specifies if the data originates from a series. This is useful for some
    operations where the casting behavior changes depending on if the input is a Series
    or an Array (specifically, S.astype(str) vs S.values.astype(str))
    """
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(data, "fix_arr_dtype()")
    do_copy = is_overload_true(copy)

    # If the new dtype is "object", we treat it as a no-op.
    is_object = (
        is_overload_constant_str(new_dtype)
        and get_overload_const_str(new_dtype) == "object"
    )
    if is_overload_none(new_dtype) or is_object:
        if do_copy:
            return (
                lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data.copy()
            )  # pragma: no cover
        return (
            lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data
        )  # pragma: no cover

    if isinstance(data, NullableTupleType):
        nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)
        if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
            nb_dtype = nb_dtype.dtype

        default_value_dict = {
            types.unicode_type: "",
            boolean_dtype: False,
            types.bool_: False,
            types.int8: np.int8(0),
            types.int16: np.int16(0),
            types.int32: np.int32(0),
            types.int64: np.int64(0),
            types.uint8: np.uint8(0),
            types.uint16: np.uint16(0),
            types.uint32: np.uint32(0),
            types.uint64: np.uint64(0),
            types.float32: np.float32(0),
            types.float64: np.float64(0),
            bodo.datetime64ns: pd.Timestamp(0),
            bodo.timedelta64ns: pd.Timedelta(0),
        }

        convert_func_dict = {
            types.unicode_type: str,
            types.bool_: bool,
            boolean_dtype: bool,
            types.int8: np.int8,
            types.int16: np.int16,
            types.int32: np.int32,
            types.int64: np.int64,
            types.uint8: np.uint8,
            types.uint16: np.uint16,
            types.uint32: np.uint32,
            types.uint64: np.uint64,
            types.float32: np.float32,
            types.float64: np.float64,
            bodo.datetime64ns: pd.to_datetime,
            bodo.timedelta64ns: pd.to_timedelta,
        }

        # If NA values properly done this should suffice for default_value_dict:
        # default_value_dict = {typ: func(0) for typ, func in convert_func_dict.items()}

        valid_types = default_value_dict.keys()
        scalar_types = list(data._tuple_typ.types)

        if nb_dtype not in valid_types:
            raise BodoError(f"type conversion to {nb_dtype} types unsupported.")
        for typ in scalar_types:
            if typ == bodo.datetime64ns:
                if nb_dtype not in (
                    types.unicode_type,
                    types.int64,
                    types.uint64,
                    bodo.datetime64ns,
                ):
                    raise BodoError(
                        f"invalid type conversion from {typ} to {nb_dtype}."
                    )
            elif typ == bodo.timedelta64ns:
                if nb_dtype not in (
                    types.unicode_type,
                    types.int64,
                    types.uint64,
                    bodo.timedelta64ns,
                ):
                    raise BodoError(
                        f"invalid type conversion from {typ} to {nb_dtype}."
                    )

        func_text = "def impl(data, new_dtype, copy=None, nan_to_str=True, from_series=False):\n"
        func_text += "  data_tup = data._data\n"
        func_text += "  null_tup = data._null_values\n"
        for i in range(len(scalar_types)):
            # may have type mismatch because default_value is treated as a literal
            # TODO: remove convert_func
            func_text += f"  val_{i} = convert_func(default_value)\n"
            func_text += f"  if not null_tup[{i}]:\n"
            func_text += f"    val_{i} = convert_func(data_tup[{i}])\n"
        vals_str = ", ".join(f"val_{i}" for i in range(len(scalar_types)))
        func_text += f"  vals_tup = ({vals_str},)\n"
        func_text += "  res_tup = bodo.libs.nullable_tuple_ext.build_nullable_tuple(vals_tup, null_tup)\n"
        func_text += "  return res_tup\n"
        loc_vars = {}
        convert_func = convert_func_dict[nb_dtype]
        default_value = default_value_dict[nb_dtype]
        exec(
            func_text,
            {
                "bodo": bodo,
                "np": np,
                "pd": pd,
                "default_value": default_value,
                "convert_func": convert_func,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]

        return impl

    # convert to string
    if _is_str_dtype(new_dtype):
        # special optimized case for int to string conversion, uses inplace write to
        # string array to avoid extra allocation
        if isinstance(data.dtype, types.Integer):

            def impl_int_str(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
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

        if data.dtype == bytes_type:
            # In pandas, binarySeries.astype(str) will call str on each of the bytes objects,
            # returning a string array.
            # For example:
            # Pandas behavior:
            #   pd.Series([b"a", b"c"]).astypes(str) == pd.Series(["b'a'", "b'c'"])
            # Desired Bodo Behavior:
            #   pd.Series([b"a", b"c"]).astypes(str) == pd.Series(["a", "c"])
            def impl_binary(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(data)
                A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
                for j in numba.parfors.parfor.internal_prange(n):

                    if bodo.libs.array_kernels.isna(data, j):
                        bodo.libs.array_kernels.setna(A, j)
                    else:
                        # TODO: replace his with .encode
                        A[j] = "".join([chr(z) for z in data[j]])

                return A

            return impl_binary

        if is_overload_true(from_series) and data.dtype in (
            bodo.datetime64ns,
            bodo.timedelta64ns,
        ):

            def impl_str_dt_series(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(data)
                A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
                for j in numba.parfors.parfor.internal_prange(n):

                    if bodo.libs.array_kernels.isna(data, j):
                        if nan_to_str:
                            A[j] = "NaT"
                        else:
                            bodo.libs.array_kernels.setna(A, j)
                        continue

                    # this is needed, as dt Series.astype(str) produces different output
                    # then Series.values.astype(str)
                    A[j] = str(box_if_dt64(data[j]))

                return A

            return impl_str_dt_series

        else:

            def impl_str_array(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
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

            return impl_str_array

    # convert to Categorical with predefined CategoricalDtype
    if isinstance(new_dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):

        def impl_cat_dtype(
            data, new_dtype, copy=None, nan_to_str=True, from_series=False
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
            data, new_dtype, copy=None, nan_to_str=True, from_series=False
        ):  # pragma: no cover

            # find categories in data, droping na
            cats = bodo.libs.array_kernels.unique(data, dropna=True)
            # sort categories to match Pandas behavior
            # TODO(ehsan): refactor to avoid long compilation time (too much inlining)
            cats = pd.Series(cats).sort_values().values
            # make sure categories are replicated since dtype is replicated
            # allgatherv should preserve sort ordering
            cats = bodo.allgatherv(cats, False)

            cat_dtype = bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                bodo.utils.conversion.index_from_array(cats, None), False, None, None
            )

            n = len(data)
            numba.parfors.parfor.init_prange()

            label_dict = bodo.hiframes.pd_categorical_ext.get_label_dict_from_categories_no_duplicates(
                cats
            )

            A = bodo.hiframes.pd_categorical_ext.alloc_categorical_array(n, cat_dtype)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A)

            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(A, i)
                    continue
                val = data[i]
                codes[i] = label_dict[val]

            return A

        return impl_category

    nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)

    # Matching data case
    if isinstance(data, bodo.libs.int_arr_ext.IntegerArrayType):
        same_typ = (
            isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype)
            and data.dtype == nb_dtype.dtype
        )
    else:
        same_typ = data.dtype == nb_dtype

    if do_copy and same_typ:
        return (
            lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data.copy()
        )  # pragma: no cover

    if same_typ:
        return (
            lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data
        )  # pragma: no cover

    # nullable int array case
    if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
        if isinstance(nb_dtype, types.Integer):
            _dtype = nb_dtype
        else:
            _dtype = nb_dtype.dtype

        if isinstance(data.dtype, types.Float):

            def impl_float(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
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
            # optimized implementation for dictionary arrays
            if data == bodo.dict_str_arr_type:

                def impl_dict(
                    data, new_dtype, copy=None, nan_to_str=True, from_series=False
                ):
                    return bodo.libs.dict_arr_ext.convert_dict_arr_to_int(data, _dtype)

                return impl_dict

            # perform specific time cast
            if isinstance(data, bodo.hiframes.time_ext.TimeArrayType):

                def impl(
                    data, new_dtype, copy=None, nan_to_str=True, from_series=False
                ):  # pragma: no cover
                    n = len(data)
                    numba.parfors.parfor.init_prange()
                    B = bodo.libs.int_arr_ext.alloc_int_array(n, _dtype)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(data, i):
                            bodo.libs.array_kernels.setna(B, i)
                        else:
                            B[i] = cast_time_to_int(data[i])
                    return B

                return impl

            # data is a string array or integer array (nullable or non-nullable)
            def impl(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                n = len(data)
                numba.parfors.parfor.init_prange()
                B = bodo.libs.int_arr_ext.alloc_int_array(n, _dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, i):
                        bodo.libs.array_kernels.setna(B, i)
                    else:
                        # Cast the data to support conversion for
                        # string arrays. There may be an extra cast
                        # for the setitem if the array is not int64,
                        # but this should never impact correctness.
                        B[i] = np.int64(data[i])
                return B

            return impl

    # nullable int array to non-nullable int array case
    if isinstance(nb_dtype, types.Integer) and isinstance(data.dtype, types.Integer):

        def impl(
            data, new_dtype, copy=None, nan_to_str=True, from_series=False
        ):  # pragma: no cover
            return data.astype(nb_dtype)

        return impl

    # nullable bool array case
    if nb_dtype == bodo.libs.bool_arr_ext.boolean_dtype:

        def impl_bool(
            data, new_dtype, copy=None, nan_to_str=True, from_series=False
        ):  # pragma: no cover
            n = len(data)
            numba.parfors.parfor.init_prange()
            B = bodo.libs.bool_arr_ext.alloc_bool_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(B, i)
                else:
                    B[i] = bool(data[i])
            return B

        return impl_bool

    # Note astype(datetime.date) isn't possible in Pandas because its treated
    # as an object type. We support it to maintain parity with Spark's cast.
    if nb_dtype == bodo.datetime_date_type:
        # This operation isn't defined in Pandas, so we opt to implement it as
        # truncating to the date, which best resembles a cast.
        if data.dtype == bodo.datetime64ns:

            def impl_date(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                n = len(data)
                out_arr = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = bodo.utils.conversion.box_if_dt64(data[i]).date()
                return out_arr

            return impl_date

    # Datetime64 case
    if nb_dtype == bodo.datetime64ns:

        if data.dtype == bodo.string_type:
            # Support String Arrays using objmode
            def impl_str(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):
                # Keep the objmode in a separate function for
                # inlining purposes.
                return bodo.hiframes.pd_timestamp_ext.series_str_dt64_astype(data)

            return impl_str

        if data == bodo.datetime_date_array_type:
            # Support Date Arrays using objmode
            # TODO: Replace with a native impl
            def impl_date(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                return bodo.hiframes.pd_timestamp_ext.datetime_date_arr_to_dt64_arr(
                    data
                )

            return impl_date

        if isinstance(data.dtype, types.Number) or data.dtype in [
            bodo.timedelta64ns,
            types.bool_,
        ]:
            # Nullable Integer/boolean/timedelta64 arrays
            def impl_numeric(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):  # pragma: no cover
                n = len(data)
                numba.parfors.parfor.init_prange()
                out_arr = np.empty(n, dtype=np.dtype("datetime64[ns]"))
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                            np.int64(data[i])
                        )
                return out_arr

            return impl_numeric

    # Timedelta64 case
    if nb_dtype == bodo.timedelta64ns:

        if data.dtype == bodo.string_type:
            # Support String Arrays using objmode
            def impl_str(
                data, new_dtype, copy=None, nan_to_str=True, from_series=False
            ):
                # Keep the objmode in a separate function for
                # inlining purposes.
                return bodo.hiframes.pd_timestamp_ext.series_str_td64_astype(data)

            return impl_str

        if isinstance(data.dtype, types.Number) or data.dtype in [
            bodo.datetime64ns,
            types.bool_,
        ]:
            if do_copy:
                # Nullable Integer/boolean/datetime64 arrays
                def impl_numeric(
                    data, new_dtype, copy=None, nan_to_str=True, from_series=False
                ):  # pragma: no cover
                    n = len(data)
                    numba.parfors.parfor.init_prange()
                    out_arr = np.empty(n, dtype=np.dtype("timedelta64[ns]"))
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(data, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[
                                i
                            ] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                                np.int64(data[i])
                            )
                    return out_arr

                return impl_numeric

            else:
                return lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data.view(
                    "int64"
                )  # pragma: no cover

    # Pandas currently only supports dt64/td64 -> int64
    if (nb_dtype == types.int64) and (
        data.dtype in [bodo.datetime64ns, bodo.timedelta64ns]
    ):

        def impl_datelike_to_integer(
            data, new_dtype, copy=None, nan_to_str=True, from_series=False
        ):  # pragma: no cover
            n = len(data)
            numba.parfors.parfor.init_prange()
            A = np.empty(n, types.int64)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(A, i)
                else:
                    A[i] = np.int64(data[i])
            return A

        return impl_datelike_to_integer

    if data.dtype != nb_dtype:
        return lambda data, new_dtype, copy=None, nan_to_str=True, from_series=False: data.astype(
            nb_dtype
        )  # pragma: no cover

    raise BodoError(f"Conversion from {data} to {new_dtype} not supported yet")


def array_type_from_dtype(dtype):
    return dtype_to_array_type(bodo.utils.typing.parse_dtype(dtype))


@overload(array_type_from_dtype)
def overload_array_type_from_dtype(dtype):
    """parse dtype and return corresponding array type TypeRef"""
    arr_type = dtype_to_array_type(bodo.utils.typing.parse_dtype(dtype))
    return lambda dtype: arr_type  # pragma: no cover


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
    assert is_str_arr_type(data), "parse_datetimes_from_strings: string array expected"

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

    if is_np_arr_typ(data, types.int64):
        return lambda data: data.view(
            bodo.utils.conversion.NS_DTYPE
        )  # pragma: no cover

    if is_np_arr_typ(data, types.NPDatetime("ns")):
        return lambda data: data  # pragma: no cover

    if is_str_arr_type(data):
        return lambda data: bodo.utils.conversion.parse_datetimes_from_strings(
            data
        )  # pragma: no cover

    raise BodoError(f"invalid data type {data} for dt64 conversion")


# TODO: use generated_jit with IR inlining
def convert_to_td64ns(data):  # pragma: no cover
    return data


@overload(convert_to_td64ns, no_unliteral=True)
def overload_convert_to_td64ns(data):
    """Converts data formats like int64 to timedelta64ns"""
    # TODO: array of strings
    # see pd.core.arrays.timedeltas.sequence_to_td64ns for constructor types
    # TODO: support datetime.timedelta
    if is_np_arr_typ(data, types.int64):
        return lambda data: data.view(
            bodo.utils.conversion.TD_DTYPE
        )  # pragma: no cover

    if is_np_arr_typ(data, types.NPTimedelta("ns")):
        return lambda data: data  # pragma: no cover

    if is_str_arr_type(data):
        # TODO: support
        raise BodoError("conversion to timedelta from string not supported yet")

    raise BodoError(f"invalid data type {data} for timedelta64 conversion")


def convert_to_index(data, name=None):  # pragma: no cover
    return data


@overload(convert_to_index, no_unliteral=True)
def overload_convert_to_index(data, name=None):
    """
    convert data to Index object if necessary.
    """
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

    # already Index
    if isinstance(
        data,
        (
            RangeIndexType,
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            StringIndexType,
            BinaryIndexType,
            CategoricalIndexType,
            PeriodIndexType,
            types.NoneType,
        ),
    ):
        return lambda data, name=None: data  # pragma: no cover

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
    if data in [bodo.string_array_type, bodo.binary_array_type, bodo.dict_str_arr_type]:
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_binary_str_index(
            data, name
        )  # pragma: no cover

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

    if isinstance(data.dtype, (types.Integer, types.Float, types.Boolean)):
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_numeric_index(
            data, name
        )  # pragma: no cover

    # interval array
    if isinstance(data, bodo.libs.interval_arr_ext.IntervalArrayType):
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_interval_index(
            data, name
        )  # pragma: no cover

    # categorical array
    if isinstance(data, bodo.hiframes.pd_categorical_ext.CategoricalArrayType):
        return (
            lambda data, name=None: bodo.hiframes.pd_index_ext.init_categorical_index(
                data, name
            )
        )  # pragma: no cover

    # datetime array
    if isinstance(data, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType):
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_datetime_index(
            data, name
        )  # pragma: no cover

    # TODO: timedelta, period
    raise BodoError(f"cannot convert {data} to Index")  # pragma: no cover


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
        CategoricalIndexType,
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
        data,
        (
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            PeriodIndexType,
            CategoricalIndexType,
        ),
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
    if val == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type:
        return lambda val: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            val.value
        )  # pragma: no cover

    # unbox datetime.datetime to dt64
    if val == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type:
        return lambda val: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            pd.Timestamp(val).value
        )  # pragma: no cover

    # unbox Timedelta to timedelta64
    if val == bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type:
        return lambda val: bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            val.value
        )  # pragma: no cover

    # Optional(timestamp)
    if val == types.Optional(bodo.hiframes.pd_timestamp_ext.pd_timestamp_type):

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


@overload(get_array_if_series_or_index)
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

    return lambda A: np.arange(len(A))  # pragma: no cover


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


def nullable_bool_to_bool_na_false(arr):
    """Takes a nullable boolean array and converts
    it to a numpy boolean array but it each entry
    that is NA is set to False.
    """


@overload(nullable_bool_to_bool_na_false)
def overload_nullable_bool_to_bool_na_false(arr):
    if arr == bodo.boolean_array:

        def impl(arr):  # pragma: no cover
            output_arr = bodo.libs.bool_arr_ext.get_bool_arr_data(arr)
            for i in range(len(arr)):
                output_arr[i] = output_arr[i] and (
                    not bodo.libs.array_kernels.isna(arr, i)
                )
            return output_arr

        return impl
    else:
        return lambda arr: arr  # pragma: no cover
