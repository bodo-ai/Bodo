"""
Utility functions for conversion of data such as list to array.
Need to be inlined for better optimization.
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.core.ir_utils import next_label
from numba.core.typing.templates import AbstractTemplate, infer_global, signature
from numba.extending import lower_builtin, overload

import bodo
from bodo.hiframes.time_ext import cast_time_to_int
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    array_to_repeated_array_item_array,
)
from bodo.libs.binary_arr_ext import bytes_type
from bodo.libs.bool_arr_ext import boolean_dtype
from bodo.libs.nullable_tuple_ext import NullableTupleType
from bodo.libs.str_arr_ext import get_utf8_size
from bodo.utils.indexing import add_nested_counts, init_nested_counts
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    get_overload_const_bool,
    get_overload_const_list,
    get_overload_const_str,
    is_heterogeneous_tuple_type,
    is_np_arr_typ,
    is_nullable_type,
    is_overload_constant_bool,
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


def coerce_to_ndarray(
    data, error_on_nonarray=True, use_nullable_array=None, scalar_to_arr_len=None
):  # pragma: no cover
    return data


@infer_global(coerce_to_ndarray)
class CoerceToNdarrayInfer(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_index_ext import (
            DatetimeIndexType,
            NumericIndexType,
            RangeIndexType,
            TimedeltaIndexType,
        )
        from bodo.hiframes.pd_series_ext import SeriesType

        pysig = numba.core.utils.pysignature(coerce_to_ndarray)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        data, error_on_nonarray, use_nullable_array, scalar_to_arr_len = folded_args
        data = types.unliteral(data)

        if isinstance(data, types.Optional) and bodo.utils.typing.is_scalar_type(
            data.type
        ):
            # If we have an optional scalar create a nullable array
            data = data.type
            use_nullable_array = True

        if isinstance(
            data, bodo.libs.int_arr_ext.IntegerArrayType
        ) and is_overload_none(use_nullable_array):
            return signature(types.Array(data.dtype, 1, "C"), *folded_args).replace(
                pysig=pysig
            )
        if isinstance(
            data, bodo.libs.float_arr_ext.FloatingArrayType
        ) and is_overload_none(use_nullable_array):
            return signature(types.Array(data.dtype, 1, "C"), *folded_args).replace(
                pysig=pysig
            )
        if data == bodo.libs.bool_arr_ext.boolean_array_type:
            return signature(data, *folded_args).replace(pysig=pysig)

        if isinstance(data, (types.List, types.UniTuple)):
            # If we have an optional type, extract the underlying type
            elem_type = data.dtype
            if isinstance(elem_type, types.Optional):
                elem_type = elem_type.type
                # If we have a scalar we need to use a nullable array
                if bodo.utils.typing.is_scalar_type(elem_type):
                    use_nullable_array = True

            arr_typ = dtype_to_array_type(elem_type)
            if not is_overload_none(use_nullable_array):
                arr_typ = to_nullable_type(arr_typ)

            return signature(arr_typ, *folded_args).replace(pysig=pysig)

        if isinstance(data, types.Array):
            if not is_overload_none(use_nullable_array) and (
                isinstance(data.dtype, (types.Boolean, types.Integer, types.Float))
                or data.dtype == bodo.types.timedelta64ns
                or data.dtype == bodo.types.datetime64ns
            ):
                if data.dtype == types.bool_:
                    output = bodo.types.boolean_array_type
                elif data.dtype == bodo.types.timedelta64ns:
                    output = bodo.types.timedelta_array_type
                elif data.dtype == bodo.types.datetime64ns:
                    output = bodo.types.DatetimeArrayType(None)
                elif isinstance(data.dtype, types.Float):
                    output = bodo.types.FloatingArrayType(data.dtype)
                else:  # Integer case
                    output = bodo.types.IntegerArrayType(data.dtype)
                return signature(output, *folded_args).replace(pysig=pysig)
            if data.layout != "C":
                return signature(data.copy(layout="C"), *folded_args).replace(
                    pysig=pysig
                )
            return signature(data, *folded_args).replace(pysig=pysig)

        if isinstance(data, RangeIndexType):
            if not is_overload_none(use_nullable_array):
                return signature(
                    bodo.types.IntegerArrayType(data.dtype), *folded_args
                ).replace(pysig=pysig)
            return signature(types.Array(data.dtype, 1, "C"), *folded_args).replace(
                pysig=pysig
            )

        if isinstance(data, types.RangeType):
            return signature(types.Array(data.dtype, 1, "C"), *folded_args).replace(
                pysig=pysig
            )

        if isinstance(data, SeriesType):
            return signature(data.data, *folded_args).replace(pysig=pysig)

        # index types
        if isinstance(data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):
            if isinstance(data, NumericIndexType) and not is_overload_none(
                use_nullable_array
            ):
                if isinstance(data.dtype, types.Integer):
                    return signature(
                        bodo.types.IntegerArrayType(data.dtype), *folded_args
                    ).replace(pysig=pysig)
                else:
                    return signature(
                        bodo.types.FloatingArrayType(data.dtype), *folded_args
                    ).replace(pysig=pysig)
            return signature(data.data, *folded_args).replace(pysig=pysig)

        if not is_overload_none(scalar_to_arr_len):
            if isinstance(data, bodo.types.Decimal128Type):
                output = bodo.libs.decimal_arr_ext.DecimalArrayType(
                    data.precision, data.scale
                )
            elif data == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type:
                output = types.Array(bodo.types.datetime64ns, 1, "C")
            elif data == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
                output = types.Array(bodo.types.timedelta64ns, 1, "C")
            elif data == bodo.hiframes.datetime_date_ext.datetime_date_type:
                output = bodo.types.datetime_date_array_type
            elif isinstance(data, bodo.hiframes.time_ext.TimeType):
                output = bodo.types.TimeArrayType(data.precision)
            elif data == bodo.types.timestamptz_type:
                output = bodo.types.timestamptz_array_type
            # Timestamp values are stored as dt64 arrays
            elif data == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type:
                output = types.Array(np.dtype("datetime64[ns]"), 1, "C")
            elif not is_overload_none(use_nullable_array):
                dtype = types.unliteral(data)
                if isinstance(dtype, types.Integer):
                    output = bodo.types.IntegerArrayType(dtype)
                elif isinstance(dtype, types.Float):
                    output = bodo.types.FloatingArrayType(dtype)
                elif dtype == types.bool_:
                    output = bodo.types.boolean_array_type
            else:
                output = types.Array(data, 1, "C")
            return signature(output, *folded_args).replace(pysig=pysig)

        if bodo.utils.utils.is_array_typ(data, False):
            return signature(data, *folded_args).replace(pysig=pysig)

        if is_overload_true(error_on_nonarray):
            raise BodoError(f"cannot coerce {data} to ndarray")

        return signature(data, *folded_args).replace(pysig=pysig)


CoerceToNdarrayInfer._no_unliteral = True  # type: ignore


def np_to_nullable_array(data):
    pass


@overload(np_to_nullable_array, jit_options={"cache": True})
def overload_np_to_nullable_array(data):
    """Converts a Numpy array (bool, float, int) to an equivalent nullable array. This
    function should not be inlined since the bitmap length calculations can cause issues
    in distributed transformation.
    """
    assert isinstance(data, types.Array), "np_to_nullable_array: Numpy array expected"

    if data.dtype == types.bool_:

        def impl(data):  # pragma: no cover
            n = len(data)
            out_array = bodo.libs.bool_arr_ext.alloc_bool_array(n)
            for i in range(n):
                out_array[i] = data[i]
            return out_array

        return impl
    elif isinstance(data.dtype, types.Float):
        if data.layout != "C":
            return lambda data: bodo.libs.float_arr_ext.init_float_array(
                np.ascontiguousarray(data),
                np.full((len(data) + 7) >> 3, 255, np.uint8),
            )  # pragma: no cover
        else:
            return lambda data: bodo.libs.float_arr_ext.init_float_array(
                data, np.full((len(data) + 7) >> 3, 255, np.uint8)
            )  # pragma: no cover
    elif isinstance(data.dtype, types.Integer):
        if data.layout != "C":
            return lambda data: bodo.libs.int_arr_ext.init_integer_array(
                np.ascontiguousarray(data),
                np.full((len(data) + 7) >> 3, 255, np.uint8),
            )  # pragma: no cover
        else:
            return lambda data: bodo.libs.int_arr_ext.init_integer_array(
                data, np.full((len(data) + 7) >> 3, 255, np.uint8)
            )  # pragma: no cover
    elif data.dtype == bodo.types.timedelta64ns:
        if data.layout != "C":
            return (
                lambda data: bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array(
                    np.ascontiguousarray(data),
                    np.full((len(data) + 7) >> 3, 255, np.uint8),
                )
            )  # pragma: no cover
        else:
            return (
                lambda data: bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array(
                    data, np.full((len(data) + 7) >> 3, 255, np.uint8)
                )
            )  # pragma: no cover
    elif data.dtype == bodo.types.datetime64ns:
        if data.layout != "C":

            def func(data):
                new_bitmask = np.full((len(data) + 7) >> 3, 255, np.uint8)

                for i in range(len(data)):
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        new_bitmask, i, 0 if np.isnat(data[i]) else 1
                    )

                return bodo.libs.pd_datetime_arr_ext.init_datetime_array(
                    np.ascontiguousarray(data), new_bitmask, None
                )

            return func
        else:

            def func(data):
                new_bitmask = np.full((len(data) + 7) >> 3, 255, np.uint8)

                for i in range(len(data)):
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        new_bitmask, i, 0 if np.isnat(data[i]) else 1
                    )

                return bodo.libs.pd_datetime_arr_ext.init_datetime_array(
                    data, new_bitmask, None
                )

            return func

    raise BodoError(
        f"np_to_nullable_array: invalid dtype {data.dtype}, integer, bool or float dtype expected"
    )


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
    if isinstance(data, bodo.libs.int_arr_ext.IntegerArrayType) and is_overload_none(
        use_nullable_array
    ):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: bodo.libs.int_arr_ext.get_int_arr_data(data)
        )  # pragma: no cover

    # nullable float array
    if isinstance(data, bodo.libs.float_arr_ext.FloatingArrayType) and is_overload_none(
        use_nullable_array
    ):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: bodo.libs.float_arr_ext.get_float_arr_data(data)
        )  # pragma: no cover

    # nullable boolean array
    if data == bodo.libs.bool_arr_ext.boolean_array_type:
        # Always keep nullable boolean arrays as nullable booleans.
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: data
        )  # pragma: no cover

    # numpy array
    if isinstance(data, types.Array):
        if not is_overload_none(use_nullable_array) and (
            isinstance(data.dtype, (types.Boolean, types.Integer, types.Float))
            or data.dtype == bodo.types.timedelta64ns
            or data.dtype == bodo.types.datetime64ns
        ):
            return (
                lambda data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None: bodo.utils.conversion.np_to_nullable_array(data)
            )  # pragma: no cover

        if data.layout != "C":
            return (
                lambda data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None: np.ascontiguousarray(data)
            )  # pragma: no cover
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: data
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

    # series
    if isinstance(data, SeriesType):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: bodo.hiframes.pd_series_ext.get_series_data(data)
        )  # pragma: no cover

    # index types
    if isinstance(data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):
        if isinstance(data, NumericIndexType) and not is_overload_none(
            use_nullable_array
        ):
            return (
                lambda data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None: bodo.utils.conversion.coerce_to_array(
                    bodo.hiframes.pd_index_ext.get_index_data(data),
                    use_nullable_array=True,
                )
            )  # pragma: no cover

        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: bodo.hiframes.pd_index_ext.get_index_data(data)
        )  # pragma: no cover

    # RangeIndex
    if isinstance(data, RangeIndexType):
        if not is_overload_none(use_nullable_array):
            return (
                lambda data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None: bodo.utils.conversion.coerce_to_array(
                    np.arange(data._start, data._stop, data._step),
                    use_nullable_array=True,
                )
            )  # pragma: no cover

        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: np.arange(data._start, data._stop, data._step)
        )  # pragma: no cover

    # types.RangeType
    if isinstance(data, types.RangeType):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: np.arange(data.start, data.stop, data.step)
        )  # pragma: no cover

    # convert scalar to ndarray
    # TODO: make sure scalar is a Numpy dtype

    if not is_overload_none(scalar_to_arr_len):
        if isinstance(data, bodo.types.Decimal128Type):
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

        if data == bodo.types.timestamptz_type:

            def impl_timestamptz(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    A[i] = data
                return A

            return impl_timestamptz

        # Timestamp values are stored as dt64 arrays
        if data == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type:
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

        if not is_overload_none(use_nullable_array) and isinstance(dtype, types.Float):

            def impl_null_float(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = scalar_to_arr_len
                out_arr = bodo.libs.float_arr_ext.alloc_float_array(n, dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    out_arr[i] = data
                return out_arr

            return impl_null_float

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
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: np.array(data)
        )  # pragma: no cover

    # data is already an array
    if bodo.utils.utils.is_array_typ(data, False):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None: data
        )  # pragma: no cover

    if is_overload_true(error_on_nonarray):
        raise BodoError(f"cannot coerce {data} to array")

    return (
        lambda data,
        error_on_nonarray=True,
        use_nullable_array=None,
        scalar_to_arr_len=None: data
    )  # pragma: no cover


@lower_builtin(coerce_to_ndarray, types.Any, types.Any, types.Any, types.Any)
def lower_coerce_to_ndarray(context, builder, sig, args):
    impl = overload_coerce_to_ndarray(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def coerce_scalar_to_array(
    scalar, length, arr_type, dict_encode=True
):  # pragma: no cover
    pass


@overload(coerce_scalar_to_array, jit_options={"cache": True})
def overload_coerce_scalar_to_array(scalar, length, arr_type, dict_encode=True):
    """
    Converts the given scalar to an array with the given length.
    If the scalar is None or optional then we generate the result
    as all NA with the given array type. If the value is optional
    we also convert the array to a nullable type.

    If the type scalar is already the required output array type, we return it.
    """
    # The array type always needs to be nullable for the gen_na_array case.
    _arr_typ = to_nullable_type(unwrap_typeref(arr_type))

    # If we were supplied unknown as the type, then we should instead use the
    # type of the scalar as the type to construct.
    if _arr_typ == types.unknown:
        _arr_typ = to_nullable_type(dtype_to_array_type(scalar))

    if _arr_typ == bodo.types.null_array_type:
        return (
            lambda scalar,
            length,
            arr_type,
            dict_encode=True: bodo.libs.null_arr_ext.init_null_array(length)
        )  # pragma: no cover

    if isinstance(_arr_typ, ArrayItemArrayType):
        if scalar == _arr_typ:
            # If the scalar is the same as the output array type we can just
            # return the scalar.
            return lambda scalar, length, arr_type, dict_encode=True: scalar

        # If the output array is ArrayItemArray
        data_arr_type = _arr_typ.dtype

        # Ideally, we would have some sort of compile time check that the scalar type is
        # compatible with the output array type, so we can throw a readable error if
        # it weren't the case. However, we run into complications due to differences
        # between our nullable types, and Numba's array types. For example,
        # ArrayItemArrayType(BooleanArrayType())'s dtype is Array(bool, 1, 'C', False, aligned=True)
        # NOT BooleanArrayType().
        # For now, we're not doing any checks, and just let the code fail with
        # a numba type coercion error.

        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            return array_to_repeated_array_item_array(scalar, length, data_arr_type)

        return impl

    if isinstance(_arr_typ, bodo.types.MapArrayType):

        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            return bodo.libs.map_arr_ext.scalar_to_map_array(scalar, length, _arr_typ)

        return impl

    if isinstance(_arr_typ, bodo.types.StructArrayType):

        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            return bodo.libs.struct_arr_ext.scalar_to_struct_array(
                scalar, length, _arr_typ
            )

        return impl

    if scalar == types.none:
        # If the scalar is None we generate an array of all NA
        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            return bodo.libs.array_kernels.gen_na_array(length, _arr_typ, True)

    elif isinstance(scalar, types.Optional):

        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            if scalar is None:
                return bodo.libs.array_kernels.gen_na_array(length, _arr_typ, True)
            else:
                # If the data may be null both paths must produce the nullable array type.
                return bodo.utils.conversion.coerce_to_array(
                    bodo.utils.indexing.unoptional(scalar),
                    True,
                    True,
                    length,
                    dict_encode,
                )

    else:

        def impl(scalar, length, arr_type, dict_encode=True):  # pragma: no cover
            return bodo.utils.conversion.coerce_to_array(
                scalar, True, None, length, dict_encode
            )

    return impl


def ndarray_if_nullable_arr(data):
    pass


@overload(ndarray_if_nullable_arr, jit_options={"cache": True})
def overload_ndarray_if_nullable_arr(data):
    """convert input to Numpy array if it is a nullable array but return any other input as-is."""
    if data == bodo.libs.bool_arr_ext.boolean_array_type:
        # Handle boolean arrays separately since coerce_to_ndarray keeps
        # the nullable boolean type.
        return lambda data: data.to_numpy()  # pragma: no cover
    if (
        isinstance(
            data,
            (
                bodo.libs.int_arr_ext.IntegerArrayType,
                bodo.libs.float_arr_ext.FloatingArrayType,
            ),
        )
        or data == bodo.libs.bool_arr_ext.boolean_array_type
    ):
        return lambda data: bodo.utils.conversion.coerce_to_ndarray(
            data
        )  # pragma: no cover

    if isinstance(data, bodo.types.DatetimeArrayType):
        return lambda data: data._data  # pragma: no cover

    return lambda data: data  # pragma: no cover


def coerce_to_array(
    data,
    error_on_nonarray=True,
    use_nullable_array=None,
    scalar_to_arr_len=None,
    dict_encode=True,
):  # pragma: no cover
    return data


@overload(coerce_to_array, no_unliteral=True, jit_options={"cache": True})
def overload_coerce_to_array(
    data,
    error_on_nonarray=True,
    use_nullable_array=None,
    scalar_to_arr_len=None,
    dict_encode=True,
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
        if not is_overload_none(use_nullable_array) and (
            not is_nullable_type(data.data)
            or isinstance(
                data.data,
                (
                    ArrayItemArrayType,
                    bodo.types.TupleArrayType,
                    bodo.types.StructArrayType,
                    bodo.types.MapArrayType,
                ),
            )
            or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(data)
            or bodo.hiframes.pd_series_ext.is_dt64_series_typ(data)
        ):

            def impl_series_to_nullable(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
                dict_encode=True,
            ):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(data)
                return bodo.utils.conversion.coerce_to_array(
                    arr, use_nullable_array=True
                )

            return impl_series_to_nullable

        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: bodo.hiframes.pd_series_ext.get_series_data(data)
        )  # pragma: no cover

    if isinstance(data, ArrayItemArrayType) and not is_overload_none(
        use_nullable_array
    ):
        # Convert inner types to nullable

        def impl_array_item_array_to_nullable(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            new_inner_data = bodo.utils.conversion.coerce_to_array(
                bodo.libs.array_item_arr_ext.get_data(data), use_nullable_array=True
            )
            new_data = bodo.libs.array_item_arr_ext.init_array_item_array(
                bodo.libs.array_item_arr_ext.get_n_arrays(data),
                new_inner_data,
                bodo.libs.array_item_arr_ext.get_offsets(data),
                bodo.libs.array_item_arr_ext.get_null_bitmap(data),
            )
            return new_data

        return impl_array_item_array_to_nullable

    if isinstance(data, bodo.types.StructArrayType) and not is_overload_none(
        use_nullable_array
    ):
        # Convert inner types to nullable
        n_fields = len(data.data)
        field_names = data.names

        if n_fields == 0:
            return (
                lambda data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
                dict_encode=True: data
            )  # pragma: no cover

        func_text = (
            "def bodo_impl_struct_array_to_nullable("
            "    data,"
            "    error_on_nonarray=True,"
            "    use_nullable_array=None,"
            "    scalar_to_arr_len=None,"
            "    dict_encode=True"
            "):\n"
        )
        func_text += "  inner_data_arrs = bodo.libs.struct_arr_ext.get_data(data)\n"

        for i in range(n_fields):
            func_text += (
                f"  new_inner_data_arr_{i} = bodo.utils.conversion.coerce_to_array("
                f"inner_data_arrs[{i}], use_nullable_array=True)\n"
            )

        new_data_tuple_str = "({},)".format(
            ", ".join([f"new_inner_data_arr_{i}" for i in range(n_fields)])
        )
        field_names_tuple_str = "({},)".format(
            ", ".join([f"'{f}'" for f in field_names])
        )

        func_text += (
            f"  new_data = bodo.libs.struct_arr_ext.init_struct_arr("
            f"{n_fields},"
            f"{new_data_tuple_str},"
            "bodo.libs.struct_arr_ext.get_null_bitmap(data),"
            f"{field_names_tuple_str}"
            ")\n"
        )

        func_text += "  return new_data"

        return bodo.utils.utils.bodo_exec(func_text, {"bodo": bodo}, {}, __name__)

    if isinstance(data, bodo.types.TupleArrayType) and not is_overload_none(
        use_nullable_array
    ):
        # Convert inner types to nullable

        def impl_tuple_array_to_nullable(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            new_data = bodo.utils.conversion.coerce_to_array(
                data._data, use_nullable_array=True
            )
            return bodo.libs.tuple_arr_ext.init_tuple_arr(new_data)

        return impl_tuple_array_to_nullable

    if isinstance(data, bodo.types.MapArrayType) and not is_overload_none(
        use_nullable_array
    ):
        # Convert inner types to nullable

        def impl_map_array_to_nullable(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            new_data = bodo.utils.conversion.coerce_to_array(
                data._data, use_nullable_array=True
            )
            return bodo.libs.map_arr_ext.init_map_arr(new_data)

        return impl_map_array_to_nullable

    # string/binary/categorical Index
    if isinstance(data, (StringIndexType, BinaryIndexType, CategoricalIndexType)):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: bodo.hiframes.pd_index_ext.get_index_data(data)
        )  # pragma: no cover

    # string/binary list
    if isinstance(data, types.List) and data.dtype in (
        bodo.types.string_type,
        bodo.types.bytes_type,
    ):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: bodo.libs.str_arr_ext.str_arr_from_sequence(data)
        )  # pragma: no cover

    # Empty Tuple
    # TODO: Remove once we can iterate with an empty tuple (next condition will capture this case)
    # Related Task: https://bodo.atlassian.net/browse/BE-1936
    if isinstance(data, types.BaseTuple) and data.count == 0:
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: bodo.libs.str_arr_ext.empty_str_arr(data)
        )  # pragma: no cover

    # string tuple
    if (
        isinstance(data, types.UniTuple)
        and isinstance(data.dtype, (types.UnicodeType, types.StringLiteral))
    ) or (
        isinstance(data, types.BaseTuple)
        and all(isinstance(t, types.StringLiteral) for t in data.types)
    ):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: bodo.libs.str_arr_ext.str_arr_from_sequence(data)
        )  # pragma: no cover

    # Return data if already an array and nullable
    if not isinstance(data, types.Array) and bodo.utils.utils.is_array_typ(data, False):
        return (
            lambda data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True: data
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
            dict_encode=True,
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
            dict_encode=True,
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
        if not is_overload_constant_bool(dict_encode):
            raise BodoError("dict_code must be a constant bool value")
        else:
            dict_encode = get_overload_const_bool(dict_encode)

        if dict_encode:

            def impl_str(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
                dict_encode=True,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                # Use str_arr_from_sequence to force rep/avoid equiv_set
                dict_arr = bodo.libs.str_arr_ext.str_arr_from_sequence([data])
                indices = bodo.libs.int_arr_ext.alloc_int_array(n, np.int32)
                numba.parfors.parfor.init_prange()
                for i in numba.parfors.parfor.internal_prange(n):
                    indices[i] = 0
                A = bodo.libs.dict_arr_ext.init_dict_arr(
                    dict_arr, indices, True, True, None
                )
                return A

        else:

            def impl_str(
                data,
                error_on_nonarray=True,
                use_nullable_array=None,
                scalar_to_arr_len=None,
                dict_encode=True,
            ):  # pragma: no cover
                n = scalar_to_arr_len
                A = bodo.libs.str_arr_ext.pre_alloc_string_array(
                    n, get_utf8_size(data) * n
                )
                for i in range(n):
                    A[i] = data
                return A

        return impl_str

    if not is_overload_none(scalar_to_arr_len) and data == bodo.types.bytes_type:

        def impl_bytes(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            n = scalar_to_arr_len
            A = bodo.libs.binary_arr_ext.pre_alloc_binary_array(n, len(data) * n)
            for i in range(n):
                A[i] = data
            return A

        return impl_bytes

    # Convert list of Timestamps to dt64 array
    if isinstance(data, types.List) and isinstance(
        data.dtype, bodo.hiframes.pd_timestamp_ext.PandasTimestampType
    ):

        def impl_list_timestamp(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            n = len(data)
            A = np.empty(n, np.dtype("datetime64[ns]"))
            for i in range(n):
                A[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(data[i].value)
            return A

        return impl_list_timestamp

    # Convert list of Timedeltas to td64 array
    if isinstance(data, types.List) and data.dtype == bodo.types.pd_timedelta_type:

        def impl_list_timedelta(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
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
            dict_encode=True,
        ):  # pragma: no cover
            A = np.empty(scalar_to_arr_len, "datetime64[ns]")
            dt64_val = data.to_datetime64()
            null_bitmap = np.full(
                bodo.utils.indexing.bitmap_size(scalar_to_arr_len),
                bodo.utils.indexing.get_dt64_bitmap_fill(dt64_val),
                dtype=np.uint8,
            )
            for i in numba.parfors.parfor.internal_prange(scalar_to_arr_len):
                A[i] = dt64_val
            return bodo.libs.pd_datetime_arr_ext.init_datetime_array(
                A, null_bitmap, tz_literal
            )

        return impl_timestamp_tz_aware

    # Timestamp/Timedelta scalars to array
    if not is_overload_none(scalar_to_arr_len) and data in [
        bodo.types.pd_timestamp_tz_naive_type,
        bodo.types.pd_timedelta_type,
    ]:
        _dtype = (
            "datetime64[ns]"
            if data == bodo.types.pd_timestamp_tz_naive_type
            else "timedelta64[ns]"
        )

        def impl_timestamp(
            data,
            error_on_nonarray=True,
            use_nullable_array=None,
            scalar_to_arr_len=None,
            dict_encode=True,
        ):  # pragma: no cover
            n = scalar_to_arr_len
            # NOTE: not using n to calculate n_chars since distributed pass will use
            # the global value of n and cannot replace it with the local version
            A = np.empty(n, _dtype)
            data = bodo.utils.conversion.unbox_if_tz_naive_timestamp(data)
            for i in numba.parfors.parfor.internal_prange(n):
                A[i] = data
            return A

        return impl_timestamp

    # assuming can be ndarray
    return (
        lambda data,
        error_on_nonarray=True,
        use_nullable_array=None,
        scalar_to_arr_len=None,
        dict_encode=True: bodo.utils.conversion.coerce_to_ndarray(
            data, error_on_nonarray, use_nullable_array, scalar_to_arr_len
        )
    )  # pragma: no cover


@numba.generated_jit
def make_replicated_array(scalar, len):
    """
    A special wrapper for coerce_to_array that takes in a scalar of any type and coerces it to an
    array of specified length. This wrapper is recognized in distributed_analysis as a signal to force
    the output to be replicated (e.g. for the output of a no-groupby aggregation).

    Args:
        scalar (any): a scalar value that needs to be coerced to an array.
        len (integer): the number of rows the array should have.

    Returns:
        (any array): a replicated array containing the scalar input with the desired length.
    """

    def impl(scalar, len):
        return coerce_to_array(scalar, scalar_to_arr_len=len)

    return impl


def _is_str_dtype(dtype):
    """return True if 'dtype' specifies a string data type."""
    return (
        isinstance(dtype, bodo.libs.str_arr_ext.StringDtype)
        or (isinstance(dtype, types.Function) and dtype.key[0] is str)
        or (is_overload_constant_str(dtype) and get_overload_const_str(dtype) == "str")
        or (
            isinstance(dtype, types.TypeRef)
            and dtype.instance_type == types.unicode_type
        )
    )


# TODO: use generated_jit with IR inlining
def fix_arr_dtype(
    data, new_dtype, copy=None, nan_to_str=False, from_series=False
):  # pragma: no cover
    pass


@overload(fix_arr_dtype, no_unliteral=True)
def overload_fix_arr_dtype(
    data, new_dtype, copy=None, nan_to_str=False, from_series=False
):
    """convert data to new_dtype, copy if copy parameter is not None.
    'nan_to_str' specifies string conversion for NA values: write as '<NA>'
    or actual NA (Pandas has inconsistent behavior in APIs).

    'from_series' specifies if the data originates from a series. This is useful for some
    operations where the casting behavior changes depending on if the input is a Series
    or an Array (specifically, S.astype(str) vs S.values.astype(str))
    """
    data_is_tz_aware = isinstance(
        data.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
    )
    do_copy = is_overload_true(copy)

    # If the new dtype is "object", we treat it as a no-op.
    is_object = (
        is_overload_constant_str(new_dtype)
        and get_overload_const_str(new_dtype) == "object"
    )
    if is_overload_none(new_dtype) or is_object:
        if do_copy:
            return (
                lambda data,
                new_dtype,
                copy=None,
                nan_to_str=False,
                from_series=False: data.copy()
            )  # pragma: no cover
        return (
            lambda data, new_dtype, copy=None, nan_to_str=False, from_series=False: data
        )  # pragma: no cover

    # Handle nested types recursively:
    if isinstance(data, bodo.types.ArrayItemArrayType):
        nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)
        if not isinstance(nb_dtype, bodo.types.ArrayItemArrayType):
            raise BodoError(
                f"Both source and target types must be ArrayTimeArrayType! Got {data} and {nb_dtype} instead."
            )

        new_inner_type = bodo.utils.typing.get_castable_arr_dtype(nb_dtype.dtype)

        if not do_copy:
            # This still requires copying the inner data, but the null-bitmap and offsets
            # are re-used.
            def impl(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):  # pragma: no cover
                new_inner_data = bodo.utils.conversion.fix_arr_dtype(
                    bodo.libs.array_item_arr_ext.get_data(data),
                    new_inner_type,
                    copy,
                    nan_to_str,
                    from_series,
                )
                new_data = bodo.libs.array_item_arr_ext.init_array_item_array(
                    bodo.libs.array_item_arr_ext.get_n_arrays(data),
                    new_inner_data,
                    bodo.libs.array_item_arr_ext.get_offsets(data),
                    bodo.libs.array_item_arr_ext.get_null_bitmap(data),
                )
                return new_data

            return impl
        else:
            # This copies the null-bitmap and offsets in addition to the inner data
            # that will be copied during the cast.
            def impl(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):  # pragma: no cover
                new_inner_data = bodo.utils.conversion.fix_arr_dtype(
                    bodo.libs.array_item_arr_ext.get_data(data),
                    new_inner_type,
                    copy,
                    nan_to_str,
                    from_series,
                )
                new_data = bodo.libs.array_item_arr_ext.init_array_item_array(
                    bodo.libs.array_item_arr_ext.get_n_arrays(data),
                    new_inner_data,
                    bodo.libs.array_item_arr_ext.get_offsets(data).copy(),
                    bodo.libs.array_item_arr_ext.get_null_bitmap(data).copy(),
                )
                return new_data

            return impl
    elif isinstance(data, bodo.types.StructArrayType):
        nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)
        if not isinstance(nb_dtype, bodo.types.StructArrayType):
            raise BodoError(
                f"Both source and target types must be StructArrayType! Got {data} and {nb_dtype} instead."
            )

        if len(data.data) != len(nb_dtype.data):
            raise BodoError(
                f"Number of fields in the source ({len(data.data)}) and target ({len(nb_dtype.data)}) struct fields don't match!"
            )
        if data.names != nb_dtype.names:
            raise BodoError(
                f"Names of the fields in the source ({data.names}) and target ({nb_dtype.names}) struct fields don't match!"
            )

        n_fields = len(data.data)
        new_inner_types = tuple(
            [bodo.utils.typing.get_castable_arr_dtype(t) for t in nb_dtype.data]
        )
        field_names = data.names

        # Handle the 0 fields case.
        if n_fields == 0:
            if do_copy:
                return (
                    lambda data,
                    new_dtype,
                    copy=None,
                    nan_to_str=False,
                    from_series=False: data.copy()
                )  # pragma: no cover
            else:
                return (
                    lambda data,
                    new_dtype,
                    copy=None,
                    nan_to_str=False,
                    from_series=False: data
                )  # pragma: no cover

        call_id = next_label()
        func_text = "def impl(data, new_dtype, copy=None, nan_to_str=False, from_series=False):\n"
        func_text += "  inner_data_arrs = bodo.libs.struct_arr_ext.get_data(data)\n"
        for i in range(n_fields):
            # String types need to be constants and are therefore inlined.
            if isinstance(new_inner_types[i], str):
                func_text += f"  new_inner_data_arr_{i} = bodo.utils.conversion.fix_arr_dtype(inner_data_arrs[{i}], '{new_inner_types[i]}', copy, nan_to_str, from_series)\n"
            else:
                func_text += f"  new_inner_data_arr_{i} = bodo.utils.conversion.fix_arr_dtype(inner_data_arrs[{i}], new_inner_types_{call_id}[{i}], copy, nan_to_str, from_series)\n"

        new_data_tuple_str = "({},)".format(
            ", ".join([f"new_inner_data_arr_{i}" for i in range(n_fields)])
        )
        field_names_tuple_str = "({},)".format(
            ", ".join([f"'{f}'" for f in field_names])
        )
        copy_str = ".copy()" if do_copy else ""
        func_text += f"  new_data = bodo.libs.struct_arr_ext.init_struct_arr({n_fields}, {new_data_tuple_str}, bodo.libs.struct_arr_ext.get_null_bitmap(data){copy_str}, {field_names_tuple_str})\n"
        func_text += "  return new_data\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                f"new_inner_types_{call_id}": new_inner_types,
            },
            loc_vars,
        )
        return loc_vars["impl"]

    elif isinstance(data, bodo.types.MapArrayType):
        nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)

        if not isinstance(nb_dtype, bodo.types.MapArrayType):
            raise BodoError(
                f"Both source and target types must be MapArrayType! Got {data} and {nb_dtype} instead."
            )

        # Get the underlying ArrayItemArray type.
        new_underlying_type = bodo.libs.map_arr_ext._get_map_arr_data_type(nb_dtype)

        def impl(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            # Call it recursively on the underlying ArrayItemArray array.
            new_underlying_data = bodo.utils.conversion.fix_arr_dtype(
                data._data,
                new_underlying_type,
                copy,
                nan_to_str,
                from_series,
            )
            # Reconstruct the Map array from the new ArrayItemArray array.
            new_data = bodo.libs.map_arr_ext.init_map_arr(new_underlying_data)
            return new_data

        return impl

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
            bodo.types.datetime64ns: pd.Timestamp(0),
            bodo.types.timedelta64ns: pd.Timedelta(0),
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
            bodo.types.datetime64ns: pd.to_datetime,
            bodo.types.timedelta64ns: pd.to_timedelta,
        }

        # If NA values properly done this should suffice for default_value_dict:
        # default_value_dict = {typ: func(0) for typ, func in convert_func_dict.items()}

        valid_types = default_value_dict.keys()
        scalar_types = list(data._tuple_typ.types)

        if nb_dtype not in valid_types:
            raise BodoError(f"type conversion to {nb_dtype} types unsupported.")
        for typ in scalar_types:
            if typ == bodo.types.datetime64ns:
                if nb_dtype not in (
                    types.unicode_type,
                    types.int64,
                    types.uint64,
                    bodo.types.datetime64ns,
                ):
                    raise BodoError(
                        f"invalid type conversion from {typ} to {nb_dtype}."
                    )
            elif typ == bodo.types.timedelta64ns:
                if nb_dtype not in (
                    types.unicode_type,
                    types.int64,
                    types.uint64,
                    bodo.types.timedelta64ns,
                ):
                    raise BodoError(
                        f"invalid type conversion from {typ} to {nb_dtype}."
                    )

        func_text = "def impl(data, new_dtype, copy=None, nan_to_str=False, from_series=False):\n"
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

    # null array input case
    if data == bodo.types.null_array_type:

        def impl_null_array(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            return data.astype(new_dtype)

        return impl_null_array

    # convert to string
    if _is_str_dtype(new_dtype):
        #
        # special optimized case for int to string conversion, uses inplace write to
        # string array to avoid extra allocation
        if isinstance(data.dtype, types.Integer):

            def impl_int_str(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
            bodo.types.datetime64ns,
            bodo.types.timedelta64ns,
        ):

            def impl_str_dt_series(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(data)
                A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
                for j in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(data, j):
                        if nan_to_str:
                            A[j] = "<NA>"
                        else:
                            bodo.libs.array_kernels.setna(A, j)
                        continue

                    A[j] = str(data[j])

                return A

            return impl_str_array

    # convert to Categorical with predefined CategoricalDtype
    if isinstance(new_dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):

        def impl_cat_dtype(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
                val = bodo.utils.conversion.unbox_if_tz_naive_timestamp(data[i])
                codes[i] = label_dict[val]

            return A

        return impl_category

    nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)

    if isinstance(data.dtype, types.Integer) and isinstance(
        nb_dtype, bodo.types.Decimal128Type
    ):
        new_prec = nb_dtype.precision
        new_scale = nb_dtype.scale

        def impl_int_to_decimal(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            dec_arr = bodo.libs.decimal_arr_ext.int_to_decimal(data)
            return bodo.libs.decimal_arr_ext.cast_decimal_to_decimal_array(
                dec_arr, new_prec, new_scale, False
            )

        return impl_int_to_decimal

    # Matching data case
    if isinstance(data, bodo.libs.int_arr_ext.IntegerArrayType):
        same_typ = (
            isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype)
            and data.dtype == nb_dtype.dtype
        )
    elif isinstance(data, bodo.libs.float_arr_ext.FloatingArrayType):
        same_typ = (
            isinstance(nb_dtype, bodo.libs.float_arr_ext.FloatDtype)
            and data.dtype == nb_dtype.dtype
        )
    elif data == bodo.types.boolean_array_type:
        same_typ = nb_dtype == boolean_dtype
    elif bodo.utils.utils.is_array_typ(nb_dtype, False):
        same_typ = data == nb_dtype
    else:
        same_typ = data.dtype == nb_dtype

    if do_copy and same_typ:
        return (
            lambda data,
            new_dtype,
            copy=None,
            nan_to_str=False,
            from_series=False: data.copy()
        )  # pragma: no cover

    if same_typ:
        return (
            lambda data, new_dtype, copy=None, nan_to_str=False, from_series=False: data
        )  # pragma: no cover

    # nullable int array case
    if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
        _dtype = nb_dtype.dtype

        if isinstance(data.dtype, types.Float):

            def impl_float(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
            if data == bodo.types.dict_str_arr_type:

                def impl_dict(
                    data, new_dtype, copy=None, nan_to_str=False, from_series=False
                ):
                    return bodo.libs.dict_arr_ext.convert_dict_arr_to_int(data, _dtype)

                return impl_dict

            # perform specific time cast
            if isinstance(data, bodo.hiframes.time_ext.TimeArrayType):

                def impl(
                    data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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

    # nullable float array case
    if isinstance(nb_dtype, bodo.libs.float_arr_ext.FloatDtype):
        _dtype = nb_dtype.dtype

        def impl(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            n = len(data)
            numba.parfors.parfor.init_prange()
            B = bodo.libs.float_arr_ext.alloc_float_array(n, _dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(B, i)
                else:
                    B[i] = float(data[i])
            return B

        return impl

    # nullable int array to non-nullable int array case
    if isinstance(nb_dtype, types.Integer) and isinstance(data.dtype, types.Integer):

        def impl(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            return data.astype(nb_dtype)

        return impl

    # nullable float array to non-nullable int array case
    if isinstance(nb_dtype, types.Float) and isinstance(data.dtype, types.Float):

        def impl(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            return data.astype(nb_dtype)

        return impl

    # nullable bool array case
    if nb_dtype == bodo.libs.bool_arr_ext.boolean_dtype:

        def impl_bool(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
    if nb_dtype == bodo.types.datetime_date_type and (
        data.dtype == bodo.types.datetime64ns or data_is_tz_aware
    ):
        # This operation isn't defined in Pandas, so we opt to implement it as
        # truncating to the date, which best resembles a cast.
        convert_tz = data_is_tz_aware and data.dtype.tz is not None

        def impl_date(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            n = len(data)
            out_arr = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
                else:
                    if convert_tz:
                        out_arr[i] = bodo.utils.conversion.box_if_dt64(
                            data[i].tz_convert(None)
                        ).date()
                    else:
                        out_arr[i] = bodo.utils.conversion.box_if_dt64(data[i]).date()
            return out_arr

        return impl_date

    # Datetime64 case
    if nb_dtype == bodo.types.datetime64ns:
        if data.dtype == bodo.types.string_type:
            # Support String Arrays using objmode
            def impl_str(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):
                # Keep the objmode in a separate function for
                # inlining purposes.
                return bodo.hiframes.pd_timestamp_ext.series_str_dt64_astype(data)

            return impl_str

        if data == bodo.types.datetime_date_array_type:
            # Support Date Arrays using objmode
            # TODO: Replace with a native impl
            def impl_date(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):  # pragma: no cover
                return bodo.hiframes.pd_timestamp_ext.datetime_date_arr_to_dt64_arr(
                    data
                )

            return impl_date

        if data_is_tz_aware:

            def impl_tz_ts(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):
                return data.tz_convert(None)

            return impl_tz_ts

        if isinstance(data.dtype, types.Number) or data.dtype in [
            bodo.types.timedelta64ns,
            types.bool_,
        ]:
            # Nullable Integer/boolean/timedelta64 arrays
            def impl_numeric(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
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
    if nb_dtype == bodo.types.timedelta64ns:
        if data.dtype == bodo.types.string_type:
            # Support String Arrays using objmode
            def impl_str(
                data, new_dtype, copy=None, nan_to_str=False, from_series=False
            ):
                # Keep the objmode in a separate function for
                # inlining purposes.
                return bodo.hiframes.pd_timestamp_ext.series_str_td64_astype(data)

            return impl_str

        if isinstance(data.dtype, types.Number) or data.dtype in [
            bodo.types.datetime64ns,
            types.bool_,
        ]:
            if do_copy:
                # Nullable Integer/boolean/datetime64 arrays
                def impl_numeric(
                    data, new_dtype, copy=None, nan_to_str=False, from_series=False
                ):  # pragma: no cover
                    n = len(data)
                    numba.parfors.parfor.init_prange()
                    out_arr = np.empty(n, dtype=np.dtype("timedelta64[ns]"))
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(data, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = (
                                bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                                    np.int64(data[i])
                                )
                            )
                    return out_arr

                return impl_numeric

            else:
                return (
                    lambda data,
                    new_dtype,
                    copy=None,
                    nan_to_str=False,
                    from_series=False: data.view("int64")
                )  # pragma: no cover

    # Pandas currently only supports dt64/td64 -> int64
    if (nb_dtype == types.int64) and (
        data.dtype
        in [
            bodo.types.datetime64ns,
            bodo.types.timedelta64ns,
            bodo.types.pd_timedelta_type,
        ]
        or data_is_tz_aware
    ):
        unbox_value = data_is_tz_aware or data.dtype == bodo.types.pd_timedelta_type

        def impl_datelike_to_integer(
            data, new_dtype, copy=None, nan_to_str=False, from_series=False
        ):  # pragma: no cover
            n = len(data)
            numba.parfors.parfor.init_prange()
            A = np.empty(n, types.int64)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(data, i):
                    bodo.libs.array_kernels.setna(A, i)
                else:
                    if unbox_value:
                        A[i] = np.int64(data[i].value)
                    else:
                        A[i] = np.int64(data[i])
            return A

        return impl_datelike_to_integer

    if data.dtype != nb_dtype:
        return (
            lambda data,
            new_dtype,
            copy=None,
            nan_to_str=False,
            from_series=False: data.astype(nb_dtype)
        )  # pragma: no cover

    raise BodoError(f"Conversion from {data} to {new_dtype} not supported yet")


def array_type_from_dtype(dtype):
    return dtype_to_array_type(bodo.utils.typing.parse_dtype(dtype))


@overload(array_type_from_dtype, jit_options={"cache": True})
def overload_array_type_from_dtype(dtype):
    """parse dtype and return corresponding array type TypeRef"""
    arr_type = dtype_to_array_type(bodo.utils.typing.parse_dtype(dtype))
    return lambda dtype: arr_type  # pragma: no cover


@numba.njit
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


@overload(parse_datetimes_from_strings, no_unliteral=True, jit_options={"cache": True})
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


@overload(convert_to_dt64ns, no_unliteral=True, jit_options={"cache": True})
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

    if is_np_arr_typ(data, types.NPDatetime("ns")) or isinstance(
        data, bodo.types.DatetimeArrayType
    ):
        return lambda data: data  # pragma: no cover

    if is_str_arr_type(data):
        return lambda data: bodo.utils.conversion.parse_datetimes_from_strings(
            data
        )  # pragma: no cover

    raise BodoError(f"invalid data type {data} for dt64 conversion")


# TODO: use generated_jit with IR inlining
def convert_to_td64ns(data):  # pragma: no cover
    return data


@overload(convert_to_td64ns, no_unliteral=True, jit_options={"cache": True})
def overload_convert_to_td64ns(data):
    """Converts data formats like int64 to timedelta64ns"""
    # TODO: array of strings
    # see pd.core.arrays.timedeltas.sequence_to_td64ns for constructor types
    # TODO: support datetime.timedelta
    if is_np_arr_typ(data, types.int64):
        return lambda data: data.view(
            bodo.utils.conversion.TD_DTYPE
        )  # pragma: no cover

    if (
        is_np_arr_typ(data, types.NPTimedelta("ns"))
        or data == bodo.types.timedelta_array_type
    ):
        return lambda data: data  # pragma: no cover

    if is_str_arr_type(data):
        # TODO: support
        raise BodoError("conversion to timedelta from string not supported yet")

    raise BodoError(f"invalid data type {data} for timedelta64 conversion")


def convert_to_index(data, name=None):  # pragma: no cover
    return data


@overload(convert_to_index, no_unliteral=True, jit_options={"cache": True})
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


@overload(force_convert_index, no_unliteral=True, jit_options={"cache": True})
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


@overload(index_from_array, no_unliteral=True, jit_options={"cache": True})
def overload_index_from_array(data, name=None):
    """
    convert data array to Index object.
    """
    if data in [
        bodo.types.string_array_type,
        bodo.types.binary_array_type,
        bodo.types.dict_str_arr_type,
    ]:
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_binary_str_index(
            data, name
        )  # pragma: no cover

    if data.dtype == types.NPDatetime("ns"):
        return lambda data, name=None: pd.DatetimeIndex(
            data, name=name
        )  # pragma: no cover

    if data.dtype in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type):
        return lambda data, name=None: pd.TimedeltaIndex(
            data, name=name
        )  # pragma: no cover

    if (
        isinstance(
            data.dtype,
            (
                types.Integer,
                types.Float,
                types.Boolean,
                bodo.types.TimeType,
                bodo.types.Decimal128Type,
            ),
        )
        or data.dtype == bodo.types.datetime_date_type
    ):
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


@overload(index_to_array, no_unliteral=True, jit_options={"cache": True})
def overload_index_to_array(I):
    """
    convert Index object to data array.
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType

    if isinstance(I, RangeIndexType):
        return lambda I: np.arange(I._start, I._stop, I._step)  # pragma: no cover

    # other indices have data
    return lambda I: bodo.hiframes.pd_index_ext.get_index_data(I)  # pragma: no cover


def index_to_array_list(data):  # pragma: no cover
    return data


@overload(index_to_array_list, no_unliteral=True, jit_options={"cache": True})
def overload_index_to_array_list(I, gen_range_index_array=True):
    """
    Convert Index to a tuple of data array(s).
    gen_range_index_array is used to determine if an array should be generated for
    range Index or not.
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    if isinstance(I, MultiIndexType):
        return (
            lambda I,
            gen_range_index_array=True: bodo.hiframes.pd_index_ext.get_index_data(I)
        )  # pragma: no cover

    if (
        isinstance(I, RangeIndexType)
        and is_overload_constant_bool(gen_range_index_array)
        and not get_overload_const_bool(gen_range_index_array)
    ):
        return lambda I, gen_range_index_array=True: ()  # pragma: no cover

    return lambda I, gen_range_index_array=True: (
        bodo.utils.conversion.index_to_array(I),
    )  # pragma: no cover


def false_if_none(val):  # pragma: no cover
    return False if val is None else val


@overload(false_if_none, no_unliteral=True, jit_options={"cache": True})
def overload_false_if_none(val):
    """Return False if 'val' is None, otherwise same value"""

    if is_overload_none(val):
        return lambda val: False  # pragma: no cover

    return lambda val: val  # pragma: no cover


def extract_name_if_none(data, name):  # pragma: no cover
    return name


@overload(extract_name_if_none, no_unliteral=True, jit_options={"cache": True})
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


@overload(extract_index_if_none, no_unliteral=True, jit_options={"cache": True})
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
        0,
        len(data),
        1,
        None,  # pragma: no cover
    )


def box_if_dt64(val):  # pragma: no cover
    return val


@overload(box_if_dt64, no_unliteral=True, jit_options={"cache": True})
def overload_box_if_dt64(val):
    """If 'val' is dt64, box it to Timestamp otherwise just return 'val'"""
    if val == types.NPDatetime("ns"):
        return (
            lambda val: bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                val
            )
        )  # pragma: no cover

    if val == types.NPTimedelta("ns"):
        return (
            lambda val: bodo.hiframes.pd_timestamp_ext.convert_numpy_timedelta64_to_pd_timedelta(
                val
            )
        )  # pragma: no cover

    return lambda val: val  # pragma: no cover


def unbox_if_tz_naive_timestamp(val):  # pragma: no cover
    return val


@overload(unbox_if_tz_naive_timestamp, no_unliteral=True, jit_options={"cache": True})
def overload_unbox_if_tz_naive_timestamp(val):
    """If 'val' is Timestamp without a Timezone,
    "unbox" it to dt64 otherwise just return 'val'"""
    # unbox Timestamp to dt64
    if val == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type:
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
    if val == types.Optional(bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type):

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


@overload(to_tuple, no_unliteral=True, jit_options={"cache": True})
def overload_to_tuple(val):
    """convert tuple-like 'val' (e.g. constant list) to a tuple"""
    if not isinstance(val, types.BaseTuple) and is_overload_constant_list(val):
        # LiteralList values may be non-constant
        n_values = len(
            val.types
            if isinstance(val, types.LiteralList)
            else get_overload_const_list(val)
        )
        func_text = "def bodo_to_tuple(val):\n"
        res = ",".join(f"val[{i}]" for i in range(n_values))
        func_text += f"  return ({res},)\n"
        return bodo.utils.utils.bodo_exec(func_text, {}, {}, __name__)

    assert isinstance(val, types.BaseTuple), "tuple type expected"
    return lambda val: val  # pragma: no cover


def get_array_if_series_or_index(data):  # pragma: no cover
    return data


@overload(get_array_if_series_or_index, jit_options={"cache": True})
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


@overload(extract_index_array, no_unliteral=True, jit_options={"cache": True})
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


@overload(ensure_contig_if_np, no_unliteral=True, jit_options={"cache": True})
def overload_ensure_contig_if_np(arr):
    """make sure array 'arr' is contiguous in memory if it is a numpy array.
    Other arrays are always contiguous.
    """
    if isinstance(arr, types.Array):
        return lambda arr: np.ascontiguousarray(arr)  # pragma: no cover

    return lambda arr: arr  # pragma: no cover


def struct_if_heter_dict(values, names):  # pragma: no cover
    return dict(zip(names, values))


@overload(struct_if_heter_dict, no_unliteral=True, jit_options={"cache": True})
def overload_struct_if_heter_dict(values, names):
    """returns a struct with fields names 'names' and data 'values' if value types are
    heterogeneous, otherwise a regular dict.
    """

    if not types.is_homogeneous(*values.types):
        return lambda values, names: bodo.libs.struct_arr_ext.init_struct(
            values, names
        )  # pragma: no cover

    n_fields = len(values.types)
    func_text = "def bodo_struct_if_heter_dict(values, names):\n"
    res = ",".join(
        f"'{get_overload_const_str(names.types[i])}': values[{i}]"
        for i in range(n_fields)
    )
    func_text += f"  return {{{res}}}\n"
    return bodo.utils.utils.bodo_exec(func_text, {}, {}, __name__)


def list_to_array(lst, arr_type, parallel=False):
    pass


@overload(list_to_array)
def overload_list_to_array(lst, arr_type, parallel=False):
    """Converts the contents of the provided list to an array with the given
    type. This function is capable of converting a list to either a replicated
    or 1D distributed result, depending on the output of distributed analysis. Since
    lists are always replicated, via the parallel flag and then statically calculating
    our chunk.

    This kernel cannot output 1DVar results because there is not enough information to
    determine size equivalence between this and another array. Please do not attempt to use
    this kernel in this setting (although there is a check in distributed analysis).

    Args:
        lst (types.List): The list to convert to an array. It may have optional types
            if it is used in context that creates list literals (e.g. Values in SQL).
        arr_type (types.TypeRef[types.ArrayType]): The desired output array type.
        parallel (types.boolean): Is the result 1D or replicated. This is set automatically
            by the compiler.
    Returns:
        An array containing a subset of the replicate list's elements. This array may
        be replicated or distributed.
    """
    func_text = "def impl(lst, arr_type, parallel=False):\n"
    func_text += "  global_len = len(lst)\n"
    func_text += "  if parallel:\n"
    func_text += "    n_pes = bodo.get_size()\n"
    func_text += "    rank = bodo.get_rank()\n"
    func_text += (
        "    start = bodo.libs.distributed_api.get_start(global_len, n_pes, rank)\n"
    )
    func_text += "    copy_len = bodo.libs.distributed_api.get_node_portion(global_len, n_pes, rank)\n"
    func_text += "  else:\n"
    func_text += "    start, copy_len = 0, global_len\n"

    # If we were supplied unknown as the type, then we should instead use the
    # type of the list as the type to construct.
    real_arr_type = arr_type
    if unwrap_typeref(arr_type) == types.unknown:
        real_arr_type = dtype_to_array_type(to_nullable_type(lst.dtype))
    glbls = {"bodo": bodo, "real_arr_type": real_arr_type}

    if arr_type == bodo.types.dict_str_arr_type:
        # For dictionary encoded arrays create a naive array containing duplicates.
        glbls["data_arr_type"] = bodo.types.string_array_type
        glbls["indices_arr_type"] = bodo.libs.dict_arr_ext.dict_indices_arr_type
        func_text += "  data_arr = bodo.utils.conversion.list_to_dict_array(lst, data_arr_type)\n"
        func_text += "  indices_arr = bodo.utils.utils.alloc_type(copy_len, indices_arr_type, (-1,))\n"
    else:
        func_text += (
            "  out_arr = bodo.utils.utils.alloc_type(copy_len, real_arr_type, (-1,))\n"
        )
    func_text += "  for i in range(start, start + copy_len):\n"
    if arr_type == bodo.types.dict_str_arr_type:
        func_text += "    indices_arr[i - start] = i\n"
        # Ensure nulls are consistent. This extra pass is fine because we assume lists are small (e.g.
        # used by VALUES in SQL).
        func_text += "  for j in range(copy_len):\n"
        func_text += "    if bodo.libs.array_kernels.isna(data_arr, j):\n"
        func_text += "      bodo.libs.array_kernels.setna(indices_arr, j):\n"
        func_text += "  out_arr = bodo.libs.dict_arr_ext.init_dict_arr(data_arr, indices_arr, True, False, None)\n"
    else:
        func_text += "    out_arr[i - start] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(lst[i])\n"
    func_text += "  return out_arr\n"
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]
