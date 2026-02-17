"""DatetimeArray extension for Pandas DatetimeArray with timezone support."""

from __future__ import annotations

import datetime
import operator
import zoneinfo
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs.str_arr_ext import null_bitmap_arr_type
from bodo.utils.conversion import ensure_contig_if_np
from bodo.utils.indexing import bitmap_size
from bodo.utils.typing import (
    BodoArrayIterator,
    BodoError,
    get_literal_value,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_constant_str,
)


@register_jitable
def build_dt_valid_bitmap(ts: npt.NDArray[np.datetime64]):  # pragma: no cover
    nbytes = bitmap_size(len(ts))
    nulls_arr = np.empty(nbytes, np.uint8)

    for i, s in enumerate(ts):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
    return nulls_arr


class PandasDatetimeTZDtype(types.Type):
    """Data type for datetime timezone"""

    def __init__(self, tz):
        if isinstance(
            tz,
            (
                pytz._FixedOffset,
                pytz.tzinfo.BaseTzInfo,
                datetime.timezone,
                zoneinfo.ZoneInfo,
            ),
        ):
            tz = get_tz_type_info(tz)

        if not isinstance(tz, (int, str)) and tz is not None:
            raise BodoError(
                "Timezone must be either a valid pytz type with a zone, a fixed offset, or None"
            )
        self.tz = tz
        super().__init__(name=f"PandasDatetimeTZDtype[{tz}]")


pd_datetime_tz_naive_type = PandasDatetimeTZDtype(None)

register_model(PandasDatetimeTZDtype)(models.OpaqueModel)


@lower_constant(PandasDatetimeTZDtype)
def lower_constant_pd_datetime_tz_dtype(context, builder, typ, pyval):
    return context.get_dummy_value()


@box(PandasDatetimeTZDtype)
def box_pd_datetime_tzdtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)
    # Create the timezone type.
    unit_str = c.context.get_constant_generic(c.builder, types.unicode_type, "ns")
    # No need to incref because unit_str is a constant
    unit_str_obj = c.pyapi.from_native_value(
        types.unicode_type, unit_str, c.env_manager
    )
    if isinstance(typ.tz, str):
        tz_str = c.context.get_constant_generic(c.builder, types.unicode_type, typ.tz)
        # No need to incref because tz_str is a constant
        tz_arg_obj = c.pyapi.from_native_value(
            types.unicode_type, tz_str, c.env_manager
        )
    elif isinstance(typ.tz, int):
        # We store ns, but the Fixed offset constructor takes minutes.
        offset = nanoseconds_to_offset(typ.tz)
        tz_arg_obj = c.pyapi.unserialize(c.pyapi.serialize_object(offset))
    else:
        tz_arg_obj = None

    if tz_arg_obj is not None:
        res = c.pyapi.call_method(
            pd_class_obj, "DatetimeTZDtype", (unit_str_obj, tz_arg_obj)
        )
        c.pyapi.decref(tz_arg_obj)
    else:
        res = c.pyapi.make_none()

    c.pyapi.decref(unit_str_obj)
    c.pyapi.decref(pd_class_obj)
    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(PandasDatetimeTZDtype)
def unbox_pd_datetime_tzdtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


@typeof_impl.register(pd.DatetimeTZDtype)
def typeof_pd_int_dtype(val, c):
    return PandasDatetimeTZDtype(val.tz)


def get_tz_type_info(tz_type):
    """
    Extracts the information used by Bodo when encountering a pytz or datetime.timezone
    type. This obtains the string name of the zone for most timezones,
    but for FixedOffsets it outputs an integer in nanoseconds.
    """
    if tz_type is datetime.timezone.utc:
        tz_val = "UTC"
    elif isinstance(tz_type, datetime.timezone):
        tz_val = pd.Timedelta(tz_type.utcoffset(None)).value
    elif isinstance(tz_type, zoneinfo.ZoneInfo):
        tz_val = tz_type.key
    elif isinstance(tz_type, pytz._FixedOffset):
        # If we have a fixed offset represent it as an integer
        # offset in ns.
        # Note: tz_type._offset is a np.timedelta.
        tz_val = pd.Timedelta(tz_type._offset).value
    else:
        tz_val = tz_type.zone
        if tz_val not in pytz.all_timezones_set:
            raise BodoError(
                "Unsupported timezone type. Timezones must be a fixedOffset or contain a zone found in pytz.all_timezones"
            )
    return tz_val


def python_timezone_from_bodo_timezone_info(tz_value: int | str | None) -> Any:
    """
    Convert the Bodo internal typing representation of a timezone which is either
    an int, string, or None to a python timezone object.

    Args:
        tz_value (Union[int, str, None]): The Bodo internal representation of a timezone.

    Returns:
        Any: An actual timezone type that can be used in Python at compile time.
    """
    if isinstance(tz_value, int):
        return nanoseconds_to_offset(tz_value)
    elif isinstance(tz_value, str):
        return pytz.timezone(tz_value)
    else:
        return None


def nanoseconds_to_offset(nanoseconds):
    """
    Converts a number of nanoseconds to the appropriate pytz.Offset type.
    """
    num_mins = nanoseconds // (60 * 1000 * 1000 * 1000)
    return pytz.FixedOffset(num_mins)


class DatetimeArrayType(types.IterableType, types.ArrayCompatible):
    """Data type for datetime array with timezones"""

    def __init__(self, tz):
        if isinstance(
            tz,
            (
                pytz._FixedOffset,
                pytz.tzinfo.BaseTzInfo,
                datetime.timezone,
                zoneinfo.ZoneInfo,
            ),
        ):
            tz = get_tz_type_info(tz)

        if not isinstance(tz, (int, str)) and tz is not None:
            raise BodoError(
                "Timezone must be either a valid pytz type with a zone, a fixed offset, or None"
            )
        self.tz = tz
        self._data_array_type = types.Array(types.NPDatetime("ns"), 1, "C")
        self._dtype = PandasDatetimeTZDtype(tz)
        super().__init__(name=f"DatetimeArrayType('{tz}')")

    @property
    def data_array_type(self):
        return self._data_array_type

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)

    @property
    def dtype(self):
        return self._dtype

    def copy(self):
        return DatetimeArrayType(self.tz)


@register_model(DatetimeArrayType)
class PandasDatetimeArrayModel(models.StructModel):
    """Datetime array model, storing datetime64 array and timezone"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data_array_type),
            ("null_bitmap", null_bitmap_arr_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DatetimeArrayType, "data", "_data")
make_attribute_wrapper(DatetimeArrayType, "null_bitmap", "_null_bitmap")


@typeof_impl.register(pd.arrays.DatetimeArray)
def typeof_pd_datetime_array(val, c):
    return DatetimeArrayType(val.tz)


@unbox(DatetimeArrayType)
def unbox_pd_datetime_array(typ, val, c):
    return bodo.libs.array.unbox_array_using_arrow(typ, val, c)


@box(DatetimeArrayType)
def box_pd_datetime_array(typ, val, c):
    """
    We box a the datetime array by extracting the object for the data,
    creating a DatetimeTZDtype from the type string, and finally by
    calling the pandas.arrays.DatetimeArray constructor.
    """
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@intrinsic(prefer_literal=True)
def init_datetime_array(typingctx, data, null_bitmap, tz):
    """
    Initialize a pandas.arrays.DatetimeArray.
    """

    def codegen(context, builder, sig, args):
        data, null_bitmap, tz = args

        pd_dt_arr = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        pd_dt_arr.data = data
        pd_dt_arr.null_bitmap = null_bitmap

        context.nrt.incref(builder, sig.args[0], data)
        context.nrt.incref(builder, sig.args[1], null_bitmap)

        return pd_dt_arr._getvalue()

    if is_overload_constant_str(tz) or is_overload_constant_int(tz):
        tz_str = get_literal_value(tz)
    elif tz is types.none:
        tz_str = None
    else:
        raise BodoError("tz must be a constant string or Fixed Offset")

    return_type = DatetimeArrayType(tz_str)
    sig = return_type(return_type.data_array_type, null_bitmap_arr_type, tz)

    return sig, codegen


def init_datetime_array_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 3 and not kws, "invalid arguments in init_datetime_array_equiv"
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_pd_datetime_arr_ext_init_datetime_array = (
    init_datetime_array_equiv
)


def alias_ext_init_datetime_array(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 3, "invalid arguments in alias_ext_init_datetime_array"
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_datetime_array", "bodo.libs.pd_datetime_arr_ext")
] = alias_ext_init_datetime_array


# high-level allocation function for tz-aware arrays arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_pd_datetime_array(n, tz):  # pragma: no cover
    data_arr = np.empty(n, dtype="datetime64[ns]")
    null_bitmap_arr = np.empty(bitmap_size(n), dtype=np.uint8)
    return init_datetime_array(data_arr, null_bitmap_arr, tz)


def alloc_pd_datetime_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_pd_datetime_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_pd_datetime_arr_ext_alloc_pd_datetime_array = (
    alloc_pd_datetime_array_equiv
)


@overload(len, no_unliteral=True)
def overload_pd_datetime_arr_len(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@lower_constant(DatetimeArrayType)
def lower_constant_pd_datetime_arr(context, builder, typ, pyval):
    npval = pyval.to_numpy("datetime64[ns]")
    numpy_data = context.get_constant_generic(builder, typ.data_array_type, npval)
    null_bitmap = np.packbits(np.logical_not(np.isnat(npval)), None, "little")
    null_bitmap_data = context.get_constant_generic(
        builder, null_bitmap_arr_type, null_bitmap
    )
    datetime_arr_val = lir.Constant.literal_struct([numpy_data, null_bitmap_data])
    return datetime_arr_val


@overload_attribute(DatetimeArrayType, "shape")
def overload_pd_datetime_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(DatetimeArrayType, "nbytes")
def overload_pd_datetime_arr_nbytes(A):
    return lambda A: A._data.nbytes  # pragma: no cover


@overload_method(DatetimeArrayType, "tz_convert", no_unliteral=True)
def overload_pd_datetime_tz_convert(A, tz):
    if tz == types.none:
        # Note this differs from Pandas in the output type.
        # Pandas would still have a DatetimeArrayType with no timezone
        # but we always represent no timezone as datetime64 array.
        def impl(A, tz):  # pragma: no cover
            return A._data.copy()

        return impl

    else:

        def impl(A, tz):  # pragma: no cover
            return init_datetime_array(A._data.copy(), A._null_bitmap.copy(), tz)

    return impl


@overload_method(DatetimeArrayType, "copy", no_unliteral=True)
def overload_pd_datetime_tz_convert(A):
    tz = A.tz

    def impl(A):  # pragma: no cover
        return init_datetime_array(A._data.copy(), A._null_bitmap.copy(), tz)

    return impl


@overload_attribute(DatetimeArrayType, "dtype", no_unliteral=True)
def overload_pd_datetime_dtype(A):
    tz = A.tz

    # Replicate pandas behavior, which returns numpy's datetime64[ns] when tz is None
    # Note that getitem still returns PandasTimestampTypes
    if A.tz is None:

        def impl(A):  # pragma: no cover
            return bodo.types.datetime64ns

        return impl
    else:
        dtype = pd.DatetimeTZDtype("ns", tz)

        def impl(A):  # pragma: no cover
            return dtype

        return impl


@overload(operator.getitem, no_unliteral=True)
def overload_getitem(A, ind):
    if not isinstance(A, DatetimeArrayType):
        return
    tz = A.tz
    if isinstance(ind, types.Integer):

        def impl(A, ind):  # pragma: no cover
            return bodo.hiframes.pd_timestamp_ext.convert_val_to_timestamp(
                bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A._data[ind]),
                tz,
            )

        return impl

    # bool arr indexing. Note nullable boolean arrays are handled in
    # bool_arr_ind_getitem to ensure NAs are converted to False.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_array(ind)
            new_data = ensure_contig_if_np(A._data[ind])
            null_bitmap = ensure_contig_if_np(build_dt_valid_bitmap(new_data))
            return init_datetime_array(new_data, null_bitmap, tz)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl_int_arr(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_array(ind)
            new_data = ensure_contig_if_np(A._data[ind])
            null_bitmap = ensure_contig_if_np(build_dt_valid_bitmap(new_data))
            return init_datetime_array(new_data, null_bitmap, tz)

        return impl_int_arr

    # slice indexing
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data = ensure_contig_if_np(A._data[ind])
            null_bitmap = ensure_contig_if_np(build_dt_valid_bitmap(new_data))
            return init_datetime_array(new_data, null_bitmap, tz)

        return impl_slice

    # This should be the only DatetimeArray implementation.
    raise BodoError(
        "operator.getitem with DatetimeArrayType is only supported with an integer index, int arr, boolean array, or slice."
    )  # pragma: no cover


def timestamp_to_dt64(val):
    return val


@overload(timestamp_to_dt64)
def overload_timestamp_to_dt64(val):
    if isinstance(val, bodo.types.PandasTimestampType):

        def impl(val):  # pragma: no cover
            return bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val.value)

        return impl
    elif val == bodo.types.datetime64ns:  # pragma: no cover

        def impl(val):  # pragma: no cover
            return val

        return impl
    elif val == bodo.types.datetime_datetime_type:  # pragma: no cover

        def impl(val):  # pragma: no cover
            return bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)

        return impl
    else:
        raise BodoError("timestamp_to_dt64 requires a timestamp")


@overload(operator.setitem, no_unliteral=True)
def overload_setitem(A, ind, val):
    if not isinstance(A, DatetimeArrayType):
        return
    tz = A.tz

    # Check the possible values
    if not (
        isinstance(val, DatetimeArrayType)
        or isinstance(val, bodo.types.PandasTimestampType)
        or val == bodo.types.datetime64ns
        or val == bodo.types.datetime_datetime_type
        or val == types.Array(bodo.types.datetime64ns, 1, "C")
    ):  # pragma: no cover
        raise BodoError(
            "operator.setitem with DatetimeArrayType requires a Timestamp value or DatetimeArrayType"
        )

    is_tz_naive_dt = (
        isinstance(val, numba.core.types.scalars.NPDatetime)
        or val == bodo.types.datetime_datetime_type
        or val == types.Array(bodo.types.datetime64ns, 1, "C")
    )

    # Ensure the timezones match, and that, if a datetime 64, we have no time zone
    if not is_tz_naive_dt and val.tz != tz:  # pragma: no cover
        raise BodoError(
            "operator.setitem with DatetimeArrayType requires the Array and values to set to share a timezone"
        )
    elif is_tz_naive_dt and tz is not None:
        raise BodoError(
            "operator.setitem with tz-aware DatetimeArrayType requires timezone-aware values"
        )

    if isinstance(ind, types.Integer):
        if isinstance(val, bodo.types.PandasTimestampType):

            def impl(A, ind, val):  # pragma: no cover
                dt64_val = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val.value)
                A._data[ind] = dt64_val
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    A._null_bitmap, ind, 0 if np.isnat(dt64_val) else 1
                )

            return impl
        elif isinstance(val, numba.core.types.scalars.NPDatetime):

            def impl(A, ind, val):  # pragma: no cover
                dt64_val = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val)
                A._data[ind] = dt64_val
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    A._null_bitmap, ind, 0 if np.isnat(dt64_val) else 1
                )

            return impl
        elif val == bodo.types.datetime_datetime_type:

            def impl(A, ind, val):  # pragma: no cover
                dt64_val = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)
                A._data[ind] = dt64_val
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    A._null_bitmap, ind, 0 if np.isnat(dt64_val) else 1
                )

            return impl
        else:  # pragma: no cover
            raise BodoError(
                "operator.setitem with DatetimeArrayType requires a Timestamp value"
            )

    # array of int indices
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        if isinstance(val, DatetimeArrayType):
            # Array case
            def impl_arr(A, ind, val):  # pragma: no cover
                n = len(ind)
                for i in range(n):
                    A._data[ind[i]] = val._data[i]
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        A._null_bitmap, ind[i], 0 if np.isnat(val._data[i]) else 1
                    )

            return impl_arr

        else:
            # Scalar case
            def impl_scalar(A, ind, val):  # pragma: no cover
                value = timestamp_to_dt64(val)
                n = len(ind)
                valid = 0 if np.isnat(value) else 1
                for i in range(n):
                    A._data[ind[i]] = value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, ind[i], valid)

            return impl_scalar

    # bool array
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        if isinstance(val, DatetimeArrayType):
            # Array case
            def impl_arr(A, ind, val):  # pragma: no cover
                ind = bodo.utils.conversion.coerce_to_array(ind)
                val_ind = 0
                n = len(ind)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                        A._data[i] = val._data[val_ind]
                        bodo.libs.int_arr_ext.set_bit_to_arr(
                            A._null_bitmap, i, 0 if np.isnat(val._data[val_ind]) else 1
                        )
                        val_ind += 1

            return impl_arr

        else:
            # Scalar case
            def impl_scalar(A, ind, val):  # pragma: no cover
                value = timestamp_to_dt64(val)
                ind = bodo.utils.conversion.coerce_to_array(ind)
                n = len(ind)
                valid = 0 if np.isnat(value) else 1
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                        A._data[i] = value
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, valid)

            return impl_scalar

    # slice case
    if isinstance(ind, types.SliceType):
        if isinstance(val, DatetimeArrayType):
            # Array case
            def impl_arr(A, ind, val):  # pragma: no cover
                # using setitem directly instead of copying in loop since
                # Array setitem checks for memory overlap and copies source
                A._data[ind] = val._data
                n = len(A)
                slice_idx = numba.cpython.unicode._normalize_slice(ind, n)

                # Set the appropriate values in the null bitmap
                val_idx = 0
                val_null_bitmap = val._null_bitmap
                val_data = val._data
                arr_null_bitmap = A._null_bitmap
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        val_null_bitmap, val_idx
                    )
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        arr_null_bitmap,
                        i,
                        0 if np.isnat(val_data[val_idx]) else bit,
                    )
                    val_idx += 1

            return impl_arr

        elif val == types.Array(bodo.types.datetime64ns, 1, "C"):

            def impl_arr(A, ind, val):  # pragma: no cover
                # using setitem directly instead of copying in loop since
                # Array setitem checks for memory overlap and copies source
                A._data[ind] = val
                n = len(A)
                slice_idx = numba.cpython.unicode._normalize_slice(ind, n)

                # Set the appropriate values in the null bitmap
                val_idx = 0
                arr_null_bitmap = A._null_bitmap
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        arr_null_bitmap,
                        i,
                        0 if np.isnat(val[val_idx]) else 1,
                    )
                    val_idx += 1

            return impl_arr
        else:
            # Scalar case
            def impl_scalar(A, ind, val):  # pragma: no cover
                value = timestamp_to_dt64(val)
                slice_idx = numba.cpython.unicode._normalize_slice(ind, len(A))
                valid = 0 if np.isnat(value) else 1
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A._data[i] = value
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, valid)

            return impl_scalar

    raise BodoError(
        f"setitem for DatetimeArrayType with indexing type {ind} not supported"
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def numpy_arr_setitem(A, idx, val):
    """Support setitem of Numpy arrays with nullable datetime arrays"""
    if not (
        isinstance(A, types.Array)
        and (A.dtype == bodo.types.datetime64ns)
        and isinstance(val, DatetimeArrayType)
    ):
        return

    nat = bodo.types.datetime64ns("NaT")

    def impl_np_setitem_datetime_arr(A, idx, val):  # pragma: no cover
        # Make sure data elements of NA values are NaT to pass the NAs to output
        data = val._data
        bitmap = val._null_bitmap
        for i in range(len(val)):
            if not bodo.libs.int_arr_ext.get_bit_bitmap_arr(bitmap, i):
                data[i] = nat

        A[idx] = data

    return impl_np_setitem_datetime_arr


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def unwrap_tz_array(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: A._data  # pragma: no cover
    return lambda A: A  # pragma: no cover


# array analysis extension
def unwrap_tz_array_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_pd_datetime_arr_ext_unwrap_tz_array = (
    unwrap_tz_array_equiv
)


def create_cmp_op_overload_arr(op):
    """create overload function for comparison operators with pandas timezone aware datetime array"""
    # Import within the function to avoid circular imports
    from bodo.hiframes.pd_timestamp_ext import PandasTimestampType

    def overload_datetime_arr_cmp(lhs, rhs):
        if not (
            isinstance(lhs, DatetimeArrayType) or isinstance(rhs, DatetimeArrayType)
        ):  # pragma: no cover
            # This implementation only handles at least 1 DatetimeArrayType
            return

        # DatetimeArrayType + Scalar tz-aware or date
        if isinstance(lhs, DatetimeArrayType) and (
            isinstance(rhs, PandasTimestampType)
            or rhs == bodo.types.datetime_date_type
            or (rhs == bodo.types.datetime64ns)
        ):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs[i], rhs)
                return out_arr

            return impl

        # Scalar tz-aware or date + DatetimeArrayType.
        elif (
            isinstance(lhs, PandasTimestampType)
            or lhs == bodo.types.datetime_date_type
            or (lhs == bodo.types.datetime64ns)
        ) and isinstance(rhs, DatetimeArrayType):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs, rhs[i])
                return out_arr

            return impl

        # DatetimeArrayType or date array + DatetimeArrayType or date array
        elif (
            isinstance(lhs, DatetimeArrayType)
            or lhs == bodo.types.datetime_date_array_type
        ) and (
            isinstance(rhs, DatetimeArrayType)
            or rhs == bodo.types.datetime_date_array_type
        ):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        lhs, i
                    ) or bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs[i], rhs[i])
                return out_arr

            return impl

        # Tz-Aware timestamp + Tz-Naive timestamp
        elif (
            isinstance(lhs, DatetimeArrayType)
            and (isinstance(rhs, types.Array) and rhs.dtype == bodo.types.datetime64ns)
        ) or (
            (isinstance(lhs, types.Array) and lhs.dtype == bodo.types.datetime64ns)
            and isinstance(rhs, DatetimeArrayType)
        ):

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        lhs, i
                    ) or bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs[i], rhs[i])
                return out_arr

            return impl

    return overload_datetime_arr_cmp


def overload_add_operator_datetime_arr(lhs, rhs):
    """
    Implementation for the supported add operations on Timezone-Aware data.
    This function is called from an overload, so it returns an overload.
    This is used for lhs + rhs.

    Either lhs or rhs is assumed to be a DatetimeArrayType based on how this
    function is used.

    Args:
        lhs (types.Type): Bodo type to add. Either (DatetimeArrayType or week_type)
        rhs (types.Type): Bodo type to add. Either (DatetimeArrayType or week_type)

    Raises:
        BodoError: If operator.add is not supported between DatetimeArrayType and the other type.

    Returns:
        func: An implementation function that would be returned from an overload
    """
    if isinstance(lhs, DatetimeArrayType):
        # TODO: Support more types
        if rhs in (bodo.types.week_type, bodo.types.pd_timedelta_type):
            tz_literal = lhs.tz

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    n, tz_literal
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs[i] + rhs
                return out_arr

            return impl

        else:
            raise BodoError(
                f"add operator not supported between Timezone-aware timestamp and {rhs}. Please convert to timezone naive with ts.tz_convert(None)"
            )
    else:
        # Note this function is only called if at least one input is a DatetimeArrayType
        # TODO: Support more types
        if lhs in (bodo.types.week_type, bodo.types.pd_timedelta_type):
            tz_literal = rhs.tz

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    n, tz_literal
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs + rhs[i]
                return out_arr

            return impl

        else:
            raise BodoError(
                f"add operator not supported between {lhs} and Timezone-aware timestamp. Please convert to timezone naive with ts.tz_convert(None)"
            )


def overload_sub_operator_datetime_arr(lhs, rhs):
    if isinstance(lhs, DatetimeArrayType):
        # TODO: Support more types
        if rhs in (bodo.types.week_type, bodo.types.pd_timedelta_type):
            tz_literal = lhs.tz

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    n, tz_literal
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs[i] - rhs
                return out_arr

            return impl

        if isinstance(rhs, DatetimeArrayType) and rhs.tz == lhs.tz:

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.hiframes.datetime_timedelta_ext.alloc_timedelta_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        lhs, i
                    ) or bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs[i] - rhs[i]
                return out_arr

            return impl
        else:
            raise BodoError(
                f"sub operator not supported between Timezone-aware timestamp and {rhs}. Please convert to timezone naive with ts.tz_convert(None)"
            )


@register_jitable
def convert_months_offset_to_days(
    curr_year, curr_month, curr_day, num_months
):  # pragma: no cover
    """Converts the number of months to move forward from a current
    year, month, and day into a Timedelta with the appropriate number of days.
    This is used to convert a DateOffset of only months into an equivalent
    pd.Timedelta for us in BodoSQL array kernels

    Args:
        curr_year (types.int64): Current year number
        curr_month (types.int64): Current month number (1-12)
        curr_day (types.int64): Current day number (1-31)
        num_months (types.int64): Number of months to add (either + or -)
    """
    # Account for the 1-indexing in computing the new month
    month_total = (curr_month + num_months) - 1
    new_month = (month_total % 12) + 1
    num_years = month_total // 12
    new_year = curr_year + num_years
    # Make sure the day is still valid in this month, otherwise we truncate
    # to the last day of the month.
    max_day = bodo.hiframes.pd_timestamp_ext.get_days_in_month(new_year, new_month)
    new_day = min(max_day, curr_day)
    curr_ts = pd.Timestamp(year=curr_year, month=curr_month, day=curr_day)
    new_ts = pd.Timestamp(year=new_year, month=new_month, day=new_day)
    return new_ts - curr_ts
