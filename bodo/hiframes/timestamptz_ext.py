import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
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
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo.libs.pd_datetime_arr_ext
from bodo.hiframes.pd_timestamp_ext import (
    PandasTimestampType,
    pd_timestamp_tz_naive_type,
)
from bodo.libs import hdatetime_ext
from bodo.utils.indexing import (
    get_new_null_mask_bool_index,
    get_new_null_mask_int_index,
    get_new_null_mask_slice_index,
    setitem_slice_index_null_bits,
)
from bodo.utils.typing import (
    BodoError,
    is_iterable_type,
    is_list_like_index_type,
    is_overload_none,
)


class TimestampTZ:
    """UTC Timestamp with offset in minutes to a local timezone."""

    def __init__(self, utc_timestamp: pd.Timestamp, offset_minutes: int):
        """Create a TimestampTZ object

        Args:
            utc_timestamp (pd.Timestamp): A timestamp that represents the UTC timestamp.
            offset (int): The offset to apply to the UTC timestamp to get the local time. This
                is the number of minutes to add to the UTC timestamp to get the local time.
        """
        self._utc_timestamp = utc_timestamp
        self._offset_minutes = offset_minutes

    @staticmethod
    def fromUTC(utc_timestamp_string, offset_minutes):
        """
        Alternative constructor for TimestampTZ taking in the timestamp string
        in local time and the offset in hours/minutes from UTC time.

        For example, fromLocal("2018-12-15 20:45:00", -330) represents the
        UTC timestamp "2018-12-16 02:15:00" with an offset of "-05:30"
        """
        utc_timestamp = pd.Timestamp(utc_timestamp_string)
        return TimestampTZ(utc_timestamp, offset_minutes)

    @staticmethod
    def fromLocal(local_timestamp_string, offset_minutes):
        """
        Alternative constructor for TimestampTZ taking in the timestamp string
        in local time and the offset in hours/minutes from UTC time.

        For example, fromLocal("2018-12-15 20:45:00", -330) represents the
        UTC timestamp "2018-12-16 02:15:00" with an offset of "-05:30"
        """
        utc_timestamp = pd.Timestamp(local_timestamp_string) - pd.Timedelta(
            minutes=offset_minutes
        )
        return TimestampTZ(utc_timestamp, offset_minutes)

    def __int__(self):
        # Dummy method for pandas' is_scalar, throw error if called
        raise Exception("Conversion to int not implemented")

    def __hash__(self) -> int:
        return hash(self.utc_timestamp)

    def offset_str(self):
        offset_sign = "+" if self.offset_minutes >= 0 else "-"
        offset_hrs = abs(self.offset_minutes) // 60
        offset_min = abs(self.offset_minutes) % 60
        return f"{offset_sign}{offset_hrs:02}{offset_min:02}"

    def __repr__(self):
        # This implementation is for human readability, not for displaying to
        # the user
        return f"TimestampTZ({self.local_timestamp()}, {self.offset_str()})"

    @property
    def utc_timestamp(self):
        return self._utc_timestamp

    @property
    def offset_minutes(self):
        return self._offset_minutes

    def local_timestamp(self):
        return self.utc_timestamp + pd.Timedelta(minutes=self.offset_minutes)

    def __str__(self):
        # This differs from __repr__ and matches snowflake
        return f"{self.local_timestamp()} {self.offset_str()}"

    def __eq__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp == other.utc_timestamp

    def __ne__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp != other.utc_timestamp

    def __lt__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp < other.utc_timestamp

    def __le__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp <= other.utc_timestamp

    def __gt__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp > other.utc_timestamp

    def __ge__(self, other):
        self._check_can_compare(other)
        return self.utc_timestamp >= other.utc_timestamp

    def _check_can_compare(self, other):
        """Determine if other is a valid object to compare with this TimestampTZ.

        Args:
            other (Any): The other type to check.

        Raises:
            TypeError: The type is not a valid type to compare with TimestampTZ.
        """
        if not isinstance(other, TimestampTZ):
            raise TypeError("Cannot compare TimestampTZ with non-TimestampTZ")


class TimestampTZType(types.Type):
    def __init__(self):
        super().__init__(name="TimestampTZ")


timestamptz_type = TimestampTZType()


@typeof_impl.register(TimestampTZ)
def typeof_pd_timestamp(val, c):
    return TimestampTZType()


@register_model(TimestampTZType)
class TimestampTZModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("utc_timestamp", pd_timestamp_tz_naive_type),
            ("offset_minutes", types.int16),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(TimestampTZType, "utc_timestamp", "utc_timestamp")
make_attribute_wrapper(TimestampTZType, "offset_minutes", "offset_minutes")


@unbox(TimestampTZType)
def unbox_timestamptz(typ, val, c):
    timestamp_obj = c.pyapi.object_getattr_string(val, "utc_timestamp")
    offset_obj = c.pyapi.object_getattr_string(val, "offset_minutes")

    timestamp_tz = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    timestamp_tz.utc_timestamp = c.pyapi.to_native_value(
        pd_timestamp_tz_naive_type, timestamp_obj
    ).value
    timestamp_tz.offset_minutes = c.pyapi.to_native_value(types.int16, offset_obj).value

    c.pyapi.decref(timestamp_obj)
    c.pyapi.decref(offset_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(timestamp_tz._getvalue(), is_error=is_error)


@box(TimestampTZType)
def box_timestamptz(typ, val, c):
    tzts = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    timestamp_obj = c.pyapi.from_native_value(
        pd_timestamp_tz_naive_type, tzts.utc_timestamp
    )
    offset_obj = c.pyapi.long_from_signed_int(tzts.offset_minutes)

    tzts_obj = c.pyapi.unserialize(c.pyapi.serialize_object(TimestampTZ))
    args = c.pyapi.tuple_pack(())
    kwargs = c.pyapi.dict_pack(
        [
            ("utc_timestamp", timestamp_obj),
            ("offset_minutes", offset_obj),
        ]
    )
    res = c.pyapi.call(tzts_obj, args, kwargs)
    c.pyapi.decref(args)
    c.pyapi.decref(kwargs)

    c.pyapi.decref(timestamp_obj)
    c.pyapi.decref(offset_obj)
    return res


def get_utc_timestamp(ts):  # pragma: no cover
    pass


@overload(get_utc_timestamp)
def overload_get_utc_timestamp(ts):
    """
    Converts a timestamp_tz to the corresponding UTC timestamp.

    Args:
        ts (TimestampTZType): the timestamp to convert

    Returns:
        (pd_timestamp_tz_naive_type): the UTC timestamp
    """
    if ts != TimestampTZType():
        return

    def impl(ts):  # pragma: no cover
        return ts.utc_timestamp

    return impl


def get_local_timestamp(ts):  # pragma: no cover
    pass


@overload(get_local_timestamp)
def overload_get_local_timestamp(ts):
    """
    Converts a timestamp_tz to the corresponding regular timestamp
    in local time. E.g. a timestamp_tz with UTC timestamp of
    "2024-07-04 16:30:00" and an offset of +08:15 would have
    a local timestamp of "2024-07-05 00:45:00"

    Args:
        ts (TimestampTZType): the timestamp to convert

    Returns:
        (pd_timestamp_tz_naive_type): the local timestamp
    """
    if ts != TimestampTZType():
        return

    def impl(ts):  # pragma: no cover
        return ts.utc_timestamp + pd.Timedelta(minutes=ts.offset_minutes)

    return impl


@intrinsic
def init_timestamptz(typingctx, utc_timestamp, offset_minutes):
    """Create a TimestampTZ object"""

    def codegen(context, builder, sig, args):
        utc_timestamp, offset_minutes = args
        ts = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        ts.utc_timestamp = utc_timestamp
        ts.offset_minutes = offset_minutes
        return ts._getvalue()

    return (
        timestamptz_type(
            pd_timestamp_tz_naive_type,
            types.int16,
        ),
        codegen,
    )


@lower_constant(TimestampTZType)
def constant_timestamptz(context, builder, ty, pyval):
    # Extracting constants. Inspired from @lower_constant(types.Complex)
    # in numba/numba/targets/numbers.py
    offset_minutes = context.get_constant(types.int16, pyval.offset_minutes)
    utc_timestamp = context.get_constant(
        pd_timestamp_tz_naive_type, pyval.utc_timestamp
    )

    return lir.Constant.literal_struct([utc_timestamp, offset_minutes])


def init_timestamptz_from_local(local_timestamp, offset_minutes):  # pragma: no cover
    return bodo.TimestampTZ.fromLocal(local_timestamp, offset_minutes)


@overload(init_timestamptz_from_local)
def overload_init_timestamptz_from_local(local_timestamp, offset_minutes):
    """
    Constructor for TIMESTAMP_TZ using the local timestamp.
    """

    def impl(local_timestamp, offset_minutes):  # pragma: no cover
        return init_timestamptz(
            local_timestamp - pd.Timedelta(minutes=offset_minutes), offset_minutes
        )

    return impl


@overload(TimestampTZ, no_unliteral=True)
def overload_timestamptz(utc_timestamp, offset_minutes):
    def impl(utc_timestamp, offset_minutes):  # pragma: no cover
        return init_timestamptz(utc_timestamp, offset_minutes)

    return impl


@overload_method(TimestampTZType, "local_timestamp")
def overload_timestamptz_local_timestamp(A):
    return lambda A: A.utc_timestamp + pd.Timedelta(
        minutes=A.offset_minutes
    )  # pragma: no cover


def create_cmp_op_overload(op):
    """create overload function for comparison operators with TimestampTZ type."""
    # TODO(aneesh) support comparing TimestampTZ with other timestamp types,
    # dates, and times.

    def overload_time_cmp(lhs, rhs):
        if isinstance(lhs, TimestampTZType) and isinstance(rhs, TimestampTZType):

            def impl(lhs, rhs):  # pragma: no cover
                x = lhs.utc_timestamp.value
                y = rhs.utc_timestamp.value
                return op(0 if x == y else 1 if x > y else -1, 0)

            return impl

        if isinstance(lhs, TimestampTZType) and is_overload_none(rhs):
            # When we compare TimestampTZ and None in order to sort or take extreme values
            # in a series/array of TimestampTZ, TimestampTZ() > None, TimestampTZ() < None should all return True
            return lambda lhs, rhs: (
                False if op is operator.eq else True
            )  # pragma: no cover

        if is_overload_none(lhs) and isinstance(rhs, TimestampTZType):
            # When we compare None and TimestampTZ in order to sort or take extreme values
            # in a series/array of TimestampTZ, None > TimestampTZ(), None < TimestampTZ() should all return False
            return lambda lhs, rhs: False  # pragma: no cover

        if isinstance(lhs, TimestampTZType) and isinstance(rhs, PandasTimestampType):

            def impl(lhs, rhs):
                x = lhs.utc_timestamp.value
                y = rhs.value
                return op(0 if x == y else 1 if x > y else -1, 0)

            return impl

        if isinstance(lhs, PandasTimestampType) and isinstance(rhs, TimestampTZType):

            def impl(lhs, rhs):
                x = lhs.value
                y = rhs.utc_timestamp.value
                return op(0 if x == y else 1 if x > y else -1, 0)

            return impl

    return overload_time_cmp


class TimestampTZArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="TimestampTZArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    @property
    def dtype(self):
        return TimestampTZType()

    def copy(self):
        return TimestampTZArrayType()

    @staticmethod
    def ts_dtype():
        return types.int64

    @staticmethod
    def offset_dtype():
        return types.int16

    @staticmethod
    def ts_arr_type():
        return types.Array(TimestampTZArrayType.ts_dtype(), 1, "C")

    @staticmethod
    def offset_array_type():
        return types.Array(TimestampTZArrayType.offset_dtype(), 1, "C")


timestamptz_array_type = TimestampTZArrayType()


# TODO(aneesh) refactor array definitions into 1 standard file
data_ts_type = TimestampTZArrayType.ts_arr_type()
data_offset_type = TimestampTZArrayType.offset_array_type()
nulls_type = types.Array(types.uint8, 1, "C")


@register_model(TimestampTZArrayType)
class TimestampTZArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data_ts", data_ts_type),
            ("data_offset", data_offset_type),
            ("null_bitmap", nulls_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(TimestampTZArrayType, "data_ts", "data_ts")
make_attribute_wrapper(TimestampTZArrayType, "data_offset", "data_offset")
make_attribute_wrapper(TimestampTZArrayType, "null_bitmap", "_null_bitmap")


@intrinsic
def init_timestamptz_array(typingctx, data_ts, data_offset, nulls):
    """Create a TimestampTZArrayType with provided data values."""
    assert data_ts == types.Array(types.int64, 1, "C"), (
        "timestamps must be an array of int64"
    )
    assert data_offset == types.Array(types.int16, 1, "C"), (
        "offsets must be an array of int16"
    )
    assert nulls == types.Array(types.uint8, 1, "C"), "nulls must be an array of uint8"

    def codegen(context, builder, signature, args):
        (data_ts_val, data_offset_val, bitmap_val) = args
        # create arr struct and store values
        ts_tz_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        ts_tz_arr.data_ts = data_ts_val
        ts_tz_arr.data_offset = data_offset_val
        ts_tz_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_ts_val)
        context.nrt.incref(builder, signature.args[1], data_offset_val)
        context.nrt.incref(builder, signature.args[2], bitmap_val)

        return ts_tz_arr._getvalue()

    sig = timestamptz_array_type(data_ts, data_offset, nulls)
    return sig, codegen


@numba.njit(no_cpython_wrapper=True)
def alloc_timestamptz_array(n):  # pragma: no cover
    data_ts = np.empty(n, dtype=np.int64)
    data_offset = np.empty(n, dtype=np.int16)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_timestamptz_array(data_ts, data_offset, nulls)


def alloc_timestamptz_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_timestamptz_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws, "alloc_timestamptz_array() takes one argument"
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_timestamptz_ext_alloc_timestamptz_array = (
    alloc_timestamptz_array_equiv
)


@overload_method(TimestampTZArrayType, "copy", no_unliteral=True)
def overload_timestamptz_arr_copy(A):
    """Copy a TimestampTZArrayType by copying the underlying data and null bitmap"""
    return lambda A: bodo.hiframes.timestamptz_ext.init_timestamptz_array(
        A.data_ts, A.data_offset, A._null_bitmap
    )  # pragma: no cover


@overload_attribute(TimestampTZArrayType, "dtype")
def overload_timestamptz_arr_dtype(A):
    return lambda A: A.data_ts.dtype  # pragma: no cover


ll.add_symbol("unbox_timestamptz_array", hdatetime_ext.unbox_timestamptz_array)
ll.add_symbol("box_timestamptz_array", hdatetime_ext.box_timestamptz_array)


@unbox(TimestampTZArrayType)
def unbox_timestamptz_array(typ, val, c):
    n = bodo.utils.utils.object_length(c, val)
    ts_arr = bodo.utils.utils._empty_nd_impl(c.context, c.builder, data_ts_type, [n])
    offset_arr = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, data_offset_type, [n]
    )
    n_bitmask_bytes = c.builder.udiv(
        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    bitmap_arr = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
    )

    # function signature of unbox_timestamptz_array
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64).as_pointer(),
            lir.IntType(16).as_pointer(),
            lir.IntType(8).as_pointer(),
        ],
    )
    fn = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="unbox_timestamptz_array"
    )
    c.builder.call(fn, [val, n, ts_arr.data, offset_arr.data, bitmap_arr.data])

    timestamptz_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    timestamptz_arr.data_ts = ts_arr._getvalue()
    timestamptz_arr.data_offset = offset_arr._getvalue()
    timestamptz_arr.null_bitmap = bitmap_arr._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(timestamptz_arr._getvalue(), is_error=is_error)


@box(TimestampTZArrayType)
def box_timestamptz_array(typ, val, c):
    in_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    data_ts_arr = c.context.make_array(types.Array(types.int64, 1, "C"))(
        c.context, c.builder, in_arr.data_ts
    )
    data_offset_arr = c.context.make_array(types.Array(types.int16, 1, "C"))(
        c.context, c.builder, in_arr.data_offset
    )
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, in_arr.null_bitmap
    ).data

    n = c.builder.extract_value(data_ts_arr.shape, 0)

    fnty = lir.FunctionType(
        c.pyapi.pyobj,
        [
            lir.IntType(64),
            lir.IntType(64).as_pointer(),
            lir.IntType(16).as_pointer(),
            lir.IntType(8).as_pointer(),
        ],
    )
    fn_get = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="box_timestamptz_array"
    )
    obj_arr = c.builder.call(
        fn_get,
        [
            n,
            data_ts_arr.data,
            data_offset_arr.data,
            bitmap_arr_data,
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return obj_arr


@overload(operator.setitem, no_unliteral=True)
def timestamptz_array_setitem(A, idx, val):
    if A != timestamptz_array_type:
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for TimestampTZ Array with indexing type {idx} received an incorrect 'value' type {val}."

    if isinstance(idx, types.Integer):
        if val == timestamptz_type:

            def impl(A, idx, val):  # pragma: no cover
                A.data_ts[idx] = val.utc_timestamp.value
                A.data_offset[idx] = val.offset_minutes
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            return impl

        else:  # pragma: no cover
            raise BodoError(typ_err_msg)

    if not (
        (is_iterable_type(val) and val.dtype == timestamptz_type)
        or types.unliteral(val) == timestamptz_type
    ):  # pragma: no cover
        raise BodoError(typ_err_msg)

    # array of integers
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):
        if types.unliteral(val) == timestamptz_type:
            # A[int_array] = ts_tz_scalar

            def impl_arr_ind_scalar(A, idx, val):  # pragma: no cover
                n = len(idx)
                for i in range(n):
                    A.data_ts[idx[i]] = val.utc_timestamp.value
                    A.data_offset[idx[i]] = val.offset_minutes
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], 1)

            return impl_arr_ind_scalar

        else:
            # A[int_array] = ts_tz_array

            def impl_arr_ind(A, idx, val):  # pragma: no cover
                val = bodo.utils.conversion.coerce_to_array(
                    val, use_nullable_array=True
                )
                n = len(val)
                for i in range(n):
                    A.data_ts[idx[i]] = val.data_ts[i]
                    A.data_offset[idx[i]] = val.data_offset[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], bit)

            return impl_arr_ind

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        if types.unliteral(val) == timestamptz_type:
            # A[bool_array] = ts_tz_scalar

            def impl_bool_ind_mask_scalar(A, idx, val):  # pragma: no cover
                n = len(idx)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        A.data_ts[i] = val.utc_timestamp.value
                        A.data_offset[i] = val.offset_minutes
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_bool_ind_mask_scalar

        else:
            # A[bool_array] = ts_tz_array

            def impl_bool_ind_mask(A, idx, val):  # pragma: no cover
                val = bodo.utils.conversion.coerce_to_array(
                    val, use_nullable_array=True
                )
                n = len(idx)
                val_ind = 0
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        A.data_ts[i] = val.data_ts[val_ind]
                        A.data_offset[i] = val.data_offset[val_ind]
                        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                            val._null_bitmap, val_ind
                        )
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
                        val_ind += 1

            return impl_bool_ind_mask

    # slice case
    if isinstance(idx, types.SliceType):
        if types.unliteral(val) == timestamptz_type:
            # A[slice] = ts_tz_scalar

            def impl_slice_scalar(A, idx, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A.data_ts[i] = val.utc_timestamp.value
                    A.data_offset[i] = val.offset_minutes
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_slice_scalar

        else:
            # A[slice] = ts_tz_array

            def impl_slice_mask(A, idx, val):  # pragma: no cover
                val = bodo.utils.conversion.coerce_to_array(
                    val,
                    use_nullable_array=True,
                )
                n = len(A)
                # using setitem directly instead of copying in loop since
                # Array setitem checks for memory overlap and copies source
                A.data_ts[idx] = val.data_ts
                A.data_offset[idx] = val.data_offset
                # XXX: conservative copy of bitmap in case there is overlap
                src_bitmap = val._null_bitmap.copy()
                setitem_slice_index_null_bits(A._null_bitmap, src_bitmap, idx, n)

            return impl_slice_mask

    # This should be the only TimestampTZ Array implementation.
    # We only expect to reach this case if more ind options are added.
    raise BodoError(
        f"setitem for TimestampTZ Array with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def timestamptz_array_getitem(A, idx):
    if A != timestamptz_array_type:
        return

    # Integer index
    if isinstance(idx, types.Integer):
        return lambda A, idx: init_timestamptz(
            pd.Timestamp(A.data_ts[idx]), A.data_offset[idx]
        )  # pragma: no cover

    # bool arr indexing.
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:

        def impl_bool(A, idx):  # pragma: no cover
            # Heavily influenced by array_getitem_bool_index.
            # Just replaces calls for new data with all 3 arrays
            idx_t = bodo.utils.conversion.coerce_to_array(idx)
            old_mask = A._null_bitmap
            new_ts_data = A.data_ts[idx_t]
            new_offset_data = A.data_offset[idx_t]
            n = len(new_ts_data)
            new_mask = get_new_null_mask_bool_index(old_mask, idx_t, n)
            return init_timestamptz_array(new_ts_data, new_offset_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):

        def impl(A, idx):  # pragma: no cover
            # Heavily influenced by array_getitem_int_index.
            # Just replaces calls for new data with all 3 arrays
            idx_t = bodo.utils.conversion.coerce_to_array(idx)
            old_mask = A._null_bitmap
            new_ts_data = A.data_ts[idx_t]
            new_offset_data = A.data_offset[idx_t]
            n = len(new_ts_data)
            new_mask = get_new_null_mask_int_index(old_mask, idx_t, n)
            return init_timestamptz_array(new_ts_data, new_offset_data, new_mask)

        return impl

    # slice case
    if isinstance(idx, types.SliceType):

        def impl_slice(A, idx):  # pragma: no cover
            # Heavily influenced by array_getitem_slice_index.
            # Just replaces calls for new data with all 3 arrays
            n = len(A)
            old_mask = A._null_bitmap
            new_ts_data = np.ascontiguousarray(A.data_ts[idx])
            new_offset_data = np.ascontiguousarray(A.data_offset[idx])
            new_mask = get_new_null_mask_slice_index(old_mask, idx, n)
            return init_timestamptz_array(new_ts_data, new_offset_data, new_mask)

        return impl_slice

    # This should be the only Timestamp TZ array implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for TimestampTZ Array with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_len_timestamptz_arr(A):
    """Overload len for TimestampTZ Array by returning the length of a component array."""
    if A == timestamptz_array_type:
        return lambda A: len(A.data_ts)  # pragma: no cover


@overload_attribute(TimestampTZArrayType, "shape")
def overload_timestamptz_arr_shape(A):
    """Overload shape for TimestampTZArrayType by returning the shape of the underlying
    ts array
    """
    return lambda A: (len(A.data_ts),)  # pragma: no cover


# Note that max only considers the timestamp and not the offset
@overload(max, no_unliteral=True)
def timestamptz_max(lhs, rhs):
    if isinstance(lhs, TimestampTZType) and isinstance(rhs, TimestampTZType):

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs.utc_timestamp > rhs.utc_timestamp else rhs

        return impl


# Note that min only considers the timestamp and not the offset
@overload(min, no_unliteral=True)
def timestamptz_min(lhs, rhs):
    if isinstance(lhs, TimestampTZType) and isinstance(rhs, TimestampTZType):

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs.utc_timestamp < rhs.utc_timestamp else rhs

        return impl


class ArrowTimestampTZType(pa.ExtensionType):
    def __init__(self):
        type_ = pa.struct(
            [
                pa.field("utc_timestamp", pa.timestamp("ns")),
                pa.field("offset_minutes", pa.int16()),
            ]
        )
        # The name passed in to the constructor is used to identify the type in C++
        pa.ExtensionType.__init__(self, type_, "arrow_timestamp_tz")

    def __arrow_ext_serialize__(self):
        # since we don't have a parameterized type, we don't need extra
        # metadata to be deserialized
        return b""

    @classmethod
    def __arrow_ext_deserialize__(self, storage_type, serialized):
        # return an instance of this subclass given the serialized
        # metadata.
        return ArrowTimestampTZType()

    # Hash the underlying data
    def __hash__(self):
        return hash(self.storage_type) | hash(self.extension_name)


pa.register_extension_type(ArrowTimestampTZType())
