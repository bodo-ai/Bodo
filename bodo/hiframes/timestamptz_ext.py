import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
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

import bodo.libs.pd_datetime_arr_ext
from bodo.hiframes.pd_timestamp_ext import pd_timestamp_tz_naive_type
from bodo.libs import hdatetime_ext


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

    def __int__(self):
        # Dummy method for pandas' is_scalar, throw error if called
        raise Exception("Conversion to int not implemented")

    def __repr__(self):
        offset_sign = "+" if self.offset_minutes >= 0 else "-"
        # TODO: Add leading 0s
        offset_hrs = abs(self.offset_minutes) // 60
        offset_min = abs(self.offset_minutes) % 60
        # TODO: Convert the utc_timestamp to the local timestamp.
        return (
            f"TimestampTZ({self.utc_timestamp}, {offset_sign}{offset_hrs}:{offset_min})"
        )

    @property
    def utc_timestamp(self):
        return self._utc_timestamp

    @property
    def offset_minutes(self):
        return self._offset_minutes

    def __str__(self):
        return self.__repr__()


class TimestampTZType(types.Type):
    def __init__(self):
        super(TimestampTZType, self).__init__(name="TimestampTZ")


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


@intrinsic(prefer_literal=True)
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


@overload(TimestampTZ, no_unliteral=True)
def overload_timestamptz(utc_timestamp, offset_minutes):
    def impl(utc_timestamp, offset_minutes):
        return init_timestamptz(utc_timestamp, offset_minutes)

    return impl


class TimestampTZArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self):
        super(TimestampTZArrayType, self).__init__(name=f"TimestampTZArrayType()")

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


timestamptz_array_type = TimestampTZArrayType()


# TODO(aneesh) refactor array definitions into 1 standard file
data_ts_type = types.Array(types.int64, 1, "C")
data_offset_type = types.Array(types.int16, 1, "C")
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
make_attribute_wrapper(TimestampTZArrayType, "null_bitmap", "null_bitmap")


@intrinsic(prefer_literal=True)
def init_timestamptz_array(typingctx, data_ts, data_offset, nulls):
    """Create a TimestampTZArrayType with provided data values."""
    assert data_ts == types.Array(
        types.int64, 1, "C"
    ), "timestamps must be an array of int64"
    assert data_offset == types.Array(
        types.int16, 1, "C"
    ), "offsets must be an array of int16"
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


@overload_method(TimestampTZArrayType, "copy", no_unliteral=True)
def overload_timestamptz_arr_copy(A):
    """Copy a TimestampTZArrayType by copying the underlying data and null bitmap"""
    return lambda A: bodo.hiframes.timestamptz_ext.init_timestamp_array(
        A.data_ts, A.data_offset, A.null_bitmap
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
    if not isinstance(A, TimestampTZArrayType):
        return
    if isinstance(idx, types.Integer):
        if isinstance(val, TimestampTZType):

            def impl(A, idx, val):
                A.data_ts[idx] = val.utc_timestamp.value
                A.data_offset[idx] = val.offset_minutes
                bodo.libs.int_arr_ext.set_bit_to_arr(A.null_bitmap, idx, 1)

            return impl
    raise Exception("TODO")


@overload(operator.getitem, no_unliteral=True)
def timestamptz_array_getitem(A, idx):
    if not isinstance(A, TimestampTZArrayType):
        return
    if isinstance(idx, types.Integer):
        return lambda A, idx: init_timestamptz(
            pd.Timestamp(A.data_ts[idx]), A.data_offset[idx]
        )  # pragma: no cover
    raise Exception("TODO")
