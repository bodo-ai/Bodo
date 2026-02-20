"""Numba extension support for datetime.timedelta objects and their arrays."""

import datetime
import operator
from collections import namedtuple

import numba
import numpy as np
import pandas as pd
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
import bodo.pandas as bd
import bodo.pandas_compat
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.utils.indexing import (
    get_new_null_mask_bool_index,
    get_new_null_mask_int_index,
    get_new_null_mask_slice_index,
    setitem_slice_index_null_bits,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_str,
    is_iterable_type,
    is_list_like_index_type,
    is_overload_constant_str,
)


# sentinel type representing no first input to pd.Timestamp() constructor
# similar to _no_input object of Pandas in timestamps.pyx
# https://github.com/pandas-dev/pandas/blob/8806ed7120fed863b3cd7d3d5f377ec4c81739d0/pandas/_libs/tslibs/timestamps.pyx#L38
# Also used by pd.Timedelta, and df.to_numpy()
class NoInput:
    pass


_no_input = NoInput()


class NoInputType(types.Type):
    def __init__(self):
        super().__init__(name="NoInput")


register_model(NoInputType)(models.OpaqueModel)


@typeof_impl.register(NoInput)
def _typ_no_input(val, c):
    return NoInputType()


@lower_constant(NoInputType)
def constant_no_input(context, builder, ty, pyval):
    return context.get_dummy_value()


# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class PDTimeDeltaType(types.Type):
    def __init__(self):
        super().__init__(name="PDTimeDeltaType()")


pd_timedelta_type = PDTimeDeltaType()
types.pd_timedelta_type = pd_timedelta_type


# 2.Teach Numba how to infer the Numba type of Python values of a certain class,
# using typeof_impl.register
@typeof_impl.register(pd.Timedelta)
def typeof_pd_timedelta(val, c):
    return pd_timedelta_type


# 3.Define the data model for a Numba type using StructModel and register_model
@register_model(PDTimeDeltaType)
class PDTimeDeltaModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("value", types.int64),
        ]
        super().__init__(dmm, fe_type, members)


# 4.Implementing a boxing function for a Numba type using the @box decorator
@box(PDTimeDeltaType)
def box_pd_timedelta(typ, val, c):
    time_delta = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    value_obj = c.pyapi.long_from_longlong(time_delta.value)

    timedelta_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pd.Timedelta))
    res = c.pyapi.call_function_objargs(timedelta_obj, (value_obj,))
    c.pyapi.decref(value_obj)
    c.pyapi.decref(timedelta_obj)
    return res


# 5.Implementing an unboxing function for a Numba type
# using the @unbox decorator and the NativeValue class
@unbox(PDTimeDeltaType)
def unbox_pd_timedelta(typ, val, c):
    value_obj = c.pyapi.object_getattr_string(val, "value")

    valuell = c.pyapi.long_as_longlong(value_obj)

    time_delta = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    time_delta.value = valuell

    c.pyapi.decref(value_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(time_delta._getvalue(), is_error=is_error)


@lower_constant(PDTimeDeltaType)
def lower_constant_pd_timedelta(context, builder, ty, pyval):
    value = context.get_constant(types.int64, pyval.value)
    return lir.Constant.literal_struct([value])


# 6. Implement the constructor
@overload(pd.Timedelta, no_unliteral=True)
@overload(bd.Timedelta, no_unliteral=True)
def pd_timedelta(
    value=_no_input,
    unit="ns",
    days=0,
    seconds=0,
    microseconds=0,
    milliseconds=0,
    minutes=0,
    hours=0,
    weeks=0,
):
    if value == _no_input:

        def impl_timedelta_kw(
            value=_no_input,
            unit="ns",
            days=0,
            seconds=0,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0,
        ):  # pragma: no cover
            days += weeks * 7
            hours += days * 24
            minutes += 60 * hours
            seconds += 60 * minutes
            milliseconds += 1000 * seconds
            microseconds += 1000 * milliseconds
            ns = 1000 * microseconds
            return init_pd_timedelta(ns)

        return impl_timedelta_kw

    # parse string input
    if value == bodo.types.string_type or is_overload_constant_str(value):
        # just call Pandas in this case since the string parsing code is complex and
        # handles several possible cases
        def impl_str(
            value=_no_input,
            unit="ns",
            days=0,
            seconds=0,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0,
        ):  # pragma: no cover
            with numba.objmode(res="pd_timedelta_type"):
                res = pd.Timedelta(value)
            return res

        return impl_str

    # Timedelta type, just return value
    if value == pd_timedelta_type:
        return (
            lambda value=_no_input,
            unit="ns",
            days=0,
            seconds=0,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0: value
        )  # pragma: no cover

    if value == datetime_timedelta_type:

        def impl_timedelta_datetime(
            value=_no_input,
            unit="ns",
            days=0,
            seconds=0,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0,
        ):  # pragma: no cover
            days = value.days
            seconds = 60 * 60 * 24 * days + value.seconds
            microseconds = 1000 * 1000 * seconds + value.microseconds
            ns = 1000 * microseconds
            return init_pd_timedelta(ns)

        return impl_timedelta_datetime

    # if we reach this point, we need to extract the value of the unit argument, and get the
    # multiplier such that value * multiplier == the correct number of nanoseconds
    if not is_overload_constant_str(unit):  # pragma: no cover
        raise BodoError("pd.to_timedelta(): unit should be a constant string")

    # internal Pandas API that normalizes variations of unit. e.g. 'seconds' -> 's'
    unit = pd._libs.tslibs.timedeltas.parse_timedelta_unit(get_overload_const_str(unit))
    # we don't need the precision value in this case
    (
        value_to_nanoseconds_multiplier,
        _,
    ) = bodo.pandas_compat.precision_from_unit_to_nanoseconds(unit)

    def impl_timedelta(
        value=_no_input,
        unit="ns",
        days=0,
        seconds=0,
        microseconds=0,
        milliseconds=0,
        minutes=0,
        hours=0,
        weeks=0,
    ):  # pragma: no cover
        return init_pd_timedelta(value * value_to_nanoseconds_multiplier)

    return impl_timedelta


@intrinsic
def init_pd_timedelta(typingctx, value):
    def codegen(context, builder, signature, args):
        typ = signature.return_type
        timedelta = cgutils.create_struct_proxy(typ)(context, builder)
        timedelta.value = args[0]
        return timedelta._getvalue()

    return PDTimeDeltaType()(value), codegen


# 2nd arg is used in LLVM level, 3rd arg is used in python level
make_attribute_wrapper(PDTimeDeltaType, "value", "_value")


# Implement the getters
@overload_attribute(PDTimeDeltaType, "value")
def pd_timedelta_get_value(td):
    def impl(td):  # pragma: no cover
        return td._value

    return impl


@overload_attribute(PDTimeDeltaType, "days")
def pd_timedelta_get_days(td):
    def impl(td):  # pragma: no cover
        return td._value // (1000 * 1000 * 1000 * 60 * 60 * 24)

    return impl


@overload_attribute(PDTimeDeltaType, "seconds")
def pd_timedelta_get_seconds(td):
    def impl(td):  # pragma: no cover
        return (td._value // (1000 * 1000 * 1000)) % (60 * 60 * 24)

    return impl


@overload_attribute(PDTimeDeltaType, "microseconds")
def pd_timedelta_get_microseconds(td):
    def impl(td):  # pragma: no cover
        return (td._value // 1000) % 1000000

    return impl


@overload_attribute(PDTimeDeltaType, "nanoseconds")
def pd_timedelta_get_nanoseconds(td):
    def impl(td):  # pragma: no cover
        return td._value % 1000

    return impl


@register_jitable
def _to_hours_pd_td(td):  # pragma: no cover
    return (td._value // (1000 * 1000 * 1000 * 60 * 60)) % 24


@register_jitable
def _to_minutes_pd_td(td):  # pragma: no cover
    return (td._value // (1000 * 1000 * 1000 * 60)) % 60


@register_jitable
def _to_seconds_pd_td(td):  # pragma: no cover
    return (td._value // (1000 * 1000 * 1000)) % 60


@register_jitable
def _to_milliseconds_pd_td(td):  # pragma: no cover
    return (td._value // (1000 * 1000)) % 1000


@register_jitable
def _to_microseconds_pd_td(td):  # pragma: no cover
    return (td._value // (1000)) % 1000


Components = namedtuple(
    "Components",
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
    defaults=[
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
)


@overload_attribute(PDTimeDeltaType, "components", no_unliteral=True)
def pd_timedelta_get_components(td):
    def impl(td):  # pragma: no cover
        a = Components(
            td.days,
            _to_hours_pd_td(td),
            _to_minutes_pd_td(td),
            _to_seconds_pd_td(td),
            _to_milliseconds_pd_td(td),
            _to_microseconds_pd_td(td),
            td.nanoseconds,
        )
        return a

    return impl


@overload_method(PDTimeDeltaType, "__hash__", no_unliteral=True)
def pd_td___hash__(td):
    """Hashcode for pd.Timedelta types."""

    def impl(td):  # pragma: no cover
        return hash(td._value)

    return impl


@overload_method(PDTimeDeltaType, "to_numpy", no_unliteral=True)
@overload_method(PDTimeDeltaType, "to_timedelta64", no_unliteral=True)
def pd_td_to_numpy(td):
    """Convert to NP.timedelta64[ns]."""
    # TODO: Fix imports
    from bodo.hiframes.pd_timestamp_ext import integer_to_timedelta64

    def impl(td):  # pragma: no cover
        return integer_to_timedelta64(td.value)

    return impl


@overload_method(PDTimeDeltaType, "to_pytimedelta", no_unliteral=True)
def pd_td_to_pytimedelta(td):
    """Convert to datetime.timedelta."""

    def impl(td):  # pragma: no cover
        return datetime.timedelta(microseconds=np.int64(td._value / 1000))

    return impl


@overload_method(PDTimeDeltaType, "total_seconds", no_unliteral=True)
def pd_td_total_seconds(td):
    """Total seconds in the duration. Pandas drops nanoseconds from this result"""

    def impl(td):  # pragma: no cover
        return (td._value // 1000) / 10**6

    return impl


@overload(min, no_unliteral=True, jit_options={"cache": True})
def timedelta_min(lhs, rhs):
    if rhs == pd_timedelta_type and lhs == pd_timedelta_type:

        def impl(lhs, rhs):  # prama: no cover
            return lhs if lhs.value < rhs.value else rhs

        return impl

    elif (
        isinstance(lhs, numba.core.typing.builtins.IndexValueType)
        and isinstance(rhs, numba.core.typing.builtins.IndexValueType)
        and lhs.val_typ == pd_timedelta_type
        and rhs.val_typ == pd_timedelta_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            # Based off of https://github.com/numba/numba/blob/249c8ff3206928b486346443ec148508f8c25f8e/numba/cpython/builtins.py#L589
            #
            # If both values are NaT, compare by index. If one value is not Nan and the other is, return the non-NaT. Else return the normal
            if pd.isna(lhs) and pd.isna(rhs):
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            elif pd.isna(lhs):
                return rhs
            elif pd.isna(rhs):
                return lhs
            elif lhs.value < rhs.value:
                return lhs
            elif lhs.value == rhs.value:
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            else:
                return rhs

        return impl


@overload(max, no_unliteral=True, jit_options={"cache": True})
def timedelta_max(lhs, rhs):
    if rhs == pd_timedelta_type and lhs == pd_timedelta_type:

        def impl(lhs, rhs):  # prama: no cover
            return lhs if lhs.value > rhs.value else rhs

        return impl

    elif (
        isinstance(lhs, numba.core.typing.builtins.IndexValueType)
        and isinstance(rhs, numba.core.typing.builtins.IndexValueType)
        and lhs.val_typ == pd_timedelta_type
        and rhs.val_typ == pd_timedelta_type
    ):  # pragma: no cover

        def impl(lhs, rhs):  # pragma: no cover
            # Based off of https://github.com/numba/numba/blob/249c8ff3206928b486346443ec148508f8c25f8e/numba/cpython/builtins.py#L589
            #
            # If both values are NaT, compare by index. If one value is not Nan and the other is, return the non-NaT. Else return the normal
            if pd.isna(lhs) and pd.isna(rhs):
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            elif pd.isna(lhs):
                return rhs
            elif pd.isna(rhs):
                return lhs
            elif lhs.value < rhs.value:
                return rhs
            elif lhs.value == rhs.value:
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            else:
                return lhs

        return impl


def overload_add_operator_datetime_timedelta(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            val = lhs.value + rhs.value
            return pd.Timedelta(val)

        return impl

    if lhs == pd_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            rhs_value = (
                rhs.microseconds
                + ((rhs.seconds + (rhs.days * 60 * 60 * 24)) * 1000 * 1000)
            ) * 1000
            val = lhs.value + rhs_value
            return pd.Timedelta(val)

        return impl

    if lhs == datetime_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            lhs_value = (
                lhs.microseconds
                + ((lhs.seconds + (lhs.days * 60 * 60 * 24)) * 1000 * 1000)
            ) * 1000
            val = lhs_value + rhs.value
            return pd.Timedelta(val)

        return impl

    if lhs == pd_timedelta_type and rhs == datetime_datetime_type:
        # Import here to avoid circular import error. Perhaps this
        # should be moved to a utils file.
        from bodo.hiframes.pd_timestamp_ext import compute_pd_timestamp

        def impl(lhs, rhs):  # pragma: no cover
            # The time itself
            days1 = rhs.toordinal()
            secs1 = rhs.second + rhs.minute * 60 + rhs.hour * 3600
            msec1 = rhs.microsecond
            # The timedelta
            msec2 = lhs.value // 1000
            nanosec2 = lhs.nanoseconds
            # Computing the difference
            msecF = msec1 + msec2
            # Getting total microsecond
            totmicrosec = 1000000 * (days1 * 86400 + secs1) + msecF
            # Getting total nano_seconds
            totnanosec = nanosec2
            return compute_pd_timestamp(totmicrosec, totnanosec)

        return impl

    if lhs == datetime_datetime_type and rhs == pd_timedelta_type:
        # In Python this becomes a datetime instead
        # of a timestamp

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + rhs.to_pytimedelta()

        return impl

    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs.days + rhs.days
            s = lhs.seconds + rhs.seconds
            us = lhs.microseconds + rhs.microseconds
            return datetime.timedelta(d, s, us)

        return impl

    if lhs == datetime_timedelta_type and rhs == datetime_datetime_type:

        def impl(lhs, rhs):  # pragma: no cover
            delta = datetime.timedelta(
                rhs.toordinal(),
                hours=rhs.hour,
                minutes=rhs.minute,
                seconds=rhs.second,
                microseconds=rhs.microsecond,
            )
            delta = delta + lhs
            hour, rem = divmod(delta.seconds, 3600)
            minute, second = divmod(rem, 60)
            if 0 < delta.days <= _MAXORDINAL:
                d = bodo.hiframes.datetime_date_ext.fromordinal_impl(delta.days)
                return datetime.datetime(
                    d.year, d.month, d.day, hour, minute, second, delta.microseconds
                )
            raise OverflowError("result out of range")

        return impl

    if lhs == datetime_datetime_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            delta = datetime.timedelta(
                lhs.toordinal(),
                hours=lhs.hour,
                minutes=lhs.minute,
                seconds=lhs.second,
                microseconds=lhs.microsecond,
            )
            delta = delta + rhs
            hour, rem = divmod(delta.seconds, 3600)
            minute, second = divmod(rem, 60)
            if 0 < delta.days <= _MAXORDINAL:
                d = bodo.hiframes.datetime_date_ext.fromordinal_impl(delta.days)
                return datetime.datetime(
                    d.year, d.month, d.day, hour, minute, second, delta.microseconds
                )
            raise OverflowError("result out of range")

        return impl


def overload_sub_operator_datetime_timedelta(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            new_val = lhs.value - rhs.value
            return pd.Timedelta(new_val)

        return impl

    if lhs == pd_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl

    if lhs == datetime_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl

    if lhs == datetime_datetime_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl

    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs.days - rhs.days
            s = lhs.seconds - rhs.seconds
            us = lhs.microseconds - rhs.microseconds
            return datetime.timedelta(d, s, us)

        return impl

    if lhs == datetime_datetime_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl

    # datetime_timedelta_array - timedelta
    if lhs == timedelta_array_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            in_arr = lhs
            numba.parfors.parfor.init_prange()
            n = len(in_arr)
            A = alloc_timedelta_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(in_arr, i):
                    bodo.libs.array_kernels.setna(A, i)
                    continue
                A[i] = in_arr[i] - rhs
            return A

        return impl

    # timedelta - datetime_timedelta_array
    if lhs == datetime_timedelta_type and rhs == timedelta_array_type:

        def impl(lhs, rhs):  # pragma: no cover
            in_arr = rhs
            numba.parfors.parfor.init_prange()
            n = len(in_arr)
            A = alloc_timedelta_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(in_arr, i):
                    bodo.libs.array_kernels.setna(A, i)
                    continue
                A[i] = lhs - in_arr[i]
            return A

        return impl


def overload_mul_operator_timedelta(lhs, rhs):
    if lhs == pd_timedelta_type and isinstance(rhs, types.Integer):

        def impl(lhs, rhs):  # pragma: no cover
            return pd.Timedelta(lhs.value * rhs)

        return impl

    elif isinstance(lhs, types.Integer) and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return pd.Timedelta(rhs.value * lhs)

        return impl

    if lhs == datetime_timedelta_type and isinstance(rhs, types.Integer):

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs.days * rhs
            s = lhs.seconds * rhs
            us = lhs.microseconds * rhs
            return datetime.timedelta(d, s, us)

        return impl

    elif isinstance(lhs, types.Integer) and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs * rhs.days
            s = lhs * rhs.seconds
            us = lhs * rhs.microseconds
            return datetime.timedelta(d, s, us)

        return impl


def overload_floordiv_operator_pd_timedelta(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs.value // rhs.value

        return impl

    elif lhs == pd_timedelta_type and isinstance(rhs, types.Integer):

        def impl(lhs, rhs):  # pragma: no cover
            return pd.Timedelta(lhs.value // rhs)

        return impl


def overload_truediv_operator_pd_timedelta(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs.value / rhs.value

        return impl

    elif lhs == pd_timedelta_type and isinstance(rhs, types.Integer):

        def impl(lhs, rhs):  # pragma: no cover
            return pd.Timedelta(int(lhs.value / rhs))

        # TODO: float division: rhs=float64 type

        return impl


def overload_mod_operator_timedeltas(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return pd.Timedelta(lhs.value % rhs.value)

        return impl

    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            r = _to_microseconds(lhs) % _to_microseconds(rhs)
            return datetime.timedelta(0, 0, r)

        return impl


@overload(min, no_unliteral=True)
def timedelta_min(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs < rhs else rhs

        return impl


@overload(max, no_unliteral=True)
def timedelta_max(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs > rhs else rhs

        return impl


def pd_create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_date_array"""

    def overload_pd_timedelta_cmp(lhs, rhs):
        if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

            def impl(lhs, rhs):  # pragma: no cover
                return op(lhs.value, rhs.value)

            return impl

        # Timedelta/td64
        if lhs == pd_timedelta_type and rhs == bodo.types.timedelta64ns:
            return lambda lhs, rhs: op(
                bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(lhs.value), rhs
            )  # pragma: no cover

        # td64/Timedelta
        if lhs == bodo.types.timedelta64ns and rhs == pd_timedelta_type:
            return lambda lhs, rhs: op(
                lhs, bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(rhs.value)
            )  # pragma: no cover

    return overload_pd_timedelta_cmp


@overload(operator.neg, no_unliteral=True)
def pd_timedelta_neg(lhs):
    if lhs == pd_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return pd.Timedelta(-lhs.value)

        return impl


@overload(operator.pos, no_unliteral=True)
def pd_timedelta_pos(lhs):
    if lhs == pd_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return lhs

        return impl


@overload(divmod, no_unliteral=True)
def pd_timedelta_divmod(lhs, rhs):
    if lhs == pd_timedelta_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            q, r = divmod(lhs.value, rhs.value)
            return q, pd.Timedelta(r)

        return impl


@overload(abs, no_unliteral=True)
def pd_timedelta_abs(lhs):
    if lhs == pd_timedelta_type:

        def impl(lhs):  # pragma: no cover
            if lhs.value < 0:
                return -lhs
            else:
                return lhs

        return impl


# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class DatetimeTimeDeltaType(types.Type):
    def __init__(self):
        super().__init__(name="DatetimeTimeDeltaType()")


datetime_timedelta_type = DatetimeTimeDeltaType()


# 2.Teach Numba how to infer the Numba type of Python values of a certain class,
# using typeof_impl.register
@typeof_impl.register(datetime.timedelta)
def typeof_datetime_timedelta(val, c):
    return datetime_timedelta_type


# 3.Define the data model for a Numba type using StructModel and register_model
@register_model(DatetimeTimeDeltaType)
class DatetimeTimeDeltaModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("days", types.int64),
            ("seconds", types.int64),
            ("microseconds", types.int64),
        ]
        super().__init__(dmm, fe_type, members)


# 4.Implementing a boxing function for a Numba type using the @box decorator
@box(DatetimeTimeDeltaType)
def box_datetime_timedelta(typ, val, c):
    time_delta = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    days_obj = c.pyapi.long_from_longlong(time_delta.days)
    seconds_obj = c.pyapi.long_from_longlong(time_delta.seconds)
    microseconds_obj = c.pyapi.long_from_longlong(time_delta.microseconds)

    timedelta_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.timedelta))
    res = c.pyapi.call_function_objargs(
        timedelta_obj, (days_obj, seconds_obj, microseconds_obj)
    )
    c.pyapi.decref(days_obj)
    c.pyapi.decref(seconds_obj)
    c.pyapi.decref(microseconds_obj)
    c.pyapi.decref(timedelta_obj)
    return res


# 5.Implementing an unboxing function for a Numba type
# using the @unbox decorator and the NativeValue class
@unbox(DatetimeTimeDeltaType)
def unbox_datetime_timedelta(typ, val, c):
    days_obj = c.pyapi.object_getattr_string(val, "days")
    seconds_obj = c.pyapi.object_getattr_string(val, "seconds")
    microseconds_obj = c.pyapi.object_getattr_string(val, "microseconds")

    daysll = c.pyapi.long_as_longlong(days_obj)
    secondsll = c.pyapi.long_as_longlong(seconds_obj)
    microsecondsll = c.pyapi.long_as_longlong(microseconds_obj)

    time_delta = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    time_delta.days = daysll
    time_delta.seconds = secondsll
    time_delta.microseconds = microsecondsll

    c.pyapi.decref(days_obj)
    c.pyapi.decref(seconds_obj)
    c.pyapi.decref(microseconds_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(time_delta._getvalue(), is_error=is_error)


@lower_constant(DatetimeTimeDeltaType)
def lower_constant_datetime_timedelta(context, builder, ty, pyval):
    days = context.get_constant(types.int64, pyval.days)
    seconds = context.get_constant(types.int64, pyval.seconds)
    microseconds = context.get_constant(types.int64, pyval.microseconds)
    return lir.Constant.literal_struct([days, seconds, microseconds])


# 6. Implement the constructor
@overload(datetime.timedelta, no_unliteral=True)
def datetime_timedelta(
    days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0
):
    def impl_timedelta(
        days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0
    ):  # pragma: no cover
        d = s = us = 0

        # Normalize everything to days, seconds, microseconds.
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000

        # convert seconds to days, microseconds to seconds
        d = days
        days, seconds = divmod(seconds, 24 * 3600)
        d += days
        s += int(seconds)

        seconds, us = divmod(microseconds, 1000000)
        days, seconds = divmod(seconds, 24 * 3600)
        d += days
        s += seconds

        return init_timedelta(d, s, us)

    return impl_timedelta


@intrinsic
def init_timedelta(typingctx, d, s, us):
    def codegen(context, builder, signature, args):
        typ = signature.return_type
        timedelta = cgutils.create_struct_proxy(typ)(context, builder)
        timedelta.days = args[0]
        timedelta.seconds = args[1]
        timedelta.microseconds = args[2]

        return timedelta._getvalue()

    return DatetimeTimeDeltaType()(d, s, us), codegen


# 2nd arg is used in LLVM level, 3rd arg is used in python level
make_attribute_wrapper(DatetimeTimeDeltaType, "days", "_days")
make_attribute_wrapper(DatetimeTimeDeltaType, "seconds", "_seconds")
make_attribute_wrapper(DatetimeTimeDeltaType, "microseconds", "_microseconds")


# Implement the getters
@overload_attribute(DatetimeTimeDeltaType, "days")
def timedelta_get_days(td):
    def impl(td):  # pragma: no cover
        return td._days

    return impl


@overload_attribute(DatetimeTimeDeltaType, "seconds")
def timedelta_get_seconds(td):
    def impl(td):  # pragma: no cover
        return td._seconds

    return impl


@overload_attribute(DatetimeTimeDeltaType, "microseconds")
def timedelta_get_microseconds(td):
    def impl(td):  # pragma: no cover
        return td._microseconds

    return impl


@overload_method(DatetimeTimeDeltaType, "total_seconds", no_unliteral=True)
def total_seconds(td):
    """Total seconds in the duration."""

    def impl(td):  # pragma: no cover
        return ((td._days * 86400 + td._seconds) * 10**6 + td._microseconds) / 10**6

    return impl


@overload_method(DatetimeTimeDeltaType, "__hash__", no_unliteral=True)
def __hash__(td):
    """Hashcode for datetimed.timedelta types. Copies the CPython implementation"""

    def impl(td):  # pragma: no cover
        return hash((td._days, td._seconds, td._microseconds))

    return impl


@register_jitable
def _to_nanoseconds(td):  # pragma: no cover
    return np.int64(
        ((td._days * 86400 + td._seconds) * 1000000 + td._microseconds) * 1000
    )


@register_jitable
def _to_microseconds(td):  # pragma: no cover
    return (td._days * (24 * 3600) + td._seconds) * 1000000 + td._microseconds


@register_jitable
def _cmp(x, y):  # pragma: no cover
    return 0 if x == y else 1 if x > y else -1


@register_jitable
def _getstate(td):  # pragma: no cover
    return (td._days, td._seconds, td._microseconds)


@register_jitable
def _divide_and_round(a, b):  # pragma: no cover
    """divide a by b and round result to the nearest integer
    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    q, r = divmod(a, b)
    # round up if either r / b > 0.5, or r / b == 0.5 and q is odd.
    # The expression r / b > 0.5 is equivalent to 2 * r > b if b is
    # positive, 2 * r < b if b negative.
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or r == b and q % 2 == 1:
        q += 1

    return q


_MAXORDINAL = 3652059


def overload_floordiv_operator_dt_timedelta(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            us = _to_microseconds(lhs)
            return us // _to_microseconds(rhs)

        return impl

    elif lhs == datetime_timedelta_type and rhs == types.int64:

        def impl(lhs, rhs):  # pragma: no cover
            us = _to_microseconds(lhs)
            return datetime.timedelta(0, 0, us // rhs)

        return impl


def overload_truediv_operator_dt_timedelta(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            us = _to_microseconds(lhs)
            return us / _to_microseconds(rhs)

        return impl

    elif lhs == datetime_timedelta_type and rhs == types.int64:

        def impl(lhs, rhs):  # pragma: no cover
            us = _to_microseconds(lhs)
            return datetime.timedelta(0, 0, _divide_and_round(us, rhs))

        # TODO: float division: rhs=float64 type

        return impl


def create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_timedelta_type."""

    def overload_timedelta_cmp(lhs, rhs):
        if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

            def impl(lhs, rhs):  # pragma: no cover
                ret = _cmp(_getstate(lhs), _getstate(rhs))
                return op(ret, 0)

            return impl

    return overload_timedelta_cmp


@overload(operator.neg, no_unliteral=True)
def timedelta_neg(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return datetime.timedelta(-lhs.days, -lhs.seconds, -lhs.microseconds)

        return impl


@overload(operator.pos, no_unliteral=True)
def timedelta_pos(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return lhs

        return impl


@overload(divmod, no_unliteral=True)
def timedelta_divmod(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            q, r = divmod(_to_microseconds(lhs), _to_microseconds(rhs))
            return q, datetime.timedelta(0, 0, r)

        return impl


@overload(abs, no_unliteral=True)
def timedelta_abs(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            if lhs.days < 0:
                return -lhs
            else:
                return lhs

        return impl


@intrinsic
def cast_numpy_timedelta_to_int(typingctx, val=None):
    """Cast timedelta64 value to int"""
    assert val in (types.NPTimedelta("ns"), types.int64, bodo.types.pd_timedelta_type)

    if val == bodo.types.pd_timedelta_type:

        def codegen(context, builder, signature, args):
            time_delta = cgutils.create_struct_proxy(val)(
                context, builder, value=args[0]
            )
            return time_delta.value

    else:

        def codegen(context, builder, signature, args):
            return args[0]

    return types.int64(val), codegen


@overload(bool, no_unliteral=True)
def timedelta_to_bool(timedelta):
    if timedelta != datetime_timedelta_type:  # pragma: no cover
        return

    zero_timedelta = datetime.timedelta(0)

    def impl(timedelta):  # pragma: no cover
        return timedelta != zero_timedelta

    return impl


@overload(bool, no_unliteral=True)
def pd_timedelta_to_bool(timedelta):
    if timedelta != pd_timedelta_type:  # pragma: no cover
        return

    def impl(timedelta):  # pragma: no cover
        return timedelta.value != 0

    return impl


##################### Array of datetime.timedelta objects ##########################


class TimeDeltaArrayType(types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="TimeDeltaArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return pd_timedelta_type

    def copy(self):
        return TimeDeltaArrayType()


timedelta_array_type = TimeDeltaArrayType()
types.timedelta_array_type = timedelta_array_type

data_array_type = types.Array(types.NPTimedelta("ns"), 1, "C")
nulls_type = types.Array(types.uint8, 1, "C")


# datetime.timedelta has three arrays of integers to store data
@register_model(TimeDeltaArrayType)
class DatetimeTimeDeltaArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", data_array_type),
            ("null_bitmap", nulls_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(TimeDeltaArrayType, "data", "_data")
make_attribute_wrapper(TimeDeltaArrayType, "null_bitmap", "_null_bitmap")


@overload_method(TimeDeltaArrayType, "copy", no_unliteral=True)
def overload_datetime_timedelta_arr_copy(A):
    return lambda A: bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array(
        A._data.copy(), A._null_bitmap.copy()
    )  # pragma: no cover


@typeof_impl.register(pd.arrays.TimedeltaArray)
def typeof_pd_timedelta_array(val, c):
    if val.unit != "ns":
        raise BodoError("Timedelta array data requires 'ns' unit")

    return timedelta_array_type


@unbox(TimeDeltaArrayType)
def unbox_pd_timedelta_array(typ, val, c):
    """
    Unbox a timedelta array using Arrow.
    """
    return bodo.libs.array.unbox_array_using_arrow(typ, val, c)


@box(TimeDeltaArrayType)
def box_pd_timedelta_array(typ, val, c):
    """
    Box a timedelta into an Arrow array.
    """
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@intrinsic
def init_datetime_timedelta_array(typingctx, data, nulls):
    """Create a TimeDeltaArrayType with provided data values."""
    assert data == data_array_type
    assert nulls == nulls_type

    def codegen(context, builder, signature, args):
        (data_val, bitmap_val) = args
        # create arr struct and store values
        dt_date_arr = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        dt_date_arr.data = data_val
        dt_date_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], bitmap_val)
        return dt_date_arr._getvalue()

    sig = timedelta_array_type(data, nulls)
    return sig, codegen


@lower_constant(TimeDeltaArrayType)
def lower_constant_datetime_timedelta_arr(context, builder, typ, pyval):
    n = len(pyval)
    data_arr = np.empty(n, np.dtype("timedelta64[ns]"))
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)

    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            data_arr[i] = s

    data_const_arr = context.get_constant_generic(builder, data_array_type, data_arr)
    nulls_const_arr = context.get_constant_generic(builder, nulls_type, nulls_arr)

    return lir.Constant.literal_struct(
        [
            data_const_arr,
            nulls_const_arr,
        ]
    )


@numba.njit(no_cpython_wrapper=True)
def alloc_timedelta_array(n):  # pragma: no cover
    data_arr = np.empty(n, dtype=bodo.types.timedelta64ns)
    # XXX: set all bits to not null since datetime.timedelta array operations do not support
    # NA yet. TODO: use 'empty' when all operations support NA
    # nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    nulls = np.full((n + 7) >> 3, 255, np.uint8)
    return init_datetime_timedelta_array(data_arr, nulls)


def alloc_timedelta_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_timedelta_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_datetime_timedelta_ext_alloc_timedelta_array = alloc_timedelta_array_equiv


@overload(operator.getitem, no_unliteral=True)
def dt_timedelta_arr_getitem(A, ind):
    if A != timedelta_array_type:
        return

    if isinstance(ind, types.Integer):

        def impl_int(A, ind):
            # TODO: Eventually support handle case where value is marked as
            # NA/None. But for now we will mark this as a github issue and fix
            # implementation later.
            return init_pd_timedelta(A._data[ind])

        return impl_int

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            # Heavily influenced by array_getitem_bool_index.
            # Just replaces calls for new data with all 3 arrays
            ind_t = bodo.utils.conversion.coerce_to_array(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind_t]
            n = len(new_data)
            new_mask = get_new_null_mask_bool_index(old_mask, ind_t, n)
            return init_datetime_timedelta_array(new_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            # Heavily influenced by array_getitem_int_index.
            # Just replaces calls for new data with all 3 arrays
            ind_t = bodo.utils.conversion.coerce_to_array(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind_t]
            n = len(new_data)
            new_mask = get_new_null_mask_int_index(old_mask, ind_t, n)
            return init_datetime_timedelta_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            # Heavily influenced by array_getitem_slice_index.
            # Just replaces calls for new data with all 3 arrays
            n = len(A._data)
            old_mask = A._null_bitmap
            new_data = np.ascontiguousarray(A._data[ind])
            new_mask = get_new_null_mask_slice_index(old_mask, ind, n)
            return init_datetime_timedelta_array(new_data, new_mask)

        return impl_slice

    # This should be the only DatetimeTimedeltaArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for DatetimeTimedeltaArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def dt_timedelta_arr_setitem(A, ind, val):
    if A != timedelta_array_type:
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for DatetimeTimedeltaArray with indexing type {ind} received an incorrect 'value' type {val}."

    # scalar case
    if isinstance(ind, types.Integer):
        if types.unliteral(val) == datetime_timedelta_type:

            def impl(A, ind, val):  # pragma: no cover
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    val
                )
                A._data[ind] = td64
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, ind, 1)

            # TODO: Confirm the coverage and if its missing add it to the test cases
            return impl

        elif types.unliteral(val) == pd_timedelta_type:

            def impl(A, ind, val):  # pragma: no cover
                td64_val = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    val.value
                )
                A._data[ind] = td64_val
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    A._null_bitmap, ind, 0 if np.isnat(td64_val) else 1
                )

            return impl
        elif types.unliteral(val) == bodo.types.timedelta64ns:

            def impl(A, ind, val):  # pragma: no cover
                A._data[ind] = val
                bodo.libs.int_arr_ext.set_bit_to_arr(
                    A._null_bitmap, ind, 0 if np.isnat(val) else 1
                )

            return impl
        else:
            raise BodoError(typ_err_msg)

    if not (
        (
            is_iterable_type(val)
            and val.dtype in (datetime_timedelta_type, pd_timedelta_type)
        )
        or types.unliteral(val)
        in (datetime_timedelta_type, pd_timedelta_type, bodo.types.timedelta64ns)
    ):
        raise BodoError(typ_err_msg)

    # array of integers
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        if types.unliteral(val) == datetime_timedelta_type:

            def impl_arr_ind_scalar(A, ind, val):  # pragma: no cover
                n = len(A)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    val
                )
                for i in range(n):
                    A._data[ind[i]] = td64
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, ind[i], 1)

            # TODO: Confirm the coverage and if its missing add it to the test cases
            return impl_arr_ind_scalar

        elif types.unliteral(val) == pd_timedelta_type:

            def impl_arr_ind_scalar(A, ind, val):  # pragma: no cover
                n = len(A)
                td64 = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val.value)
                for i in range(n):
                    A._data[ind[i]] = td64
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, ind[i], 1)

            return impl_arr_ind_scalar

        else:

            def impl_arr_ind(A, ind, val):  # pragma: no cover
                # Heavily influenced by array_setitem_int_index.
                # Just replaces calls for new data with all 3 arrays
                val = bodo.utils.conversion.coerce_to_array(
                    val, use_nullable_array=True
                )
                n = len(val._data)
                for i in range(n):
                    A._data[ind[i]] = val._data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, ind[i], bit)

            # TODO: Confirm the coverage and if its missing add it to the test cases
            return impl_arr_ind

    # bool array
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        if types.unliteral(val) == datetime_timedelta_type:

            def impl_bool_ind_mask_scalar(A, ind, val):  # pragma: no cover
                # Heavily influenced by array_setitem_bool_index.
                # Just replaces calls for new data with all 3 arrays
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    val
                )
                n = len(ind)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                        A._data[i] = td64
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_bool_ind_mask_scalar

        elif types.unliteral(val) == pd_timedelta_type:

            def impl_bool_ind_mask_scalar(A, ind, val):  # pragma: no cover
                td64 = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val.value)
                n = len(ind)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                        A._data[i] = td64
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_bool_ind_mask_scalar
        else:

            def impl_bool_ind_mask(A, ind, val):  # pragma: no cover
                # Heavily influenced by array_setitem_bool_index.
                # Just replaces calls for new data with all 3 arrays
                val = bodo.utils.conversion.coerce_to_array(
                    val, use_nullable_array=True
                )
                n = len(ind)
                val_ind = 0
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                        A._data[i] = val._data[val_ind]
                        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                            val._null_bitmap, val_ind
                        )
                        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
                        val_ind += 1

            return impl_bool_ind_mask

    # slice case
    if isinstance(ind, types.SliceType):
        if types.unliteral(val) == datetime_timedelta_type:

            def impl_slice_scalar(A, ind, val):  # pragma: no cover
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    val
                )
                slice_idx = numba.cpython.unicode._normalize_slice(ind, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A._data[i] = td64
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_slice_scalar

        elif types.unliteral(val) == pd_timedelta_type:

            def impl_slice_scalar(A, ind, val):  # pragma: no cover
                td64 = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val.value)
                slice_idx = numba.cpython.unicode._normalize_slice(ind, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A._data[i] = td64
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_slice_scalar
        elif types.unliteral(val) == bodo.types.timedelta64ns:

            def impl_slice_scalar(A, ind, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(ind, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A._data[i] = val
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

            return impl_slice_scalar
        else:

            def impl_slice_mask(A, ind, val):  # pragma: no cover
                # Heavily influenced by array_setitem_slice_index.
                # Just replaces calls for new data with all 3 arrays
                val = bodo.utils.conversion.coerce_to_array(
                    val,
                    use_nullable_array=True,
                )
                n = len(A._data)
                # using setitem directly instead of copying in loop since
                # Array setitem checks for memory overlap and copies source
                A._data[ind] = val._data
                # XXX: conservative copy of bitmap in case there is overlap
                # TODO: check for overlap and copy only if necessary
                src_bitmap = val._null_bitmap.copy()
                setitem_slice_index_null_bits(A._null_bitmap, src_bitmap, ind, n)

            return impl_slice_mask

    # This should be the only DatetimeTimedeltaArray implementation.
    # We only expect to reach this case if more ind options are added.
    raise BodoError(
        f"setitem for DatetimeTimedeltaArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_len_datetime_timedelta_arr(A):
    if A == timedelta_array_type:
        return lambda A: len(A._data)


@overload_attribute(TimeDeltaArrayType, "shape")
def overload_datetime_timedelta_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(TimeDeltaArrayType, "nbytes")
def timedelta_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


def overload_datetime_timedelta_arr_sub(arg1, arg2):
    # datetime_timedelta_array - timedelta
    if arg1 == timedelta_array_type and arg2 == datetime_timedelta_type:

        def impl(arg1, arg2):  # pragma: no cover
            in_arr = arg1
            numba.parfors.parfor.init_prange()
            n = len(in_arr)
            A = alloc_timedelta_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                A[i] = in_arr[i] - arg2
            return A

        return impl


def create_cmp_op_overload_arr(op):
    """create overload function for comparison operators with datetime_timedelta_array"""

    def overload_date_arr_cmp(lhs, rhs):
        if op == operator.ne:
            default_value = True
        else:
            default_value = False
        # both timedelta_array_type
        if lhs == timedelta_array_type and rhs == timedelta_array_type:

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    bit1 = bodo.libs.array_kernels.isna(lhs, i)
                    bit2 = bodo.libs.array_kernels.isna(rhs, i)
                    if bit1 or bit2:
                        ret_val = default_value
                    else:
                        ret_val = op(lhs[i], rhs[i])
                    out_arr[i] = ret_val
                return out_arr

            return impl
        # 1st arg is array
        elif lhs == timedelta_array_type:

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    bit = bodo.libs.array_kernels.isna(lhs, i)
                    if bit:
                        ret_val = default_value
                    else:
                        ret_val = op(lhs[i], rhs)
                    out_arr[i] = ret_val
                return out_arr

            return impl
        # 2nd arg is array
        elif rhs == timedelta_array_type:

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    bit = bodo.libs.array_kernels.isna(rhs, i)
                    if bit:
                        ret_val = default_value
                    else:
                        ret_val = op(lhs, rhs[i])
                    out_arr[i] = ret_val
                return out_arr

            return impl

    return overload_date_arr_cmp


@overload(operator.setitem, no_unliteral=True)
def dt_timedelta_arr_setitem(A, ind, val):
    if A != types.Array(bodo.types.timedelta64ns, 1, "C") or val != pd_timedelta_type:
        return

    def impl(A, ind, val):  # pragma: no cover
        A[ind] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val.value)

    return impl


timedelta_unsupported_attrs = [
    "asm8",
    "resolution_string",
    "freq",
    "is_populated",
]

timedelta_unsupported_methods = [
    "isoformat",
]

# class methods/attrs handled in untyped pass
# pandas.Timedelta.max
# pandas.Timedelta.min
# pandas.Timedelta.resolution


def _install_pd_timedelta_unsupported():
    for attr_name in timedelta_unsupported_attrs:
        full_name = "pandas.Timedelta." + attr_name
        overload_unsupported_attribute(PDTimeDeltaType, attr_name, full_name)
    for fname in timedelta_unsupported_methods:
        full_name = "pandas.Timedelta." + fname
        overload_unsupported_method(PDTimeDeltaType, fname, full_name)


_install_pd_timedelta_unsupported()
