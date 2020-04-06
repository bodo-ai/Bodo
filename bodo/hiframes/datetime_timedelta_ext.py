# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Numba extension support for datetime.timedelta objects and their arrays.
"""
import operator
import datetime
from numba.core import types
from numba.extending import (
    typeof_impl,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    box,
    unbox,
    lower_getattr,
    overload,
    intrinsic,
    overload_attribute,
    overload_method,
    register_jitable,
)
from numba.core import cgutils
import bodo
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type

# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class DatetimeTimeDeltaType(types.Type):
    def __init__(self):
        super(DatetimeTimeDeltaType, self).__init__(name="DatetimeTimeDeltaType()")


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
        super(DatetimeTimeDeltaModel, self).__init__(dmm, fe_type, members)


# 4.Implementing a boxing function for a Numba type using the @box decorator
@box(DatetimeTimeDeltaType)
def box_datetime_date(typ, val, c):
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


# 6. Implement the constructor
@overload(datetime.timedelta)
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


@overload_method(DatetimeTimeDeltaType, "total_seconds")
def total_seconds(td):
    """Total seconds in the duration."""

    def impl(td):  # pragma: no cover
        return ((td._days * 86400 + td._seconds) * 10 ** 6 + td._microseconds) / 10 ** 6

    return impl


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


@overload(operator.add)
def timedelta_add(lhs, rhs):
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


@overload(operator.sub)
def timedelta_sub(lhs, rhs):
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


@overload(operator.mul)
def timedelta_mul(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == types.int64:

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs.days * rhs
            s = lhs.seconds * rhs
            us = lhs.microseconds * rhs
            return datetime.timedelta(d, s, us)

        return impl

    elif lhs == types.int64 and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            d = lhs * rhs.days
            s = lhs * rhs.seconds
            us = lhs * rhs.microseconds
            return datetime.timedelta(d, s, us)

        return impl


@overload(operator.floordiv)
def timedelta_floordiv(lhs, rhs):
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


@overload(operator.truediv)
def timedelta_truediv(lhs, rhs):
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


@overload(operator.mod)
def timedelta_mod(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            r = _to_microseconds(lhs) % _to_microseconds(rhs)
            return datetime.timedelta(0, 0, r)

        return impl


@overload(operator.eq)
def timedelta_eq(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret == 0

        return impl


@overload(operator.ne)
def timedelta_ne(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret != 0

        return impl


@overload(operator.le)
def timedelta_le(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret <= 0

        return impl


@overload(operator.lt)
def timedelta_lt(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret < 0

        return impl


@overload(operator.ge)
def timedelta_ge(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret >= 0

        return impl


@overload(operator.gt)
def timedelta_gt(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            ret = _cmp(_getstate(lhs), _getstate(rhs))
            return ret > 0

        return impl


@overload(operator.neg)
def timedelta_neg(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return datetime.timedelta(-lhs.days, -lhs.seconds, -lhs.microseconds)

        return impl


@overload(operator.pos)
def timedelta_pos(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            return lhs

        return impl


@overload(divmod)
def timedelta_divmod(lhs, rhs):
    if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            q, r = divmod(_to_microseconds(lhs), _to_microseconds(rhs))
            return q, datetime.timedelta(0, 0, r)

        return impl


@overload(abs)
def timedelta_abs(lhs):
    if lhs == datetime_timedelta_type:

        def impl(lhs):  # pragma: no cover
            if lhs.days < 0:
                return -lhs
            else:
                return lhs

        return impl

