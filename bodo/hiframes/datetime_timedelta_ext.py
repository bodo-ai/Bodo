# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Numba extension support for datetime.timedelta objects and their arrays.
"""
import datetime
import numba
from numba import types
from numba.typing import signature
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
)
from numba import cgutils
from numba.typing.templates import signature
from llvmlite import ir as lir
import bodo

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
    ):
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
    def impl(td):
        return td._days

    return impl


@overload_attribute(DatetimeTimeDeltaType, "seconds")
def timedelta_get_seconds(td):
    def impl(td):
        return td._seconds

    return impl


@overload_attribute(DatetimeTimeDeltaType, "microseconds")
def timedelta_get_microseconds(td):
    def impl(td):
        return td._microseconds

    return impl

