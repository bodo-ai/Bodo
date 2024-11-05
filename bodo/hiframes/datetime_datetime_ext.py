import datetime

import numba
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_cast,
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

import bodo

"""
Implementation is based on
https://github.com/python/cpython/blob/39a5c889d30d03a88102e56f03ee0c95db198fb3/Lib/datetime.py
"""


class DatetimeDatetimeType(types.Type):
    def __init__(self):
        super().__init__(name="DatetimeDatetimeType()")


datetime_datetime_type = DatetimeDatetimeType()
types.datetime_datetime_type = datetime_datetime_type


@typeof_impl.register(datetime.datetime)
def typeof_datetime_datetime(val, c):
    return datetime_datetime_type


@register_model(DatetimeDatetimeType)
class DatetimeDateTimeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("year", types.int64),
            ("month", types.int64),
            ("day", types.int64),
            ("hour", types.int64),
            ("minute", types.int64),
            ("second", types.int64),
            ("microsecond", types.int64),
        ]
        super().__init__(dmm, fe_type, members)


@box(DatetimeDatetimeType)
def box_datetime_datetime(typ, val, c):
    datetime_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    year_obj = c.pyapi.long_from_longlong(datetime_struct.year)
    month_obj = c.pyapi.long_from_longlong(datetime_struct.month)
    day_obj = c.pyapi.long_from_longlong(datetime_struct.day)
    hour_obj = c.pyapi.long_from_longlong(datetime_struct.hour)
    minute_obj = c.pyapi.long_from_longlong(datetime_struct.minute)
    second_obj = c.pyapi.long_from_longlong(datetime_struct.second)
    microsecond_obj = c.pyapi.long_from_longlong(datetime_struct.microsecond)

    datetime_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.datetime))
    res = c.pyapi.call_function_objargs(
        datetime_obj,
        (
            year_obj,
            month_obj,
            day_obj,
            hour_obj,
            minute_obj,
            second_obj,
            microsecond_obj,
        ),
    )
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)
    c.pyapi.decref(datetime_obj)
    return res


@unbox(DatetimeDatetimeType)
def unbox_datetime_datetime(typ, val, c):
    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")
    hour_obj = c.pyapi.object_getattr_string(val, "hour")
    minute_obj = c.pyapi.object_getattr_string(val, "minute")
    second_obj = c.pyapi.object_getattr_string(val, "second")
    microsecond_obj = c.pyapi.object_getattr_string(val, "microsecond")

    datetime_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    datetime_struct.year = c.pyapi.long_as_longlong(year_obj)
    datetime_struct.month = c.pyapi.long_as_longlong(month_obj)
    datetime_struct.day = c.pyapi.long_as_longlong(day_obj)
    datetime_struct.hour = c.pyapi.long_as_longlong(hour_obj)
    datetime_struct.minute = c.pyapi.long_as_longlong(minute_obj)
    datetime_struct.second = c.pyapi.long_as_longlong(second_obj)
    datetime_struct.microsecond = c.pyapi.long_as_longlong(microsecond_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(datetime_struct._getvalue(), is_error=is_error)


@lower_constant(DatetimeDatetimeType)
def constant_datetime(context, builder, ty, pyval):
    # Extracting constants. Inspired from @lower_constant(types.Complex)
    # in numba/numba/targets/numbers.py
    year = context.get_constant(types.int64, pyval.year)
    month = context.get_constant(types.int64, pyval.month)
    day = context.get_constant(types.int64, pyval.day)
    hour = context.get_constant(types.int64, pyval.hour)
    minute = context.get_constant(types.int64, pyval.minute)
    second = context.get_constant(types.int64, pyval.second)
    microsecond = context.get_constant(types.int64, pyval.microsecond)

    return lir.Constant.literal_struct(
        [year, month, day, hour, minute, second, microsecond]
    )


@overload(datetime.datetime, no_unliteral=True)
def datetime_datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0):
    # TODO: tzinfo is currently not supported
    def impl_datetime(
        year, month, day, hour=0, minute=0, second=0, microsecond=0
    ):  # pragma: no cover
        return init_datetime(year, month, day, hour, minute, second, microsecond)

    return impl_datetime


@intrinsic
def init_datetime(typingctx, year, month, day, hour, minute, second, microsecond):
    def codegen(context, builder, signature, args):
        typ = signature.return_type
        datetime_struct = cgutils.create_struct_proxy(typ)(context, builder)
        datetime_struct.year = args[0]
        datetime_struct.month = args[1]
        datetime_struct.day = args[2]
        datetime_struct.hour = args[3]
        datetime_struct.minute = args[4]
        datetime_struct.second = args[5]
        datetime_struct.microsecond = args[6]

        return datetime_struct._getvalue()

    return (
        DatetimeDatetimeType()(year, month, day, hour, minute, second, microsecond),
        codegen,
    )


make_attribute_wrapper(DatetimeDatetimeType, "year", "_year")
make_attribute_wrapper(DatetimeDatetimeType, "month", "_month")
make_attribute_wrapper(DatetimeDatetimeType, "day", "_day")
make_attribute_wrapper(DatetimeDatetimeType, "hour", "_hour")
make_attribute_wrapper(DatetimeDatetimeType, "minute", "_minute")
make_attribute_wrapper(DatetimeDatetimeType, "second", "_second")
make_attribute_wrapper(DatetimeDatetimeType, "microsecond", "_microsecond")


@overload_attribute(DatetimeDatetimeType, "year")
def datetime_get_year(dt):
    def impl(dt):  # pragma: no cover
        return dt._year

    return impl


@overload_attribute(DatetimeDatetimeType, "month")
def datetime_get_month(dt):
    def impl(dt):  # pragma: no cover
        return dt._month

    return impl


@overload_attribute(DatetimeDatetimeType, "day")
def datetime_get_day(dt):
    def impl(dt):  # pragma: no cover
        return dt._day

    return impl


@overload_attribute(DatetimeDatetimeType, "hour")
def datetime_get_hour(dt):
    def impl(dt):  # pragma: no cover
        return dt._hour

    return impl


@overload_attribute(DatetimeDatetimeType, "minute")
def datetime_get_minute(dt):
    def impl(dt):  # pragma: no cover
        return dt._minute

    return impl


@overload_attribute(DatetimeDatetimeType, "second")
def datetime_get_second(dt):
    def impl(dt):  # pragma: no cover
        return dt._second

    return impl


@overload_attribute(DatetimeDatetimeType, "microsecond")
def datetime_get_microsecond(dt):
    def impl(dt):  # pragma: no cover
        return dt._microsecond

    return impl


@overload_method(DatetimeDatetimeType, "date", no_unliteral=True)
def date(dt):
    """Return the date part."""

    # TODO: support datetime.datetime.time() method once datetime.time is implemented
    def impl(dt):  # pragma: no cover
        return datetime.date(dt.year, dt.month, dt.day)

    return impl


@register_jitable
def now_impl():  # pragma: no cover
    """Internal call to support datetime.datetime.now().
    Untyped pass replaces datetime.date.now() with this call since class methods are
    not supported in Numba's typing
    """
    with bodo.objmode(d="datetime_datetime_type"):
        d = datetime.datetime.now()
    return d


@register_jitable
def today_impl():  # pragma: no cover
    """Internal call to support datetime.datetime.today().
    Untyped pass replaces datetime.datetime.today() with this call since class methods are
    not supported in Numba's typing
    """
    with bodo.objmode(d="datetime_datetime_type"):
        d = datetime.datetime.today()
    return d


@register_jitable
def strptime_impl(date_string, dtformat):  # pragma: no cover
    """Internal call to support datetime.datetime.strptime().
    Untyped pass replaces datetime.datetime.strptime() with this call since class methods are
    not supported in Numba's typing
    """
    with bodo.objmode(d="datetime_datetime_type"):
        d = datetime.datetime.strptime(date_string, dtformat)
    return d


@register_jitable
def _cmp(x, y):  # pragma: no cover
    return 0 if x == y else 1 if x > y else -1


def create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_datetime_type."""

    def overload_datetime_cmp(lhs, rhs):
        if lhs == datetime_datetime_type and rhs == datetime_datetime_type:

            def impl(lhs, rhs):  # pragma: no cover
                y, y2 = lhs.year, rhs.year
                m, m2 = lhs.month, rhs.month
                d, d2 = lhs.day, rhs.day
                h, h2 = lhs.hour, rhs.hour
                mi, mi2 = lhs.minute, rhs.minute
                s, s2 = lhs.second, rhs.second
                us, us2 = lhs.microsecond, rhs.microsecond
                return op(
                    _cmp((y, m, d, h, mi, s, us), (y2, m2, d2, h2, mi2, s2, us2)), 0
                )

            return impl

    return overload_datetime_cmp


def overload_sub_operator_datetime_datetime(lhs, rhs):
    if lhs == datetime_datetime_type and rhs == datetime_datetime_type:

        def impl(lhs, rhs):  # pragma: no cover
            days1 = lhs.toordinal()
            days2 = rhs.toordinal()
            secs1 = lhs.second + lhs.minute * 60 + lhs.hour * 3600
            secs2 = rhs.second + rhs.minute * 60 + rhs.hour * 3600
            base = datetime.timedelta(
                days1 - days2, secs1 - secs2, lhs.microsecond - rhs.microsecond
            )
            return base

        return impl


@lower_cast(
    types.Optional(numba.core.types.NPTimedelta("ns")),
    numba.core.types.NPTimedelta("ns"),
)
@lower_cast(
    types.Optional(numba.core.types.NPDatetime("ns")), numba.core.types.NPDatetime("ns")
)
def optional_dt64_to_dt64(context, builder, fromty, toty, val):
    optval = context.make_helper(builder, fromty, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    with builder.if_else(validbit) as (then, orelse):
        with then:
            res_if = context.cast(builder, optval.data, fromty.type, toty)
            then_bb = builder.block
        with orelse:
            res_else = numba.np.npdatetime.NAT
            orelse_bb = builder.block
    res = builder.phi(res_if.type)
    res.add_incoming(res_if, then_bb)
    res.add_incoming(res_else, orelse_bb)
    return res
