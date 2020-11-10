# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implement support for the various classes in pd.tseries.offsets.
"""
import operator

import llvmlite.binding as ll
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
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (
    convert_datetime64_to_timestamp,
    get_days_in_month,
    pandas_timestamp_type,
)
from bodo.libs import hdatetime_ext
from bodo.utils.indexing import unoptional
from bodo.utils.typing import is_overload_none

ll.add_symbol("box_date_offset", hdatetime_ext.box_date_offset)
ll.add_symbol("unbox_date_offset", hdatetime_ext.unbox_date_offset)


# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class MonthEndType(types.Type):
    """Class for pd.tseries.offset.MonthEnd"""

    def __init__(self):
        super(MonthEndType, self).__init__(name="MonthEndType()")


month_end_type = MonthEndType()


# 2.Teach Numba how to infer the Numba type of Python values of a certain class,
# using typeof_impl.register
@typeof_impl.register(pd.tseries.offsets.MonthEnd)
def typeof_month_end(val, c):
    return month_end_type


# 3.Define the data model for a Numba type using StructModel and register_model
@register_model(MonthEndType)
class MonthEndModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("n", types.int64), ("normalize", types.boolean)]
        super(MonthEndModel, self).__init__(dmm, fe_type, members)


# 4.Implementing a boxing function for a Numba type using the @box decorator


@box(MonthEndType)
def box_month_end(typ, val, c):
    month_end = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    n_obj = c.pyapi.long_from_longlong(month_end.n)
    normalize_obj = c.pyapi.from_native_value(
        types.boolean, month_end.normalize, c.env_manager
    )
    month_end_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(pd.tseries.offsets.MonthEnd)
    )
    res = c.pyapi.call_function_objargs(month_end_obj, (n_obj, normalize_obj))
    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)
    c.pyapi.decref(month_end_obj)
    return res


# 5.Implementing an unboxing function for a Numba type
# using the @unbox decorator and the NativeValue class
@unbox(MonthEndType)
def unbox_month_end(typ, val, c):

    n_obj = c.pyapi.object_getattr_string(val, "n")
    normalize_obj = c.pyapi.object_getattr_string(val, "normalize")

    n = c.pyapi.long_as_longlong(n_obj)
    normalize = c.pyapi.to_native_value(types.bool_, normalize_obj).value

    month_end = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    month_end.n = n
    month_end.normalize = normalize

    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(month_end._getvalue(), is_error=is_error)


@lower_constant(MonthEndType)
def lower_constant_month_end(context, builder, ty, pyval):
    n = context.get_constant(types.int64, pyval.n)
    normalize = context.get_constant(types.boolean, pyval.normalize)
    month_end = cgutils.create_struct_proxy(ty)(context, builder)
    month_end.n = n
    month_end.normalize = normalize
    return month_end._getvalue()


# 6. Implement the constructor
@overload(pd.tseries.offsets.MonthEnd, no_unliteral=True)
def MonthEnd(
    n=1,
    normalize=False,
):
    def impl(
        n=1,
        normalize=False,
    ):  # pragma: no cover
        return init_month_end(n, normalize)

    return impl


@intrinsic
def init_month_end(typingctx, n, normalize):
    def codegen(context, builder, signature, args):  # pragma: no cover
        typ = signature.return_type
        month_end = cgutils.create_struct_proxy(typ)(context, builder)
        month_end.n = args[0]
        month_end.normalize = args[1]
        return month_end._getvalue()

    return MonthEndType()(n, normalize), codegen


# 2nd arg is used in LLVM level, 3rd arg is used in python level
make_attribute_wrapper(MonthEndType, "n", "_n")
make_attribute_wrapper(MonthEndType, "normalize", "_normalize")


# Implement the getters
@overload_attribute(MonthEndType, "n")
def month_end_get_n(me):
    def impl(me):  # pragma: no cover
        return me._n

    return impl


@overload_attribute(MonthEndType, "normalize")
def month_end_get_normalize(me):
    def impl(me):  # pragma: no cover
        return me._normalize

    return impl


# TODO: Implement the rest of the getters

# Implement the necessary operators


# Code is generalized and split across multiple functions in Pandas
# General structure: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L2137
# Changing n: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L3938
# Shifting the month: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L3711
@register_jitable
def calculate_month_end_date(year, month, day, n):  # pragma: no cover
    """Inputs: A date described by a year, month, and
    day and the number of month ends to move by, n.
    Returns: The new date in year, month, day
    """
    if n > 0:
        # Determine the last day of this month
        month_end = get_days_in_month(year, month)
        if month_end > day:
            n -= 1
    # Alter the number of months, then year. Note this is 1 indexed.
    month = month + n
    # Subtract 1 to map (1, 12) to (0, 11), then add the 1 back.
    month -= 1
    year += month // 12
    month = (month % 12) + 1
    day = get_days_in_month(year, month)
    return year, month, day


@overload(operator.add, no_unliteral=True)
def month_end_add_scalar(lhs, rhs):
    """Implement all of the relevant scalar types additions.
    These will be reused to implement arrays.
    """
    # rhs is a datetime
    if lhs == month_end_type and rhs == datetime_datetime_type:

        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_end_date(
                rhs.year, rhs.month, rhs.day, lhs.n
            )
            if lhs.normalize:
                return pd.Timestamp(year=year, month=month, day=day)
            else:
                return pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                    hour=rhs.hour,
                    minute=rhs.minute,
                    second=rhs.second,
                    microsecond=rhs.microsecond,
                )

        return impl

    # rhs is a timestamp
    if lhs == month_end_type and rhs == pandas_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_end_date(
                rhs.year, rhs.month, rhs.day, lhs.n
            )
            if lhs.normalize:
                return pd.Timestamp(year=year, month=month, day=day)
            else:
                return pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                    hour=rhs.hour,
                    minute=rhs.minute,
                    second=rhs.second,
                    microsecond=rhs.microsecond,
                    nanosecond=rhs.nanosecond,
                )

        return impl

    # rhs is a datetime.date
    if lhs == month_end_type and rhs == datetime_date_type:
        # No need to consider normalize because datetime only goes down to day.
        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_end_date(
                rhs.year, rhs.month, rhs.day, lhs.n
            )
            return pd.Timestamp(year=year, month=month, day=day)

        return impl

    # rhs is the offset
    if (
        lhs in [datetime_datetime_type, pandas_timestamp_type, datetime_date_type]
        and rhs == month_end_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl


@overload(operator.sub, no_unliteral=True)
def month_end_sub(lhs, rhs):
    """Implement all of the relevant scalar types subtractions.
    These will be reused to implement arrays.
    """
    # lhs date/datetime/timestamp and rhs month_end
    if (
        lhs in [datetime_datetime_type, pandas_timestamp_type, datetime_date_type]
    ) and rhs == month_end_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl


# TODO: Support operators with arrays


@overload(operator.neg, no_unliteral=True)
def month_end_neg(lhs):
    if lhs == month_end_type:

        def impl(lhs):  # pragma: no cover
            return pd.tseries.offsets.MonthEnd(-lhs.n, lhs.normalize)

        return impl


class DateOffsetType(types.Type):
    """Class for pd.tseries.offset.DateOffset"""

    def __init__(self):
        super(DateOffsetType, self).__init__(name="DateOffsetType()")


date_offset_type = DateOffsetType()

date_offset_fields = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "microseconds",
    "nanoseconds",
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
]


@typeof_impl.register(pd.tseries.offsets.DateOffset)
def type_of_date_offset(val, c):
    return date_offset_type


@register_model(DateOffsetType)
class DateOffsetModel(models.StructModel):
    """Date offsets support the use of n, but this argument is discouraged.
    Functionality is mostly implemented between kwargs that replace and those
    that add. Replace have no s (i.e. year) whereas add have an s (i.e. years).
    The proper behavior is that fields are first replaced and then addition occurs.
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("n", types.int64),
            ("normalize", types.boolean),
            # Fields that add to offset value
            ("years", types.int64),
            ("months", types.int64),
            ("weeks", types.int64),
            ("days", types.int64),
            ("hours", types.int64),
            ("minutes", types.int64),
            ("seconds", types.int64),
            ("microseconds", types.int64),
            ("nanoseconds", types.int64),
            # Fields that replace offset value
            ("year", types.int64),
            ("month", types.int64),
            ("day", types.int64),
            ("weekday", types.int64),
            ("hour", types.int64),
            ("minute", types.int64),
            ("second", types.int64),
            ("microsecond", types.int64),
            ("nanosecond", types.int64),
            # Keyword to distinguish if any fields were passed in.
            # No kwds has different behavior
            ("has_kws", types.boolean),
        ]
        super(DateOffsetModel, self).__init__(dmm, fe_type, members)


@box(DateOffsetType)
def box_date_offset(typ, val, c):
    """Boxes a native date offset as a Python DateOffset object. The has_kws field
    is used to determine if fields should be transferred (except the nano values).
    This is because there is different behavior on add/sub when there is a non_nano
    keyword. Nano values are transferred anyways because they do not impact correctness.
    """
    date_offset = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # Allocate stack array for extracting C++ values
    fields_arr = c.builder.alloca(
        lir.IntType(64), size=lir.Constant(lir.IntType(64), 18)
    )
    for i, field in enumerate(date_offset_fields):
        c.builder.store(
            getattr(date_offset, field),
            c.builder.inttoptr(
                c.builder.add(
                    c.builder.ptrtoint(fields_arr, lir.IntType(64)),
                    lir.Constant(lir.IntType(64), 8 * i),
                ),
                lir.IntType(64).as_pointer(),
            ),
        )
    fnty = lir.FunctionType(
        c.pyapi.pyobj,
        [
            lir.IntType(64),
            lir.IntType(1),
            lir.IntType(64).as_pointer(),
            lir.IntType(1),
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(fnty, name="box_date_offset")
    date_offset_obj = c.builder.call(
        fn_get,
        [
            date_offset.n,
            date_offset.normalize,
            fields_arr,
            date_offset.has_kws,
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return date_offset_obj


@unbox(DateOffsetType)
def unbox_date_offset(typ, val, c):

    n_obj = c.pyapi.object_getattr_string(val, "n")
    normalize_obj = c.pyapi.object_getattr_string(val, "normalize")
    n = c.pyapi.long_as_longlong(n_obj)
    normalize = c.pyapi.to_native_value(types.bool_, normalize_obj).value

    # Allocate stack array for extracting C++ values
    fields_arr = c.builder.alloca(
        lir.IntType(64), size=lir.Constant(lir.IntType(64), 18)
    )

    # function signature of unbox_date_offset
    fnty = lir.FunctionType(
        lir.IntType(1),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64).as_pointer(),
        ],
    )
    fn = c.builder.module.get_or_insert_function(fnty, name="unbox_date_offset")
    has_kws = c.builder.call(
        fn,
        [
            val,
            fields_arr,
        ],
    )

    date_offset = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    date_offset.n = n
    date_offset.normalize = normalize
    # Manually load the fields from the array in LLVM
    for i, field in enumerate(date_offset_fields):
        setattr(
            date_offset,
            field,
            c.builder.load(
                c.builder.inttoptr(
                    c.builder.add(
                        c.builder.ptrtoint(fields_arr, lir.IntType(64)),
                        lir.Constant(lir.IntType(64), 8 * i),
                    ),
                    lir.IntType(64).as_pointer(),
                )
            ),
        )
    date_offset.has_kws = has_kws

    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(date_offset._getvalue(), is_error=is_error)


@lower_constant(DateOffsetType)
def lower_constant_date_offset(context, builder, ty, pyval):
    """Lowers a constant DateOffset python type to a native type.
    has_kws is determined by checking if any of the non_nanos attributes
    exist.
    """
    n = context.get_constant(types.int64, pyval.n)
    normalize = context.get_constant(types.boolean, pyval.normalize)
    date_offset = cgutils.create_struct_proxy(ty)(context, builder)
    date_offset.n = n
    date_offset.normalize = normalize
    has_kws = False
    # Default value if the value doesn't exist
    default_values = [0] * 9 + [-1] * 9
    for i, field in enumerate(date_offset_fields):
        if hasattr(pyval, field):
            setattr(
                date_offset,
                field,
                context.get_constant(types.int64, getattr(pyval, field)),
            )
            if field != "nanoseconds" and field != "nanosecond":
                has_kws = True
        else:
            setattr(
                date_offset, field, context.get_constant(types.int64, default_values[i])
            )
    date_offset.has_kws = context.get_constant(types.boolean, has_kws)
    return date_offset._getvalue()


@overload(pd.tseries.offsets.DateOffset, no_unliteral=True)
def DateOffset(
    n=1,
    normalize=False,
    years=None,
    months=None,
    weeks=None,
    days=None,
    hours=None,
    minutes=None,
    seconds=None,
    microseconds=None,
    nanoseconds=None,
    year=None,
    month=None,
    day=None,
    weekday=None,
    hour=None,
    minute=None,
    second=None,
    microsecond=None,
    nanosecond=None,
):
    # has_kws is true if any value that is not n, normalize, nanoseconds, or nanosecond is passed in
    # Set kws to None to allow checking for if kws were passed in
    has_kws = False
    kws_args = [
        years,
        months,
        weeks,
        days,
        hours,
        minutes,
        seconds,
        microseconds,
        year,
        month,
        day,
        weekday,
        hour,
        minute,
        second,
        microsecond,
    ]
    for kws_arg in kws_args:
        if not is_overload_none(kws_arg):
            has_kws = True
            break

    def impl(
        n=1,
        normalize=False,
        years=None,
        months=None,
        weeks=None,
        days=None,
        hours=None,
        minutes=None,
        seconds=None,
        microseconds=None,
        nanoseconds=None,
        year=None,
        month=None,
        day=None,
        weekday=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
    ):  # pragma: no cover
        # Convert any none values to default int values
        years = 0 if years is None else years
        months = 0 if months is None else months
        weeks = 0 if weeks is None else weeks
        days = 0 if days is None else days
        hours = 0 if hours is None else hours
        minutes = 0 if minutes is None else minutes
        seconds = 0 if seconds is None else seconds
        microseconds = 0 if microseconds is None else microseconds
        nanoseconds = 0 if nanoseconds is None else nanoseconds
        year = -1 if year is None else year
        month = -1 if month is None else month
        weekday = -1 if weekday is None else weekday
        day = -1 if day is None else day
        hour = -1 if hour is None else hour
        minute = -1 if minute is None else minute
        second = -1 if second is None else second
        microsecond = -1 if microsecond is None else microsecond
        nanosecond = -1 if nanosecond is None else nanosecond
        return init_date_offset(
            n,
            normalize,
            unoptional(years),
            unoptional(months),
            unoptional(weeks),
            unoptional(days),
            unoptional(hours),
            unoptional(minutes),
            unoptional(seconds),
            unoptional(microseconds),
            unoptional(nanoseconds),
            unoptional(year),
            unoptional(month),
            unoptional(day),
            unoptional(weekday),
            unoptional(hour),
            unoptional(minute),
            unoptional(second),
            unoptional(microsecond),
            unoptional(nanosecond),
            has_kws,
        )

    return impl


@intrinsic
def init_date_offset(
    typingctx,
    n,
    normalize,
    years,
    months,
    weeks,
    days,
    hours,
    minutes,
    seconds,
    microseconds,
    nanoseconds,
    year,
    month,
    day,
    weekday,
    hour,
    minute,
    second,
    microsecond,
    nanosecond,
    has_kws,
):
    def codegen(context, builder, signature, args):  # pragma: no cover
        typ = signature.return_type
        date_offset = cgutils.create_struct_proxy(typ)(context, builder)
        date_offset.n = args[0]
        date_offset.normalize = args[1]
        date_offset.years = args[2]
        date_offset.months = args[3]
        date_offset.weeks = args[4]
        date_offset.days = args[5]
        date_offset.hours = args[6]
        date_offset.minutes = args[7]
        date_offset.seconds = args[8]
        date_offset.microseconds = args[9]
        date_offset.nanoseconds = args[10]
        date_offset.year = args[11]
        date_offset.month = args[12]
        date_offset.day = args[13]
        date_offset.weekday = args[14]
        date_offset.hour = args[15]
        date_offset.minute = args[16]
        date_offset.second = args[17]
        date_offset.microsecond = args[18]
        date_offset.nanosecond = args[19]
        date_offset.has_kws = args[20]
        return date_offset._getvalue()

    return (
        DateOffsetType()(
            n,
            normalize,
            years,
            months,
            weeks,
            days,
            hours,
            minutes,
            seconds,
            microseconds,
            nanoseconds,
            year,
            month,
            day,
            weekday,
            hour,
            minute,
            second,
            microsecond,
            nanosecond,
            has_kws,
        ),
        codegen,
    )


make_attribute_wrapper(DateOffsetType, "n", "_n")
make_attribute_wrapper(DateOffsetType, "normalize", "_normalize")
make_attribute_wrapper(DateOffsetType, "years", "_years")
make_attribute_wrapper(DateOffsetType, "months", "_months")
make_attribute_wrapper(DateOffsetType, "weeks", "_weeks")
make_attribute_wrapper(DateOffsetType, "days", "_days")
make_attribute_wrapper(DateOffsetType, "hours", "_hours")
make_attribute_wrapper(DateOffsetType, "minutes", "_minutes")
make_attribute_wrapper(DateOffsetType, "seconds", "_seconds")
make_attribute_wrapper(DateOffsetType, "microseconds", "_microseconds")
make_attribute_wrapper(DateOffsetType, "nanoseconds", "_nanoseconds")
make_attribute_wrapper(DateOffsetType, "year", "_year")
make_attribute_wrapper(DateOffsetType, "month", "_month")
make_attribute_wrapper(DateOffsetType, "weekday", "_weekday")
make_attribute_wrapper(DateOffsetType, "day", "_day")
make_attribute_wrapper(DateOffsetType, "hour", "_hour")
make_attribute_wrapper(DateOffsetType, "minute", "_minute")
make_attribute_wrapper(DateOffsetType, "second", "_second")
make_attribute_wrapper(DateOffsetType, "microsecond", "_microsecond")
make_attribute_wrapper(DateOffsetType, "nanosecond", "_nanosecond")
make_attribute_wrapper(DateOffsetType, "has_kws", "_has_kws")


# Add implementation derived from https://dateutil.readthedocs.io/en/stable/relativedelta.html
@register_jitable
def relative_delta_addition(dateoffset, ts):  # pragma: no cover
    """Performs the general addition performed by relative delta according to the
    passed in DateOffset
    """
    if dateoffset._has_kws:
        sign = -1 if dateoffset._n < 0 else 1
        for _ in range(np.abs(dateoffset._n)):
            year = ts.year
            month = ts.month
            day = ts.day
            hour = ts.hour
            minute = ts.minute
            second = ts.second
            microsecond = ts.microsecond
            nanosecond = ts.nanosecond

            if dateoffset._year != -1:
                year = dateoffset._year
            year += sign * dateoffset._years
            if dateoffset._month != -1:
                month = dateoffset._month
            month += sign * dateoffset._months

            year, month, new_day = calculate_month_end_date(year, month, day, 0)
            # If the day is out of bounds roll back to a legal date
            if day > new_day:
                day = new_day

            # Remaining values can be handled with a timedelta to give the same
            # effect as happening 1 at a time
            if dateoffset._day != -1:
                day = dateoffset._day
            if dateoffset._hour != -1:
                hour = dateoffset._hour
            if dateoffset._minute != -1:
                minute = dateoffset._minute
            if dateoffset._second != -1:
                second = dateoffset._second
            if dateoffset._microsecond != -1:
                microsecond = dateoffset._microsecond

            ts = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
                nanosecond=nanosecond,
            )

            # Pandas ignores nanosecond/nanoseconds because it uses relative delta
            td = pd.Timedelta(
                days=dateoffset._days + 7 * dateoffset._weeks,
                hours=dateoffset._hours,
                minutes=dateoffset._minutes,
                seconds=dateoffset._seconds,
                microseconds=dateoffset._microseconds,
            )
            if sign == -1:
                td = -td

            ts = ts + td

            if dateoffset._weekday != -1:
                # roll foward by determining the difference in day of the week
                # We only accept labeling a day of the week 0..6
                curr_weekday = ts.weekday()
                days_forward = (dateoffset._weekday - curr_weekday) % 7
                ts = ts + pd.Timedelta(days=days_forward)
        return ts
    else:
        return pd.Timedelta(days=dateoffset._n) + ts


@overload(operator.add, no_unliteral=True)
def date_offset_add_scalar(lhs, rhs):
    """Implement all of the relevant scalar types additions.
    These will be reused to implement arrays.
    """
    # rhs is a timestamp
    if lhs == date_offset_type and rhs == pandas_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            ts = relative_delta_addition(lhs, rhs)
            if lhs._normalize:
                return ts.normalize()
            return ts

        return impl

    # rhs is a date or datetime
    if lhs == date_offset_type and rhs in [datetime_date_type, datetime_datetime_type]:

        def impl(lhs, rhs):  # pragma: no cover
            ts = relative_delta_addition(lhs, pd.Timestamp(rhs))
            if lhs._normalize:
                return ts.normalize()
            return ts

        return impl

    # offset is the rhs
    if (
        lhs in [datetime_datetime_type, pandas_timestamp_type, datetime_date_type]
        and rhs == date_offset_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl


@overload(operator.sub, no_unliteral=True)
def month_end_sub(lhs, rhs):
    """Implement all of the relevant scalar types subtractions.
    These will be reused to implement arrays.
    """
    # lhs date/datetime/timestamp and rhs date_offset
    if (
        lhs in [datetime_datetime_type, pandas_timestamp_type, datetime_date_type]
    ) and rhs == date_offset_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl


@overload(operator.neg, no_unliteral=True)
def date_offset_neg(lhs):
    if lhs == date_offset_type:

        def impl(lhs):  # pragma: no cover
            # Negate only n
            n = -lhs._n
            normalize = lhs._normalize
            nanoseconds = lhs._nanoseconds
            nanosecond = lhs._nanosecond
            # Make sure has_kws behavior doesn't change
            if lhs._has_kws:
                years = lhs._years
                months = lhs._months
                weeks = lhs._weeks
                days = lhs._days
                hours = lhs._hours
                minutes = lhs._minutes
                seconds = lhs._seconds
                microseconds = lhs._microseconds
                year = lhs._year
                month = lhs._month
                day = lhs._day
                weekday = lhs._weekday
                hour = lhs._hour
                minute = lhs._minute
                second = lhs._second
                microsecond = lhs._microsecond
                return pd.tseries.offsets.DateOffset(
                    n,
                    normalize,
                    years,
                    months,
                    weeks,
                    days,
                    hours,
                    minutes,
                    seconds,
                    microseconds,
                    nanoseconds,
                    year,
                    month,
                    day,
                    weekday,
                    hour,
                    minute,
                    second,
                    microsecond,
                    nanosecond,
                )
            else:
                return pd.tseries.offsets.DateOffset(
                    n, normalize, nanoseconds=nanoseconds, nanosecond=nanosecond
                )

        return impl


def is_offsets_type(val):
    """Function containing all the support offset types"""
    return val in [date_offset_type, month_end_type]
