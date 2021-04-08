# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implement support for the various classes in pd.tseries.offsets.
"""
import operator

import llvmlite.binding as ll
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
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.pd_timestamp_ext import get_days_in_month, pd_timestamp_type
from bodo.libs import hdatetime_ext
from bodo.utils.typing import BodoError, is_overload_none

ll.add_symbol("box_date_offset", hdatetime_ext.box_date_offset)
ll.add_symbol("unbox_date_offset", hdatetime_ext.unbox_date_offset)


# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class MonthBeginType(types.Type):
    """Class for pd.tseries.offset.MonthBegin"""

    def __init__(self):
        super(MonthBeginType, self).__init__(name="MonthBeginType()")


month_begin_type = MonthBeginType()


# 2.Teach Numba how to infer the Numba type of Python values of a certain class,
# using typeof_impl.register
@typeof_impl.register(pd.tseries.offsets.MonthBegin)
def typeof_month_begin(val, c):
    return month_begin_type


# 3.Define the data model for a Numba type using StructModel and register_model
@register_model(MonthBeginType)
class MonthBeginModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("n", types.int64), ("normalize", types.boolean)]
        super(MonthBeginModel, self).__init__(dmm, fe_type, members)


# 4.Implementing a boxing function for a Numba type using the @box decorator
@box(MonthBeginType)
def box_month_begin(typ, val, c):
    month_begin = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    n_obj = c.pyapi.long_from_longlong(month_begin.n)
    normalize_obj = c.pyapi.from_native_value(
        types.boolean, month_begin.normalize, c.env_manager
    )
    month_begin_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(pd.tseries.offsets.MonthBegin)
    )
    res = c.pyapi.call_function_objargs(month_begin_obj, (n_obj, normalize_obj))
    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)
    c.pyapi.decref(month_begin_obj)
    return res


# 5.Implementing an unboxing function for a Numba type
# using the @unbox decorator and the NativeValue class
@unbox(MonthBeginType)
def unbox_month_begin(typ, val, c):
    n_obj = c.pyapi.object_getattr_string(val, "n")
    normalize_obj = c.pyapi.object_getattr_string(val, "normalize")

    n = c.pyapi.long_as_longlong(n_obj)
    normalize = c.pyapi.to_native_value(types.bool_, normalize_obj).value

    month_begin = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    month_begin.n = n
    month_begin.normalize = normalize

    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(month_begin._getvalue(), is_error=is_error)


# 6. Implement the constructor
@overload(pd.tseries.offsets.MonthBegin, no_unliteral=True)
def MonthBegin(
    n=1,
    normalize=False,
):
    def impl(
        n=1,
        normalize=False,
    ):  # pragma: no cover
        return init_month_begin(n, normalize)

    return impl


@intrinsic
def init_month_begin(typingctx, n, normalize):
    def codegen(context, builder, signature, args):  # pragma: no cover
        typ = signature.return_type
        month_begin = cgutils.create_struct_proxy(typ)(context, builder)
        month_begin.n = args[0]
        month_begin.normalize = args[1]
        return month_begin._getvalue()

    return MonthBeginType()(n, normalize), codegen


# 2nd arg is used in LLVM level, 3rd arg is used in python level
make_attribute_wrapper(MonthBeginType, "n", "n")
make_attribute_wrapper(MonthBeginType, "normalize", "normalize")


# Code is generalized and split across multiple functions in Pandas
# General structure: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L2137
# Changing n: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L3938
# Shifting the month: https://github.com/pandas-dev/pandas/blob/41ec93a5a4019462c4e461914007d8b25fb91e48/pandas/_libs/tslibs/offsets.pyx#L3711
@register_jitable
def calculate_month_begin_date(year, month, day, n):  # pragma: no cover
    """Inputs: A date described by a year, month, and
    day and the number of month begins to move by, n.
    Returns: The new date in year, month, day
    """
    # If n < 0, we need to increment n when rolling back to the start of
    # the month. The exception is if we are already at the start of the
    # month.
    if n < 0:
        if day > 1:
            n += 1
    # Alter the number of months by n, then update year. Note this is 1 indexed.
    month = month + n
    # Subtract 1 to map (1, 12) to (0, 11) so we can use the modulo operator.
    # Then add the 1 back to restore 1 indexing for months.
    month -= 1
    year += month // 12
    month = (month % 12) + 1
    day = 1
    return year, month, day


# Implement the necessary operators
def overload_add_operator_month_begin_offset_type(lhs, rhs):
    """Implement all of the relevant scalar types additions.
    These will be reused to implement arrays.
    """
    # rhs is a datetime
    if lhs == month_begin_type and rhs == datetime_datetime_type:

        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_begin_date(
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
    if lhs == month_begin_type and rhs == pd_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_begin_date(
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
    if lhs == month_begin_type and rhs == datetime_date_type:
        # No need to consider normalize because datetime only goes down to day.
        def impl(lhs, rhs):  # pragma: no cover
            year, month, day = calculate_month_begin_date(
                rhs.year, rhs.month, rhs.day, lhs.n
            )
            return pd.Timestamp(year=year, month=month, day=day)

        return impl

    # rhs is the offset
    if (
        lhs in [datetime_datetime_type, pd_timestamp_type, datetime_date_type]
        and rhs == month_begin_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl
    # Raise Bodo error if not supported
    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


class MonthEndType(types.Type):
    """Class for pd.tseries.offset.MonthEnd"""

    def __init__(self):
        super(MonthEndType, self).__init__(name="MonthEndType()")


month_end_type = MonthEndType()


@typeof_impl.register(pd.tseries.offsets.MonthEnd)
def typeof_month_end(val, c):
    return month_end_type


@register_model(MonthEndType)
class MonthEndModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("n", types.int64), ("normalize", types.boolean)]
        super(MonthEndModel, self).__init__(dmm, fe_type, members)


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
make_attribute_wrapper(MonthEndType, "n", "n")
make_attribute_wrapper(MonthEndType, "normalize", "normalize")


# MonthBegin and MonthEnd have the same constant lowering code
@lower_constant(MonthBeginType)
@lower_constant(MonthEndType)
def lower_constant_month_end(context, builder, ty, pyval):
    n = context.get_constant(types.int64, pyval.n)
    normalize = context.get_constant(types.boolean, pyval.normalize)
    month_end = cgutils.create_struct_proxy(ty)(context, builder)
    month_end.n = n
    month_end.normalize = normalize
    return month_end._getvalue()


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


def overload_add_operator_month_end_offset_type(lhs, rhs):
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
    if lhs == month_end_type and rhs == pd_timestamp_type:

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
        lhs in [datetime_datetime_type, pd_timestamp_type, datetime_date_type]
        and rhs == month_end_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl
    # Raise Bodo error if not supported
    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


# TODO: Support operators with arrays


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


make_attribute_wrapper(DateOffsetType, "n", "n")
make_attribute_wrapper(DateOffsetType, "normalize", "normalize")
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
        sign = -1 if dateoffset.n < 0 else 1
        for _ in range(np.abs(dateoffset.n)):
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
        return pd.Timedelta(days=dateoffset.n) + ts


def overload_add_operator_date_offset_type(lhs, rhs):
    """Implement all of the relevant scalar types additions.
    These will be reused to implement arrays.
    """
    # rhs is a timestamp
    if lhs == date_offset_type and rhs == pd_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            ts = relative_delta_addition(lhs, rhs)
            if lhs.normalize:
                return ts.normalize()
            return ts

        return impl

    # rhs is a date or datetime
    if lhs == date_offset_type and rhs in [datetime_date_type, datetime_datetime_type]:

        def impl(lhs, rhs):  # pragma: no cover
            ts = relative_delta_addition(lhs, pd.Timestamp(rhs))
            if lhs.normalize:
                return ts.normalize()
            return ts

        return impl

    # offset is the rhs
    if (
        lhs in [datetime_datetime_type, pd_timestamp_type, datetime_date_type]
        and rhs == date_offset_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl

    # Raise Bodo error if not supported
    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


def overload_sub_operator_offsets(lhs, rhs):
    """Implement all of the relevant scalar types subtractions.
    These will be reused to implement arrays.
    """
    # lhs date/datetime/timestamp and rhs date_offset/month_end_type/week_type
    if (
        lhs in [datetime_datetime_type, pd_timestamp_type, datetime_date_type]
    ) and rhs in [date_offset_type, month_begin_type, month_end_type, week_type]:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl


@overload(operator.neg, no_unliteral=True)
def overload_neg(lhs):
    if lhs == month_begin_type:

        def impl(lhs):  # pragma: no cover
            return pd.tseries.offsets.MonthBegin(-lhs.n, lhs.normalize)

    if lhs == month_end_type:

        def impl(lhs):  # pragma: no cover
            return pd.tseries.offsets.MonthEnd(-lhs.n, lhs.normalize)

    if lhs == week_type:

        def impl(lhs):  # pragma: no cover
            return pd.tseries.offsets.Week(-lhs.n, lhs.normalize, lhs.weekday)

    if lhs == date_offset_type:

        def impl(lhs):  # pragma: no cover
            # Negate only n
            n = -lhs.n
            normalize = lhs.normalize
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
    return val in [date_offset_type, month_begin_type, month_end_type, week_type]


####### tseries.offset.Week #########
# create a new Numba Type
class WeekType(types.Type):
    """ Numba type for tseries.offset.Week. """

    def __init__(self):
        super(WeekType, self).__init__(name="WeekType()")


week_type = WeekType()

# Tell Numba's type inference to map Week's type to WeekType
@typeof_impl.register(pd.tseries.offsets.Week)
def typeof_week(val, c):
    return week_type


# Data Model: Define the data model for the Numba type
@register_model(WeekType)
class WeekModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("n", types.int64),
            ("normalize", types.boolean),
            ("weekday", types.int64),
        ]
        super(WeekModel, self).__init__(dmm, fe_type, members)


# Data Model: expose DM attributes to Numba functions
make_attribute_wrapper(WeekType, "n", "n")
make_attribute_wrapper(WeekType, "normalize", "normalize")
make_attribute_wrapper(WeekType, "weekday", "weekday")

# Constructor Overload
@overload(pd.tseries.offsets.Week, no_unliteral=True)
def Week(
    n=1,
    normalize=False,
    weekday=None,
):
    def impl(
        n=1,
        normalize=False,
        weekday=None,
    ):  # pragma: no cover
        # deal with None to preserve the int type for Numba
        _weekday = -1 if weekday is None else weekday
        return init_week(n, normalize, _weekday)

    return impl


# LLVM helper for constructing the object
@intrinsic
def init_week(typingctx, n, normalize, weekday):
    def codegen(context, builder, signature, args):  # pragma: no cover
        typ = signature.return_type
        week = cgutils.create_struct_proxy(typ)(context, builder)
        week.n = args[0]
        week.normalize = args[1]
        week.weekday = args[2]
        return week._getvalue()

    return WeekType()(n, normalize, weekday), codegen


# Implement the constant creation for Numba 'typespec'
@lower_constant(WeekType)
def lower_constant_week(context, builder, ty, pyval):  # pragma: no cover
    week = cgutils.create_struct_proxy(ty)(context, builder)
    week.n = context.get_constant(types.int64, pyval.n)
    week.normalize = context.get_constant(types.boolean, pyval.normalize)

    if pyval.weekday is not None:
        week.weekday = context.get_constant(types.int64, pyval.weekday)
    else:
        week.weekday = context.get_constant(types.int64, -1)
    return week._getvalue()


# Boxing and Unboxing
@box(WeekType)
def box_week(typ, val, c):  # pragma: no cover
    week = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    n_obj = c.pyapi.long_from_longlong(week.n)
    normalize_obj = c.pyapi.from_native_value(
        types.boolean, week.normalize, c.env_manager
    )
    weekday_obj = c.pyapi.long_from_longlong(week.weekday)
    week_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pd.tseries.offsets.Week))
    cond = c.builder.icmp_signed("!=", lir.Constant(lir.IntType(64), -1), week.weekday)
    with c.builder.if_else(cond) as (weekday_defined, weekday_undefined):
        with weekday_defined:
            res_if = c.pyapi.call_function_objargs(
                week_obj, (n_obj, normalize_obj, weekday_obj)
            )
            weekday_defined_bb = c.builder.block

        with weekday_undefined:
            res_else = c.pyapi.call_function_objargs(week_obj, (n_obj, normalize_obj))
            weekday_undefined_bb = c.builder.block

    res = c.builder.phi(res_if.type)
    res.add_incoming(res_if, weekday_defined_bb)
    res.add_incoming(res_else, weekday_undefined_bb)
    c.pyapi.decref(weekday_obj)
    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)
    c.pyapi.decref(week_obj)
    return res


@unbox(WeekType)
def unbox_week(typ, val, c):
    n_obj = c.pyapi.object_getattr_string(val, "n")
    normalize_obj = c.pyapi.object_getattr_string(val, "normalize")
    weekday_obj = c.pyapi.object_getattr_string(val, "weekday")

    n = c.pyapi.long_as_longlong(n_obj)
    normalize = c.pyapi.to_native_value(types.bool_, normalize_obj).value

    none_obj = c.pyapi.make_none()
    is_none = c.builder.icmp_unsigned("==", weekday_obj, none_obj)

    with c.builder.if_else(is_none) as (weekday_undefined, weekday_defined):

        with weekday_defined:
            res_if = c.pyapi.long_as_longlong(weekday_obj)
            weekday_defined_bb = c.builder.block

        with weekday_undefined:
            res_else = lir.Constant(lir.IntType(64), -1)
            weekday_undefined_bb = c.builder.block

    res = c.builder.phi(res_if.type)
    res.add_incoming(res_if, weekday_defined_bb)
    res.add_incoming(res_else, weekday_undefined_bb)

    week = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    week.n = n
    week.normalize = normalize
    week.weekday = res

    c.pyapi.decref(n_obj)
    c.pyapi.decref(normalize_obj)
    c.pyapi.decref(weekday_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    return NativeValue(week._getvalue(), is_error=is_error)


def overload_add_operator_week_offset_type(lhs, rhs):
    """Implement all of the relevant scalar types additions.
    These will be reused to implement arrays.
    """
    # rhs is a datetime, date or a timestamp
    if lhs == week_type and rhs == pd_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            time_delta = calculate_week_date(lhs.n, lhs.weekday, rhs.weekday())

            if lhs.normalize:
                new_time = pd.Timestamp(year=rhs.year, month=rhs.month, day=rhs.day)
            else:
                new_time = rhs

            return new_time + time_delta

        return impl

    if lhs == week_type and rhs == datetime_datetime_type:

        def impl(lhs, rhs):  # pragma: no cover
            time_delta = calculate_week_date(lhs.n, lhs.weekday, rhs.weekday())

            if lhs.normalize:
                new_time = pd.Timestamp(year=rhs.year, month=rhs.month, day=rhs.day)
            else:
                new_time = pd.Timestamp(
                    year=rhs.year,
                    month=rhs.month,
                    day=rhs.day,
                    hour=rhs.hour,
                    minute=rhs.minute,
                    second=rhs.second,
                    microsecond=rhs.microsecond,
                )

            return new_time + time_delta

        return impl

    if lhs == week_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            time_delta = calculate_week_date(lhs.n, lhs.weekday, rhs.weekday())
            return rhs + time_delta

        return impl

    # rhs is the offset
    if (
        lhs in [datetime_datetime_type, pd_timestamp_type, datetime_date_type]
        and rhs == week_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl

    # Raise Bodo error if not supported
    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


@register_jitable
def calculate_week_date(n, weekday, other_weekday):  # pragma: no cover
    """ Calculate the date n weeks from the input. """

    # if weekday = None (int representation is -1) return the offset
    if weekday == -1:
        return pd.Timedelta(weeks=n)

    # adjust the offset by a week if the weekdays are different
    if weekday != other_weekday:
        offset = (weekday - other_weekday) % 7
        if n > 0:
            n = n - 1

    return pd.Timedelta(weeks=n, days=offset)
