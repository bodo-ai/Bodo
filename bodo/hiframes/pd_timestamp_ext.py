# Copyright (C) 2019 Bodo Inc. All rights reserved.

import calendar
import datetime
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    ConcreteTemplate,
    infer_global,
    signature,
)
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

import bodo.libs.str_ext
import bodo.utils.utils
from bodo.hiframes.datetime_date_ext import (
    DatetimeDateType,
    _ord2ymd,
    _ymd2ord,
    get_isocalendar,
)
from bodo.hiframes.datetime_timedelta_ext import (
    PDTimeDeltaType,
    _no_input,
    datetime_timedelta_type,
    pd_timedelta_type,
)
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.libs import hdatetime_ext
from bodo.libs.str_arr_ext import string_array_type
from bodo.utils.typing import (
    BodoError,
    get_overload_const_str,
    is_iterable_type,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)

ll.add_symbol("extract_year_days", hdatetime_ext.extract_year_days)
ll.add_symbol("get_month_day", hdatetime_ext.get_month_day)

ll.add_symbol(
    "npy_datetimestruct_to_datetime", hdatetime_ext.npy_datetimestruct_to_datetime
)
npy_datetimestruct_to_datetime = types.ExternalFunction(
    "npy_datetimestruct_to_datetime",
    types.int64(
        types.int64,
        types.int32,
        types.int32,
        types.int32,
        types.int32,
        types.int32,
        types.int32,
    ),
)


date_fields = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "quarter",
    "dayofyear",
    "dayofweek",
    "daysinmonth",
    "days_in_month",
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "week",
    "weekofyear",
    "weekday",
]
date_methods = ["normalize"]

# Timedelta fields separated by return type
timedelta_fields = ["days", "seconds", "microseconds", "nanoseconds"]
timedelta_methods = ["total_seconds", "to_pytimedelta"]
iNaT = pd._libs.tslibs.iNaT


class PandasTimestampType(types.Type):
    def __init__(self):
        super(PandasTimestampType, self).__init__(name="PandasTimestampType()")


pandas_timestamp_type = PandasTimestampType()


@typeof_impl.register(pd.Timestamp)
def typeof_pd_timestamp(val, c):
    return pandas_timestamp_type


ts_field_typ = types.int64


@register_model(PandasTimestampType)
class PandasTimestampModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("year", ts_field_typ),
            ("month", ts_field_typ),
            ("day", ts_field_typ),
            ("hour", ts_field_typ),
            ("minute", ts_field_typ),
            ("second", ts_field_typ),
            ("microsecond", ts_field_typ),
            ("nanosecond", ts_field_typ),
            ("value", ts_field_typ),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(PandasTimestampType, "year", "year")
make_attribute_wrapper(PandasTimestampType, "month", "month")
make_attribute_wrapper(PandasTimestampType, "day", "day")
make_attribute_wrapper(PandasTimestampType, "hour", "hour")
make_attribute_wrapper(PandasTimestampType, "minute", "minute")
make_attribute_wrapper(PandasTimestampType, "second", "second")
make_attribute_wrapper(PandasTimestampType, "microsecond", "microsecond")
make_attribute_wrapper(PandasTimestampType, "nanosecond", "nanosecond")
make_attribute_wrapper(PandasTimestampType, "value", "value")


@unbox(PandasTimestampType)
def unbox_pandas_timestamp(typ, val, c):
    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")
    hour_obj = c.pyapi.object_getattr_string(val, "hour")
    minute_obj = c.pyapi.object_getattr_string(val, "minute")
    second_obj = c.pyapi.object_getattr_string(val, "second")
    microsecond_obj = c.pyapi.object_getattr_string(val, "microsecond")
    nanosecond_obj = c.pyapi.object_getattr_string(val, "nanosecond")
    value_obj = c.pyapi.object_getattr_string(val, "value")

    pd_timestamp = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    pd_timestamp.year = c.pyapi.long_as_longlong(year_obj)
    pd_timestamp.month = c.pyapi.long_as_longlong(month_obj)
    pd_timestamp.day = c.pyapi.long_as_longlong(day_obj)
    pd_timestamp.hour = c.pyapi.long_as_longlong(hour_obj)
    pd_timestamp.minute = c.pyapi.long_as_longlong(minute_obj)
    pd_timestamp.second = c.pyapi.long_as_longlong(second_obj)
    pd_timestamp.microsecond = c.pyapi.long_as_longlong(microsecond_obj)
    pd_timestamp.nanosecond = c.pyapi.long_as_longlong(nanosecond_obj)
    pd_timestamp.value = c.pyapi.long_as_longlong(value_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)
    c.pyapi.decref(nanosecond_obj)
    c.pyapi.decref(value_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(pd_timestamp._getvalue(), is_error=is_error)


@box(PandasTimestampType)
def box_pandas_timestamp(typ, val, c):
    pdts = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    year_obj = c.pyapi.long_from_longlong(pdts.year)
    month_obj = c.pyapi.long_from_longlong(pdts.month)
    day_obj = c.pyapi.long_from_longlong(pdts.day)
    hour_obj = c.pyapi.long_from_longlong(pdts.hour)
    minute_obj = c.pyapi.long_from_longlong(pdts.minute)
    second_obj = c.pyapi.long_from_longlong(pdts.second)
    us_obj = c.pyapi.long_from_longlong(pdts.microsecond)
    ns_obj = c.pyapi.long_from_longlong(pdts.nanosecond)

    pdts_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pd.Timestamp))
    res = c.pyapi.call_function_objargs(
        pdts_obj,
        (
            year_obj,
            month_obj,
            day_obj,
            hour_obj,
            minute_obj,
            second_obj,
            us_obj,
            ns_obj,
        ),
    )
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(us_obj)
    c.pyapi.decref(ns_obj)
    return res


@intrinsic
def init_timestamp(
    typingctx,
    year,
    month,
    day,
    hour,
    minute,
    second,
    microsecond,
    nanosecond,
    value=None,
):
    """Create a PandasTimestampType with provided data values."""

    def codegen(context, builder, sig, args):
        year, month, day, hour, minute, second, us, ns, value = args
        ts = cgutils.create_struct_proxy(pandas_timestamp_type)(context, builder)
        ts.year = year
        ts.month = month
        ts.day = day
        ts.hour = hour
        ts.minute = minute
        ts.second = second
        ts.microsecond = us
        ts.nanosecond = ns
        ts.value = value
        return ts._getvalue()

    return (
        pandas_timestamp_type(
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
        ),
        codegen,
    )


@numba.generated_jit
def zero_if_none(value):
    """return zero if value is None. Otherwise, return value"""
    if value == types.none:
        return lambda value: 0
    return lambda value: value


@lower_constant(PandasTimestampType)
def constant_timestamp(context, builder, ty, pyval):
    # Extracting constants. Inspired from @lower_constant(types.Complex)
    # in numba/numba/targets/numbers.py
    year = context.get_constant(types.int64, pyval.year)
    month = context.get_constant(types.int64, pyval.month)
    day = context.get_constant(types.int64, pyval.day)
    hour = context.get_constant(types.int64, pyval.hour)
    minute = context.get_constant(types.int64, pyval.minute)
    second = context.get_constant(types.int64, pyval.second)
    microsecond = context.get_constant(types.int64, pyval.microsecond)
    nanosecond = context.get_constant(types.int64, pyval.nanosecond)
    value = context.get_constant(types.int64, pyval.value)
    pd_timestamp = cgutils.create_struct_proxy(ty)(context, builder)
    pd_timestamp.year = year
    pd_timestamp.month = month
    pd_timestamp.day = day
    pd_timestamp.hour = hour
    pd_timestamp.minute = minute
    pd_timestamp.second = second
    pd_timestamp.microsecond = microsecond
    pd_timestamp.nanosecond = nanosecond
    pd_timestamp.value = value
    return pd_timestamp._getvalue()


# -------------------------------------------------------------------------------


@overload(pd.Timestamp, no_unliteral=True)
def overload_pd_timestamp(
    ts_input=_no_input,
    freq=None,
    tz=None,
    unit=None,
    year=None,
    month=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    microsecond=None,
    nanosecond=None,
    tzinfo=None,
):
    # The code for creating Timestamp from year/month/... is complex in Pandas but it
    # eventually just sets year/month/... values, and calculates dt64 "value" attribute
    # Timestamp.__new__()
    # https://github.com/pandas-dev/pandas/blob/8806ed7120fed863b3cd7d3d5f377ec4c81739d0/pandas/_libs/tslibs/timestamps.pyx#L399
    # convert_to_tsobject()
    # https://github.com/pandas-dev/pandas/blob/8806ed7120fed863b3cd7d3d5f377ec4c81739d0/pandas/_libs/tslibs/conversion.pyx#L267
    # convert_datetime_to_tsobject()
    # pydatetime_to_dt64()
    # https://github.com/pandas-dev/pandas/blob/8806ed7120fed863b3cd7d3d5f377ec4c81739d0/pandas/_libs/tslibs/np_datetime.pyx#L145
    # create_timestamp_from_ts()

    # User passed keyword arguments
    if ts_input == _no_input or getattr(ts_input, "value", None) == _no_input:

        def impl_kw(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover
            value = npy_datetimestruct_to_datetime(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
            )
            value += zero_if_none(nanosecond)
            return init_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                value,
            )

        return impl_kw

    # User passed positional arguments:
    # Timestamp(year, month, day[, hour[, minute[, second[,
    # microsecond[, nanosecond[, tzinfo]]]]]])
    if isinstance(types.unliteral(freq), types.Integer):

        def impl_pos(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover
            value = npy_datetimestruct_to_datetime(
                ts_input,
                freq,
                tz,
                zero_if_none(unit),
                zero_if_none(year),
                zero_if_none(month),
                zero_if_none(day),
            )
            value += zero_if_none(hour)
            return init_timestamp(
                ts_input,
                freq,
                tz,
                zero_if_none(unit),
                zero_if_none(year),
                zero_if_none(month),
                zero_if_none(day),
                zero_if_none(hour),
                value,
            )

        return impl_pos

    # Pandas converts to dt64 and then back to a timestamp
    # https://github.com/pandas-dev/pandas/blob/4aa0783d65dc20dc450ef3f58defda14ebab5f6f/pandas/_libs/tslibs/conversion.pyx#L406
    if isinstance(ts_input, types.Number):
        if is_overload_none(unit):
            unit = "ns"
        if not is_overload_constant_str(unit):
            raise BodoError("pd.Timedelta(): unit argument must be a constant str")
        unit = pd._libs.tslibs.timedeltas.parse_timedelta_unit(
            get_overload_const_str(unit)
        )
        nanoseconds, precision = pd._libs.tslibs.conversion.precision_from_unit(unit)
        if isinstance(ts_input, types.Integer):

            def impl_int(
                ts_input=_no_input,
                freq=None,
                tz=None,
                unit=None,
                year=None,
                month=None,
                day=None,
                hour=None,
                minute=None,
                second=None,
                microsecond=None,
                nanosecond=None,
                tzinfo=None,
            ):  # pragma: no cover
                # Create nanosecond value for dt64
                value = ts_input * nanoseconds
                return convert_datetime64_to_timestamp(integer_to_dt64(value))

            return impl_int

        def impl_float(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover
            # Create nanosecond value for td64
            base = np.int64(ts_input)
            frac = ts_input - base
            if precision:
                frac = np.round(frac, precision)
            value = base * nanoseconds + np.int64(frac * nanoseconds)
            return convert_datetime64_to_timestamp(integer_to_dt64(value))

        return impl_float

    # parse string input
    if ts_input == bodo.string_type or is_overload_constant_str(ts_input):
        # just call Pandas in this case since the string parsing code is complex and
        # handles several possible cases
        types.pandas_timestamp_type = pandas_timestamp_type

        def impl_str(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover
            with numba.objmode(res="pandas_timestamp_type"):
                res = pd.Timestamp(ts_input)
            return res

        return impl_str

    # for pd.Timestamp(), just return input
    if ts_input == pandas_timestamp_type:
        return (
            lambda ts_input=_no_input, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None: ts_input
        )

    if ts_input == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type:

        def impl_datetime(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover

            year = ts_input.year
            month = ts_input.month
            day = ts_input.day
            hour = ts_input.hour
            minute = ts_input.minute
            second = ts_input.second
            microsecond = ts_input.microsecond

            value = npy_datetimestruct_to_datetime(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
            )
            value += zero_if_none(nanosecond)
            return init_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                value,
            )

        return impl_datetime

    if ts_input == bodo.hiframes.datetime_date_ext.datetime_date_type:

        def impl_date(
            ts_input=_no_input,
            freq=None,
            tz=None,
            unit=None,
            year=None,
            month=None,
            day=None,
            hour=None,
            minute=None,
            second=None,
            microsecond=None,
            nanosecond=None,
            tzinfo=None,
        ):  # pragma: no cover

            year = ts_input.year
            month = ts_input.month
            day = ts_input.day

            value = npy_datetimestruct_to_datetime(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
            )
            value += zero_if_none(nanosecond)
            return init_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                value,
            )

        return impl_date


@overload_attribute(PandasTimestampType, "dayofyear")
def overload_pd_dayofyear(ptt):
    def pd_dayofyear(ptt):  # pragma: no cover
        return get_day_of_year(ptt.year, ptt.month, ptt.day)

    return pd_dayofyear


@overload_method(PandasTimestampType, "weekday")
@overload_attribute(PandasTimestampType, "dayofweek")
@overload_method(PandasTimestampType, "weekday")
def overload_pd_dayofweek(ptt):
    def pd_dayofweek(ptt):  # pragma: no cover
        return get_day_of_week(ptt.year, ptt.month, ptt.day)

    return pd_dayofweek


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L138
@overload_attribute(PandasTimestampType, "week")
@overload_attribute(PandasTimestampType, "weekofyear")
def overload_week_number(ptt):
    def pd_week_number(ptt):
        # Offset the day of the year by the (day of the week of jan 1st) and add 1 week because 1 indexed
        # Year starting on a Monday should be 7
        return (
            get_day_of_year(ptt.year, ptt.month, ptt.day)
            + get_day_of_week(ptt.year, 1, 1)
            + 6
        ) // 7

    return pd_week_number


@overload_method(PandasTimestampType, "__hash__", no_unliteral=True)
def dt64_hash(val):
    return lambda val: hash(val.value)


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L59
@overload_attribute(PandasTimestampType, "days_in_month")
@overload_attribute(PandasTimestampType, "daysinmonth")
def overload_pd_daysinmonth(ptt):
    def pd_daysinmonth(ptt):  # pragma: no cover
        return get_days_in_month(ptt.year, ptt.month)

    return pd_daysinmonth


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L132
@overload_attribute(PandasTimestampType, "is_leap_year")
def overload_pd_is_leap_year(ptt):
    def pd_is_leap_year(ptt):  # pragma: no cover
        return is_leap_year(ptt.year)

    return pd_is_leap_year


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L425
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_month_start")
def overload_pd_is_month_start(ptt):
    def pd_is_month_start(ptt):  # pragma: no cover
        return ptt.day == 1

    return pd_is_month_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L436
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_month_end")
def overload_pd_is_month_end(ptt):
    def pd_is_month_end(ptt):  # pragma: no cover
        return ptt.day == get_days_in_month(ptt.year, ptt.month)

    return pd_is_month_end


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L445
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_quarter_start")
def overload_pd_is_quarter_start(ptt):
    def pd_is_quarter_start(ptt):  # pragma: no cover
        return ptt.day == 1 and (ptt.month % 3) == 1

    return pd_is_quarter_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L456
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_quarter_end")
def overload_pd_is_quarter_end(ptt):
    def pd_is_quarter_end(ptt):  # pragma: no cover
        return (ptt.month % 3) == 0 and ptt.day == get_days_in_month(
            ptt.year, ptt.month
        )

    return pd_is_quarter_end


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L466
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_year_start")
def overload_pd_is_year_start(ptt):
    def pd_is_year_start(ptt):  # pragma: no cover
        return ptt.day == 1 and ptt.month == 1

    return pd_is_year_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L476
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_year_end")
def overload_pd_is_year_end(ptt):
    def pd_is_year_end(ptt):  # pragma: no cover
        return ptt.day == 31 and ptt.month == 12

    return pd_is_year_end


@overload_attribute(PandasTimestampType, "quarter")
def overload_quarter(ptt):
    # copied implementation from https://github.com/pandas-dev/pandas/blob/4859be9cd145e3da0a7f596c3e27636a58920c1c/pandas/_libs/tslibs/timestamps.pyx#L547
    def quarter(ptt):  # pragma: no cover
        return ((ptt.month - 1) // 3) + 1

    return quarter


@overload_method(PandasTimestampType, "date", no_unliteral=True)
def overload_pd_timestamp_date(ptt):
    def pd_timestamp_date_impl(ptt):  # pragma: no cover
        return datetime.date(ptt.year, ptt.month, ptt.day)

    return pd_timestamp_date_impl


@overload_method(PandasTimestampType, "isocalendar", no_unliteral=True)
def overload_pd_timestamp_isocalendar(ptt):
    def impl(ptt):  # pragma: no cover
        year, week, day_of_week = get_isocalendar(ptt.year, ptt.month, ptt.day)
        return (year, week, day_of_week)

    return impl


@overload_method(PandasTimestampType, "isoformat", no_unliteral=True)
def overload_pd_timestamp_isoformat(ts, sep=None):
    if is_overload_none(sep):

        def timestamp_isoformat_impl(ts, sep=None):  # pragma: no cover
            assert ts.nanosecond == 0  # TODO: handle nanosecond (timestamps.pyx)
            _time = str_2d(ts.hour) + ":" + str_2d(ts.minute) + ":" + str_2d(ts.second)
            res = (
                str(ts.year)
                + "-"
                + str_2d(ts.month)
                + "-"
                + str_2d(ts.day)
                + "T"
                + _time
            )
            return res

        return timestamp_isoformat_impl

    else:

        def timestamp_isoformat_impl(ts, sep=None):  # pragma: no cover
            assert ts.nanosecond == 0  # TODO: handle nanosecond (timestamps.pyx)
            _time = str_2d(ts.hour) + ":" + str_2d(ts.minute) + ":" + str_2d(ts.second)
            res = (
                str(ts.year)
                + "-"
                + str_2d(ts.month)
                + "-"
                + str_2d(ts.day)
                + sep
                + _time
            )
            return res

    return timestamp_isoformat_impl


@overload_method(PandasTimestampType, "normalize", no_unliteral=True)
def overload_pd_timestamp_normalize(ptt):
    def impl(ptt):  # pragma: no cover
        return pd.Timestamp(year=ptt.year, month=ptt.month, day=ptt.day)

    return impl


# TODO: support general string formatting
@numba.njit
def str_2d(a):  # pragma: no cover
    res = str(a)
    if len(res) == 1:
        return "0" + res
    return res


@overload(str, no_unliteral=True)
def ts_str_overload(a):
    if a == pandas_timestamp_type:
        return lambda a: a.isoformat(" ")


@intrinsic
def extract_year_days(typingctx, dt64_t=None):
    """Extracts year and days from dt64 value.
    Returns a 3-tuple of (leftover_dt64_values, year, days)
    """
    assert dt64_t in (types.int64, types.NPDatetime("ns"))

    def codegen(context, builder, sig, args):
        dt = cgutils.alloca_once(builder, lir.IntType(64))
        builder.store(args[0], dt)
        year = cgutils.alloca_once(builder, lir.IntType(64))
        days = cgutils.alloca_once(builder, lir.IntType(64))
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="extract_year_days")
        builder.call(fn_tp, [dt, year, days])
        return cgutils.pack_array(
            builder, [builder.load(dt), builder.load(year), builder.load(days)]
        )

    return types.Tuple([types.int64, types.int64, types.int64])(dt64_t), codegen


@intrinsic
def get_month_day(typingctx, year_t, days_t=None):
    """Converts number of days within a year to month and day, returned as a 2-tuple."""
    assert year_t == types.int64
    assert days_t == types.int64

    def codegen(context, builder, sig, args):
        month = cgutils.alloca_once(builder, lir.IntType(64))
        day = cgutils.alloca_once(builder, lir.IntType(64))
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="get_month_day")
        builder.call(fn_tp, [args[0], args[1], month, day])
        return cgutils.pack_array(builder, [builder.load(month), builder.load(day)])

    return types.Tuple([types.int64, types.int64])(types.int64, types.int64), codegen


@register_jitable
def get_day_of_year(year, month, day):  # pragma: no cover
    """gets day offset within year"""
    # mostly copied from https://github.com/pandas-dev/pandas/blob/6b2d0260c818e62052eaf535767f3a8c4b446c69/pandas/_libs/tslibs/ccalendar.pyx#L215
    month_offset = [
        0,
        31,
        59,
        90,
        120,
        151,
        181,
        212,
        243,
        273,
        304,
        334,
        365,
        0,
        31,
        60,
        91,
        121,
        152,
        182,
        213,
        244,
        274,
        305,
        335,
        366,
    ]

    is_leap = is_leap_year(year)
    mo_off = month_offset[is_leap * 13 + month - 1]

    day_of_year = mo_off + day
    return day_of_year


@register_jitable
def get_day_of_week(y, m, d):  # pragma: no cover
    """
    gets the day of the week for the date described by the year month day tuple. Assumes that the arguments are valid.
    """
    # mostly copied from https://github.com/pandas-dev/pandas/blob/6b2d0260c818e62052eaf535767f3a8c4b446c69/pandas/_libs/tslibs/ccalendar.pyx#L83
    sakamoto_arr = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    y -= m < 3
    day = (y + y // 4 - y // 100 + y // 400 + sakamoto_arr[m - 1] + d) % 7
    # convert to python day
    return (day + 6) % 7


@register_jitable
def get_days_in_month(year, month):  # pragma: no cover
    """
    gets the number of days in month
    """
    # mostly copied from   https://github.com/pandas-dev/pandas/blob/6b2d0260c818e62052eaf535767f3a8c4b446c69/pandas/_libs/tslibs/ccalendar.pyx#L59
    is_leap_year = (year & 0x3) == 0 and ((year % 100) != 0 or (year % 400) == 0)
    days_per_month_array = [
        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
        31,
        29,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    return days_per_month_array[12 * is_leap_year + month - 1]


@register_jitable
def is_leap_year(year):  # pragma: no cover
    """returns 1 if leap year 0 otherwise"""
    # copied from https://github.com/pandas-dev/pandas/blob/6b2d0260c818e62052eaf535767f3a8c4b446c69/pandas/_libs/tslibs/ccalendar.pyx#L161
    return (year & 0x3) == 0 and ((year % 100) != 0 or (year % 400) == 0)


@numba.njit(no_cpython_wrapper=True)
def convert_datetime64_to_timestamp(dt64):  # pragma: no cover
    """Converts dt64 value to pd.Timestamp"""
    dt, year, days = extract_year_days(dt64)
    month, day = get_month_day(year, days)

    return pd.Timestamp(
        year,
        month,
        day,
        dt // (60 * 60 * 1000000000),  # hour
        (dt // (60 * 1000000000)) % 60,  # minute
        (dt // 1000000000) % 60,  # second
        (dt // 1000) % 1000000,  # microsecond
        dt % 1000,
    )  # nanosecond


@numba.njit(no_cpython_wrapper=True)
def convert_numpy_timedelta64_to_datetime_timedelta(dt64):  # pragma: no cover
    """Convertes numpy.timedelta64 to datetime.timedelta"""
    n_int64 = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(dt64)
    n_day = n_int64 // (86400 * 1000000000)
    res1 = n_int64 - n_day * 86400 * 1000000000
    n_sec = res1 // 1000000000
    res2 = res1 - n_sec * 1000000000
    n_microsec = res2 // 1000
    return datetime.timedelta(n_day, n_sec, n_microsec)


@numba.njit(no_cpython_wrapper=True)
def convert_numpy_timedelta64_to_pd_timedelta(dt64):  # pragma: no cover
    """Convertes numpy.timedelta64 to pd.Timedelta"""
    n_int64 = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(dt64)
    return pd.Timedelta(n_int64)


@intrinsic
def integer_to_timedelta64(typingctx, val=None):
    """Cast an int value to timedelta64"""

    def codegen(context, builder, sig, args):
        return args[0]

    return types.NPTimedelta("ns")(val), codegen


@intrinsic
def integer_to_dt64(typingctx, val=None):
    """Cast an int value to datetime64"""

    def codegen(context, builder, sig, args):
        return args[0]

    return types.NPDatetime("ns")(val), codegen


@intrinsic
def dt64_to_integer(typingctx, val=None):
    """Cast a datetime64 value to integer"""

    def codegen(context, builder, sig, args):
        return args[0]

    return types.int64(val), codegen


@lower_cast(types.NPDatetime("ns"), types.int64)
def cast_dt64_to_integer(context, builder, fromty, toty, val):
    # dt64 is stored as int64 so just return value
    return val


# TODO: fix in Numba
@overload_method(types.NPDatetime, "__hash__", no_unliteral=True)
def dt64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


# TODO: fix in Numba
@overload_method(types.NPTimedelta, "__hash__", no_unliteral=True)
def td64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


@intrinsic
def timedelta64_to_integer(typingctx, val=None):
    """Cast a timedelta64 value to integer"""

    def codegen(context, builder, sig, args):
        return args[0]

    return types.int64(val), codegen


@numba.njit
def parse_datetime_str(val):  # pragma: no cover
    """Parse datetime string value to dt64
    Just calling Pandas since the Pandas code is complex
    """
    with numba.objmode(res="int64"):
        res = pd.Timestamp(val).value
    return integer_to_dt64(res)


@numba.njit
def datetime_timedelta_to_timedelta64(val):  # pragma: no cover
    """convert datetime.timedelta to np.timedelta64"""
    with numba.objmode(res='NPTimedelta("ns")'):
        res = pd.to_timedelta(val)
        res = res.to_timedelta64()
    return res


@numba.njit
def datetime_datetime_to_dt64(val):  # pragma: no cover
    """convert datetime.datetime to np.datetime64"""
    with numba.objmode(res='NPDatetime("ns")'):
        res = np.datetime64(val).astype("datetime64[ns]")

    return res


@register_jitable
def datetime_date_arr_to_dt64_arr(arr):  # pragma: no cover
    """convert array of datetime.date to np.datetime64"""
    with numba.objmode(res='NPDatetime("ns")[:]'):
        res = np.array(arr, dtype="datetime64[ns]")
    return res


types.pandas_timestamp_type = pandas_timestamp_type


@register_jitable
def to_datetime_scalar(
    a,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):  # pragma: no cover
    """call pd.to_datetime() with scalar value 'a'
    separate call to avoid adding extra basic blocks to user function for simplicity
    """
    with numba.objmode(t="pandas_timestamp_type"):
        t = pd.to_datetime(
            a,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
    return t


@numba.njit
def pandas_string_array_to_datetime(
    arr,
    errors,
    dayfirst,
    yearfirst,
    utc,
    format,
    exact,
    unit,
    infer_datetime_format,
    origin,
    cache,
):  # pragma: no cover
    with numba.objmode(result="datetime_index"):
        # pd.to_datetime(string_array) returns DatetimeIndex
        result = pd.to_datetime(
            arr,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
    return result


@overload(pd.to_datetime, inline="always", no_unliteral=True)
def overload_to_datetime(
    arg_a,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):
    """implementation for pd.to_datetime"""
    # TODO: change 'arg_a' to 'arg' when inliner can handle it

    # This covers string as a literal or not
    # and integer as a literal or not
    if (
        arg_a == bodo.string_type
        or is_overload_constant_str(arg_a)
        or is_overload_constant_int(arg_a)
        or isinstance(arg_a, types.Integer)
    ):

        def pd_to_datetime_impl(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            return to_datetime_scalar(
                arg_a,
                errors=errors,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                utc=utc,
                format=format,
                exact=exact,
                unit=unit,
                infer_datetime_format=infer_datetime_format,
                origin=origin,
                cache=cache,
            )

        return pd_to_datetime_impl

    # Series input, call on values and wrap to Series
    if isinstance(arg_a, bodo.hiframes.pd_series_ext.SeriesType):  # pragma: no cover

        def impl_series(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(arg_a)
            index = bodo.hiframes.pd_series_ext.get_series_index(arg_a)
            name = bodo.hiframes.pd_series_ext.get_series_name(arg_a)
            A = bodo.utils.conversion.coerce_to_ndarray(
                pd.to_datetime(
                    arr,
                    errors=errors,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    utc=utc,
                    format=format,
                    exact=exact,
                    unit=unit,
                    infer_datetime_format=infer_datetime_format,
                    origin=origin,
                    cache=cache,
                )
            )
            return bodo.hiframes.pd_series_ext.init_series(A, index, name)

        return impl_series

    # datetime.date() array
    if (
        arg_a == bodo.hiframes.datetime_date_ext.datetime_date_array_type
    ):  # pragma: no cover
        dt64_dtype = np.dtype("datetime64[ns]")
        iNaT = pd._libs.tslibs.iNaT

        def impl_date_arr(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            n = len(arg_a)
            B = np.empty(n, dt64_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                val = iNaT
                if not bodo.libs.array_kernels.isna(arg_a, i):
                    data = arg_a[i]
                    val = bodo.hiframes.pd_timestamp_ext.npy_datetimestruct_to_datetime(
                        data.year, data.month, data.day, 0, 0, 0, 0
                    )
                B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val)
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_date_arr

    # return DatetimeIndex if input is array(dt64)
    if arg_a == types.Array(types.NPDatetime("ns"), 1, "C"):
        return lambda arg_a, errors="raise", dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin="unix", cache=True: bodo.hiframes.pd_index_ext.init_datetime_index(
            arg_a, None
        )  # pragma: no cover

    # string_array_type as input
    if arg_a == string_array_type:
        dt64_dtype = np.dtype("datetime64[ns]")
        iNaT = pd._libs.tslibs.iNaT

        def impl_string_array(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            # need to call a separately compiled function because inlining
            # doesn't work with objmode currently
            return pandas_string_array_to_datetime(
                arg_a,
                errors,
                dayfirst,
                yearfirst,
                utc,
                format,
                exact,
                unit,
                infer_datetime_format,
                origin,
                cache,
            )

        return impl_string_array

    # datetime.date() array
    if isinstance(arg_a, types.Array) and isinstance(arg_a.dtype, types.Integer):
        dt64_dtype = np.dtype("datetime64[ns]")

        def impl_date_arr(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            n = len(arg_a)
            B = np.empty(n, dt64_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                data = arg_a[i]
                val = to_datetime_scalar(
                    data,
                    errors=errors,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    utc=utc,
                    format=format,
                    exact=exact,
                    unit=unit,
                    infer_datetime_format=infer_datetime_format,
                    origin=origin,
                    cache=cache,
                )
                B[i] = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_date_arr

    # Categorical array with string values
    if (
        isinstance(arg_a, CategoricalArray)
        and arg_a.dtype.elem_type == bodo.string_type
    ):
        dt64_dtype = np.dtype("datetime64[ns]")

        def impl_cat_arr(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            infer_datetime_format=False,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            n = len(arg_a)
            B = np.empty(n, dt64_dtype)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arg_a)
            # get datetime64 value for each category code
            dt64_vals = pandas_string_array_to_datetime(
                arg_a.dtype.categories.values,
                errors,
                dayfirst,
                yearfirst,
                utc,
                format,
                exact,
                unit,
                infer_datetime_format,
                origin,
                cache,
            ).values
            for i in numba.parfors.parfor.internal_prange(n):
                c = codes[i]
                if c == -1:
                    bodo.libs.array_kernels.setna(B, i)
                    continue
                B[i] = dt64_vals[c]
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_cat_arr

    # TODO: input Type of a dataframe
    raise_bodo_error(f"pd.to_datetime(): cannot convert date type {arg_a}")


@overload(pd.to_timedelta, inline="always", no_unliteral=True)
def overload_to_timedelta(arg_a, unit="ns", errors="raise"):
    # changed 'arg' to 'arg_a' since inliner uses vname.startswith("arg.") to find
    # argument variables which causes conflict
    # TODO: fix call inliner to hande 'arg' name properly

    if not is_overload_constant_str(unit):  # pragma: no cover
        raise BodoError("pd.to_timedelta(): unit should be a constant string")

    # internal Pandas API that normalizes variations of unit. e.g. 'seconds' -> 's'
    unit = pd._libs.tslibs.timedeltas.parse_timedelta_unit(get_overload_const_str(unit))

    # Series input, call on values and wrap to Series
    if isinstance(arg_a, bodo.hiframes.pd_series_ext.SeriesType):  # pragma: no cover

        def impl_series(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(arg_a)
            index = bodo.hiframes.pd_series_ext.get_series_index(arg_a)
            name = bodo.hiframes.pd_series_ext.get_series_name(arg_a)
            # calls to_timedelta() recursively to pick up the array implementation
            # such as the one for float arrays below. Inlined recursively in series pass
            A = bodo.utils.conversion.coerce_to_ndarray(
                pd.to_timedelta(arr, unit, errors)
            )
            return bodo.hiframes.pd_series_ext.init_series(A, index, name)

        return impl_series

    # Timedelta, Datetime, or String input, just create a Timedelta value
    if is_overload_constant_str(arg_a) or arg_a in (
        pd_timedelta_type,
        datetime_timedelta_type,
        bodo.string_type,
    ):

        def impl_string(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            return pd.Timedelta(arg_a)

        return impl_string

    # Float scalar
    if isinstance(arg_a, types.Float):
        m, p = pd._libs.tslibs.conversion.precision_from_unit(unit)

        def impl_float_scalar(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            val = float_to_timedelta_val(arg_a, p, m)
            return pd.Timedelta(val)

        return impl_float_scalar

    # Integer scalar
    if isinstance(arg_a, types.Integer):
        m, _ = pd._libs.tslibs.conversion.precision_from_unit(unit)

        def impl_integer_scalar(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            return pd.Timedelta(arg_a * m)

        return impl_integer_scalar

    # TODO: Add tuple support. These require separate kernels becasue we cannot check isna for tuples
    if is_iterable_type(arg_a) and not isinstance(arg_a, types.BaseTuple):
        m, p = pd._libs.tslibs.conversion.precision_from_unit(unit)
        td64_dtype = np.dtype("timedelta64[ns]")
        if isinstance(arg_a.dtype, types.Float):
            # float input
            # from Pandas implementation:
            # https://github.com/pandas-dev/pandas/blob/2e0e013703390377faad57ee97f2cfaf98ba039e/pandas/core/arrays/timedeltas.py#L956
            def impl_float(arg_a, unit="ns", errors="raise"):  # pragma: no cover
                n = len(arg_a)
                B = np.empty(n, td64_dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    val = iNaT
                    if not bodo.libs.array_kernels.isna(arg_a, i):
                        val = float_to_timedelta_val(arg_a[i], p, m)
                    B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(B, None)

            return impl_float

        if isinstance(arg_a.dtype, types.Integer):

            def impl_int(arg_a, unit="ns", errors="raise"):  # pragma: no cover
                n = len(arg_a)
                B = np.empty(n, td64_dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    val = iNaT
                    if not bodo.libs.array_kernels.isna(arg_a, i):
                        val = arg_a[i] * m
                    B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(B, None)

            return impl_int

        if arg_a.dtype == bodo.timedelta64ns:

            def impl_td64(arg_a, unit="ns", errors="raise"):  # pragma: no cover
                arr = bodo.utils.conversion.coerce_to_ndarray(arg_a)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(arr, None)

            return impl_td64

        # Either a string array or numpy unichr array
        if arg_a.dtype == bodo.string_type or isinstance(
            arg_a.dtype, types.UnicodeCharSeq
        ):
            # Call a kernel that enters objmode once for all conversion to avoid overhead
            def impl_str(arg_a, unit="ns", errors="raise"):  # pragma: no cover
                return pandas_string_array_to_timedelta(arg_a, unit, errors)

            return impl_str

        if arg_a.dtype == datetime_timedelta_type:

            def impl_datetime_timedelta(
                arg_a, unit="ns", errors="raise"
            ):  # pragma: no cover
                n = len(arg_a)
                B = np.empty(n, td64_dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    val = iNaT
                    if not bodo.libs.array_kernels.isna(arg_a, i):
                        datetime_val = arg_a[i]
                        val = (
                            datetime_val.microseconds
                            + 1000
                            * 1000
                            * (
                                datetime_val.seconds
                                + (24 * 60 * 60 * datetime_val.days)
                            )
                        ) * 1000
                    B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(B, None)

            return impl_datetime_timedelta


@register_jitable
def float_to_timedelta_val(data, precision, multiplier):  # pragma: no cover
    """Helper function to convert floating point data to an integer
    representing a timedelta val with the given precision and multplier.
    The multiplier increase the integer component of the float the rescale
    the original unit to ns, while precision rounds the decimal component.
    """
    base = np.int64(data)
    frac = data - base
    if precision:
        frac = np.round(frac, precision)
    return base * multiplier + np.int64(frac * multiplier)


@numba.njit
def pandas_string_array_to_timedelta(
    arg_a, unit="ns", errors="raise"
):  # pragma: no cover
    with numba.objmode(result="timedelta_index"):
        # pd.to_timedelta(string_array) returns TimedeltaIndex
        # Cannot pass in a unit if args are strings
        result = pd.to_timedelta(arg_a, errors=errors)
    return result


# comparison of Timestamp and datetime.date
def create_timestamp_cmp_op_overload(op):
    """
    create overloads for comparison operators with datetime.date and Timestamp
    """

    def overload_date_timestamp_cmp(A1, A2):
        # Timestamp, datetime.date
        if (
            A1 == pandas_timestamp_type
            and A2 == bodo.hiframes.datetime_date_ext.datetime_date_type
        ):
            return lambda A1, A2: op(
                A1.value,
                bodo.hiframes.pd_timestamp_ext.npy_datetimestruct_to_datetime(
                    A2.year, A2.month, A2.day, 0, 0, 0, 0
                ),
            )

        # datetime.date, Timestamp
        if (
            A1 == bodo.hiframes.datetime_date_ext.datetime_date_type
            and A2 == pandas_timestamp_type
        ):
            return lambda A1, A2: op(
                bodo.hiframes.pd_timestamp_ext.npy_datetimestruct_to_datetime(
                    A1.year, A1.month, A1.day, 0, 0, 0, 0
                ),
                A2.value,
            )

        # Timestamp/Timestamp
        if A1 == pandas_timestamp_type and A2 == pandas_timestamp_type:
            return lambda A1, A2: op(A1.value, A2.value)

        # Timestamp/dt64
        if A1 == pandas_timestamp_type and A2 == bodo.datetime64ns:
            return lambda A1, A2: op(
                bodo.hiframes.pd_timestamp_ext.integer_to_dt64(A1.value), A2
            )  # pragma: no cover

        # dt64/Timestamp
        if A1 == bodo.datetime64ns and A2 == pandas_timestamp_type:
            return lambda A1, A2: op(
                A1, bodo.hiframes.pd_timestamp_ext.integer_to_dt64(A2.value)
            )  # pragma: no cover

    return overload_date_timestamp_cmp


def _install_timestamp_cmp_ops():
    """install overloads for comparison operators with datetime.date and datetime64"""
    for op in (
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ):
        overload_impl = create_timestamp_cmp_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_timestamp_cmp_ops()


@overload_method(PandasTimestampType, "toordinal", no_unliteral=True)
def toordinal(date):
    """Return proleptic Gregorian ordinal for the year, month and day.
    January 1 of year 1 is day 1.  Only the year, month and day values
    contribute to the result.
    """

    def impl(date):  # pragma: no cover
        return _ymd2ord(date.year, date.month, date.day)

    return impl


# Relevant Pandas code
# https://github.com/pandas-dev/pandas/blob/009ffa8d2c019ffb757fb0a4b53cc7a9a948afdd/pandas/_libs/tslibs/timestamps.pyx#L1228
# https://github.com/pandas-dev/pandas/blob/009ffa8d2c019ffb757fb0a4b53cc7a9a948afdd/pandas/_libs/tslibs/timestamps.pyx#L1189
# https://github.com/pandas-dev/pandas/blob/009ffa8d2c019ffb757fb0a4b53cc7a9a948afdd/pandas/_libs/tslibs/timestamps.pyx#L1149
# https://github.com/pandas-dev/pandas/blob/009ffa8d2c019ffb757fb0a4b53cc7a9a948afdd/pandas/_libs/tslibs/timedeltas.pyx#L1219
def overload_freq_methods(method):
    def freq_overload(td, freq):
        freq_conditions = [
            "freq == 'D'",
            "freq == 'H'",
            "freq == 'min' or freq == 'T'",
            "freq == 'S'",
            "freq == 'ms' or freq == 'L'",
            "freq == 'U' or freq == 'us'",
            "freq == 'N'",
        ]
        unit_values = [
            24 * 60 * 60 * 1000000 * 1000,
            60 * 60 * 1000000 * 1000,
            60 * 1000000 * 1000,
            1000000 * 1000,
            1000 * 1000,
            1000,
            1,
        ]
        func_text = "def impl(td, freq):\n"
        for i, cond in enumerate(freq_conditions):
            cond_label = "if" if i == 0 else "elif"
            func_text += "    {} {}:\n".format(cond_label, cond)
            func_text += "        unit_value = {}\n".format(unit_values[i])
        func_text += "    else:\n"
        func_text += "        raise ValueError('Incorrect Frequency specification')\n"
        if td == pd_timedelta_type:
            func_text += "    return pd.Timedelta(unit_value * np.int64(np.{}(td.value / unit_value)))\n".format(
                method
            )
        elif td == pandas_timestamp_type:
            if method == "ceil":
                func_text += (
                    "    value = td.value + np.remainder(-td.value, unit_value)\n"
                )
            if method == "floor":
                func_text += (
                    "    value = td.value - np.remainder(td.value, unit_value)\n"
                )
            if method == "round":
                # Unit value is always even except value = 1
                func_text += "    if unit_value == 1:\n"
                func_text += "        value = td.value\n"
                func_text += "    else:\n"
                func_text += (
                    "        quotient, remainder = np.divmod(td.value, unit_value)\n"
                )
                func_text += "        mask = np.logical_or(remainder > (unit_value // 2), np.logical_and(remainder == (unit_value // 2), quotient % 2))\n"
                func_text += "        if mask:\n"
                func_text += "            quotient = quotient + 1\n"
                func_text += "        value = quotient * unit_value\n"
            func_text += (
                "    return pd.Timestamp(year=1970, month=1, day=1, nanosecond=value)\n"
            )
        loc_vars = {}
        exec(
            func_text,
            {"np": np, "pd": pd},
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return freq_overload


def _install_freq_methods():
    freq_methods = ["ceil", "floor", "round"]
    for method in freq_methods:
        overload_impl = overload_freq_methods(method)
        overload_method(PDTimeDeltaType, method, no_unliteral=True)(overload_impl)
        overload_method(PandasTimestampType, method, no_unliteral=True)(overload_impl)


_install_freq_methods()


# @intrinsic
@register_jitable
def compute_pd_timestamp(totmicrosec, nanosecond):  # pragma: no cover
    # number of microsecond
    microsecond = totmicrosec % 1000000
    totsecond = totmicrosec // 1000000
    # number of second
    second = totsecond % 60
    totminute = totsecond // 60
    # number of minute
    minute = totminute % 60
    tothour = totminute // 60
    # number of hour
    hour = tothour % 24
    totday = tothour // 24
    # computing year, month, day
    year, month, day = _ord2ymd(totday)
    #
    value = npy_datetimestruct_to_datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
    )
    value += zero_if_none(nanosecond)
    return init_timestamp(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        nanosecond,
        value,
    )


@overload(operator.sub, no_unliteral=True)
def timestamp_sub(lhs, rhs):
    if lhs == pandas_timestamp_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            # The time itself
            days1 = lhs.toordinal()
            secs1 = lhs.second + lhs.minute * 60 + lhs.hour * 3600
            msec1 = lhs.microsecond
            nanosecond = (
                lhs.nanosecond
            )  # That entries remain unchanged because timestamp has no nanosecond
            # The timedelta
            days2 = rhs.days
            secs2 = rhs.seconds
            msec2 = rhs.microseconds
            # Computing the difference
            daysF = days1 - days2
            secsF = secs1 - secs2
            msecF = msec1 - msec2
            # Getting total microsecond
            totmicrosec = 1000000 * (daysF * 86400 + secsF) + msecF
            return compute_pd_timestamp(totmicrosec, nanosecond)

        return impl

    if lhs == pandas_timestamp_type and rhs == pandas_timestamp_type:

        def impl_timestamp(lhs, rhs):  # pragma: no cover
            return convert_numpy_timedelta64_to_pd_timedelta(lhs.value - rhs.value)

        return impl_timestamp

    if lhs == pandas_timestamp_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl


def overload_add_operator_timestamp(lhs, rhs):
    if lhs == pandas_timestamp_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            # The time itself
            days1 = lhs.toordinal()
            secs1 = lhs.second + lhs.minute * 60 + lhs.hour * 3600
            msec1 = lhs.microsecond
            nanosecond = (
                lhs.nanosecond
            )  # That entries remain unchanged because timestamp has no nanosecond
            # The timedelta
            days2 = rhs.days
            secs2 = rhs.seconds
            msec2 = rhs.microseconds
            # Computing the difference
            daysF = days1 + days2
            secsF = secs1 + secs2
            msecF = msec1 + msec2
            # Getting total microsecond
            totmicrosec = 1000000 * (daysF * 86400 + secsF) + msecF
            return compute_pd_timestamp(totmicrosec, nanosecond)

        return impl

    if lhs == pandas_timestamp_type and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            # The time itself
            days1 = lhs.toordinal()
            secs1 = lhs.second + lhs.minute * 60 + lhs.hour * 3600
            msec1 = lhs.microsecond
            nanosec1 = lhs.nanosecond
            # The timedelta
            msec2 = rhs.value // 1000
            nanosec2 = rhs.nanoseconds
            # Computing the difference
            msecF = msec1 + msec2
            # Getting total microsecond
            totmicrosec = 1000000 * (days1 * 86400 + secs1) + msecF
            # Getting total nano_seconds
            totnanosec = nanosec1 + nanosec2
            return compute_pd_timestamp(totmicrosec, totnanosec)

        return impl

    # if lhs and rhs flipped, flip args and call add again
    if (lhs == pd_timedelta_type and rhs == pandas_timestamp_type) or (
        lhs == datetime_timedelta_type and rhs == pandas_timestamp_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl


@overload(min, no_unliteral=True)
def timestamp_min(lhs, rhs):
    if lhs == pandas_timestamp_type and rhs == pandas_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs < rhs else rhs

        return impl


@overload(max, no_unliteral=True)
def timestamp_max(lhs, rhs):
    if lhs == pandas_timestamp_type and rhs == pandas_timestamp_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs > rhs else rhs

        return impl


@overload_method(DatetimeDateType, "strftime", no_unliteral=True)
@overload_method(PandasTimestampType, "strftime", no_unliteral=True)
def strftime(ts, format_str):
    def impl(ts, format_str):  # pragma: no cover
        with numba.objmode(res="unicode_type"):
            res = ts.strftime(format_str)
        return res

    return impl


# -- builtin operators for dt64 ----------------------------------------------
# TODO: move to Numba


class CompDT64(ConcreteTemplate):
    cases = signature(types.boolean, types.NPDatetime("ns"), types.NPDatetime("ns"))


@infer_global(operator.lt)
class CmpOpLt(CompDT64):
    key = operator.lt


@infer_global(operator.le)
class CmpOpLe(CompDT64):
    key = operator.le


@infer_global(operator.gt)
class CmpOpGt(CompDT64):
    key = operator.gt


@infer_global(operator.ge)
class CmpOpGe(CompDT64):
    key = operator.ge


@infer_global(operator.eq)
class CmpOpEq(CompDT64):
    key = operator.eq


@infer_global(operator.ne)
class CmpOpNe(CompDT64):
    key = operator.ne


@typeof_impl.register(calendar._localized_month)
def typeof_python_calendar(val, c):
    return types.Tuple([types.StringLiteral(v) for v in val])
