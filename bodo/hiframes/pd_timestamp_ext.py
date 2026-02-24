"""Timestamp extension for Pandas Timestamp with timezone support."""

from __future__ import annotations

import calendar
import datetime
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pytz
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.core.typing.builtins import IndexValueType
from numba.core.typing.templates import (
    ConcreteTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
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
import bodo.libs.str_ext
import bodo.pandas as bd
import bodo.pandas_compat
import bodo.types
from bodo.hiframes.datetime_date_ext import (
    DatetimeDateType,
    _ord2ymd,
    _ymd2ord,
    get_isocalendar,
    str_2d,
)
from bodo.hiframes.datetime_timedelta_ext import (
    PDTimeDeltaType,
    _no_input,
    datetime_timedelta_type,
    pd_timedelta_type,
)
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.libs import hdatetime_ext
from bodo.libs.pd_datetime_arr_ext import get_tz_type_info
from bodo.libs.str_arr_ext import string_array_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_str,
    is_iterable_type,
    is_literal_type,
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
    "day_of_year",
    "dayofweek",
    "day_of_week",
    "daysinmonth",
    "days_in_month",
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "weekday",
]
date_methods = ["normalize", "day_name", "month_name"]

# Timedelta fields separated by return type
timedelta_fields = ["days", "seconds", "microseconds", "nanoseconds"]
timedelta_methods = ["total_seconds", "to_pytimedelta"]
iNaT = pd._libs.tslibs.iNaT


class PandasTimestampType(types.Type):
    def __init__(self, tz_val=None):
        self.tz = tz_val
        if tz_val is None:
            name = "PandasTimestampType()"
        else:
            name = f"PandasTimestampType({tz_val})"
        super().__init__(name=name)


pd_timestamp_tz_naive_type = PandasTimestampType()


def check_tz_aware_unsupported(val, func_name):
    """
    Checks if Timestamp, Array, DatetimeIndex, Series, or Series.dt
    if Timezone-aware but the intended operation doesn't support it.

    Raises an exception indicating the user must convert to timezone-naive
    """
    if isinstance(val, bodo.hiframes.series_dt_impl.SeriesDatetimePropertiesType):
        val = val.stype

    if isinstance(val, PandasTimestampType) and val.tz is not None:
        raise BodoError(
            f"{func_name} on Timezone-aware timestamp not yet supported. Please convert to timezone naive with ts.tz_convert(None)"
        )
    elif isinstance(val, bodo.types.DatetimeArrayType) and val.tz is not None:
        raise BodoError(
            f"{func_name} on Timezone-aware array not yet supported. Please convert to timezone naive with arr.tz_convert(None)"
        )
    elif (
        isinstance(val, bodo.types.DatetimeIndexType)
        and isinstance(val.data, bodo.types.DatetimeArrayType)
        and val.data.tz is not None
    ):
        raise BodoError(
            f"{func_name} on Timezone-aware index not yet supported. Please convert to timezone naive with index.tz_convert(None)"
        )
    elif (
        isinstance(val, bodo.types.SeriesType)
        and isinstance(val.data, bodo.types.DatetimeArrayType)
        and val.data.tz is not None
    ):
        raise BodoError(
            f"{func_name} on Timezone-aware series not yet supported. Please convert to timezone naive with series.dt.tz_convert(None)"
        )
    elif isinstance(val, bodo.types.DataFrameType):
        for arr_typ in val.data:
            if (
                isinstance(arr_typ, bodo.types.DatetimeArrayType)
                and arr_typ.tz is not None
            ):
                raise BodoError(
                    f"{func_name} on Timezone-aware columns not yet supported. Please convert each column to timezone naive with series.dt.tz_convert(None)"
                )


@typeof_impl.register(pd.Timestamp)
def typeof_pd_timestamp(val, c):
    return PandasTimestampType(get_tz_type_info(val.tz) if val.tz else None)


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
    args = c.pyapi.tuple_pack(())
    # NOTE: nanosecond argument is keyword-only as of Pandas 2
    kwargs = c.pyapi.dict_pack(
        [
            ("year", year_obj),
            ("month", month_obj),
            ("day", day_obj),
            ("hour", hour_obj),
            ("minute", minute_obj),
            ("second", second_obj),
            ("microsecond", us_obj),
            ("nanosecond", ns_obj),
        ]
    )
    res = c.pyapi.call(pdts_obj, args, kwargs)
    c.pyapi.decref(args)
    c.pyapi.decref(kwargs)

    if typ.tz is not None:
        if isinstance(typ.tz, int):
            tz_obj = c.pyapi.long_from_longlong(lir.Constant(lir.IntType(64), typ.tz))
        else:
            tz_str = c.context.insert_const_string(c.builder.module, str(typ.tz))
            tz_obj = c.pyapi.string_from_string(tz_str)

        false_obj = c.pyapi.from_native_value(
            types.bool_,
            c.context.get_constant_generic(c.builder, types.bool_, False),
            c.env_manager,
        )
        # Call ts.tz_localize(tz, False) instead of passing tz to Timestamp constructor
        # since the constructor will not allow ambiguous daylight saving times as of
        # Pandas 2.2.2.
        # See bodo/tests/test_timestamp_timezones.py::test_datetime_timedelta_sub
        ts_naive = res
        res = c.pyapi.call_method(ts_naive, "tz_localize", (tz_obj, false_obj))
        c.pyapi.decref(ts_naive)
        c.pyapi.decref(false_obj)
        c.pyapi.decref(tz_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(us_obj)
    c.pyapi.decref(ns_obj)
    return res


@intrinsic(prefer_literal=True)
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
    value,
    tz,
):
    """Create a PandasTimestampType with provided data values."""

    def codegen(context, builder, sig, args):
        year, month, day, hour, minute, second, us, ns, value, _ = args
        ts = cgutils.create_struct_proxy(sig.return_type)(context, builder)
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

    if is_overload_none(tz):
        typ = pd_timestamp_tz_naive_type
    elif is_overload_constant_str(tz):
        typ = PandasTimestampType(get_overload_const_str(tz))
    elif is_overload_constant_int(tz):
        typ = PandasTimestampType(get_overload_const_int(tz))
    else:
        raise_bodo_error("tz must be a constant string, int, or None")
    return (
        typ(
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            tz,
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
    """Constant lowering for PandasTimestampType"""

    year = context.get_constant(types.int64, pyval.year)
    month = context.get_constant(types.int64, pyval.month)
    day = context.get_constant(types.int64, pyval.day)
    hour = context.get_constant(types.int64, pyval.hour)
    minute = context.get_constant(types.int64, pyval.minute)
    second = context.get_constant(types.int64, pyval.second)
    microsecond = context.get_constant(types.int64, pyval.microsecond)
    nanosecond = context.get_constant(types.int64, pyval.nanosecond)
    value = context.get_constant(types.int64, pyval.value)

    return lir.Constant.literal_struct(
        (year, month, day, hour, minute, second, microsecond, nanosecond, value)
    )


# -------------------------------------------------------------------------------


def tz_has_transition_times(tz: str | int | None):
    """
    Return if a tz has different offsets from UTC at different times.
    This is useful for operations that moved by non-fixed amount (e.g. 1 Day)
    to determine what is required to compute the computation.

    Args:
        tz (Union[str, int, None]): Value stored in the tz field of the
        PandasTimestampType. Str is the name of a zone, int means a fixed
        offset (so no transition), and None is tz-naive.

    Returns:
        bool: Does this tz have different offsets from utc at different times.
    """
    # All timezones with transition times are strings in types.
    if isinstance(tz, str):
        # Compute the timezone.
        tz_info = pytz.timezone(tz)
        # timezones that ever transition are all DstTzInfo.
        # Unfortunately many of these may be very far in the past (e.g. Hawaii),
        # so there may be False positives for full correctness.
        return isinstance(tz_info, pytz.tzinfo.DstTzInfo)
    return False


# Overload regular Pandas and our exported Pandas Timestamp so
# "import bodo.pandas as pd" works correctly inside JIT as well.
@overload(pd.Timestamp, no_unliteral=True, jit_options={"cache": True})
@overload(bd.Timestamp, no_unliteral=True, jit_options={"cache": True})
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

    # check tz argument
    if (
        not is_overload_none(tz)
        and is_overload_constant_str(tz)
        and get_overload_const_str(tz) not in pytz.all_timezones_set
    ):
        raise BodoError(
            "pandas.Timestamp(): 'tz', if provided, must be constant string found in pytz.all_timezones"
        )

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
            return compute_val_for_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                tz,
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
            return compute_val_for_timestamp(
                ts_input,
                freq,
                tz,
                zero_if_none(unit),
                zero_if_none(year),
                zero_if_none(month),
                zero_if_none(day),
                zero_if_none(hour),
                # This result cannot have a timezone because the argument is overloaded.
                None,
            )

        return impl_pos

    # Pandas converts to dt64 and then back to a timestamp
    # https://github.com/pandas-dev/pandas/blob/4aa0783d65dc20dc450ef3f58defda14ebab5f6f/pandas/_libs/tslibs/conversion.pyx#L406
    if isinstance(ts_input, types.Number):
        if is_overload_none(unit):
            unit = "ns"
        if not is_overload_constant_str(unit):
            raise BodoError("pandas.Timedelta(): unit argument must be a constant str")
        unit = pd._libs.tslibs.timedeltas.parse_timedelta_unit(
            get_overload_const_str(unit)
        )
        nanoseconds, precision = bodo.pandas_compat.precision_from_unit_to_nanoseconds(
            unit
        )
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
                return convert_val_to_timestamp(value, tz)

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
            return convert_val_to_timestamp(value, tz)

        return impl_float

    # parse string input
    if ts_input == bodo.types.string_type or is_overload_constant_str(ts_input):
        # just call Pandas in this case since the string parsing code is complex and
        # handles several possible cases
        types.pd_timestamp_tz_naive_type = pd_timestamp_tz_naive_type

        if is_overload_none(tz):
            tz_val = None
        elif is_overload_constant_str(tz):
            tz_val = get_overload_const_str(tz)
        else:
            raise_bodo_error(
                "pandas.Timestamp(): tz argument must be a constant string or None"
            )

        typ = PandasTimestampType(tz_val)

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
            with numba.objmode(res=typ):
                res = pd.Timestamp(ts_input, tz=tz)
            return res

        return impl_str

    # for pd.Timestamp(), just return input
    if isinstance(ts_input, PandasTimestampType):
        return (
            lambda ts_input=_no_input,
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
            tzinfo=None: ts_input
        )  # pragma: no cover

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
            return compute_val_for_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                tz,
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
            year, month, day = ts_input._ymd

            return compute_val_for_timestamp(
                year,
                month,
                day,
                zero_if_none(hour),
                zero_if_none(minute),
                zero_if_none(second),
                zero_if_none(microsecond),
                zero_if_none(nanosecond),
                tz,
            )

        return impl_date

    if isinstance(ts_input, numba.core.types.scalars.NPDatetime):
        nanoseconds, precision = bodo.pandas_compat.precision_from_unit_to_nanoseconds(
            ts_input.unit
        )

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
            value = np.int64(ts_input) * nanoseconds
            # Pandas treats datetime64 as wall clock time
            return convert_val_to_timestamp(value, tz)

        return impl_date


@overload_attribute(PandasTimestampType, "dayofyear", jit_options={"cache": True})
@overload_attribute(PandasTimestampType, "day_of_year", jit_options={"cache": True})
def overload_pd_dayofyear(ptt):
    def pd_dayofyear(ptt):  # pragma: no cover
        return get_day_of_year(ptt.year, ptt.month, ptt.day)

    return pd_dayofyear


@overload_method(PandasTimestampType, "weekday", jit_options={"cache": True})
@overload_attribute(PandasTimestampType, "dayofweek", jit_options={"cache": True})
@overload_attribute(PandasTimestampType, "day_of_week", jit_options={"cache": True})
def overload_pd_dayofweek(ptt):
    def pd_dayofweek(ptt):  # pragma: no cover
        return get_day_of_week(ptt.year, ptt.month, ptt.day)

    return pd_dayofweek


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L138
@overload_attribute(PandasTimestampType, "week", jit_options={"cache": True})
@overload_attribute(PandasTimestampType, "weekofyear", jit_options={"cache": True})
def overload_week_number(ptt):
    def pd_week_number(ptt):
        # In the Gregorian calendar, week 1 is considered the week of the first Thursday
        # of the month. https://en.wikipedia.org/wiki/ISO_week_date#First_week
        # As a result, we need special handling depending on the first day of the year.

        # https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L161

        # Offset the day of the year by the (day of the week of jan 1st) and add 1 week because 1 indexed
        # Year starting on a Monday should be 7
        _, week, _ = get_isocalendar(ptt.year, ptt.month, ptt.day)
        return week

    return pd_week_number


@overload_method(PandasTimestampType, "utcoffset", jit_options={"cache": True})
def overload_utcoffset(ptt):
    """
    Overload for PandasTimestampType.utcoffset() method.
    """
    if ptt.tz is None:
        raise BodoError("utcoffset on Timezone-naive timestamp not supported")

    def pd_utcoffset(ptt):
        """
        Return the time zone offset from UTC.
        """
        return ptt.tz_localize(None) - ptt.tz_convert(None)

    return pd_utcoffset


@overload_method(
    PandasTimestampType, "__hash__", no_unliteral=True, jit_options={"cache": True}
)
def dt64_hash(val):
    return lambda val: hash(val.value)


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L59
@overload_attribute(PandasTimestampType, "days_in_month", jit_options={"cache": True})
@overload_attribute(PandasTimestampType, "daysinmonth", jit_options={"cache": True})
def overload_pd_daysinmonth(ptt):
    def pd_daysinmonth(ptt):  # pragma: no cover
        return get_days_in_month(ptt.year, ptt.month)

    return pd_daysinmonth


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L132
@overload_attribute(PandasTimestampType, "is_leap_year", jit_options={"cache": True})
def overload_pd_is_leap_year(ptt):
    def pd_is_leap_year(ptt):  # pragma: no cover
        return is_leap_year(ptt.year)

    return pd_is_leap_year


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L425
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_month_start", jit_options={"cache": True})
def overload_pd_is_month_start(ptt):
    def pd_is_month_start(ptt):  # pragma: no cover
        return ptt.day == 1

    return pd_is_month_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L436
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_month_end", jit_options={"cache": True})
def overload_pd_is_month_end(ptt):
    def pd_is_month_end(ptt):  # pragma: no cover
        return ptt.day == get_days_in_month(ptt.year, ptt.month)

    return pd_is_month_end


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L445
# Note we don't support business frequencies
@overload_attribute(
    PandasTimestampType, "is_quarter_start", jit_options={"cache": True}
)
def overload_pd_is_quarter_start(ptt):
    def pd_is_quarter_start(ptt):  # pragma: no cover
        return ptt.day == 1 and (ptt.month % 3) == 1

    return pd_is_quarter_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L456
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_quarter_end", jit_options={"cache": True})
def overload_pd_is_quarter_end(ptt):
    def pd_is_quarter_end(ptt):  # pragma: no cover
        return (ptt.month % 3) == 0 and ptt.day == get_days_in_month(
            ptt.year, ptt.month
        )

    return pd_is_quarter_end


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L466
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_year_start", jit_options={"cache": True})
def overload_pd_is_year_start(ptt):
    def pd_is_year_start(ptt):  # pragma: no cover
        return ptt.day == 1 and ptt.month == 1

    return pd_is_year_start


# Pandas Implementation:
# https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/timestamps.pyx#L476
# Note we don't support business frequencies
@overload_attribute(PandasTimestampType, "is_year_end", jit_options={"cache": True})
def overload_pd_is_year_end(ptt):
    def pd_is_year_end(ptt):  # pragma: no cover
        return ptt.day == 31 and ptt.month == 12

    return pd_is_year_end


@overload_attribute(PandasTimestampType, "quarter", jit_options={"cache": True})
def overload_quarter(ptt):
    # copied implementation from https://github.com/pandas-dev/pandas/blob/4859be9cd145e3da0a7f596c3e27636a58920c1c/pandas/_libs/tslibs/timestamps.pyx#L547
    def quarter(ptt):  # pragma: no cover
        return ((ptt.month - 1) // 3) + 1

    return quarter


@overload_method(
    PandasTimestampType, "date", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_date(ptt):
    def pd_timestamp_date_impl(ptt):  # pragma: no cover
        return datetime.date(ptt.year, ptt.month, ptt.day)

    return pd_timestamp_date_impl


@overload_method(
    PandasTimestampType, "isocalendar", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_isocalendar(ptt):
    def impl(ptt):  # pragma: no cover
        year, week, day_of_week = get_isocalendar(ptt.year, ptt.month, ptt.day)
        return (year, week, day_of_week)

    return impl


@overload_method(
    PandasTimestampType, "isoformat", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_isoformat(ts, sep=None):
    has_tz = ts.tz is not None
    if is_overload_none(sep):

        def timestamp_isoformat_impl(ts, sep=None):  # pragma: no cover
            _time = str_2d(ts.hour) + ":" + str_2d(ts.minute) + ":" + str_2d(ts.second)
            if ts.microsecond != 0:
                _time += "." + str_2d(ts.microsecond)
                if ts.nanosecond != 0:
                    _time += str_2d(ts.nanosecond)
            _tz = ""
            if has_tz:
                # strftime returns (-/+) HHMM for UTC offset, when the default Bodo
                # timezone format is (-/+) HH:MM. So we must manually insert a ":" character
                utc_offset = ts.strftime("%z")
                _tz = f"{utc_offset[:3]}:{utc_offset[3:]}"
            res = (
                str(ts.year)
                + "-"
                + str_2d(ts.month)
                + "-"
                + str_2d(ts.day)
                + "T"
                + _time
                + _tz
            )
            return res

        return timestamp_isoformat_impl

    else:

        def timestamp_isoformat_impl(ts, sep=None):  # pragma: no cover
            _time = str_2d(ts.hour) + ":" + str_2d(ts.minute) + ":" + str_2d(ts.second)
            if ts.microsecond != 0:
                _time += "." + str_2d(ts.microsecond)
                if ts.nanosecond != 0:
                    _time += str_2d(ts.nanosecond)
            _tz = ""
            if has_tz:
                # strftime returns (-/+) HHMM for UTC offset, when the default Bodo
                # timezone format is (-/+) HH:MM. So we must manually insert a ":" character
                utc_offset = ts.strftime("%z")
                _tz = f"{utc_offset[:3]}:{utc_offset[3:]}"

            res = (
                str(ts.year)
                + "-"
                + str_2d(ts.month)
                + "-"
                + str_2d(ts.day)
                + sep
                + _time
                + _tz
            )
            return res

    return timestamp_isoformat_impl


@overload_method(
    PandasTimestampType, "normalize", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_normalize(ptt):
    tz_literal = ptt.tz

    def impl(ptt):  # pragma: no cover
        return pd.Timestamp(year=ptt.year, month=ptt.month, day=ptt.day, tz=tz_literal)

    return impl


@overload_method(
    PandasTimestampType, "day_name", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_day_name(ptt, locale=None):
    """
    Support for Timestamp.day_name(). This returns the full name
    of the day of the week as a string.
    """
    unsupported_args = {"locale": locale}
    arg_defaults = {"locale": None}
    check_unsupported_args(
        "Timestamp.day_name",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Timestamp",
    )

    def impl(ptt, locale=None):  # pragma: no cover
        day_names = (
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        )
        # Gets the day of the week 1-indexed: Monday = 1, Sunday = 7
        _, _, day_num = ptt.isocalendar()
        return day_names[day_num - 1]

    return impl


@overload_method(
    PandasTimestampType, "month_name", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_month_name(ptt, locale=None):
    """
    Support for Timestamp.month_name(). This returns the full name
    of the month as a string.
    """
    unsupported_args = {"locale": locale}
    arg_defaults = {"locale": None}
    check_unsupported_args(
        "Timestamp.month_name",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Timestamp",
    )

    def impl(ptt, locale=None):  # pragma: no cover
        month_names = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        return month_names[ptt.month - 1]

    return impl


@overload_method(
    PandasTimestampType, "tz_convert", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_tz_convert(ptt, tz):
    if ptt.tz is None:
        # TODO: tz_localize
        raise BodoError(
            "Cannot convert tz-naive Timestamp, use tz_localize to localize"
        )

    if is_overload_none(tz):
        return lambda ptt, tz: convert_val_to_timestamp(ptt.value)
    elif is_overload_constant_str(tz):
        return lambda ptt, tz: convert_val_to_timestamp(ptt.value, tz=tz)


@overload_method(
    PandasTimestampType, "tz_localize", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_tz_localize(ptt, tz, ambiguous="raise", nonexistent="raise"):
    if ptt.tz is not None and not is_overload_none(tz):
        raise BodoError(
            "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
        )
    unsupported_args = {"ambiguous": ambiguous, "nonexistent": nonexistent}
    defaults_args = {"ambiguous": "raise", "nonexistent": "raise"}
    check_unsupported_args(
        "Timestamp.tz_localize",
        unsupported_args,
        defaults_args,
        package_name="pandas",
        module_name="Timestamp",
    )
    # Create a fast path for a naive timezone that remains naive
    if is_overload_none(tz) and ptt.tz is None:
        # Create a fast path for a naive timezone that remains naive
        return (
            lambda ptt, tz, ambiguous="raise", nonexistent="raise": ptt
        )  # pragma: no cover
    if is_overload_none(tz):
        # If we are converting to naive then we add
        # the delta for the current timestamp.
        tz_value = ptt.tz
        negate = False
    else:
        # If we are converting from naive to tz-aware then
        # we need to subtract the delta.
        if not is_literal_type(tz):  # pragma: no cover
            raise_bodo_error(
                "Timestamp.tz_localize(): tz value must be a literal string, integer, or None"
            )
        tz_value = get_literal_value(tz)
        negate = True

    deltas = None
    trans = None
    check_transitions = False
    if tz_has_transition_times(tz_value):
        # We only need to check transitions if we are converting
        # from UTC.
        check_transitions = negate
        tz_obj = pytz.timezone(tz_value)
        trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view("i8")
        deltas = np.array(tz_obj._transition_info)[:, 0]
        deltas = (
            (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
            .astype(np.int64)
            .values
        )
        delta_str = "deltas[np.searchsorted(trans, value, side='right') - 1]"
    elif isinstance(tz_value, str):
        # Here we are certain there are no transition times so we can compute a fixed offset
        # from the timezone
        tz_obj = pytz.timezone(tz_value)
        # Convert to nanoseconds.
        delta_str = str(np.int64(tz_obj._utcoffset.total_seconds() * 1_000_000_000))
    elif isinstance(tz_value, int):
        # Integers are always the offset in nanoseconds
        delta_str = str(tz_value)
    else:  # pragma: no cover
        raise_bodo_error(
            "Timestamp.tz_localize(): tz value must be a literal string, integer, or None"
        )

    if negate:
        sign = "-"
    else:
        sign = "+"

    func_text = "def bodo_pd_timestamp_tz_localize(ptt, tz, ambiguous='raise', nonexistent='raise'):\n"
    func_text += "    value =  ptt.value\n"
    func_text += f"    delta =  {delta_str}\n"
    func_text += f"    new_value = value {sign} delta\n"
    if check_transitions:
        func_text += "    end_delta = deltas[np.searchsorted(trans, new_value, side='right') - 1]\n"
        func_text += "    offset = delta - end_delta\n"
        func_text += "    new_value = new_value + offset\n"
    func_text += "    return convert_val_to_timestamp(new_value, tz=tz)\n"
    return bodo.utils.utils.bodo_exec(
        func_text,
        {
            "np": np,
            "convert_val_to_timestamp": convert_val_to_timestamp,
            "trans": trans,
            "deltas": deltas,
        },
        {},
        __name__,
    )


@overload_method(PandasTimestampType, "__str__", jit_options={"cache": True})
def timestamp_str_overload(a):
    return lambda a: a.isoformat(" ")  # pragma: no cover


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
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="extract_year_days"
        )
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
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_month_day"
        )
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


@numba.generated_jit(nopython=True, no_unliteral=True)
def compute_val_for_timestamp(
    year,
    month,
    day,
    hour,
    minute,
    second,
    microsecond,
    nanosecond,
    tz,
):
    """
    Computes the correct value for the given timezone and outputs the correct Timestamp
    given the appropriate field values.
    """
    delta_str = "0"
    tz_value = get_literal_value(tz)
    deltas = None
    trans = None
    check_transitions = False
    if tz_has_transition_times(tz_value):
        check_transitions = True
        tz_obj = pytz.timezone(tz_value)
        trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view("i8")
        deltas = np.array(tz_obj._transition_info)[:, 0]
        deltas = (
            (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
            .astype(np.int64)
            .values
        )
        delta_str = "deltas[np.searchsorted(trans, original_value, side='right') - 1]"
    elif isinstance(tz_value, str):
        # Here we are certain there are no transition times so we can compute a fixed offset
        # from the timezone
        tz_obj = pytz.timezone(tz_value)
        # Convert to nanoseconds.
        delta_str = str(np.int64(tz_obj._utcoffset.total_seconds() * 1_000_000_000))
    elif isinstance(tz_value, int):
        # Integers are always the offset in nanoseconds
        delta_str = str(tz_value)
    elif tz_value is not None:
        raise_bodo_error(
            "compute_val_for_timestamp(): tz value must be a constant string, integer or None"
        )

    func_text = "def impl(year, month, day, hour, minute, second, microsecond, nanosecond, tz):\n"
    # Subtract the offset
    func_text += "  original_value = npy_datetimestruct_to_datetime(year, month, day, hour, minute, second, microsecond) + nanosecond\n"
    # In this case the delta is a fixed offset in nanoseconds that is the amount to add from utc to the current
    # timezone. As a result we subtract this value.
    # Example:
    # In [1]: import pytz
    # In [2]: pd.Timestamp('2020-03-15', tz=pytz.FixedOffset(8)).value
    # Out[2]: 1584229920000000000
    # In [3]: pd.Timestamp('2020-03-15').value
    # Out[3]: 1584230400000000000
    # Here delta would be 480000000000 (to convert from 3 to 2)
    func_text += f"  value = original_value - {delta_str}\n"
    if check_transitions:
        func_text += (
            "  start_trans = np.searchsorted(trans, original_value, side='right') - 1\n"
        )
        func_text += "  end_trans = np.searchsorted(trans, value, side='right') - 1\n"
        func_text += "  offset = deltas[start_trans] - deltas[end_trans]\n"
        func_text += "  value = value + offset\n"
    func_text += "  return init_timestamp(\n"
    func_text += "    year=year,\n"
    func_text += "    month=month,\n"
    func_text += "    day=day,\n"
    func_text += "    hour=hour,\n"
    func_text += "    minute=minute,"
    func_text += "    second=second,\n"
    func_text += "    microsecond=microsecond,\n"
    func_text += "    nanosecond=nanosecond,\n"
    func_text += "    value=value,\n"
    func_text += "    tz=tz,\n"
    func_text += "  )\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "pd": pd,
            "init_timestamp": init_timestamp,
            "npy_datetimestruct_to_datetime": npy_datetimestruct_to_datetime,
            "trans": trans,
            "deltas": deltas,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def convert_val_to_timestamp(ts_input, tz=None, is_convert=True):
    """
    Converts value given in nanoseconds to timestamp with appropriate timezone.
    If is_convert, ts_input's value is taken to be in UTC and is stored directly.
    Otherwise a new value in the appropriate tz is calculated (e.g. tz_localize).
    """
    trans = deltas = np.array([])
    delta_str = "0"
    if is_overload_constant_str(tz):
        tz_str = get_overload_const_str(tz)
        tz_obj = pytz.timezone(tz_str)
        if isinstance(tz_obj, pytz.tzinfo.DstTzInfo):
            # Most timezones are pytz.tzinfo.DstTzInfo
            trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view("i8")
            deltas = np.array(tz_obj._transition_info)[:, 0]
            deltas = (
                (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                .astype(np.int64)
                .values
            )  # np.array(tz_obj._transition_info)[:, 0]
            delta_str = "deltas[np.searchsorted(trans, ts_input, side='right') - 1]"
        else:
            # Some are pytz.tzinfo.StaticTzInfo (e.g. 'HST') or pytz.utc
            deltas = np.int64(tz_obj._utcoffset.total_seconds() * 1_000_000_000)
            delta_str = "deltas"
    elif is_overload_constant_int(tz):
        ns_int = get_overload_const_int(tz)
        delta_str = str(ns_int)
    elif not is_overload_none(tz):
        raise_bodo_error(
            "convert_val_to_timestamp(): tz value must be a constant string or None"
        )
    is_convert = get_overload_const_bool(is_convert)
    if is_convert:
        dt_val = "tz_ts_input"
        new_val = "ts_input"
    else:
        dt_val = "ts_input"
        new_val = "tz_ts_input"

    func_text = "def impl(ts_input, tz=None, is_convert=True):\n"
    func_text += f"  tz_ts_input = ts_input + {delta_str}\n"
    func_text += f"  dt, year, days = extract_year_days(integer_to_dt64({dt_val}))\n"
    func_text += "  month, day = get_month_day(year, days)\n"
    func_text += "  return init_timestamp(\n"
    func_text += "    year=year,\n"
    func_text += "    month=month,\n"
    func_text += "    day=day,\n"
    func_text += "    hour=dt // (60 * 60 * 1_000_000_000),\n"  # hour
    func_text += "    minute=(dt // (60 * 1_000_000_000)) % 60,\n"  # minute
    func_text += "    second=(dt // 1_000_000_000) % 60,\n"  # second
    func_text += "    microsecond=(dt // 1000) % 1_000_000,\n"  # microsecond
    func_text += "    nanosecond=dt % 1000,\n"  # nanosecond
    func_text += f"    value={new_val},\n"
    func_text += "    tz=tz,\n"
    func_text += "  )\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "pd": pd,
            "trans": trans,
            "deltas": deltas,
            "integer_to_dt64": integer_to_dt64,
            "extract_year_days": extract_year_days,
            "get_month_day": get_month_day,
            "init_timestamp": init_timestamp,
            "zero_if_none": zero_if_none,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


@numba.njit(cache=True, no_cpython_wrapper=True)
def convert_datetime64_to_timestamp(dt64):  # pragma: no cover
    """Converts dt64 value to pd.Timestamp"""
    dt, year, days = extract_year_days(dt64)
    month, day = get_month_day(year, days)

    return init_timestamp(
        year=year,
        month=month,
        day=day,
        hour=dt // (60 * 60 * 1000000000),  # hour
        minute=(dt // (60 * 1000000000)) % 60,  # minute
        second=(dt // 1000000000) % 60,  # second
        microsecond=(dt // 1000) % 1000000,  # microsecond
        nanosecond=dt % 1000,  # nanosecond
        value=dt64,
        tz=None,
    )


@numba.njit(cache=True, no_cpython_wrapper=True)
def convert_numpy_timedelta64_to_datetime_timedelta(dt64):  # pragma: no cover
    """Converts numpy.timedelta64 to datetime.timedelta"""
    n_int64 = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(dt64)
    n_day = n_int64 // (86400 * 1000000000)
    res1 = n_int64 - n_day * 86400 * 1000000000
    n_sec = res1 // 1000000000
    res2 = res1 - n_sec * 1000000000
    n_microsec = res2 // 1000
    return datetime.timedelta(n_day, n_sec, n_microsec)


def convert_numpy_timedelta64_to_pd_timedelta(td64):  # pragma: no cover
    return td64


@overload(convert_numpy_timedelta64_to_pd_timedelta, jit_options={"cache": True})
def overload_convert_numpy_timedelta64_to_pd_timedelta(td64):
    """Converts numpy.timedelta64 to pd.Timedelta"""

    if td64 == bodo.types.pd_timedelta_type:
        return lambda td64: td64  # pragma: no cover

    def impl(td64):  # pragma: no cover
        n_int64 = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(td64)
        return pd.Timedelta(n_int64)

    return impl


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

    if val == bodo.types.pd_timestamp_tz_naive_type:

        def codegen(context, builder, sig, args):
            timestamp = cgutils.create_struct_proxy(val)(
                context, builder, value=args[0]
            )
            return timestamp.value
    else:

        def codegen(context, builder, sig, args):
            return args[0]

    return types.int64(val), codegen


@lower_cast(types.NPDatetime("ns"), types.int64)
def cast_dt64_to_integer(context, builder, fromty, toty, val):
    # dt64 is stored as int64 so just return value
    return val


# TODO: fix in Numba
@overload_method(
    types.NPDatetime, "__hash__", no_unliteral=True, jit_options={"cache": True}
)
def dt64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


# TODO: fix in Numba
@overload_method(
    types.NPTimedelta, "__hash__", no_unliteral=True, jit_options={"cache": True}
)
def td64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


@intrinsic
def timedelta64_to_integer(typingctx, val=None):
    """Cast a timedelta64 value to integer"""

    if val == pd_timedelta_type:

        def codegen(context, builder, sig, args):
            timedelta = cgutils.create_struct_proxy(val)(
                context, builder, value=args[0]
            )
            return timedelta.value
    else:

        def codegen(context, builder, sig, args):
            return args[0]

    return types.int64(val), codegen


@lower_cast(numba.core.types.NPTimedelta("ns"), types.int64)
def cast_td64_to_integer(context, builder, fromty, toty, val):
    # td64 is stored as int64 so just return value
    return val


@numba.njit(cache=True)
def parse_datetime_str(val):  # pragma: no cover
    """Parse datetime string value to dt64
    Just calling Pandas since the Pandas code is complex
    """
    with numba.objmode(res="int64"):
        res = pd.Timestamp(val).value
    return integer_to_dt64(res)


@numba.njit(cache=True)
def datetime_timedelta_to_timedelta64(val):  # pragma: no cover
    """convert datetime.timedelta to np.timedelta64"""
    with numba.objmode(res='NPTimedelta("ns")'):
        res = pd.to_timedelta(val)
        # Pandas 2 returns us precision for some reason
        res = res.to_timedelta64().astype(np.dtype("timedelta64[ns]"))
    return res


@numba.njit(cache=True)
def series_str_dt64_astype(data):  # pragma: no cover
    """convert string array to datetime64 array using
    objmode and Series implementation."""
    with numba.objmode(res="NPDatetime('ns')[::1]"):
        # Convert to series to enable Pandas str parsing.
        # This enables conversions not supported in just Numba.
        # call ArrowStringArray.to_numpy() since PyArrow can't convert all datetime
        # formats, see test_dt64_str_astype
        res = pd.to_datetime(pd.Series(data.to_numpy()), format="mixed").values.astype(
            np.dtype("datetime64[ns]")
        )
    return res


@numba.njit(cache=True)
def series_str_td64_astype(data):  # pragma: no cover
    """convert string array to timedelta64 array using
    objmode."""
    with numba.objmode(res="NPTimedelta('ns')[::1]"):
        # No need to use Series because Timedelta doesn't
        # have extra parsing.
        # NOTE: Pandas 2 returns a TimedeltaArray so to_numpy is needed
        res = data.astype("timedelta64[ns]").to_numpy()
    return res


@numba.njit(cache=True)
def datetime_datetime_to_dt64(val):  # pragma: no cover
    """convert datetime.datetime to np.datetime64"""
    return integer_to_dt64(pd.Timestamp(val).value)


@register_jitable
def datetime_date_arr_to_dt64_arr(arr):  # pragma: no cover
    """convert array of datetime.date to np.datetime64"""
    n = len(arr)
    res = np.empty(n, bodo.types.datetime64ns)
    for i in range(n):
        if bodo.libs.array_kernels.isna(arr, i):
            bodo.libs.array_kernels.setna(res, i)
            continue
        res[i] = integer_to_dt64(pd.Timestamp(arr[i]).value)

    return res


types.pd_timestamp_tz_naive_type = pd_timestamp_tz_naive_type


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
    origin="unix",
    cache=True,
):  # pragma: no cover
    """call pd.to_datetime() with scalar value 'a'
    separate call to avoid adding extra basic blocks to user function for simplicity
    """
    with numba.objmode(t="pd_timestamp_tz_naive_type"):
        # A `tz_localize(None)` is required to handle inputs with a tz offset
        # because the return type is a naive timestamp.
        t = pd.to_datetime(
            a,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            origin=origin,
            cache=cache,
        ).tz_localize(None)
    return t


@numba.njit(cache=True)
def pandas_string_array_to_datetime(
    arr,
    errors,
    dayfirst,
    yearfirst,
    utc,
    format,
    exact,
    unit,
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
            origin=origin,
            cache=cache,
        )
    return result


@numba.njit(cache=True)
def pandas_dict_string_array_to_datetime(
    arr,
    errors,
    dayfirst,
    yearfirst,
    utc,
    format,
    exact,
    unit,
    origin,
    cache,
):  # pragma: no cover
    """
    Implementation of converting a dictionary array of strings
    to datetime. This is grouped into an intermediate function to
    avoid parallelism issues from extracting arr._data. Calling
    pandas_string_array_to_datetime shouldn't be exposed to the main
    IR for this example because arr._data isn't truly REP or DIST.
    """
    n = len(arr)
    B = np.empty(n, "datetime64[ns]")
    indices = arr._indices
    # get datetime64 value for each dictionary value.
    # Use an intermediate functions to avoid distribution issues.
    dt64_vals = pandas_string_array_to_datetime(
        arr._data,
        errors,
        dayfirst,
        yearfirst,
        utc,
        format,
        exact,
        unit,
        origin,
        cache,
    ).values
    for i in range(n):
        if bodo.libs.array_kernels.isna(indices, i):
            bodo.libs.array_kernels.setna(B, i)
            continue
        B[i] = dt64_vals[indices[i]]
    return B


@overload(
    pd.to_datetime, inline="always", no_unliteral=True, jit_options={"cache": True}
)
def overload_to_datetime(
    arg_a,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    origin="unix",
    cache=True,
):
    """implementation for pd.to_datetime"""
    # TODO: change 'arg_a' to 'arg' when inliner can handle it

    args_dict = {
        "errors": errors,
    }
    args_default_dict = {
        "errors": "raise",
    }

    # We don't support errors, as 'ignore' and 'coerce' both can cause type instability
    check_unsupported_args(
        "pd.to_datetime",
        args_dict,
        args_default_dict,
        package_name="pandas",
    )

    # This covers string as a literal or not
    # and integer as a literal or not
    if (
        arg_a == bodo.types.string_type
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
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            n = len(arg_a)
            B = np.empty(n, dt64_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                val = iNaT
                if not bodo.libs.array_kernels.isna(arg_a, i):
                    data = arg_a[i]
                    year, month, day = data._ymd
                    val = bodo.hiframes.pd_timestamp_ext.npy_datetimestruct_to_datetime(
                        year, month, day, 0, 0, 0, 0
                    )
                B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val)
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_date_arr

    # return DatetimeIndex if input is array(dt64)
    if arg_a == types.Array(types.NPDatetime("ns"), 1, "C") or isinstance(
        arg_a, bodo.types.DatetimeArrayType
    ):
        return (
            lambda arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True: bodo.hiframes.pd_index_ext.init_datetime_index(arg_a, None)
        )  # pragma: no cover

    # string_array_type as input
    if arg_a == string_array_type:

        def impl_string_array(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
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
                    origin=origin,
                    cache=cache,
                )
                B[i] = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_date_arr

    # Categorical array with string values
    if (
        isinstance(arg_a, CategoricalArrayType)
        and arg_a.dtype.elem_type == bodo.types.string_type
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

    # Dictionary-encoded string array
    if arg_a == bodo.types.dict_str_arr_type:

        def impl_dict_str_arr(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            B = pandas_dict_string_array_to_datetime(
                arg_a,
                errors,
                dayfirst,
                yearfirst,
                utc,
                format,
                exact,
                unit,
                origin,
                cache,
            )
            return bodo.hiframes.pd_index_ext.init_datetime_index(B, None)

        return impl_dict_str_arr

    # Timestamp input. This ignores other fields and just returns Timestamp
    # TODO: Support useful fields like unit without objmode
    if isinstance(arg_a, PandasTimestampType):

        def impl_timestamp(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            # TODO: Support other args like unit
            return arg_a

        return impl_timestamp

    # np datetime input. This ignores other fields and just returns value wrapped
    # in a timestamp
    if arg_a == bodo.types.datetime64ns:

        def impl_np_datetime(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            return pd.Timestamp(arg_a)

        return impl_np_datetime

    # datetime.date input. This ignores other fields and just returns value wrapped
    # in a timestamp
    if arg_a == bodo.hiframes.datetime_date_ext.datetime_date_type:

        def impl_date(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            return pd.Timestamp(arg_a)

        return impl_date

    if is_overload_none(arg_a):  # pragma: no cover

        def impl_np_datetime(
            arg_a,
            errors="raise",
            dayfirst=False,
            yearfirst=False,
            utc=None,
            format=None,
            exact=True,
            unit=None,
            origin="unix",
            cache=True,
        ):  # pragma: no cover
            return None

        return impl_np_datetime

    # TODO: input Type of a dataframe
    raise_bodo_error(
        f"pd.to_datetime(): cannot convert data type {arg_a}"
    )  # pragma: no cover


@overload(
    pd.to_timedelta, inline="always", no_unliteral=True, jit_options={"cache": True}
)
def overload_to_timedelta(arg_a, unit="ns", errors="raise"):
    # changed 'arg' to 'arg_a' since inliner uses vname.startswith("arg.") to find
    # argument variables which causes conflict
    # TODO: fix call inliner to hande 'arg' name properly

    if not is_overload_constant_str(unit):  # pragma: no cover
        raise BodoError("pandas.to_timedelta(): unit should be a constant string")

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
        bodo.types.string_type,
    ):

        def impl_string(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            return pd.Timedelta(arg_a)

        return impl_string

    # Float scalar
    if isinstance(arg_a, types.Float):
        m, p = bodo.pandas_compat.precision_from_unit_to_nanoseconds(unit)

        def impl_float_scalar(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            val = float_to_timedelta_val(arg_a, p, m)
            return pd.Timedelta(val)

        return impl_float_scalar

    # Integer scalar
    if isinstance(arg_a, types.Integer):
        m, _ = bodo.pandas_compat.precision_from_unit_to_nanoseconds(unit)

        def impl_integer_scalar(arg_a, unit="ns", errors="raise"):  # pragma: no cover
            return pd.Timedelta(arg_a * m)

        return impl_integer_scalar

    # TODO: Add tuple support. These require separate kernels because we cannot check isna for tuples
    if is_iterable_type(arg_a) and not isinstance(arg_a, types.BaseTuple):
        m, p = bodo.pandas_compat.precision_from_unit_to_nanoseconds(unit)
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

        if arg_a.dtype == bodo.types.timedelta64ns:

            def impl_td64(arg_a, unit="ns", errors="raise"):  # pragma: no cover
                arr = bodo.utils.conversion.coerce_to_ndarray(arg_a)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(arr, None)

            return impl_td64

        # Either a string array or numpy unichr array
        if arg_a.dtype == bodo.types.string_type or isinstance(
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

        if arg_a == bodo.types.timedelta_array_type:

            def impl_timedelta_arr(
                arg_a, unit="ns", errors="raise"
            ):  # pragma: no cover
                n = len(arg_a)
                B = np.empty(n, td64_dtype)
                for i in numba.parfors.parfor.internal_prange(n):
                    val = iNaT
                    if not bodo.libs.array_kernels.isna(arg_a, i):
                        timedelta_val = arg_a[i]
                        val = timedelta_val.value
                    B[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val)
                return bodo.hiframes.pd_index_ext.init_timedelta_index(B, None)

            return impl_timedelta_arr

    if is_overload_none(arg_a):  # pragma: no cover
        # None input
        return lambda arg_a, unit="ns", errors="raise": None

    raise_bodo_error(f"pd.to_timedelta(): cannot convert data type {arg_a.dtype}")


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


@numba.njit(cache=True)
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

    def overload_date_timestamp_cmp(lhs, rhs):
        # Timestamp, datetime.date
        if (
            isinstance(lhs, PandasTimestampType)
            and rhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        ):
            tz_literal = lhs.tz
            # Compare using date to handle timezones

            if tz_literal is None:
                # Fast path for timezone-naive case: simply compare
                # integers (ns).
                return lambda lhs, rhs: op(
                    lhs.value,
                    bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int_ns(rhs),
                )  # pragma: no cover
            else:
                return lambda lhs, rhs: op(
                    lhs,
                    # Convert the date to the same tz timestamp.
                    pd.Timestamp(rhs, tz=tz_literal),
                )  # pragma: no cover

        # datetime.date, Timestamp
        if lhs == bodo.hiframes.datetime_date_ext.datetime_date_type and isinstance(
            rhs, PandasTimestampType
        ):
            tz_literal = rhs.tz
            if tz_literal is None:
                # Fast path for timezone-naive case: simply compare
                # integers (ns).
                return lambda lhs, rhs: op(
                    bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int_ns(lhs),
                    rhs.value,
                )  # pragma: no cover
            else:
                return lambda lhs, rhs: op(
                    # Convert the date to the same tz timestamp.
                    pd.Timestamp(lhs, tz=tz_literal),
                    rhs,
                )  # pragma: no cover

        # Timestamp/Timestamp
        if isinstance(lhs, PandasTimestampType) and isinstance(
            rhs, PandasTimestampType
        ):
            if lhs.tz == rhs.tz:
                return lambda lhs, rhs: op(lhs.value, rhs.value)
            elif lhs.tz is None:
                return lambda lhs, rhs: op(lhs.value, rhs.tz_localize(None).value)
            elif rhs.tz is None:
                return lambda lhs, rhs: op(lhs.tz_localize(None).value, rhs.value)
            else:
                # TODO: Support comparison operator between timestamps with different time-zone
                raise BodoError(
                    f"{numba.core.utils.OPERATORS_TO_BUILTINS[op]} with two Timestamps requires both Timestamps share the same timezone. "
                    + f"Argument 0 has timezone {lhs.tz} and argument 1 has timezone {rhs.tz}. "
                    + "To compare these values please convert to timezone naive with ts.tz_convert(None)."
                )

        # Timestamp/dt64 scalar
        if isinstance(lhs, PandasTimestampType) and rhs == bodo.types.datetime64ns:
            if lhs.tz is not None:
                return lambda lhs, rhs: op(
                    bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        lhs.tz_localize(None).value
                    ),
                    rhs,
                )  # pragma: no cover
            else:
                return lambda lhs, rhs: op(
                    bodo.hiframes.pd_timestamp_ext.integer_to_dt64(lhs.value), rhs
                )  # pragma: no cover

        # dt64 scalar/Timestamp
        if lhs == bodo.types.datetime64ns and isinstance(rhs, PandasTimestampType):
            if rhs.tz is not None:
                return lambda lhs, rhs: op(
                    lhs,
                    bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        rhs.tz_localize(None).value
                    ),
                )  # pragma: no cover
            else:
                return lambda lhs, rhs: op(
                    lhs, bodo.hiframes.pd_timestamp_ext.integer_to_dt64(rhs.value)
                )  # pragma: no cover

    return overload_date_timestamp_cmp


@overload_method(
    PandasTimestampType, "toordinal", no_unliteral=True, jit_options={"cache": True}
)
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
    def freq_overload(td, freq, ambiguous="raise", nonexistent="raise"):
        unsupported_args = {"ambiguous": ambiguous, "nonexistent": nonexistent}
        floor_defaults = {"ambiguous": "raise", "nonexistent": "raise"}
        check_unsupported_args(
            f"Timestamp.{method}",
            unsupported_args,
            floor_defaults,
            package_name="pandas",
            module_name="Timestamp",
        )
        freq_conditions = [
            "freq == 'D'",
            "freq == 'h'",
            "freq == 'min'",
            "freq == 's'",
            "freq == 'ms'",
            "freq == 'us'",
            "freq == 'ns'",
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
        # Used by tz-aware timezones
        deltas = None
        trans = None
        tz_literal = None
        func_text = "def impl(td, freq, ambiguous='raise', nonexistent='raise'):\n"
        for i, cond in enumerate(freq_conditions):
            cond_label = "if" if i == 0 else "elif"
            func_text += f"    {cond_label} {cond}:\n"
            func_text += f"        unit_value = {unit_values[i]}\n"
        func_text += "    else:\n"
        func_text += "        raise ValueError('Incorrect Frequency specification')\n"
        if td == pd_timedelta_type:
            func_text += f"    return pd.Timedelta(unit_value * np.int64(np.{method}(td.value / unit_value)))\n"
        else:
            assert isinstance(td, PandasTimestampType), "Value must be a timestamp"
            func_text += "    value = td.value\n"
            tz_literal = td.tz
            if tz_literal is not None:
                # If we a timezone we need to remove the offset from UTC before computing
                # the result.
                # TODO: We can check the timezone offsets and only apply changes to the offset
                # for frequencies >= minimum offset unit.
                delta_str = "0"
                has_transitions = False
                if tz_has_transition_times(tz_literal):
                    has_transitions = True
                    tz_obj = pytz.timezone(tz_literal)
                    trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view(
                        "i8"
                    )
                    deltas = np.array(tz_obj._transition_info)[:, 0]
                    deltas = (
                        (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                        .astype(np.int64)
                        .values
                    )
                    delta_str = (
                        "deltas[np.searchsorted(trans, value, side='right') - 1]"
                    )
                elif isinstance(tz_literal, str):
                    # Here we are certain there are no transition times so we can compute a fixed offset
                    # from the timezone
                    tz_obj = pytz.timezone(tz_literal)
                    # Convert to nanoseconds.
                    delta_str = str(
                        np.int64(tz_obj._utcoffset.total_seconds() * 1_000_000_000)
                    )
                elif isinstance(tz_literal, int):
                    # Integers are always the offset in nanoseconds
                    delta_str = str(tz_literal)
                func_text += f"    delta = {delta_str}\n"
                func_text += "    value = value + delta\n"

            if method == "ceil":
                func_text += "    value = value + np.remainder(-value, unit_value)\n"
            if method == "floor":
                func_text += "    value = value - np.remainder(value, unit_value)\n"
            if method == "round":
                # Unit value is always even except value = 1
                func_text += "    if unit_value == 1:\n"
                func_text += "        value = value\n"
                func_text += "    else:\n"
                func_text += (
                    "        quotient, remainder = np.divmod(value, unit_value)\n"
                )
                func_text += "        mask = np.logical_or(remainder > (unit_value // 2), np.logical_and(remainder == (unit_value // 2), quotient % 2))\n"
                func_text += "        if mask:\n"
                func_text += "            quotient = quotient + 1\n"
                func_text += "        value = quotient * unit_value\n"
            if tz_literal is not None:
                if has_transitions:
                    func_text += "    original_value = value\n"
                    func_text += "    start_trans = deltas[np.searchsorted(trans, original_value, side='right') - 1]\n"
                    # Restore the delta which may have changed
                    func_text += "    value = value - start_trans\n"
                    func_text += "    end_trans = deltas[np.searchsorted(trans, value, side='right') - 1]\n"
                    # There are rare cases where start_trans is not the actual correct delta. This
                    # occurs because deltas[np.searchsorted(trans, original_value, side='right') - 1]
                    # has its values set by the UTC values, which is the result we are calculating.
                    #
                    # If we are wrong then the proposed UTC time will actually use a different offset
                    # than the starting time. We account for this by adjusting the transition time
                    # accordingly.
                    #
                    # For example this timestamp calculate would be wrong
                    # In [49]: pd.Timestamp("2022-11-06 03:00:00").value
                    # Out[49]: 1667703600000000000 -- this is the original_value
                    # In [50]: value = pd.Timestamp("2022-11-06 03:00:00").value
                    # In [51]: deltas[np.searchsorted(trans, value, side='right') - 1]
                    # Out[51]: -14400000000000 -- This is start_trans
                    # In [52]: pd.Timestamp("2022-11-06 03:00:00", tz="US/Eastern").value
                    # Out[52]: 1667721600000000000 -- This should be the final value.
                    # In [53]: value = pd.Timestamp("2022-11-06 03:00:00", tz="US/Eastern").value
                    # In [54]: deltas[np.searchsorted(trans, value, side='right') - 1]
                    # Out[54]: -18000000000000 -- This is the actual value/end_trans
                    func_text += "    offset = start_trans - end_trans\n"
                    func_text += "    value = value + offset\n"
                else:
                    # Restore the delta which hasn't changed.
                    func_text += "    value = value - delta\n"
            func_text += "    return pd.Timestamp(value, tz=tz_literal)\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "np": np,
                "pd": pd,
                "deltas": deltas,
                "trans": trans,
                "tz_literal": tz_literal,
            },
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
        None,
    )


def overload_sub_operator_timestamp(lhs, rhs):
    if isinstance(lhs, PandasTimestampType) and rhs == datetime_timedelta_type:
        tz_literal = lhs.tz

        def impl(lhs, rhs):  # pragma: no cover
            # Compute total nanoseconds to allow the timestamp constructor
            rhs_nanoseconds = bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(rhs)
            return pd.Timestamp(lhs.value - rhs_nanoseconds, tz=tz_literal)

        return impl

    if isinstance(lhs, PandasTimestampType) and isinstance(rhs, PandasTimestampType):
        if lhs.tz == rhs.tz:

            def impl_timestamp(lhs, rhs):  # pragma: no cover
                return convert_numpy_timedelta64_to_pd_timedelta(lhs.value - rhs.value)

        else:

            def impl_timestamp(lhs, rhs):  # pragma: no cover
                return convert_numpy_timedelta64_to_pd_timedelta(
                    lhs.tz_convert(None).value - rhs.tz_convert(None).value
                )

        return impl_timestamp

    if isinstance(lhs, PandasTimestampType) and rhs == pd_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + -rhs

        return impl


def to_nanoseconds(td):
    pass


@overload(to_nanoseconds, jit_options={"cache": True})
def to_nanoseconds_impl(td):
    if td == datetime_timedelta_type:

        def impl(td):  # pragma: no cover
            return bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(td)

        return impl
    elif isinstance(td, numba.types.NPTimedelta):

        def impl(td):  # pragma: no cover
            return td.value

        return impl


def overload_add_operator_timestamp(lhs, rhs):
    if isinstance(lhs, PandasTimestampType) and (
        rhs == datetime_timedelta_type or isinstance(rhs, numba.types.NPTimedelta)
    ):
        tz_literal = lhs.tz

        def impl(lhs, rhs):  # pragma: no cover
            # Compute total nanoseconds to allow the timestamp constructor
            rhs_nanoseconds = to_nanoseconds(rhs)
            return pd.Timestamp(lhs.value + rhs_nanoseconds, tz=tz_literal)

        return impl

    if isinstance(lhs, PandasTimestampType) and rhs == pd_timedelta_type:
        tz_literal = lhs.tz

        def impl(lhs, rhs):  # pragma: no cover
            # Sum the values and run the Timestamp function.
            return pd.Timestamp(lhs.value + rhs.value, tz=tz_literal)

        return impl

    # if lhs and rhs flipped, flip args and call add again
    if (lhs == pd_timedelta_type and isinstance(rhs, PandasTimestampType)) or (
        lhs == datetime_timedelta_type and isinstance(rhs, PandasTimestampType)
    ):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

        return impl


@overload(min, no_unliteral=True, jit_options={"cache": True})
def timestamp_min(lhs, rhs):
    if isinstance(lhs, PandasTimestampType) and isinstance(rhs, PandasTimestampType):
        if lhs.tz == rhs.tz:

            def impl(lhs, rhs):  # prama: no cover
                return lhs if lhs.value < rhs.value else rhs

        elif lhs.tz is not None and rhs.tz is not None:
            raise BodoError(
                "Cannot use min/max on timestamps with different timezones. Use tz_convert"
            )

        else:
            raise BodoError("Cannot compare tz-naive and tz-aware timestamps")

        return impl

    elif (
        isinstance(lhs, IndexValueType)
        and isinstance(rhs, IndexValueType)
        and isinstance(lhs.val_typ, PandasTimestampType)
        and isinstance(rhs.val_typ, PandasTimestampType)
    ):

        def impl(lhs, rhs):  # pragma: no cover
            # Based off of https://github.com/numba/numba/blob/249c8ff3206928b486346443ec148508f8c25f8e/numba/cpython/builtins.py#L589
            #
            # If both values are NaT, compare by index. If one value is not Nan and the other is, return the non-NaT. Else return the normal
            if pd.isna(lhs.value) and pd.isna(rhs.value):
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            elif pd.isna(lhs.value):
                return rhs
            elif pd.isna(rhs.value):
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
def timestamp_max(lhs, rhs):
    if isinstance(lhs, PandasTimestampType) and isinstance(rhs, PandasTimestampType):
        if lhs.tz == rhs.tz:

            def impl(lhs, rhs):  # prama: no cover
                return lhs if lhs.value > rhs.value else rhs

        elif lhs.tz is not None and rhs.tz is not None:
            raise BodoError(
                "Cannot use min/max on timestamps with different timezones. Use tz_convert"
            )

        else:
            raise BodoError("Cannot compare tz-naive and tz-aware timestamps")

        return impl

    # Won't be covered until final nullable TS changes
    elif (
        isinstance(lhs, IndexValueType)
        and isinstance(rhs, IndexValueType)
        and isinstance(lhs.val_typ, PandasTimestampType)
        and isinstance(rhs.val_typ, PandasTimestampType)
    ):  # pragma: no cover

        def impl(lhs, rhs):  # pragma: no cover
            # Based off of https://github.com/numba/numba/blob/249c8ff3206928b486346443ec148508f8c25f8e/numba/cpython/builtins.py#L589
            #
            # If both values are NaT, compare by index. If one value is not Nan and the other is, return the non-NaT. Else return the normal
            if pd.isna(lhs.value) and pd.isna(rhs.value):
                if lhs.index < rhs.index:
                    return lhs
                else:
                    return rhs
            elif pd.isna(lhs.value):
                return rhs
            elif pd.isna(rhs.value):
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


@overload_method(DatetimeDateType, "strftime", jit_options={"cache": True})
@overload_method(PandasTimestampType, "strftime", jit_options={"cache": True})
def strftime(ts, format):
    if isinstance(ts, DatetimeDateType):
        cls_name = "datetime.date"
    else:
        cls_name = "pandas.Timestamp"
    if types.unliteral(format) != types.unicode_type:
        raise BodoError(f"{cls_name}.strftime(): 'strftime' argument must be a string")

    def impl(ts, format):  # pragma: no cover
        with numba.objmode(res="unicode_type"):
            res = ts.strftime(format)
        return res

    return impl


@overload_method(PandasTimestampType, "to_datetime64", jit_options={"cache": True})
def to_datetime64(ts):
    def impl(ts):
        return integer_to_dt64(ts.value)

    return impl


def now_impl(tz=None):  # pragma: no cover
    pass


@overload(now_impl, no_unilteral=True, jit_options={"cache": True})
def now_impl_overload(tz=None):
    """Internal call to support pd.Timestamp.now().
    Untyped pass replaces pd.Timestamp.now() with this call since class methods are
    not supported in Numba's typing
    """

    if is_overload_none(tz):
        tz_typ = PandasTimestampType(None)
    elif is_overload_constant_str(tz):
        tz_typ = PandasTimestampType(get_overload_const_str(tz))
    elif is_overload_constant_int(tz):
        tz_typ = PandasTimestampType(get_overload_const_int(tz))
    else:
        raise_bodo_error(
            "pandas.Timestamp.now(): tz argument must be a constant string or integer literal if provided"
        )

    def impl(tz=None):  # pragma: no cover
        with numba.objmode(d=tz_typ):
            d = pd.Timestamp.now(tz)
        return d

    return impl


@register_jitable
def now_impl_consistent(tz_value_or_none=None):  # pragma: no cover
    """Wrapper around now_impl that ensure the result
    is consistent on all ranks.
    """
    if bodo.get_rank() == 0:
        ts = now_impl(tz_value_or_none)
    else:
        # Give a dummy date for type stability
        ts = pd.Timestamp(0, tz=tz_value_or_none)
    return bodo.libs.distributed_api.bcast_scalar(ts)


# -- builtin operators for dt64 ----------------------------------------------
# TODO: move to Numba


class CompDT64(ConcreteTemplate):
    cases = [signature(types.boolean, types.NPDatetime("ns"), types.NPDatetime("ns"))]


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


@overload_method(types.NPDatetime, "__str__", jit_options={"cache": True})
def overload_datetime64_str(val):
    if val == bodo.types.datetime64ns:
        # for right now, just going to use isoformat. This will omit fractional values,
        # similar to how the current str(timestamp) implementation omits fractional values.
        # see BE-1407
        def impl(val):  # pragma: no cover
            return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                val
            ).isoformat("T")

        return impl


timestamp_unsupported_attrs = [
    "asm8",
    "components",
    "freqstr",
    "tz",
    "fold",
    "tzinfo",
    "freq",
]

timestamp_unsupported_methods = [
    "astimezone",
    "ctime",
    "dst",
    "isoweekday",
    "replace",
    "strptime",
    "time",
    "timestamp",
    "timetuple",
    "timetz",
    "to_julian_date",
    "to_numpy",
    "to_period",
    "to_pydatetime",
    "tzname",
    "utctimetuple",
]

# class method(s) handled in untyped pass
# Timestamp.combine
# Timestamp.fromisocalendar
# Timestamp.fromisoformat
# Timestamp.fromordinal
# Timestamp.fromtimestamp
# Timestamp.today()
# Timestamp.utcfromtimestamp()
# Timestamp.utcnow()
# Timestamp.max
# Timestamp.min
# Timestamp.resolution


def _install_pd_timestamp_unsupported():
    for attr_name in timestamp_unsupported_attrs:
        full_name = "pandas.Timestamp." + attr_name
        overload_unsupported_attribute(PandasTimestampType, attr_name, full_name)
    for fname in timestamp_unsupported_methods:
        full_name = "pandas.Timestamp." + fname
        overload_unsupported_method(PandasTimestampType, fname, full_name)


_install_pd_timestamp_unsupported()


@lower_builtin(
    numba.core.types.functions.NumberClass,
    pd_timestamp_tz_naive_type,
    types.StringLiteral,
)
def datetime64_constructor(context, builder, sig, args):
    def datetime64_constructor_impl(a, b):
        return integer_to_dt64(a.value)

    return context.compile_internal(builder, datetime64_constructor_impl, sig, args)


@overload(operator.setitem, no_unliteral=True)
def dt_timedelta_arr_setitem(A, ind, val):
    """Handle A[i] = pd.Timestamp(...) where A is a datetime64 array"""
    if (
        not isinstance(A, types.Array)
        or A.dtype != bodo.types.datetime64ns
        or not isinstance(ind, types.Integer)
        or not val == pd_timestamp_tz_naive_type
    ):
        return

    def impl(A, ind, val):  # pragma: no cover
        dt64 = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val.value)
        A[ind] = dt64

    return impl
