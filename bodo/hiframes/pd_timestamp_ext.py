# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import numpy as np
import pandas as pd
import numba
from numba import types
from numba.extending import (
    typeof_impl,
    type_callable,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    lower_builtin,
    box,
    unbox,
    lower_cast,
    lower_getattr,
    infer_getattr,
    overload_method,
    overload,
    intrinsic,
    overload_attribute,
)
from numba.targets.imputils import lower_constant
from numba import cgutils
from numba.targets.arrayobj import (
    make_array,
    store_item,
    basic_indexing,
)
from numba.typing.templates import (
    infer_getattr,
    AttributeTemplate,
    bound_function,
    signature,
    infer_global,
    AbstractTemplate,
    ConcreteTemplate,
)
import bodo.libs.str_ext
import bodo.utils.utils
from bodo.utils.typing import is_overload_constant_str

from llvmlite import ir as lir


import datetime
from bodo.libs import hdatetime_ext
import llvmlite.binding as ll


ll.add_symbol("extract_year_days", hdatetime_ext.extract_year_days)
ll.add_symbol("get_month_day", hdatetime_ext.get_month_day)

date_fields = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
]
timedelta_fields = ["days", "seconds", "microseconds", "nanoseconds"]


# sentinel type representing no first input to pd.Timestamp() constructor
# similar to _no_input object of Pandas in timestamps.pyx
# https://github.com/pandas-dev/pandas/blob/8806ed7120fed863b3cd7d3d5f377ec4c81739d0/pandas/_libs/tslibs/timestamps.pyx#L38
class NoInput:
    pass


_no_input = NoInput()


class NoInputType(types.Type):
    def __init__(self):
        super(NoInputType, self).__init__(name='NoInput')


register_model(NoInputType)(models.OpaqueModel)


@typeof_impl.register(NoInput)
def _typ_no_input(val, c):
  return NoInputType()


@lower_constant(NoInputType)
def constant_no_input(context, builder, ty, pyval):
    return context.get_dummy_value()


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


class MinMaxBaseDT64(numba.typing.builtins.MinMaxBase):
    def _unify_minmax(self, tys):
        for ty in tys:
            if not ty == types.NPDatetime("ns"):
                return
        return self.context.unify_types(*tys)


@infer_global(max)
class Max(MinMaxBaseDT64):
    pass


@infer_global(min)
class Min(MinMaxBaseDT64):
    pass


class PandasTimestampType(types.Type):
    def __init__(self):
        super(PandasTimestampType, self).__init__(name="PandasTimestampType()")


pandas_timestamp_type = PandasTimestampType()


@typeof_impl.register(pd.Timestamp)
def typeof_pd_timestamp(val, c):
    return pandas_timestamp_type


@typeof_impl.register(datetime.datetime)
def typeof_datetime_datetime(val, c):
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


@overload_method(PandasTimestampType, "date")
def overload_pd_timestamp_date(ptt):
    def pd_timestamp_date_impl(ptt):  # pragma: no cover
        return datetime.date(ptt.year, ptt.month, ptt.day)

    return pd_timestamp_date_impl


@overload_method(PandasTimestampType, "isoformat")
def overload_pd_timestamp_isoformat(ts_typ, sep=None):
    if sep is None:

        def timestamp_isoformat_impl(ts):  # pragma: no cover
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

    else:

        def timestamp_isoformat_impl(ts, sep):  # pragma: no cover
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


# TODO: support general string formatting
@numba.njit
def str_2d(a):  # pragma: no cover
    res = str(a)
    if len(res) == 1:
        return "0" + res
    return res


@overload(str)
def ts_str_overload(a):
    if a == pandas_timestamp_type:
        return lambda a: a.isoformat(" ")


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

    pd_timestamp = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    pd_timestamp.year = c.pyapi.long_as_longlong(year_obj)
    pd_timestamp.month = c.pyapi.long_as_longlong(month_obj)
    pd_timestamp.day = c.pyapi.long_as_longlong(day_obj)
    pd_timestamp.hour = c.pyapi.long_as_longlong(hour_obj)
    pd_timestamp.minute = c.pyapi.long_as_longlong(minute_obj)
    pd_timestamp.second = c.pyapi.long_as_longlong(second_obj)
    pd_timestamp.microsecond = c.pyapi.long_as_longlong(microsecond_obj)
    pd_timestamp.nanosecond = c.pyapi.long_as_longlong(nanosecond_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)
    c.pyapi.decref(nanosecond_obj)

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


@type_callable(datetime.datetime)
def type_datetime_datetime(context):
    def typer(year, month, day):  # how to handle optional hour, minute, second, us, ns?
        # TODO: check types
        return pandas_timestamp_type

    return typer


@intrinsic
def init_timestamp(typingctx, year, month, day, hour, minute, second, microsecond, nanosecond=None):
    """Create a PandasTimestampType with provided data values.
    """

    def codegen(context, builder, sig, args):
        year, month, day, hour, minute, second, us, ns = args
        ts = cgutils.create_struct_proxy(pandas_timestamp_type)(context, builder)
        ts.year = year
        ts.month = month
        ts.day = day
        ts.hour = hour
        ts.minute = minute
        ts.second = second
        ts.microsecond = us
        ts.nanosecond = ns
        return ts._getvalue()

    return pandas_timestamp_type(types.int64, types.int64, types.int64, types.int64,
        types.int64, types.int64, types.int64, types.int64), codegen



@numba.generated_jit
def zero_if_none(value):
    """return zero if value is None. Otherwise, return value
    """
    if value == types.none:
        return lambda value: 0
    return lambda value: value


@overload(pd.Timestamp)
def overload_pd_timestamp(ts_input=_no_input,
                freq=None, tz=None, unit=None,
                year=None, month=None, day=None,
                hour=None, minute=None, second=None, microsecond=None,
                nanosecond=None, tzinfo=None):

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
        def impl_kw(ts_input=_no_input,
                freq=None, tz=None, unit=None,
                year=None, month=None, day=None,
                hour=None, minute=None, second=None, microsecond=None,
                nanosecond=None, tzinfo=None):  # pragma: no cover
            return init_timestamp(year, month, day, zero_if_none(hour),
                zero_if_none(minute), zero_if_none(second), zero_if_none(microsecond),
                zero_if_none(nanosecond)
            )
        return impl_kw

    # User passed positional arguments:
    # Timestamp(year, month, day[, hour[, minute[, second[,
    # microsecond[, nanosecond[, tzinfo]]]]]])
    if isinstance(freq, types.Integer):
        def impl_pos(ts_input=_no_input,
                freq=None, tz=None, unit=None,
                year=None, month=None, day=None,
                hour=None, minute=None, second=None, microsecond=None,
                nanosecond=None, tzinfo=None):  # pragma: no cover
            return init_timestamp(ts_input, freq, tz, zero_if_none(unit),
                zero_if_none(year), zero_if_none(month), zero_if_none(day),
                zero_if_none(hour)
            )
        return impl_pos

    # parse string input
    if ts_input == bodo.string_type or is_overload_constant_str(ts_input):
        # just call Pandas in this case since the string parsing code is complex and
        # handles several possible cases
        types.pandas_timestamp_type = pandas_timestamp_type
        def impl_str(ts_input=_no_input,
                freq=None, tz=None, unit=None,
                year=None, month=None, day=None,
                hour=None, minute=None, second=None, microsecond=None,
                nanosecond=None, tzinfo=None):  # pragma: no cover
            with numba.objmode(res="pandas_timestamp_type"):
                res = pd.Timestamp(ts_input)
            return res

        return impl_str

    # for pd.Timestamp(), just return input
    if ts_input == pandas_timestamp_type:
        return lambda ts_input=_no_input, freq=None, tz=None, unit=None, \
                year=None, month=None, day=None, \
                hour=None, minute=None, second=None, microsecond=None, \
                nanosecond=None, tzinfo=None: ts_input


@lower_builtin(datetime.datetime, types.int64, types.int64, types.int64)
@lower_builtin(
    datetime.datetime, types.IntegerLiteral, types.IntegerLiteral, types.IntegerLiteral
)
def impl_ctor_datetime(context, builder, sig, args):
    typ = sig.return_type
    year, month, day = args
    # year, month, day, hour, minute, second, us, ns = args
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    ts.year = year
    ts.month = month
    ts.day = day
    ts.hour = lir.Constant(lir.IntType(64), 0)
    ts.minute = lir.Constant(lir.IntType(64), 0)
    ts.second = lir.Constant(lir.IntType(64), 0)
    ts.microsecond = lir.Constant(lir.IntType(64), 0)
    ts.nanosecond = lir.Constant(lir.IntType(64), 0)
    # ts.hour = hour
    # ts.minute = minute
    # ts.second = second
    # ts.microsecond = us
    # ts.nanosecond = ns
    return ts._getvalue()


@lower_cast(types.NPDatetime("ns"), types.int64)
def cast_dt64_to_integer(context, builder, fromty, toty, val):
    # dt64 is stored as int64 so just return value
    return val


@overload_attribute(PandasTimestampType, "value")
def overload_timestamp_value(t):
    def impl(t):  # pragma: no cover
        return convert_timestamp_to_datetime64(t)

    return impl


@numba.njit
def convert_timestamp_to_datetime64(ts):  # pragma: no cover
    year = ts.year - 1970
    days = year * 365
    if days >= 0:
        year += 1
        days += year // 4
        year += 68
        days -= year // 100
        year += 300
        days += year // 400
    else:
        year -= 2
        days += year // 4
        year -= 28
        days -= year // 100
        days += year // 400
    leapyear = (ts.year % 400 == 0) or (ts.year % 4 == 0 and ts.year % 100 != 0)
    month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leapyear:
        month_len[1] = 29

    for i in range(ts.month - 1):
        days += month_len[i]

    days += ts.day - 1

    return (
        (((days * 24 + ts.hour) * 60 + ts.minute) * 60 + ts.second) * 1000000
        + ts.microsecond
    ) * 1000 + ts.nanosecond


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
    """Converts number of days within a year to month and day, returned as a 2-tuple.
    """
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


@numba.njit
def convert_datetime64_to_timestamp(dt64):  # pragma: no cover
    """Converts dt64 value to pd.Timestamp
    """
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


@intrinsic
def integer_to_timedelta64(typingctx, val=None):
    """Cast an int value to timedelta64
    """
    def codegen(context, builder, sig, args):
        return args[0]

    return types.NPTimedelta("ns")(val), codegen


@intrinsic
def integer_to_dt64(typingctx, val=None):
    """Cast an int value to datetime64
    """
    def codegen(context, builder, sig, args):
        return args[0]

    return types.NPDatetime("ns")(val), codegen


@intrinsic
def dt64_to_integer(typingctx, val=None):
    """Cast a datetime64 value to integer
    """
    def codegen(context, builder, sig, args):
        return args[0]

    return types.int64(val), codegen


# TODO: fix in Numba
@overload_method(types.NPDatetime, "__hash__")
def dt64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


@intrinsic
def timedelta64_to_integer(typingctx, val=None):
    """Cast a timedelta64 value to integer
    """
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
