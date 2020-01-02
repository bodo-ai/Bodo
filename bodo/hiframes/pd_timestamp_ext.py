# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
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

import numpy as np
import ctypes
import inspect
import bodo.libs.str_ext
import bodo.utils.utils

from llvmlite import ir as lir

import pandas as pd

# TODO: make pandas optional, not import this file if no pandas
# pandas_present = True
# try:
#     import pandas as pd
# except ImportError:
#     pandas_present = False

import datetime
from bodo.libs import hdatetime_ext
import llvmlite.binding as ll

ll.add_symbol("parse_iso_8601_datetime", hdatetime_ext.parse_iso_8601_datetime)
ll.add_symbol(
    "convert_datetimestruct_to_datetime",
    hdatetime_ext.convert_datetimestruct_to_datetime,
)
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

# --------------------------------------------------------------


class PANDAS_DATETIMESTRUCT(ctypes.Structure):
    _fields_ = [
        ("year", ctypes.c_longlong),
        ("month", ctypes.c_int),
        ("day", ctypes.c_int),
        ("hour", ctypes.c_int),
        ("min", ctypes.c_int),
        ("sec", ctypes.c_int),
        ("us", ctypes.c_int),
        ("ps", ctypes.c_int),
        ("as", ctypes.c_int),
    ]


class PandasDtsType(types.Type):
    def __init__(self):
        super(PandasDtsType, self).__init__(name="PandasDtsType()")


pandas_dts_type = PandasDtsType()


@typeof_impl.register(PANDAS_DATETIMESTRUCT)
def typeof_pandas_dts(val, c):
    return pandas_dts_type


@register_model(PandasDtsType)
class PandasDtsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("year", types.int64),
            ("month", types.int32),
            ("day", types.int32),
            ("hour", types.int32),
            ("min", types.int32),
            ("sec", types.int32),
            ("us", types.int32),
            ("ps", types.int32),
            ("as", types.int32),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(PandasDtsType, "year", "year")
make_attribute_wrapper(PandasDtsType, "month", "month")
make_attribute_wrapper(PandasDtsType, "day", "day")


@type_callable(PANDAS_DATETIMESTRUCT)
def type_pandas_dts(context):
    def typer():
        return pandas_dts_type

    return typer


@lower_builtin(PANDAS_DATETIMESTRUCT)
def impl_ctor_pandas_dts(context, builder, sig, args):
    typ = sig.return_type
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    return ts._getvalue()


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


@type_callable(pd.Timestamp)
def type_timestamp(context):
    def typer(year, month, day, hour, minute, second, us, ns):
        # TODO: check types
        return pandas_timestamp_type

    return typer


@type_callable(pd.Timestamp)
def type_timestamp(context):
    def typer(data):
        if data == pandas_timestamp_type:
            return pandas_timestamp_type

    return typer


@type_callable(datetime.datetime)
def type_datetime_datetime(context):
    def typer(year, month, day):  # how to handle optional hour, minute, second, us, ns?
        # TODO: check types
        return pandas_timestamp_type

    return typer


@lower_builtin(
    pd.Timestamp,
    types.int64,
    types.int64,
    types.int64,
    types.int64,
    types.int64,
    types.int64,
    types.int64,
    types.int64,
)
def impl_ctor_timestamp(context, builder, sig, args):
    typ = sig.return_type
    year, month, day, hour, minute, second, us, ns = args
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    ts.year = year
    ts.month = month
    ts.day = day
    ts.hour = hour
    ts.minute = minute
    ts.second = second
    ts.microsecond = us
    ts.nanosecond = ns
    return ts._getvalue()


@lower_builtin(pd.Timestamp, pandas_timestamp_type)
def impl_ctor_ts_ts(context, builder, sig, args):
    typ = sig.return_type
    rhs = args[0]
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    rhsproxy = cgutils.create_struct_proxy(typ)(context, builder)
    rhsproxy._setvalue(rhs)
    cgutils.copy_struct(ts, rhsproxy)
    return ts._getvalue()


@overload(pd.Timestamp)
def overload_pd_timestamp(ts_input):
    if ts_input == bodo.string_type:

        def impl(ts_input):  # pragma: no cover
            dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(ts_input)
            idt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dt64)
            return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(idt64)

        return impl


#              , types.int64, types.int64, types.int64, types.int64, types.int64)
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
def dt64_to_integer(context, builder, fromty, toty, val):
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


# -----------------------------------------------------------


def myref(val):  # pragma: no cover
    pass


@type_callable(myref)
def type_myref(context):
    def typer(val):
        return types.voidptr

    return typer


# -----------------------------------------------------------


def integer_to_timedelta64(val):  # pragma: no cover
    return np.timedelta64(val)


@type_callable(integer_to_timedelta64)
def type_int_to_timedelta64(context):
    def typer(val):
        return types.NPTimedelta("ns")

    return typer


@lower_builtin(integer_to_timedelta64, types.int64)
def impl_int_to_timedelta64(context, builder, sig, args):
    return args[0]


# -----------------------------------------------------------


def integer_to_dt64(val):  # pragma: no cover
    return np.datetime64(val)


@type_callable(integer_to_dt64)
def type_int_to_dt64(context):
    def typer(val):
        return types.NPDatetime("ns")

    return typer


@lower_builtin(integer_to_dt64, types.int64)
@lower_builtin(integer_to_dt64, types.uint64)
@lower_builtin(integer_to_dt64, types.IntegerLiteral)
def impl_int_to_dt64(context, builder, sig, args):
    return args[0]


# -----------------------------------------------------------


def dt64_to_integer(val):  # pragma: no cover
    return int(val)


@type_callable(dt64_to_integer)
def type_dt64_to_int(context):
    def typer(val):
        return types.int64

    return typer


@lower_builtin(dt64_to_integer, types.NPDatetime("ns"))
def impl_dt64_to_int(context, builder, sig, args):
    return args[0]


# TODO: fix in Numba
@overload_method(types.NPDatetime, "__hash__")
def dt64_hash(val):
    return lambda val: hash(dt64_to_integer(val))


# -----------------------------------------------------------
def timedelta64_to_integer(val):  # pragma: no cover
    return int(val)


@type_callable(timedelta64_to_integer)
def type_dt64_to_int(context):
    def typer(val):
        return types.int64

    return typer


@lower_builtin(timedelta64_to_integer, types.NPTimedelta("ns"))
def impl_dt64_to_int(context, builder, sig, args):
    return args[0]


# -----------------------------------------------------------


@lower_builtin(myref, types.int32)
@lower_builtin(myref, types.int64)
@lower_builtin(myref, types.IntegerLiteral)
def impl_myref_int32(context, builder, sig, args):
    typ = types.voidptr
    val = args[0]
    assert isinstance(val, lir.instructions.LoadInstr)
    return builder.bitcast(val.operands[0], lir.IntType(8).as_pointer())


@lower_builtin(myref, PandasDtsType)
def impl_myref_pandas_dts_type(context, builder, sig, args):
    typ = types.voidptr
    val = args[0]
    assert isinstance(val, lir.instructions.LoadInstr)
    return builder.bitcast(val.operands[0], lir.IntType(8).as_pointer())


# tslib_so = inspect.getfile(pd._libs.tslib)
# tslib_cdll = ctypes.CDLL(tslib_so)
# func_parse_iso = tslib_cdll.parse_iso_8601_datetime
# func_parse_iso.restype = ctypes.c_int32
# func_parse_iso.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
# func_dts_to_dt = tslib_cdll.pandas_datetimestruct_to_datetime
# func_dts_to_dt.restype = ctypes.c_int64
# func_dts_to_dt.argtypes = [ctypes.c_int, ctypes.c_void_p]

sig = types.intp(
    types.voidptr,  # C str
    types.intp,  # len(str)
    types.voidptr,  # struct ptr
    types.voidptr,  # int ptr
    types.voidptr,  # int ptr
)
parse_iso_8601_datetime = types.ExternalFunction("parse_iso_8601_datetime", sig)
sig = types.intp(
    types.intp,  # fr magic number
    types.voidptr,  # struct ptr
    types.voidptr,  # out int ptr
)
convert_datetimestruct_to_datetime = types.ExternalFunction(
    "convert_datetimestruct_to_datetime", sig
)


iNaT = np.iinfo(np.int64).min


@numba.njit(locals={"arg1": numba.int32, "arg3": numba.int32, "arg4": numba.int32})
def parse_datetime_str(_str):  # pragma: no cover
    nat_strings = ("NaT", "nat", "NAT", "nan", "NaN", "NAN")
    if len(_str) == 0 or _str in nat_strings:
        return integer_to_dt64(iNaT)
    arg0 = bodo.libs.str_ext.unicode_to_char_ptr(_str)
    arg1 = len(_str)
    arg2 = PANDAS_DATETIMESTRUCT()
    arg3 = np.int32(13)
    arg4 = np.int32(13)
    arg2ref = myref(arg2)
    retval = parse_iso_8601_datetime(arg0, arg1, arg2ref, myref(arg3), myref(arg4))
    out = 0
    retval2 = convert_datetimestruct_to_datetime(10, arg2ref, myref(out))
    return integer_to_dt64(out)


#     retval = func_parse_iso(arg0, arg1, arg2ref, myref(arg3), myref(arg4))
#     # "10" is magic enum value for PANDAS_FR_ns (nanosecond date time unit)
# #        return func_dts_to_dt(10, arg2ref)
#     return integer_to_dt64(func_dts_to_dt(10, arg2ref))
