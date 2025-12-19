"""Numba extension support for datetime.date objects and their arrays."""

from __future__ import annotations

import datetime
import operator
import warnings

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pytz
from llvmlite import ir as lir
from numba.core import cgutils, types, typing
from numba.core.imputils import lower_builtin, lower_constant
from numba.core.typing.templates import AttributeTemplate, infer_getattr
from numba.extending import (
    NativeValue,
    box,
    infer_getattr,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.datetime_datetime_ext import DatetimeDatetimeType
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_type
from bodo.libs import hdatetime_ext
from bodo.utils.indexing import (
    array_getitem_bool_index,
    array_getitem_int_index,
    array_getitem_slice_index,
    array_setitem_bool_index,
    array_setitem_int_index,
    array_setitem_slice_index,
)
from bodo.utils.typing import (
    BodoError,
    is_iterable_type,
    is_list_like_index_type,
    is_overload_int,
    is_overload_none,
)

ll.add_symbol("get_isocalendar", hdatetime_ext.get_isocalendar)
ll.add_symbol("get_days_from_date", hdatetime_ext.get_days_from_date)


# datetime.date implementation that uses a single int to store year/month/day
# Does not need refcounted object wrapping since it is immutable
#
# We represent dates as the Arrow date32 type, which is the difference in days from the UNIX epoch.
class DatetimeDateType(types.Type):
    def __init__(self):
        super().__init__(name="DatetimeDateType()")
        self.bitwidth = 32  # needed for using IntegerModel


datetime_date_type = DatetimeDateType()


@typeof_impl.register(datetime.date)
def typeof_datetime_date(val, c):
    return datetime_date_type


register_model(DatetimeDateType)(models.IntegerModel)


# extraction of year/month/day attributes
@infer_getattr
class DatetimeAttribute(AttributeTemplate):
    key = DatetimeDateType

    def resolve_year(self, typ):
        return types.int64

    def resolve_month(self, typ):
        return types.int64

    def resolve_day(self, typ):
        return types.int64


# '_ymd' is a Bodo-specific attribute for convenience.
# It simply returns a tuple: (year, month, day)
# corresponding to this Date object. This is useful
# for cases such as comparison operators where we
# need all 3 values.
@overload_attribute(DatetimeDateType, "_ymd", jit_options={"cache": True})
def overload_datetime_date_get_ymd(a):
    def get_ymd(a):  # pragma: no cover
        return _ord2ymd(cast_datetime_date_to_int(a) + UNIX_EPOCH_ORD)

    return get_ymd


@overload_attribute(DatetimeDateType, "year", jit_options={"cache": True})
def overload_datetime_date_get_year(a):
    def get_year(a):  # pragma: no cover
        year, _, _ = _ord2ymd(cast_datetime_date_to_int(a) + UNIX_EPOCH_ORD)
        return year

    return get_year


@overload_attribute(DatetimeDateType, "month", jit_options={"cache": True})
def overload_datetime_date_get_month(a):
    def get_month(a):  # pragma: no cover
        _, month, _ = _ord2ymd(cast_datetime_date_to_int(a) + UNIX_EPOCH_ORD)
        return month

    return get_month


@overload_attribute(DatetimeDateType, "day", jit_options={"cache": True})
def overload_datetime_date_get_day(a):
    def get_day(a):  # pragma: no cover
        _, _, day = _ord2ymd(cast_datetime_date_to_int(a) + UNIX_EPOCH_ORD)
        return day

    return get_day


@unbox(DatetimeDateType)
def unbox_datetime_date(typ, val, c):
    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")

    yll = c.pyapi.long_as_longlong(year_obj)
    mll = c.pyapi.long_as_longlong(month_obj)
    dll = c.pyapi.long_as_longlong(day_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)

    fnty = lir.FunctionType(
        lir.IntType(64), [lir.IntType(64), lir.IntType(64), lir.IntType(64)]
    )
    fn_get_days_from_date = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="get_days_from_date"
    )
    nopython_date = c.builder.call(fn_get_days_from_date, [yll, mll, dll])

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    return NativeValue(
        c.builder.trunc(nopython_date, lir.IntType(32)), is_error=is_error
    )


@lower_constant(DatetimeDateType)
def lower_constant_datetime_date(context, builder, ty, pyval):
    days = (pyval - datetime.date(1970, 1, 1)).days

    return context.get_constant(types.int32, days)


@box(DatetimeDateType)
def box_datetime_date(typ, val, c):
    ord_obj = c.pyapi.long_from_longlong(
        c.builder.add(
            c.builder.sext(val, lir.IntType(64)),
            lir.Constant(lir.IntType(64), UNIX_EPOCH_ORD),
        )
    )

    dt_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.date.fromordinal))
    res = c.pyapi.call_function_objargs(dt_obj, (ord_obj,))
    c.pyapi.decref(ord_obj)
    c.pyapi.decref(dt_obj)
    return res


@type_callable(datetime.date)
def type_datetime_date(context):
    def typer(year, month, day):
        # TODO: check types
        return datetime_date_type

    return typer


@lower_builtin(
    datetime.date, types.IntegerLiteral, types.IntegerLiteral, types.IntegerLiteral
)
@lower_builtin(datetime.date, types.int64, types.int64, types.int64)
def impl_ctor_datetime_date(context, builder, sig, args):
    def build_date(year, month, day):  # pragma: no cover
        days_since_1_1_0001 = _ymd2ord(year, month, day)
        return days_since_1_1_0001 - UNIX_EPOCH_ORD

    sig = typing.signature(types.int64, types.int64, types.int64, types.int64)
    o = context.compile_internal(builder, build_date, sig, args)
    return builder.trunc(o, lir.IntType(32))


@intrinsic
def cast_int_to_datetime_date(typingctx, val):
    """Cast int value to datetime.date"""
    assert val == types.int32

    def codegen(context, builder, signature, args):
        return args[0]

    return datetime_date_type(types.int32), codegen


@intrinsic
def cast_datetime_date_to_int(typingctx, val):
    """Cast datetime.date value to int"""
    assert val == datetime_date_type

    def codegen(context, builder, signature, args):
        return args[0]

    return types.int32(datetime_date_type), codegen


DATE_TO_NS = 24 * 60 * 60 * 10**9


@register_jitable
def cast_datetime_date_to_int_ns(dt):
    """
    Cast datetime.date to nanoseconds (int)
    """
    return cast_datetime_date_to_int(dt) * DATE_TO_NS


###############################################################################
"""
Following codes are copied from
https://github.com/python/cpython/blob/39a5c889d30d03a88102e56f03ee0c95db198fb3/Lib/datetime.py
"""

_MAXORDINAL = 3652059

# -1 is a placeholder for indexing purposes.
_DAYS_IN_MONTH = np.array(
    [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64
)

_DAYS_BEFORE_MONTH = np.array(
    [-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64
)


@register_jitable(cache=True)
def _is_leap(year):  # pragma: no cover
    "year -> 1 if leap year, else 0."
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@register_jitable(cache=True)
def _days_before_year(year):  # pragma: no cover
    "year -> number of days before January 1st of year."
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


@register_jitable(cache=True)
def _days_in_month(year, month):  # pragma: no cover
    "year, month -> number of days in that month in that year."
    if month == 2 and _is_leap(year):
        return 29
    return _DAYS_IN_MONTH[month]


@register_jitable(cache=True)
def _days_before_month(year, month):  # pragma: no cover
    "year, month -> number of days in year preceding first day of month."
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))


@register_jitable(cache=True)
def _day_of_year(year, month, day):  # pragma: no cover
    "year, month, day -> how many days into the year is it"
    days = day
    for m in range(1, month):
        days += _days_in_month(year, m)
    return days


_DI400Y = _days_before_year(401)  # number of days in 400 years
_DI100Y = _days_before_year(101)  #    "    "   "   " 100   "
_DI4Y = _days_before_year(5)  #    "    "   "   "   4   "


@register_jitable(cache=True)
def _ymd2ord(year, month, day):  # pragma: no cover
    "year, month, day -> ordinal, considering 01-Jan-0001 as day 1."
    # If we pass in 2/29 on a non-leap-year, decrement to 2/28
    if month == 2 and (not _is_leap(year)) and day == 29:
        day -= 1
    return _days_before_year(year) + _days_before_month(year, month) + day


@register_jitable(cache=True)
def _ord2ymd(n):  # pragma: no cover
    "ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."

    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    #
    #     D  M   Y            n              n-1
    #     -- --- ----        ----------     ----------------
    #     31 Dec -400        -_DI400Y       -_DI400Y -1
    #      1 Jan -399         -_DI400Y +1   -_DI400Y      400-year boundary
    #     ...
    #     30 Dec  000        -1             -2
    #     31 Dec  000         0             -1
    #      1 Jan  001         1              0            400-year boundary
    #      2 Jan  001         2              1
    #      3 Jan  001         3              2
    #     ...
    #     31 Dec  400         _DI400Y        _DI400Y -1
    #      1 Jan  401         _DI400Y +1     _DI400Y      400-year boundary
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year = n400 * 400 + 1  # ..., -399, 1, 401, ...

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100, n = divmod(n, _DI100Y)

    # Now compute how many 4-year cycles precede it.
    n4, n = divmod(n, _DI4Y)

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1, n = divmod(n, 365)

    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return year - 1, 12, 31

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    leapyear = n1 == 3 and (n4 != 24 or n100 == 3)
    month = (n + 50) >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:  # estimate is too large
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding

    # Now the year and month are correct, and n is the offset from the
    # start of that month:  we're done!
    return year, month, n + 1


@register_jitable(cache=True)
def _cmp(x, y):  # pragma: no cover
    return 0 if x == y else 1 if x > y else -1


@intrinsic
def get_isocalendar(typingctx, dt_year, dt_month, dt_day):
    def codegen(context, builder, sig, args):
        year = cgutils.alloca_once(builder, lir.IntType(64))
        week = cgutils.alloca_once(builder, lir.IntType(64))
        dow = cgutils.alloca_once(builder, lir.IntType(64))
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_isocalendar"
        )
        builder.call(fn_tp, [args[0], args[1], args[2], year, week, dow])
        return cgutils.pack_array(
            builder, [builder.load(year), builder.load(week), builder.load(dow)]
        )

    res = (
        types.Tuple([types.int64, types.int64, types.int64])(
            types.int64, types.int64, types.int64
        ),
        codegen,
    )
    return res


###############################################################################

types.datetime_date_type = datetime_date_type


@register_jitable(cache=True)
def today_impl():  # pragma: no cover
    """Internal call to support datetime.date.today().
    Untyped pass replaces datetime.date.today() with this call since class methods are
    not supported in Numba's typing
    """
    with numba.objmode(d="datetime_date_type"):
        d = datetime.date.today()
    return d


@register_jitable(cache=True)
def today_rank_consistent():  # pragma: no cover
    """Internal wrapper around today_impl that is used to ensure all
    ranks return the same value.
    """
    if bodo.get_rank() == 0:
        d = today_impl()
    else:
        # Give a dummy date for type stability
        d = datetime.date(2023, 1, 1)
    return bodo.libs.distributed_api.bcast_scalar(d)


@register_jitable(cache=True)
def fromordinal_impl(n):  # pragma: no cover
    """Internal call to support datetime.date.fromordinal().
    Untyped pass replaces datetime.date.fromordinal() with this call since class methods are
    not supported in Numba's typing
    """
    y, m, d = _ord2ymd(n)
    return datetime.date(y, m, d)


# TODO: support general string formatting
@numba.njit(cache=True)
def str_2d(a):  # pragma: no cover
    """Takes in a number representing an date/time unit and formats it as a
    2 character string, adding a leading zero if necessary."""
    res = str(a)
    if len(res) == 1:
        return "0" + res
    return res


@overload_method(DatetimeDateType, "__str__", jit_options={"cache": True})
def overload_date_str(val):
    def impl(val):  # pragma: no cover
        year, month, day = val._ymd
        return str(year) + "-" + str_2d(month) + "-" + str_2d(day)

    return impl


@overload_method(DatetimeDateType, "replace", jit_options={"cache": True})
def replace_overload(date, year=None, month=None, day=None):
    if not is_overload_none(year) and not is_overload_int(year):
        raise BodoError("date.replace(): year must be an integer")
    elif not is_overload_none(month) and not is_overload_int(month):
        raise BodoError("date.replace(): month must be an integer")
    elif not is_overload_none(day) and not is_overload_int(day):
        raise BodoError("date.replace(): day must be an integer")

    def impl(date, year=None, month=None, day=None):  # pragma: no cover
        year_, month_, day_ = date._ymd
        year_val = year_ if year is None else year
        month_val = month_ if month is None else month
        day_val = day_ if day is None else day
        return datetime.date(year_val, month_val, day_val)

    return impl


@overload_method(
    DatetimeDatetimeType, "toordinal", no_unliteral=True, jit_options={"cache": True}
)
def toordinal(dt):
    """Return proleptic Gregorian ordinal for the year, month and day.
    January 1 of year 1 is day 1.  Only the year, month and day values
    contribute to the result.
    """

    def impl(dt):  # pragma: no cover
        return _ymd2ord(dt.year, dt.month, dt.day)

    return impl


@overload_method(
    DatetimeDateType, "toordinal", no_unliteral=True, jit_options={"cache": True}
)
def toordinal(date):
    """Return proleptic Gregorian ordinal for the year, month and day.
    January 1 of year 1 is day 1.  Only the year, month and day values
    contribute to the result.
    """

    def impl(date):  # pragma: no cover
        return cast_datetime_date_to_int(date) + UNIX_EPOCH_ORD

    return impl


@overload_method(
    DatetimeDatetimeType, "weekday", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    DatetimeDateType, "weekday", no_unliteral=True, jit_options={"cache": True}
)
def weekday(date):
    "Return day of the week, where Monday == 0 ... Sunday == 6."

    def impl(date):  # pragma: no cover
        return (date.toordinal() + 6) % 7

    return impl


@overload_method(
    DatetimeDateType, "isocalendar", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_timestamp_isocalendar(date):
    def impl(date):  # pragma: no cover
        year, month, day = date._ymd
        year, week, day_of_week = get_isocalendar(year, month, day)
        return (year, week, day_of_week)

    return impl


def overload_add_operator_datetime_date(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            o = lhs.toordinal() + rhs.days
            if 0 < o <= _MAXORDINAL:
                return fromordinal_impl(o)
            raise OverflowError("result out of range")

        return impl

    elif lhs == datetime_timedelta_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            o = lhs.days + rhs.toordinal()
            if 0 < o <= _MAXORDINAL:
                return fromordinal_impl(o)
            raise OverflowError("result out of range")

        return impl


def overload_sub_operator_datetime_date(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs + datetime.timedelta(-rhs.days)

        return impl

    elif lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            days1 = lhs.toordinal()
            days2 = rhs.toordinal()
            return datetime.timedelta(days1 - days2)

        return impl

    # datetime_date_array - timedelta
    if lhs == datetime_date_array_type and rhs == datetime_timedelta_type:

        def impl(lhs, rhs):  # pragma: no cover
            in_arr = lhs
            numba.parfors.parfor.init_prange()
            n = len(in_arr)
            A = alloc_datetime_date_array(n)
            for i in numba.parfors.parfor.internal_prange(n):
                A[i] = in_arr[i] - rhs
            return A

        return impl


@overload(min, no_unliteral=True, jit_options={"cache": True})
def date_min(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs < rhs else rhs

        return impl

    # Support min of IndexValue with date because Numba 0.56 IndexValueType implementation uses np.isnan which isn't
    # implemented for dates.
    if (
        isinstance(lhs, numba.core.typing.builtins.IndexValueType)
        and lhs.val_typ == datetime_date_type
        and isinstance(rhs, numba.core.typing.builtins.IndexValueType)
        and rhs.val_typ == datetime_date_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            if lhs.value == rhs.value:
                # If the values are equal return the smaller index
                return lhs if lhs.index < rhs.index else rhs
            return lhs if lhs.value < rhs.value else rhs

        return impl


@overload(max, no_unliteral=True, jit_options={"cache": True})
def date_max(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs > rhs else rhs

        return impl

    # Support max of IndexValue with date because np.isnan isn't
    # implemented for dates.
    if (
        isinstance(lhs, numba.core.typing.builtins.IndexValueType)
        and lhs.val_typ == datetime_date_type
        and isinstance(rhs, numba.core.typing.builtins.IndexValueType)
        and rhs.val_typ == datetime_date_type
    ):

        def impl(lhs, rhs):  # pragma: no cover
            if lhs.value == rhs.value:
                # If the values are equal return the smaller index
                return lhs if lhs.index < rhs.index else rhs
            return lhs if lhs.value > rhs.value else rhs

        return impl


@overload_method(
    DatetimeDateType, "__hash__", no_unliteral=True, jit_options={"cache": True}
)
def __hash__(td):
    """Hashcode for datetime.date types. Copies the CPython implementation"""

    def impl(td):  # pragma: no cover
        y, m, d = _ord2ymd(cast_datetime_date_to_int(td))
        yhi = (np.uint8)(y // 256)
        ylo = (np.uint8)(y % 256)
        month = (np.uint8)(m)
        day = (np.uint8)(d)
        state = (yhi, ylo, month, day)
        return hash(state)

    return impl


@overload(bool, inline="always", no_unliteral=True, jit_options={"cache": True})
def date_to_bool(date):
    """All dates evaluate to True"""
    if date != datetime_date_type:  # pragma: no cover
        return

    def impl(date):  # pragma: no cover
        return True

    return impl


# Python 3.9 uses a namedtuple-like calss for isocalendar output instead of tuple
# IsoCalendarDate class is hidden from import, so use a value to retrieve it
IsoCalendarDate = datetime.date(2011, 1, 1).isocalendar().__class__

# TODO: [BE-251] support full functionality


class IsoCalendarDateType(types.Type):
    def __init__(self):
        super().__init__(name="IsoCalendarDateType()")


iso_calendar_date_type = DatetimeDateType()


@typeof_impl.register(IsoCalendarDate)
def typeof_datetime_date(val, c):
    return iso_calendar_date_type


##################### Array of datetime.date objects ##########################


class DatetimeDateArrayType(types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="DatetimeDateArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return datetime_date_type

    def copy(self):
        return DatetimeDateArrayType()


datetime_date_array_type = DatetimeDateArrayType()
types.datetime_date_array_type = datetime_date_array_type

data_type = types.Array(types.int32, 1, "C")
nulls_type = types.Array(types.uint8, 1, "C")


# datetime.date array has only an array integers to store data
@register_model(DatetimeDateArrayType)
class DatetimeDateArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", data_type),
            ("null_bitmap", nulls_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DatetimeDateArrayType, "data", "_data")
make_attribute_wrapper(DatetimeDateArrayType, "null_bitmap", "_null_bitmap")


@overload_method(
    DatetimeDateArrayType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_datetime_date_arr_copy(A):
    return lambda A: bodo.hiframes.datetime_date_ext.init_datetime_date_array(
        A._data.copy(),
        A._null_bitmap.copy(),
    )  # pragma: no cover


@overload_attribute(DatetimeDateArrayType, "dtype", jit_options={"cache": True})
def overload_datetime_date_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@unbox(DatetimeDateArrayType)
def unbox_datetime_date_array(typ, val, c):
    return bodo.libs.array.unbox_array_using_arrow(typ, val, c)


def int_to_datetime_date_python(ia):
    return datetime.date(ia >> 32, (ia >> 16) & 0xFFFF, ia & 0xFFFF)


def int_array_to_datetime_date(ia):
    # setting 'otypes' is necessary since input can be empty
    return np.vectorize(int_to_datetime_date_python, otypes=[object])(ia)


@box(DatetimeDateArrayType)
def box_datetime_date_array(typ, val, c):
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@intrinsic
def init_datetime_date_array(typingctx, data, nulls):
    """Create a DatetimeDateArrayType with provided data values."""
    assert data == types.Array(types.int32, 1, "C") or data == types.Array(
        types.NPDatetime("ns"), 1, "C"
    )
    assert nulls == types.Array(types.uint8, 1, "C")

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

    sig = datetime_date_array_type(data, nulls)
    return sig, codegen


@lower_constant(DatetimeDateArrayType)
def lower_constant_datetime_date_arr(context, builder, typ, pyval):
    n = len(pyval)
    data_arr = np.zeros(n, np.int32)
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)

    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            data_arr[i] = _ymd2ord(s.year, s.month, s.day) - UNIX_EPOCH_ORD

    data_const_arr = context.get_constant_generic(builder, data_type, data_arr)
    nulls_const_arr = context.get_constant_generic(builder, nulls_type, nulls_arr)

    # create datetime arr struct
    return lir.Constant.literal_struct([data_const_arr, nulls_const_arr])


@numba.njit(cache=True, no_cpython_wrapper=True)
def alloc_datetime_date_array(n):  # pragma: no cover
    data_arr = np.empty(n, dtype=np.int32)
    # XXX: set all bits to not null since datetime.date array operations do not support
    # NA yet. TODO: use 'empty' when all operations support NA
    # nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    nulls = np.full((n + 7) >> 3, 255, np.uint8)
    return init_datetime_date_array(data_arr, nulls)


def alloc_datetime_date_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_datetime_date_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_datetime_date_ext_alloc_datetime_date_array = alloc_datetime_date_array_equiv


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
def dt_date_arr_getitem(A, ind):
    if A != datetime_date_array_type:
        return

    if isinstance(types.unliteral(ind), types.Integer):
        return lambda A, ind: cast_int_to_datetime_date(A._data[ind])

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_bool_index(A, ind)
            return init_datetime_date_array(new_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_int_index(A, ind)
            return init_datetime_date_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_slice_index(A, ind)
            return init_datetime_date_array(new_data, new_mask)

        return impl_slice

    # This should be the only DatetimeDateArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for DatetimeDateArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True, jit_options={"cache": True})
def dt_date_arr_setitem(A, idx, val):
    if A != datetime_date_array_type:
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for DatetimeDateArray with indexing type {idx} received an incorrect 'value' type {val}."

    # scalar case
    if isinstance(idx, types.Integer):
        if types.unliteral(val) == datetime_date_type:

            def impl(A, idx, val):  # pragma: no cover
                A._data[idx] = cast_datetime_date_to_int(val)
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            # Covered by test_series_iat_setitem , test_series_iloc_setitem_int , test_series_setitem_int
            return impl

        else:
            raise BodoError(typ_err_msg)

    if not (
        (is_iterable_type(val) and val.dtype == bodo.types.datetime_date_type)
        or types.unliteral(val) == datetime_date_type
    ):
        raise BodoError(typ_err_msg)

    # array of integers
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):
        if types.unliteral(val) == datetime_date_type:
            return lambda A, idx, val: array_setitem_int_index(
                A, idx, cast_datetime_date_to_int(val)
            )  # pragma: no cover

        def impl_arr_ind(A, idx, val):  # pragma: no cover
            array_setitem_int_index(A, idx, val)

        # covered by test_series_iloc_setitem_list_int
        return impl_arr_ind

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        if types.unliteral(val) == datetime_date_type:
            return lambda A, idx, val: array_setitem_bool_index(
                A, idx, cast_datetime_date_to_int(val)
            )  # pragma: no cover

        def impl_bool_ind_mask(A, idx, val):  # pragma: no cover
            array_setitem_bool_index(A, idx, val)

        return impl_bool_ind_mask

    # slice case
    if isinstance(idx, types.SliceType):
        if types.unliteral(val) == datetime_date_type:
            return lambda A, idx, val: array_setitem_slice_index(
                A, idx, cast_datetime_date_to_int(val)
            )  # pragma: no cover

        def impl_slice_mask(A, idx, val):  # pragma: no cover
            array_setitem_slice_index(A, idx, val)

        # covered by test_series_setitem_slice
        return impl_slice_mask

    # This should be the only DatetimeDateArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for DatetimeDateArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_len_datetime_date_arr(A):
    if A == datetime_date_array_type:
        return lambda A: len(A._data)


@overload_attribute(DatetimeDateArrayType, "shape", jit_options={"cache": True})
def overload_datetime_date_arr_shape(A):
    return lambda A: (len(A._data),)


@overload_attribute(DatetimeDateArrayType, "nbytes", jit_options={"cache": True})
def datetime_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


def create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_date_type."""

    def overload_date_cmp(lhs, rhs):
        # datetime.date and datetime.date
        if lhs == datetime_date_type and rhs == datetime_date_type:

            def impl(lhs, rhs):  # pragma: no cover
                ord, ord2 = (
                    cast_datetime_date_to_int(lhs),
                    cast_datetime_date_to_int(rhs),
                )
                return op(ord, ord2)

            return impl

        # datetime.date and datetime64
        if lhs == datetime_date_type and rhs == bodo.types.datetime64ns:
            # Convert both to integers (ns) for comparison.
            return lambda lhs, rhs: op(
                cast_datetime_date_to_int_ns(lhs),
                bodo.hiframes.pd_timestamp_ext.dt64_to_integer(rhs),
            )  # pragma: no cover

        # datetime64 and datetime.date
        if rhs == datetime_date_type and lhs == bodo.types.datetime64ns:
            # Convert both to integers (ns) for comparison.
            return lambda lhs, rhs: op(
                bodo.hiframes.pd_timestamp_ext.dt64_to_integer(lhs),
                cast_datetime_date_to_int_ns(rhs),
            )  # pragma: no cover

    return overload_date_cmp


def create_datetime_date_cmp_op_overload(op):
    """create overload function for supported comparison operators
    between datetime_date_type and datetime_datetime_type."""

    def overload_cmp(lhs, rhs):
        # equality between datetime and date doesn't look at the values.
        # We raise a warning because this may be a bug.
        datetime_warning = f"{lhs} {numba.core.utils.OPERATORS_TO_BUILTINS[op]} {rhs} is always {op == operator.ne} in Python. If this is unexpected there may be a bug in your code."
        warnings.warn(datetime_warning, bodo.BodoWarning)
        if op == operator.eq:
            return lambda lhs, rhs: False  # pragma: no cover
        elif op == operator.ne:
            return lambda lhs, rhs: True  # pragma: no cover

    return overload_cmp


def create_datetime_array_date_cmp_op_overload(op):
    """create overload function for comparison operators with datetime64ns array
    and date types."""

    def overload_arr_cmp(lhs, rhs):
        if isinstance(lhs, types.Array) and lhs.dtype == bodo.types.datetime64ns:
            # datetime64 + date scalar
            if rhs == datetime_date_type:

                def impl(lhs, rhs):  # pragma: no cover
                    numba.parfors.parfor.init_prange()
                    n = len(lhs)
                    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(lhs, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(
                                lhs[i],
                                bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                                    pd.Timestamp(rhs)
                                ),
                            )
                    return out_arr

                return impl

            # datetime64 + date array
            elif rhs == datetime_date_array_type:

                def impl(lhs, rhs):  # pragma: no cover
                    numba.parfors.parfor.init_prange()
                    n = len(lhs)
                    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(
                            lhs, i
                        ) or bodo.libs.array_kernels.isna(rhs, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(
                                lhs[i],
                                bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                                    pd.Timestamp(rhs[i])
                                ),
                            )
                    return out_arr

                return impl

        elif isinstance(rhs, types.Array) and rhs.dtype == bodo.types.datetime64ns:
            # date scalar + datetime64
            if lhs == datetime_date_type:

                def impl(lhs, rhs):  # pragma: no cover
                    numba.parfors.parfor.init_prange()
                    n = len(rhs)
                    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(rhs, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(
                                bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                                    pd.Timestamp(lhs)
                                ),
                                rhs[i],
                            )
                    return out_arr

                return impl

            # date array + datetime64
            elif lhs == datetime_date_array_type:

                def impl(lhs, rhs):  # pragma: no cover
                    numba.parfors.parfor.init_prange()
                    n = len(rhs)
                    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if bodo.libs.array_kernels.isna(
                            lhs, i
                        ) or bodo.libs.array_kernels.isna(rhs, i):
                            bodo.libs.array_kernels.setna(out_arr, i)
                        else:
                            out_arr[i] = op(
                                bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                                    pd.Timestamp(lhs[i])
                                ),
                                rhs[i],
                            )
                    return out_arr

                return impl

    return overload_arr_cmp


def create_cmp_op_overload_arr(op):
    """create overload function for comparison operators with datetime_date_array"""

    def overload_date_arr_cmp(lhs, rhs):
        # both datetime_date_array_type
        if op == operator.ne:
            default_value = True
        else:
            default_value = False
        if lhs == datetime_date_array_type and rhs == datetime_date_array_type:

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
        elif lhs == datetime_date_array_type:

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
        elif rhs == datetime_date_array_type:

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


UNIX_EPOCH_ORD = _ymd2ord(1970, 1, 1)


def now_date_python(tz_value_or_none: str | int | None):
    """Pure python function run in object mode
    to return the equivalent of datetime.datetime.now(tzInfo).date().

    This function is responsible for converting tz_value_or_none
    to a proper timezone.

    Args:
        tz_value_or_none (Union[str, int, None]): The input to
        create a new tzInfo.

    Returns:
        datetime.date: Current date in the local timezone.
    """
    if tz_value_or_none is None:
        return datetime.date.today()
    else:
        if isinstance(tz_value_or_none, int):
            tz_info = bodo.libs.pd_datetime_arr_ext.nanoseconds_to_offset(
                tz_value_or_none
            )
        else:
            assert isinstance(tz_value_or_none, str)
            tz_info = pytz.timezone(tz_value_or_none)
        return datetime.datetime.now(tz_info).date()


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def now_date_wrapper(tz_value_or_none=None):
    """JIT wrapper equivalent to datetime.datetime.now(tzInfo).date(),
    but accepting a string/int instead of an actual timezone.
    This is because we cannot represent timezone objects directly yet.

    Args:
        tz_str_or_none Union[str, int, None]: The input to
        create a new tzInfo.

    Returns:
        datetime.date: Current date in the local timezone.

    Returns: t
    """

    def impl(tz_value_or_none=None):  # pragma: no cover
        with numba.objmode(d="datetime_date_type"):
            d = now_date_python(tz_value_or_none)
        return d

    return impl


@register_jitable
def now_date_wrapper_consistent(tz_value_or_none=None):  # pragma: no cover
    """Wrapper around now_date_wrapper that ensure the result
    is consistent on all ranks.
    """
    if bodo.get_rank() == 0:
        d = now_date_wrapper(tz_value_or_none)
    else:
        # Give a dummy date for type stability
        d = datetime.date(2023, 1, 1)
    return bodo.libs.distributed_api.bcast_scalar(d)
