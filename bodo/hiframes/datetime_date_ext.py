# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Numba extension support for datetime.date objects and their arrays.
"""
import operator
import numpy as np
import datetime
import numba
from numba import types
from numba.targets.imputils import lower_builtin
from numba.typing import signature
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
    register_jitable,
)
from numba import cgutils
from numba.typing.templates import (
    infer_getattr,
    AttributeTemplate,
    bound_function,
    signature,
    infer_global,
    AbstractTemplate,
    ConcreteTemplate,
)
from numba.array_analysis import ArrayAnalysis
from llvmlite import ir as lir
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_type
from bodo.hiframes.datetime_datetime_ext import DatetimeDatetimeType
import bodo
from bodo.utils.typing import is_list_like_index_type
from bodo.libs import hdatetime_ext
import llvmlite.binding as ll


ll.add_symbol("box_datetime_date_array", hdatetime_ext.box_datetime_date_array)
ll.add_symbol("unbox_datetime_date_array", hdatetime_ext.unbox_datetime_date_array)


# datetime.date implementation that uses a single int to store year/month/day
# Does not need refcounted object wrapping since it is immutable
class DatetimeDateType(types.Type):
    def __init__(self):
        super(DatetimeDateType, self).__init__(name="DatetimeDateType()")
        self.bitwidth = 64  # needed for using IntegerModel


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


@lower_getattr(DatetimeDateType, "year")
def datetime_get_year(context, builder, typ, val):
    return builder.lshr(val, lir.Constant(lir.IntType(64), 32))


@lower_getattr(DatetimeDateType, "month")
def datetime_get_month(context, builder, typ, val):
    return builder.and_(
        builder.lshr(val, lir.Constant(lir.IntType(64), 16)),
        lir.Constant(lir.IntType(64), 0xFFFF),
    )


@lower_getattr(DatetimeDateType, "day")
def datetime_get_day(context, builder, typ, val):
    return builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF))


@unbox(DatetimeDateType)
def unbox_datetime_date(typ, val, c):

    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")

    yll = c.pyapi.long_as_longlong(year_obj)
    mll = c.pyapi.long_as_longlong(month_obj)
    dll = c.pyapi.long_as_longlong(day_obj)

    nopython_date = c.builder.add(
        dll,
        c.builder.add(
            c.builder.shl(yll, lir.Constant(lir.IntType(64), 32)),
            c.builder.shl(mll, lir.Constant(lir.IntType(64), 16)),
        ),
    )

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(nopython_date, is_error=is_error)


@box(DatetimeDateType)
def box_datetime_date(typ, val, c):
    year_obj = c.pyapi.long_from_longlong(
        c.builder.lshr(val, lir.Constant(lir.IntType(64), 32))
    )
    month_obj = c.pyapi.long_from_longlong(
        c.builder.and_(
            c.builder.lshr(val, lir.Constant(lir.IntType(64), 16)),
            lir.Constant(lir.IntType(64), 0xFFFF),
        )
    )
    day_obj = c.pyapi.long_from_longlong(
        c.builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF))
    )

    dt_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.date))
    res = c.pyapi.call_function_objargs(dt_obj, (year_obj, month_obj, day_obj))
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
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
    year, month, day = args
    nopython_date = builder.add(
        day,
        builder.add(
            builder.shl(year, lir.Constant(lir.IntType(64), 32)),
            builder.shl(month, lir.Constant(lir.IntType(64), 16)),
        ),
    )
    return nopython_date


@intrinsic
def cast_int_to_datetime_date(typingctx, val=None):
    """Cast int value to datetime.date
    """
    assert val == types.int64

    def codegen(context, builder, signature, args):
        return args[0]

    return datetime_date_type(types.int64), codegen


@intrinsic
def cast_datetime_date_to_int(typingctx, val=None):
    """Cast datetime.date value to int
    """
    assert val == datetime_date_type

    def codegen(context, builder, signature, args):
        return args[0]

    return types.int64(datetime_date_type), codegen


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


@register_jitable
def _is_leap(year):  # pragma: no cover
    "year -> 1 if leap year, else 0."
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@register_jitable
def _days_before_year(year):  # pragma: no cover
    "year -> number of days before January 1st of year."
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


@register_jitable
def _days_in_month(year, month):  # pragma: no cover
    "year, month -> number of days in that month in that year."
    if month == 2 and _is_leap(year):
        return 29
    return _DAYS_IN_MONTH[month]


@register_jitable
def _days_before_month(year, month):  # pragma: no cover
    "year, month -> number of days in year preceding first day of month."
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))


_DI400Y = _days_before_year(401)  # number of days in 400 years
_DI100Y = _days_before_year(101)  #    "    "   "   " 100   "
_DI4Y = _days_before_year(5)  #    "    "   "   "   4   "


@register_jitable
def _ymd2ord(year, month, day):  # pragma: no cover
    "year, month, day -> ordinal, considering 01-Jan-0001 as day 1."
    dim = _days_in_month(year, month)
    return _days_before_year(year) + _days_before_month(year, month) + day


@register_jitable
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


@register_jitable
def _cmp(x, y):  # pragma: no cover
    return 0 if x == y else 1 if x > y else -1


###############################################################################

types.datetime_date_type = datetime_date_type


@register_jitable
def today_impl():  # pragma: no cover
    """Internal call to support datetime.date.today().
    Untyped pass replaces datetime.date.today() with this call since class methods are
    not supported in Numba's typing
    """
    with numba.objmode(d="datetime_date_type"):
        d = datetime.date.today()
    return d


@register_jitable
def fromordinal_impl(n):  # pragma: no cover
    """Internal call to support datetime.date.fromordinal().
    Untyped pass replaces datetime.date.fromordinal() with this call since class methods are
    not supported in Numba's typing
    """
    y, m, d = _ord2ymd(n)
    return datetime.date(y, m, d)


@overload_method(DatetimeDatetimeType, "toordinal")
@overload_method(DatetimeDateType, "toordinal")
def toordinal(date):
    """Return proleptic Gregorian ordinal for the year, month and day.
    January 1 of year 1 is day 1.  Only the year, month and day values
    contribute to the result.
    """

    def impl(date):  # pragma: no cover
        return _ymd2ord(date.year, date.month, date.day)

    return impl


@overload_method(DatetimeDatetimeType, "weekday")
@overload_method(DatetimeDateType, "weekday")
def weekday(date):
    "Return day of the week, where Monday == 0 ... Sunday == 6."

    def impl(date):  # pragma: no cover
        return (date.toordinal() + 6) % 7

    return impl


@overload(operator.add)
def date_add(lhs, rhs):
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


@overload(operator.sub)
def date_sub(lhs, rhs):
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


@overload(operator.eq)
def date_eq(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) == 0

        return impl


@overload(operator.ne)
def date_ne(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) != 0

        return impl


@overload(operator.le)
def date_le(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) <= 0

        return impl


@overload(operator.lt)
def date_lt(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) < 0

        return impl


@overload(operator.ge)
def date_ge(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) >= 0

        return impl


@overload(operator.gt)
def date_gt(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            y, y2 = lhs.year, rhs.year
            m, m2 = lhs.month, rhs.month
            d, d2 = lhs.day, rhs.day
            return _cmp((y, m, d), (y2, m2, d2)) > 0

        return impl


@overload(min)
def date_min(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs < rhs else rhs

        return impl


@overload(max)
def date_max(lhs, rhs):
    if lhs == datetime_date_type and rhs == datetime_date_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs > rhs else rhs

        return impl


##################### Array of datetime.date objects ##########################


class DatetimeDateArrayType(types.ArrayCompatible):
    def __init__(self):
        super(DatetimeDateArrayType, self).__init__(name="DatetimeDateArrayType()")

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

# datetime.date array has only an array integers to store data
@register_model(DatetimeDateArrayType)
class DatetimeDateArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(types.int64, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DatetimeDateArrayType, "data", "_data")
make_attribute_wrapper(DatetimeDateArrayType, "null_bitmap", "_null_bitmap")


@unbox(DatetimeDateArrayType)
def unbox_datetime_date_array(typ, val, c):
    n = bodo.utils.utils.object_length(c, val)
    arr_typ = types.Array(types.intp, 1, "C")
    data_arr = bodo.utils.utils._empty_nd_impl(c.context, c.builder, arr_typ, [n])
    n_bitmask_bytes = c.builder.udiv(
        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    bitmap_arr = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
    )

    # function signature of unbox_datetime_date_array
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64).as_pointer(),
            lir.IntType(8).as_pointer(),
        ],
    )
    fn = c.builder.module.get_or_insert_function(fnty, name="unbox_datetime_date_array")
    c.builder.call(fn, [val, n, data_arr.data, bitmap_arr.data])

    out_dt_date_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out_dt_date_arr.data = data_arr._getvalue()
    out_dt_date_arr.null_bitmap = bitmap_arr._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(out_dt_date_arr._getvalue(), is_error=is_error)


def int_to_datetime_date_python(ia):
    return datetime.date(ia >> 32, (ia >> 16) & 0xFFFF, ia & 0xFFFF)


def int_array_to_datetime_date(ia):
    # setting 'otypes' is necessary since input can be empty
    return np.vectorize(int_to_datetime_date_python, otypes=[np.object])(ia)


@box(DatetimeDateArrayType)
def box_datetime_date_array(typ, val, c):
    in_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    data_arr = c.context.make_array(types.Array(types.int64, 1, "C"))(
        c.context, c.builder, in_arr.data
    )
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, in_arr.null_bitmap
    ).data

    n = c.builder.extract_value(data_arr.shape, 0)

    fnty = lir.FunctionType(
        c.pyapi.pyobj,
        [lir.IntType(64), lir.IntType(64).as_pointer(), lir.IntType(8).as_pointer()],
    )
    fn_get = c.builder.module.get_or_insert_function(
        fnty, name="box_datetime_date_array"
    )
    obj_arr = c.builder.call(fn_get, [n, data_arr.data, bitmap_arr_data])

    c.context.nrt.decref(c.builder, typ, val)
    return obj_arr


@intrinsic
def init_datetime_date_array(typingctx, data, nulls=None):
    """Create a DatetimeDateArrayType with provided data values.
    """
    assert data == types.Array(types.int64, 1, "C")
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
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], bitmap_val)

        return dt_date_arr._getvalue()

    sig = datetime_date_array_type(data, nulls)
    return sig, codegen


@numba.njit(no_cpython_wrapper=True)
def alloc_datetime_date_array(n):  # pragma: no cover
    data_arr = np.empty(n, dtype=np.int64)
    # XXX: set all bits to not null since datetime.date array operations do not support
    # NA yet. TODO: use 'empty' when all operations support NA
    # nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    nulls = np.full((n + 7) >> 3, 255, np.uint8)
    return init_datetime_date_array(data_arr, nulls)


def alloc_datetime_date_array_equiv(self, scope, equiv_set, args, kws):
    """Array analysis function for alloc_datetime_date_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_hiframes_datetime_date_ext_alloc_datetime_date_array = (
    alloc_datetime_date_array_equiv
)


@overload(operator.getitem)
def dt_date_arr_getitem(A, ind):
    if A != datetime_date_array_type:
        return

    if isinstance(ind, types.Integer):
        return lambda A, ind: cast_int_to_datetime_date(A._data[ind])

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_ndarray(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind]
            n = len(new_data)
            n_bytes = (n + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in numba.parfor.internal_prange(len(ind)):
                if ind[i]:
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                    curr_bit += 1
            return init_datetime_date_array(new_data, new_mask)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl(A, ind):  # pragma: no cover
            ind_t = bodo.utils.conversion.coerce_to_ndarray(ind)
            old_mask = A._null_bitmap
            new_data = A._data[ind_t]
            n = len(new_data)
            n_bytes = (n + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in range(len(ind)):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, ind_t[i])
                bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                curr_bit += 1
            return init_datetime_date_array(new_data, new_mask)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            n = len(A._data)
            old_mask = A._null_bitmap
            new_data = np.ascontiguousarray(A._data[ind])
            slice_idx = numba.unicode._normalize_slice(ind, n)
            span = numba.unicode._slice_span(slice_idx)
            n_bytes = (span + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                curr_bit += 1
            return init_datetime_date_array(new_data, new_mask)

        return impl_slice


@overload(operator.setitem)
def dt_date_arr_setitem(A, ind, val):
    if A != datetime_date_array_type:
        return

    if isinstance(ind, types.Integer):

        def impl(A, ind, val):
            A._data[ind] = cast_datetime_date_to_int(val)

        return impl


@overload(len)
def overload_len_datetime_date_arr(A):
    if A == datetime_date_array_type:
        return lambda A: len(A._data)


@overload_attribute(DatetimeDateArrayType, "shape")
def overload_datetime_date_arr_shape(A):
    return lambda A: (len(A._data),)


@overload(operator.sub)
def overload_datetime_date_arr_sub(arg1, arg2):
    from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_type

    # datetime_date_array - timedelta
    if arg1 == datetime_date_array_type and arg2 == datetime_timedelta_type:

        def impl(arg1, arg2):  # pragma: no cover
            in_arr = arg1
            numba.parfor.init_prange()
            n = len(in_arr)
            A = alloc_datetime_date_array(n)
            for i in numba.parfor.internal_prange(n):
                A[i] = in_arr[i] - arg2
            return A

        return impl


def create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_date_array
    """

    def overload_date_arr_cmp(A1, A2):
        # both datetime_date_array_type
        if A1 == datetime_date_array_type and A2 == datetime_date_array_type:

            def impl(A1, A2):
                numba.parfor.init_prange()
                n = len(A1)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    out_arr[i] = op(A1[i], A2[i])
                return out_arr

            return impl
        # 1st arg is array
        elif A1 == datetime_date_array_type:

            def impl(A1, A2):
                numba.parfor.init_prange()
                n = len(A1)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    out_arr[i] = op(A1[i], A2)
                return out_arr

            return impl
        # 2nd arg is array
        elif A2 == datetime_date_array_type:

            def impl(A1, A2):
                numba.parfor.init_prange()
                n = len(A2)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    out_arr[i] = op(A1, A2[i])
                return out_arr

            return impl

    return overload_date_arr_cmp


def _install_cmp_ops():
    """install overloads for comparison operators with datetime_date_array
    """
    for op in (
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ):
        overload_impl = create_cmp_op_overload(op)
        overload(op)(overload_impl)


_install_cmp_ops()
