# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implement support for the various classes in pd.tseries.offsets.
"""
import operator

import numba
import numpy as np
import pandas as pd
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


# 1.Define a new Numba type class by subclassing the Type class
#   Define a singleton Numba type instance for a non-parametric type
class MonthEndType(types.Type):
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

    # lhs is a datetime
    if lhs == datetime_datetime_type and rhs == month_end_type:

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

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

    # lhs is a timestamp
    if lhs == pandas_timestamp_type and rhs == month_end_type:

        def impl(lhs, rhs):  # pragma: no cover
            return rhs + lhs

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

    # lhs is a datetime.date
    if lhs == datetime_date_type and rhs == month_end_type:

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
        lhs == datetime_datetime_type
        or lhs == pandas_timestamp_type
        or lhs == datetime_date_type
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
