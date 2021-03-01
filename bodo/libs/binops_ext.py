# Copyright (C) 2021 Bodo Inc. All rights reserved.
""" Implementation of binary operators for the different types. 
    Currently implemented operators: add
"""

import operator
import bodo

from bodo.utils.typing import BodoError, is_timedelta_type
from bodo.hiframes.pd_offsets_ext import week_type, month_end_type, date_offset_type
from bodo.hiframes.datetime_date_ext import datetime_date_type, datetime_timedelta_type
from bodo.hiframes.datetime_timedelta_ext import (
    pd_timedelta_type,
    datetime_datetime_type,
)
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type

from numba.extending import overload
from numba.core import types


@overload(operator.add, no_unliteral=True)
def overload_add_operator(lhs, rhs):
    """Overload the add operator. Note that the order is important.
    Please don't change unless it's intentional.
    """

    # Offsets
    if lhs == week_type or rhs == week_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_week_offset_type(
            lhs, rhs
        )
    if lhs == month_end_type or rhs == month_end_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_month_end_offset_type(
            lhs, rhs
        )
    if lhs == date_offset_type or rhs == date_offset_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_perator_date_offset_type(
            lhs, rhs
        )

    # The order matters here: make sure offset types are before datetime types
    # Datetime types
    if adding_timestamp(lhs, rhs):
        return bodo.hiframes.pd_timestamp_ext.overload_add_operator_timestamp(lhs, rhs)

    if adding_dt_td_and_dt_date(lhs, rhs):
        return bodo.hiframes.datetime_date_ext.overload_add_operator_datetime_date(
            lhs, rhs
        )

    if adding_datetime_and_timedeltas(lhs, rhs):
        return bodo.hiframes.datetime_timedelta_ext.overload_add_operator_datetime_timedelta(
            lhs, rhs
        )

    # String arrays
    if lhs == string_array_type or types.unliteral(lhs) == string_type:
        return bodo.libs.str_arr_ext.overload_add_operator_string_array(lhs, rhs)

    # Series are doing an overload_method. No need to be included here

    # Types passed to the binary operator are not supported
    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


def adding_dt_td_and_dt_date(lhs, rhs):
    """ Helper function to check types supported in datetime_date_ext overload. """

    lhs_td = lhs == datetime_timedelta_type and rhs == datetime_date_type
    rhs_td = rhs == datetime_timedelta_type and lhs == datetime_date_type
    return lhs_td or rhs_td


def adding_timestamp(lhs, rhs):
    """ Helper function to check types supported in pd_timestamp_ext overload. """

    ts_and_td = lhs == pandas_timestamp_type and is_timedelta_type(rhs)
    td_and_ts = is_timedelta_type(lhs) and rhs == pandas_timestamp_type

    return ts_and_td or td_and_ts


def adding_datetime_and_timedeltas(lhs, rhs):
    """ Helper function to check types supported in datetime_timedelta_ext overload. """

    td_types = [datetime_timedelta_type, pd_timedelta_type]
    lhs_types = [datetime_timedelta_type, pd_timedelta_type, datetime_datetime_type]
    deltas = lhs in td_types and rhs in td_types
    dt = (lhs == datetime_datetime_type and rhs in td_types) or (
        rhs == datetime_datetime_type and lhs in td_types
    )

    return deltas or dt
