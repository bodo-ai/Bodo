# Copyright (C) 2021 Bodo Inc. All rights reserved.
""" Implementation of binary operators for the different types. 
    Currently implemented operators: add, sub
"""

import operator
import bodo

from bodo.utils.typing import BodoError, is_timedelta_type
from bodo.hiframes.pd_index_ext import DatetimeIndexType
from bodo.hiframes.pd_offsets_ext import week_type, month_end_type, date_offset_type
from bodo.hiframes.series_impl import SeriesType
from bodo.hiframes.datetime_date_ext import (
    datetime_date_type,
    datetime_timedelta_type,
    datetime_date_array_type,
)
from bodo.hiframes.datetime_timedelta_ext import (
    pd_timedelta_type,
    datetime_datetime_type,
    datetime_timedelta_array_type,
)
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs import hdatetime_ext
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.decimal_arr_ext import Decimal128Type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntegerArrayType

from numba.extending import overload
from numba.core import types

### Operator.add
@overload(operator.add, no_unliteral=True)
def overload_add_operator(lhs, rhs):
    """Overload the add operator. Note that the order is important.
    Please don't change unless it's intentional.
    """

    # Time series
    if time_series_operation(lhs, rhs):
        overload_impl = bodo.hiframes.series_dt_impl.create_bin_op_overload(
            operator.add
        )
        return overload_impl(lhs, rhs)

    if isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType):
        return bodo.hiframes.series_impl.create_binary_op_overload(operator.add)(
            lhs, rhs
        )

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

    # if supported by Numba, pass
    if add_and_sub_supported_by_numba(lhs, rhs) or add_only_supported_by_numba(
        lhs, rhs
    ):
        return

    raise BodoError(f"add operator not supported for data types {lhs} and {rhs}.")


### Operator.sub
@overload(operator.sub, no_unliteral=True)
def overload_sub_operator(lhs, rhs):
    """Overload the sub operator. Note that the order is important.
    Please don't change unless it's intentional.
    """

    # Offsets
    if subtracting_offset_to_datetime_or_timestamp(lhs, rhs):
        return bodo.hiframes.pd_offsets_ext.overload_sub_operator_offsets(lhs, rhs)

    # The order matters here: make sure offset types are before datetime types
    # Datetime types
    if lhs == pandas_timestamp_type and rhs in [
        pandas_timestamp_type,
        datetime_timedelta_type,
        pd_timedelta_type,
    ]:
        return bodo.hiframes.pd_timestamp_ext.overload_sub_operator_timestamp(lhs, rhs)

    if substracting_dt_or_td(lhs, rhs):
        return bodo.hiframes.datetime_date_ext.overload_sub_operator_datetime_date(
            lhs, rhs
        )

    if subtracting_datetime_and_timedeltas(lhs, rhs):
        return bodo.hiframes.datetime_timedelta_ext.overload_sub_operator_datetime_timedelta(
            lhs, rhs
        )

    if lhs == datetime_datetime_type and rhs == datetime_datetime_type:
        return (
            bodo.hiframes.datetime_datetime_ext.overload_sub_operator_datetime_datetime(
                lhs, rhs
            )
        )

    if subtracting_dt_index_and_timestamp(lhs, rhs):
        return bodo.hiframes.pd_index_ext.overload_sub_operator_datetime_index(lhs, rhs)

    # Time Series
    if time_series_operation(lhs, rhs):
        overload_impl = bodo.hiframes.series_dt_impl.create_bin_op_overload(
            operator.sub
        )
        return overload_impl(lhs, rhs)

    if isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType):
        return bodo.hiframes.series_impl.create_binary_op_overload(operator.sub)(
            lhs, rhs
        )

    # int array
    if isinstance(lhs, IntegerArrayType) or isinstance(rhs, IntegerArrayType):
        return bodo.libs.int_arr_ext.create_op_overload(operator.sub, 2)(lhs, rhs)

    # if supported by Numba, pass
    if add_and_sub_supported_by_numba(lhs, rhs):
        return

    raise BodoError(f"sub operator not supported for data types {lhs} and {rhs}.")


### Helper Functions For Checking Types
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


def subtracting_offset_to_datetime_or_timestamp(lhs, rhs):
    """ Helper function to check types supported in pd_offsets_ext add op overload. """

    dt_types = [datetime_datetime_type, pandas_timestamp_type, datetime_date_type]
    offset_types = [date_offset_type, month_end_type, week_type]
    lhs_offset = lhs in offset_types and rhs in dt_types
    rhs_offset = rhs in offset_types and lhs in dt_types

    return lhs_offset or rhs_offset


def subtracting_dt_index_and_timestamp(lhs, rhs):
    """ Helper function to check types supported in pd_index_ext sub op overload. """

    lhs_index = isinstance(lhs, DatetimeIndexType) and rhs == pandas_timestamp_type
    rhs_index = isinstance(rhs, DatetimeIndexType) and lhs == pandas_timestamp_type

    return lhs_index or rhs_index


def substracting_dt_or_td(lhs, rhs):
    """ Helper function to check types supported in datetime_date_ext sub op overload. """

    date_and_timedelta = lhs == datetime_date_type and rhs == datetime_timedelta_type
    date_and_date = lhs == datetime_date_type and rhs == datetime_date_type
    date_array_and_timedelta = (
        lhs == datetime_date_array_type and rhs == datetime_timedelta_type
    )

    return date_and_timedelta or date_and_date or date_array_and_timedelta


def subtracting_datetime_and_timedeltas(lhs, rhs):
    """ Helper function to check types supported in datetime_timedelta_ext sub op overload. """

    td_cond = (is_timedelta_type(lhs) or lhs == datetime_datetime_type) and (
        is_timedelta_type(rhs)
    )
    array_cond = lhs == datetime_timedelta_array_type and rhs == datetime_timedelta_type

    return td_cond or array_cond


def helper_time_series_checks(operand):
    """Helper function that checks whether the operand
    type is supported with the dt64_series add/sub ops in series_dt_impl."""
    ret = (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(operand)
        or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(operand)
        or operand
        in [datetime_timedelta_type, datetime_datetime_type, pandas_timestamp_type]
    )
    return ret


def time_series_operation(lhs, rhs):
    """ Helper function to check types supported in series_dt_impl by add/sub op overload. """
    td64series_and_timedelta = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
        and rhs == datetime_timedelta_type
    )
    timedelta_and_td64series = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
        and lhs == datetime_timedelta_type
    )
    dt64series_lhs = bodo.hiframes.pd_series_ext.is_dt64_series_typ(
        lhs
    ) and helper_time_series_checks(rhs)
    dt64series_rhs = bodo.hiframes.pd_series_ext.is_dt64_series_typ(
        rhs
    ) and helper_time_series_checks(lhs)

    return (
        td64series_and_timedelta
        or timedelta_and_td64series
        or dt64series_lhs
        or dt64series_rhs
    )


def create_overload_cmp_operator(op):
    """ create overloads for the comparison operators. """

    def overload_cmp_operator(lhs, rhs):
        # datetime.date
        if lhs == datetime_date_type and rhs == datetime_date_type:
            return bodo.hiframes.datetime_date_ext.create_cmp_op_overload(op)(lhs, rhs)

        # datetime.date array
        if lhs == datetime_date_array_type or rhs == datetime_date_array_type:
            return bodo.hiframes.datetime_date_ext.create_cmp_op_overload_arr(op)(
                lhs, rhs
            )

        # datetime.datetime
        if lhs == datetime_datetime_type and rhs == datetime_datetime_type:
            return bodo.hiframes.datetime_datetime_ext.create_cmp_op_overload(op)(
                lhs, rhs
            )

        # datetime.timedelta
        if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:
            return bodo.hiframes.datetime_timedelta_ext.create_cmp_op_overload(op)(
                lhs, rhs
            )

        # datetime.timedelta array
        if lhs == datetime_timedelta_array_type or rhs == datetime_timedelta_array_type:
            impl = bodo.hiframes.datetime_timedelta_ext.create_cmp_op_overload(op)
            return impl(lhs, rhs)

        # pd.timedelta
        if comparing_timedeltas(lhs, rhs):
            impl = bodo.hiframes.datetime_timedelta_ext.pd_create_cmp_op_overload(op)
            return impl(lhs, rhs)

        # index
        if comparing_dt_index_to_string(lhs, rhs):
            return bodo.hiframes.pd_index_ext.overload_binop_dti_str(op)(lhs, rhs)

        # timestamp
        if comparing_timestamp_or_date(lhs, rhs):
            return bodo.hiframes.pd_timestamp_ext.create_timestamp_cmp_op_overload(op)(
                lhs, rhs
            )

        # time series (order matters: time series check should be before the generic series check)
        if is_timeseries_op(lhs, rhs):
            return bodo.hiframes.series_dt_impl.create_cmp_op_overload(op)(lhs, rhs)

        # series
        if isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType):
            return bodo.hiframes.series_impl.create_binary_op_overload(op)(lhs, rhs)

        # str_arr
        if lhs == string_array_type or rhs == string_array_type:
            return bodo.libs.str_arr_ext.create_binary_op_overload(op)(lhs, rhs)

        # decimal_arr
        if isinstance(lhs, Decimal128Type) and isinstance(rhs, Decimal128Type):
            return bodo.libs.decimal_arr_ext.decimal_create_cmp_op_overload(op)(
                lhs, rhs
            )

        # boolean array
        if lhs == boolean_array or rhs == boolean_array:
            return bodo.libs.bool_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # int array
        if isinstance(lhs, IntegerArrayType) or isinstance(rhs, IntegerArrayType):
            return bodo.libs.int_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # if supported by Numba, pass
        if cmp_op_supported_by_numba(lhs, rhs):
            return

        # Types passed to the binary operator are not supported
        # raise BodoError(f"{op} operator not supported for data types {lhs} and {rhs}.")

    return overload_cmp_operator


def comparing_dt_index_to_string(lhs, rhs):
    """ Helper function to check types supported in pd_index_ext by cmp op overload. """

    lhs_index = (
        isinstance(lhs, DatetimeIndexType) and types.unliteral(rhs) == string_type
    )
    rhs_index = (
        isinstance(rhs, DatetimeIndexType) and types.unliteral(lhs) == string_type
    )

    return lhs_index or rhs_index


def comparing_timestamp_or_date(lhs, rhs):
    """ Helper function to check types supported in pd_timestamp_ext by cmp op overload. """

    ts_and_date = (
        lhs == pandas_timestamp_type
        and rhs == bodo.hiframes.datetime_date_ext.datetime_date_type
    )
    date_and_ts = (
        lhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        and rhs == pandas_timestamp_type
    )
    ts_and_ts = lhs == pandas_timestamp_type and rhs == pandas_timestamp_type

    ts_and_dt64 = lhs == pandas_timestamp_type and rhs == bodo.datetime64ns
    dt64_and_ts = rhs == pandas_timestamp_type and lhs == bodo.datetime64ns

    return ts_and_date or date_and_ts or ts_and_ts or ts_and_dt64 or dt64_and_ts


def is_timeseries_op(lhs, rhs):
    """ Helper function to check types supported in series_dt_impl by cmp op overload. """

    dt64s_with_string_or_ts = bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs) and (
        bodo.utils.typing.is_overload_constant_str(lhs)
        or lhs == bodo.libs.str_ext.string_type
        or lhs == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
    )
    string_or_ts_with_dt64s = bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs) and (
        bodo.utils.typing.is_overload_constant_str(rhs)
        or rhs == bodo.libs.str_ext.string_type
        or rhs == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
    )
    dt64_series_ops = dt64s_with_string_or_ts or string_or_ts_with_dt64s

    tds_and_td = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
        and lhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
    )
    td_and_tds = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
        and rhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
    )
    td_series_ops = tds_and_td or td_and_tds

    return dt64_series_ops or td_series_ops


def comparing_timedeltas(lhs, rhs):
    """ Helper function to check types supported in datetime_timedelta_ext by cmp op overload. """

    deltas = [pd_timedelta_type, bodo.timedelta64ns]
    return lhs in deltas and rhs in deltas


def add_and_sub_supported_by_numba(lhs, rhs):
    """ Signatures supported by Numba for operator.add and operator.sub. """

    # timedelta
    timedeltas = isinstance(lhs, types.NPTimedelta) and isinstance(
        rhs, types.NPTimedelta
    )

    # NPDatetimes
    np_dt = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPDatetime)

    # NPDatetime and NPTimedelta
    dt_td = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPTimedelta)

    # Sets
    sets = isinstance(lhs, types.Set) and isinstance(rhs, types.Set)

    # Numbers
    ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
    reals = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)
    cmplx = isinstance(lhs, types.Complex) and isinstance(rhs, types.Complex)
    numbers = ints or reals or cmplx

    # Arrays
    arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

    return timedeltas or np_dt or dt_td or sets or numbers or arrs


def add_only_supported_by_numba(lhs, rhs):
    """ Signatures supported by Numba for operator.add. """

    # Tuples
    tuples = isinstance(lhs, types.BaseTuple) and isinstance(rhs, types.BaseTuple)

    # Lists
    lists = isinstance(lhs, types.List) and isinstance(rhs, types.List)

    # Chars
    char_seq_char = isinstance(lhs, types.UnicodeCharSeq) and isinstance(
        rhs, types.UnicodeType
    )
    char_char_seq = isinstance(rhs, types.UnicodeCharSeq) and isinstance(
        lhs, types.UnicodeType
    )
    char_seq_char_seq = isinstance(lhs, types.UnicodeCharSeq) and isinstance(
        rhs, types.UnicodeCharSeq
    )
    char_seq_bytes = isinstance(lhs, (types.CharSeq, types.Bytes)) and isinstance(
        rhs, (types.CharSeq, types.Bytes)
    )

    char_add = char_seq_char or char_char_seq or char_seq_char_seq or char_seq_bytes

    # Strings
    unicodes = isinstance(lhs, types.UnicodeType) and isinstance(rhs, types.UnicodeType)
    char_seq_unicode = isinstance(lhs, types.UnicodeType) and isinstance(
        rhs, types.UnicodeCharSeq
    )

    string_add = unicodes or char_seq_unicode

    # NPTimedelta and NPDatetime
    np_dt = lhs == types.NPTimedelta and rhs == types.NPDatetime

    return tuples or lists or char_add or string_add or np_dt


def cmp_op_supported_by_numba(lhs, rhs):
    """ Signatures supported by Numba for cmp operator. """

    # Lists
    lists = isinstance(lhs, types.ListType) and isinstance(rhs, types.ListType)

    # timedelta
    timedeltas = isinstance(lhs, types.NPTimedelta) and isinstance(
        rhs, types.NPTimedelta
    )

    # datetime.datetime
    datetimes = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPDatetime)

    # unicodes
    unicode_types = (
        types.UnicodeType,
        types.StringLiteral,
        types.CharSeq,
        types.Bytes,
        types.UnicodeCharSeq,
    )
    unicodes = isinstance(lhs, unicode_types) and isinstance(rhs, unicode_types)

    # tuples
    tuples = isinstance(lhs, types.BaseTuple) and isinstance(rhs, types.BaseTuple)

    # sets
    sets = isinstance(lhs, types.Set) and isinstance(rhs, types.Set)

    # numbers
    numbers = isinstance(lhs, types.Number) and isinstance(rhs, types.Number)

    # bools
    bools = isinstance(lhs, types.Boolean) and isinstance(rhs, types.Boolean)

    # None
    nones = isinstance(lhs, types.NoneType) or isinstance(rhs, types.NoneType)

    # dictionaries
    dicts = isinstance(lhs, types.DictType) and isinstance(rhs, types.DictType)

    # arrays
    arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

    # enums
    enums = isinstance(lhs, types.EnumMember) and isinstance(rhs, types.EnumMember)

    # literals
    literals = isinstance(lhs, types.Literal) and isinstance(rhs, types.Literal)

    return (
        lists
        or timedeltas
        or datetimes
        or unicodes
        or tuples
        or sets
        or numbers
        or bools
        or nones
        or dicts
        or arrs
        or enums
        or literals
    )


def _install_cmp_ops():
    for op in (
        operator.lt,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
    ):
        overload_impl = create_overload_cmp_operator(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_cmp_ops()
