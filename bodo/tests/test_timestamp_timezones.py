# Copyright (C) 2022 Bodo Inc. All rights reserved.

import datetime
import re

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.timezone_common import representative_tz, sample_tz  # noqa
from bodo.tests.utils import check_func, generate_comparison_ops_func
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.tz_aware


@pytest.fixture(
    params=[
        "2019-01-01",
        "2020-01-01",
        "2030-01-01",
    ]
)
def timestamp_str(request):
    return request.param


@pytest.fixture(
    params=[
        "UTC",
        "US/Eastern",
        "US/Pacific",
        "Europe/Berlin",
    ],
)
def timezone(request):
    return request.param


def test_timestamp_timezone_boxing(timestamp_str, timezone, memory_leak_check):
    def test_impl(timestamp):
        return timestamp

    check_func(test_impl, (pd.Timestamp(timestamp_str, tz=timezone),))


def test_timestamp_timezone_constant_lowering(
    timestamp_str, timezone, memory_leak_check
):
    timestamp = pd.Timestamp(timestamp_str, tz=timezone)

    def test_impl():
        return timestamp

    check_func(test_impl, ())


def test_timestamp_timezone_constructor(timestamp_str, timezone, memory_leak_check):
    def test_impl(ts, tz):
        return pd.Timestamp(ts, tz=tz)

    check_func(test_impl, (timestamp_str, timezone))


def test_timestamp_tz_convert(representative_tz):
    def test_impl(ts, tz):
        return ts.tz_convert(tz=tz)

    ts = pd.Timestamp("09-30-2020", tz="Poland")
    check_func(
        test_impl,
        (
            ts,
            representative_tz,
        ),
    )

    ts = pd.Timestamp("09-30-2020")
    with pytest.raises(
        BodoError,
        match="Cannot convert tz-naive Timestamp, use tz_localize to localize",
    ):
        bodo.jit(test_impl)(ts, representative_tz)


def test_timestamp_tz_localize(representative_tz):
    def test_impl1(ts, tz):
        return ts.tz_localize(tz=tz)

    def test_impl2(ts):
        return ts.tz_localize(None)

    ts = pd.Timestamp("09-30-2020 14:00")
    check_func(
        test_impl1,
        (
            ts,
            representative_tz,
        ),
    )
    ts = pd.Timestamp("09-30-2020 14:00", tz=representative_tz)
    check_func(test_impl2, (ts,))


def test_timestamp_tz_ts_input():
    def test_impl(ts_input, tz_str):
        return pd.Timestamp(ts_input, tz="US/Pacific")

    # ts_input represents 04-05-2021 12:00:00 for tz='US/Pacific'
    ts_input = 1617649200000000000
    tz_str = "US/Pacific"

    check_func(
        test_impl,
        (
            ts_input,
            tz_str,
        ),
    )


def test_tz_timestamp_max_min():
    def impl_max(s):
        return s.max()

    def impl_min(s):
        return s.min()

    s1 = pd.Series(
        [
            pd.Timestamp("2020-03-15", tz="Europe/Stockholm"),
            pd.Timestamp("2020-03-16", tz="Europe/Stockholm"),
            pd.Timestamp("2020-03-17", tz="Europe/Stockholm"),
        ]
    )

    s2 = pd.Series(
        [
            pd.Timestamp("1999-10-31 12:23:33", tz="US/Eastern"),
            pd.Timestamp("2005-01-01 00:00:00", tz="US/Eastern"),
            pd.Timestamp("2016-02-28 12:23:33", tz="US/Eastern"),
        ]
    )

    check_func(impl_max, (s1,))
    check_func(impl_max, (s2,))
    check_func(impl_min, (s1,))
    check_func(impl_min, (s2,))


def test_tz_datetime_arr_unsupported():
    def impl(arr):
        return np.hstack([arr, arr])

    non_tz_arr = pd.array([pd.Timestamp("2020-01-01")] * 10)
    tz_arr = pd.array([pd.Timestamp("2020-01-01", tz="US/Eastern")] * 10)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Cannot support timezone naive pd.arrays.DatetimeArray. Please convert to a numpy array with .astype('datetime64[ns]')"
        ),
    ):
        bodo.jit(impl)(non_tz_arr)

    with pytest.raises(
        BodoError,
        match=".*Timezone-aware array not yet supported.*",
    ):
        bodo.jit(impl)(tz_arr)


def test_tz_index_unsupported():
    def impl(idx):
        return idx.min()

    non_tz_idx = pd.date_range("2020-01-01", periods=10)
    tz_idx = pd.date_range("2020-01-01", periods=10, tz="US/Eastern")

    check_func(impl, (non_tz_idx,))

    with pytest.raises(
        BodoError,
        match=".*Timezone-aware index not yet supported.*",
    ):
        bodo.jit(impl)(tz_idx)


def test_tz_date_scalar_cmp(sample_tz, cmp_op, memory_leak_check):
    """Check that scalar comparison operators work between dates and
    Timestamps
    """
    func = generate_comparison_ops_func(cmp_op)
    d = datetime.date(2022, 4, 4)
    ts = pd.Timestamp("4/4/2022", tz=sample_tz)
    # date + Timestamp comparison is deprecated. The current library truncates to date,
    # but to match SQL expectations we cast the date to the timestamp instead.
    d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
    check_func(func, (d, ts), py_output=cmp_op(d_ts, ts))
    check_func(func, (ts, d), py_output=cmp_op(ts, d_ts))
    # Check where they aren't equal
    d = datetime.date(2022, 4, 3)
    d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
    check_func(func, (d, ts), py_output=cmp_op(d_ts, ts))
    check_func(func, (ts, d), py_output=cmp_op(ts, d_ts))


def test_date_array_tz_scalar(sample_tz, cmp_op, memory_leak_check):
    """Check that comparison operators work between an array
    of dates and a Timestamp
    """
    func = generate_comparison_ops_func(cmp_op)
    arr = (
        pd.date_range(start="2/1/2022", freq="8D2H30T", periods=30, tz=sample_tz)
        .to_series()
        .dt.date.values
    )
    ts = pd.Timestamp("4/4/2022", tz=sample_tz)
    check_func(func, (arr, ts))
    check_func(func, (ts, arr))


def test_date_series_tz_scalar(sample_tz, cmp_op, memory_leak_check):
    """Check that comparison operators work between an Series
    of dates and a Timestamp
    """
    func = generate_comparison_ops_func(cmp_op)
    S = (
        pd.date_range(start="2/1/2022", freq="8D2H30T", periods=30, tz=sample_tz)
        .to_series()
        .dt.date
    )
    ts = pd.Timestamp("4/4/2022", tz=sample_tz)
    check_func(func, (S, ts))
    check_func(func, (ts, S))


def test_tz_tz_scalar_cmp(cmp_op, memory_leak_check):
    """Check that scalar comparison operators work between
    Timestamps with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    timezone = "Poland"
    ts = pd.Timestamp("4/4/2022", tz=timezone)
    check_func(func, (ts, ts))
    check_func(func, (ts, ts))
    ts2 = pd.Timestamp("1/4/2022", tz=timezone)
    # Check where they aren't equal
    d = datetime.date(2022, 4, 3)
    check_func(func, (ts2, ts))
    check_func(func, (ts, ts2))


def test_different_tz_unsupported(cmp_op):
    """Check that scalar comparison operators work between
    Timestamps with different timezone.
    """
    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    ts1 = pd.Timestamp("4/4/2022")
    ts2 = pd.Timestamp("4/4/2022", tz="Poland")
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts1, ts2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts2, ts1)
    # Check different timezones aren't supported
    ts3 = pd.Timestamp("4/4/2022", tz="US/Pacific")
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts3, ts2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts2, ts3)


def test_pd_timedelta_add(representative_tz, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    ts = pd.Timestamp(
        year=2022,
        month=11,
        day=6,
        hour=0,
        minute=36,
        second=11,
        microsecond=113,
        nanosecond=204,
        tz=representative_tz,
    )
    td = pd.Timedelta(hours=2, seconds=11, nanoseconds=45)
    check_func(test_impl, (td, ts))
    check_func(test_impl, (ts, td))


def test_datetime_timedelta_add(representative_tz, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    ts = pd.Timestamp(
        year=2022,
        month=11,
        day=6,
        hour=0,
        minute=36,
        second=11,
        microsecond=113,
        nanosecond=204,
        tz=representative_tz,
    )
    td = datetime.timedelta(hours=2, seconds=11, microseconds=45)
    check_func(test_impl, (td, ts))
    check_func(test_impl, (ts, td))


def test_pd_timedelta_sub(representative_tz, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    ts = pd.Timestamp(
        year=2022,
        month=11,
        day=6,
        hour=3,
        minute=36,
        second=11,
        microsecond=113,
        nanosecond=204,
        tz=representative_tz,
    )
    td = pd.Timedelta(hours=2, seconds=11, nanoseconds=45)
    check_func(test_impl, (ts, td))


def test_datetime_timedelta_sub(representative_tz, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    ts = pd.Timestamp(
        year=2022,
        month=11,
        day=6,
        hour=3,
        minute=36,
        second=11,
        microsecond=113,
        nanosecond=204,
        tz=representative_tz,
    )
    td = datetime.timedelta(hours=2, seconds=11, microseconds=45)
    check_func(test_impl, (ts, td))


def test_timestamp_now_with_tz_str(representative_tz, memory_leak_check):

    # Note: we have to lower this a global so that it's constant, since we require it to be
    # a constant at this time
    @bodo.jit()
    def test_impl():
        return pd.Timestamp.now(representative_tz)

    # Note: have to test this manually as pd.Timestamp.now() will return slightly differing values
    out = test_impl()
    assert pd.Timestamp.now(representative_tz) - out < pd.Timedelta(1, "min")


def test_tz_constructor_values(representative_tz, memory_leak_check):
    tz = representative_tz

    def test_constructor_kw(year, month, day, hour):
        # Test constructor with year/month/day passed as keyword arguments
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, tz=tz)

    def test_constructor_kw_value(year, month, day, hour):
        # Check the utc value
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, tz=tz).value

    # Note: This checks both sides of daylight savings.
    check_func(test_constructor_kw, (2022, 11, 6, 0))
    check_func(test_constructor_kw, (2022, 11, 6, 3))
    check_func(test_constructor_kw, (2022, 3, 13, 0))
    check_func(test_constructor_kw, (2022, 3, 13, 3))

    check_func(test_constructor_kw_value, (2022, 11, 6, 0))
    check_func(test_constructor_kw_value, (2022, 11, 6, 3))
    check_func(test_constructor_kw_value, (2022, 3, 13, 0))
    check_func(test_constructor_kw_value, (2022, 3, 13, 3))


def test_tz_add_month_begin(representative_tz, memory_leak_check):
    """
    Add tests for adding TZ-Aware timezones with a Pandas
    MonthBegin type.
    """

    def impl(lhs, rhs):
        return lhs + rhs

    ts = pd.Timestamp("11/6/2022 11:30:15", tz=representative_tz)
    offset = pd.tseries.offsets.MonthBegin(n=4, normalize=True)
    check_func(impl, (ts, offset))
    check_func(impl, (offset, ts))
    # Check normalize=False
    offset = pd.tseries.offsets.MonthBegin(n=4, normalize=False)
    check_func(impl, (ts, offset))
    check_func(impl, (offset, ts))


def test_tz_sub_month_begin(representative_tz, memory_leak_check):
    """
    Add tests for subtracting a Pandas
    MonthBeing type from TZ-Aware timezones.
    """

    def impl(lhs, rhs):
        return lhs - rhs

    ts = pd.Timestamp("11/6/2022 11:30:15", tz=representative_tz)
    offset = pd.tseries.offsets.MonthBegin(n=3, normalize=True)
    check_func(impl, (ts, offset))
    # Check normalize=False
    offset = pd.tseries.offsets.MonthBegin(n=3, normalize=False)
    check_func(impl, (ts, offset))


def test_tz_add_month_end(representative_tz, memory_leak_check):
    """
    Add tests for adding TZ-Aware timezones with a Pandas
    MonthEnd type.
    """

    def impl(lhs, rhs):
        return lhs + rhs

    ts = pd.Timestamp("11/6/2022 11:30:15", tz=representative_tz)
    offset = pd.tseries.offsets.MonthEnd(n=4, normalize=True)
    check_func(impl, (ts, offset))
    check_func(impl, (offset, ts))
    # Check normalize=False
    offset = pd.tseries.offsets.MonthEnd(n=4, normalize=False)
    check_func(impl, (ts, offset))
    check_func(impl, (offset, ts))


def test_tz_sub_month_end(representative_tz, memory_leak_check):
    """
    Add tests for subtracting a Pandas
    MonthEnd type from TZ-Aware timezones.
    """

    def impl(lhs, rhs):
        return lhs - rhs

    ts = pd.Timestamp("11/6/2022 11:30:15", tz=representative_tz)
    offset = pd.tseries.offsets.MonthEnd(n=3, normalize=True)
    check_func(impl, (ts, offset))
    # Check normalize=False
    offset = pd.tseries.offsets.MonthEnd(n=3, normalize=False)
    check_func(impl, (ts, offset))


@pytest.mark.parametrize("freq", ["D", "H", "T", "S", "ms", "L", "U", "us", "N"])
def test_timestamp_freq_methods(freq, representative_tz, memory_leak_check):
    """Tests the timestamp freq methods with various frequencies"""

    ts = pd.Timestamp("11/6/2022 11:30:15", tz=representative_tz)

    def impl1(ts, freq):
        return ts.ceil(freq)

    def impl2(ts, freq):
        return ts.floor(freq)

    def impl3(ts, freq):
        return ts.round(freq)

    check_func(impl1, (ts, freq))
    check_func(impl2, (ts, freq))
    check_func(impl3, (ts, freq))
