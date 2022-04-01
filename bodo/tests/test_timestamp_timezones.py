# Copyright (C) 2022 Bodo Inc. All rights reserved.

import re

import numpy as np
import pandas as pd
import pytest
import pytz

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


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


@pytest.fixture(params=pytz.all_timezones)
def all_tz(request):
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


def test_timestamp_tz_convert(all_tz):
    def test_impl(ts, tz):
        return ts.tz_convert(tz=tz)

    ts = pd.Timestamp("09-30-2020", tz="Poland")
    check_func(
        test_impl,
        (
            ts,
            all_tz,
        ),
    )

    ts = pd.Timestamp("09-30-2020")
    with pytest.raises(
        BodoError,
        match="Cannot convert tz-naive Timestamp, use tz_localize to localize",
    ):
        bodo.jit(test_impl)(ts, all_tz)


def test_timestamp_tz_localize(all_tz):
    def test_impl(ts, tz):
        return ts.tz_localize(tz=tz)

    ts = pd.Timestamp("09-30-2020 14:00")
    check_func(
        test_impl,
        (
            ts,
            all_tz,
        ),
    )


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


def test_tz_timestamp_unsupported():
    def impl(ts):
        return max(ts, ts)

    non_tz_ts = pd.Timestamp("2020-01-01")
    tz_ts = pd.Timestamp("2020-01-01", tz="US/Eastern")

    check_func(impl, (non_tz_ts,))

    with pytest.raises(
        BodoError,
        match=".*Timezone-aware timestamp not yet supported.*",
    ):
        bodo.jit(impl)(tz_ts)


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


def test_tz_series_unsupported():
    def impl(s):
        return s.dtype

    non_tz_s = pd.Series([pd.Timestamp(f"2020-01-0{i}") for i in range(1, 10)])
    tz_s = pd.Series(
        [pd.Timestamp(f"2020-01-0{i}", tz="US/Eastern") for i in range(1, 10)]
    )

    check_func(impl, (non_tz_s,))

    with pytest.raises(
        BodoError,
        match=".*Timezone-aware series not yet supported.*",
    ):
        bodo.jit(impl)(tz_s)


def test_tz_dataframe_unsupported(memory_leak_check):
    def impl(df):
        return df.astype("int64")

    non_tz_df = pd.DataFrame(
        {"a": [pd.Timestamp("2020-01-01")] * 10},
    )
    tz_df = pd.DataFrame(
        {"a": [pd.Timestamp("2020-01-01", tz="US/Eastern")] * 10},
    )

    check_func(impl, (non_tz_df,))

    with pytest.raises(
        BodoError,
        match=".*Timezone-aware columns not yet supported.*",
    ):
        bodo.jit(impl)(tz_df)
