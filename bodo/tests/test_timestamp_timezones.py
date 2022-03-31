# Copyright (C) 2022 Bodo Inc. All rights reserved.

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
