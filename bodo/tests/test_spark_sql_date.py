# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests of series.map/apply and dataframe.apply used for parity
with pyspark.sql.functions that operation on columns of dates.

Test names refer to the names of the spark function they map to.
"""

import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pd.DataFrame(
            {"A": pd.date_range(start="2018-04-24", end="2020-04-29", periods=5)}
        ),
        pd.DataFrame(
            {"A": pd.date_range(start="2018-04-24", end="2020-04-29", periods=5).date}
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.mark.slow
def test_add_months(dataframe_val, memory_leak_check):
    def test_impl(df, num_months):
        return df.A.apply(
            lambda x, num_months: x + pd.DateOffset(months=num_months),
            num_months=num_months,
        )

    check_func(test_impl, (dataframe_val, 3))
    check_func(test_impl, (dataframe_val, 0))
    check_func(test_impl, (dataframe_val, -2))


@pytest.mark.slow
def test_date_add(dataframe_val, memory_leak_check):
    def test_impl(df, num_days):
        return df.A.apply(
            lambda x, num_days: x + pd.DateOffset(num_days),
            num_days=num_days,
        )

    check_func(test_impl, (dataframe_val, 3))
    check_func(test_impl, (dataframe_val, 0))
    check_func(test_impl, (dataframe_val, -2))


@pytest.mark.slow
def test_date_sub(dataframe_val, memory_leak_check):
    def test_impl(df, num_days):
        return df.A.apply(
            lambda x, num_days: x - pd.DateOffset(num_days),
            num_days=num_days,
        )

    check_func(test_impl, (dataframe_val, 3))
    check_func(test_impl, (dataframe_val, 0))
    check_func(test_impl, (dataframe_val, -2))


@pytest.mark.slow
def test_from_unixtime(memory_leak_check):
    def test_impl(S, format_str):
        return S.map(lambda x: pd.Timestamp(x, "s")).dt.strftime(format_str)

    S = (
        pd.date_range(start="1/1/2018", end="1/08/2018", freq="h")
        .to_series()
        .map(lambda x: int(x.value / (1000 * 1000 * 1000)))
    )
    check_func(test_impl, (S, "%B %d, %Y, %r"))


@pytest.mark.slow
def test_last_day(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: x + pd.tseries.offsets.MonthEnd())

    check_func(test_impl, (dataframe_val,))


@pytest.mark.skip("No Support for Week")
def test_next_day(dataframe_val, memory_leak_check):
    def test_impl(df, dayOfWeek):
        return df.A.apply(
            lambda x, dayOfWeek: x + pd.tseries.offsets.Week(0, weekday=dayOfWeek),
            dayOfWeek=dayOfWeek,
        )

    check_func(test_impl, (dataframe_val, 3))
    check_func(test_impl, (dataframe_val, 0))
    check_func(test_impl, (dataframe_val, 6))
