# Copyright (C) 2022 Bodo Inc. All rights reserved.


import pandas as pd
import pytest

from bodo.tests.utils import check_func

_timestamp_strs = [
    "2019-01-01",
    "2020-01-01",
    "2030-01-01",
]

_timezones = [
    "UTC",
    "US/Eastern",
    "US/Pacific",
    "Europe/Berlin",
]

_dt_arrs = [
    pytest.param(
        pd.array([pd.Timestamp("2000-01-01", tz=timezone)] * 10),
        id=f"single-{timezone}",
    )
    for timezone in _timezones
] + [
    pytest.param(
        pd.array(
            [
                pd.Timestamp(timestamp_str, tz=timezone)
                for timestamp_str in _timestamp_strs
            ]
            * 5
        ),
        id=f"multiple-{timezone}",
    )
    for timezone in _timezones
]


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_boxing(arr, memory_leak_check):
    def test_impl(arr):
        return arr

    check_func(test_impl, (arr,))


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_len(arr):
    def test_impl(arr):
        return len(arr)

    check_func(test_impl, (arr,))


@pytest.mark.skip(reason="Lowering for `pd.DatetimeArray` not implemented yet")
@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_constant_lowering(arr, memory_leak_check):
    def test_impl():
        return arr

    check_func(test_impl, ())


@pytest.mark.skip(reason="Constructor not implemented yet")
@pytest.mark.parametrize("values", _dt_arrs)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(pd.DatetimeTZDtype(tz=timezone), id=timezone)
        for timezone in _timezones
    ],
)
def test_pd_datetime_arr_constructor(values, dtype, memory_leak_check):
    def test_impl(values, dtype):
        return pd.arrays.DatetimeArray(values, dtype=dtype)

    check_func(test_impl, (values, dtype))


@pytest.mark.skip(reason="Construction from `pd.array` not implemented yet")
@pytest.mark.parametrize(
    "timestamp_list",
    [
        pytest.param([pd.Timestamp("2000-01-01", tz=timezone)], id=f"single-{timezone}")
        for timezone in _timezones
    ]
    + [
        pytest.param(
            [
                pd.Timestamp(timestamp_str, tz=timezone)
                for timestamp_str in _timestamp_strs
            ],
            id=f"multiple-{timezone}",
        )
        for timezone in _timezones
    ],
)
def test_pd_datetime_arr_from_pd_array(timestamp_list, memory_leak_check):
    def test_impl(timestamp_list):
        return pd.array(timestamp_list)

    check_func(test_impl, (timestamp_list,))


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_index_conversion(arr, memory_leak_check):
    def test_impl(idx):
        return idx

    check_func(test_impl, (pd.DatetimeIndex(arr),))


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_series_dt_conversion(arr, memory_leak_check):
    def test_impl(idx):
        return idx

    check_func(test_impl, (pd.Series(arr),))


@pytest.mark.parametrize("arr", _dt_arrs)
@pytest.mark.parametrize("timezone", _timezones)
def test_pd_datetime_arr_tz_convert(arr, timezone, memory_leak_check):
    def test_impl(arr, timezone):
        return arr.tz_convert(timezone)

    check_func(test_impl, (arr, timezone))


@pytest.mark.parametrize("arr", _dt_arrs)
@pytest.mark.parametrize("timezone", _timezones)
def test_pd_datetime_index_tz_convert(arr, timezone, memory_leak_check):
    def test_impl(idx, timezone):
        return idx.tz_convert(timezone)

    check_func(test_impl, (pd.DatetimeIndex(arr), timezone))


@pytest.mark.parametrize("arr", _dt_arrs)
@pytest.mark.parametrize("timezone", _timezones)
def test_pd_datetime_arr_series_dt_tz_convert(arr, timezone, memory_leak_check):
    def test_impl(s, timezone):
        return s.dt.tz_convert(timezone)

    check_func(test_impl, (pd.Series(arr), timezone))
