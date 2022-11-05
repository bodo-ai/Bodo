# Copyright (C) 2022 Bodo Inc. All rights reserved.

import datetime

import pandas as pd
import pytest

import bodo
from bodo.tests.timezone_common import sample_tz  # noqa
from bodo.tests.utils import check_func, generate_comparison_ops_func
from bodo.utils.typing import BodoError

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

_dt_na_arrs = [
    pytest.param(
        pd.array(
            [
                pd.Timestamp("2000-01-01", tz=timezone),
                pd.Timestamp("2001-01-01", tz=timezone),
                pd.NaT,
                pd.Timestamp("2002-01-01", tz=timezone),
            ]
            * 10
        ),
        id=f"na-{timezone}",
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


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_datetime_arr_constant_lowering(arr, memory_leak_check):
    def test_impl():
        return arr

    check_func(test_impl, (), dist_test=False)


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_tz_series_constant_lowering(arr, memory_leak_check):
    S = pd.Series(arr)

    def test_impl():
        return S

    check_func(test_impl, (), dist_test=False)


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_tz_index_constant_lowering(arr, memory_leak_check):
    I = pd.Index(arr)

    def test_impl():
        return I

    check_func(test_impl, (), dist_test=False)


@pytest.mark.parametrize("arr", _dt_arrs)
def test_pd_tz_df_constant_lowering(arr, memory_leak_check):
    df = pd.DataFrame({"A": arr})

    def test_impl():
        return df

    check_func(test_impl, (), dist_test=False)


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


@pytest.mark.parametrize("arr", _dt_arrs + _dt_na_arrs)
def test_pd_datetime_arr_isna(arr, memory_leak_check):
    def test_impl(arr):
        return pd.isna(arr)

    check_func(test_impl, (arr,))


def test_dt_str_method_tz(memory_leak_check):
    """
    Test Series.dt methods that are supported with timezone-aware data.
    """

    def impl1(S):
        return S.dt.day_name()

    def impl2(S):
        return S.dt.month_name()

    def impl3(S):
        return S.dt.date

    def impl4(S):
        return S.dt.strftime("%m/%d/%Y, %H:%M:%S")

    S = pd.Series(
        pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland")
    )
    check_func(impl1, (S,))
    check_func(impl2, (S,))
    check_func(impl3, (S,))
    check_func(impl4, (S,))


def test_series_value_tz(memory_leak_check):
    """
    Tests using Series.values in JIT with tz-aware series.
    """

    def impl1(S):
        return S.values

    S = pd.Series(
        pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland")
    )
    check_func(impl1, (S,))


def test_nbytes_arr_seq(memory_leak_check):
    """
    Test using array.nbytes inside JIT. We just do a sequential test.
    """

    @bodo.jit
    def impl1(arr):
        return arr.nbytes

    arr = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    nbytes = impl1(arr)
    # Array is just a 64-bit integer array
    assert nbytes == (len(arr) * 8), "JIT nbytes doesn't match expected value"


def test_tz_index_getitem(memory_leak_check):
    """
    Test integer getitem with a datetime index
    """

    def impl(arr):
        return I[17]

    I = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland")
    check_func(impl, (I,))


def test_tz_index_fields(memory_leak_check):
    """
    Test fields with a datetime index
    """

    def impl1(I):
        return I.year

    def impl2(I):
        return I.hour

    def impl3(I):
        return I.is_leap_year

    def impl4(I):
        return I.date

    def impl5(I):
        return I.weekday

    I = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland")
    check_func(impl1, (I,))
    check_func(impl2, (I,))
    check_func(impl3, (I,))
    check_func(impl4, (I,))
    check_func(impl5, (I,))


def test_tz_index_isocalendar(memory_leak_check):
    """
    Test methods with a datetime index
    """

    def impl(I):
        return I.isocalendar()

    I = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland")
    check_func(impl, (I,))


def test_setna_compiles(memory_leak_check):
    """
    Tests that bodo.libs.array_kernels.setna compiles.
    In the future this should be replaced with a test that
    checks functionality.
    """

    @bodo.jit
    def impl(arr):
        bodo.libs.array_kernels.setna(arr, 8)

    arr = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    impl(arr)


def test_tz_array_tz_scalar_comparison(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the tz-aware array and a tz-aware scalar with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    arr = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    ts = pd.Timestamp("4/4/2022", tz="Poland")
    check_func(func, (arr, ts))
    check_func(func, (ts, arr))


def test_scalar_different_tz_unsupported(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the tz-aware array and a tz-aware scalar with the same timezone.
    """
    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    arr = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    ts1 = pd.Timestamp("4/4/2022")
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr, ts1)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts1, arr)
    # Check different timezones aren't supported
    ts2 = pd.Timestamp("4/4/2022", tz="US/Pacific")
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr, ts2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts2, arr)


def test_tz_array_date_scalar_comparison(sample_tz, cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the timestamp array and a date scalar.
    """
    func = generate_comparison_ops_func(cmp_op)
    d_range = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz=sample_tz)
    if sample_tz is None:
        arr = d_range.values
    else:
        arr = d_range.array
    d = datetime.date(2022, 4, 4)
    # Use the casting for the generated output.
    d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
    # This isn't supported in pandas so generate a py output.
    py_output = pd.array([False] * len(arr), dtype="boolean")
    for i, elem in enumerate(arr):
        # Wrap elem in pd.Timestamp to protect against dt64 array.
        py_output[i] = cmp_op(pd.Timestamp(elem), d_ts)
    check_func(func, (arr, d), py_output=py_output)
    # Update the py_output
    for i, elem in enumerate(arr):
        py_output[i] = cmp_op(d_ts, pd.Timestamp(elem))
    check_func(func, (d, arr), py_output=py_output)


def test_tz_array_tz_array_comparison(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    two tz-aware arrays with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    arr1 = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    arr2 = pd.date_range(
        start="2/1/2022", freq="8D2H30T", periods=30, tz="Poland"
    ).array
    check_func(func, (arr1, arr2))
    check_func(func, (arr2, arr1))


def test_tz_array_date_array_comparison(sample_tz, cmp_op, memory_leak_check):
    """Check that comparison operators work between
    a tz-aware array and a date array.
    """
    func = generate_comparison_ops_func(cmp_op)
    d_range = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz=sample_tz)
    if sample_tz is None:
        arr1 = d_range.values
    else:
        arr1 = d_range.array
    arr2 = (
        pd.date_range(start="2/1/2022", freq="8D2H30T", periods=30, tz=sample_tz)
        .to_series()
        .dt.date.values
    )

    # This isn't supported in pandas so generate a py output.
    py_output = pd.array([False] * len(arr1), dtype="boolean")
    for i in range(len(arr1)):
        d = arr2[i]
        d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
        py_output[i] = cmp_op(pd.Timestamp(arr1[i]), d_ts)
    check_func(func, (arr1, arr2), py_output=py_output)
    for i in range(len(arr1)):
        d = arr2[i]
        d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
        py_output[i] = cmp_op(d_ts, pd.Timestamp(arr1[i]))
    check_func(func, (arr2, arr1), py_output=py_output)


def test_tz_array_date_series_comparison(sample_tz, cmp_op, memory_leak_check):
    """Check that comparison operators work between
    a tz-aware array and a date array.
    """
    func = generate_comparison_ops_func(cmp_op)
    d_range = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz=sample_tz)
    if sample_tz is None:
        arr = d_range.values
    else:
        arr = d_range.array
    S = (
        pd.date_range(start="2/1/2022", freq="8D2H30T", periods=30, tz=sample_tz)
        .to_series()
        .reset_index(drop=True)
        .dt.date
    )

    # This isn't supported in pandas so generate a py output.
    py_output = pd.array([False] * len(arr), dtype="boolean")
    for i in range(len(arr)):
        d = S[i]
        d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
        py_output[i] = cmp_op(pd.Timestamp(arr[i]), d_ts)
    check_func(func, (arr, S), py_output=pd.Series(py_output))
    for i in range(len(arr)):
        d = S[i]
        d_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, tz=sample_tz)
        py_output[i] = cmp_op(d_ts, pd.Timestamp(arr[i]))
    check_func(func, (S, arr), py_output=pd.Series(py_output))


def test_array_different_tz_unsupported(cmp_op, memory_leak_check):
    """Check that comparison operators throw exceptions between
    the 2 arrays with different timezones.
    """
    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    arr1 = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    arr2 = pd.date_range(start="2/1/2022", freq="8D2H30T", periods=30).values
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr1, arr2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr2, arr1)
    # Check different timezones aren't supported
    arr3 = pd.date_range(
        start="2/1/2022", freq="8D2H30T", periods=30, tz="US/Pacific"
    ).array
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr1, arr3)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr3, arr1)


def test_tz_convert_none(memory_leak_check):
    """
    Test tz_convert with argument None on a timezone aware array.
    """

    def impl(arr):
        return arr.tz_convert(None)

    arr = pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland").array
    # Python will have a different output type until we handle no timezone in Datetime array
    py_output = arr.tz_convert(None)._data
    check_func(impl, (arr,), py_output=py_output)
