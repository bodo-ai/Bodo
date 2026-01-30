"""Checks for functionality on Series containing timezone values."""

import pandas as pd
import pytest

import bodo
from bodo.tests.timezone_common import representative_tz, sample_tz  # noqa: F401
from bodo.tests.utils import check_func, generate_comparison_ops_func

pytestmark = pytest.mark.tz_aware


def test_tz_series_tz_scalar_comparison(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the tz-aware series and a tz-aware scalar with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    S = pd.date_range(
        start="1/1/2022", freq="16D5h", periods=30, tz="Poland"
    ).to_series()
    ts = pd.Timestamp("4/4/2022", tz="Poland")
    check_func(func, (S, ts))
    check_func(func, (ts, S))


def test_tz_series_tz_array_comparison(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the tz-aware series and a tz-aware array with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    S = pd.date_range(
        start="1/1/2022", freq="16D5h", periods=30, tz="Poland"
    ).to_series()
    arr = pd.date_range(
        start="2/1/2022", freq="8D2h30min", periods=30, tz="Poland"
    ).array
    check_func(func, (S, arr))
    check_func(func, (arr, S))


def test_tz_series_tz_series_comparison(cmp_op, memory_leak_check):
    """Check that comparison operators work between
    the tz-aware series and a tz-aware series with the same timezone.
    """
    func = generate_comparison_ops_func(cmp_op)
    S1 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="Poland")
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="2/1/2022", freq="8D2h30min", periods=30, tz="Poland")
        .to_series()
        .reset_index(drop=True)
    )
    check_func(func, (S1, S2))
    check_func(func, (S2, S1))


def test_series_scalar_different_tz_unsupported(cmp_op, memory_leak_check):
    """Check that comparison operators throw exceptions between
    the 2 Series with different timezones.
    """
    from bodo.utils.typing import BodoError

    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    S = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="Poland")
        .to_series()
        .reset_index(drop=True)
    )
    ts1 = pd.Timestamp("4/4/2022")
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S, ts1)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts1, S)
    # Check different timezones aren't supported
    ts2 = pd.Timestamp("4/4/2022", tz="US/Pacific")
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S, ts2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(ts2, S)


def test_series_array_different_tz_unsupported(cmp_op, memory_leak_check):
    """Check that comparison operators throw exceptions between
    the 2 Series with different timezones.
    """
    from bodo.utils.typing import BodoError

    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    S = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="Poland")
        .to_series()
        .reset_index(drop=True)
    )
    arr1 = (
        pd.date_range(start="2/1/2022", freq="8D2h30min", periods=30).to_series().array
    )
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S, arr1)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr1, S)
    # Check different timezones aren't supported
    arr2 = pd.date_range(
        start="2/1/2022", freq="8D2h30min", periods=30, tz="US/Pacific"
    ).array
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S, arr2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(arr2, S)


def test_series_different_tz_unsupported(cmp_op, memory_leak_check):
    """Check that comparison operators throw exceptions between
    the 2 Series with different timezones.
    """
    from bodo.utils.typing import BodoError

    func = bodo.jit(generate_comparison_ops_func(cmp_op))
    S1 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="Poland")
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="2/1/2022", freq="8D2h30min", periods=30)
        .to_series()
        .reset_index(drop=True)
    )
    # Check that comparison is not support between tz-aware and naive
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S1, S2)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S2, S1)
    # Check different timezones aren't supported
    S3 = (
        pd.date_range(start="2/1/2022", freq="8D2h30min", periods=30, tz="US/Pacific")
        .to_series()
        .reset_index(drop=True)
    )
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S1, S3)
    with pytest.raises(
        BodoError, match="requires both Timestamps share the same timezone"
    ):
        func(S3, S1)


def test_dt_tz_convert_none(memory_leak_check):
    """
    Test Series.dt.tz_convert with argument None on a timezone aware Series.
    """

    def impl(S):
        return S.dt.tz_convert(None)

    S = pd.date_range(
        start="1/1/2022", freq="16D5h", periods=30, tz="Poland"
    ).to_series()
    check_func(impl, (S,))


def test_pd_concat_series(memory_leak_check):
    """
    Tests pd.Concat on two Series Arguments with tz_aware
    Data.
    """

    def impl1(S1, S2):
        return pd.concat([S1, S2])

    def impl2(S1, S2):
        return pd.concat((S1, S2))

    def impl3(S1, S2):
        df = pd.concat((S1, S2), axis=1)
        # Rename to avoid issues
        df.columns = ["A", "B"]
        return df

    S1 = (
        pd.date_range(
            start="1/1/2022",
            freq="16D5h",
            periods=30,
            tz="Poland",
        )
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="UTC")
        .to_series()
        .reset_index(drop=True)
    )
    # pd.concat doesn't match the order of Pandas across multiple ranks
    check_func(impl1, (S1, S1), sort_output=True, reset_index=True)
    check_func(impl2, (S1, S1), sort_output=True, reset_index=True)
    check_func(impl3, (S1, S1), sort_output=True, reset_index=True)
    # Different timezones should be allowed because they become different columns.
    check_func(impl3, (S1, S2), sort_output=True, reset_index=True)


def test_series_dtype(memory_leak_check, representative_tz):
    """
    Tests support for Series.dtype on various timezone types.
    """

    def impl(S):
        return S.dtype

    S = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz=representative_tz)
        .to_series()
        .astype(f"datetime64[ns, {representative_tz}]")
    )
    check_func(impl, (S,))


def test_pd_concat_series_error(memory_leak_check):
    """Tests trying to concatenate rows of a Series
    with different Timezones throw reasonable errors.
    """
    from bodo.utils.typing import BodoError

    @bodo.jit
    def impl(S1, S2):
        return pd.concat((S1, S2))

    S1 = (
        pd.date_range(
            start="1/1/2022",
            freq="16D5h",
            periods=30,
            tz="Poland",
        )
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="UTC")
        .to_series()
        .reset_index(drop=True)
    )
    S3 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30)
        .to_series()
        .reset_index(drop=True)
    )
    with pytest.raises(
        BodoError,
        match="Cannot concatenate the rows of Timestamp data with different timezones",
    ):
        impl(S1, S2)
    with pytest.raises(
        BodoError,
        match="Cannot concatenate the rows of Timestamp data with different timezones",
    ):
        impl(S1, S3)


def test_series_shift(memory_leak_check, sample_tz):
    """
    Tests the functionality of Series.shift with tz-aware data.
    """

    def impl1(S, shift_amount):
        return S.shift(shift_amount)

    def impl2(S, shift_amount, fill_value):
        return S.shift(shift_amount, fill_value=fill_value)

    data = (
        [None] * 4
        + list(pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz=sample_tz))
        + [None] * 4
    )
    shift_amount = 3
    fill_value = pd.Timestamp("2019-1-24", tz=sample_tz)
    S = pd.Series(data)
    check_func(impl1, (S, shift_amount))
    check_func(impl2, (S, shift_amount, fill_value))


@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
def test_series_dt_freq_methods(freq, representative_tz, memory_leak_check):
    """Tests the Series.dt freq methods with various frequencies"""

    S = pd.Series(
        [None] * 4
        + list(
            pd.date_range(
                start="1/1/2022 4:31:15.4814",
                freq="16D5h",
                periods=30,
                tz=representative_tz,
            )
        )
        + [None] * 4
    )

    def impl1(S, freq):
        return S.dt.ceil(freq)

    def impl2(S, freq):
        return S.dt.floor(freq)

    def impl3(S, freq):
        return S.dt.round(freq)

    check_func(impl1, (S, freq))
    check_func(impl2, (S, freq))
    check_func(impl3, (S, freq))
