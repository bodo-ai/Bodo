"""Checks for functionality on DataFrames containing timezone values."""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.timezone_common import representative_tz_or_none  # noqa
from bodo.tests.utils import check_func, pytest_pandas

pytestmark = pytest_pandas


def test_pd_concat_df(memory_leak_check):
    """
    Tests pd.Concat on DataFrame Arguments with tz_aware
    Data.
    """

    def impl1(df1, df2):
        return pd.concat([df1, df2])

    def impl2(df1, df2):
        return pd.concat((df1, df2))

    def impl3(df1, df2):
        df = pd.concat((df1, df2), axis=1)
        # Keep names consistent
        df.columns = ["A", "B", "C", "D"]
        return df

    def impl4(df, S1, S2):
        df = pd.concat((df, S1, S2), axis=1)
        # Keep names consistent
        df.columns = ["A", "B", "C", "D"]
        return df

    S1 = (
        pd.date_range(
            start="1/1/2022",
            freq="16D5h",
            periods=30,
            tz="Poland",
            unit="ns",
        )
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="UTC", unit="ns")
        .to_series()
        .reset_index(drop=True)
    )
    df1 = pd.DataFrame(
        {
            "A": S1,
            "B": S2,
        }
    )
    # Add another DataFrame to test gen_na_array
    df2 = pd.DataFrame(
        {
            "C": S1,
            "D": S2,
        }
    )
    # pd.concat doesn't match the order of Pandas across multiple ranks
    check_func(impl1, (df1, df1), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df1), sort_output=True, reset_index=True)
    check_func(impl2, (df1, df2), sort_output=True, reset_index=True)
    check_func(impl3, (df1, df1), sort_output=True, reset_index=True)
    check_func(
        impl4, (df1, S1.rename("q"), S2.rename("w")), sort_output=True, reset_index=True
    )


def test_df_dtypes(memory_leak_check, representative_tz_or_none):
    """
    Tests support for DataFrames.dtypes with various timezone types.
    """
    if representative_tz_or_none is None:
        pytest.skip("TODO: match datetime64[ns] dtype behavior with new numpy")

    def impl(df):
        return df.dtypes

    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz=representative_tz_or_none,
                unit="ns",
            ).to_series(),
            "B": [1.2, 1.5, 1.6] * 10,
        }
    )
    check_func(impl, (df,), only_seq=True)


def test_df_dtypes_astype(memory_leak_check, representative_tz_or_none):
    """
    Tests support for astype using DataFrames.dtypes and casting to the same
    type. This is meant to emulate when the tz-aware type is unchanged but other
    types are changed.
    """

    def impl(df):
        return df.astype(df.dtypes, copy=False)

    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz=representative_tz_or_none,
                unit="ns",
            ).to_series(),
            "B": [1.2, 1.5, 1.6] * 10,
        }
    )
    check_func(impl, (df,))


def test_pd_concat_dataframe_error(memory_leak_check):
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
            unit="ns",
        )
        .to_series()
        .reset_index(drop=True)
    )
    S2 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, tz="UTC", unit="ns")
        .to_series()
        .reset_index(drop=True)
    )
    S3 = (
        pd.date_range(start="1/1/2022", freq="16D5h", periods=30, unit="ns")
        .to_series()
        .reset_index(drop=True)
    )
    df1 = pd.DataFrame(
        {
            "A": S1,
            "B": S2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": S2,
            "B": S2,
        }
    )
    df3 = pd.DataFrame(
        {
            "A": S1,
            "B": S3,
        }
    )

    with pytest.raises(
        BodoError,
        match="Cannot concatenate the rows of Timestamp data with different timezones",
    ):
        impl(df1, df2)
    with pytest.raises(
        BodoError,
        match="Cannot concatenate the rows of Timestamp data with different timezones",
    ):
        impl(df1, df3)


def test_tz_aware_unsupported(memory_leak_check):
    """Test that a tz-naive values cannot be assigned to tz-aware series"""
    from bodo.utils.typing import BodoError

    def impl(df, value):
        df[0] = value

    with pytest.raises(
        BodoError,
        match=".*setitem with DatetimeArrayType requires a Timestamp value or DatetimeArrayType.*",
    ):
        bodo.jit(impl)(
            pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz="Poland",
                unit="ns",
            ).to_series(),
            np.datetime64("2023-01-01"),
        )
