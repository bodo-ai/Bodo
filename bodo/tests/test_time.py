# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.testing import ensure_clean, ensure_clean_dir


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda: bodo.Time(precision=0),
            id="none",
        ),
        pytest.param(
            lambda: bodo.Time(12, precision=0),
            id="hour",
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, precision=0),
            id="minute",
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, precision=0),
            id="second",
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, precision=3),
            id="millisecond",
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, 12, precision=6),
            id="microsecond",
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, 12, 34, precision=9),
            id="nanosecond",
        ),
    ],
)
def test_time_constructor(impl, memory_leak_check):

    check_func(impl, ())


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda t: t.hour,
            id="hour",
        ),
        pytest.param(
            lambda t: t.minute,
            id="minute",
        ),
        pytest.param(
            lambda t: t.second,
            id="second",
        ),
        pytest.param(
            lambda t: t.microsecond,
            id="microsecond",
        ),
    ],
)
def test_time_extraction(impl, memory_leak_check):
    t = bodo.Time(1, 2, 3, 4, 5, 6, precision=9)

    check_func(impl, (t,))


@pytest.mark.parametrize(
    "precision,dtype",
    [
        # TODO: parquet doesn't support second precision currently
        # pytest.param(
        #     0,
        #     pa.time32("s"),
        #     id="0-time32[s]",
        # ),
        pytest.param(
            3,
            pa.time32("ms"),
            id="3-time32[ms]",
        ),
        pytest.param(
            6,
            pa.time64("us"),
            id="6-time64[us]",
        ),
        pytest.param(
            9,
            pa.time64("ns"),
            id="9-time64[ns]",
        ),
    ],
)
def test_time_arrow_conversions(precision, dtype, memory_leak_check):
    """Test the conversion between Arrow and Bodos Time types by doing the following:
    1. Test conversion from pandas df to Arrow table and check types
    2. Test writing said df to parquet
    3. Test reading parquet and checking types match the original df
    """
    fname = "time_test.pq"
    fname2 = "time_test_2.pq"
    with ensure_clean(fname), ensure_clean_dir(fname2):
        df_orig = pd.DataFrame(
            {
                "A": bodo.Time(0, 0, 0, precision=precision),
                "B": bodo.Time(1, 1, 1, precision=precision),
                "C": bodo.Time(2, 2, 2, precision=precision),
            },
            index=np.arange(3),
        )
        table_orig = pa.Table.from_pandas(
            df_orig,
            schema=pa.schema(
                [
                    pa.field("A", dtype),
                    pa.field("B", dtype),
                    pa.field("C", dtype),
                ]
            ),
        )
        pq.write_table(table_orig, fname)

        @bodo.jit(distributed=False)
        def impl():
            df = pd.read_parquet(fname)
            df.to_parquet(fname2, index=False)

        impl()

        # TODO: Because all data is loaded as ns, we can compare to the original
        # dataframe, but this should change when we support other time units.
        bodo.barrier()

        # read in bodo because of pandas type differences
        @bodo.jit(distributed=False)
        def reader():
            return pd.read_parquet(fname2)

        df = reader()

        assert df.equals(df_orig)


@pytest.mark.parametrize(
    "cmp_fn",
    [
        pytest.param(
            lambda a, b: a == b,
            id="op_eq",
        ),
        pytest.param(
            lambda a, b: a != b,
            id="op_not_eq",
        ),
        pytest.param(
            lambda a, b: a < b,
            id="op_lt",
        ),
        pytest.param(
            lambda a, b: a <= b,
            id="op_le",
        ),
        pytest.param(
            lambda a, b: a > b,
            id="op_gt",
        ),
        pytest.param(
            lambda a, b: a >= b,
            id="op_ge",
        ),
    ],
)
@pytest.mark.parametrize(
    "a,b",
    [
        pytest.param(
            bodo.Time(1, 15, 12, 0, precision=3),
            bodo.Time(1, 15, 12, 0, precision=3),
            id="data_eq",
        ),
        pytest.param(
            bodo.Time(1, 15, 12, 0, precision=3),
            bodo.Time(1, 15, 12, 1, precision=3),
            id="data_lt",
        ),
        pytest.param(
            bodo.Time(1, 15, 12, 1, precision=3),
            bodo.Time(1, 15, 12, 0, precision=3),
            id="data_gt",
        ),
    ],
)
def test_time_cmp(cmp_fn, a, b, memory_leak_check):
    check_func(cmp_fn, (a, b))


def test_time_sort(memory_leak_check):
    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0),
                bodo.Time(1, 1, 3, 1),
                bodo.Time(2),
            ]
        }
    )

    def impl(df):
        df.sort_values(by="A")
        return df

    check_func(impl, (df,))


def test_time_merge(memory_leak_check):
    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0),
                bodo.Time(1, 1, 3, 1),
                bodo.Time(2),
            ]
        }
    )

    df2 = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0),
                bodo.Time(1, 1, 3, 1),
                bodo.Time(2),
            ]
        }
    )

    def impl(df, df2):
        df.merge(df2, how="inner", on="A")
        return df

    check_func(impl, (df, df2))


def test_time_head(memory_leak_check):
    df = pd.DataFrame({"A": [bodo.Time(x) for x in range(10)]})

    def impl(df):
        return df.head()

    check_func(impl, (df,))


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda dt: bodo.Time(dt.hour, precision=0),
            id="hour",
        ),
        pytest.param(
            lambda dt: bodo.Time(dt.hour, dt.minute, precision=0),
            id="minute",
        ),
        pytest.param(
            lambda dt: bodo.Time(dt.hour, dt.minute, dt.second, precision=0),
            id="second",
        ),
        pytest.param(
            lambda dt: bodo.Time(
                dt.hour, dt.minute, dt.second, dt.millisecond, precision=3
            ),
            id="millisecond",
        ),
        pytest.param(
            lambda dt: bodo.Time(
                dt.hour,
                dt.minute,
                dt.second,
                dt.millisecond,
                dt.microsecond,
                precision=6,
            ),
            id="microsecond",
        ),
        pytest.param(
            lambda dt: bodo.Time(
                dt.hour,
                dt.minute,
                dt.second,
                dt.millisecond,
                dt.microsecond,
                dt.nanosecond,
                precision=9,
            ),
            id="nanosecond",
        ),
    ],
)
@pytest.mark.parametrize(
    "dt",
    [
        pytest.param(
            bodo.Time(precision=9),
            id="none",
        ),
        pytest.param(
            bodo.Time(12, precision=9),
            id="hour",
        ),
        pytest.param(
            bodo.Time(12, 30, precision=9),
            id="minute",
        ),
        pytest.param(
            bodo.Time(12, 30, 42, precision=9),
            id="second",
        ),
        pytest.param(
            bodo.Time(12, 30, 42, 64, precision=9),
            id="millisecond",
        ),
        pytest.param(
            bodo.Time(12, 30, 42, 64, 43, precision=9),
            id="microsecond",
        ),
        pytest.param(
            bodo.Time(12, 30, 42, 64, 43, 58, precision=9),
            id="nanosecond",
        ),
    ],
)
def test_time_construction_from_parts(impl, dt, memory_leak_check):
    """Test that time can be constructed from parts of a time.
    Needed for SQL `TRUNC` and `TIME_SLICE` functionality.
    """

    check_func(impl, (dt,))


def test_time_array_setitem_none(memory_leak_check):
    df = pd.DataFrame({"A": [bodo.Time(x) for x in range(10)]})

    def impl(df):
        df["A"][0] = None
        return df

    check_func(impl, (df,))
