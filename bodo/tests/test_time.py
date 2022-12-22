# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.testing import ensure_clean


@pytest.mark.parametrize(
    "str_time,int_time",
    [
        pytest.param(
            bodo.time_from_str("12:34:56", precision=0),
            bodo.Time(12, 34, 56, precision=0),
            id="second_str",
        ),
        pytest.param(
            bodo.time_from_str("12:34:56.789", precision=3),
            bodo.Time(12, 34, 56, 789, precision=3),
            id="millisecond_str",
        ),
        pytest.param(
            bodo.time_from_str("12:34:56.789123", precision=6),
            bodo.Time(12, 34, 56, 789, 123, precision=6),
            id="microsecond_str",
        ),
        pytest.param(
            bodo.time_from_str("12:34:56.789123456", precision=9),
            bodo.Time(12, 34, 56, 789, 123, 456, precision=9),
            id="nanosecond_str",
        ),
        pytest.param(
            bodo.time_from_str("12:34:56.789123456"),
            bodo.Time(12, 34, 56, 789, 123, 456),
            id="non_specified_str",
        ),
    ],
)
def test_time_python_constructor(str_time, int_time):
    assert str_time == int_time


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
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, precision=0),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, precision=0),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, precision=3),
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, 12, precision=6),
            id="microsecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.Time(12, 34, 56, 78, 12, 34, precision=9),
            id="nanosecond",
        ),
        pytest.param(
            lambda: bodo.time_from_str("12:34:56", precision=0),
            id="second_str",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.time_from_str("12:34:56.789", precision=3),
            id="millisecond_str",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.time_from_str("12:34:56.789123", precision=6),
            id="microsecond_str",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.time_from_str("12:34:56.789123456", precision=9),
            id="nanosecond_str",
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
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda t: t.minute,
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda t: t.second,
            id="second",
            marks=pytest.mark.slow,
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


@pytest.mark.slow
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

    df_orig = pd.DataFrame(
        {
            "A": bodo.Time(0, 0, 0, precision=precision),
            "B": bodo.Time(1, 1, 1, precision=precision),
            "C": bodo.Time(2, 2, 2, precision=precision),
        },
        index=np.arange(3),
    )

    if bodo.get_rank() == 0:
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

    bodo.barrier()

    with ensure_clean(fname), ensure_clean(fname2):

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
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda a, b: a != b,
            id="op_not_eq",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda a, b: a < b,
            id="op_lt",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda a, b: a <= b,
            id="op_le",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda a, b: a > b,
            id="op_gt",
            marks=pytest.mark.slow,
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
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(1, 15, 12, 0, precision=3),
            bodo.Time(1, 15, 12, 1, precision=3),
            id="data_lt",
            marks=pytest.mark.slow,
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


@pytest.mark.slow
@pytest.mark.parametrize("precision", [0, 3, 6, 9])
def test_time_sort(precision, memory_leak_check):
    """Test sort by a Time column

    Args:
        precision (int): Time precision argument
        memory_leak_check (fixture function): check memory leak in the test.

    """
    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0, precision=precision),
                bodo.Time(1, 1, 3, 1, precision=precision),
                None,
                bodo.Time(2, precision=precision),
                bodo.Time(15, 0, 50, 10, 100, precision=precision),
                bodo.Time(9, 1, 3, 10, precision=precision),
                None,
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
            ]
        }
    )

    def impl(df):
        return df.sort_values(by="A")

    check_func(impl, (df,), reset_index=True)


@pytest.mark.parametrize("precision", [0, 3, 6, 9])
def test_time_merge(precision, memory_leak_check):
    """Test join on a Time column

    Args:
        precision (int): Time precision argument
        memory_leak_check (fixture function): check memory leak in the test.

    """
    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0, precision=precision),
                bodo.Time(1, 1, 3, 1, precision=precision),
                bodo.Time(2, precision=precision),
                bodo.Time(15, 0, 50, 10, 100, precision=precision),
                bodo.Time(9, 1, 3, 10, precision=precision),
                None,
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
            ],
            "B": [
                None,
                bodo.Time(12, 0, precision=precision),
                bodo.Time(1, 11, 3, 1, precision=precision),
                bodo.Time(2, precision=precision),
                bodo.Time(14, 0, 50, 10, 100, precision=precision),
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
                bodo.Time(9, 1, 30, 10, precision=precision),
            ],
        }
    )

    df2 = pd.DataFrame(
        {
            "A": [
                None,
                bodo.Time(12, 0, precision=precision),
                bodo.Time(1, 1, 3, 1, precision=precision),
                bodo.Time(2, precision=precision),
                bodo.Time(1, 10, precision=precision),
                None,
                bodo.Time(1, 11, 30, 100, precision=precision),
                bodo.Time(12, precision=precision),
            ],
            "D": [
                bodo.Time(11, 0, precision=precision),
                None,
                bodo.Time(6, 11, 3, 1, precision=precision),
                bodo.Time(9, precision=precision),
                bodo.Time(14, 10, 50, 10, 100, precision=precision),
                bodo.Time(9, 1, 30, 10, precision=precision),
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
                bodo.Time(11, 59, 59, 100, 1000, 50, precision=precision),
            ],
        }
    )

    def impl(df, df2):
        return df.merge(df2, how="inner", on="A")

    check_func(impl, (df, df2), sort_output=True, reset_index=True)

    def impl2(df, df2):
        return df.merge(df2, how="inner", on="left.A == right.A & left.B < right.D")

    py_out = df.merge(df2, left_on=["A"], right_on=["A"])
    py_out = py_out.query("B < D")
    check_func(
        impl2,
        (df, df2),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_out,
    )


@pytest.mark.slow
@pytest.mark.parametrize("precision", [0, 3, 6, 9])
def test_time_groupby(precision, memory_leak_check):
    """Test groupby with Time column as key with index=False and as an aggregation column
        NOTE: [BE-4109] Not testing Time as groupby key with as_index=True
        since Time is not supported as an index.

    Args:
        precision (int): Time precision argument
        memory_leak_check (fixture function): check memory leak in the test.

    """
    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0, precision=precision),
                bodo.Time(1, 1, 3, 1, precision=precision),
                bodo.Time(2, precision=precision),
                bodo.Time(15, 0, 50, 10, 100, precision=precision),
                bodo.Time(9, 1, 3, 10, precision=precision),
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
            ],
            "B": [0, 0, 1, 0, 0, 1],
        }
    )

    # Test Time as column to compute aggregation on
    def impl(df):
        return df.groupby("B")["A"].agg(["min", "max", "first", "last"])

    check_func(impl, (df,), sort_output=True, reset_index=True)

    df = pd.DataFrame(
        {
            "A": [
                bodo.Time(12, 0, precision=precision),
                None,
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
                bodo.Time(2, precision=precision),
                bodo.Time(12, 0, precision=precision),
                bodo.Time(15, 0, 50, 10, 100, precision=precision),
                None,
                bodo.Time(2, precision=precision),
                bodo.Time(11, 59, 59, 100, 100, 50, precision=precision),
            ],
            "B": [0, 0, 1, 0, 0, 1, 2, 1, -1],
        }
    )

    # Test Time as column to compute aggregation on with None values
    def impl2(df):
        return df.groupby("B")["A"].max()

    # Hard-code py_output (See [BE-4107])
    py_output = df.dropna().groupby("B")["A"].max().append(pd.Series([None], name="A"))
    check_func(impl2, (df,), py_output=py_output, sort_output=True, reset_index=True)

    # Test Time as key with index=False and keeping None group
    def impl3(df):
        return df.groupby("A", as_index=False, dropna=False)["B"].min()

    check_func(impl3, (df,), sort_output=True, reset_index=True)

    # Test Time as key with index=False and dropping None group
    def impl4(df):
        return df.groupby("A", as_index=False)["B"].max()

    check_func(impl4, (df,), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_time_head(memory_leak_check):
    df = pd.DataFrame({"A": [bodo.Time(1, x) for x in range(15)]})

    def impl(df):
        return df.head()

    check_func(impl, (df,))


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda dt: bodo.Time(dt.hour, precision=0),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.Time(dt.hour, dt.minute, precision=0),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.Time(dt.hour, dt.minute, dt.second, precision=0),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.Time(
                dt.hour, dt.minute, dt.second, dt.millisecond, precision=3
            ),
            id="millisecond",
            marks=pytest.mark.slow,
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
            marks=pytest.mark.slow,
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
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(12, precision=9),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(12, 30, precision=9),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(12, 30, 42, precision=9),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(12, 30, 42, 64, precision=9),
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            bodo.Time(12, 30, 42, 64, 43, precision=9),
            id="microsecond",
            marks=pytest.mark.slow,
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


@pytest.mark.slow
def test_time_array_setitem_none(memory_leak_check):
    df = pd.DataFrame({"A": [bodo.Time(1, x) for x in range(15)]})

    def impl(df):
        df["A"][0] = None
        return df

    check_func(impl, (df,))


@pytest.mark.slow
def test_comparison_error(memory_leak_check):

    # Time vs. non-Time
    def impl():
        return bodo.Time(2) < None

    with pytest.raises(
        TypeError,
        match="Cannot compare Time with non-Time type",
    ):
        impl()

    # Time different precisions
    def impl2():
        return bodo.Time(2) < bodo.Time(2, precision=0)

    with pytest.raises(
        TypeError,
        match="Cannot compare times with different precisions",
    ):
        impl2()
