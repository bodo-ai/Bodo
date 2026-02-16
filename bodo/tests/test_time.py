import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import bodo
from bodo.tests.utils import _test_equal_guard, check_func
from bodo.utils.testing import ensure_clean


@pytest.fixture
def lazy_time_fixture(request):
    """Lazyily import Time to avoid importing the compiler at test collection time."""
    import bodo.decorators  # noqa

    val = request.param

    return val()


@pytest.mark.parametrize(
    "time_str, lazy_time_fixture",
    [
        pytest.param(
            "0",
            lambda: bodo.types.Time(0, 0, 0, precision=9),
            id="numeric_string-zero",
        ),
        pytest.param(
            "30",
            lambda: bodo.types.Time(0, 0, 30, precision=9),
            id="numeric_string-thirty",
        ),
        pytest.param(
            "10000",
            lambda: bodo.types.Time(2, 46, 40, precision=9),
            id="numeric_string-ten_thousand",
        ),
        pytest.param(
            "12:30",
            lambda: bodo.types.Time(12, 30, 0, precision=9),
            id="hour_minute-no_leading",
        ),
        pytest.param(
            "10:5",
            lambda: bodo.types.Time(10, 5, 0, precision=9),
            id="hour_minute-short_minute",
        ),
        pytest.param(
            "7:45",
            lambda: bodo.types.Time(7, 45, 0, precision=9),
            id="hour_minute-short_hour",
        ),
        pytest.param(
            "1:6",
            lambda: bodo.types.Time(1, 6, 0, precision=9),
            id="hour_minute-short_both",
        ),
        pytest.param(
            "10:20:30",
            lambda: bodo.types.Time(10, 20, 30, precision=9),
            id="hour_minute_second-no_leading",
        ),
        pytest.param(
            "23:01:59",
            lambda: bodo.types.Time(23, 1, 59, precision=9),
            id="hour_minute_second-short_minute",
        ),
        pytest.param(
            "20:50:03",
            lambda: bodo.types.Time(20, 50, 3, precision=9),
            id="hour_minute_second-short_second",
        ),
        pytest.param(
            "1:2:3",
            lambda: bodo.types.Time(1, 2, 3, precision=9),
            id="hour_minute_second-short_all",
        ),
        pytest.param(
            "16:17:18.",
            lambda: bodo.types.Time(16, 17, 18, precision=9),
            id="hour_minute_second_dot-no_leading",
        ),
        pytest.param(
            "6:30:9.",
            lambda: bodo.types.Time(6, 30, 9, precision=9),
            id="hour_minute_second_dot-short_hour_sec",
        ),
        pytest.param(
            "00:4:0.",
            lambda: bodo.types.Time(0, 4, 0, precision=9),
            id="hour_minute_second_dot-short_minute_sec",
        ),
        pytest.param(
            "12:30:15.5",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=500_000_000, precision=9),
            id="hour_minute_second_nanoseconds-one_digit",
        ),
        pytest.param(
            "12:30:15.99",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=990_000_000, precision=9),
            id="hour_minute_second_nanoseconds-two_digits",
        ),
        pytest.param(
            "12:30:15.607",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=607_000_000, precision=9),
            id="hour_minute_second_nanoseconds-three_digits",
        ),
        pytest.param(
            "12:30:15.0034",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=3_400_000, precision=9),
            id="hour_minute_second_nanoseconds-four_digits",
        ),
        pytest.param(
            "12:30:15.000250",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=250_000, precision=9),
            id="hour_minute_second_nanoseconds-six_digits",
        ),
        pytest.param(
            "12:30:15.67108864",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=671_088_640, precision=9),
            id="hour_minute_second_nanoseconds-eight_digits",
        ),
        pytest.param(
            "12:30:15.123456789",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=123_456_789, precision=9),
            id="hour_minute_second_nanoseconds-nine_digits",
        ),
        pytest.param(
            "12:30:15.989796859493",
            lambda: bodo.types.Time(12, 30, 15, nanosecond=989_796_859, precision=9),
            id="hour_minute_second_nanoseconds-twelve_digits",
        ),
    ],
    indirect=["lazy_time_fixture"],
)
def test_time_parsing(time_str, lazy_time_fixture):
    answer = lazy_time_fixture

    def impl(time_str):
        hr, mi, sc, ns, succeeded = bodo.hiframes.time_ext.parse_time_string(time_str)
        if succeeded:
            return bodo.types.Time(hr, mi, sc, nanosecond=ns, precision=9)
        else:
            return bodo.types.Time(0, 0, 0, nanosecond=0, precision=9)

    check_func(impl, (time_str,), py_output=answer)


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda: bodo.types.Time(precision=0),
            id="none",
        ),
        pytest.param(
            lambda: bodo.types.Time(12, precision=0),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 34, precision=0),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 34, 56, precision=0),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 34, 56, 78, precision=3),
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 34, 56, 78, 12, precision=6),
            id="microsecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 34, 56, 78, 12, 34, precision=9),
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
    t = bodo.types.Time(1, 2, 3, 4, 5, 6, precision=9)

    check_func(impl, (t,))


@pytest.mark.parquet
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
            "A": bodo.types.Time(0, 0, 0, precision=precision),
            "B": bodo.types.Time(1, 1, 1, precision=precision),
            "C": bodo.types.Time(2, 2, 2, precision=precision),
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
            df = pd.read_parquet(fname, dtype_backend="pyarrow")
            df.to_parquet(fname2, index=False)

        impl()

        # TODO: Because all data is loaded as ns, we can compare to the original
        # dataframe, but this should change when we support other time units.
        bodo.barrier()

        # read in bodo because of pandas type differences
        @bodo.jit(distributed=False)
        def reader():
            return pd.read_parquet(fname2, dtype_backend="pyarrow")

        df = reader()
        _test_equal_guard(df, df_orig)


@pytest.fixture
def a(request):
    """Lazily construct arguments for comparison tests to avoid importing at
    collection time."""
    import bodo.decorators  # noqa

    h, m, s, us, prec = request.param
    return bodo.types.Time(h, m, s, us, precision=prec)


@pytest.fixture
def b(request):
    """Lazily construct arguments for comparison tests to avoid importing at
    collection time."""
    import bodo.decorators  # noqa

    h, m, s, us, prec = request.param
    return bodo.types.Time(h, m, s, us, precision=prec)


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
            (1, 15, 12, 0, 3), (1, 15, 12, 0, 3), id="data_eq", marks=pytest.mark.slow
        ),
        pytest.param(
            (1, 15, 12, 0, 3), (1, 15, 12, 1, 3), id="data_lt", marks=pytest.mark.slow
        ),
        pytest.param((1, 15, 12, 1, 3), (1, 15, 12, 0, 3), id="data_gt"),
    ],
    indirect=["a", "b"],
)
def test_time_cmp(cmp_fn, a, b, memory_leak_check):
    check_func(cmp_fn, (a, b))


@pytest.mark.slow
@pytest.mark.parametrize("precision", [6, 9])
def test_time_sort(precision, memory_leak_check):
    """Test sort by a Time column

    Args:
        precision (int): Time precision argument
        memory_leak_check (fixture function): check memory leak in the test.

    """
    df = pd.DataFrame(
        {
            "A": pd.array(
                [
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(1, 1, 3, 1, precision=precision),
                    None,
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(15, 0, 50, 10, 100, precision=precision),
                    bodo.types.Time(9, 1, 3, 10, precision=precision),
                    None,
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            )
        }
    )

    def impl(df):
        return df.sort_values(by="A")

    check_func(impl, (df,), reset_index=True)


@pytest.mark.parametrize("precision", [6, 9])
def test_time_merge(precision, memory_leak_check):
    """Test join on a Time column

    Args:
        precision (int): Time precision argument
        memory_leak_check (fixture function): check memory leak in the test.

    """
    df = pd.DataFrame(
        {
            "A": pd.array(
                [
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(1, 1, 3, 1, precision=precision),
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(15, 0, 50, 10, 100, precision=precision),
                    bodo.types.Time(9, 1, 3, 10, precision=precision),
                    None,
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
            "B": pd.array(
                [
                    None,
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(1, 11, 3, 1, precision=precision),
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(14, 0, 50, 10, 100, precision=precision),
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                    bodo.types.Time(9, 1, 30, 10, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
        }
    )

    df2 = pd.DataFrame(
        {
            "A": pd.array(
                [
                    None,
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(1, 1, 3, 1, precision=precision),
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(1, 10, precision=precision),
                    None,
                    bodo.types.Time(1, 11, 30, 100, precision=precision),
                    bodo.types.Time(12, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
            "D": pd.array(
                [
                    bodo.types.Time(11, 0, precision=precision),
                    None,
                    bodo.types.Time(6, 11, 3, 1, precision=precision),
                    bodo.types.Time(9, precision=precision),
                    bodo.types.Time(14, 10, 50, 10, 100, precision=precision),
                    bodo.types.Time(9, 1, 30, 10, precision=precision),
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                    bodo.types.Time(11, 59, 59, 100, 1000, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
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
@pytest.mark.parametrize("precision", [6, 9])
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
            "A": pd.array(
                [
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(1, 1, 3, 1, precision=precision),
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(15, 0, 50, 10, 100, precision=precision),
                    bodo.types.Time(9, 1, 3, 10, precision=precision),
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
            "B": pd.array([0, 0, 1, 0, 0, 1], dtype=pd.ArrowDtype(pa.int64())),
        }
    )

    # Test Time as column to compute aggregation on
    def impl(df):
        return df.groupby("B")["A"].agg(["min", "max", "first", "last"])

    check_func(impl, (df,), sort_output=True, reset_index=True)

    df = pd.DataFrame(
        {
            "A": pd.array(
                [
                    bodo.types.Time(12, 0, precision=precision),
                    None,
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(12, 0, precision=precision),
                    bodo.types.Time(15, 0, 50, 10, 100, precision=precision),
                    None,
                    bodo.types.Time(2, precision=precision),
                    bodo.types.Time(11, 59, 59, 100, 100, precision=precision),
                ],
                dtype=pd.ArrowDtype(pa.time64("ns" if precision == 9 else "us")),
            ),
            "B": pd.array(
                [0, 0, 1, 0, 0, 1, 2, 1, -1], dtype=pd.ArrowDtype(pa.int64())
            ),
        }
    )

    # Test Time as column to compute aggregation on with None values
    def impl2(df):
        return df.groupby("B")["A"].max()

    # Hard-code py_output (See [BE-4107])
    py_output = pd.concat(
        (df.dropna().groupby("B")["A"].max(), pd.Series([None], name="A"))
    )
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
    df = pd.DataFrame(
        {
            "A": pd.array(
                [bodo.types.Time(1, x) for x in range(15)],
                dtype=pd.ArrowDtype(pa.time64("ns")),
            )
        }
    )

    def impl(df):
        return df.head()

    check_func(impl, (df,))


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda dt: bodo.types.Time(dt.hour, precision=0),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.types.Time(dt.hour, dt.minute, precision=0),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.types.Time(dt.hour, dt.minute, dt.second, precision=0),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.types.Time(
                dt.hour, dt.minute, dt.second, dt.millisecond, precision=3
            ),
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda dt: bodo.types.Time(
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
            lambda dt: bodo.types.Time(
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
    "lazy_time_fixture",
    [
        pytest.param(
            lambda: bodo.types.Time(precision=9),
            id="none",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, precision=9),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 30, precision=9),
            id="minute",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 30, 42, precision=9),
            id="second",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 30, 42, 64, precision=9),
            id="millisecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 30, 42, 64, 43, precision=9),
            id="microsecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: bodo.types.Time(12, 30, 42, 64, 43, 58, precision=9),
            id="nanosecond",
        ),
    ],
    indirect=["lazy_time_fixture"],
)
def test_time_construction_from_parts(impl, lazy_time_fixture, memory_leak_check):
    """Test that time can be constructed from parts of a time.
    Needed for SQL `TRUNC` and `TIME_SLICE` functionality.
    """
    dt = lazy_time_fixture

    check_func(impl, (dt,))


@pytest.mark.skip("Pandas 3 issue with setitem")
@pytest.mark.slow
def test_time_array_setitem_none(memory_leak_check):
    df = pd.DataFrame(
        {
            "A": pd.array(
                [bodo.types.Time(1, x) for x in range(15)],
                dtype=pd.ArrowDtype(pa.time64("ns")),
            )
        }
    )

    def impl(df):
        df.loc[0, "A"] = None
        return df

    check_func(impl, (df,))


@pytest.mark.slow
def test_compare_with_none(memory_leak_check):
    """
    Tests to compare Time with None
    """

    def impl1():
        return bodo.types.Time(1, 2, 3) < None

    check_func(impl1, (), py_output=True)

    def impl2():
        return None > bodo.types.Time(4, 5, 6)

    check_func(impl2, (), py_output=False)

    def impl3():
        return bodo.types.Time(1, 2, 3) != None

    check_func(impl3, (), py_output=True)

    def impl4():
        return None == bodo.types.Time(4, 5, 6)

    check_func(impl4, (), py_output=False)


@pytest.mark.slow
def test_compare_different_precisions(memory_leak_check):
    """
    Tests to compare Time objects with different precisions
    """

    def impl1():
        return bodo.types.Time(5, 6, 7, 8) != bodo.types.Time(5, 6, 7, 8, precision=3)

    check_func(impl1, ())

    def impl2():
        return bodo.types.Time(2) == bodo.types.Time(2, precision=0)

    check_func(impl2, ())

    def impl3():
        return bodo.types.Time(12, precision=0) < bodo.types.Time(12, 13, 14)

    check_func(impl3, ())

    def impl4():
        return bodo.types.Time(13, precision=6) >= bodo.types.Time(
            12, 13, 14, 15, 16, 17
        )

    check_func(impl4, ())


@pytest.mark.slow
def test_compare_same_precision(memory_leak_check):
    """
    Tests to compare Time objects with the same precision
    """

    def impl1():
        return bodo.types.Time(12) != bodo.types.Time(12, 13, 14)

    check_func(impl1, ())

    def impl2():
        return bodo.types.Time(12) == bodo.types.Time(12)

    check_func(impl2, ())

    def impl3():
        return bodo.types.Time(12) > bodo.types.Time(9, 8, 7)

    check_func(impl3, ())

    def impl4():
        return bodo.types.Time(22) <= bodo.types.Time(9, 8, 7)

    check_func(impl4, ())


def test_time_series_min_max(time_df, memory_leak_check):
    """
    Test Series.min() and Series.max() with a bodo.types.Time Series.
    """
    np.random.seed(1)
    time_arr = [
        bodo.types.Time(17, 33, 26, 91, 8, 79),
        bodo.types.Time(0, 24, 43, 365, 18, 74),
        bodo.types.Time(3, 59, 6, 25, 757, 3),
        bodo.types.Time(),
        bodo.types.Time(4),
        bodo.types.Time(6, 41),
        bodo.types.Time(22, 13, 57),
        bodo.types.Time(17, 34, 29, 90),
        bodo.types.Time(7, 3, 45, 876, 234),
    ]
    np.random.shuffle(time_arr)
    S = pd.Series(time_arr)

    def impl_min(S):
        return S.min()

    def impl_max(S):
        return S.max()

    check_func(impl_min, (S,))
    check_func(impl_max, (S,))


def test_time_series_min_max_none(memory_leak_check):
    """
    Test Series.min() and Series.max() with a bodo.types.Time Series
    and a None entry. This isn't supported in Pandas but should work
    in Bodo.
    """
    np.random.seed(1)
    time_arr = [
        bodo.types.Time(17, 33, 26, 91, 8, 79),
        bodo.types.Time(0, 24, 43, 365, 18, 74),
        bodo.types.Time(3, 59, 6, 25, 757, 3),
        bodo.types.Time(),
        bodo.types.Time(4),
        bodo.types.Time(6, 41),
        bodo.types.Time(22, 13, 57),
        bodo.types.Time(17, 34, 29, 90),
        bodo.types.Time(7, 3, 45, 876, 234),
        None,
    ]
    np.random.shuffle(time_arr)
    S = pd.Series(time_arr)

    def impl_min(S):
        return S.min()

    def impl_max(S):
        return S.max()

    py_output = S.dropna().min()
    check_func(impl_min, (S,), py_output=py_output)
    py_output = S.dropna().max()
    check_func(impl_max, (S,), py_output=py_output)
