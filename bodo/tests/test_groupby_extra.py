import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func, pytest_pandas

pytestmark = pytest_pandas


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": np.arange(100) % 10,
                "B": np.arange(100) % 7,
                "C": np.arange(100),
                # These two columns should be skipped for numeric only computation
                "D": ["a", "ecw", "e2r", "2314", " 342"] * 20,
                "E": [b"a", b"ecw", b"e2r", b"2314", b" 342"] * 20,
            }
        )
    ]
)
def test_groupby_agg_df(request):
    return request.param


def test_agg_single_str(test_groupby_agg_df, memory_leak_check):
    """
    Check that groupby.agg works with a single
    string argument that matches a legal aggregation
    function.
    """

    def test_impl1(df):
        return df.groupby("B").agg("sum")

    def test_impl2(df):
        return df.groupby("B", as_index=False)[["A", "C"]].agg("mean")

    def test_impl3(df):
        return df.groupby(["A", "B"], as_index=False)["C"].agg("mean")

    check_func(test_impl1, (test_groupby_agg_df,), sort_output=True, reset_index=True)
    check_func(
        test_impl2,
        (test_groupby_agg_df,),
        sort_output=True,
        reset_index=True,
        # Mean always outputs a nullable float for Bodo.
        check_dtype=False,
        convert_to_nullable_float=False,
    )
    # Pandas seems to detect that every value can be represented as an integer and returns
    # int instead of float for the dtype of column C.
    check_func(
        test_impl3,
        (test_groupby_agg_df,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        convert_to_nullable_float=False,
    )


@pytest.mark.skip("Not supported in Pandas 3")
def test_agg_single_builtin(test_groupby_agg_df, memory_leak_check):
    """
    Check that groupby.agg works with a single
    builtin function argument that matches a legal aggregation
    function.
    """

    def test_impl1(df):
        return df.groupby("B").agg(sum)

    def test_impl2(df):
        return df.groupby("B", as_index=False).agg(min)

    def test_impl3(df):
        return df.groupby(["A", "B"], as_index=False).agg(sum)

    check_func(test_impl1, (test_groupby_agg_df,), sort_output=True, reset_index=True)
    check_func(test_impl2, (test_groupby_agg_df,), sort_output=True, reset_index=True)
    check_func(test_impl3, (test_groupby_agg_df,), sort_output=True, reset_index=True)


def test_agg_multi_builtin(test_groupby_agg_df, memory_leak_check):
    """
    Check that groupby.agg works with multiple
    builtin function argument that match legal aggregation
    functions.
    """

    def test_impl(df):
        return df.groupby("A").agg({"C": sum, "D": max})

    check_func(test_impl, (test_groupby_agg_df,), sort_output=True, reset_index=True)


def test_transform_builtin(test_groupby_agg_df, memory_leak_check):
    """
    Check that groupby.agg works with a builtin function argument
    that matches a legal aggregation
    function.
    """

    def test_impl1(df):
        return df.groupby("B").transform(sum)

    def test_impl2(df):
        return df.groupby("B", as_index=False).transform(min)

    def test_impl3(df):
        return df.groupby(["A", "B"], as_index=False).transform(max)

    # Omit D because string types aren't supported yet.
    test_groupby_agg_df = test_groupby_agg_df[["A", "B", "C"]]
    check_func(test_impl1, (test_groupby_agg_df,), sort_output=True, reset_index=True)
    check_func(test_impl2, (test_groupby_agg_df,), sort_output=True, reset_index=True)
    check_func(test_impl3, (test_groupby_agg_df,), sort_output=True, reset_index=True)


def test_groupby_window_all_dead():
    # TODO: Fix memory leak
    """
    Tests the custom groupby.window kernel outputs the correct length
    when all input columns are dead.
    """

    def impl(df):
        res1 = df.groupby(["A"]).window((("row_number",),), ("B",), (True,), ("last",))
        return len(res1)

    df = pd.DataFrame(
        {
            "A": [1, 4, 4, 3] * 2,
            "B": [3, 2, 1, 5, 6, 7, 8, -1],
        }
    )
    check_func(impl, (df,), py_output=8)


@pytest.mark.slow
def test_groupby_min_null_datetime_timedelta(memory_leak_check):
    """
    Tests groupby.min() with null datetime and timedelta arrays
    """

    def impl(df):
        return df.groupby(["A"]).min()

    df = pd.DataFrame(
        {
            "A": [
                1,
            ]
            * 7,
            "B": [pd.Timestamp(None)] * 7,
            "C": [pd.Timedelta(None)] * 7,
            "D": [pd.Timestamp(None, tz="US/Pacific")] * 7,
        }
    )
    check_func(impl, (df,))
