import pandas as pd
import pyarrow as pa
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.pandas.utils import BodoLibFallbackWarning, convert_to_pandas_types
from bodo.tests.utils import _test_equal

pytestmark = pytest.mark.jit_dependency


@pytest.fixture
def groupby_df():
    return pd.DataFrame(
        {
            "A": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "C": [1, 0, 1, -2, 1, -2, -1, 2, 1],
            "B": [-2, -2, 3, 4, 5, 6, 4, 8, 9],
        }
    )


@pytest.fixture(
    params=[
        pytest.param(True, id="dropna-True"),
        pytest.param(False, id="dropna-False"),
    ],
    scope="module",
)
def dropna(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(True, id="as_index-True"),
        pytest.param(False, id="as_index-False"),
    ],
    scope="module",
)
def as_index(request):
    return request.param


def test_basic_agg_udf(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    df2 = df.groupby(by=["A", "C"]).agg(udf)
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A", "C"]).agg(udf)

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


def test_df_agg_multiple_udf(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    def udf2(x):
        if sum(x) == 0:
            return None
        return sum(x[x < 0]) / sum(x)

    df2 = df.groupby(by=["A"]).agg({"B": udf2, "C": udf})
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"]).agg({"B": udf2, "C": udf})

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


def test_series_agg_multiple_udf(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    def udf2(x):
        if sum(x) == 0:
            return None
        return sum(x[x < 0]) / sum(x)

    df2 = df.groupby(by=["A"])["B"].agg([udf2, udf])
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"])["B"].agg([udf2, udf])

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


def test_agg_mix_udf_builtin(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    df2 = df.groupby(by=["A"]).agg({"B": "sum", "C": udf})
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"]).agg({"B": "sum", "C": udf})

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: 4, id="int_ret"),
        pytest.param(lambda x: True, id="bool_ret"),
        pytest.param(lambda x: 1.23, id="float_ret", marks=pytest.mark.slow),
        pytest.param(lambda x: pd.Timestamp(2021, 11, 3), id="timestamp_ret"),
        pytest.param(
            lambda x: pd.Timedelta(3), id="timedelta_ret", marks=pytest.mark.slow
        ),
        pytest.param(lambda x: "A", id="string_ret"),
        pytest.param(
            lambda x: {"id": 1, "text": "hello"},
            id="struct_ret",
            marks=pytest.mark.slow,
        ),
        pytest.param(lambda x: [1, 2], id="list_ret"),
    ],
)
@pytest.mark.parametrize(
    "val_col",
    [
        pytest.param(
            pd.array([1.23, 1.23, 3.2, pd.NA] * 3, dtype="Float64"), id="float_col"
        ),
        pytest.param(
            pd.array([True, False, True, True] * 3, dtype="boolean"), id="bool_col"
        ),
        pytest.param(
            pd.array(["a", None, "b", "c"] * 3, dtype="string[pyarrow]"),
            id="string_col",
        ),
        pytest.param(
            pd.array(
                [pd.Timestamp(2021, 10, 2), None, pd.Timestamp(2021, 11, 3), None] * 3,
                dtype="datetime64[ns]",
            ),
            id="datetime_col",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.array([None, None, 3, 100] * 3, dtype="timedelta64[ns]"),
            id="timedelta_col",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.array(
                [None, [1, 2, 3], [4, 5, 6], [1, 2, 3]] * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
            id="list_col",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.array(
                [{"A": 1, "B": "a"}, {"A": 1, "B": "a"}, None, {"A": 2, "B": "b"}] * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [pa.field("A", pa.int64()), pa.field("B", pa.large_string())]
                    )
                ),
            ),
            id="struct_col",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(lambda df, func: df.groupby(by=["A"]).agg({"B": func}), id="agg"),
        pytest.param(
            lambda df, func: df.groupby(by=["A"]).apply(func, include_groups=False),
            id="apply",
            marks=pytest.mark.skip(
                "Skip tests on Nightly CI to reduce memory footprint."
            ),
        ),
    ],
)
def test_groupby_udf_types(impl, val_col, func):
    """Test agg and apply with custom funcs of different types."""

    df = pd.DataFrame({"A": ["A", "B", "C"] * 4, "B": val_col})
    bdf = bd.from_pandas(df)
    pdf = convert_to_pandas_types(df)

    df2 = impl(pdf, func)
    with assert_executed_plan_count(0):
        bdf2 = impl(bdf, func)

    _test_equal(bdf2, df2, check_pandas_types=False, reset_index=True)


def test_agg_udf_errorchecking(groupby_df):
    bdf = bd.from_pandas(groupby_df)

    # x is a series
    def udf1(x):
        return x

    # raises an exception
    def udf2(x):
        if len(x):
            raise Exception
        else:
            return 1

    # unsupported function, fallback to Pandas
    def udf3(x, flag=3):
        if flag < 3:
            return "A"
        else:
            return 0

    # Series output not supported raises ValueError to match Pandas
    with pytest.raises(
        ValueError, match="User defined function must produce aggregated value"
    ):
        bdf.groupby("A").agg(udf1).execute_plan()

    with pytest.raises(
        Exception,
        match=r"Groupby.agg\(\) \| Groupby.apply\(\): An error occured while executing user defined function.",
    ):
        bdf.groupby("A").agg(udf2).execute_plan()

    # UDF cannot be compiled, fallback to Pandas
    df2 = groupby_df.groupby("A").agg(udf3)
    with pytest.warns(
        BodoLibFallbackWarning,
        match="An error occured while compiling user defined function",
    ):
        bdf2 = bdf.groupby("A").agg(udf3)

    _test_equal(bdf2, df2)


def test_agg_null_keys(dropna, as_index):
    df = pd.DataFrame({"A": [None, None, None, 1, 2, 3], "B": [1, 2, 3, 4, 5, 6]})

    bdf = bd.from_pandas(df)

    df2 = df.groupby("A", dropna=dropna, as_index=as_index).agg(lambda x: x.sum())
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby("A", dropna=dropna, as_index=as_index).agg(lambda x: x.sum())

    # If as_index=False, then the output before sorting will use a RangeIndex,
    # which will result in mismatched indexes after sorting, so we need to reset the index.
    _test_equal(bdf2, df2, sort_output=True, reset_index=(not as_index))


@pytest.mark.skip(reason="TODO[BSE-5254]: Fix CI error with map")
def test_apply_basic(dropna, as_index):
    """Test basic groupby apply example"""
    df = pd.DataFrame(
        {
            "B": ["a", "c", "b", "c"] * 3,
            "A": ["A", "B", None] * 4,
            "C": [1, 2, 3, 4, 5, 6] * 2,
            "BB": ["a", "b"] * 6,
        }
    )
    df["A"] = df["A"].astype("string[pyarrow]")

    bdf = bd.from_pandas(df)

    def udf(df):
        denom = df["C"].sum()
        df = df[df["B"] == "c"]
        numer = df["C"].sum()
        return numer / denom

    df2 = df.groupby(["A", "BB"], dropna=dropna, as_index=as_index).apply(
        udf, include_groups=False
    )
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(["A", "BB"], dropna=dropna, as_index=as_index).apply(
            udf, include_groups=False
        )

    # Pandas/BD returns None column when as_index=False
    # Bodo DataFrames just returns "None" for now,
    if not as_index:
        df2 = df2.rename(columns={None: "None"})

    _test_equal(
        bdf2,
        df2,
        sort_output=True,
        check_pandas_types=False,
        reset_index=(not as_index),
    )


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            lambda df, func: df.groupby("A", dropna=dropna, as_index=as_index)["C"].agg(
                func
            ),
            id="agg",
        ),
        pytest.param(
            lambda df, func: df.groupby("A", dropna=dropna, as_index=as_index)[
                "C"
            ].apply(func, include_groups=False),
            id="apply",
        ),
    ],
)
def test_series_udf(dropna, as_index, impl):
    df = pd.DataFrame(
        {
            "B": ["a", "c", "b", "c"] * 3,
            "A": ["A", "B", None] * 4,
            "C": [1, 2, 3, 4, 5, 6] * 2,
        }
    )

    bdf = bd.from_pandas(df)

    def udf(x):
        return x.max() - x.min()

    df2 = impl(df, udf)
    with assert_executed_plan_count(0):
        bdf2 = impl(bdf, udf)

    _test_equal(
        bdf2,
        df2,
        sort_output=True,
        check_pandas_types=False,
        reset_index=(not as_index),
    )


def test_apply_fallback(groupby_df: pd.DataFrame, as_index):
    # DataFrame return value
    def udf1(x):
        return x

    # Unsupported operation in JIT (string Series sum)
    def udf2(x):
        return x.A.sum()

    bdf = bd.from_pandas(groupby_df)

    df2 = groupby_df.groupby("C", as_index=as_index).apply(udf1, include_groups=False)
    with pytest.warns(
        BodoLibFallbackWarning,
        match="functions returning Series or DataFrame not implemented yet",
    ):
        bdf2 = bdf.groupby("C", as_index=as_index).apply(udf1, include_groups=False)

    if not as_index:
        df2 = df2.rename(columns={None: "None"})
        bdf2 = df2.rename(columns={None: "None"})

    _test_equal(bdf2, df2, sort_output=True, check_pandas_types=False, reset_index=True)

    bdf = bd.from_pandas(groupby_df)

    df2 = groupby_df.groupby("C", as_index=as_index).apply(udf2, include_groups=False)
    with pytest.warns(
        BodoLibFallbackWarning,
        match="An error occured while compiling user defined function 'udf2'",
    ):
        bdf2 = bdf.groupby("C", as_index=as_index).apply(udf2, include_groups=False)

    if not as_index:
        df2 = df2.rename(columns={None: "None"})
        bdf2 = df2.rename(columns={None: "None"})

    _test_equal(
        bdf2,
        df2,
        sort_output=True,
        check_pandas_types=False,
        reset_index=(not as_index),
    )
