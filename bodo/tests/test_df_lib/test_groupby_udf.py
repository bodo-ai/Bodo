import pandas as pd
import pyarrow as pa
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.pandas.utils import BodoLibFallbackWarning, convert_to_pandas_types
from bodo.tests.utils import _test_equal


@pytest.fixture
def groupby_df():
    return pd.DataFrame(
        {
            "A": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "C": [1, 0, 1, -2, 1, -2, -1, 2, 1],
            "B": [-2, -2, 3, 4, 5, 6, 4, 8, 9],
        }
    )


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
        pytest.param(lambda x: {"id": 1, "text": "hello"}, id="struct_ret"),
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
        ),
    ],
)
def test_agg_udf_types(val_col, func):
    """Test agg with custom funcs of different types."""

    df = pd.DataFrame({"A": ["A", "B", "C"] * 4, "B": val_col})
    bdf = bd.from_pandas(df)
    pdf = convert_to_pandas_types(df)

    df2 = pdf.groupby(by=["A"]).agg({"B": func})
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"]).agg({"B": func})

    _test_equal(bdf2, df2, check_pandas_types=False)


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
    with pytest.raises(ValueError, match="Must produce aggregated value"):
        bdf.groupby("A").agg(udf1).execute_plan()

    with pytest.raises(
        Exception,
        match=r"Groupby.agg\(\): An error occured while executing user defined function.",
    ):
        bdf.groupby("A").agg(udf2).execute_plan()

    # UDF cannot be compiled, fallback to Pandas
    df2 = groupby_df.groupby("A").agg(udf3)
    with pytest.warns(
        BodoLibFallbackWarning, match=r"Groupby.agg\(\): unable to compile udf3"
    ):
        bdf2 = bdf.groupby("A").agg(udf3)

    _test_equal(bdf2, df2)
