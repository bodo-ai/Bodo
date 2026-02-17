"""
Test correctness of SQL aggregation operations without groupby on BodoSQL
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    count_array_REPs,
    dist_IR_contains,
    pytest_slow_unless_groupby,
)
from bodo.tests.utils_jit import DistTestPipeline
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query

# Skip unless any groupby-related files were changed
pytestmark = pytest_slow_unless_groupby


def test_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test agg func calls in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "select median(A) from table1",
                pd.DataFrame(
                    {
                        "A": [3.35],
                    }
                ),
            ),
        ),
        pytest.param(
            (
                "select median(B) from table1",
                pd.DataFrame(
                    {
                        "B": [-3.35],
                    }
                ),
            ),
        ),
        pytest.param(
            (
                "select median(C) from table1",
                pd.DataFrame(
                    {
                        "C": [0.0],
                    }
                ),
            ),
        ),
        pytest.param(
            (
                "select median(D) from table1",
                pd.DataFrame(
                    {
                        "D": [1.0],
                    }
                ),
            ),
        ),
    ],
)
@pytest.mark.slow
def test_median(args, spark_info, memory_leak_check):
    df1 = pd.DataFrame(
        {
            "A": [1.0, 2.5, 1000.0, 100.0, 4.2, 1.001],
            "B": [-1.0, -2.5, -1000.0, -100.0, -4.2, 4.5],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "D": [0, 0, 1, 1, 1, 1],
        }
    )
    query, expected_output = args

    check_query(
        query,
        {"TABLE1": df1},
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_aliasing_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aliasing of aggregations in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["TABLE1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B) as testCol from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_repeat_columns(basic_df, spark_info, memory_leak_check):
    """
    Tests that a column that won't produce a conflicting name
    even if it performs the same operation.
    """
    query = "Select sum(A), sum(A) as alias from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_count_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """test various count queries on nullable numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test various count queries on Timestamp data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_interval(bodosql_interval_types, memory_leak_check):
    """test various count queries on Timedelta data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_interval_types,
        None,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame(
            {"B": [bodosql_interval_types["TABLE1"]["B"].nunique()]}
        ),
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_interval_types,
        None,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame({"B": [len(bodosql_interval_types["TABLE1"])]}),
    )


@pytest.mark.slow
def test_count_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """test various count queries on boolean data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_string(bodosql_string_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


def test_count_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_numeric_alias(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data with aliases."""
    check_query(
        "SELECT COUNT(Distinct B) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.skip("[BS-81]")
def test_max_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on string types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(query, bodosql_string_types, spark_info, check_names=False)


def test_max_datetime_types(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on datetime types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        is_out_distributed=False,
        use_duckdb=True,
    )


@pytest.mark.slow
def test_max_interval_types(bodosql_interval_types, memory_leak_check):
    """
    Simple test to ensure that max is working on timedelta types
    """
    query = """
        SELECT
            max(A) as output
        FROM
            table1
        """
    check_query(
        query,
        bodosql_interval_types,
        None,
        is_out_distributed=False,
        expected_output=pd.DataFrame(
            {"OUTPUT": [bodosql_interval_types["TABLE1"]["A"].max()]}
        ),
    )


@pytest.mark.slow
def test_max_literal(basic_df, memory_leak_check):
    """tests that max works on a scalar value"""
    # This query does not get optimized, by manual check
    query = "Select A, scalar_max from table1, (Select Max(1) as scalar_max)"

    check_query(
        query,
        basic_df,
        None,
        # Max outputs a nullable output by default to handle empty Series values
        check_dtype=False,
        expected_output=pd.DataFrame({"A": basic_df["TABLE1"]["A"], "SCALAR_MAX": 1}),
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT COUNT_IF(A) FROM table1", id="bool_col"),
        pytest.param("SELECT COUNT_IF(B = 'B') FROM table1", id="string_match"),
        pytest.param(
            "SELECT COUNT_IF(C % 2 = 1) FROM table1",
            id="int_cond",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_count_if(query, spark_info, memory_leak_check):
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [True, False, None, True, True, None, False, True] * 5,
                    dtype=pd.BooleanDtype(),
                ),
                "B": pd.Series(list("AABAABCBAC") * 4),
                "C": pd.Series((list(range(7)) + [None]) * 5, dtype=pd.Int32Dtype()),
            }
        )
    }

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_having(bodosql_numeric_types, comparison_ops, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) {comparison_ops} 1
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_max_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on boolean types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


def test_having_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = """
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) <> True
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_agg_replicated(datapath, memory_leak_check):
    """
    Tests that an aggregation query produces a
    replicated output.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"T1": bodosql.TablePath(filename, "parquet")})
        return bc.sql("select count(B) as cnt from t1")

    filename = datapath("sample-parquet-data/no_index.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    count = read_df.B.count()
    expected_output = pd.DataFrame({"CNT": count}, index=pd.Index([0]))
    check_func(impl, (filename,), py_output=expected_output, is_out_distributed=False)
    # Check that the function returns replicated data.
    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(impl)
    bodo_func(filename)
    # This function needs to return at least two distributed outputs, the data column
    # and the index. This functions may contain more if there are aliases or other
    # intermediate variables.
    assert count_array_REPs() > 2, "Expected replicated return value"
    # The parquet data should still be loaded as 1D
    # TODO: Determine why we're no longer inserting a project before the aggregate
    #       see https://bodo.atlassian.net/browse/BSE-1970
    # assert count_array_OneDs() > 1, "Expected distributed read from parquet"
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce"), (
        "Expected distributed reduction in the compute."
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT ANY_VALUE(A) FROM table1", id="int32"),
        pytest.param(
            "SELECT ANY_VALUE(B) FROM table1", id="string", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT ANY_VALUE(C) FROM table1", id="float", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT ANY_VALUE(A), ANY_VALUE(B), ANY_VALUE(C) FROM table1", id="all"
        ),
    ],
)
def test_any_value(query, spark_info, memory_leak_check):
    """Tests ANY_VALUE, which is normally nondeterministic but has been
    implemented in a way that is reproducible (by always returning the first
    value)"""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [5, 3, 1, 10, 30, -1, 0, None, 3, 1, 5, 4, -1, None, 10] * 2,
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series(list("AABAABCBAABCDCB") * 2),
                "C": pd.Series(
                    [
                        (((i + 3) ** 2) % 50) / 10 if i % 5 != 2 else None
                        for i in range(30)
                    ]
                ),
            }
        )
    }

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        is_out_distributed=False,
    )


def test_any_value_nulls(memory_leak_check):
    """
    Test that ANY_VALUE works when the first element is NULL.
    See BSE-2934
    """

    df = pd.DataFrame(
        {
            "A": pd.array([None, 1, 1, 1, 1, 1], dtype=pd.ArrowDtype(pa.int64())),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT ANY_VALUE(A) FROM table1"

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame(
            {0: pd.array([None], dtype=pd.ArrowDtype(pa.int64()))}
        ),
    )


@pytest.mark.tz_aware
def test_max_min_tz_aware(memory_leak_check):
    """
    Test max and min on a tz-aware timestamp column
    """
    S = pd.Series(
        list(
            pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            )
        )
        + [None] * 4
    )
    df = pd.DataFrame(
        {
            "A": S,
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {"OUTPUT1": S.max(), "OUTPUT2": S.min()}, index=pd.RangeIndex(0, 1, 1)
    )
    query = "Select max(A) as output1, min(A) as output2 from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_count_tz_aware(memory_leak_check):
    """
    Test count and count(*) on a tz-aware timestamp column
    """
    S = pd.Series(
        list(
            pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            )
        )
        + [None] * 4
    )
    df = pd.DataFrame(
        {
            "A": S,
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {"OUTPUT1": S.count(), "OUTPUT2": len(S)}, index=pd.RangeIndex(0, 1, 1)
    )
    query = "Select count(A) as output1, Count(*) as output2 from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_any_value_tz_aware(memory_leak_check):
    """
    Test any_value on a tz-aware timestamp column
    """
    S = pd.Series(
        list(
            pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            )
        )
        + [None] * 4
    )
    df = pd.DataFrame(
        {
            "A": S,
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({"OUTPUT1": S.iloc[0]}, index=pd.RangeIndex(0, 1, 1))
    # Note: We also output the first value although this is not strictly defined.
    query = "Select ANY_VALUE(A) as output1 from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_having(memory_leak_check):
    """
    Test having with tz-aware values.
    """
    df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    pd.Timestamp("2022/1/1", tz="Poland"),
                    pd.Timestamp("2022/1/2", tz="Poland"),
                    pd.Timestamp("2022/1/3", tz="Poland"),
                    pd.Timestamp("2016/1/1", tz="Poland"),
                    pd.Timestamp("2019/1/4", tz="Poland"),
                ]
                * 7,
                dtype="datetime64[ns, Poland]",
            ),
            "B": pd.Series(
                [
                    pd.Timestamp("2021/1/12", tz="Poland"),
                    pd.Timestamp("2022/2/4", tz="Poland"),
                    pd.Timestamp("2021/1/4", tz="Poland"),
                    None,
                    pd.Timestamp("2022/1/1", tz="Poland"),
                    pd.Timestamp("2027/1/1", tz="Poland"),
                    None,
                ]
                * 5,
                dtype="datetime64[ns, Poland]",
            ),
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({"OUTPUT1": df.A.min()}, index=pd.RangeIndex(0, 1, 1))
    query = "Select MIN(A) as output1 from table1 HAVING MAX(A) > min(B)"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


@pytest.fixture
def timestamptz_data():
    """TimestampTZ values for testing aggregation"""
    return np.array(
        [
            None,
            bodo.types.TimestampTZ.fromUTC("2022-01-01 00:00:00", 0),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-01-01 00:00:00", 100),
            bodo.types.TimestampTZ.fromUTC("2022-01-01 00:00:00", -100),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-01-02 01:02:03.123456789", 0),
            bodo.types.TimestampTZ.fromUTC("2022-01-02 01:02:03.123456789", 100),
            bodo.types.TimestampTZ.fromUTC("2022-01-02 01:02:03.123456789", -100),
            None,
        ]
    )


def test_max_min_timestamptz(timestamptz_data, memory_leak_check):
    """
    Test max and min on a TIMESTAMPTZ column
    """
    df = pd.DataFrame({"A": timestamptz_data})
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {
            "OUT1": bodo.types.TimestampTZ.fromUTC("2022-01-02 01:02:03.123456789", 0),
            "OUT2": bodo.types.TimestampTZ.fromUTC("2022-01-01 00:00:00", 0),
        },
        index=pd.RangeIndex(0, 1, 1),
    )
    query = "Select max(A) as OUT1, min(A) as OUT2 from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


def test_count_timestamptz(timestamptz_data, memory_leak_check):
    """
    Test count and count(*) on a TIMESTAMPTZ column
    """
    df = pd.DataFrame({"A": timestamptz_data})
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {"OUTPUT1": 6, "OUTPUT2": 10},
        index=pd.RangeIndex(0, 1, 1),
    )
    query = "Select count(A) as output1, Count(*) as output2 from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


@pytest.mark.skip(reason="[BSE-2934] NULLs in ANY_VALUE column returns the wrong value")
def test_anyvalue_timestamptz(timestamptz_data, memory_leak_check):
    """
    Test ANY_VALUE on a TIMESTAMPTZ column
    """
    # note that this is using the UTC timestamp constructor to make it easier to
    # see what the values being compared are
    df = pd.DataFrame({"A": timestamptz_data})
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {"OUTPUT": bodo.types.TimestampTZ.fromUTC("2022-01-01 00:00:00", 0)},
        index=pd.RangeIndex(0, 1),
    )
    query = "Select ANY_VALUE(A) as output from table1"
    check_query(
        query,
        ctx,
        None,
        is_out_distributed=False,
        expected_output=py_output,
    )


def test_single_value(spark_info, memory_leak_check):
    """Test Calcite's SINGLE_VALUE Agg function"""
    query = "select B from t1 where t1.A = (select C from t2)"

    df1 = pd.DataFrame(
        {
            "A": [0, 0, 1, 1, 1, 1],
            "B": [1.0, 2.5, 1000.0, 100.0, 4.2, 1.001],
            "C": [-1.0, -2.5, -1000.0, -100.0, -4.2, 4.5],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [0.1],
            "B": [2],
            "C": [1],
            "D": [1.1],
        }
    )

    check_query(
        query,
        {"T1": df1, "T2": df2},
        spark_info,
        check_names=False,
        check_dtype=False,
        # ensure_single_value() makes input replicated, causing error for dist arg
        only_jit_seq=True,
    )


def test_single_value2(spark_info, memory_leak_check):
    """Test Calcite's SINGLE_VALUE Agg function in a max query"""
    query = "select max((select max(A)-1 from t1)) from t1"

    df1 = pd.DataFrame(
        {
            "A": pd.date_range("2022-02-05", periods=8, unit="ns").date,
        }
    )

    check_query(
        query,
        {"T1": df1},
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="only_jit_seq=True disables spawn testing so pytest.raises fails",
)
@pytest.mark.slow
def test_single_value_error():
    """Make sure Calcite's SINGLE_VALUE Agg implementation raises an error for input
    with more than one value
    """
    query = "select * from t1 where t1.A = (select A from t2)"

    df1 = pd.DataFrame(
        {
            "A": [0, 0, 1],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [1, 2],
        }
    )

    # only run on np=1 since check_query turns off seq run on np>1
    if bodo.get_size() == 1:
        with pytest.raises(ValueError, match=r"Expected single value in column"):
            check_query(
                query,
                {"T1": df1, "T2": df2},
                None,
                check_names=False,
                check_dtype=False,
                # ensure_single_value() makes input replicated, causing error for dist arg
                only_jit_seq=True,
                # dummy output to avoid Spark errors
                expected_output=1,
            )


@pytest.mark.parametrize(
    "data, quantiles",
    [
        pytest.param("A", (0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 0.9), id="linear"),
        pytest.param("B", (0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99), id="uniform"),
        pytest.param("C", (0.15, 0.35, 0.45, 0.5, 0.55, 0.65, 0.85), id="gaussian"),
    ],
)
def test_approx_percentile(data, quantiles, memory_leak_check):
    """Test APPROX_PERCENTILE"""
    calculations = [f"APPROX_PERCENTILE({data}, {q})" for q in quantiles]
    query = f"select {', '.join(calculations)} FROM table1"

    n = 10000
    np.random.seed(100)
    unif = np.round(np.random.uniform(0, 1000, n))
    gaus = np.random.normal(100, 30, n)
    df = pd.DataFrame(
        {
            "A": pd.Series(list(range(n)), dtype=pd.Int32Dtype()),
            "B": pd.Series(
                [None if i % 7 == 3 else unif[i] for i in range(n)],
                dtype=pd.Int32Dtype(),
            ),
            "C": pd.Series(gaus, dtype=np.float64),
        }
    )
    expected_output = pd.DataFrame(
        {i: df[data].quantile(q) for i, q in enumerate(quantiles)}, index=np.arange(1)
    )
    # 40% was chosen as an arbitrary cutoff for the relative tolerance based on
    # observations of the test data outputs.
    check_query(
        query,
        {"TABLE1": df},
        None,
        expected_output=expected_output,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
        atol=0.01,
        rtol=0.4,
    )


@pytest.mark.skip(reason="fix Pandas 3 NA/NaN mismatch in testing")
@pytest.mark.parametrize(
    "agg_cols",
    [
        pytest.param("AD", id="fast_tests_a"),
        pytest.param("BCE", id="slow_tests_b"),
        pytest.param("FHI", id="slow_tests_a", marks=pytest.mark.slow),
        pytest.param("JKL", id="slow_tests_b", marks=pytest.mark.slow),
    ],
)
def test_kurtosis_skew(agg_cols, spark_info, memory_leak_check):
    """Tests the Kurtosis and Skew functions"""
    query = (
        "SELECT "
        + ", ".join(f"Skew({col}), Kurtosis({col})" for col in agg_cols)
        + " FROM table1"
    )
    # Datasets designed to exhibit different distributions, thus producing myriad
    # cases of kurtosis and skew calculations
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [int(np.log2(i**2 + 10)) for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series([float(i) for i in range(100)]),
                "C": pd.Series([i**2 for i in range(100)], dtype=pd.Int32Dtype()),
                "D": pd.Series(
                    [None if i % 2 == 0 else i for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "E": pd.Series(
                    [None if i % 3 == 0 else float(i**2) for i in range(100)]
                ),
                "F": pd.Series([float((i**3) % 100) for i in range(100)]),
                "H": pd.Series([(i / 100) ** 0.5 for i in range(100)]),
                "I": pd.Series(
                    [np.arctanh(np.pi * (i - 49.5) / 160.5) for i in range(100)]
                ),
                "J": pd.Series([np.cbrt(i) for i in range(-49, 50)]),
                "K": pd.Series(
                    [i if i % 30 == 29 else None for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "L": pd.Series(
                    [i if i % 50 == 30 else None for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
            }
        )
    }

    def kurt_skew_refsol(cols):
        result = pd.DataFrame({"A0": [0]})
        i = 0
        for col in cols:
            result[f"A{i}"] = ctx["TABLE1"][col].skew()
            i += 1
            result[f"A{i}"] = ctx["TABLE1"][col].kurtosis()
            i += 1
        return result

    answer = kurt_skew_refsol(agg_cols)

    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    "func, results",
    [
        pytest.param(
            "BOOLOR_AGG", [None, False, True, True, True] * 2, id="boolor_agg"
        ),
        pytest.param(
            "BOOLAND_AGG", [None, False, False, True, True] * 2, id="booland_agg"
        ),
        pytest.param(
            "BOOLXOR_AGG", [None, False, True, False, True] * 2, id="boolxor_agg"
        ),
    ],
)
def test_boolor_booland_boolxor_agg(func, results, memory_leak_check):
    """Tests the BOOLOR_AGG, BOOLAND_AGG and BOOLXOR_AGG functions"""
    selects = ", ".join([f"{func}({col})" for col in "ABCDEFGHIJ"])
    query = f"SELECT {selects} FROM table1"
    # Datasets designed to exhibit different distributions, thus producing myriad
    # cases of kurtosis and skew calculations
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                # All null (int32)
                "A": pd.Series([None] * 5, dtype=pd.Int32Dtype()),
                # All zero (int32)
                "B": pd.Series([0] * 5, dtype=pd.Int32Dtype()),
                # One nonzero (int32)
                "C": pd.Series([0, 0, 1, 0, 0], dtype=pd.Int32Dtype()),
                # All nonzero (int32)
                "D": pd.Series([i + 1 for i in range(5)], dtype=pd.Int32Dtype()),
                # One nonzero, rest null (int32)
                "E": pd.Series([None] * 4 + [-1], dtype=pd.Int32Dtype()),
                # All null (float64)
                "F": pd.Series([None] * 5, dtype=np.float64),
                # All zero (float64)
                "G": pd.Series([0.0] * 5, dtype=np.float64),
                # One nonzero (float64)
                "H": pd.Series([0, 0, -1.0, 0, 0], dtype=np.float64),
                # All nonzero (float64)
                "I": pd.Series([np.tan(i + 1) for i in range(5)], dtype=np.float64),
                # One nonzero, rest null (float64)
                "J": pd.Series([2.71828] + [None] * 4, dtype=np.float64),
            }
        )
    }

    answer = pd.DataFrame(dict(enumerate(results)), index=np.arange(1))

    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    "col, expected",
    [
        pytest.param(
            pd.Series([None] * 20, dtype=pd.Int32Dtype()),
            (None, None, None),
            id="all_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([1, 2, 4, 63, 4, None, 20], dtype=pd.Int32Dtype()),
            (63, 0, 40),
            id="ints",
        ),
        pytest.param(
            pd.Series([10, 253, 253, None, 42], dtype=pd.UInt8Dtype()),
            (255, 8, 32),
            id="unsigned",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([0, 2, 62, 60, 20, 16, 4], dtype=pd.Int32Dtype()),
            (62, 0, 0),
            id="ints-xor",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([-64, 1, 4, 8, 1, 4], dtype=pd.Int32Dtype()),
            (-51, 0, -56),
            id="neg_ints",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [3625133335, 7285961799, 4755749177, 7850278502], dtype=pd.Int64Dtype()
            ),
            (8522825599, 268443648, 7026152975),
            id="big_ints",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([2.0, 50.5, None, 602.4, 61.6], dtype=pd.Float32Dtype()),
            (639, 2, 597),
            id="floats",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(["2", "50.5", "601.5", None, "2"]),
            (635, 2, 617),
            id="strings",
        ),
    ],
)
def test_bit_agg(col, expected, memory_leak_check):
    """Tests the BITOR_AGG, BITAND_AGG, and BITXOR_AGG aggregation functions
    without groupby on string data.

    Args:
        col (pd.Series): Input column
        expected (int): Expected output
        memory_leak_check (): Fixture, see `conftest.py`.
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": col,
            }
        )
    }

    query = "SELECT bitor_agg(A), bitand_agg(A), bitxor_agg(A) from table1"

    expected_df = pd.DataFrame(
        {
            0: expected[0],
            1: expected[1],
            2: expected[2],
        },
        index=np.arange(1),
    )

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_df,
        is_out_distributed=False,
    )


def test_all_null(memory_leak_check):
    """
    Tests that no-groupby aggregations work correctly when the
    data is all-null on the remaining functions.

    Todo items:
    [BSE-2183] ensure numeric no-groupby aggregations return null on all-null data
    [BSE-2184] fix listagg segfault in no-groupby on all-null data
    """
    selects = [
        # [BSE-2183]
        # "SUM(I) as SU",
        # "MIN(I) as MI",
        # "MAX(I) as MA",
        # "VARIANCE(I) as V",
        # "VARIANCE_POP(I) as VP",
        # "STDDEV(I) as S",
        # "STDDEV_POP(I) as SP",
        # "SKEW(I) as SK",
        # "KURTOSIS(I) as KU",
        # "MEDIAN(I) as ME",
        # "APPROX_PERCENTILE(I, 0.5) as AP",
        # "ANY_VALUE(I) as AV",
        "COUNT(*) as CS",
        "COUNT(B) as C",
        "COUNT_IF(B) as CI",
        "BOOLOR_AGG(B) as BO",
        "BOOLAND_AGG(B) as BA",
        "BOOLXOR_AGG(B) as BX",
        "BITOR_AGG(I) as BIO",
        "BITAND_AGG(I) as BIA",
        "BITXOR_AGG(I) as BIX",
        # [BSE-2184]
        # "LISTAGG(I::varchar, ',') as LA",
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY I) as PC",
        "PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY I) as PD",
    ]
    query = f"SELECT {', '.join(selects)} FROM table1"
    df = pd.DataFrame(
        {
            "I": pd.array([None] * 10, dtype=pd.Int32Dtype()),
            "B": pd.array([None] * 10, dtype=pd.BooleanDtype()),
        }
    )

    expected = pd.DataFrame(
        {
            # [BSE-2183]
            # "SU": pd.Series([None], dtype=pd.Int32Dtype()),
            # "MI": pd.Series([None], dtype=pd.Int32Dtype()),
            # "MA": pd.Series([None], dtype=pd.Int32Dtype()),
            # "V": pd.Series([None], dtype=pd.Int32Dtype()),
            # "VP": pd.Series([None], dtype=pd.Int32Dtype()),
            # "S": pd.Series([None], dtype=pd.Int32Dtype()),
            # "SP": pd.Series([None], dtype=pd.Int32Dtype()),
            # "SK": pd.Series([None], dtype=pd.Int32Dtype()),
            # "KU": pd.Series([None], dtype=pd.Int32Dtype()),
            # "ME": pd.Series([None], dtype=pd.Int32Dtype()),
            # "AP": pd.Series([None], dtype=pd.Int32Dtype()),
            # "AV": pd.Series([None], dtype=pd.Int32Dtype()),
            "CS": pd.Series([10], dtype=pd.Int32Dtype()),
            "C": pd.Series([0], dtype=pd.Int32Dtype()),
            "CI": pd.Series([0], dtype=pd.Int32Dtype()),
            "BO": pd.Series([None], dtype=pd.Int32Dtype()),
            "BA": pd.Series([None], dtype=pd.Int32Dtype()),
            "BX": pd.Series([None], dtype=pd.Int32Dtype()),
            "BIO": pd.Series([None], dtype=pd.Int32Dtype()),
            "BIA": pd.Series([None], dtype=pd.Int32Dtype()),
            "BIX": pd.Series([None], dtype=pd.Int32Dtype()),
            # [BSE-2184]
            # "LA": pd.Series([""]),
            "PC": pd.Series([None], dtype=pd.Int32Dtype()),
            "PD": pd.Series([None], dtype=pd.Int32Dtype()),
        }
    )

    check_query(
        query,
        {"TABLE1": df},
        None,
        expected_output=expected,
        check_dtype=False,
        sort_output=False,
        is_out_distributed=False,
    )
