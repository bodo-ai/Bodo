"""
Test correctness of SQL join queries on BodoSQL
"""

import copy
import io
import os
from datetime import date

import numba
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    pytest_mark_one_rank,
    pytest_slow_unless_join,
    temp_env_override,
)
from bodo.utils.typing import BodoError
from bodosql.tests.utils import check_query

# Skip unless any join-related files were changed
pytestmark = pytest_slow_unless_join


@pytest.fixture(params=["INNER", "LEFT", "RIGHT", "FULL OUTER"])
def join_type(request):
    return request.param


@pytest.mark.bodosql_cpp
def test_join(
    join_dataframes, spark_info, join_type, comparison_ops, memory_leak_check
):
    """test simple join queries"""

    # TODO[BSE-5149]: support non-equi joins in BodoSQL C++ backend
    # TODO[BSE-5151]: support filter in BodoSQL C++ backend to enable non-outer join plans
    if bodosql.use_cpp_backend and (comparison_ops != "=" or join_type != "FULL OUTER"):
        return

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    if comparison_ops == "<=>":
        # TODO: Add support for <=> from general-join cond
        return
    query = f"select table1.B, C, D from table1 {join_type} join table2 on table1.A {comparison_ops} table2.A"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
        use_duckdb=True,
    )


def test_multitable_join_cond(join_dataframes, spark_info, memory_leak_check):
    """tests selecting from multiple tables based upon a where clause"""
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["A", "B"]
    else:
        convert_columns_bytearray = None
    check_query(
        "select table1.A, table2.B from table1, table2 where table2.B > table2.A",
        join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_join_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Test that checks that joining two tables that share a column name
    can be merged if aliased.
    """
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["A1", "A2"]
    else:
        convert_columns_bytearray = None
    query = """SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_natural_join(join_dataframes, spark_info, join_type, memory_leak_check):
    """test simple natural join queries"""
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select table1.B, C, D from table1 NATURAL {join_type} join table2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
        use_duckdb=True,
    )


@pytest.mark.slow
def test_and_join(join_dataframes, spark_info, memory_leak_check):
    """
    Query that demonstrates that a join with an AND expression
    will merge on a common column, rather than just merge the entire tables.
    """
    query = """
        SELECT
            table1.A, table2.B
        from
            table1, table2
        where
            (table1.A = table2.A and table1.B = table2.B)
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        # TODO[BE-3478]: enable dict-encoded string test when fixed
        use_dict_encoded_strings=False,
    )


def test_or_join(join_dataframes, spark_info, memory_leak_check):
    """
    Query that demonstrates that a join with an OR expression and common conds
    will merge on the common cond, rather than just merge the entire tables.
    """

    if isinstance(join_dataframes["TABLE1"]["A"][0], bytes):
        byte_array_cols = ["A", "B"]
    else:
        byte_array_cols = []
    query = """
        SELECT
            table1.A, table2.B
        from
            table1, table2
        where
            (table1.A = table2.A or table1.B = table2.B)
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        convert_columns_bytearray=byte_array_cols,
    )


def test_join_types(join_dataframes, spark_info, join_type, memory_leak_check):
    """test all possible join types"""
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select table2.B, C, D from table1 {join_type} join table2 on table1.A = table2.A"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_join_different_size_tables(
    join_dataframes, spark_info, join_type, memory_leak_check
):
    """tests that join operations still works when the dataframes have different sizes"""
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    df = pd.DataFrame({"A": [1, 2, 3]})
    copied_join_dataframes = copy.copy(join_dataframes)

    copied_join_dataframes["TABLE3"] = df
    query = f"select table2.B, C, D from table1 {join_type} join table2 on table1.A = table2.A"
    check_query(
        query,
        copied_join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nested_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work properly"""

    # for context, the nested right join should create a number of null values in table4.A,
    # which we then use in the join condition for the top level join
    # the null values in table4.A shouldn't match to anything, and shouldn't raise an error

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["T1", "T2"]
    else:
        convert_columns_bytearray = None

    query = """
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3
    JOIN
        (select table1.A from table1 RIGHT join table2 on table1.A = table2.A) as table4
    ON
        table4.A = table3.Y
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nested_or_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work with implicit joins using 'or'"""

    # for context, the nested outer join should create a number of null values in table4.A and table4.B,
    # which we then use in the join condition for the top level join
    # assumedly, the null values in table4.A/B shouldn't match to anything, and shouldn't raise an error
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["T1", "T2"]
    else:
        convert_columns_bytearray = None

    query = """
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A) as table4
    WHERE
        table3.Y = table4.A or table3.Y = table4.B
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nested_and_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work with implicit joins using 'and'"""

    # for context, the nested right join should create a number of null values in table4.A,
    # which we then use in the join condition for the top level join
    # assumedly, the null values in table4.A should match to anything, and shouldn't raise an error
    query = """
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A) as table4
    WHERE
        table3.Y = table4.A and table3.Y = table4.B
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_join_boolean(bodosql_boolean_types, spark_info, join_type, memory_leak_check):
    """test all possible join types on boolean table"""

    newCtx = {
        "TABLE1": bodosql_boolean_types["TABLE1"],
        "TABLE2": bodosql_boolean_types["TABLE1"],
    }
    query = f"select table1.B, table2.C from table1 {join_type} join table2 on table1.A"
    check_query(
        query,
        newCtx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_multi_key_join_types(
    join_dataframes, spark_info, join_type, memory_leak_check
):
    """test that for all possible join types "and equality conditions" turn into multi key join"""
    # Note: We don't check the generated code because column ordering isn't deterministic
    # Join code doesn't properly trim the filter yet, so outer joins will drop any NA columns
    # when applying the filter.
    query = f"select C, D from table1 {join_type} join table2 on table1.A = table2.A and table1.B = table2.B"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        use_duckdb=True,
    )


@pytest.mark.slow
def test_trimmed_multi_key_cond_inner_join(
    join_dataframes, spark_info, memory_leak_check
):
    """test that with inner join, equality conditions that are used in AND become keys and don't appear in the filter."""
    query = "select C, D from table1 inner join table2 on table1.A = table2.A and table1.B < table2.B"
    # Note: We don't check the generated code because column ordering isn't deterministic
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        use_duckdb=True,
    )


def test_nonascii_in_implicit_join(spark_info, memory_leak_check):
    """
    Tests using non-ascii in an implicit join via select distinct.
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "D": pd.Series(
                    list(pd.date_range("2011", "2018", 5, unit="ns")) * 20,
                    dtype="datetime64[ns]",
                ),
                "S": pd.Series(
                    [
                        None if i % 7 == 0 else chr(65 + (i**2) % 8 + i // 48)
                        for i in range(100)
                    ]
                ),
            }
        ),
        "TABLE2": pd.DataFrame(
            {
                "T": pd.Series(
                    [
                        a + b + c + d
                        for a in ["", *"ALPHABETS♫UP"]
                        for b in ["", *"ÉPSI∫øN"]
                        for c in ["", *"ZE฿Rä"]
                        for d in "THETA"
                    ]
                )
            }
        ),
    }

    query = """
    SELECT
        S,
        D,
        COUNT(*)
    FROM table1
    WHERE s IN (SELECT DISTINCT LEFT(t, 1) FROM table2)
    GROUP BY s, d
    """

    check_query(query, ctx, spark_info, check_names=False, check_dtype=False)


def test_tz_aware_join(representative_tz, memory_leak_check):
    """
    Test join, including non-equality, on tz_aware data
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="4D7h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022",
                    freq="12D21h",
                    periods=20,
                    tz=representative_tz,
                    unit="ns",
                )
            ),
            "C": pd.date_range(
                start="3/1/2022", freq="1h", periods=34, tz=representative_tz, unit="ns"
            ),
            "D": pd.date_range(
                start="1/1/2022",
                freq="14D20min",
                periods=34,
                tz=representative_tz,
                unit="ns",
            ),
        }
    )
    query = """
        select
            t1.A as A,
            t2.B as B,
            t1.C as C,
            t2.D as D
        FROM
            table1 t1
        JOIN table2 t2
            on t1.A = t2.B and t2.C > t1.B
    """
    ctx = {
        "TABLE1": df,
        "TABLE2": df,
    }
    py_output = df.merge(df, left_on="A", right_on="B")
    # Drop nulls to match SQL
    py_output = py_output[py_output.A_x.notna()]
    # Add the comparison
    py_output = py_output[py_output.C_y > py_output.B_x]
    py_output = py_output[["A_x", "B_y", "C_x", "D_y"]]
    py_output.columns = ["A", "B", "C", "D"]
    check_query(
        query, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


def test_join_pow(spark_info, join_type, memory_leak_check):
    """
    Make sure pow() works inside join conditions
    """
    df1 = pd.DataFrame({"A": [2, 4, 3] * 4, "B": [3.1, 2.2, 0.1] * 4})
    df2 = pd.DataFrame({"C": [1, 2] * 3, "D": [1.1, 3.3] * 3})
    query1 = f"select * from ARG1 {join_type} join ARG2 on pow(ARG1.A - ARG2.C, 2) > 11"
    query2 = f"select * from ARG1 {join_type} join ARG2 on pow(pow(ARG1.A - ARG2.C, 2) + pow(ARG1.B - ARG2.D,2),.5)<2"
    ctx = {
        "ARG1": df1,
        "ARG2": df2,
    }
    check_query(query1, ctx, spark_info, check_dtype=False, check_names=False)
    check_query(query2, ctx, spark_info, check_dtype=False, check_names=False)


def test_interval_join_compilation(memory_leak_check):
    """
    Tests that the Interval Join detection code correctly determines that
    Interval Join should be used in this case. This is useful for ensuring:
        * That Bodo can handle BodoSQL column names (EXPR$1, ...)
        * That BodoSQL performs the casts as a projection before the join.
          Interval join currently does not support operations inside of the condition
    """
    if bodo.bodosql_use_streaming_plan:
        # Ignore this test when using streaming plan
        return

    df1 = pd.DataFrame(
        {
            "P": [date(2023, 1, 1)],
        }
    )
    df2 = pd.DataFrame(
        {
            "L": pd.date_range(
                start="2023-01-01", periods=10, freq="D", unit="ns"
            ).to_series(),
            "R": pd.date_range(
                start="2023-01-01", periods=10, freq="D", unit="ns"
            ).to_series(),
        }
    )
    bc = bodosql.BodoSQLContext({"ARG1": df1, "ARG2": df2})
    query = "select P, L from ARG1 inner join ARG2 on ARG1.P >= ARG2.L::date and ARG1.P < ARG2.R::date"

    def impl(bc, query):
        return bc.sql(query)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit((bodo.typeof(bc), numba.types.literal(query)))(impl)
        check_logger_msg(stream, "Using optimized interval range join")


@pytest.mark.slow
def test_join_div(spark_info, join_type, memory_leak_check):
    """
    Make sure div operation works inside join conditions
    """
    df1 = pd.DataFrame({"A": [2, 4, 3] * 4, "B": [3.1, 2.2, 0.1] * 4})
    df2 = pd.DataFrame({"A": [1, 2] * 4, "D": [1.1, 3.3] * 4})
    query1 = f"select B from T1 {join_type} join T2 on true where T1.B / T2.D > 2.0"
    ctx = {
        "T1": df1,
        "T2": df2,
    }
    check_query(query1, ctx, spark_info, check_names=False)


@pytest_mark_one_rank
def test_join_invalid_condition(memory_leak_check):
    """
    Make sure an invalid join condition throws a proper error.
    """
    df1 = pd.DataFrame({"A": pd.array([2, 4, 3, None] * 4, dtype="Int64")})
    df2 = pd.DataFrame({"C": pd.array([1, None, 2] * 3, dtype="Int64")})
    query1 = "select * from ARG1 full outer join ARG2 on COALESCE(ARG1.A, ARG2.C) = 11"
    ctx = {
        "ARG1": df1,
        "ARG2": df2,
    }
    with pytest.raises(
        BodoError,
        match=".* Encountered an unsupported join condition in an outer join that cannot be rewritten",
    ):
        check_query(
            query1, ctx, None, check_dtype=False, check_names=False, expected_output=1
        )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.skipif(
    "AGENT_NAME" in os.environ, reason="Assertion fails on Azure only [BSE-3585]"
)
def test_join_broadcast_hint(memory_leak_check, capfd):
    """
    Test that providing a broadcast hint for a join provides
    a runtime broadcast even though the testing infrastructure
    disables the dynamic broadcast decision
    """
    df1 = pd.DataFrame({"A": pd.array([2, 4, 3, None] * 4, dtype="Int64")})
    df2 = pd.DataFrame({"A": pd.array([1, None, 2] * 3, dtype="Int64")})
    query = "select /*+ broadcast(t1) */ t1.A as OUTPUT from table1 t1 join table2 t2 on t1.A = t2.A"
    ctx = {
        "TABLE1": df1,
        "TABLE2": df2,
    }
    py_output = pd.DataFrame({"OUTPUT": pd.array([2] * 12, dtype="Int64")})

    with temp_env_override({"BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1"}):
        check_query(
            query,
            ctx,
            None,
            expected_output=py_output,
            check_dtype=False,
            check_names=False,
            # Sequential execution won't trigger the "broadcast" message
            only_jit_1DVar=True,
        )
        stdout, stderr = capfd.readouterr()
        if bodo.get_rank() == 0:
            expected_log_messages = ["Converting to a broadcast hash join"]
        else:
            expected_log_messages = [None]

        comm = MPI.COMM_WORLD
        for expected_log_message in expected_log_messages:
            assert_success = True
            if expected_log_message is not None:
                assert_success = expected_log_message in stderr
            assert_success = comm.allreduce(assert_success, op=MPI.LAND)
            assert assert_success
