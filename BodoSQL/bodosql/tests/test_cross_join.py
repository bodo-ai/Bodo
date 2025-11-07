"""Tests cross join"""

import pytest

import bodosql
from bodo.tests.utils import pytest_slow_unless_join, temp_env_override
from bodosql.tests.utils import check_query

# Skip unless any join-related files were changed
pytestmark = pytest_slow_unless_join


@pytest.mark.parametrize("broadcast", [True, False])
def test_cross_join_simpl(basic_df, spark_info, broadcast: bool, memory_leak_check):
    """
    Tests a simple cross join on small tables
    """
    ctx = {
        "TABLE1": basic_df["TABLE1"],
        "TABLE2": basic_df["TABLE1"],
    }
    query = "Select table1.C, table2.B from table1 cross join table2"
    # Set BODO_BCAST_JOIN_THRESHOLD to 0 to force non-broadcast (fully-parallel) case,
    # and to 1GiB to force the broadcast join case.
    with temp_env_override(
        {"BODO_BCAST_JOIN_THRESHOLD": "0" if not broadcast else str(1024 * 1024 * 1024)}
    ):
        check_query(
            query,
            ctx,
            spark_info,
            check_dtype=False,
            check_names=False,
        )


@pytest.mark.slow
def test_nested_cross_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested cross joins work properly with all types,
    and with nullable values in one or more tables"""

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["T1", "T2", "T3"]
    else:
        convert_columns_bytearray = None

    query = """
    SELECT
        table3.Y as T1, table4.A as T2, table4.C as T3
    FROM
        table3
    JOIN
        (select table5.A, C from table1 as not_table_1 cross join (select table1.A from table1 RIGHT join table2 on table1.A = table2.A) as table5 ) as table4
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


@pytest.mark.slow
def test_cross_join_select_star(join_dataframes, spark_info, memory_leak_check):
    """
    Tests a simple cross join + select star
    """

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        # Have to skip, as we can't cast the columns without explicitly providing the names
        # (BodoSQL will assign temp column names)
        pytest.skip()
    query = "Select * from (SELECT A as A_1, B as B_1, C as C_1 from table1) cross join table2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_cross_join_select_filter(join_dataframes, spark_info, memory_leak_check):
    """
    Tests a simple cross join + filter
    """

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["C1", "C2"]
    else:
        convert_columns_bytearray = None

    query = "Select table1.B as C1, table3.Y as C2 from table1 cross join table3 where table1.A = table3.Y"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_nested_cross_join_with_filter_select_star(
    join_dataframes, spark_info, memory_leak_check
):
    """tests that nested cross joins work properly with all types,
    and with nullable values in one or more tables, with filters and select star's"""

    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["T1", "T2"]
    else:
        convert_columns_bytearray = None

    query = """
    SELECT
        table4.Y as T1, table4.A as T2
    FROM
        (SELECT * from table3 FULL OUTER join (select table1.A, table2.B from table1 cross join table2 where table1.A = table2.B) as table5 on table5.A = table3.Y) as table4
    WHERE
        table4.Y = table4.A and table4.Y = table4.B
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_cross_join_error(basic_df, spark_info, memory_leak_check):
    """
    Tests that we throw a reasonable error if the cross join syntax is wrong
    """
    ctx = {
        "TABLE1": basic_df["TABLE1"],
        "TABLE2": basic_df["TABLE1"],
    }
    query = "select table1.a from table1 cross join on table1.A = table2.B table2"
    bc = bodosql.BodoSQLContext(ctx)
    # NOTE: the error message will be Encountered "join on"... if/when
    # we revert the bodo parser to using a lookahead of 1
    # https://bodo.atlassian.net/browse/BE-4404
    with pytest.raises(ValueError, match='.*Encountered "cross join on".*'):
        bc.sql(query)
