from io import StringIO

import pandas as pd

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    assert_tables_equal,
    check_func,
    create_tabular_iceberg_table,
    gen_unique_table_id,
    get_rest_catalog_connection_string,
    pytest_tabular,
)
from bodo.utils.utils import run_rank0

pytestmark = pytest_tabular


def test_basic_read(memory_leak_check, tabular_catalog):
    """
    Test reading an entire Iceberg table from Tabular in SQL
    """
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "A": pd.array(["ally", "bob", "cassie", "david", pd.NA]),
            "B": pd.array([10.5, -124.0, 11.11, 456.2, -8e2], dtype="float64"),
            "C": pd.array([True, pd.NA, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )

    query = "SELECT A, B, C FROM CI.BODOSQL_ICEBERG_READ_TEST"
    check_func(
        impl,
        (bc, query),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def test_column_pruning(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg table from Tabular in SQL
    where columns are pruned and reordered
    """
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "B": [10.5, 124.0, 11.11, 456.2, 8e2],
            "A": ["ally", "bob", "cassie", "david", pd.NA],
        }
    )

    query = "SELECT ABS(B) as B, A FROM CI.BODOSQL_ICEBERG_READ_TEST"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")


def test_filter_pushdown(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg table from Tabular with filter pushdown
    """

    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "B": [10.5, 11.11, 456.2],
            "A": ["ally", "cassie", "david"],
        }
    )

    # A IS NOT NULL can be pushed down to iceberg
    # LIKE A '%a%' cannot be pushed down to iceberg
    query = "SELECT ABS(B) as B, A FROM CI.BODOSQL_ICEBERG_READ_TEST WHERE B > 0 AND A LIKE '%a%'"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('AND', [bic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f0)]), bic.FilterExpr('IS_NOT_NULL', [bic.ColumnRef('A')])])",
        )


def test_filter_pushdown_col_not_read(memory_leak_check, tabular_catalog):
    """
    Test reading a Iceberg table with BodoSQL filter pushdown
    where a column used in the filter is not read in
    """
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "cassie", "david"],
        }
    )

    # A IS NOT NULL can be pushed down to iceberg
    # LIKE A '%a%' cannot be pushed down to iceberg
    query = "SELECT A FROM CI.BODOSQL_ICEBERG_READ_TEST WHERE B > 0 AND A LIKE '%a%'"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A']")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('AND', [bic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f0)]), bic.FilterExpr('IS_NOT_NULL', [bic.ColumnRef('A')])])",
        )


def test_tabular_catalog_iceberg_write(
    tabular_catalog, tabular_connection, memory_leak_check
):
    """tests that writing tables works"""
    import bodo_iceberg_connector as bic

    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)
    schema = "CI"
    con_str = get_rest_catalog_connection_string(
        rest_uri, tabular_warehouse, tabular_credential
    )

    in_df = pd.DataFrame(
        {
            "ints": list(range(100)),
            "floats": [float(x) for x in range(100)],
            "str": [str(x) for x in range(100)],
            "dict_str": ["abc", "df"] * 50,
        }
    )
    bc = bc.add_or_replace_view("TABLE1", in_df)

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    table_name = run_rank0(
        lambda: gen_unique_table_id("bodosql_catalog_write_iceberg_table").upper()
    )()

    ctas_query = f"CREATE OR REPLACE TABLE {schema}.{table_name} AS SELECT * from __bodolocal__.table1"
    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, ctas_query),
            only_1D=True,
            py_output=5,
            use_table_format=True,
        )

        @bodo.jit
        def read_results(con_str, schema, table_name):
            output_df = pd.read_sql_table(table_name, con=con_str, schema=schema)
            return bodo.allgatherv(output_df)

        output_df = read_results(con_str, schema, table_name)
        assert_tables_equal(output_df, in_df, check_dtype=False)

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call delete_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                run_rank0(bic.delete_table)(
                    bodo.io.iceberg.format_iceberg_conn(con_str),
                    schema,
                    table_name,
                )
            except:
                pass
        else:
            run_rank0(bic.delete_table)(
                bodo.io.iceberg.format_iceberg_conn(con_str),
                schema,
                table_name,
            )


def test_limit_pushdown(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg from Tabular with limit pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [2]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 2 rows")


def test_limit_filter_pushdown(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg table from Tabular with limit + filter pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST where B > 200 LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 2 rows")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f0)])",
        )


def test_multi_limit_pushdown(memory_leak_check, tabular_catalog):
    """
    Verify multiple limits are still simplified even though Iceberg trees
    only support a single limit.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})
    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST LIMIT 2) LIMIT 1)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        # The planner should simplify the two limits into a single limit
        check_logger_msg(stream, "Constant limit detected, reading at most 1 rows")


def test_limit_filter_limit_pushdown(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg table from Tabular with limit pushdown. We can push down
    both limits and filters in a way that meets the requirements of this query
    (pushes the smallest limit and ensures the filter is applied).

    This may not result in a correct result since the ordering is not defined.
    """
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [2]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST LIMIT 4) where B > 11 LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 2 rows")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f0)])",
        )


def test_filter_limit_filter_pushdown(memory_leak_check, tabular_catalog):
    """
    Test reading an Iceberg table from Tabular with filters after the limit
    computes a valid result (enforcing the limit and the filters). This query
    doesn't have a strict ordering since limit can return any result and we opt
    to apply the filter then limit (which is always correct but may be suboptimal).
    """

    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM CI.BODOSQL_ICEBERG_READ_TEST where B > 11 LIMIT 4) where A <> 'david')"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 4 rows")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('AND', [bic.FilterExpr('!=', [bic.ColumnRef('A'), bic.Scalar(f0)]), bic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f1)])])",
        )


def test_dynamic_scalar_filter_pushdown(
    memory_leak_check, tabular_catalog, tabular_connection
):
    """
    Test that a dynamically generated filter can be pushed down to Iceberg.
    """
    _, tabular_warehouse, tabular_credential = tabular_connection
    catalog = tabular_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)
    schema = "CI"

    def impl(bc, query):
        return bc.sql(query)

    current_date = pd.Timestamp.now().date()
    # Use a large delta so we don't have to worry about the current date changing
    offsets = [-30, -20, -10, 10, 20, 30]
    column = [current_date + pd.Timedelta(days=offset) for offset in offsets]
    input_df = pd.DataFrame({"A": column})
    py_output = pd.DataFrame({"A": [x for x in column if x <= current_date]})
    with create_tabular_iceberg_table(
        input_df, "current_date_table", tabular_warehouse, schema, tabular_credential
    ) as table_name:
        query = f'SELECT * FROM {schema}."{table_name}" WHERE A <= CURRENT_DATE'
        stream = StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, query),
                py_output=py_output,
                sort_output=True,
                reset_index=True,
            )
            # Verify filter pushdown
            check_logger_msg(
                stream,
                "Iceberg Filter Pushed Down:\nbic.FilterExpr('<=', [bic.ColumnRef('A'), bic.Scalar(f0)])",
            )
