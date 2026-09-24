import os
import tempfile
from io import StringIO

import numba
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.spawn.utils import run_rank0
from bodo.tests.iceberg_database_helpers.utils import (
    SparkAwsIcebergCatalog,
    get_spark,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    assert_tables_equal,
    check_func,
    gen_unique_table_id,
    get_rest_catalog_connection_string,
    pytest_polaris,
    temp_env_override,
)
from bodosql.bodosql_types.rest_catalog_ext import get_REST_connection
from bodosql.tests.test_types.utils import create_iceberg_table

pytestmark = pytest_polaris


def test_basic_read(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an entire Iceberg table from Polaris in SQL
    """
    catalog = polaris_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    query = "SELECT A, B, C FROM CI.BODOSQL_ICEBERG_READ_TEST"
    check_func(
        impl,
        (bc, query),
        py_output=polaris_catalog_iceberg_read_df,
        sort_output=True,
        reset_index=True,
    )


def test_column_pruning(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg table from Polaris in SQL
    where columns are pruned and reordered
    """
    catalog = polaris_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "B": polaris_catalog_iceberg_read_df["B"],
            "A": polaris_catalog_iceberg_read_df["A"],
        }
    )

    query = "SELECT B, A FROM CI.BODOSQL_ICEBERG_READ_TEST"
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


def test_filter_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg table from Polaris with filter pushdown
    """

    catalog = polaris_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = (
        polaris_catalog_iceberg_read_df.where(
            polaris_catalog_iceberg_read_df["A"].str.contains("a")
            & polaris_catalog_iceberg_read_df["B"]
            > 0
        )[["B", "A"]]
        .dropna()
        .reset_index(drop=True)
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
            "Iceberg Filter Pushed Down:\npie.And(pie.GreaterThan('B', literal(f0)), pie.NotNull('A'))",
        )


def test_filter_pushdown_col_not_read(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading a Iceberg table with BodoSQL filter pushdown
    where a column used in the filter is not read in
    """
    catalog = polaris_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = (
        polaris_catalog_iceberg_read_df.where(
            polaris_catalog_iceberg_read_df["A"].str.contains("a")
            & polaris_catalog_iceberg_read_df["B"]
            > 0
        )[["A"]]
        .dropna()
        .reset_index(drop=True)
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
            "Iceberg Filter Pushed Down:\npie.And(pie.GreaterThan('B', literal(f0)), pie.NotNull('A'))",
        )


def parse_table_property(s):
    """Given property column from "DESCRIBE TABLE EXTENDED", parse it into dict
        There might be builtin functions to extract proerties, but currently
        we implement our own version.

        Input string is in format of "['Foo'='A','Bar'='B, C, D']".
        Since "," can appear in values or keys, we can't simply use s.split(",") and
        then s.split('=').
        But a pattern we observe is when serving as key-value separators there are no
        spaces around ",". So we customize this split(",") to only split on "," with
        no subsequent space.

    Args:
        s (string): Input table property string

    Returns:
        dict: A dictionary with key-value pairs of table properties in string
    """

    i = 1
    properties = {}
    while i < len(s) - 1:
        j = i + 1
        while j < len(s) - 1 and (s[j] != "," or s[j + 1] == " "):
            j += 1
        properties[s[i:j].split("=")[0]] = s[i:j].split("=")[1]
        i = j + 1
    return properties


def check_table_comment(
    polaris_connection,
    schema,
    table_name,
    number_columns,
    table_comments: str | None = None,
    column_comments=False,
    table_properties: dict | None = None,
):
    """Helper function to test table comments are correctly added

    Args:
        polaris_connection:
        schema (_type_): Databse schema
        table_name (_type_): Table name
        number_columns (int): Number of columns of the table
        table_comments (bool, optional): Whether the test case will test table comments. Defaults to False.
        column_comments (bool, optional): Whether the test case will test column comments. Defaults to False.
    """

    if bodo.get_rank() != 0:
        return

    uri, warehouse, credential = polaris_connection
    spark = get_spark(
        SparkAwsIcebergCatalog(
            catalog_name=warehouse,
            warehouse=warehouse,
            uri=uri,
            credential=credential,
        )
    )
    table_cmt = (
        spark.sql(f"DESCRIBE TABLE EXTENDED {schema}.{table_name}")
        .filter("col_name = 'Comment'")
        .select("data_type")
        .head()
    )
    if table_comments is not None:
        assert table_cmt[0] == table_comments, (
            f'Expected table comment to be "{table_comments}", got "{table_cmt}"'
        )

    df = spark.sql(f"DESCRIBE TABLE {schema}.{table_name}").toPandas()
    for i in range(number_columns):
        if not column_comments or i % 2 == 1:
            assert pd.isna(df.iloc[i]["comment"]), (
                f"Expected column {i} comment to be None, but actual comment is not None"
            )
        else:
            assert df.iloc[i]["comment"] == f"{table_name}_test_colcmt_{i}", (
                f'Expected column {i} comment to be "{table_name}_test_colcmt_{i}", got a different one'
            )

    if table_properties is not None:
        str = (
            spark.sql(f"DESCRIBE TABLE EXTENDED {schema}.{table_name}")
            .filter("col_name = 'Table Properties'")
            .select("data_type")
            .head()[0]
        )
        parsed_properties = parse_table_property(str)

        for keys in table_properties:
            assert keys in parsed_properties, (
                f"Expected table properties {keys}, find nothing"
            )
            assert parsed_properties[keys] == table_properties[keys], (
                f"Expected key {keys} with value {table_properties[keys]}, got {parsed_properties[keys]}"
            )


@pytest.mark.parametrize("column_comments", [True, False])
@pytest.mark.parametrize("table_properties", [True, False])
@pytest.mark.parametrize("table_comments", ["test_tbl_comments", "", None])
@pytest.mark.parametrize("construct", ["CREATE", "CREATE OR REPLACE"])
def test_rest_catalog_iceberg_write(
    polaris_catalog,
    polaris_connection,
    table_comments,
    # Here we are testing different combinations of column comments, so the comments are generated later
    column_comments,
    table_properties,
    construct,
    memory_leak_check,
):
    """tests that writing tables works"""

    rest_uri, polaris_warehouse, polaris_credential = polaris_connection
    catalog = polaris_catalog
    bc = bodosql.BodoSQLContext(catalog=catalog)
    schema = "CI"
    con_str = get_rest_catalog_connection_string(
        rest_uri, polaris_warehouse, polaris_credential
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

    column_comment = [
        f"{table_name}_test_colcmt_{i}" if i % 2 == 0 else None
        for i in range(len(in_df.columns))
    ]
    if not column_comments:
        column_comment_str = ""
    else:
        column_comment_str = "("
        for i in range(len(column_comment)):
            column_comment_str += f"c{i + 1} "
            column_comment_str += chr(65 + i)
            column_comment_str += (
                "" if column_comment[i] is None else f" comment '{column_comment[i]}'"
            )
            if i < len(column_comment) - 1:
                column_comment_str += ", "
        column_comment_str += ")"
    table_property = (
        "TBLPROPERTIES ('TBLPROPERTIES' = 'A', 'TEST_TBL' = 'T', 'TEST_TBL' = 'TRUE')"
        if table_properties
        else ""
    )
    ref_table_properties = (
        None if not table_properties else {"TBLPROPERTIES": "A", "TEST_TBL": "TRUE"}
    )
    table_comment = (
        f"COMMENT = '{table_comments}'" if table_comments is not None else ""
    )

    ctas_query = f"{construct} TABLE {schema}.{table_name} {column_comment_str} {table_comment} {table_property} AS SELECT * from __bodolocal__.table1"
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
        check_table_comment(
            polaris_connection,
            schema,
            table_name,
            len(output_df.columns),
            table_comments=table_comments,
            column_comments=column_comments,
            table_properties=ref_table_properties,
        )
        assert_tables_equal(output_df, in_df, check_dtype=False)

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call delete_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        con_str = get_rest_catalog_connection_string(
            rest_uri, polaris_warehouse, polaris_credential
        )
        py_catalog = conn_str_to_catalog(con_str)
        try:
            run_rank0(lambda: py_catalog.purge_table(f"{schema}.{table_name}"))()
        except Exception:
            if exception_occurred_in_test_body:
                pass
            else:
                raise


def test_limit_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg from Polaris with limit pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = polaris_catalog
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


def test_limit_filter_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg table from Polaris with limit + filter pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = polaris_catalog
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
            "Iceberg Filter Pushed Down:\npie.GreaterThan('B', literal(f0))",
        )


def test_multi_limit_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Verify multiple limits are still simplified even though Iceberg trees
    only support a single limit.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """
    catalog = polaris_catalog
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


def test_limit_filter_limit_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg table from Polaris with limit pushdown. We can push down
    both limits and filters in a way that meets the requirements of this query
    (pushes the smallest limit and ensures the filter is applied).

    This may not result in a correct result since the ordering is not defined.
    """
    catalog = polaris_catalog
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
            "Iceberg Filter Pushed Down:\npie.GreaterThan('B', literal(f0))",
        )


def test_filter_limit_filter_pushdown(
    memory_leak_check, polaris_catalog, polaris_catalog_iceberg_read_df
):
    """
    Test reading an Iceberg table from Polaris with filters after the limit
    computes a valid result (enforcing the limit and the filters). This query
    doesn't have a strict ordering since limit can return any result and we opt
    to apply the filter then limit (which is always correct but may be suboptimal).
    """

    catalog = polaris_catalog
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
            "Iceberg Filter Pushed Down:\npie.And(pie.NotEqualTo('A', literal(f0)), pie.GreaterThan('B', literal(f1)))",
        )


def test_dynamic_scalar_filter_pushdown(
    memory_leak_check, polaris_catalog, polaris_connection
):
    """
    Test that a dynamically generated filter can be pushed down to Iceberg.
    """
    rest_url, warehouse, credential = polaris_connection
    conn_str = get_rest_catalog_connection_string(rest_url, warehouse, credential)
    catalog = polaris_catalog
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
    conn_str_to_catalog(conn_str)
    table_name = "current_date_table"
    table_id = f"{schema}.{table_name}"
    with create_iceberg_table(conn_str, table_id, input_df):
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
                "Iceberg Filter Pushed Down:\npie.LessThanOrEqual('A', literal(f0))",
            )


def test_rest_catalog_token_caching(memory_leak_check):
    prev_cache_loc = numba.config.CACHE_DIR
    try:
        tempdir = run_rank0(tempfile.TemporaryDirectory)()
        cache_loc = tempdir.name
        # In certain cases, numba reloads its config variables from the
        # environment. In those cases, the above line would be overridden.
        # Therefore, we also set it to the env var that numba reloads from.
        with temp_env_override(
            {"NUMBA_CACHE_DIR": cache_loc, "__BODOSQL_REST_TOKEN": "test_token1"}
        ):
            numba.config.CACHE_DIR = cache_loc

            def f():
                tc = get_REST_connection(
                    "test_uri", "test_warehouse", "PRINCIPAL_ROLE:ALL"
                )
                return tc.conn_str

            dispatcher = bodo.jit(cache=True)(f)
            assert (
                dispatcher()
                == "iceberg+test_uri?warehouse=test_warehouse&scope=PRINCIPAL_ROLE:ALL&token=test_token1&sigv4=false"
            )
            sig = dispatcher.signatures[0]
            assert dispatcher._cache_hits[sig] == 0, (
                "Expected no cache hit for function signature"
            )

            # Ensure that the cache is shared across all ranks
            bodo.barrier()

            dispatcher_2 = bodo.jit(cache=True)(f)
            os.environ["__BODOSQL_REST_TOKEN"] = "test_token2"
            assert (
                dispatcher_2()
                == "iceberg+test_uri?warehouse=test_warehouse&scope=PRINCIPAL_ROLE:ALL&token=test_token2&sigv4=false"
            )
            sig = dispatcher_2.signatures[0]
            assert dispatcher_2._cache_hits[sig] == 1, (
                "Expected a cache hit for function signature"
            )

    finally:
        # Ensure all ranks are done before cleanup
        bodo.barrier()
        run_rank0(tempdir.cleanup)()
        numba.config.CACHE_DIR = prev_cache_loc
