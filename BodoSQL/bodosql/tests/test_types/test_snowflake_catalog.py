# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests various components of the SnowflakeCatalog type both inside and outside a
direct BodoSQLContext.
"""

import datetime
import io
import os
import random

import bodosql
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.test_datetime_fns import compute_valid_times
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    test_db_snowflake_catalog,
)
from mpi4py import MPI

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    create_snowflake_table,
    drop_snowflake_table,
    gen_unique_table_id,
    get_snowflake_connection_string,
    pytest_snowflake,
)
from bodo.utils.typing import BodoError

pytestmark = pytest_snowflake


@pytest.fixture(
    params=[
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        ),
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        ),
    ]
)
def dummy_snowflake_catalogs(request):
    """
    List of table paths that should be suppported.
    None of these actually point to valid data
    """
    return request.param


@pytest.fixture(
    params=[
        bodosql.SnowflakeCatalog(
            os.environ.get("SF_USERNAME", ""),
            os.environ.get("SF_PASSWORD", ""),
            "bodopartner.us-east-1",
            "DEMO_WH",
            "SNOWFLAKE_SAMPLE_DATA",
            connection_params={
                "schema": "TPCH_SF1",
                "query_tag": "folder=folder1+ folder2&",
            },
        )
    ]
)
def snowflake_sample_data_snowflake_catalog(request):
    """
    The snowflake_sample_data snowflake catalog used for most tests.
    Although this is a fixture there is intentionally a
    single element.
    """
    return request.param


@pytest.fixture(params=[False, True])
def use_default_schema(request):
    """
    Should this test assume the snowflake catalog has a default
    schema.
    """
    return request.param


def test_snowflake_catalog_lower_constant(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test lowering a constant snowflake catalog.
    """

    def impl():
        return dummy_snowflake_catalogs

    check_func(impl, ())


def test_snowflake_catalog_boxing(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test boxing and unboxing a table path type.
    """

    def impl(snowflake_catalog):
        return snowflake_catalog

    check_func(impl, (dummy_snowflake_catalogs,))


def test_snowflake_catalog_constructor(memory_leak_check):
    """
    Test using the table path constructor from JIT.
    """

    def impl1():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        )

    def impl2():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Explicitly pass None
            None,
        )

    def impl3():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        )

    def impl4():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Try passing an empty dictionary
            {},
        )

    check_func(impl1, ())
    check_func(impl2, ())
    check_func(impl3, ())
    # Note: Empty dictionary passed via args or literal map not supported yet.
    # [BE-3455]
    # check_func(impl4, ())


def test_snowflake_catalog_read(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    def impl(bc):
        return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    py_output = pd.read_sql(
        "Select r_name from REGION ORDER BY r_name",
        get_snowflake_connection_string(
            snowflake_sample_data_snowflake_catalog.database, "TPCH_SF1"
        ),
    )
    check_func(impl, (bc,), py_output=py_output)


def test_snowflake_catalog_aggregate_pushdown_sum(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    def impl(bc, query):
        return bc.sql("SELECT SUM(l_quantity) as total FROM TPCH_SF1.LINEITEM")

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    py_output = pd.read_sql(
        "SELECT SUM(l_quantity) as total FROM LINEITEM",
        get_snowflake_connection_string(
            snowflake_sample_data_snowflake_catalog.database, "TPCH_SF1"
        ),
    )

    # Case insensitive.
    query1 = "SELECT SUM(l_quantity) as total FROM TPCH_SF1.LINEITEM"
    check_func(impl, (bc, query1), py_output=py_output)
    # Case sensitive.
    query2 = 'SELECT SUM("L_QUANTITY") as total FROM TPCH_SF1.LINEITEM'
    check_func(impl, (bc, query2), py_output=py_output)


def test_snowflake_catalog_aggregate_pushdown_count(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    def impl(bc):
        return bc.sql("SELECT COUNT(*) as cnt FROM TPCH_SF1.LINEITEM")

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    py_output = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM LINEITEM",
        get_snowflake_connection_string(
            snowflake_sample_data_snowflake_catalog.database, "TPCH_SF1"
        ),
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl)(bc)
        check_logger_msg(stream, "Columns loaded ['cnt']")

    # Count produces an int64 type from pandas and an int32 from calcite.
    check_func(impl, (bc,), py_output=py_output, check_dtype=False)


def test_snowflake_catalog_insert_into(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """
    Tests executing insert into with a Snowflake Catalog.
    """
    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))
    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_test1", db, schema
    ) as table_name:
        # Write to Snowflake
        insert_table = table_name if use_default_schema else f"{schema}.{table_name}"
        query = f"INSERT INTO {insert_table}(B, C) Select 'literal', A + 1 from __bodolocal__.table1"
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]
        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.concat(
            (new_df, pd.DataFrame({"B": "literal", "C": np.arange(1, 11)}))
        )
        assert_tables_equal(output_df, result_df)


def test_snowflake_catalog_insert_into_read(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests insert into in a snowflake catalog and afterwards reading the result.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )

    def impl(bc, write_query, read_query):
        bc.sql(write_query)
        return bc.sql(read_query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))
    # Recreate the expected output with an append.
    py_output = pd.concat(
        (new_df, pd.DataFrame({"B": "literal", "C": np.arange(1, 11)}))
    )
    # Rename columns for comparison with the default capitalization when
    # read from Snowflake.
    py_output.columns = ["a", "b", "c"]  # type: ignore
    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_test3", db, schema
    ) as table_name:
        write_query = f"INSERT INTO {schema}.{table_name}(B, C) Select 'literal', A + 1 from __bodolocal__.table1"
        read_query = f"Select * from {schema}.{table_name}"
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, write_query, read_query),
            sort_output=True,
            reset_index=True,
            only_1D=True,
            py_output=py_output,
        )


def test_snowflake_catalog_insert_into_null_literal(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests insert into in a snowflake catalog with a literal null value.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )

    def impl(bc, write_query, read_query):
        bc.sql(write_query)
        return bc.sql(read_query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))
    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_test_nulls", db, schema
    ) as table_name:
        # Generate the expected output.
        py_output = pd.concat(
            (new_df, pd.DataFrame({"B": "literal", "C": np.arange(1, 11)}))
        )
        # Rename columns for comparison
        py_output.columns = ["a", "b", "c"]  # type: ignore
        write_query = f"INSERT INTO {schema}.{table_name}(A, B, C) Select NULL as A, 'literal', A + 1 from __bodolocal__.table1"
        read_query = f"Select * from {schema}.{table_name}"
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, write_query, read_query),
            sort_output=True,
            reset_index=True,
            only_1D=True,
            py_output=py_output,
        )


def test_snowflake_catalog_insert_into_date(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests insert into in a snowflake catalog with a datetime.date column.
    """
    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": [
                datetime.date(2022, 11, 1),
                datetime.date(2023, 12, 1),
                datetime.date(2025, 1, 1),
            ]
        }
    )

    def impl(bc, write_query):
        bc.sql(write_query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    date_table = pd.DataFrame({"A": [datetime.date(2023, 12, 12)] * 12})
    bc = bc.add_or_replace_view("table1", date_table)
    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_test_nulls", db, schema
    ) as table_name:
        write_query = (
            f"INSERT INTO {schema}.{table_name}(A) Select A from __bodolocal__.table1"
        )
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, write_query),
            sort_output=True,
            reset_index=True,
            only_1D=True,
            py_output=5,
        )
        output_df = None
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]
        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.concat((new_df, date_table))
        assert_tables_equal(output_df, result_df)


def test_snowflake_catalog_default(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """
    Tests passing a default schema to SnowflakeCatalog.
    """

    def impl1(bc):
        # Load with snowflake or local default
        return bc.sql("SELECT r_name FROM REGION ORDER BY r_name")

    def impl2(bc):
        return bc.sql("SELECT r_name FROM LOCAL_REGION ORDER BY r_name")

    def impl3(bc):
        return bc.sql("SELECT r_name FROM __bodolocal__.REGION ORDER BY r_name")

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    py_output = pd.read_sql(
        "Select r_name from REGION ORDER BY r_name",
        get_snowflake_connection_string(
            snowflake_sample_data_snowflake_catalog.database,
            snowflake_sample_data_snowflake_catalog.connection_params["schema"],
        ),
    )
    check_func(impl1, (bc,), py_output=py_output)

    # We use a different type for the local table to clearly tell if we
    # got the correct table.
    local_table = pd.DataFrame({"r_name": np.arange(100)})
    bc = bc.add_or_replace_view("LOCAL_REGION", local_table)
    # We should select the local table
    check_func(impl2, (bc,), py_output=local_table, reset_index=True)
    bc = bc.add_or_replace_view("REGION", local_table)
    # We two default conflicts we should get the snowflake option
    check_func(impl1, (bc,), py_output=py_output)
    # If we manually specify "__bodolocal__" we should get the local one.
    check_func(impl3, (bc,), py_output=local_table, reset_index=True)


def test_snowflake_catalog_read_tpch(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    tpch_query = """select
                      n_name,
                      sum(l_extendedprice * (1 - l_discount)) as revenue
                    from
                      tpch_sf1.customer,
                      tpch_sf1.orders,
                      tpch_sf1.lineitem,
                      tpch_sf1.supplier,
                      tpch_sf1.nation,
                      tpch_sf1.region
                    where
                      c_custkey = o_custkey
                      and l_orderkey = o_orderkey
                      and l_suppkey = s_suppkey
                      and c_nationkey = s_nationkey
                      and s_nationkey = n_nationkey
                      and n_regionkey = r_regionkey
                      and r_name = 'ASIA'
                      and o_orderdate >= '1994-01-01'
                      and o_orderdate < '1995-01-01'
                    group by
                      n_name
                    order by
                      revenue desc
    """

    def impl(bc):
        return bc.sql(tpch_query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    py_output = pd.read_sql(
        tpch_query,
        get_snowflake_connection_string(
            snowflake_sample_data_snowflake_catalog.database, "TPCH_SF1"
        ),
    )

    check_func(impl, (bc,), py_output=py_output, reset_index=True)


def test_read_timing_debug_message(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """
    Tests that loading a table using a SnowflakeCatalog with bodo.set_verbose_level(1)
    automatically adds a debug message about IO.
    """

    @bodo.jit
    def impl(bc):
        # Load with snowflake or local default
        return bc.sql("SELECT r_name FROM REGION ORDER BY r_name")

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        impl(bc)
        check_logger_msg(stream, "Execution time for reading table REGION")


def test_insert_into_timing_debug_message(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that using insert into with a SnowflakeCatalog and bodo.set_verbose_level(1)
    automatically adds a debug message about IO.
    """
    if bodo.get_size() > 1:
        # Only test on 1 rank to simplify testing.
        return

    @bodo.jit
    def impl(bc, query):
        # Load with snowflake or local default
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    typed_df = pd.DataFrame({"name_column": ["sample_name"]})

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    with create_snowflake_table(
        typed_df, "insert_into_timer_table", db, schema
    ) as table_name:
        query = f"INSERT INTO {table_name} (name_column) SELECT 'this is a name'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            impl(bc, query)
            check_logger_msg(
                stream, f"Execution time for writing table {table_name.upper()}"
            )


def test_create_table_timing_debug_message(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that using insert into with a SnowflakeCatalog and bodo.set_verbose_level(1)
    automatically adds a debug message about IO.
    """
    if bodo.get_size() > 1:
        # Only test on 1 rank to simplify testing.
        return

    @bodo.jit
    def impl(bc, query):
        # Load with snowflake or local default
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    table_name = gen_unique_table_id("create_table_timer_table")
    exception_occurred_in_test_body = False
    try:
        query = f"create table {table_name} as SELECT 'this is a name' as A"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            impl(bc, query)
            check_logger_msg(stream, f"Execution time for writing table {table_name}")
    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        db = test_db_snowflake_catalog.database
        schema = test_db_snowflake_catalog.connection_params["schema"]
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_delete_simple(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests a simple delete clause inside of Snowflake.
    """
    comm = MPI.COMM_WORLD

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 10,
            "B": np.arange(30),
        }
    )

    def impl1(query, bc):
        df = bc.sql(query)
        # Use the column name to confirm we use a standard name.
        return df["ROWCOUNT"].iloc[0]

    def impl2(query, bc):
        # Verify the delete is still performed even if the output
        # is unused
        bc.sql(query)
        return 10

    with create_snowflake_table(
        new_df, "bodosql_delete_test", db, schema
    ) as table_name:
        bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
        num_rows_to_delete = 10

        query1 = f"DELETE FROM {schema}.{table_name} WHERE A = 1"

        query2 = f"DELETE FROM {schema}.{table_name} WHERE A = 2"

        # We run only 1 distribution because DELETE has side effects
        check_func(impl1, (query1, bc), only_1D=True, py_output=num_rows_to_delete)
        check_func(impl2, (query2, bc), only_1D=True, py_output=num_rows_to_delete)

        # Load the table on rank 0 to verify the drop.
        output_df = None
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Convert output to match the input.
            output_df.columns = [colname.upper() for colname in output_df.columns]  # type: ignore
        output_df = comm.bcast(output_df)
        result_df = new_df[new_df.A == 3]
        assert_tables_equal(output_df, result_df)


def test_delete_named_param(test_db_snowflake_catalog):
    """
    Tests submitting a delete query with a named param. Since we just
    push the query into Snowflake without any changes, this query
    should just raise a reasonable error.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 10,
            "B": np.arange(30),
        }
    )

    @bodo.jit
    def impl(query, bc, pyvar):
        return bc.sql(query, {"pyvar": pyvar})

    with create_snowflake_table(
        new_df, "bodosql_dont_delete_test_param", db, schema
    ) as table_name:
        bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
        # Snowflake cannot support out named_params
        query = f"DELETE FROM {schema}.{table_name} WHERE A = @pyvar"
        with pytest.raises(
            BodoError,
            match="Please verify that all of your Delete query syntax is supported inside of Snowflake",
        ):
            impl(query, bc, 1)


def test_delete_bodosql_syntax(test_db_snowflake_catalog):
    """
    Tests submitting a delete query with SQL that does not match
    Snowflake syntax. Since we just push the query into Snowflake
    without any changes, this query should just raise a reasonable error.
    """
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 10,
            "B": np.arange(30),
        }
    )

    @bodo.jit
    def impl(query, bc):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "bodosql_dont_delete_test", db, schema
    ) as table_name:
        bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
        # Snowflake doesn't have a CEILING function
        query = f"DELETE FROM {schema}.{table_name} WHERE A = CEILING(1.5)"
        with pytest.raises(
            BodoError,
            match="Please verify that all of your Delete query syntax is supported inside of Snowflake",
        ):
            impl(query, bc)


def assert_tables_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """Asserts df1 and df2 have the same data without regard
    for ordering or index.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
    """
    # Output ordering is not defined so we sort.
    df1 = df1.sort_values(by=[col for col in df1.columns])
    df2 = df2.sort_values(by=[col for col in df2.columns])
    # Drop the index
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def test_current_timestamp_case(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """Tests CURRENT_TIMESTAMP and equivalents with a snowflake catalog to verify that
    the output uses the default Snowflake timezone.
    """
    current_timestamp = pd.Timestamp.now(tz="US/Pacific")
    _, valid_hours, _ = compute_valid_times(current_timestamp)

    def impl(bc):
        return bc.sql(
            f"SELECT "
            f"  DATE_TRUNC('DAY', NOW()) AS now_trunc, "
            f"  DATE_TRUNC('DAY', LOCALTIMESTAMP()) AS local_trunc, "
            f"  DATE_TRUNC('DAY', GETDATE()) AS getdate_trunc, "
            f"  DATE_TRUNC('DAY', SYSTIMESTAMP()) AS sys_trunc, "
            f"  CASE WHEN A THEN DATE_TRUNC('DAY', CURRENT_TIMESTAMP()) END AS case_current_trunc, "
            f"  CASE WHEN A THEN EXTRACT(HOUR from NOW()) IN ({valid_hours}) END AS is_valid_now, "
            f"  CASE WHEN A THEN EXTRACT(HOUR from LOCALTIME()) IN ({valid_hours}) END AS is_valid_localtime "
            f"FROM __bodolocal__.table1"
        )

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    bc = bc.add_or_replace_view("table1", df)
    normalize_val = current_timestamp.normalize()
    S = pd.Series(normalize_val, index=np.arange(len(df)))
    S[~df.A] = None
    V = pd.Series(True, index=np.arange(len(df)))
    V[~df.A] = None
    py_output = pd.DataFrame(
        {
            "now_trunc": normalize_val,
            "local_trunc": normalize_val,
            "getdate_trunc": normalize_val,
            "sys_trunc": normalize_val,
            "case_current_trunc": S,
            "is_valid_now": V,
            "is_valid_localtime": V,
        },
    )
    check_func(
        impl,
        (bc,),
        None,
        py_output=py_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "transient_default",
    [
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ],
)
def test_default_table_type(
    transient_default, test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that the default table type is dependent on the default
    type for the database/schema it's created in/
    """

    # The default table type depends on the default for the database/schema
    if transient_default:
        # default for test_db_snowflake_catalog is transient
        catalog = test_db_snowflake_catalog
        db = test_db_snowflake_catalog.database
        schema = test_db_snowflake_catalog.connection_params["schema"]
    else:
        # E2E_TESTS_DB.PUBLIC defaults to permanent
        catalog = bodosql.SnowflakeCatalog(
            os.environ.get("SF_USERNAME", ""),
            os.environ.get("SF_PASSWORD", ""),
            "bodopartner.us-east-1",
            "DEMO_WH",
            "E2E_TESTS_DB",
            connection_params={"schema": "PUBLIC"},
        )
        db = "E2E_TESTS_DB"
        schema = "PUBLIC"

    bc = bodosql.BodoSQLContext(catalog=catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))

    # Create the table

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id(
            "bodosql_catalog_write_create_table_does_not_already_exist"
        )
    table_name = comm.bcast(table_name)

    # Write to Snowflake
    succsess_query = f"CREATE OR REPLACE TABLE {schema}.{table_name} AS Select 'literal' as column1, A + 1 as column2, '2023-02-21'::date as column3 from __bodolocal__.table1"

    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, succsess_query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.DataFrame(
            {
                "column1": "literal",
                "column2": np.arange(1, 11),
                "column3": datetime.date(2023, 2, 21),
            }
        )
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)

        table_type = None
        if bodo.get_rank() == 0:
            table_type = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}' in SCHEMA IDENTIFIER('{db}.{schema}')",
                conn_str,
            )["kind"][0]

        table_type = comm.bcast(table_type)
        expected_table_type = "TRANSIENT" if transient_default else "TABLE"
        assert (
            table_type == expected_table_type
        ), f"Table type is not as expected. Expected {expected_table_type} but found {table_type}"

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_create_table_temporary(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that explicitly supplying "TEMPORARY" as the table type works.
    Since the table will be dropped before we have the opportunity to read
    it, we're just testing that it runs without error.
    """
    # default for test_db_snowflake_catalog is transient
    catalog = test_db_snowflake_catalog
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    bc = bodosql.BodoSQLContext(catalog=catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id(
            "bodosql_catalog_write_create_table_does_not_already_exist"
        )
    table_name = comm.bcast(table_name)

    succsess_query = f"CREATE OR REPLACE TEMPORARY TABLE {schema}.{table_name} AS Select 'literal' as column1, A + 1 as column2, '2023-02-21'::date as column3 from __bodolocal__.table1"
    check_func(impl, (bc, succsess_query), only_1D=True, py_output=5)


def test_snowflake_catalog_create_table_transient(memory_leak_check):
    """tests that explicitly supplying "TRANSIENT" as the table type works"""

    # E2E_TESTS_DB defaults to permanent, this is needed to make sure
    # that the default table type is NOT transient
    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "E2E_TESTS_DB",
        connection_params={"schema": "PUBLIC"},
    )
    db = "E2E_TESTS_DB"
    schema = "PUBLIC"

    bc = bodosql.BodoSQLContext(catalog=catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id(
            "bodosql_catalog_write_create_table_does_not_already_exist"
        )
    table_name = comm.bcast(table_name)

    succsess_query = f"CREATE OR REPLACE TRANSIENT TABLE {schema}.{table_name} AS Select 'literal' as column1, A + 1 as column2, '2023-02-21'::date as column3 from __bodolocal__.table1"
    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, succsess_query), only_1D=True, py_output=5)

        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.DataFrame(
            {
                "column1": "literal",
                "column2": np.arange(1, 11),
                "column3": datetime.date(2023, 2, 21),
            }
        )
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)

        output_table_type = None
        if bodo.get_rank() == 0:
            output_table_type = pd.read_sql(
                f"SHOW TABLES LIKE '{table_name}' in SCHEMA IDENTIFIER('{db}.{schema}')",
                conn_str,
            )["kind"][0]

        output_table_type = comm.bcast(output_table_type)
        assert (
            output_table_type == "TRANSIENT"
        ), f"Table type is not as expected. Expected TRANSIENT but found {output_table_type}"

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_create_table_does_not_already_exists(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """
    Test Snowflake CREATE TABLE, without a pre-existing table
    """

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))
    # Create the table

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id(
            "bodosql_catalog_write_create_table_does_not_already_exist"
        )
    table_name = comm.bcast(table_name)

    # Write to Snowflake
    insert_table = table_name if use_default_schema else f"{schema}.{table_name}"
    succsess_query = f"CREATE TABLE IF NOT EXISTS {insert_table} AS Select 'literal' as column1, A + 1 as column2, '2023-02-21'::date as column3 from __bodolocal__.table1"
    # This should succseed, since the table does not exist

    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, succsess_query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.DataFrame(
            {
                "column1": "literal",
                "column2": np.arange(1, 11),
                "column3": datetime.date(2023, 2, 21),
            }
        )
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)
    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_create_table_already_exists_error(
    test_db_snowflake_catalog,
    use_default_schema,
):
    """
    Test that Snowflake CREATE TABLE errors with a pre-existing table, when we expect an error.
    """
    # Note: we have a memory leak due to the runtime error case

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))

    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_create_table_test_already_exists", db, schema
    ) as table_name:
        # Write to Snowflake
        insert_table = table_name if use_default_schema else f"{schema}.{table_name}"
        fail_query = f"CREATE TABLE IF NOT EXISTS {insert_table} AS Select 'literal_not' as columnA, A + 10 as columnB, '2023-02-2'::date as columnC from __bodolocal__.table1"

        # This should pass, since the table does not currently exist
        with pytest.raises(RuntimeError, match=".*Object .* already exists.*"):
            check_func(impl, (bc, fail_query), only_1D=True, py_output=5)

        # Default behavior is IF NOT EXISTS, if it is not explicitly specified
        fail_query_2 = f"CREATE TABLE IF NOT EXISTS {insert_table} AS Select 'literal' as columnA, A + 1 as columnB, '2023-02-21'::date as columnC from __bodolocal__.table1"

        with pytest.raises(RuntimeError, match=".*Object .* already exists.*"):
            check_func(impl, (bc, fail_query_2), only_1D=True, py_output=5)
        # create_snowflake_table handles dropping the table for us


def test_snowflake_catalog_create_table_already_exists(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """
    Test Snowflake CREATE TABLE, with a pre-existing table
    """

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": [1, 3, 5, 7, 9] * 10, "B": ["Afe", "fewfe"] * 25, "C": 1.1}
    )

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    bc = bc.add_or_replace_view("table1", pd.DataFrame({"A": np.arange(10)}))

    # Create the table
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_create_table_test_already_exists", db, schema
    ) as table_name:
        # Write to Snowflake
        insert_table = table_name if use_default_schema else f"{schema}.{table_name}"

        succsess_query = f"CREATE OR REPLACE TABLE {insert_table} AS Select 2 as column1, A as column2, 'hello world' as column3 from __bodolocal__.table1"
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, succsess_query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        # Recreate the expected output by manually doing an append.
        result_df = pd.DataFrame(
            {"column1": 2, "column2": np.arange(10), "column3": "hello world"}
        )
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)
        # create_snowflake_table handles dropping the table for us


def test_snowflake_catalog_simple_rewrite(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """tests that create table can handle simple queries that require some amount of re-write during validation."""

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    local_table = pd.DataFrame({"A": np.arange(10)})
    bc = bc.add_or_replace_view("table1", local_table)

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("bodosql_ctas_test_rewrite")
    table_name = comm.bcast(table_name)

    # Write to Snowflake
    insert_table = table_name if use_default_schema else f"{schema}.{table_name}"

    # Limit does nothing in this case, since the table has less than 100 values
    # but the clause does require a re-write
    query = f"CREATE TABLE IF NOT EXISTS {insert_table} AS Select * from __bodolocal__.table1 limit 100"

    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        result_df = local_table
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)
    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_simple_rewrite_2(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """tests that create table can handle simple queries that require some amount of re-write during validation.
    This is a secondary test that covers a subset of Q7"""

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    random.seed(42)
    local_table = pd.DataFrame(
        {
            "r_name": random.choices(
                ["abc gobrandon", "xyz bong", "gobrandon", "bong", "a98y7guhb"], k=20
            )
        }
    )

    def row_fn(v):
        if v == "abc gobrandon":
            return "gobrandon"
        elif v == "xyz bong":
            return "bong"
        else:
            return None

    expected_output_col = local_table["r_name"].apply(lambda x: row_fn(x))
    expected_output_col = expected_output_col.dropna()
    expected_output = pd.DataFrame({"output_case": expected_output_col})

    bc = bc.add_or_replace_view("table1", local_table)

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("bodosql_ctas_test_rewrite")
    table_name = comm.bcast(table_name)

    # Write to Snowflake
    insert_table = table_name if use_default_schema else f"{schema}.{table_name}"

    # Limit does nothing in this case, since the table has less than 100 values
    # but the clause does require a re-write
    query = f"""CREATE TABLE {insert_table} as SELECT
                case
                    when endswith(r_name, ' gobrandon') then 'gobrandon'
                    when endswith(r_name, ' bong') then 'bong'
                    else null end as output_case
                FROM __bodolocal__.table1
                WHERE output_case is not null
    """

    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, query), only_1D=True, py_output=5)
        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]

        output_df = comm.bcast(output_df)
        expected_result_df = expected_output
        output_df.columns = output_df.columns.str.upper()
        expected_result_df.columns = expected_result_df.columns.str.upper()
        assert_tables_equal(output_df, expected_result_df)
    finally:
        # dropping the table
        drop_snowflake_table(table_name, db, schema)


@pytest.mark.slow
def test_snowflake_catalog_create_table_tpch(
    test_db_snowflake_catalog, tpch_data, memory_leak_check
):
    """
    Test Snowflake CREATE TABLE, with a larger e2e example that is
    a modified version of the TPCH benchmark.
    """

    SIZE = 15
    TYPE = "BRASS"
    REGION = "EUROPE"
    base_tpch_query = f"""
    with region_filtered as (
        select * from region where r_name = '{REGION}'
    )
    select
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
    from
        part,
        supplier,
        partsupp,
        nation,
        region_filtered
    where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and p_size = {SIZE}
        and p_type like '%{TYPE}'
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and ps_supplycost = (
            select
            min(ps_supplycost)
            from
            partsupp, supplier,
            nation, region_filtered
            where
            p_partkey = ps_partkey
            and s_suppkey = ps_suppkey
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            )
    order by
        s_acctbal desc,
        n_name,
        s_name,
        p_partkey
    """

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn_str = get_snowflake_connection_string(db, schema)
    expected_output_table_name = (
        "test_snowflake_catalog_create_table_tpch_expected_output"
    )

    # # Code for generating the expected output. Can be uncommented to regenerate
    # # the expected output if the query changes, or the output table in SF is deleted
    # output = check_query(
    #     base_tpch_query,
    #     tpch_data,
    #     spark_info,
    #     check_dtype=False,
    #     return_seq_dataframe=True,
    # )
    # expected_output = output["output_df"]

    # if bodo.get_rank() == 0:
    #     expected_output.to_sql(
    #         expected_output_table_name, conn_str, if_exists='replace', index=False)

    # Now, run the query and push the results to SF
    # NOTE: We don't use the snowflake_sample_data_snowflake_catalog fixture for reading
    # the data from Snowflake. This is because whatever database we read from,
    # we have to write to, and we don't have write permissions to the tpch sample database.
    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("bodosql_catalog_write_create_tpch_test")
    table_name = comm.bcast(table_name)
    ctas_tpch_query = (
        f"CREATE OR REPLACE TABLE {table_name} AS (\n" + base_tpch_query + ")"
    )

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    for tpch_table_name, df in tpch_data.items():
        bc = bc.add_or_replace_view(tpch_table_name, df)

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    try_body_threw_error = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, ctas_tpch_query), only_1D=True, py_output=5)

        output_df = None
        expected_output = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            expected_output = pd.read_sql(
                f"select * from {expected_output_table_name}", conn_str
            )
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]
            expected_output.columns = [
                colname.upper() for colname in expected_output.columns
            ]

        output_df = comm.bcast(output_df)
        expected_output = comm.bcast(expected_output)
        # Recreate the expected output by manually doing an append.
        output_df.columns = output_df.columns.str.upper()
        expected_output.columns = expected_output.columns.str.upper()
        assert_tables_equal(output_df, expected_output, check_dtype=False)
    except Exception as e:
        try_body_threw_error = True
        raise e
    finally:
        # Drop the table. In the case that the try body throws an error,
        # we may not have succseeded in creating the table. Therefore, we
        # need to try to drop the table, but we don't want to mask the
        # original error, so we use a try/except block.
        if try_body_threw_error:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_create_table_orderby_with():
    """
    Test Snowflake CREATE TABLE, for a specific edgecase found with handling WITH and
    ORDER BY clauses simultaneously.
    """

    base_query = f"""
    with part_two as (
        select 'foo' as p_partkey from (VALUES (1, 2, 3))
    )
    select
                       p_partkey
                     from
                       part_two
                     order by
                       p_partkey
    """

    ctas_query = f"CREATE OR REPLACE TABLE WILL_THROW_ERROR AS (\n" + base_query + ")"

    bc = bodosql.BodoSQLContext(dict())

    @bodo.jit
    def bodo_impl(bc, query):
        bc.sql(query)

    # Note: we previously ran into a null pointer exception in validation.
    # We still expect this to throw an error, because we only support CREATE TABLE for Snowflake Catalog Schemas.
    # The full correctness test is test_snowflake_catalog_create_table_tpch. However, this is a slow test,
    # so I included a smaller test that runs on PR CI to catch potential regressions earlier.
    with pytest.raises(
        BodoError,
        match=".*CREATE TABLE is only supported for Snowflake Catalog Schemas.*",
    ):
        bodo_impl(bc, ctas_query)


@pytest.mark.parametrize(
    "table_name_qualifer",
    [
        f"SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER",
        f"TPCH_SF1.CUSTOMER",
        f"CUSTOMER",
    ],
)
def test_snowflake_catalog_fully_qualified(
    snowflake_sample_data_snowflake_catalog, table_name_qualifer, memory_leak_check
):
    """Tests that the snowflake catalog correctly handles table names with varying levels of qualification"""
    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

    query = f"SELECT C_ADDRESS from {table_name_qualifer} where C_CUSTKEY = 60000"
    py_output = pd.DataFrame({"C_ADDRESS": ["gUTQNtV,KAQve"]})

    # Only testing Seq, since output is tiny and we're only really testing
    # name resolution is successful
    check_func(
        impl,
        (bc, query),
        None,
        py_output=py_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )


# Table Names present in each location

# The specified Snowflake catalog is TEST_DB.TEST_SCHEMA
# The non-default Snowflake catalog is TEST_DB.PUBLIC
# the local schema is __bodolocal__
# These schemas/tables have been manually added to the bnsare snowflake account,
# and should not be modified.
#
# For all tables present in public, the tables are created as follows:
# Create TABLE *NAME* as SELECT 'TEST_DB.PUBLIC' as A from VALUES (1,2,3)
#
# For all tables present in the TEST_SCHEMA, the tables are created as follows:
# Create TABLE *NAME* as SELECT 'TEST_DB.TEST_SCHEMA' as A from VALUES (1,2,3)
#
# For the local table, it's defined as:
# pd.DataFrame({"a": ["local"]})
# The locations in which each table can be found are as follows:
# tableScopingTest: TEST_DB.PUBLIC, TEST_DB.TEST_SCHEMA, __bodolocal__
# tableScopingTest2: TEST_DB.TEST_SCHEMA, __bodolocal__
# tableScopingTest3: TEST_DB.PUBLIC, TEST_DB.TEST_SCHEMA
# tableScopingTest4: TEST_DB.PUBLIC, __bodolocal__

tableScopingTest1TableName = "tableScopingTest"
tableScopingTest2TableName = "tableScopingTest2"
tableScopingTest3TableName = "tableScopingTest3"
tableScopingTest4TableName = "tableScopingTest4"

# Expected outputs, assuming we select the local/non-default catalog/default catalog schemas
expected_out_local = pd.DataFrame({"a": ["local"]})
expected_out_TEST_SCHEMA_catalog_schema = pd.DataFrame({"a": ["TEST_DB.TEST_SCHEMA"]})
expected_out_PUBLIC_catalog_schema = pd.DataFrame({"a": ["TEST_DB.PUBLIC"]})


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (tableScopingTest1TableName, expected_out_TEST_SCHEMA_catalog_schema),
            id="present_in_all_locations_table_name. Should resolve to default catalog schema",
        ),
        pytest.param(
            (tableScopingTest2TableName, expected_out_TEST_SCHEMA_catalog_schema),
            id="present_in_local_and_default_catalog_schema. Should resolve to default catalog schema",
        ),
        pytest.param(
            (tableScopingTest3TableName, expected_out_TEST_SCHEMA_catalog_schema),
            id="present_in_default_and_PUBLIC_catalog_schema. Should resolve to default catalog schema",
        ),
        pytest.param(
            (tableScopingTest4TableName, expected_out_PUBLIC_catalog_schema),
            id="present_in_local_and_PUBLIC_catalog_schema.  Should resolve to PUBLIC catalog schema",
        ),
    ],
)
def test_snowflake_catalog_table_priority(args, memory_leak_check):
    """
    Tests that BodoSQL selects the correct table in the case that the tablename is
    present in multiple locations in the catalog/database.

    In Snowflake, the default schema is PUBLIC.
    In the case that a schema is specified, the default schema is the specified schema,
    but the PUBLIC schema is still implicitly available.
    (
        by default, this can change base on connection parameters, see
        https://docs.snowflake.com/en/sql-reference/name-resolution
    )
    For example, in Snowflake, if the specified schema is TEST_DB, when attempting to resolve
    an unqualified table name,
    Snowflake will attempt to resolve it as TEST_DB.TABLE_NAME, and then PUBLIC.TABLE_NAME.

    In BodoSQL, we prioritize snowflake tables over locally defined tables. Therefore,
    The correct attempted order of resolution for an unqualified table name in BodoSQL should be
        default_database.specified_schema.(table_identifier)
        default_database.PUBLIC.(table_identifier)
        __bodo_local__.(table_identifier)

    To test this, we have several tablenames that are present across the different
    locations (PUBLIC, TEST_DB, and __bodo_local__),
    and we test that the correct table is selected in each case.
    """
    default_db_name = "TEST_DB"
    default_schema_name = "TEST_SCHEMA"

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        default_db_name,
        connection_params={
            "schema": default_schema_name,
        },
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)
    for table_name in [
        tableScopingTest1TableName,
        tableScopingTest2TableName,
        tableScopingTest4TableName,
    ]:
        bc = bc.add_or_replace_view(table_name, expected_out_local)

    current_table_name, expected_output = args
    query = f"SELECT * FROM {current_table_name}"

    def impl(bc, q):
        return bc.sql(q)

    # Only testing Seq, since output is tiny and we're only really testing
    # that name resolution is successful
    check_func(
        impl,
        (bc, query),
        None,
        py_output=expected_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("__bodolocal__", expected_out_local), id="__bodolocal__"),
        pytest.param(("PUBLIC", expected_out_PUBLIC_catalog_schema), id="PUBLIC"),
    ],
)
def test_snowflake_catalog_table_priority_override(args, memory_leak_check):
    """Tests that BodoSQL properly selects the correct table in the case that the tablename is
    not ambiguous. IE, if we explicitly specify the catalog/schema,
    we should select the table present in the specified catalog/schema, instead of a different
    table with the same name in a different catalog/schema.
    IE If the user specifies "__bodolocal__.tablename", it should not resolve to the table
    found at "TEST_DB.PUBLIC.tablename", and visa versa.
    """
    default_db_name = "TEST_DB"
    default_schema_name = "TEST_SCHEMA"

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        default_db_name,
        connection_params={
            "schema": default_schema_name,
        },
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)
    for table_name in [
        tableScopingTest1TableName,
        tableScopingTest2TableName,
        tableScopingTest4TableName,
    ]:
        bc = bc.add_or_replace_view(table_name, expected_out_local)

    prefix, expected_output = args
    query = f"SELECT * FROM {prefix}.{tableScopingTest1TableName}"

    def impl(bc, q):
        return bc.sql(q)

    # Only testing Seq, since output is tiny and we're only really testing
    # name resolution is successful
    check_func(
        impl,
        (bc, query),
        None,
        py_output=expected_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )


def test_snowflake_catalog_defaults_to_public(memory_leak_check):
    """Tests that BodoSQL defaults to the public schema if none are specified."""
    default_db_name = "TEST_DB"

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        default_db_name,
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)

    # Table is present in TEST_DB.TEST_SCHEMA, __bodolocal__, but not PUBLIC.
    # We should be able to find the table if the schema is specified,
    # even if it's not in the default search path
    succsess_query = f"SELECT * FROM {tableScopingTest1TableName}"

    def impl(bc, q):
        return bc.sql(q)

    # Only testing Seq, since output is tiny and we're only really testing
    # name resolution is successful
    check_func(
        impl,
        (bc, succsess_query),
        None,
        py_output=expected_out_PUBLIC_catalog_schema,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )


def test_snowflake_catalog_table_not_found(memory_leak_check):
    """Tests that BodoSQL does NOT find a tablename that is present in the catalog/database,
    but not present in any of the implicit schemas.
    """
    default_db_name = "TEST_DB"
    default_schema_name = "PUBLIC"

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        default_db_name,
        connection_params={
            "schema": default_schema_name,
        },
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)

    # Table is present in TEST_DB.TEST_SCHEMA, __bodolocal__, but not PUBLIC.
    # Therefore, we shouldn't be able to find it
    fail_query = f"SELECT * FROM {tableScopingTest2TableName}"

    @bodo.jit()
    def impl(bc, q):
        return bc.sql(q)

    with pytest.raises(BodoError, match=".*Object 'tableScopingTest2' not found.*"):
        impl(bc, fail_query)


def test_snowflake_catalog_table_find_able_if_not_default(memory_leak_check):
    """Tests that BodoSQL does can find all tables in the catalog/database,
    provided that they are appropriately qualified.
    """
    default_db_name = "TEST_DB"
    default_schema_name = "PUBLIC"

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        default_db_name,
        connection_params={
            "schema": default_schema_name,
        },
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)

    # Table is present in TEST_DB.TEST_SCHEMA, __bodolocal__, but not PUBLIC.
    # Therefore, we should be able to find it IF we qualify it
    succsess_query = f"SELECT * FROM TEST_SCHEMA.{tableScopingTest2TableName}"

    def impl(bc, q):
        return bc.sql(q)

    # Only testing Seq, since output is tiny and we're only really testing
    # name resolution is successful
    check_func(
        impl,
        (bc, succsess_query),
        None,
        py_output=expected_out_TEST_SCHEMA_catalog_schema,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )


def test_snowflake_catalog_create_table_like(
    test_db_snowflake_catalog, memory_leak_check
):
    """
        Test Snowflake CREATE TABLE LIKE.

    NOTE: The current behavior of CREATE TABLE LIKE does not exactly match the behavior of Snowflake.
    It doesn't copy permissions, and some column types are not copied exactly:


    Expected:
                   name          type    kind null? default  ... unique key check expression comment policy name
    ...
    7             L_TAX  NUMBER(12,2)  COLUMN     Y    None  ...          N  None       None    None        None
    8      L_RETURNFLAG    VARCHAR(1)  COLUMN     Y    None  ...          N  None       None    None        None
    9      L_LINESTATUS    VARCHAR(1)  COLUMN     Y    None  ...          N  None       None    None        None
    10       L_SHIPDATE          DATE  COLUMN     Y    None  ...          N  None       None    None        None
    ...

    Actuall:
                   name               type    kind null?  ... check expression comment policy name
    7             L_TAX              FLOAT  COLUMN     Y  ...  None       None    None        None
    8      L_RETURNFLAG  VARCHAR(16777216)  COLUMN     Y  ...  None       None    None        None
    9      L_LINESTATUS  VARCHAR(16777216)  COLUMN     Y  ...  None       None    None        None
    10       L_SHIPDATE   TIMESTAMP_NTZ(9)  COLUMN     Y  ...  None       None    None        None
    ...

    In order to properly implement this, we would likely need to push the query directly to Snowflake.
    However, I'm going to leave this to a followup issue, since this will likely be easier post
    Jonathan's refactor to the volcanno planner:
    https://bodo.atlassian.net/browse/BE-4578

    """

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn_str = get_snowflake_connection_string(db, schema)

    output_table_name = None
    if bodo.get_rank() == 0:
        output_table_name = gen_unique_table_id("LINEITEM_EMPTY_COPY")
    output_table_name = comm.bcast(output_table_name)

    query = f"CREATE TABLE {output_table_name} LIKE LINEITEM1"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    try_body_threw_error = False
    try:
        # Only test with only_1D=True so we only create the table once.
        check_func(impl, (bc, query), only_1D=True, py_output=5)

        output_df = None
        expected_output = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            expected_output = pd.read_sql(f"select * from LINEITEM1 LIMIT 0", conn_str)
            output_df = pd.read_sql(f"select * from {output_table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]
            expected_output.columns = [
                colname.upper() for colname in expected_output.columns
            ]

        output_df = comm.bcast(output_df)
        expected_output = comm.bcast(expected_output)
        # Recreate the expected output by manually doing an append.
        output_df.columns = output_df.columns.str.upper()
        expected_output.columns = expected_output.columns.str.upper()
        assert_tables_equal(output_df, expected_output)
    except Exception as e:
        try_body_threw_error = True
        raise e
    finally:
        # Drop the table. In the case that the try body throws an error,
        # we may not have succseeded in creating the table. Therefore, we
        # need to try to drop the table, but we don't want to mask the
        # original error, so we use a try/except block.
        if try_body_threw_error:
            try:
                drop_snowflake_table(output_table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(output_table_name, db, schema)
