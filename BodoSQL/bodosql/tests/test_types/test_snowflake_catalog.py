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

from bodosql.tests.utils import check_query

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
        check_logger_msg(stream, "Finished reading table REGION")


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


def test_snowflake_catalog_create_table_does_not_already_exists(
    test_db_snowflake_catalog, use_default_schema, memory_leak_check
):
    """
    Test Snowflake CREATE TABLE, without a pre-existing table
    """

    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

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
                "column3": pd.Timestamp("2023-02-21"),
            }
        )
        output_df.columns = output_df.columns.str.upper()
        result_df.columns = result_df.columns.str.upper()
        assert_tables_equal(output_df, result_df)
    finally:
        # dropping the table for us
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

    exception_occured_in_test_body = False
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
        exception_occured_in_test_body = True
        raise e
    finally:
        if exception_occured_in_test_body:
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
    expected_output_table_name = "test_snowflake_catalog_create_table_tpch_expected_output"

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
        table_name = gen_unique_table_id(
            "bodosql_catalog_write_create_tpch_test"
        )
    table_name = comm.bcast(table_name)
    ctas_tpch_query = f"CREATE OR REPLACE TABLE {table_name} AS (\n" + base_tpch_query + ")"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    for (tpch_table_name, df) in tpch_data.items():
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
            expected_output = pd.read_sql(f"select * from {expected_output_table_name}", conn_str)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            # Reset the columns to the original names for simpler testing.
            output_df.columns = [colname.upper() for colname in output_df.columns]
            expected_output.columns = [colname.upper() for colname in expected_output.columns]

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
    with pytest.raises(BodoError, match=".*CREATE TABLE is only supported for Snowflake Catalog Schemas.*"):
        bodo_impl(bc, ctas_query)

