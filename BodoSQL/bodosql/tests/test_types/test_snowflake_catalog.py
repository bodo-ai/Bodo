# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests various components of the SnowflakeCatalog type both inside and outside a
direct BodoSQLContext.
"""

import datetime
import io
import os
import random
import time
from urllib.parse import urlencode

import bodosql
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.test_datetime_fns import compute_valid_times
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_conn_str,
    snowflake_sample_data_snowflake_catalog,
    test_db_snowflake_catalog,
)
from mpi4py import MPI
from numba.core import types
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    check_func,
    create_snowflake_table,
    drop_snowflake_table,
    gen_unique_table_id,
    get_snowflake_connection_string,
    pytest_snowflake,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call_assign

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
    List of table paths that should be supported.
    None of these actually point to valid data
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


@pytest.mark.parametrize(
    "conn_str",
    [
        # Basic Connection Str
        "snowflake://myusername:mypassword@myaccount/mydatabase?warehouse=mywarehouse",
        # With Schema
        "snowflake://myusername:mypassword@myaccount/mydatabase/myschema?warehouse=mywarehouse",
        # Additional Connection Param. Note, order of params matter for test
        "snowflake://myusername:mypassword@myaccount/mydatabase/myschema?role=USERADMIN&warehouse=mywarehouse",
    ],
)
def test_snowflake_catalog_from_conn_str(conn_str: str):
    c = bodosql.SnowflakeCatalog.from_conn_str(conn_str)

    params = c.connection_params.copy()
    params["warehouse"] = c.warehouse

    schema = params.pop("schema", None)
    schema_str = f"/{schema}" if schema is not None else ""

    params_sorted = sorted(params.items())
    expected = f"snowflake://{c.username}:{c.password}@{c.account}/{c.database}{schema_str}?{urlencode(params_sorted)}"
    assert (
        expected == conn_str
    ), "Connection String from SnowflakeCatalog does not match input arg"


@pytest.mark.parametrize(
    "conn_str",
    [
        # Not a URL at all
        "test",
        # Non-Snowflake location
        "http://myaccount/mydatabase",
        # Invalid Connection Params
        "snowflake://myusername:mypassword@myaccount?test?test2",
    ],
)
def test_snowflake_catalog_from_conn_str_invalid_err(conn_str):
    """Test that invalid URIs fail when parsing"""
    with pytest.raises(BodoError, match="Invalid Snowflake Connection URI Provided"):
        bodosql.SnowflakeCatalog.from_conn_str(conn_str)


@pytest.mark.parametrize(
    "conn_str",
    [
        # Missing Username and Password
        "snowflake://myaccount/mydatabase",
        # Missing Username
        "snowflake://:mypassword@myaccount/mydatabase",
        # Missing Password
        "snowflake://myusername@myaccount/mydatabase",
        # Missing Database
        "snowflake://myusername:mypassword@myaccount",
        # Missing Warehouse
        "snowflake://myusername:mypassword@myaccount/mydatabase",
    ],
)
def test_snowflake_catalog_from_conn_str_missing_err(conn_str):
    """Test that valid URIs fail due to missing contents"""
    with pytest.raises(BodoError, match="`conn_str` must contain a"):
        bodosql.SnowflakeCatalog.from_conn_str(conn_str)


def test_snowflake_catalog_from_conn_str_jit_err():
    def impl():
        return bodosql.SnowflakeCatalog.from_conn_str(
            "snowflake://user:pass@acc/db/schema"
        )

    with pytest.raises(
        BodoError,
        match="This constructor can not be called from inside of a Bodo-JIT function",
    ):
        bodo.jit(impl)()  # type: ignore


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


def test_snowflake_catalog_from_conn_str_read(
    snowflake_sample_data_conn_str: str, memory_leak_check
):
    def impl(bc):
        return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog.from_conn_str(snowflake_sample_data_conn_str)
    )
    py_output = pd.read_sql(
        "Select r_name from REGION ORDER BY r_name",
        snowflake_sample_data_conn_str,
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


# TODO: re add memory_leak_check. See https://bodo.atlassian.net/browse/BSE-990
def test_snowflake_catalog_insert_into_read(
    test_db_snowflake_catalog,
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


# TODO: re add memory_leak_check. See https://bodo.atlassian.net/browse/BSE-990
def test_snowflake_catalog_insert_into_null_literal(
    test_db_snowflake_catalog,
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
    tpch_query = """\
        select
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
        check_logger_msg(stream, "Execution time for reading table ")


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
                conn_str,  # type: ignore
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


@pytest.mark.parametrize(
    "params, answers",
    [
        pytest.param(
            ("0", "0"),
            ([1, 1, 1, 5, 52], [2020, 2016, 1980, 1971, 2000]),
            id="iso",
        ),
        pytest.param(
            ("0", "7"),
            ([1, 1, 1, 5, 1], [2020, 2016, 1980, 1971, 2001]),
            id="iso-sunday",
        ),
        pytest.param(
            ("1", "0"),
            ([1, 2, 1, 6, 53], [2020, 2016, 1980, 1971, 2000]),
            id="policy-monday",
        ),
        pytest.param(
            ("1", "7"),
            ([1, 2, 1, 6, 54], [2020, 2016, 1980, 1971, 2000]),
            id="policy-sunday",
        ),
    ],
)
def test_snowflake_catalog_week_policy_parameters(params, answers, memory_leak_check):
    """tests that explicitly supplying WEEK_START and WEEK_OF_YEAR_POLICY works"""

    week_of_year_policy, week_start = params
    expected_woy, expected_yow = answers

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={
            "schema": "PUBLIC",
            "WEEK_START": week_start,
            "WEEK_OF_YEAR_POLICY": week_of_year_policy,
        },
    )

    extra_df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2016-01-04"),
                    pd.Timestamp("1980-01-02"),
                    pd.Timestamp("1971-02-02"),
                    pd.Timestamp("2000-12-31"),
                ]
            )
        }
    )

    expected_output = pd.DataFrame(
        {
            "A": expected_woy,
            "B": expected_yow,
        }
    )

    bc = bodosql.BodoSQLContext(catalog=catalog)
    bc = bc.add_or_replace_view("week_policy_table1", extra_df)

    def impl(bc, query):
        return bc.sql(query)

    query = "select WEEKOFYEAR(A) as A, YEAROFWEEK(A) as B from week_policy_table1"
    check_func(
        impl,
        (bc, query),
        py_output=expected_output,
        check_dtype=False,
    )


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
                conn_str,  # type: ignore
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
    # This should succeed, since the table does not exist

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

    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, ctas_tpch_query),
            only_1D=True,
            py_output=5,
            use_dict_encoded_strings=False,
        )

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
        # In the case that the try body throws an error,
        # we may not have succeeded in creating the table. Therefore, we
        # need to try to drop the table, but we don't want to mask the
        # original error, so we use a try/except block.
        try:
            drop_snowflake_table(table_name, db, schema)
        except Exception:
            pass
        raise e
    else:
        # Drop the table.
        drop_snowflake_table(table_name, db, schema)


def test_snowflake_catalog_create_table_orderby_with():
    """
    Test Snowflake CREATE TABLE, for a specific edge-case found with handling WITH and
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

    To test this, we have several table names that are present across the different
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

    Actual:
                   name               type    kind null?  ... check expression comment policy name
    7             L_TAX              FLOAT  COLUMN     Y  ...  None       None    None        None
    8      L_RETURNFLAG  VARCHAR(16777216)  COLUMN     Y  ...  None       None    None        None
    9      L_LINESTATUS  VARCHAR(16777216)  COLUMN     Y  ...  None       None    None        None
    10       L_SHIPDATE   TIMESTAMP_NTZ(9)  COLUMN     Y  ...  None       None    None        None
    ...

    In order to properly implement this, we would likely need to push the query directly to Snowflake.
    However, I'm going to leave this to a followup issue, since this will likely be easier post
    Jonathan's refactor to the volcano planner:
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
        # Drop the table. In the case that the try body throws an error,
        # we may not have succeeded in creating the table. Therefore, we
        # need to try to drop the table, but we don't want to mask the
        # original error, so we use a try/except block.
        try:
            drop_snowflake_table(output_table_name, db, schema)
        except Exception:
            pass
        raise e
    else:
        drop_snowflake_table(output_table_name, db, schema)


def test_sf_filter_pushdown_rowcount_estimate(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that filter pushdown cost estimate works for snowflake catalog.
    Note that we can't do this in the maven unit tests, as the TPCH table/row counts
    are Mocked, and therefore we won't actually push any snowflake queries to snowflake.
    """

    if bodo.get_size() > 1:
        pytest.skip("This test should only run on a single rank")
    if not bodo.bodosql_use_streaming_plan:
        pytest.skip(
            "This filter pushdown cost estimate is only enabled with the streaming plan"
        )

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # This is a specifically exteneded version of the TPCH Table.
    # The table has 1.5M rows, and the filter will return 1 row.
    # Assuming we're using some heuristic to estimate the number of rows,
    # it's almost certain that the expected ammount will be greater than 1 row.
    # if we're pushing the filter directly to snowflake, we should only
    # see 1 row in the cost estimate.
    sql = "SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'"

    plan = bc.generate_plan(sql, show_cost=True)
    assert "1 rows" in plan, "Plan should have 1 row in the cost estimate"


def test_filter_pushdown_row_count_caching(
    test_db_snowflake_catalog, memory_leak_check
):
    """E2E test that checks that we don't ping snowflake more
    than once when getting the row count for a duplicate filter"""
    bodo.bodosql_use_streaming_plan = True
    comm = MPI.COMM_WORLD
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    conn_str = get_snowflake_connection_string(db, schema)

    if bodo.get_size() > 1:
        pytest.skip("This test should only run on a single rank")
    if not bodo.bodosql_use_streaming_plan:
        pytest.skip(
            "This filter pushdown cost estimate is only enabled with the streaming plan"
        )

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # See test_sf_filter_pushdown_rowcount_estimate for the full
    # Explanation of the TPCH_SF10_CUSTOMER_WITH_ADDITIONS table
    # TLDR we should only see 1 row in the cost estimate. For the
    # Filter1 = "WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'"
    # We expect to see 1 row in the cost estimate.
    # Filter2 = "WHERE C_NATIONKEY = 3"
    # We expect to see 59849 rows in the cost estimate.

    sql = """
    with filter1_table1 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'
    ),
    filter2_table1 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_NATIONKEY = 3
    ),
    filter1_table2 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'
    ),
    filter2_table2 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_NATIONKEY = 3
    ),
    filter1_table3 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'
    ),
    filter1_table4 as (
        SELECT * FROM TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY WHERE C_COMMENT = 'I am the inserted dummy row. I am the only row with this comment'
    )

    Select *
    from
    filter1_table1,
    filter2_table1,
    filter1_table2,
    filter2_table2,
    filter1_table3,
    filter1_table4
    """

    plan = bc.generate_plan(sql, show_cost=True)
    # Since TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY should only
    # be called by this function, and should ONLY be called
    # in the 1 row case on nighlty
    # we can be fairly certain that the any queries to this table in the recent
    # query history stem from this test.
    assert "1 rows" in plan, "Plan should have 1 row in the cost estimate"
    assert "59.849e3 rows" in plan, "Plan should have 59849 rows in the cost estimate"

    # Empirically, it takes a moment for the query history to update,
    # so we sleep for a few seconds to ensure that the query history is updated
    time.sleep(2)

    # This query will get the list of all queries that match the specified pattern
    # in the past minute
    metadata_query = """select * from table(information_schema.QUERY_HISTORY_BY_WAREHOUSE(
                            WAREHOUSE_NAME=>'DEMO_WH',
                            END_TIME_RANGE_START=>dateadd('minutes',-1,current_timestamp()),
                            END_TIME_RANGE_END=>current_timestamp()
                        )
                    ) WHERE CONTAINS(QUERY_TEXT, 'SELECT COUNT(*) FROM (SELECT * FROM "TEST_DB"."PUBLIC"."TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY" WHERE "C_NATIONKEY" = 3)') OR
                            CONTAINS(QUERY_TEXT, 'SELECT COUNT(*) FROM (SELECT * FROM "TEST_DB"."PUBLIC"."TPCH_SF10_CUSTOMER_WITH_ADDITIONS_COPY" WHERE "C_COMMENT" = ')
                    """

    df = pd.read_sql(metadata_query, conn_str)
    # We expect two rows, one for each filter
    assert len(df) == 2, "We should have two rows in the query history"
    assert (
        df["query_text"].str.contains("SELECT COUNT(*)", regex=False).all()
    ), "We should have two queries for the row count"
    assert (
        df["query_text"].str.contains('WHERE "C_NATIONKEY" = 3', regex=False).sum() == 1
    ), "We should have one query for the C_NATIONKEY row estimate"
    assert (
        df["query_text"].str.contains('WHERE "C_COMMENT" = ', regex=False).sum() == 1
    ), "We should have one query for the C_COMMENT row estimate"


def test_snowflake_catalog_string_format(test_db_snowflake_catalog, memory_leak_check):
    """Tests a specific issue with the unparsing of strings for snowflake query submission."""
    bodo.bodosql_use_streaming_plan = True

    if bodo.get_size() > 1:
        pytest.skip("This test should only run on a single rank")

    # Created a special table with a single addition to the lineitem table
    # should be a single row ouput
    query = """
        select
            count(*) as output_count
        from
            TPCH_SF1_LINEITEM_WITH_ADDITIONS
        where
            l_comment = '♪ ♫ ♬ ♭ ♮ \n\r\t À È Ì © € ∞ ½⅓¼⅕⅙⅐'
    """

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    out = bc.sql(query)
    assert out.iloc[0, 0] == 1, f"Expected one row in output, found {out.iloc[0,0]}"


def test_read_with_array(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests reading a table with a Snowflake column.
    The table BODOSQL_ARRAY_READ_TEST has 3 columns:
    - I: integers
    - V: strings
    - A: arrays of integers
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    out: pd.DataFrame = bc.sql("SELECT * FROM BODOSQL_ARRAY_READ_TEST")
    assert len(out) == 100
    assert len(out.columns) == 3
    assert all(isinstance(i, pd.core.arrays.integer.IntegerArray) for i in out["a"])


def test_float_array_read(test_db_snowflake_catalog, memory_leak_check):
    """Test reading an array of floating-point column from Snowflake"""

    def impl(bc):
        return bc.sql("SELECT * FROM ARRAY_NUMBER_TEST")

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "a": [
                [pd.NA, 12.4, -0.57],
                [np.nan, np.inf, -np.inf],
                [-1235.0, 0.01234567890123456789],
                [10.0, 10.0, pd.NA],
                [12345678901234567890.0],
                np.nan,
            ],
        }
    )
    check_func(
        impl,
        (bc,),
        py_output=py_output,
        sort_output=True,
        reset_index=True,
    )


def test_string_array_read(test_db_snowflake_catalog, memory_leak_check):
    """Test reading an array of string column from Snowflake"""

    def impl(bc):
        return bc.sql("SELECT * FROM ARRAY_STRING_TEST")

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "a": [
                ["\n    test multiline \\t  \n    string with junk\n    "],
                ["\041", "\x21", "\u26c4", "z", "\b", "\f", "/"],
                ["'", '"', '"', "\t\n", "\\"],
                ["test \0 zero"],
                ["true", "10", "2023-10-20", "hello"],
                ["why", "does", "snowflake", "use", pd.NA],
                np.nan,
            ],
        }
    )
    check_func(
        impl,
        (bc,),
        py_output=py_output,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )


@pytest.mark.parametrize(
    "condition, answer",
    [
        pytest.param(
            "data:container = 'SM'",
            [0, 4, 11, 13, 16, 17, 20, 24, 31, 33, 36, 37],
            id="string_to_variant",
        ),
        pytest.param(
            "GET_PATH(data, 'color') LIKE '%o%' AND data:price >= 1025.0",
            [4, 5, 7, 11, 12, 14, 15, 16, 17, 19, 24, 25, 27, 28, 29, 31, 32, 39],
            id="variant_to_string_and_float",
        ),
    ],
)
def test_snowflake_json_filter_pushdown(
    condition, answer, test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests reading from a Snowflake table with a filter condition based
    on a JSON field.
    """
    # Only test on rank zero
    if bodo.get_rank() != 0:
        return
    query = f"SELECT id FROM BODOSQL_JSON_READ_TEST WHERE {condition}"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    answer_df = pd.DataFrame({"id": answer})
    out = bc.sql(query)
    pd.testing.assert_frame_equal(
        out,
        answer_df,
        check_names=False,
        check_dtype=False,
        check_index_type=False,
        check_column_type=False,
    )


@pytest.mark.parametrize(
    "fields, clauses, answer",
    [
        pytest.param(
            ["LEFT(data:color, 1) AS color_first_letter", "COUNT(*) as color_count"],
            ["GROUP BY color_first_letter", "ORDER BY color_first_letter"],
            pd.DataFrame(
                {
                    "color_first_letter": list("abcdfghlmoprstvwy"),
                    "color_count": [2, 2, 2, 1, 5, 1, 1, 1, 4, 2, 4, 3, 5, 3, 1, 1, 2],
                }
            ),
            id="count_color_first_letter",
        ),
        pytest.param(
            ["data:container::varchar AS size", "AVG(data:price::float) AS avg_price"],
            ["GROUP BY size", "ORDER BY avg_price"],
            pd.DataFrame(
                {
                    "size": ["LG", "WRAP", "MED", "SM", "JUMBO"],
                    "avg_price": [
                        1_026.67,
                        1_028.2525,
                        1_031.006666667,
                        1_031.173333333,
                        1_034.26,
                    ],
                }
            ),
            id="price_avg_size",
        ),
        pytest.param(
            [
                "UPPER(data:color) AS ucolor",
                "WIDTH_BUCKET(data:price::float, 1020, 1040, 25) AS prange",
            ],
            ["WHERE data:price < 1025.0", "ORDER BY data:rank::integer"],
            pd.DataFrame(
                {
                    "ucolor": [
                        "YELLOW",
                        "PALE",
                        "YELLOW",
                        "HOT",
                        "PURPLE",
                        "CREAM",
                        "METALLIC",
                        "ROSY",
                    ],
                    "prange": [3, 3, 2, 2, 6, 6, 4, 4],
                }
            ),
            id="cheap_ranked_colors",
        ),
    ],
)
def test_snowflake_json_field_pushdown(
    fields, clauses, answer, test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests reading from a Snowflake table with the extraction of JSON fields ocurring in Snowflake.

    Args:
        fields: entries that are placed in the SELECT clause of the query.
        clauses: any additional clauses provided after the FROM clause (e.g. GROUP BY, WHERE, ORDER BY).
        answer: the expected output for the query.
    """
    # Only test with one rank
    if bodo.get_size() != 1:
        return
    query = f"SELECT {', '.join(fields)} FROM BODOSQL_JSON_READ_TEST"
    for clause in clauses:
        query += f"\n{clause}"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    out = bc.sql(query)
    pd.testing.assert_frame_equal(
        out,
        answer,
        check_names=False,
        check_dtype=False,
        check_index_type=False,
        check_column_type=False,
        rtol=1e-5,
        atol=1e-8,
    )


def _check_stream_unify_opt(impl, bc, query, fdef, arg_no):
    """Check the IR to make sure unification optimization worked ('fdef' call arg
    'arg_no' is set to types.Omitted(True))
    """
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(bc, query)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    calltypes = bodo_func.overloads[bodo_func.signatures[0]].metadata[
        "preserved_calltypes"
    ]
    init_found = False
    for block in fir.blocks.values():
        for stmt in block.body:
            if is_call_assign(stmt) and guard(find_callname, fir, stmt.value) == fdef:
                assert calltypes[stmt.value].args[arg_no] == types.Omitted(
                    True
                ), "input_dicts_unified is not set to true in init"
                init_found = True

    assert init_found


@pytest.mark.skipif(
    not bodo.bodosql_use_streaming_plan, reason="Only relevant for streaming"
)
def test_stream_unification_opt(test_db_snowflake_catalog, memory_leak_check):
    """Make sure compiler optimization sets input_dicts_unified flag of
    snowflake_writer_init() and init_table_builder_state() to true after a join
    """
    catalog = test_db_snowflake_catalog
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    bc = bodosql.BodoSQLContext(catalog=catalog)

    df1 = pd.DataFrame(
        {
            "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
            "C": ["T1_1", "T1_2", "T1_3", "T1_4", "T1_5", "T1_6"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
            "D": ["T2_1", "T2_2", "T2_3", "T2_4", "T2_5", "T2_6"] * 2,
        }
    )
    bc = bc.add_or_replace_view("table1", df1)
    bc = bc.add_or_replace_view("table2", df2)

    def impl(bc, query):
        bc.sql(query)

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("bodosql_catalog_write_unification_opt")
    table_name = comm.bcast(table_name)

    # test streaming write
    query = f"CREATE OR REPLACE TEMPORARY TABLE {schema}.{table_name} as select C, D from __bodolocal__.table1 inner join __bodolocal__.table2 on __bodolocal__.table1.A = __bodolocal__.table2.A"
    _check_stream_unify_opt(
        impl,
        bc,
        query,
        (
            "snowflake_writer_init",
            "bodo.io.snowflake_write",
        ),
        -2,
    )

    # test combine exchange
    query = "select C, D from __bodolocal__.table1 inner join __bodolocal__.table2 on __bodolocal__.table1.A = __bodolocal__.table2.A"
    _check_stream_unify_opt(
        impl, bc, query, ("init_table_builder_state", "bodo.libs.table_builder"), -1
    )


def test_hidden_credentials(snowflake_sample_data_snowflake_catalog, memory_leak_check):
    """
    Test that the given username and password are not embedded into the generate code
    when calling convert_to_pandas.
    """
    query = "Select * from CUSTOMER"
    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    code = bc.convert_to_pandas(query)
    assert snowflake_sample_data_snowflake_catalog.username not in code
    assert snowflake_sample_data_snowflake_catalog.password not in code
    assert snowflake_sample_data_snowflake_catalog.account not in code
