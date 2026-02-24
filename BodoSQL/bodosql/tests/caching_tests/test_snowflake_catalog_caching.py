"""
Test that Snowflake Catalogs are cached properly in both
sequential and parallel code.
"""

import io
import os
import traceback
from uuid import getnode

import numpy as np
import pandas as pd
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.caching_tests.caching_tests_common import (  # noqa
    fn_distribution,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_caching,
    drop_snowflake_table,
    get_snowflake_connection_string,
    pytest_mark_snowflake,
)
from bodo.tests.utils_jit import reduce_sum


@pytest_mark_snowflake
def test_snowflake_catalog_caching(fn_distribution, is_cached):
    def impl(bc, query):
        return bc.sql(query)

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defaults to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    query = "SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name"

    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string(db, schema),
    )
    py_output.columns = py_output.columns.str.upper()

    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USERNAME"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        db,
        connection_params={"schema": schema},
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    check_caching(
        impl,
        (bc, query),
        check_cache,
        input_dist=fn_distribution,
        py_output=py_output,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_snowflake
def test_snowflake_catalog_write_caching(fn_distribution, is_cached):
    def impl(bc, write_query, read_query):
        # Write step
        bc.sql(write_query)
        return bc.sql(read_query)

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defaults to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    db = "TEST_DB"
    schema = "PUBLIC"

    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USERNAME"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        db,
        connection_params={"schema": schema},
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    new_df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 10,
            "B": np.arange(30),
        }
    )

    # We must generate consistent names that are consistent between runs for caching,
    # However the names must also be different on different machines so we don't
    # run into issues if
    # we're running this test in two places at the same time.
    # Therefore we use the MAC address of the current machine, since this
    # shouldn't change between runs, but should be different if this test is being run
    # on two different machines.

    read_table = None
    write_table = None
    if bodo.get_rank() == 0:
        # I don't think the mac address should change between mpi nodes in a cluster
        # (at least, not for CI, since it's all on one worker machine)
        # but just in case, we'll bcast it.
        hardware_address = str(getnode())
        read_table = "EXAMPLE_CACHE_READ_TABLE_" + hardware_address
        write_table = "EXAMPLE_CACHE_WRITE_TABLE_" + hardware_address

    read_table = MPI.COMM_WORLD.bcast(read_table)
    write_table = MPI.COMM_WORLD.bcast(write_table)

    passed = 1
    try:
        err = "see error on rank 0"
        if bodo.get_rank() == 0:
            try:
                conn_str = get_snowflake_connection_string(db, schema)
                new_df.to_sql(
                    read_table.lower(), conn_str, index=False, if_exists="replace"
                )
                # Create the write table for caching.
                new_df.to_sql(
                    write_table.lower(), conn_str, index=False, if_exists="replace"
                )
            except Exception as e:
                passed = 0
                err = "".join(traceback.format_exception(None, e, e.__traceback__))
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), err

        write_query = (
            f"create or replace table {write_table} as select * from {read_table}"
        )
        read_query = f"select * from {write_table}"
        check_caching(
            impl,
            (bc, write_query, read_query),
            check_cache,
            input_dist=fn_distribution,
            py_output=new_df,
            sort_output=True,
            reset_index=True,
        )
    finally:
        # Try to drop the tables.
        # If the error ocurred before their creation, they may not exist,
        # hence the try/except blocks
        error = None
        try:
            drop_snowflake_table(read_table, db, schema)
        except Exception as e:
            error = e

        try:
            drop_snowflake_table(write_table, db, schema)
        except Exception as e:
            error = e

        if error is not None:
            raise error


@pytest_mark_snowflake
def test_snowflake_runtime_join_filter_caching(is_cached):
    """
    Tests usage of caching on a query that will also trigger a
    min/max runtime join filter pushed into the I/O.
    """

    def impl(bc, query):
        return bc.sql(query)

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defaults to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    db = "TEST_DB"
    schema = "PUBLIC"

    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USERNAME"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        db,
        connection_params={"schema": schema},
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    answer = pd.DataFrame({"N_SUPPLIERS": [131]})

    # Count the number of unique suppliers who supply at least one
    # part that has the specified properties. Should create a runtime
    # join filter on partsupp with the following predicates:
    # - ps_partkey >= 1977
    # - ps_suppkey <= 195380
    query = """
    SELECT COUNT(DISTINCT ps_suppkey) AS n_suppliers
    FROM tpch_sf1.part, tpch_sf1.partsupp
    WHERE p_name ILIKE '%yellow%'
        AND p_name ILIKE '%blue%'
        AND p_type ILIKE '%anodized%'
        AND p_size > 30
        AND p_partkey = ps_partkey
    """
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_caching(
            impl,
            (bc, query),
            check_cache,
            input_dist=fn_distribution,
            py_output=answer,
            sort_output=True,
            reset_index=True,
            is_out_dist=False,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM "TEST_DB"."TPCH_SF1"."PARTSUPP" WHERE "PS_PARTKEY" IS NOT NULL GROUP BY "PS_PARTKEY", "PS_SUPPKEY") as TEMP) WHERE TRUE AND ($1 >= 1977) AND ($1 <= 195380)',
        )
