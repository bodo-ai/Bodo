"""
Test that Snowflake Catalogs are cached properly in both
sequential and parallel code.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import os
import traceback

import bodosql
import numpy as np
import pandas as pd

import bodo
from bodo.tests.caching_tests.caching_tests_common import (  # noqa
    fn_distribution,
)
from bodo.tests.utils import (
    check_caching,
    drop_snowflake_table,
    get_snowflake_connection_string,
    pytest_mark_snowflake,
    reduce_sum,
)


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
            "a": [1, 2, 3] * 10,
            "b": np.arange(30),
        }
    )

    # We must generate consistent names for caching.
    read_table = "EXAMPLE_CACHE_READ_TABLE"
    write_table = "EXAMPLE_CACHE_WRITE_TABLE"

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
        drop_snowflake_table(read_table, db, schema)
        drop_snowflake_table(write_table, db, schema)
