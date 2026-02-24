from uuid import getnode

import pandas as pd
from mpi4py import MPI

import bodosql
from bodo.tests.caching_tests.caching_tests_common import (  # noqa
    fn_distribution,
)
from bodo.tests.utils import (
    check_caching,
    pytest_mark_polaris,
)
from bodosql.tests.test_types.conftest import (  # noqa
    aws_polaris_warehouse,
    azure_polaris_warehouse,
    polaris_catalog,
    polaris_catalog_iceberg_read_df,
    polaris_connection,
    polaris_package,
    polaris_server,
    polaris_token,
)


@pytest_mark_polaris
def test_rest_catalog_read_caching(
    fn_distribution, is_cached, polaris_catalog, polaris_catalog_iceberg_read_df
):
    def impl(bc, query):
        return bc.sql(query)

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defaults to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    query = 'SELECT "A" FROM "CI"."BODOSQL_ICEBERG_READ_TEST"'

    bc = bodosql.BodoSQLContext(catalog=polaris_catalog)

    check_caching(
        impl,
        (bc, query),
        check_cache,
        input_dist=fn_distribution,
        py_output=polaris_catalog_iceberg_read_df[["A"]],
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_polaris
def test_rest_catalog_write_caching(
    fn_distribution, is_cached, polaris_catalog, polaris_catalog_iceberg_read_df
):
    def impl(bc, write_query, read_query):
        # Write step
        bc.sql(write_query)
        return bc.sql(read_query)

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defaults to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    # Get hardware address to use as part of table name
    # I don't think the mac address should change between mpi nodes in a cluster
    # (at least, not for CI, since it's all on one worker machine)
    # but just in case, we'll bcast it.
    hardware_address = str(getnode())
    table_name = "EXAMPLE_CACHE_WRITE_TABLE_" + hardware_address
    table_name = MPI.COMM_WORLD.bcast(table_name, root=0)

    write_query = f'create or replace table CI."{table_name}" as select * from CI.BODOSQL_ICEBERG_READ_TEST'
    read_query = f'select * from CI."{table_name}"'
    py_out = pd.DataFrame(
        {
            "A": pd.array(["ally", "bob", "cassie", "david", pd.NA]),
            "B": pd.array([10.5, -124.0, 11.11, 456.2, -8e2], dtype="float64"),
            "C": pd.array([True, pd.NA, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )

    bc = bodosql.BodoSQLContext(catalog=polaris_catalog)
    created = False
    try:
        # Make sure the table exists so the the read query doesn't fail to compile
        bc.sql(write_query)
        created = True

        check_caching(
            impl,
            (bc, write_query, read_query),
            check_cache,
            input_dist=fn_distribution,
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
    finally:
        if created:
            bc.sql(f'DROP TABLE IF EXISTS CI."{table_name}"')
