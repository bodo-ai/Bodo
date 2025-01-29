import os
from io import StringIO

import pandas as pd
import pytest

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
    pytest_snowflake,
)

pytestmark = [pytest.mark.iceberg] + pytest_snowflake


def test_prefetch_flag(fn_distribution, is_cached, tmp_path, memory_leak_check):
    """
    Test that if the prefetch flag is set, a prefetch occurs
    """

    old_prefetch_flag = bodo.prefetch_sf_iceberg
    try:
        bodo.prefetch_sf_iceberg = True

        catalog = bodosql.SnowflakeCatalog(
            os.environ["SF_USERNAME"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "TEST_DB",
            connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
            iceberg_volume="exvol",
        )
        bc = bodosql.BodoSQLContext(catalog=catalog)

        def impl(bc, query):
            return bc.sql(query)

        py_out = pd.DataFrame(
            {
                "A": ["ally", "bob", "cassie", "david", pd.NA],
                "B": [10.5, -124.0, 11.11, 456.2, -8e2],
                "C": [True, pd.NA, False, pd.NA, pd.NA],
            }
        )

        # The "is_cached" fixture is essentially a wrapper that returns the value
        # of the --is_cached flag used when invoking pytest (defaults to "n").
        # runtests_caching will pass this flag, depending on if we expect the
        # current test to be cached.
        check_cache = is_cached == "y"

        query = "SELECT A, B, C FROM BODOSQL_ICEBERG_READ_TEST"
        stream = StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_caching(
                impl,
                (bc, query),
                check_cache,
                input_dist=fn_distribution,
                py_output=py_out,
                sort_output=True,
                reset_index=True,
            )

            check_logger_msg(
                stream,
                'Execution time for prefetching SF-managed Iceberg metadata "TEST_DB"."PUBLIC"."BODOSQL_ICEBERG_READ_TEST"',
            )
            check_logger_msg(
                stream,
                'Loading table `"TEST_DB"."PUBLIC"."BODOSQL_ICEBERG_READ_TEST"` from prefetch',
            )

    finally:
        bodo.prefetch_sf_iceberg = old_prefetch_flag
