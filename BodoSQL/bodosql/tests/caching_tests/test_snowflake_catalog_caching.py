"""
Test that Snowflake Catalogs are cached properly in both
sequential and parallel code.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import os

import bodosql

from bodo.tests.caching_tests.caching_tests_common import (  # noqa
    fn_distribution,
)
from bodo.tests.utils import check_caching
import pytest


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_catalog_caching(fn_distribution, is_cached):
    def impl(bc):
        return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

    # The "is_cached" fixture is essentially a wrapper that returns the value
    # of the --is_cached flag used when invoking pytest (defualts to "n").
    # runtests_caching will pass this flag, depending on if we expect the
    # current test to be cached.
    check_cache = is_cached == "y"

    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USER"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        "SNOWFLAKE_SAMPLE_DATA",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    check_caching(
        impl,
        (bc,),
        check_cache,
        input_dist=fn_distribution,
)
