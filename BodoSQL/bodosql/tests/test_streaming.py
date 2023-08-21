# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test correctness of BodoSQL operations that handle streaming data.
"""
import bodosql
import pytest
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_snowflake_catalog,
)

import bodo
from bodo.tests.utils import check_func


@pytest.mark.pytest_snowflake
def test_streaming_cache(snowflake_sample_data_snowflake_catalog, memory_leak_check):
    if not bodo.bodosql_use_streaming_plan:
        pytest.skip("Test only runs with streaming plan enabled")

    query = """SELECT * FROM
        (SELECT C_ADDRESS FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER) t1
        JOIN
        (SELECT C_ADDRESS FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER) t2
        ON t1.C_ADDRESS = t2.C_ADDRESS
        JOIN
        (SELECT C_ADDRESS FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER) t3
        ON t1.C_ADDRESS = t3.C_ADDRESS
        JOIN
        (SELECT C_ADDRESS FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER) t4
        ON t1.C_ADDRESS = t4.C_ADDRESS
    """
    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

    # First, confirm that we're actually caching the read.
    pd_code = bc.convert_to_pandas(query)
    assert pd_code.count("read_sql") == 1, "read_sql called more than once"

    # Then, check the correctness of the streaming caching
    check_func(
        impl,
        (bc, query),
        sort_output=True,
        reset_index=True,
    )
