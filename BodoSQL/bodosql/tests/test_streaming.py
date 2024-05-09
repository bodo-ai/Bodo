# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test correctness of BodoSQL operations that handle streaming data.
"""
import io

import pytest

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, pytest_snowflake
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_snowflake_catalog,
)

pytestmark = pytest_snowflake


@pytest.mark.skip(
    "Appears to be running on AWS CI, which is causing a number of PR CI failures, so temporarily skipping: https://bodo.atlassian.net/browse/BSE-1059"
)
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


def test_streaming_groupby_aggregate_timer(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """Tests that we properly handle timer information for streaming groupby aggregate"""

    if not bodo.bodosql_use_streaming_plan:
        pytest.skip("Test only runs with streaming plan enabled")

    # Row number query is needed to prevent Aggregate push down into Snowflake
    query = """SELECT C_ADDRESS, MAX(ROW_NUM) FROM
        (SELECT *, ROW_NUMBER() OVER (PARTITION BY C_ADDRESS ORDER BY C_CUSTKEY) as ROW_NUM FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER)
        GROUP BY C_ADDRESS"""

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    def impl(bc):
        return bc.sql(query)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 2):
        plan = bc.generate_plan(query)
        bodo.jit(impl)(bc)

        assert (
            "BodoPhysicalAggregate" in plan
        ), "Streaming plan did not actually generate a BodoPhysicalAggregate node"
        for relNodeStr in plan.split("\n"):
            relNodeStr = relNodeStr.strip()
            # Confirm that all the nodes in the plan that should
            # show up in the generated log, actually do.
            if not (
                relNodeStr.startswith("PandasTableScan")
                or relNodeStr.startswith("CombineStreamsExchange")
                or relNodeStr.startswith("SeparateStreamExchange")
                or relNodeStr.startswith("Snowflake")
            ):
                check_logger_msg(stream, relNodeStr, check_case=False)
