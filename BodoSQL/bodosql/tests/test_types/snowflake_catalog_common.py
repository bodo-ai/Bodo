# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""File with common fixtures used for testing snowflake catalog."""
import os

import bodosql
import pytest


@pytest.fixture(
    params=[
        bodosql.SnowflakeCatalog(
            os.environ.get("SF_USERNAME", ""),
            os.environ.get("SF_PASSWORD", ""),
            "bodopartner.us-east-1",
            "DEMO_WH",
            "TEST_DB",
            connection_params={"schema": "PUBLIC"},
        )
    ]
)
def test_db_snowflake_catalog(request):
    """
    The test_db snowflake catalog used for most tests.
    Although this is a fixture there is intentionally a
    single element.
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
