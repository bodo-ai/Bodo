import os

import bodosql
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_invalid_credentials_err():
    def impl(bc, query):
        return bc.sql(query)

    # Incorrect Snowflake Username Catalog
    invalid_catalog1 = bodosql.SnowflakeCatalog(
        "invalid",
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        "SNOWFLAKE_SAMPLE_DATA",
    )

    # Incorrect Snowflake username + password
    invalid_catalog2 = bodosql.SnowflakeCatalog(
        "invalid",
        "invalid",
        "bodopartner.us-east-1",
        "DEMO_WH",
        "SNOWFLAKE_SAMPLE_DATA",
    )

    query = "SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name"
    bc = bodosql.BodoSQLContext(catalog=invalid_catalog1)

    # Test for incorrect Snowflake Username
    with pytest.raises(BodoError, match="Incorrect username or password was specified"):
        bodo.jit(impl)(bc, query)

    # Test for incorrect Snowflake Password
    bc = bodosql.BodoSQLContext(catalog=invalid_catalog2)
    with pytest.raises(BodoError, match="Incorrect username or password was specified"):
        bodo.jit(impl)(bc, query)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_data_not_found_err():
    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog(
            os.environ["SF_USER"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "SNOWFLAKE_SAMPLE_DATA",
        )
    )

    invalid_schema_query = "SELECT r_name FROM invalid.REGION ORDER BY r_name"
    invalid_table_query = "SELECT r_name FROM TPCH_SF1.invalid ORDER BY r_name"

    with pytest.raises(BodoError, match="Object .+ not found"):
        bodo.jit(impl)(bc, invalid_schema_query)

    with pytest.raises(BodoError, match="Object .+ not found"):
        bodo.jit(impl)(bc, invalid_table_query)
