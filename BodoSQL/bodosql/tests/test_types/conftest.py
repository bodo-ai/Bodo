import pandas as pd
import pyarrow as pa
import pytest

import bodosql
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.spawn.utils import run_rank0
from bodo.tests.test_iceberg.conftest import (  # noqa
    aws_polaris_connection,
    aws_polaris_warehouse,
    azure_polaris_warehouse,
    polaris_connection,
    polaris_package,
    polaris_server,
    polaris_token,
)
from bodo.tests.utils import get_rest_catalog_connection_string


@pytest.fixture()
def polaris_catalog_iceberg_read_df(polaris_connection):
    rest_uri, polaris_warehouse, polaris_credential = polaris_connection
    df = pd.DataFrame(
        {
            "A": pd.array(["ally", "bob", "cassie", "david", pd.NA]),
            "B": pd.array([10.5, -124.0, 11.11, 456.2, -8e2], dtype="float64"),
            "C": pd.array([True, pd.NA, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )
    con_str = get_rest_catalog_connection_string(
        rest_uri, polaris_warehouse, polaris_credential
    )
    py_catalog = conn_str_to_catalog(con_str)
    table_id = "CI.BODOSQL_ICEBERG_READ_TEST"
    run_rank0(
        lambda: py_catalog.create_table(table_id, pa.Schema.from_pandas(df)).append(
            pa.Table.from_pandas(df)
        )
    )()
    yield df
    run_rank0(lambda: py_catalog.purge_table(table_id))()


@pytest.fixture
def polaris_catalog(polaris_connection):
    """
    Returns a polaris catalog object
    """
    rest_uri, polaris_warehouse, polaris_credential = polaris_connection

    return bodosql.RESTCatalog(
        rest_uri=rest_uri,
        warehouse=polaris_warehouse,
        credential=polaris_credential,
        scope="PRINCIPAL_ROLE:ALL",
    )


@pytest.fixture
def aws_polaris_catalog(aws_polaris_connection):
    """
    Returns a polaris catalog object
    For cases where we can't used paratmerized fixtures
    like the iceberg ddl test harness
    """
    rest_uri, polaris_warehouse, polaris_credential = aws_polaris_connection

    return bodosql.RESTCatalog(
        rest_uri=rest_uri,
        warehouse=polaris_warehouse,
        credential=polaris_credential,
        scope="PRINCIPAL_ROLE:ALL",
    )
