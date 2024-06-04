import os

import pandas as pd
import pytest

import bodosql
from bodo.tests.utils import check_func, pytest_glue

pytestmark = pytest_glue


@pytest.mark.skipif(
    "AGENT_NAME" in os.environ,
    reason="BSE-3425: Permissions error only in azure environment",
)
def test_basic_read(memory_leak_check, glue_catalog):
    """
    Test reading an entire Iceberg table from Glue in SQL
    """
    bc = bodosql.BodoSQLContext(catalog=glue_catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "a": pd.array(["ally", "bob", "cassie", "david", pd.NA]),
            "b": pd.array([10.5, -124.0, 11.11, 456.2, -8e2], dtype="float64"),
            "c": pd.array([True, pd.NA, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )

    query = 'SELECT * FROM "icebergglueci"."iceberg_read_test"'
    check_func(
        impl,
        (bc, query),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )
