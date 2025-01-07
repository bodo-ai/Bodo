import pandas as pd

import bodosql
from bodo.tests.utils import (
    check_func,
    pytest_s3_tables,
)

pytestmark = pytest_s3_tables


def test_basic_read(memory_leak_check, s3_tables_catalog):
    """
    Test reading an entire Iceberg table from S3 Tables in SQL
    """
    bc = bodosql.BodoSQLContext(catalog=s3_tables_catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "bob", "cassie", "david", None],
            "B": [10.5, -124.0, 11.11, 456.2, -8e2],
            "C": [True, None, False, None, None],
        }
    )

    query = 'SELECT * FROM "read_namespace"."bodo_iceberg_read_test"'
    check_func(
        impl,
        (bc, query),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )
