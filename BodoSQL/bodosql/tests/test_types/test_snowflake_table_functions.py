"""
Test correctness of table functions that ping Snowflake for data.
"""

import pandas as pd

import bodosql
from bodo.tests.utils import (
    check_func,
    pytest_snowflake,
)
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa: F401
    snowflake_sample_data_conn_str,
    snowflake_sample_data_snowflake_catalog,
    test_db_snowflake_catalog,
)

pytestmark = pytest_snowflake
from decimal import Decimal

from bodo.tests.utils import check_func


def test_external_table_files(test_db_snowflake_catalog, datapath, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    query = 'SELECT * FROM TABLE(INFORMATION_SCHEMA.EXTERNAL_TABLE_FILES(TABLE_NAME=>\'"TEST_DB"."PUBLIC"."CITIES_EXTERNAL_TABLE"\'))'
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    # Read the refsol and transform the non-string columns to the appropriate types
    answer = pd.read_csv(datapath("external_table_file_results.csv"))
    answer["REGISTERED_ON"] = answer["REGISTERED_ON"].apply(pd.Timestamp)
    answer["FILE_SIZE"] = answer["FILE_SIZE"].apply(Decimal)
    answer["LAST_MODIFIED"] = answer["LAST_MODIFIED"].apply(pd.Timestamp)
    check_func(
        impl,
        (bc, query),
        py_output=answer,
        check_names=False,
        check_dtype=False,
    )
