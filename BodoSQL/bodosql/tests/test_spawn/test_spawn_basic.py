import bodosql
from bodo.submit.spawner import submit_jit
from bodo.tests.utils import pytest_mark_snowflake, pytest_spawn_mode
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_snowflake_catalog,
)

pytestmark = pytest_spawn_mode

QUERY = """
SELECT * FROM SUPPLIER
"""


@submit_jit
def exec_query(bc):
    # We only support passing in BodoSQLContext to Spawn mode, so
    # we will read the query as a global.
    output = bc.sql(QUERY)
    # We don't support returning the output in Spawn mode yet,
    # so we will just print it for now.
    print("Output:")
    print(output)


@pytest_mark_snowflake
def test_basic_sql(snowflake_sample_data_snowflake_catalog):
    """
    Simple test to ensure that we can run SQL queries in Spawn mode.
    """
    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    exec_query(bc)
