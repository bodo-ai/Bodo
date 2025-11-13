import os

import pandas as pd
import pytest
from pyiceberg.catalog.glue import GlueCatalog

import bodo
import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.utils import (
    assert_tables_equal,
    check_func,
    gen_unique_table_id,
    pytest_glue,
)
from bodosql.bodosql_types.glue_catalog_ext import GlueConnectionType

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


@pytest.mark.skipif(
    "AGENT_NAME" in os.environ,
    reason="BSE-3425: Permissions error only in azure environment",
)
def test_glue_catalog_iceberg_write(glue_catalog, memory_leak_check):
    """tests that writing tables works"""
    bc = bodosql.BodoSQLContext(catalog=glue_catalog)
    con_str = GlueConnectionType(glue_catalog.warehouse).conn_str
    schema = "icebergglueci"

    in_df = pd.DataFrame(
        {
            "ints": list(range(100)),
            "floats": [float(x) for x in range(100)],
            "str": [str(x) for x in range(100)],
            "dict_str": ["abc", "df"] * 50,
        }
    )
    bc = bc.add_or_replace_view("TABLE1", in_df)

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    table_name = run_rank0(
        lambda: gen_unique_table_id("bodosql_catalog_write_iceberg_table").upper()
    )()

    # glue requires schema/table names to be lowercase
    table_name = table_name.lower()

    ctas_query = f'CREATE OR REPLACE TABLE "{schema}"."{table_name}" AS SELECT * from __bodolocal__.table1'
    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(
            impl,
            (bc, ctas_query),
            only_1D=True,
            py_output=5,
            use_table_format=True,
        )

        @bodo.jit
        def read_results(con_str, schema, table_name):
            output_df = pd.read_sql_table(table_name, con=con_str, schema=schema)
            return bodo.allgatherv(output_df)

        output_df = read_results(con_str, schema, table_name)
        assert_tables_equal(output_df, in_df, check_dtype=False)

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call delete_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        try:
            run_rank0(
                lambda: GlueCatalog("glue_catalog").purge_table(
                    f"{schema}.{table_name}"
                )
            )()
        except Exception:
            if exception_occurred_in_test_body:
                pass
            else:
                raise
