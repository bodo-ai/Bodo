import pandas as pd
import pyarrow as pa

import bodo
import bodosql
from bodo.io.iceberg.catalog.s3_tables import S3TablesCatalog
from bodo.spawn.utils import run_rank0
from bodo.tests.utils import (
    assert_tables_equal,
    check_func,
    gen_unique_table_id,
    pytest_s3_tables,
    temp_env_override,
)
from bodosql.bodosql_types.s3_tables_catalog_ext import S3TablesConnectionType

pytestmark = pytest_s3_tables


# Refer to bodo/tests/test_s3_tables_iceberg.py for infrastructure
# required to run these tests
@temp_env_override({"AWS_DEFAULT_REGION": "us-east-2"})
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
            "C": pd.array(
                [True, None, False, None, None], dtype=pd.ArrowDtype(pa.bool_())
            ),
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


@temp_env_override({"AWS_DEFAULT_REGION": "us-east-2", "AWS_REGION": "us-east-2"})
def test_s3_tables_catalog_iceberg_write(s3_tables_catalog, memory_leak_check):
    """tests that writing tables works"""

    in_df = pd.DataFrame(
        {
            "ints": list(range(100)),
            "floats": [float(x) for x in range(100)],
            "str": [str(x) for x in range(100)],
            "dict_str": ["abc", "df"] * 50,
        }
    )
    bc = bodosql.BodoSQLContext(catalog=s3_tables_catalog)
    bc = bc.add_or_replace_view("TABLE1", in_df)
    con_str = S3TablesConnectionType(s3_tables_catalog.warehouse).conn_str
    table_name = run_rank0(
        lambda: gen_unique_table_id("bodosql_catalog_write_iceberg_table").upper()
    )().lower()

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    ctas_query = f'CREATE OR REPLACE TABLE "write_namespace"."{table_name}" AS SELECT * from __bodolocal__."TABLE1"'
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

        output_df = read_results(con_str, "write_namespace", table_name)
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
                lambda: (
                    S3TablesCatalog(
                        "s3_tables_catalog",
                        **{"s3tables.warehouse": s3_tables_catalog.warehouse},
                    ).purge_table(f"write_namespace.{table_name}")
                )
            )()
        except Exception:
            if exception_occurred_in_test_body:
                pass
            else:
                raise
