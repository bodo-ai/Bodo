from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyiceberg.catalog import Catalog

import bodo
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    get_rest_catalog_connection_string,
    pytest_polaris,
)

pytestmark = pytest_polaris


def test_iceberg_polaris_read(polaris_connection, memory_leak_check):
    """
    Test reading an Iceberg table from a Polaris REST catalog.
    Checksum is used to verify the data is read correctly.
    Column names are used to verify the schema is read correctly.
    """
    from bodo.io.iceberg.catalog import conn_str_to_catalog
    from bodo.utils.utils import run_rank0

    rest_uri, polaris_warehouse, polaris_credential = polaris_connection
    con_str = get_rest_catalog_connection_string(
        rest_uri, polaris_warehouse, polaris_credential
    )
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [1.1, 2.2, 3.3],
            "D": [True, False, True],
        }
    )
    namespace = "CI"
    table_name = "read_test"
    table_id = f"{namespace}.{table_name}"
    try:
        py_catalog: Catalog = conn_str_to_catalog(con_str)
        run_rank0(
            lambda: py_catalog.create_table(table_id, pa.Schema.from_pandas(df)).append(
                pa.Table.from_pandas(df)
            )
        )()

        def f():
            df = pd.read_sql_table(
                table_name,
                con=con_str,
                schema=namespace,
            )
            return df

        check_func(f, (), py_output=df)
    finally:
        run_rank0(lambda: py_catalog.purge_table(table_id))()


@pytest.mark.parametrize(
    "df, rank_skew",
    [
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), False, id="values_all_ranks"
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), True, id="values_rank_0"
        ),
        pytest.param(
            pd.DataFrame({"A": np.empty((0,), dtype=np.int32)}), False, id="empty"
        ),
    ],
)
def test_iceberg_polaris_write_basic(
    df, polaris_connection, memory_leak_check, rank_skew
):
    """
    Test writing to an Iceberg table in a Polaris REST catalog.
    Checksum is used to verify the data is written correctly.
    """
    from bodo.io.iceberg.catalog import conn_str_to_catalog
    from bodo.utils.utils import run_rank0

    rest_uri, polaris_warehouse, polaris_credential = polaris_connection
    table_uuid = run_rank0(uuid4)()
    table_name = f"bodo_write_test_{table_uuid}"

    con_str = get_rest_catalog_connection_string(
        rest_uri, polaris_warehouse, polaris_credential
    )
    write_complete = False
    try:

        def f(df, table_name):
            df.to_sql(
                table_name,
                con=con_str,
                schema="CI",
                index=True,
                if_exists="replace",
            )

        dist_df = _get_dist_arg(df)
        if rank_skew and bodo.get_rank() != 0:
            dist_df = dist_df[:0]
        bodo.jit(f, distributed=["df"])(dist_df, table_name)
        write_complete = True

        def read(table_name):
            return pd.read_sql_table(
                table_name,
                con=con_str,
                schema="CI",
            )

        check_func(
            read,
            (table_name,),
            py_output=bodo.allgatherv(dist_df),
            sort_output=True,
            reset_index=True,
        )
    finally:
        try:
            run_rank0(
                lambda: conn_str_to_catalog(con_str).purge_table(f"CI.{table_name}")
            )()
        except Exception:
            assert not write_complete, (
                f"Cleanup failed, {table_name} may need manual cleanup"
            )


def test_get_table_len(polaris_connection, memory_leak_check):
    """
    Test getting the length of an Iceberg table in a Polaris REST catalog.
    """
    from bodo.io.iceberg.catalog import conn_str_to_catalog
    from bodo.utils.utils import run_rank0

    rest_uri, polaris_warehouse, polaris_credential = polaris_connection
    con_str = get_rest_catalog_connection_string(
        rest_uri, polaris_warehouse, polaris_credential
    )
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    namespace = "CI"
    table_name = "get_table_len_test"
    table_id = f"{namespace}.{table_name}"
    try:
        py_catalog: Catalog = conn_str_to_catalog(con_str)
        run_rank0(
            lambda: py_catalog.create_table(table_id, pa.Schema.from_pandas(df)).append(
                pa.Table.from_pandas(df)
            )
        )()

        assert bodo.io.iceberg.read_metadata.get_table_length(
            py_catalog.load_table(table_id)
        ) == len(df)
    finally:
        run_rank0(lambda: py_catalog.purge_table(table_id))()
