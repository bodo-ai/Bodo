from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.fs as pa_fs
import pytest
from pyiceberg.catalog.rest import RestCatalog

import bodo
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.io.iceberg.common import _fs_from_file_path
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    get_rest_catalog_connection_string,
    pytest_polaris,
    temp_env_override,
)
from bodo.utils.utils import run_rank0

pytestmark = pytest_polaris


def test_iceberg_tabular_read(tabular_connection, memory_leak_check):
    """
    Test reading an Iceberg table from a Tabular REST catalog.
    Checksum is used to verify the data is read correctly.
    Column names are used to verify the schema is read correctly.
    """

    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
    con_str = get_rest_catalog_connection_string(
        rest_uri, tabular_warehouse, tabular_credential
    )

    def f():
        df = pd.read_sql_table(
            "nyc_taxi_locations",
            con=con_str,
            schema="examples",
        )
        checksum = df["location_id"].sum()
        return checksum, len(df), list(df.columns)

    check_func(f, (), py_output=(35245, 265, ["location_id", "borough", "zone_name"]))


def test_iceberg_tabular_read_region_detection(tabular_connection, memory_leak_check):
    """
    Creates an s3 fs instance and checks that the region is detected correctly.
    """
    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
    con_str = get_rest_catalog_connection_string(
        rest_uri, tabular_warehouse, tabular_credential
    )
    catalog = conn_str_to_catalog(con_str)
    fs = _fs_from_file_path(tabular_connection, catalog._load_file_io())
    assert isinstance(fs, pa_fs.S3FileSystem)
    assert fs.region == "us-east-1"


def test_iceberg_tabular_read_credential_refresh(
    tabular_connection, memory_leak_check, capfd
):
    """
    Test reading an Iceberg table from a Tabular REST catalog. Sets credentials to be refreshed every request and confirms appropriate logs are present.
    """
    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
    conn_str = get_rest_catalog_connection_string(
        rest_uri, tabular_warehouse, tabular_credential
    )
    with temp_env_override(
        {
            "DEFAULT_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER_TIMEOUT": "0",
            "DEBUG_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER": "1",
        }
    ):
        try:

            @bodo.jit
            def f(conn_str):
                df = pd.read_sql_table(
                    "nyc_taxi_locations",
                    con=conn_str,
                    schema="examples",
                )
                return df

            f(conn_str)
        except Exception:
            out, err = capfd.readouterr()
            with capfd.disabled():
                print(f"STDOUT:\n{out}")
                print(f"STDERR:\n{err}")
            raise

        out, err = capfd.readouterr()
        assert "[DEBUG] Reloading AWS Credentials" in err


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
def test_iceberg_tabular_write_basic(
    df, polaris_connection, memory_leak_check, rank_skew
):
    """
    Test writing to an Iceberg table in a Tabular REST catalog.
    Checksum is used to verify the data is written correctly.
    """
    rest_uri, tabular_warehouse, tabular_credential = polaris_connection
    table_uuid = run_rank0(uuid4)()
    table_name = f"bodo_write_test_{table_uuid}"

    write_complete = False
    try:
        con_str = get_rest_catalog_connection_string(
            rest_uri, tabular_warehouse, tabular_credential
        )

        def f(df, table_name):
            df.to_sql(
                table_name,
                con=con_str,
                schema="CI",
                index=True,
                if_exists="replace",
            )

        dist_df = _get_dist_arg(df)
        if rank_skew and bodo.get_rank != 0:
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
                lambda: RestCatalog("rest_catalog", uri=rest_uri).purge_table(
                    f"CI.{table_name}"
                )
            )()
        except Exception:
            assert not write_complete, (
                f"Cleanup failed, {table_name} may need manual cleanup"
            )
