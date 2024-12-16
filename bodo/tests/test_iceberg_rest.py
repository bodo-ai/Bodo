from uuid import uuid4

import bodo_iceberg_connector as bic
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.io.iceberg import get_rest_catalog_config, get_rest_catalog_fs
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    get_rest_catalog_connection_string,
    pytest_tabular,
    temp_env_override,
)
from bodo.utils.utils import run_rank0

pytestmark = pytest_tabular


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
    ).removeprefix("iceberg+")
    _, token, _ = get_rest_catalog_config(con_str)

    @bodo.jit
    def f():
        return get_rest_catalog_fs(
            rest_uri, token, tabular_warehouse, "examples", "nyc_taxi_locations"
        )

    assert f().region == "us-east-1"


def test_iceberg_tabular_read_credential_refresh(
    tabular_connection, memory_leak_check, capfd
):
    """
    Test reading an Iceberg table from a Tabular REST catalog. Sets credentials to be refreshed every request and confirms appropriate logs are present.
    """
    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
    with temp_env_override(
        {
            "DEFAULT_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER_TIMEOUT": "0",
            "DEBUG_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER": "1",
        }
    ):
        try:

            @bodo.jit
            def f():
                df = pd.read_sql_table(
                    "nyc_taxi_locations",
                    con=f"iceberg+{rest_uri.replace('https://', 'REST://')}?warehouse={tabular_warehouse}&credential={tabular_credential}",
                    schema="examples",
                )
                return df

            f()
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
    df, tabular_connection, memory_leak_check, rank_skew
):
    """
    Test writing to an Iceberg table in a Tabular REST catalog.
    Checksum is used to verify the data is written correctly.
    """
    rest_uri, tabular_warehouse, tabular_credential = tabular_connection
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
        delete_succeeded = run_rank0(bic.delete_table)(
            bodo.io.iceberg.format_iceberg_conn(con_str),
            "CI",
            table_name,
        )
        assert (
            not write_complete or delete_succeeded
        ), f"Cleanup failed, {table_name} may need manual cleanup"
