import pandas as pd

from bodo.tests.utils import check_func, pytest_tabular

pytestmark = pytest_tabular


def test_iceberg_tabular_read(tabular_connection, memory_leak_check):
    """
    Test reading an Iceberg table from a Tabular REST catalog.
    Checksum is used to verify the data is read correctly.
    Column names are used to verify the schema is read correctly.
    """

    rest_uri, tabular_warehouse, tabular_credential = tabular_connection

    def f():
        df = pd.read_sql_table(
            "nyc_taxi_locations",
            con=f"iceberg+{rest_uri.replace('https://', 'REST://')}?warehouse={tabular_warehouse}&credential={tabular_credential}",
            schema="examples",
        )
        checksum = df["location_id"].sum()
        return checksum, len(df), list(df.columns)

    check_func(f, (), py_output=(35245, 265, ["location_id", "borough", "zone_name"]))
