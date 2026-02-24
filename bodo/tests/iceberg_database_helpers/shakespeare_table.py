import os

import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "SHAKESPEARE_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.read_csv(
        f"{os.path.dirname(__file__)}/shakespeare_sample.csv", dtype_backend="pyarrow"
    ).loc[:, ["PLAY", "ACTSCENELINE", "PLAYER", "PLAYERLINE"]]
    sql_schema = [
        ("PLAY", "string", True),
        ("ACTSCENELINE", "string", True),
        ("PLAYER", "string", True),
        ("PLAYERLINE", "string", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)


if __name__ == "__main__":
    create_table()
