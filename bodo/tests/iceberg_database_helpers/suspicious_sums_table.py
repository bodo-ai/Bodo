import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "SUSPICIOUS_SUMS_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.DataFrame(
        {
            "BALANCE": [0.0, 375000.81, 488000.09, 78125.42],
            "FLAG": [
                "EMPTY",
                "HACK",
                "CAYMAN",
                "SHELL",
            ],
        }
    )
    sql_schema = [
        ("BALANCE", "double", True),
        ("FLAG", "string", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)


if __name__ == "__main__":
    create_table()
