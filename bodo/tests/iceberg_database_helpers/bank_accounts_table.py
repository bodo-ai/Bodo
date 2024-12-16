import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "BANK_ACCOUNTS_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.DataFrame(
        {
            "ACCTNMBR": [10000 + i for i in range(50000)],
            "BALANCE": [
                float(f"{((i**3)%(10**6))}.{((i**4)%91):02}") for i in range(50000)
            ],
        }
    )
    sql_schema = [
        ("ACCTNMBR", "long", True),
        ("BALANCE", "double", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)


if __name__ == "__main__":
    create_table()
