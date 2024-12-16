import datetime

import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "MOCK_HOLIDAY_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.DataFrame(
        {
            "DAY": [
                datetime.date(2021, 2, 14),
                datetime.date(2021, 7, 4),
                datetime.date(2021, 12, 25),
                datetime.date(2021, 10, 31),
                datetime.date(2021, 6, 19),
                datetime.date(2021, 9, 6),
                datetime.date(2021, 5, 31),
                datetime.date(2021, 11, 25),
            ],
            "NAME": [
                "VALENTINE'S DAY",
                "INDEPENDENCE DAY",
                "CHRISTMAS",
                "HALLOWEEN",
                "JUNETEENTH",
                "LABOR DAY",
                "MEMORIAL DAY",
                "THANKSGIVING",
            ],
        }
    )
    sql_schema = [
        ("DAY", "date", True),
        ("NAME", "string", True),
    ]
    create_iceberg_table(
        df,
        sql_schema,
        table_name,
        spark,
    )


if __name__ == "__main__":
    create_table()
