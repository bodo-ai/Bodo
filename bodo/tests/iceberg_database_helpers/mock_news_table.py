import datetime

import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "MOCK_NEWS_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.DataFrame(
        {
            "DAY": [datetime.date.fromordinal(737425 + i) for i in range(1000)],
            "EVENT": [f"{chr(i//26 + 65)}{i%100:02}" for i in range(1000)],
        }
    )
    sql_schema = [
        ("DAY", "date", True),
        ("EVENT", "string", True),
    ]
    # The table is partitioned by the year
    create_iceberg_table(
        df,
        sql_schema,
        table_name,
        spark,
        par_spec=[PartitionField("DAY", "year", -1)],
    )


if __name__ == "__main__":
    create_table()
