import os

import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
    get_spark,
)

TABLE_NAME = "ENGLISH_DICTIONARY_TABLE"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()
    df = pd.read_csv(
        f"{os.path.dirname(__file__)}/raw_dict_sample.csv", dtype_backend="pyarrow"
    )
    sql_schema = [
        ("WORD", "string", True),
        ("COUNT", "string", True),
        ("POS", "string", True),
        ("DEFINITION", "string", True),
    ]
    # The table is partitioned by the first letter of each word
    create_iceberg_table(
        df,
        sql_schema,
        table_name,
        spark,
        par_spec=[PartitionField("WORD", "truncate", 1)],
    )


if __name__ == "__main__":
    create_table()
