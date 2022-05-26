from datetime import datetime

import pandas as pd
import pyspark.sql.types as spark_types
import pytz

from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark


def create_table(table_name="simple_dt_tsz_table", spark=None):

    if spark is None:
        spark = get_spark()

    df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    datetime.strptime("12/11/2018", "%d/%m/%Y"),
                    datetime.strptime("12/11/2019", "%d/%m/%Y"),
                    datetime.strptime("12/12/2018", "%d/%m/%Y"),
                    datetime.strptime("13/11/2018", "%d/%m/%Y"),
                ]
                * 5
            ),
            "B": pd.Series(
                [
                    datetime.strptime("12/11/2018", "%d/%m/%Y"),
                    datetime.strptime("12/11/2019", "%d/%m/%Y"),
                    datetime.strptime("12/12/2018", "%d/%m/%Y"),
                    datetime.strptime("13/11/2018", "%d/%m/%Y"),
                ]
                * 5
            ),
            "C": pd.Series(
                [
                    datetime(2019, 8, 21, 15, 23, 45, 0, pytz.timezone("US/Eastern")),
                    datetime(
                        2019, 8, 21, 15, 23, 45, 0, pytz.timezone("Asia/Calcutta")
                    ),
                ]
                * 10
            ),
        }
    )

    schema = spark_types.StructType(
        [
            spark_types.StructField("A", spark_types.DateType(), True),
            spark_types.StructField("B", spark_types.DateType(), True),
            spark_types.StructField("C", spark_types.TimestampType(), True),
        ]
    )

    create_iceberg_table(df, schema, table_name, spark)


if __name__ == "__main__":
    create_table()
