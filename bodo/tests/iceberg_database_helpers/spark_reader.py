import sys

from bodo.tests.iceberg_database_helpers.utils import get_spark


def read_iceberg_table(table_name, database_name, spark=None):
    if not spark:
        spark = get_spark()

    # This is so when reading timestamp columns, the output will
    # match that of Bodo.
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = spark.table(f"hadoop_prod.{database_name}.{table_name}")
    count = df.count()
    spark_schema = df.schema
    pd_df = df.toPandas()
    return pd_df, count, spark_schema


if __name__ == "__main__":
    table_name = sys.argv[1]
    database_name = sys.argv[2]
    pd_df, count, spark_schema = read_iceberg_table(table_name, database_name)
    print(f"Count: {count}")
    print(f"Schema:\n{spark_schema}")
    print(f"Dataframe:\n{pd_df}")
