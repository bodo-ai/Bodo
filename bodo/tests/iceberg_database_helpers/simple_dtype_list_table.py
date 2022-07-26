from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="simple_dtype_list_table", spark=None):
    if spark is None:
        spark = get_spark()

    df, sql_schema, spark_schema = TABLE_MAP["simple_dtype_list_table"]

    create_iceberg_table(df, sql_schema, spark_schema, table_name, spark)


if __name__ == "__main__":
    create_table()
