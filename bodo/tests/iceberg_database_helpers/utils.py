from typing import List, NamedTuple, Optional

import pandas as pd
from pyspark.sql import SparkSession

DATABASE_NAME = "iceberg_db"


class PartitionField(NamedTuple):
    col_name: str
    transform: str
    transform_val: int


class SortField(NamedTuple):
    col_name: str
    transform: str
    transform_val: int
    asc: bool  # Ascending when True, Descending when False
    nulls_first: bool  # First when True, Last when False


def get_spark():
    spark = (
        SparkSession.builder.appName("Iceberg with Spark")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.2_2.12:0.13.0",
        )
        .config(
            "spark.sql.catalog.hadoop_prod", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config("spark.sql.catalog.hadoop_prod.type", "hadoop")
        .config("spark.sql.catalog.hadoop_prod.warehouse", ".")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    # Spark throws a WARNING with a very long stacktrace whenever creating am
    # Iceberg table with Hadoop because it is initially unable to determine that
    # it wrote a `version-hint.text` file, even though it does.
    # Setting the Log Level to "ERROR" hides it
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def transform_str(col_name: str, transform: str, val: int) -> str:
    if transform == "identity":
        return col_name
    elif transform == "truncate" or transform == "bucket":
        return f"{transform}({val}, {col_name})"
    else:
        return f"{transform}({col_name})"


def create_iceberg_table(
    df: pd.DataFrame,
    sql_schema,
    table_name: str,
    spark: Optional[SparkSession] = None,
    par_spec: Optional[List[PartitionField]] = None,
    sort_order: Optional[List[SortField]] = None,
):
    if spark is None:
        spark = get_spark()

    sql_strs = [
        f"{name} {type} {'' if nullable else 'not null'}"
        for (name, type, nullable) in sql_schema
    ]
    sql_col_defs = ",\n".join(sql_strs)
    spark_schema_str = ", ".join(sql_strs)

    if not par_spec:
        partition_str = ""
    else:
        part_defs = []
        for par_field in par_spec:
            col_name, transform, transform_val = par_field
            inner = transform_str(col_name, transform, transform_val)
            part_defs.append(inner)

        partition_str = f"PARTITIONED BY ({', '.join(part_defs)})"

    # Create the table and then add the data to it.
    # We create table using SQL syntax, because DataFrame API
    # doesn't write the nullability in Iceberg metadata correctly.
    spark.sql(
        f""" 
        CREATE TABLE hadoop_prod.{DATABASE_NAME}.{table_name} (
            {sql_col_defs})
        USING iceberg {partition_str}
        TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
    """
    )

    if sort_order:
        sort_defs = []
        for sort_field in sort_order:
            col_name, transform, transform_val, asc, nulls_first = sort_field
            trans_str = transform_str(col_name, transform, transform_val)
            asc_str = "ASC" if asc else "DESC"
            null_str = "FIRST" if nulls_first else "LAST"
            sort_defs.append(f"{trans_str} {asc_str} NULLS {null_str}")
        spark.sql(
            f"""
            ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} 
            WRITE ORDERED BY {', '.join(sort_defs)}
        """
        )

    df = df.astype("object").where(pd.notnull(df), None)
    for col_info in sql_schema:
        col_name = col_info[0]
        col_type = col_info[1]
        if col_type == "timestamp":
            df[col_name] = pd.to_datetime(df[col_name])

    df = spark.createDataFrame(df, schema=spark_schema_str)  # type: ignore
    df.writeTo(f"hadoop_prod.{DATABASE_NAME}.{table_name}").append()
