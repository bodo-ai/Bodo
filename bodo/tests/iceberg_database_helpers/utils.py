import shutil
from typing import List, NamedTuple, Optional

import pandas as pd
import pyspark
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


# All spark instances must share the same set of jars - this is because
# `spark.jars.packages` is not a run time option and cannot be modified.
SPARK_JAR_PACKAGES = [
    "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.5.2",
    "software.amazon.awssdk:bundle:2.19.13",
    "software.amazon.awssdk:url-connection-client:2.19.13",
]


def reset_spark():
    """Stop the running spark session if there is one. This allows a new session
    to be made with a different config."""
    active = SparkSession.getActiveSession()
    if active:
        # Before we muck with internal state, we need to make sure there's no
        # active sessions.
        active.stop()

    # TODO(aneesh) Overtime, the spark instances will leak memory. There doesn't
    # seem to be a work around without stopping the jvm process and letting the
    # library create a new one. Eventually we should move to spawning spark
    # servers (one per config) that live for the duration of all tests.
    if pyspark.SparkContext._gateway:
        pyspark.SparkContext._gateway.proc.kill()
        pyspark.SparkContext._gateway = None
        pyspark.SparkContext._jvm = None


def get_spark(path: str = ".") -> SparkSession:
    def do_get_spark():
        reset_spark()
        spark = (
            SparkSession.builder.appName("spark_filesystem")
            .config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))
            .config(
                "spark.sql.catalog.hadoop_prod", "org.apache.iceberg.spark.SparkCatalog"
            )
            .config("spark.sql.catalog.hadoop_prod.type", "hadoop")
            .config("spark.sql.catalog.hadoop_prod.warehouse", path)
            .config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            .config("spark.sql.session.timeZone", "UTC")
            # https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas
            .config("spark.sql.execution.arrow.enabled", "true")
            .getOrCreate()
        )
        # Spark throws a WARNING with a very long stacktrace whenever creating am
        # Iceberg table with Hadoop because it is initially unable to determine that
        # it wrote a `version-hint.text` file, even though it does.
        # Setting the Log Level to "ERROR" hides it
        spark.sparkContext.setLogLevel("ERROR")

        return spark

    try:
        return do_get_spark()
    except Exception:
        # Clear cache and try again - note that this is only for use in CI
        shutil.rmtree("/root/.ivy2", ignore_errors=True)
        shutil.rmtree("/root/.m2/repository", ignore_errors=True)
    return do_get_spark()


def get_spark_tabular(tabular_connection):
    reset_spark()
    rest_uri, tabular_warehouse, tabular_credential = tabular_connection

    spark = (
        SparkSession.builder.appName("spark_tabular")
        .config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))
        .config("spark.sql.catalog.rest_prod", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.rest_prod.catalog-impl",
            "org.apache.iceberg.rest.RESTCatalog",
        )
        .config("spark.sql.catalog.rest_prod.uri", rest_uri)
        .config("spark.sql.catalog.rest_prod.credential", tabular_credential)
        .config("spark.sql.catalog.rest_prod.warehouse", tabular_warehouse)
        .config("spark.sql.defaultCatalog", "rest_prod")
        .config(
            "spark.sql.catalog.rest_prod.io-impl", "org.apache.iceberg.aws.s3.S3FileIO"
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.session.timeZone", "UTC")
        # https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas
        .config("spark.sql.execution.arrow.enabled", "true")
        .getOrCreate()
    )

    # Spark throws a WARNING with a very long stacktrace whenever creating am
    # Iceberg table with Hadoop because it is initially unable to determine that
    # it wrote a `version-hint.text` file, even though it does.
    # Setting the Log Level to "ERROR" hides it
    spark.sparkContext.setLogLevel("ERROR")
    spark.sql("use default;")
    return spark


def transform_str(col_name: str, transform: str, val: int) -> str:
    if transform == "identity":
        return col_name
    elif transform == "truncate" or transform == "bucket":
        return f"{transform}({val}, {col_name})"
    else:
        return f"{transform}({col_name})"


def parse_sql_schema_into_col_defs(sql_schema: list[tuple[str, str, bool]]) -> str:
    """Convert a SQL schema into a string of column definitions
    e.g. [("col1", "int", True), ("col2", "string", False)] -> "col1 int,\n col2 string not null"
    """
    sql_strs = [
        f"{name} {type} {'' if nullable else 'not null'}"
        for (name, type, nullable) in sql_schema
    ]
    sql_col_defs = ",\n".join(sql_strs)
    return sql_col_defs


def parse_sql_schema_into_spark_schema(sql_schema: list[tuple[str, str, bool]]) -> str:
    """Convert a SQL schema into a string of column definitions
    e.g. [("col1", "int", True), ("col2", "string", False)] -> "col1 int, col2 string not null"
    """
    sql_strs = [
        f"{name} {type} {'' if nullable else 'not null'}"
        for (name, type, nullable) in sql_schema
    ]
    spark_schema_str = ", ".join(sql_strs)
    return spark_schema_str


def append_to_iceberg_table(
    df: pd.DataFrame, sql_schema, table_name: str, spark: Optional[SparkSession]
):
    """Append a pandas DataFrame to an existing Iceberg table"""
    spark_schema_str = parse_sql_schema_into_spark_schema(sql_schema)
    if spark is None:
        spark = get_spark()
    df = df.astype("object").where(pd.notnull(df), None)
    for col_info in sql_schema:
        col_name = col_info[0]
        col_type = col_info[1]
        if col_type == "timestamp":
            df[col_name] = pd.to_datetime(df[col_name])

    df = spark.createDataFrame(df, schema=spark_schema_str)  # type: ignore
    df.writeTo(f"hadoop_prod.{DATABASE_NAME}.{table_name}").append()


def create_iceberg_table(
    df: pd.DataFrame,
    sql_schema: list[tuple[str, str, bool]],
    table_name: str,
    spark: Optional[SparkSession] = None,
    par_spec: Optional[List[PartitionField]] = None,
    sort_order: Optional[List[SortField]] = None,
):
    if spark is None:
        spark = get_spark()
    sql_col_defs = parse_sql_schema_into_col_defs(sql_schema)

    # if the table already exists do nothing
    try:
        spark.sql(f"SELECT * FROM hadoop_prod.{DATABASE_NAME}.{table_name} LIMIT 1")
        return None
    except Exception:
        pass

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
    append_to_iceberg_table(df, sql_schema, table_name, spark)

    return table_name
