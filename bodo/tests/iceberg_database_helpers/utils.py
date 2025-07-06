from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import NamedTuple

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


# All spark instances must share the same set of jars - this is because
# `spark.jars.packages` is not a run time option and cannot be modified.
SPARK_JAR_PACKAGES = [
    "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.5.2",
    "software.amazon.awssdk:bundle:2.29.19",
    "software.amazon.awssdk:url-connection-client:2.29.19",
    "org.apache.iceberg:iceberg-azure-bundle:1.7.1",
]


@dataclass(frozen=True)
class SparkIcebergCatalog:
    catalog_name: str
    default_schema: str | None = field(default=None)


@dataclass(frozen=True)
class SparkFilesystemIcebergCatalog(SparkIcebergCatalog):
    path: str = None


@dataclass(frozen=True)
class SparkRestIcebergCatalog(SparkIcebergCatalog):
    uri: str = None
    credential: str = None
    warehouse: str = None


@dataclass(frozen=True)
class SparkAzureIcebergCatalog(SparkRestIcebergCatalog):
    pass


@dataclass(frozen=True)
class SparkAwsIcebergCatalog(SparkRestIcebergCatalog):
    region: str = field(default="us-east-2")


def get_spark_catalog_for_connection(connection: tuple[str, str, str]):
    uri, warehouse, credential = connection
    if "aws" in warehouse:
        return SparkAwsIcebergCatalog(
            catalog_name=warehouse,
            uri=uri,
            warehouse=warehouse,
            credential=credential,
        )
    elif "azure" in warehouse:
        return SparkAzureIcebergCatalog(
            catalog_name=warehouse,
            uri=uri,
            warehouse=warehouse,
            credential=credential,
        )
    else:
        raise ValueError(f"Unknown warehouse: {warehouse}")


# This should probably be wrapped into a class or fixture in the future
spark_catalogs: set[SparkIcebergCatalog] = set()
spark: SparkSession | None = None


def get_spark(
    catalog: SparkIcebergCatalog = SparkFilesystemIcebergCatalog(
        catalog_name="hadoop_prod", path="."
    ),
) -> SparkSession:
    global spark
    global spark_catalogs
    import bodo

    # Only run Spark on one rank to run faster and avoid Spark issues
    if bodo.get_rank() != 0:
        return None

    # TODO[BSE-4883]: enable Iceberg tests after Iceberg 1.10 upgrade
    builder = SparkSession.builder.appName("spark")
    builder.config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))
    builder.config("spark.sql.execution.arrow.enabled", "true")
    builder.config("spark.sql.ansi.enabled", "false")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

    def add_catalog(builder: SparkSession.Builder, catalog: SparkIcebergCatalog):
        if catalog == SparkFilesystemIcebergCatalog():
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}",
                "org.apache.iceberg.spark.SparkCatalog",
            )
            builder.config(f"spark.sql.catalog.{catalog.catalog_name}.type", "hadoop")
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.warehouse", catalog.path
            )
        elif catalog == SparkRestIcebergCatalog():
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}",
                "org.apache.iceberg.spark.SparkCatalog",
            )
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.catalog-impl",
                "org.apache.iceberg.rest.RESTCatalog",
            )
            builder.config(f"spark.sql.catalog.{catalog.catalog_name}.uri", catalog.uri)
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.credential",
                catalog.credential,
            )
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.warehouse",
                catalog.warehouse,
            )
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.scope",
                "PRINCIPAL_ROLE:ALL",
            )
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.X-Iceberg-Access-Delegation",
                "vended-credentials",
            )
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.token-refresh-enabled",
                "true",
            )
        if catalog.default_schema:
            builder.config(
                f"spark.sql.catalog.{catalog.catalog_name}.default-namespace",
                catalog.default_schema,
            )

    def do_get_spark():
        builder = SparkSession.builder.appName("spark")
        builder.config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))
        # NOTE: This is deprecated, but some Iceberg tests with nested data types
        # don't work without it. Let's try to avoid using Spark for those tests
        # as much as possible.
        builder.config("spark.sql.execution.arrow.enabled", "true")
        builder.config("spark.sql.ansi.enabled", "false")
        builder.config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )

        for catalog in spark_catalogs:
            add_catalog(builder, catalog)

        spark = builder.getOrCreate()

        # Spark throws a WARNING with a very long stacktrace whenever creating am
        # Iceberg table with Hadoop because it is initially unable to determine that
        # it wrote a `version-hint.text` file, even though it does.
        # Setting the Log Level to "ERROR" hides it
        spark.sparkContext.setLogLevel("ERROR")

        return spark

    if catalog not in spark_catalogs:
        # Clear the spark instance and reinitialize
        # we can't add a new catalog to an existing spark instance
        if spark is not None:
            spark.stop()
        spark_catalogs.add(catalog)

        try:
            spark = do_get_spark()
        except Exception:
            # Clear cache and try again - note that this is only for use in CI.
            # Sometimes packages fail to download - if this happens to you locally,
            # clear your cache manually. The path is in the logs.
            shutil.rmtree("/root/.ivy2", ignore_errors=True)
            shutil.rmtree("/root/.m2/repository", ignore_errors=True)
            spark = do_get_spark()
    spark.catalog.setCurrentCatalog(catalog.catalog_name)
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
    df: pd.DataFrame, sql_schema, table_name: str, spark: SparkSession | None
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
    df: pd.DataFrame | None,
    sql_schema: list[tuple[str, str, bool]],
    table_name: str,
    spark: SparkSession | None = None,
    par_spec: list[PartitionField] | None = None,
    sort_order: list[SortField] | None = None,
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
            WRITE ORDERED BY {", ".join(sort_defs)}
        """
        )

    if df is not None:
        append_to_iceberg_table(df, sql_schema, table_name, spark)

    return table_name
