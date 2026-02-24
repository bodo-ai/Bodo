from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from queries.common_utils import (
    check_query_result_pd,
    get_table_path,
    run_query_generic,
)
from settings import Settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

settings = Settings()


def get_or_create_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("spark_queries")
        .master("local[*]")
        .config("spark.driver.memory", settings.run.spark_driver_memory)
        .config("spark.executor.memory", settings.run.spark_executor_memory)
        .config("spark.log.level", settings.run.spark_log_level)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    if settings.dataset_base_dir.startswith(
        "s3://"
    ) or settings.dataset_base_dir.startswith("s3a://"):
        builder = builder.config(
            "spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4"
        )
        if settings.dataset_base_dir.startswith("s3a://"):
            builder = builder.config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
            )

    spark = builder.getOrCreate()

    ps.set_option("compute.default_index_type", "distributed-sequence")

    return spark


def _read_ds(table_name: str) -> DataFrame:
    if settings.run.io_type == "skip":
        # TODO: Persist data in memory before query
        msg = "cannot run PySpark starting from an in-memory representation"
        raise RuntimeError(msg)

    path = get_table_path(table_name)

    if settings.run.io_type == "parquet":
        df = ps.read_parquet(path)
        for c in df.columns:
            if c.endswith("DATE"):
                df[c] = ps.to_datetime(df[c], format="%Y-%m-%d")
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        raise ValueError(msg)

    return df


def get_line_item_ds() -> DataFrame:
    return _read_ds("lineitem")


def get_orders_ds() -> DataFrame:
    return _read_ds("orders")


def get_customer_ds() -> DataFrame:
    return _read_ds("customer")


def get_region_ds() -> DataFrame:
    return _read_ds("region")


def get_nation_ds() -> DataFrame:
    return _read_ds("nation")


def get_supplier_ds() -> DataFrame:
    return _read_ds("supplier")


def get_part_ds() -> DataFrame:
    return _read_ds("part")


def get_part_supp_ds() -> DataFrame:
    return _read_ds("partsupp")


def run_query(query_number: int, query_func) -> None:
    def query_final():
        res = query_func()
        if not isinstance(res, pd.DataFrame):
            res = res.to_pandas()
        return res

    run_query_generic(
        query_final, query_number, "pyspark", query_checker=check_query_result_pd
    )
