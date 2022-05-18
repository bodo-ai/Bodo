import datetime
import os

import numpy as np
import pandas as pd

import bodoicebergconnector.bodo_apis.parquet_info
import bodoicebergconnector.bodo_apis.schema

iceberg_poc_dir = os.path.dirname(os.path.abspath(__file__))
WAREHOUSE_LOC = iceberg_poc_dir + "/../test-dataset-creation/"
DB_NAME = "iceberg_db"
TABLE_NAME = "simple_bool_binary_table"

bodo_typing_schema_info = (
    bodoicebergconnector.bodo_apis.schema.get_bodo_connector_typing_schema(
        WAREHOUSE_LOC, DB_NAME, TABLE_NAME
    )
)
print(f"bodo_typing_schema:\n {bodo_typing_schema_info}")

bodo_runtime_schema_info = (
    bodoicebergconnector.bodo_apis.schema.get_bodo_connector_runtime_schema(
        WAREHOUSE_LOC, DB_NAME, TABLE_NAME
    )
)
print(f"bodo_runtime_schema:\n {bodo_runtime_schema_info}")

if TABLE_NAME == "partitions_dt_table":
    date_lower = datetime.date(2018, 12, 31)
    date_upper = datetime.date(2020, 1, 1)
    expr1 = [[("A", ">", date_lower), ("A", "<", date_upper)]]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr1
        )
    )
    print(f"bodo_parquet_info1:\n {bodo_parquet_infos}")

    expr2 = [
        [("B", ">=", np.int64(67)), ("B", "<=", np.int32(102))],
        [("B", "==", 987)],
    ]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr2
        )
    )
    print(f"bodo_parquet_info2:\n {bodo_parquet_infos}")

    # This is an example filter that is not a partition. This should at least "work"
    # because partitions should be hidden
    expr3 = [[("C", "!=", "Bodo.ai")]]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr3
        )
    )
    print(f"bodo_parquet_info3:\n {bodo_parquet_infos}")
elif TABLE_NAME == "simple_numeric_table":
    # This doesn't contain any partitions, but passing an exrpession with
    # numeric types must be supported.
    # TODO: Support decimal
    expr = [
        [("C", ">", np.float32(0.0))],
        [("D", "<", np.float64(0.0)), ("D", "!=", float(1.1))],
    ]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr
        )
    )
    print(f"bodo_parquet_info:\n {bodo_parquet_infos}")
elif TABLE_NAME == "simple_dt_tsz_table":
    # This doesn't contain any partitions, but passing an exrpession with
    # different Timestamp representations must be supported.
    expr1 = [
        [
            ("C", "<", pd.Timestamp(year=2021, month=4, day=15)),
            ("C", ">", pd.Timestamp(year=2018, month=4, day=3).to_numpy()),
        ]
    ]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr1
        )
    )
    print(f"bodo_parquet_info:\n {bodo_parquet_infos}")
    # Seems like Iceberg may also be using metadata to know it can skip full files
    # even without partitions. All data is from 2019 and it returns 0 files.
    expr2 = [
        [
            ("C", "<", pd.Timestamp(year=2021, month=4, day=15)),
            ("C", ">", pd.Timestamp(year=2021, month=4, day=3).to_numpy()),
        ]
    ]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr2
        )
    )
    print(f"bodo_parquet_info:\n {bodo_parquet_infos}")
elif TABLE_NAME == "simple_bool_binary_table":
    # This doesn't contain any partitions, but passing an exrpession with
    # booleans must be supported.
    expr = [[("A", "==", False), ("B", "!=", np.bool_(True))]]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, expr
        )
    )
    print(f"bodo_parquet_info:\n {bodo_parquet_infos}")
    # bytes isn't supported yet, but this should still resturn all files.
    bad_expr = [[("C", "==", b"str")]]
    bodo_parquet_infos = (
        bodoicebergconnector.bodo_apis.parquet_info.bodo_connector_get_parquet_info(
            WAREHOUSE_LOC, DB_NAME, TABLE_NAME, bad_expr
        )
    )
    print(f"bodo_parquet_info:\n {bodo_parquet_infos}")
