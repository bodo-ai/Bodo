"""End to end tests for the query profiler to ensure BodoSQL profiles contain correct op ids"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from bodo.spawn.utils import run_rank0
from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table
from bodo.tests.utils import temp_env_override
from bodosql import BodoSQLContext, FileSystemCatalog, TablePath


@pytest.mark.bodosql_cpp
def test_query_profiler_end_to_end(iceberg_database, tmp_path, datapath):
    """
    Test that the query profiler correctly captures operation IDs for a simple query.
    """

    pq_path = datapath("sample-parquet-data/no_index.pq")
    iceberg_table = "INT_TABLE"

    input_df = pd.DataFrame(
        {
            "ID": np.arange(1000),
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "bigint", True),
            ],
            iceberg_table,
        )

    setup()

    db_schema, warehouse_loc = iceberg_database()
    tables = {"TABLE1": TablePath(pq_path, file_type="parquet")}
    catalog = FileSystemCatalog(warehouse_loc)
    bc = BodoSQLContext(catalog=catalog, tables=tables)

    query = f""" 
        SELECT SUM(C) FROM TABLE1
        JOIN "{db_schema}".{iceberg_table} AS t2
        ON TABLE1.A = t2.ID
        GROUP BY B
    """

    # Expected query plan
    # (some operators like exchange, iceberg filters not shown in output are omitted):
    # BodoPhysicalProject: OpId=11
    #   BodoPhysicalAggregate: OpId=10
    #     BodoPhysicalProject: OpId=9
    #       IcbergRuntimeJoinFilter: OpId=3
    #         IcebergTableScan: OpId=1
    #     BodoPhysicalFilter: OpId=7
    #       PandasTableScan: OpId=5

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": str(tmp_path)}
    ):
        bc.sql(query)

    expected_report = {
        "10001": {"name": "19PhysicalReadIceberg(INT_TABLE)"},
        "10003": {"name": "18PhysicalJoinFilter"},
        "10005": {"name": "19PhysicalReadParquet"},
        "10007": {"name": "14PhysicalFilter"},
        "10008": {"name": "12PhysicalJoin"},
        "10009": {"name": "18PhysicalProjection"},
        "10010": {"name": "17PhysicalAggregate"},
        "10011": {"name": "18PhysicalProjection"},
    }

    for trace_dir in os.listdir(tmp_path):
        trace_path = os.path.join(tmp_path, trace_dir, "query_profile_0.json")
        with open(trace_path) as f:
            query_profile = json.load(f)
            op_reports = query_profile.get("operator_reports", [])
            for (op_id, op_report), (expected_op_id, expected_op_report) in zip(
                op_reports.items(), expected_report.items()
            ):
                assert op_id == expected_op_id
                assert op_report["name"] == expected_op_report["name"]
