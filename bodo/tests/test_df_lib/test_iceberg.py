from __future__ import annotations

import datetime
import operator
import os
import random
import tempfile

import numba.core.utils  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pyiceberg.catalog
import pyiceberg.expressions
import pytest
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table.sorting import SortField, SortOrder
from pyiceberg.transforms import IdentityTransform

import bodo.pandas as bpd
from bodo.tests.iceberg_database_helpers import pyiceberg_reader
from bodo.tests.utils import _test_equal

pytestmark = [pytest.mark.iceberg]


@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "SIMPLE_MAP_TABLE",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "SIMPLE_STRING_TABLE",
        "PARTITIONS_DT_TABLE",
        "SIMPLE_DT_TSZ_TABLE",
        "SIMPLE_DECIMALS_TABLE",
    ],
)
def test_simple_table_read(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    bodo_out = bpd.read_iceberg(f"{db_schema}.{table_name}", location=warehouse_loc)
    _test_equal(
        bodo_out,
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_STRING_TABLE",
    ],
)
def test_table_read_limit(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        limit=10,
    )
    assert bodo_out.is_lazy_plan()

    # Check that the plan has been optimized to a single read
    pre, post = bpd.plan.getPlanStatistics(bodo_out._plan)
    assert pre == 3
    assert post == 2

    _test_equal(
        bodo_out,
        py_out.iloc[:10],
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_STRING_TABLE",
    ],
)
def test_table_read_head(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(
        table_name, db_schema
    ).head(10)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    ).head(10)
    assert bodo_out.is_lazy_plan()

    # Check that the plan has been optimized to a single read
    pre, post = bpd.plan.getPlanStatistics(bodo_out._plan)
    assert pre == 3
    assert post == 2

    _test_equal(
        bodo_out,
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_STRING_TABLE",
    ],
)
def test_table_read_selected_fields(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        selected_fields=("A", "C"),
    )

    _test_equal(
        bodo_out,
        py_out[["A", "C"]],
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_STRING_TABLE",
    ],
)
def test_table_read_select_columns(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)[
        ["A", "C"]
    ]
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )[["A", "C"]]

    _test_equal(
        bodo_out,
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
def test_table_read_row_filter(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    from bodo.io.iceberg.catalog.dir import DirCatalog

    db_schema, warehouse_loc = iceberg_database(table_name)
    catalog = DirCatalog(
        None,
        **{
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    table = catalog.load_table(f"{db_schema}.{table_name}")
    filter_expr = pyiceberg.expressions.And(
        pyiceberg.expressions.LessThan("A", 5),
        pyiceberg.expressions.Not(pyiceberg.expressions.GreaterThanOrEqual("C", 10)),
    )
    py_out = table.scan(row_filter=filter_expr).to_pandas()

    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        row_filter=filter_expr,
    )

    _test_equal(
        bodo_out,
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
def test_table_read_time_travel(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    from bodo.io.iceberg.catalog.dir import DirCatalog

    db_schema, warehouse_loc = iceberg_database(table_name)
    catalog = DirCatalog(
        None,
        **{
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )

    table = catalog.load_table(f"{db_schema}.{table_name}")
    # Get the current snapshot ID
    snapshot_id = table.current_snapshot().snapshot_id
    # Read the table at the current snapshot to get the initial state
    py_out_orig = table.scan().to_pandas()

    # Append to the table to create a new snapshot
    table.append(
        pa.Table.from_pydict(
            {
                "A": [10],
                "B": [15],
                "C": [20],
                "D": [25],
                "E": [30],
                "F": [36],
            },
            schema=table.schema().as_arrow(),
        )
    )

    # Read the table at the previous snapshot ID
    bodo_out_orig = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        snapshot_id=snapshot_id,
    )

    _test_equal(
        bodo_out_orig,
        py_out_orig,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    # Read the table at the current snapshot to get the new state
    py_out_new = table.scan().to_pandas()
    bodo_out_new = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    _test_equal(
        bodo_out_new,
        py_out_new,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_table_read_filter_pushdown(
    table_name,
    op,
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    bodo_out2 = bodo_out[eval(f"bodo_out.A {op_str} 3")]
    assert bodo_out2.is_lazy_plan()

    pre, post = bpd.plan.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 3
    assert post == 2

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[eval(f"py_out.A {op_str} 3")]

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
def test_table_read_filter_pushdown_multiple(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[(bodo_out.A < 5) & (bodo_out.C >= 3) & (bodo_out.A != 3)]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.plan.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 3
    assert post == 2
    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[(py_out.A < 5) & (py_out.C >= 3) & (py_out.A != 3)]

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
def test_table_read_filter_pushdown_and_row_filter(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        row_filter="A < 3",
    )

    bodo_out2 = bodo_out[bodo_out.C >= 3]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.plan.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 3
    assert post == 2

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[(py_out.C >= 3) & (py_out.A < 3)]

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


# Need to test on schema evolved table
@pytest.mark.parametrize(
    "table_name",
    ["ADVERSARIAL_SCHEMA_EVOLUTION_TABLE"],
)
def test_table_read_schema_evolved_filter_pushdown(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[bodo_out.B < 4]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.plan.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 3
    assert post == 2

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[py_out.B < 4]

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "PARTITIONS_DT_TABLE",
    ],
)
def test_table_read_partitioned_file_pruning(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[bodo_out.A <= pd.Timestamp("2018-12-12").date()]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.plan.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 3
    assert post == 2

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[py_out.A <= datetime.date(2018, 12, 12)]
    # TODO: Figure out how to test that not all files are read automatically.
    # It was manually confirmed when the test was written.

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_write():
    """Simple test for writing a DataFrame to Iceberg."""
    # TODO[BSE-4883]: verify and improve this test when Iceberg CI is enabled
    df = pd.DataFrame(
        {
            "one": [-1.0, np.nan, 2.5, 3.0, 4.0, 6.0, 10.0],
            "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
            "three": [True, False, True, True, True, False, False],
            "four": [-1.0, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],
            "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],
        }
    )

    bdf = bpd.from_pandas(df)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "iceberg_warehouse")

        part_spec = PartitionSpec(
            PartitionField(2, 1001, IdentityTransform(), "id_part")
        )
        sort_order = SortOrder(SortField(source_id=4, transform=IdentityTransform()))
        bdf.to_iceberg(
            "test_table",
            location=path,
            partition_spec=part_spec,
            sort_order=sort_order,
            properties={"p_a1": "pvalue_a1"},
            snapshot_properties={"p_key": "p_value"},
        )
        assert bdf.is_lazy_plan()

        # Read using PyIceberg to verify the write
        out_df = pyiceberg_reader.read_iceberg_table("test_table", path)

        _test_equal(
            out_df,
            df,
            check_pandas_types=False,
            sort_output=True,
            reset_index=True,
        )

        # Check that the snapshot properties are set correctly
        catalog = pyiceberg.catalog.load_catalog(
            None,
            **{
                pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
                pyiceberg.catalog.WAREHOUSE_LOCATION: path,
            },
        )
        table = catalog.load_table("test_table")
        assert table.properties.get("p_a1") == "pvalue_a1"
        snapshot = table.current_snapshot()
        assert snapshot.summary.get("p_key") == "p_value"


def test_read_s3_tables_location():
    from bodo.io.iceberg.catalog.s3_tables import (
        S3TABLES_REGION,
        S3TABLES_TABLE_BUCKET_ARN,
        S3TablesCatalog,
    )

    location = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
    region = "us-east-2"
    catalog_properties = {
        S3TABLES_TABLE_BUCKET_ARN: location,
        S3TABLES_REGION: region,
    }
    catalog = S3TablesCatalog(None, **catalog_properties)
    pdf = catalog.load_table("sf1000.nation").scan().to_pandas()
    bdf = bpd.read_iceberg("sf1000.nation", location=location)
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_read_s3_tables_read_iceberg_table():
    from bodo.io.iceberg.catalog.s3_tables import (
        S3TABLES_REGION,
        S3TABLES_TABLE_BUCKET_ARN,
        S3TablesCatalog,
    )

    location = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
    region = "us-east-2"
    catalog_properties = {
        S3TABLES_TABLE_BUCKET_ARN: location,
        S3TABLES_REGION: region,
    }
    catalog = S3TablesCatalog(None, **catalog_properties)
    table = catalog.load_table("sf1000.nation")

    pdf = table.scan().to_pandas()
    bdf = bpd.read_iceberg_table(table)

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_write_s3_tables_location():
    from bodo.io.iceberg.catalog.s3_tables import (
        S3TABLES_REGION,
        S3TABLES_TABLE_BUCKET_ARN,
        S3TablesCatalog,
    )

    location = "arn:aws:s3tables:us-east-2:427443013497:bucket/unittest-bucket"
    region = "us-east-2"
    catalog_properties = {
        S3TABLES_TABLE_BUCKET_ARN: location,
        S3TABLES_REGION: region,
    }
    catalog = S3TablesCatalog(None, **catalog_properties)
    df = pd.DataFrame(
        {
            "one": [-1.0, np.nan, 2.5, 3.0, 4.0, 6.0, 10.0],
            "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
            "three": [True, False, True, True, True, False, False],
            "four": [-1.0, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],
            "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],
        }
    )
    bdf = bpd.from_pandas(df)
    rand_str = random.randint(100000, 999999)
    table_id = f"write_namespace.bodoicebergwritetest{rand_str}"
    bdf.to_iceberg(table_id, location=location)
    try:
        # Read using PyIceberg to verify the write
        out_df = catalog.load_table(table_id).scan().to_pandas()
        _test_equal(
            out_df,
            df,
            check_pandas_types=False,
            sort_output=True,
            reset_index=True,
        )
    finally:
        # Clean up the table after the test
        catalog.purge_table(table_id)


def test_read_iceberg_rename():
    from bodo.io.iceberg.catalog.s3_tables import (
        S3TABLES_REGION,
        S3TABLES_TABLE_BUCKET_ARN,
        S3TablesCatalog,
    )

    location = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
    region = "us-east-2"
    catalog_properties = {
        S3TABLES_TABLE_BUCKET_ARN: location,
        S3TABLES_REGION: region,
    }
    catalog = S3TablesCatalog(None, **catalog_properties)
    table = catalog.load_table("sf1000.nation")

    pdf = table.scan().to_pandas()
    pdf = pdf.rename(
        columns={
            "N_NATIONKEY": "NATIONKEY",
            "N_NAME": "NAME",
            "N_REGIONKEY": "N_REGIONKEY",
            "N_COMMENT": "COMMENT",
        }
    )
    bdf = bpd.read_iceberg_table(table)
    bdf = bdf.rename(
        columns={
            "N_NATIONKEY": "NATIONKEY",
            "N_NAME": "NAME",
            "N_REGIONKEY": "N_REGIONKEY",
            "N_COMMENT": "COMMENT",
        }
    )

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_join_filter(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "SIMPLE_NUMERIC_TABLE"
    part_table_name = "part_NUMERIC_TABLE_E_identity"
    table_names = [table_name, part_table_name]
    db_schema, warehouse_loc = iceberg_database(table_names)
    df1 = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    df2 = pyiceberg_reader.read_iceberg_table_single_rank(part_table_name, db_schema)
    df3 = df1.merge(df2, left_on="A", right_on="E")

    bdf1 = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bdf2 = bpd.read_iceberg(
        f"{db_schema}.{part_table_name}",
        None,
        catalog_properties={
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bdf3 = bdf1.merge(bdf2, left_on="A", right_on="E")
    # TODO: Figure out how to test that no files are read from the partition table automatically.
    # (A and E have no matching values)
    # It was manually confirmed when the test was written.

    _test_equal(
        bdf3,
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )
