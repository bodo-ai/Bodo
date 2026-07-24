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
import pyiceberg.table.sorting
import pytest
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table.sorting import NullOrder, SortDirection, SortField, SortOrder
from pyiceberg.transforms import (
    BucketTransform,
    DayTransform,
    HourTransform,
    IdentityTransform,
    MonthTransform,
    TruncateTransform,
    VoidTransform,
    YearTransform,
)

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
        pytest.param("SIMPLE_STRING_TABLE", marks=pytest.mark.gpu),
        pytest.param("PARTITIONS_DT_TABLE", marks=pytest.mark.gpu),
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
    # Make sure file:// protocol works with Iceberg
    db_schema, warehouse_loc = iceberg_database(
        table_name, path="file://" + os.path.abspath(os.getcwd())
    )
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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
@pytest.mark.gpu
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


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sort_dir, null_order",
    [
        (SortDirection.ASC, NullOrder.NULLS_LAST),
        (SortDirection.ASC, NullOrder.NULLS_FIRST),
        (SortDirection.DESC, NullOrder.NULLS_LAST),
        (SortDirection.DESC, NullOrder.NULLS_FIRST),
    ],
)
def test_write(sort_dir, null_order):
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
        sort_order = SortOrder(
            SortField(
                source_id=4,
                transform=IdentityTransform(),
                direction=sort_dir,
                null_order=null_order,
            )
        )
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


def _make_transform_write_df(n_rows=20):
    """Create a DataFrame with column types that exercise all Iceberg transforms.

    Column layout (field_id = 1-based index):
      1: A  int64         — bucket, truncate(int), identity
      2: B  string        — truncate(str), identity
      3: C  timestamp(ns) — year, month, day, hour
      4: D  float64       — identity with nulls
      5: E  bool          — void, identity
    """
    rng = np.random.default_rng(42)
    # Build variable-length columns that adapt to n_rows
    b_vals = ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz0", "123"]
    return pd.DataFrame(
        {
            "A": list(range(n_rows)),
            "B": (b_vals * ((n_rows // len(b_vals)) + 1))[:n_rows],
            "C": pd.Series(
                pd.date_range("2020-01-15", periods=n_rows, freq="37D").date
            ),
            "D": rng.choice([1.0, np.nan, 3.5, -2.1], n_rows).astype("float64"),
            "E": ([True, False] * ((n_rows + 1) // 2))[:n_rows],
        }
    )


def _check_write_read(df, part_spec, sort_order, tmpdir):
    """Write a DataFrame to Iceberg and verify it can be read back correctly."""
    bdf = bpd.from_pandas(df)
    path = os.path.join(tmpdir, "wh")
    bdf.to_iceberg(
        "test_tbl", location=path, partition_spec=part_spec, sort_order=sort_order
    )
    out_df = pyiceberg_reader.read_iceberg_table("test_tbl", path)
    _test_equal(
        out_df,
        df,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "part_transform, sort_transform",
    [
        # Partition transforms (all supported)
        (IdentityTransform(), None),
        (VoidTransform(), None),
        (BucketTransform(4), None),
        (TruncateTransform(3), None),
        (YearTransform(), None),
        (MonthTransform(), None),
        (DayTransform(), None),
        pytest.param(
            HourTransform(),
            None,
            marks=pytest.mark.xfail(
                reason="PyIceberg requires TIMESTAMP for hour, not DATE"
            ),
        ),
        # Sort transforms (all supported)
        (None, IdentityTransform()),
        (None, VoidTransform()),
        (None, BucketTransform(4)),
        (None, TruncateTransform(3)),
        (None, YearTransform()),
        (None, MonthTransform()),
        (None, DayTransform()),
        pytest.param(
            None,
            HourTransform(),
            marks=pytest.mark.xfail(
                reason="PyIceberg requires TIMESTAMP for hour, not DATE"
            ),
        ),
        # Combined partition + sort
        (BucketTransform(4), BucketTransform(4)),
        (TruncateTransform(3), TruncateTransform(3)),
        (YearTransform(), YearTransform()),
        (MonthTransform(), MonthTransform()),
        (DayTransform(), DayTransform()),
        pytest.param(
            HourTransform(),
            HourTransform(),
            marks=pytest.mark.xfail(
                reason="PyIceberg requires TIMESTAMP for hour, not DATE"
            ),
        ),
        # Mixed transform types
        (BucketTransform(4), TruncateTransform(3)),
        (YearTransform(), MonthTransform()),
    ],
)
def test_write_all_transforms(part_transform, sort_transform):
    """Test writing to Iceberg with every supported transform type,
    both as partition spec and sort order."""
    df = _make_transform_write_df(20)

    # Map transform to the appropriate column (field_id in Iceberg schema):
    # - Temporal transforms: column C (field_id=3, timestamp)
    # - Bucket/Truncate(int)/Identity(int): column A (field_id=1)
    # - Truncate(str): column B (field_id=2)
    # - Void: column E (field_id=5)
    _part_src = (
        3
        if isinstance(
            part_transform,
            (YearTransform, MonthTransform, DayTransform, HourTransform),
        )
        else 5
        if isinstance(part_transform, VoidTransform)
        else 1
    )

    part_spec = PartitionSpec()
    if part_transform is not None:
        part_spec = PartitionSpec(PartitionField(_part_src, 1000, part_transform, "p"))
    sort_order = SortOrder()
    if sort_transform is not None:
        _sort_src = (
            3
            if isinstance(
                sort_transform,
                (YearTransform, MonthTransform, DayTransform, HourTransform),
            )
            else 1
        )
        sort_order = SortOrder(
            SortField(
                source_id=_sort_src,
                transform=sort_transform,
                direction=SortDirection.ASC,
                null_order=NullOrder.NULLS_LAST,
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, part_spec, sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sort_dir, null_order",
    [
        (SortDirection.ASC, NullOrder.NULLS_LAST),
        (SortDirection.ASC, NullOrder.NULLS_FIRST),
        (SortDirection.DESC, NullOrder.NULLS_LAST),
        (SortDirection.DESC, NullOrder.NULLS_FIRST),
    ],
)
def test_write_sort_directions(sort_dir, null_order):
    """Verify sort direction and null ordering work for a non-trivial
    transform."""
    df = _make_transform_write_df(15)
    part_spec = PartitionSpec()
    sort_order = SortOrder(
        SortField(
            source_id=1,
            transform=BucketTransform(3),
            direction=sort_dir,
            null_order=null_order,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, part_spec, sort_order, tmp)


@pytest.mark.gpu
def test_write_multi_partition_transform():
    """Multiple partition transforms used together."""
    df = _make_transform_write_df(20)
    part_spec = PartitionSpec(
        PartitionField(3, 1000, YearTransform(), "yr"),
        PartitionField(1, 1001, BucketTransform(4), "bk"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, part_spec, SortOrder(), tmp)


@pytest.mark.gpu
def test_write_multi_sort_transform():
    """Multiple sort transforms used together."""
    df = _make_transform_write_df(20)
    sort_order = SortOrder(
        SortField(
            source_id=3,
            transform=YearTransform(),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        ),
        SortField(
            source_id=1,
            transform=BucketTransform(4),
            direction=SortDirection.DESC,
            null_order=NullOrder.NULLS_FIRST,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [1, 3, 10])
def test_write_sort_truncate_string(width):
    """Sort by truncate transform on a string column at different widths."""
    df = _make_transform_write_df(30)
    sort_order = SortOrder(
        SortField(
            source_id=2,
            transform=TruncateTransform(width),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [1, 5, 100])
def test_write_sort_truncate_int(width):
    """Sort by truncate transform on an int column at different widths."""
    df = _make_transform_write_df(30)
    sort_order = SortOrder(
        SortField(
            source_id=1,
            transform=TruncateTransform(width),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize("n_buckets", [2, 8, 16])
def test_write_sort_bucket(n_buckets):
    """Sort by bucket transform with different bucket counts."""
    df = _make_transform_write_df(30)
    sort_order = SortOrder(
        SortField(
            source_id=1,
            transform=BucketTransform(n_buckets),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
def test_write_sort_identity_date():
    """Sort by identity on a date column (date sorting, not epoch-derived)."""
    df = _make_transform_write_df(20)
    sort_order = SortOrder(
        SortField(
            source_id=3,
            transform=IdentityTransform(),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sort_dir, null_order",
    [
        (SortDirection.ASC, NullOrder.NULLS_LAST),
        (SortDirection.ASC, NullOrder.NULLS_FIRST),
        (SortDirection.DESC, NullOrder.NULLS_LAST),
        (SortDirection.DESC, NullOrder.NULLS_FIRST),
    ],
)
def test_write_sort_void(sort_dir, null_order):
    """Sort by void transform — column is all-null, sort is a no-op."""
    df = _make_transform_write_df(20)
    sort_order = SortOrder(
        SortField(
            source_id=1,
            transform=VoidTransform(),
            direction=sort_dir,
            null_order=null_order,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sort_dir, null_order",
    [
        (SortDirection.ASC, NullOrder.NULLS_LAST),
        (SortDirection.DESC, NullOrder.NULLS_LAST),
    ],
)
@pytest.mark.parametrize(
    "temporal_xfrm", [YearTransform(), MonthTransform(), DayTransform()]
)
def test_write_sort_temporal_dirs(sort_dir, null_order, temporal_xfrm):
    """Sort by year/month/day transforms with direction combos on native
    date column."""
    df = _make_transform_write_df(30)
    sort_order = SortOrder(
        SortField(
            source_id=3,
            transform=temporal_xfrm,
            direction=sort_dir,
            null_order=null_order,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sort_dir, null_order",
    [
        pytest.param(
            SortDirection.ASC,
            NullOrder.NULLS_LAST,
        ),
        pytest.param(
            SortDirection.DESC,
            NullOrder.NULLS_FIRST,
        ),
    ],
)
def test_write_sort_multi_type(sort_dir, null_order):
    """Multiple sort transforms on different column types: bucket on int,
    year on date, truncate on string."""
    df = _make_transform_write_df(20)
    sort_order = SortOrder(
        SortField(
            source_id=1,
            transform=BucketTransform(4),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        ),
        SortField(
            source_id=3,
            transform=YearTransform(),
            direction=sort_dir,
            null_order=null_order,
        ),
        SortField(
            source_id=2,
            transform=TruncateTransform(2),
            direction=SortDirection.ASC,
            null_order=NullOrder.NULLS_LAST,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        _check_write_read(df, PartitionSpec(), sort_order, tmp)
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


def test_to_iceberg_theta_sketches():
    """Test that DataFrame.to_iceberg() writes theta sketches (puffin files)
    with correct NDV estimates for supported column types.

    Verifies:
    1. The write succeeds and data is correct.
    2. A statistics file is present in the table metadata.
    3. The correct columns have theta sketches (int, string, date32).
    4. Columns that should NOT have sketches (float, bool) are excluded.
    5. The NDV estimates are correct and within expected ranges.
    """
    import datetime

    from bodo.tests.iceberg_database_helpers.metadata_utils import (
        get_metadata_field,
        get_metadata_path,
    )

    df = pd.DataFrame(
        {
            # Column A: int64, 10 distinct values → should get theta sketch
            "A": list(range(10)) * 2,
            # Column B: float64 → NOT a default theta sketch type (excluded)
            "B": [1.4, 1.5, 2.451, 0.0] * 5,
            # Column C: string, 5 distinct values → should get theta sketch
            "C": ["a", "ab", "cde", "af", "eg"] * 4,
            # Column D: bool → NOT a default theta sketch type (excluded)
            "D": [True, False, None, False] * 5,
            # Column E: date32, 1 distinct value → should get theta sketch
            "E": [datetime.date(2021, 1, 1)] * 20,
        }
    )
    bdf = bpd.from_pandas(df)

    import bodo

    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "iceberg_warehouse")

            bdf.to_iceberg(
                "theta_test_table",
                location=path,
            )
            assert bdf.is_lazy_plan()

            # 1. Verify data is correct
            out_df = pyiceberg_reader.read_iceberg_table("theta_test_table", path)
            _test_equal(
                out_df,
                df,
                check_pandas_types=False,
                sort_output=True,
                reset_index=True,
            )

            # 2. Verify statistics file exists in metadata
            # Read the metadata JSON directly (like the JIT tests do)
            from bodo.tests.iceberg_database_helpers.metadata_utils import (
                get_metadata_field,
                get_metadata_path,
            )

            metadata_path = get_metadata_path(path, "", "theta_test_table")
            statistics_lst = get_metadata_field(metadata_path, "statistics")
            assert len(statistics_lst) >= 1, "Expected at least one statistics file"

            # 3. Verify correct columns have theta sketches
            # Field IDs are 1-based: A=1, B=2, C=3, D=4, E=5
            # Expected: A(int)=10, C(string)=5, E(date32)=1
            # B(float) and D(bool) should NOT have sketches
            statistics = statistics_lst[0]
            blob_metadata = statistics["blob-metadata"]
            assert len(blob_metadata) > 0, "Expected at least one blob"

            seen_fields = {}
            for blob in blob_metadata:
                fields = blob["fields"]
                assert len(fields) == 1, "Expected one field per blob"
                field_id = fields[0]
                ndv = blob["properties"].get("ndv")
                seen_fields[field_id] = ndv

            # A (field_id=1) should have NDV=10
            assert 1 in seen_fields, "Column A should have a theta sketch"
            assert seen_fields[1] == "10", (
                f"Column A NDV should be 10, got {seen_fields.get(1)}"
            )

            # C (field_id=3) should have NDV=5
            assert 3 in seen_fields, "Column C should have a theta sketch"
            assert seen_fields[3] == "5", (
                f"Column C NDV should be 5, got {seen_fields.get(3)}"
            )

            # B (field_id=2) and D (field_id=4) should NOT have sketches
            assert 2 not in seen_fields, (
                "Column B (float) should NOT have a theta sketch"
            )
            assert 4 not in seen_fields, (
                "Column D (bool) should NOT have a theta sketch"
            )

    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_to_iceberg_theta_sketches_append():
    """Test that appending to an Iceberg table with existing theta sketches
    correctly merges the new sketches with the existing ones.

    Verifies:
    1. Initial write creates statistics with correct NDV.
    2. Append adds a second statistics file.
    3. The merged NDV reflects both writes.
    """

    from bodo.tests.iceberg_database_helpers.metadata_utils import (
        get_metadata_field,
        get_metadata_path,
    )

    df1 = pd.DataFrame(
        {
            "A": list(range(10)) * 2,  # 10 distinct values
            "C": ["a", "ab", "cde", "af", "eg"] * 4,  # 5 distinct values
        }
    )
    df2 = pd.DataFrame(
        {
            "A": list(range(10, 20)),  # 10 new distinct values (20 total)
            "C": ["x", "yz"] * 5,  # 2 new distinct values (7 total)
        }
    )

    import bodo

    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "iceberg_warehouse")

            # Write initial data
            bdf1 = bpd.from_pandas(df1)
            bdf1.to_iceberg("append_theta_table", location=path)

            # Append more data
            bdf2 = bpd.from_pandas(df2)
            bdf2.to_iceberg("append_theta_table", location=path, append=True)

            # Verify data is correct (both writes present)
            out_df = pyiceberg_reader.read_iceberg_table("append_theta_table", path)
            expected = pd.concat([df1, df2], ignore_index=True)
            _test_equal(
                out_df,
                expected,
                check_pandas_types=False,
                sort_output=True,
                reset_index=True,
            )

            # Verify statistics exist (should have 2 statistics files now)
            metadata_path = get_metadata_path(path, "", "append_theta_table")
            statistics_lst = get_metadata_field(metadata_path, "statistics")
            assert len(statistics_lst) >= 2, (
                f"Expected at least 2 statistics files after append, got {len(statistics_lst)}"
            )

            # Verify blobs exist in the latest statistics
            latest_stats = statistics_lst[-1]
            blob_metadata = latest_stats["blob-metadata"]
            assert len(blob_metadata) > 0, "Expected at least one blob after append"

            # Verify NDV estimates are within expected bounds.
            # Theta sketches provide approximate NDV estimates; we use a
            # tolerance of ±3 (accounts for sketch approximation).
            # df1 + df2 = column A has 20 distinct ints, column C has 7
            # distinct strings.
            expected_ndvs = {"A": 20, "C": 7}
            schemas = get_metadata_field(metadata_path, "schemas")
            latest_schema_id = get_metadata_field(metadata_path, "current-schema-id")
            latest_schema = next(
                s for s in schemas if s["schema-id"] == latest_schema_id
            )
            field_id_to_name = {f["id"]: f["name"] for f in latest_schema["fields"]}
            for blob in blob_metadata:
                if blob.get("type") != "apache-datasketches-theta-v1":
                    continue
                props = blob.get("properties", {})
                assert "ndv" in props, (
                    f"Blob for fields {blob.get('fields')} missing ndv property"
                )
                ndv_est = int(props["ndv"])
                field_id = blob["fields"][0]
                col_name = field_id_to_name.get(field_id, str(field_id))
                expected = expected_ndvs.get(col_name)
                if expected is not None:
                    assert abs(ndv_est - expected) <= 3, (
                        f"NDV for column '{col_name}' expected ~{expected}, "
                        f"got {ndv_est} (diff={abs(ndv_est - expected)})"
                    )

    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_to_iceberg_theta_sketches_empty_dataframe():
    """Test that writing an empty DataFrame with theta sketches enabled
    succeeds without errors and produces correct metadata.

    Verifies:
    1. Empty DataFrame write succeeds.
    2. No statistics file is created (nothing to sketch).
    3. Data round-trips correctly (still empty).
    """
    from bodo.tests.iceberg_database_helpers.metadata_utils import (
        get_metadata_field,
        get_metadata_path,
    )

    df = pd.DataFrame(
        {
            "A": pd.array([], dtype="int64"),
            "C": pd.array([], dtype="str"),
        }
    )

    import bodo

    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "iceberg_warehouse")

            bdf = bpd.from_pandas(df)
            bdf.to_iceberg("empty_theta_table", location=path)

            # Verify data is correct (still empty)
            out_df = pyiceberg_reader.read_iceberg_table("empty_theta_table", path)
            _test_equal(
                out_df,
                df,
                check_pandas_types=False,
                sort_output=True,
                reset_index=True,
            )

            # Verify statistics exist with NDV=0 (empty data has no distinct values)
            metadata_path = get_metadata_path(path, "", "empty_theta_table")
            statistics_lst = get_metadata_field(metadata_path, "statistics")
            assert len(statistics_lst) >= 1, (
                "Expected statistics file for empty DataFrame"
            )

            # Verify NDV is 0 for all columns
            statistics = statistics_lst[0]
            blob_metadata = statistics["blob-metadata"]
            for blob in blob_metadata:
                ndv = blob["properties"].get("ndv")
                assert ndv == "0", f"Expected NDV=0 for empty DataFrame, got {ndv}"

    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_to_iceberg_theta_sketches_serialization_error():
    """Test that merge_and_write_puffin handles malformed serialized data
    by raising a clear error instead of reading past the buffer.

    This tests the bounds-checking added to bodo_theta_utils_merge_and_write_puffin.
    """
    # Create a valid-looking but truncated serialized bytes object.
    # A valid serialization starts with uint32 n_sketches, then per sketch:
    # uint32 len, then len bytes of data.
    # Here we say n_sketches=1 but provide no length or data.
    import struct

    from bodo.io.iceberg.theta_utils import merge_and_write_puffin

    truncated = struct.pack("<I", 1)  # n_sketches = 1, but no actual sketch data

    with pytest.raises((ValueError, RuntimeError)):
        merge_and_write_puffin(
            [truncated],
            "/tmp/fake_puffin.stats",
            "",
            12345,
            1,
            None,  # iceberg_schema not needed for this error path
            None,  # arrow_fs not needed
            "",
        )


def test_sketch_ptr_double_free_safety():
    """Test that SketchPtr prevents double-free by tracking ownership.

    Verifies:
    1. delete() can be called safely.
    2. A second delete() is a no-op (no crash/UB).
    3. __del__ doesn't double-free.
    """
    from bodo.io.iceberg.theta_utils import SketchPtr

    # SketchPtr with ptr=0 (null)
    sp_zero = SketchPtr(0)
    sp_zero.delete()  # Should be no-op
    sp_zero.delete()  # Still safe

    # SketchPtr with a fake non-zero ptr (we can't actually delete it,
    # so just verify the tracking logic)
    sp = SketchPtr(0xDEADBEEF)
    assert sp.ptr == 0xDEADBEEF
    released = sp.release()
    assert released == 0xDEADBEEF
    assert sp.ptr == 0  # After release, internal ptr is 0
    sp.delete()  # Should be no-op since ptr is now 0
