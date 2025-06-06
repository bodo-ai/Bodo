import datetime
import operator

import numba.core.utils
import pandas as pd
import pyarrow as pa
import pyiceberg.catalog
import pyiceberg.expressions
import pytest

import bodo.pandas as bpd
from bodo.io.iceberg.catalog.dir import DirCatalog
from bodo.tests.iceberg_database_helpers import pyiceberg_reader
from bodo.tests.utils import _test_equal

pytest_mark = pytest.mark.iceberg


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
    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        limit=10,
    )
    assert bodo_out.is_lazy_plan()

    # Check that the plan has been optimized to a single read
    pre, post = bpd.utils.getPlanStatistics(bodo_out._plan)
    assert pre == 2
    assert post == 1

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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    ).head(10)
    assert bodo_out.is_lazy_plan()

    # Check that the plan has been optimized to a single read
    pre, post = bpd.utils.getPlanStatistics(bodo_out._plan)
    assert pre == 2
    assert post == 1

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
        {
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
        {
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
        {
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
        {
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
        {
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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    bodo_out2 = bodo_out[eval(f"bodo_out.A {op_str} 3")]
    assert bodo_out2.is_lazy_plan()

    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 2
    assert post == 1

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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[(bodo_out.A < 5) & (bodo_out.C >= 3) & (bodo_out.A != 3)]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 2
    assert post == 1
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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
        row_filter="A < 3",
    )

    bodo_out2 = bodo_out[bodo_out.C >= 3]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 2
    assert post == 1

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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[bodo_out.B < 4]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 2
    assert post == 1

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
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[bodo_out.A <= pd.Timestamp("2018-12-12")]
    assert bodo_out2.is_lazy_plan()
    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    assert pre == 2
    assert post == 1

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
