import operator

import numba.core.utils
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


# TODO: Test filter pushdown
@pytest.mark.parametrize(
    "table_name",
    [
        "SIMPLE_NUMERIC_TABLE",
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_table_read_row_filter_pushdown(
    table_name,
    op,
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    # This is a placeholder for a more complex test that would involve filter pushdown
    # and multiple filters.
    db_schema, warehouse_loc = iceberg_database(table_name)
    DirCatalog(
        None,
        **{
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_out = bpd.read_iceberg(
        f"{db_schema}.{table_name}",
        None,
        {
            pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
            pyiceberg.catalog.WAREHOUSE_LOCATION: warehouse_loc,
        },
    )
    bodo_out2 = bodo_out[eval(f"bodo_out.A {op_str} 20")]

    assert bodo_out2.is_lazy_plan()

    pre, post = bpd.utils.getPlanStatistics(bodo_out2._mgr._plan)
    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out2 = py_out[eval(f"py_out.A {op_str} 20")]

    _test_equal(
        bodo_out2,
        py_out2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


# Need multiple filters
# Need to test on schema evolution
# Need to test file and row filters
# Need to test with the row_filter argument and filter pushdown
