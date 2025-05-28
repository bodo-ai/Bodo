import pyiceberg.catalog
import pytest

import bodo.pandas as bpd
from bodo.tests.iceberg_database_helpers import pyiceberg_reader
from bodo.tests.utils import _test_equal

pytest_mark = pytest.mark.iceberg


@pytest.mark.slow
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
