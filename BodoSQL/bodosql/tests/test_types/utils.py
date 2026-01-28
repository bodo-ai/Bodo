import contextlib

import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from pyiceberg.catalog import Catalog

import bodo
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.spawn.utils import run_rank0
from bodo.tests.utils import _test_equal_guard, gen_unique_table_id
from bodo.tests.utils_jit import reduce_sum


def gen_unique_id(name_prefix: str) -> str:
    path = None
    if bodo.get_rank() == 0:
        path = gen_unique_table_id(name_prefix)
    path = MPI.COMM_WORLD.bcast(path)
    return path


def test_equal_par(bodo_output, py_output):
    passed = _test_equal_guard(
        bodo_output,
        py_output,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Parallel test failed"


@contextlib.contextmanager
def create_iceberg_table(conn_str: str, table_id: str, df: pd.DataFrame):
    catalog: Catalog = conn_str_to_catalog(conn_str)
    run_rank0(
        lambda: catalog.create_table(table_id, pa.Schema.from_pandas(df)).append(
            pa.Table.from_pandas(df)
        )
    )()
    yield
    run_rank0(lambda: catalog.purge_table(table_id))()
