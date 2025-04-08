import pandas as pd

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def test_read_parquet(datapath):
    """Very simple test to read a parquet file for sanity checking."""
    path = datapath("example.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    _test_equal(
        bodo_out,
        py_out,
    )


def test_read_parquet_md(datapath):
    """Test to ensure we can access Metadata after executing plan while keeping
    result distributed across workers.
    """
    path = datapath("example.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    assert len(bodo_out) == len(py_out)
    assert bodo_out.shape == py_out.shape

    bodo_out.head()

    assert bodo_out._lazy
