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


def test_read_parquet_len_shape(datapath):
    """Test length/shape after read parquet is correct"""
    path = datapath("example.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    len(bodo_out) == py_out

    # create a new lazy DF
    bodo_out2 = bd.read_parquet(path)

    # test shape
    bodo_out2.shape == py_out.shape
