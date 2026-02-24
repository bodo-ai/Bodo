import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import datapath_util
from bodo.tests.utils import pytest_spawn_mode

pytestmark = pytest_spawn_mode


CUSTOMER_TABLE_PATH = datapath_util("tpch-test_data/parquet/customer.pq")


def test_recursive():
    """Test that a JIT function that calls another JIT function in spawn mode is
    supported"""

    @bodo.jit(spawn=True)
    def read_customer_table():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH, dtype_backend="pyarrow")
        return df

    @bodo.jit(spawn=True)
    def aggregate_customer_data(df):
        max_acctbal = (
            df[df["C_ACCTBAL"] > 1000.0]
            .groupby(["C_MKTSEGMENT", "C_NATIONKEY"])
            .agg({"C_ACCTBAL": "max"})
        )
        return max_acctbal

    @bodo.jit(spawn=True)
    def jit_fn_that_calls_other_jit_fns():
        """Function decorated with @bodo.jit(spawn=True) that calls other functions that are
        also decorated"""
        df = read_customer_table()
        max_acctbal = aggregate_customer_data(df)
        print("Max acctbal:")
        print(max_acctbal)

    jit_fn_that_calls_other_jit_fns()


@pytest.mark.skip(reason="This test is flaky, revisit when output is supported")
def test_recursive_across_module(capfd):
    import bodo.tests.test_spawn.mymodule as mymod

    @bodo.jit(spawn=True)
    def f():
        mymod.jit_fn()

    f()

    with capfd.disabled():
        out, _ = capfd.readouterr()
        n_calls = sum("called mymodule.jit_fn" in l for l in out.split("\n"))
        # Regardless of the number of workers, if the function was called in the
        # compiled mode, the print should only happen once
        assert n_calls == 1
