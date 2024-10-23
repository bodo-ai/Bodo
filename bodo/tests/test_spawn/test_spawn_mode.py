import os

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.submit.spawner import get_num_workers, submit_jit
from bodo.tests.utils import _test_equal, pytest_spawn_mode

pytestmark = pytest_spawn_mode


def test_propogate_same_exception_all_ranks():
    """Test that a concise message is returned when all ranks fail"""

    @submit_jit
    def fn():
        # This will actually fail to compile since all paths end in an
        # exception, but that's ok for this test
        raise Exception("failed")

    with pytest.raises(Exception, match="All ranks failed"):
        fn()


@pytest.mark.skipif(
    int(os.environ.get("BODO_NUM_WORKERS", "0")) <= 1,
    reason="requires multiple workers",
)
def test_propogate_different_exceptions_some_ranks():
    """Test that a detailed message is returned when some ranks fail"""

    @submit_jit
    def fn():
        rank = bodo.get_rank()
        if rank != 0:
            raise Exception(f"failed on rank {rank}")

    with pytest.raises(Exception, match="Some ranks failed"):
        fn()


global_val = 0


def modify_global():
    global global_val
    global_val = 10


def test_modify_global():
    """Test that modifying a global value isn't supported and returns an
    informative error to the user"""
    fn = submit_jit(cache=True)(modify_global)
    with pytest.raises(Exception):
        fn()


@pytest.mark.skip(
    "capfd tests are unreliable with MPI see mymodule.py for details on unblocking this test"
)
def test_import_module(capfd):
    import bodo.tests.test_spawn.mymodule as mymod

    def impl():
        with bodo.no_warning_objmode:
            mymod.f()

    fn = submit_jit(cache=True)(impl)
    fn()

    fn2 = submit_jit(cache=True)(impl)
    fn2()

    with capfd.disabled():
        out, _ = capfd.readouterr()
        lines = out.split("\n")

        n_pes = get_num_workers()
        # Check that mymodule is only imported once per worker and once from the
        # spawning process
        assert sum("imported mymodule" in l for l in lines) == (n_pes + 1)
        # ensure that the function in mymodule was ran twice on every rank
        assert sum("called mymodule.f" in l for l in lines) == 2 * n_pes


def f0():
    return 0


def test_closure():
    """Check that cloudpickle correctly handles variables captured by a
    closure"""

    def create_closure():
        f = 10

        def closure():
            return "in closure: " + str(f)

        return closure

    fn = submit_jit(cache=True)(create_closure())
    assert fn() == "in closure: 10"


def test_return_scalar():
    """Test that scalars can be returned"""

    @submit_jit
    def fn():
        return 42

    assert fn() == 42


def test_return_array():
    """Test that arrays can be returned"""

    @submit_jit
    def fn():
        A = np.zeros(100, dtype=np.int64)
        return A

    _test_equal(fn(), np.zeros(100, dtype=np.int64))


def test_compute_return_df(datapath):
    """Simple test that reads data and computes in spawn mode, returning a
    dataframe"""

    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.parquet")

    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH)
        max_acctbal = (
            df[df["C_ACCTBAL"] > 1000.0]
            .groupby(["C_MKTSEGMENT", "C_NATIONKEY"])
            .agg({"C_ACCTBAL": "max"})
        )
        return max_acctbal

    submit_fn = submit_jit(cache=True)(impl)
    py_out = impl()
    bodo_submit_out = submit_fn()
    _test_equal(
        bodo_submit_out, py_out, sort_output=True, reset_index=True, check_dtype=False
    )


def test_compute_return_scalar(datapath):
    """Simple test that reads data and computes in spawn mode, returning a
    scalar"""
    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.parquet")

    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH)
        max_acctbal = (
            df[df["C_ACCTBAL"] > 1000.0]
            .groupby(["C_MKTSEGMENT", "C_NATIONKEY"])
            .agg({"C_ACCTBAL": "max"})
        )
        return max_acctbal.sum()

    submit_fn = submit_jit(cache=True)(impl)
    py_out = impl()
    bodo_submit_out = submit_fn()
    _test_equal(bodo_submit_out, py_out)


@pytest.mark.skip("submit_jit does not support output")
def test_environment():
    os.environ["BODO_TESTS_VARIABLE"] = "42"
    try:

        @submit_jit
        def get_from_env():
            with bodo.no_warning_objmode(ret_val="int64"):
                ret_val = int(os.environ["BODO_TESTS_VARIABLE"])
            return ret_val

        assert get_from_env() == 42
    finally:
        # Remove environment variable
        del os.environ["BODO_TESTS_VARIABLE"]


def test_args():
    """Make sure arguments work for submit_jit functions properly"""

    @submit_jit
    def impl1(a, arr, c=1):
        print(a + c, arr.sum())

    A = np.arange(6)
    impl1(1, A, 3)
    impl1(1, arr=A, c=3)

    # Test replicated flag
    @submit_jit(replicated=["arr"])
    def impl2(a, arr, c=1):
        print(a + c, arr.sum())

    A = np.arange(6)
    impl2(1, A, 3)
    impl2(1, arr=A, c=3)

    # Test BodoSQLContext if bodosql installed in test environment
    try:
        import bodosql
    except ImportError:
        return

    @submit_jit
    def impl3(a, bc, b):
        df = bc.sql('select * from "source"')
        print(df, a + b)

    df1 = pd.DataFrame({"id": [1, 1], "dep": ["software", "hr"]})
    df2 = pd.DataFrame({"id": [1, 2], "dep": ["finance", "hardware"]})
    bc = bodosql.context.BodoSQLContext({"target": df1, "source": df2})
    impl3(1, bc, 4)
