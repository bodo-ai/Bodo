import os

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.submit.spawner import get_num_workers
from bodo.tests.utils import check_func, pytest_spawn_mode

pytestmark = pytest_spawn_mode


def test_propogate_same_exception_all_ranks():
    """Test that a concise message is returned when all ranks fail"""

    @bodo.jit(spawn=True)
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

    @bodo.jit(spawn=True)
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
    fn = bodo.jit(spawn=True, cache=True)(modify_global)
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

    fn = bodo.jit(spawn=True, cache=True)(impl)
    fn()

    fn2 = bodo.jit(spawn=True, cache=True)(impl)
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


def test_closure():
    """Check that cloudpickle correctly handles variables captured by a
    closure"""

    def create_closure():
        f = 10

        def closure():
            return "in closure: " + str(f)

        return closure

    fn = create_closure()
    check_func(fn, (), only_spawn=True)


def test_return_scalar():
    """Test that scalars can be returned"""

    def fn():
        return 42

    check_func(fn, (), only_spawn=True)


def test_return_array():
    """Test that arrays can be returned"""

    def fn():
        A = np.zeros(100, dtype=np.int64)
        return A

    check_func(fn, (), only_spawn=True)


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

    check_func(
        impl,
        (),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        only_spawn=True,
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

    check_func(impl, (), is_out_distributed=False, only_spawn=True)


@pytest.mark.skip("submit_jit does not support output")
def test_environment():
    os.environ["BODO_TESTS_VARIABLE"] = "42"
    try:

        @bodo.jit(spawn=True)
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

    @bodo.jit(spawn=True)
    def impl1(a, arr, c=1):
        print(a + c, arr.sum())

    A = np.arange(6)
    impl1(1, A, 3)
    impl1(1, arr=A, c=3)

    # Test replicated flag
    @bodo.jit(spawn=True, replicated=["arr"])
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

    @bodo.jit(spawn=True)
    def impl3(a, bc, b):
        df = bc.sql('select * from "source"')
        print(df, a + b)

    df1 = pd.DataFrame({"id": [1, 1], "dep": ["software", "hr"]})
    df2 = pd.DataFrame({"id": [1, 2], "dep": ["finance", "hardware"]})
    bc = bodosql.context.BodoSQLContext({"target": df1, "source": df2})
    impl3(1, bc, 4)
