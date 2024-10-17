import os

import cloudpickle
import pandas as pd
import pytest

import bodo
from bodo.submit.spawner import submit_jit
from bodo.tests.conftest import datapath_util
from bodo.tests.utils import pytest_spawn_mode

pytestmark = pytest_spawn_mode


def test_return_not_supported():
    @submit_jit
    def fn():
        return 0

    with pytest.raises(NotImplementedError, match="not supported by submit_jit"):
        fn()


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


import bodo.tests.test_spawn.mymodule as mymod


@pytest.mark.skip(
    reason="TODO: pytest prevents module resolution from working correctly"
)
def test_import_module():
    """Check that modules referenced by submitted fns will be imported on the
    spawned process"""

    def print_np():
        with bodo.no_warning_objmode:
            mymod.f()

    print(print_np)
    fn = submit_jit(cache=True)(print_np)
    fn()

    fn2 = submit_jit(cache=True)(print_np)
    fn2()


def f0():
    return 0


def test_cloudpickle():
    """Test that cloudpickle preserves arbitrary properties attached to
    serialized functions"""
    f0.my_prop = "hello world"
    f0_pickle = cloudpickle.dumps(f0)
    f1 = cloudpickle.loads(f0_pickle)
    assert f1.my_prop == "hello world"
    assert f1() == 0


def test_closure():
    """Check that cloudpickle correctly handles variables captured by a
    closure"""

    def create_closure():
        f = 10

        def closure():
            print("in closure:", f)

        return closure

    fn = submit_jit(cache=True)(create_closure())
    fn()


CUSTOMER_TABLE_PATH = datapath_util("tpch-test_data/parquet/customer.parquet")


@submit_jit(cache=True)
def simple_test():
    df = pd.read_parquet(CUSTOMER_TABLE_PATH)
    max_acctbal = (
        df[df["C_ACCTBAL"] > 1000.0]
        .groupby(["C_MKTSEGMENT", "C_NATIONKEY"])
        .agg({"C_ACCTBAL": "max"})
    )
    print("Max acctbal:")
    print(max_acctbal)


def test_simple():
    """Simple test that reads data and computes in spawn mode"""
    simple_test()
