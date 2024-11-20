import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.managers import LazyBlockManager
from bodo.submit.spawner import SubmitDispatcher, destroy_spawner, get_num_workers
from bodo.tests.utils import (
    _test_equal,
    check_func,
    pytest_spawn_mode,
    temp_env_override,
)

pytestmark = pytest_spawn_mode


def test_propogate_same_exception_all_ranks():
    """Test that a single exact exception is raised when all ranks raise the same
    exception"""

    @bodo.jit(spawn=True)
    def fn(a):
        if a:
            raise ValueError("bad value")
        return a

    with pytest.raises(ValueError, match="bad value"):
        fn(3)


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


@pytest.mark.skip("BSE-4099: Requires pandas multi index support in BodoDataFrame")
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


def test_environment():
    @bodo.jit(spawn=True)
    def get_from_env(env_var):
        with bodo.no_warning_objmode(ret_val="int64"):
            ret_val = int(os.environ[env_var])
        return ret_val

    with temp_env_override(
        {
            "BODO_TESTS_VARIABLE": "42",
            "AWS_TESTS_VARIABLE": "43",
            "AZURE_TESTS_VARIABLE": "44",
        }
    ):
        assert get_from_env("BODO_TESTS_VARIABLE") == 42
        assert get_from_env("AWS_TESTS_VARIABLE") == 43
        assert get_from_env("AZURE_TESTS_VARIABLE") == 44

    # track variables specified in decorator
    @bodo.jit(
        spawn=True,
        propagate_env=["EXTRA_ENV"],
    )
    def get_from_env_decorator(env_var):
        with bodo.no_warning_objmode(ret_val="unicode_type"):
            ret_val = os.environ.get(env_var, "DOES_NOT_EXIST")
        return ret_val

    with temp_env_override(
        {
            "BODO_TESTS_VARIABLE": "42",
            "EXTRA_ENV": "45",
            "BODO_DYLD_INSERT_LIBRARIES": "xx",
        }
    ):
        assert get_from_env_decorator("BODO_TESTS_VARIABLE") == "42"
        assert get_from_env_decorator("EXTRA_ENV") == "45"
        assert get_from_env_decorator("DYLD_INSERT_LIBRARIES") == "xx"

    assert get_from_env_decorator("BODO_TESTS_VARIABLE") == "DOES_NOT_EXIST"
    assert get_from_env_decorator("EXTRA_ENV") == "DOES_NOT_EXIST"
    assert get_from_env_decorator("DYLD_INSERT_LIBRARIES") == "DOES_NOT_EXIST"

    # tests decorator error raising
    with temp_env_override(
        {
            "EXTRA_ENV": "45",
        }
    ):
        err_msg = (
            "spawn=False while propagate_env is set. No worker to propagate env vars."
        )
        with pytest.raises(
            bodo.utils.typing.BodoError,
            match=err_msg,
        ):

            @bodo.jit(
                spawn=False,
                propagate_env=["EXTRA_ENV"],
            )
            def env_decorator_error():
                return 1

            env_decorator_error()


def test_args():
    """Make sure arguments work for submit_jit functions properly"""

    @bodo.jit(spawn=True)
    def impl1(a, arr, c=1):
        return a + c + arr.sum()

    A = np.arange(6)
    _test_equal(impl1(1, A, 3), 19)
    _test_equal(impl1(1, arr=A, c=3), 19)

    # Test replicated flag
    @bodo.jit(spawn=True, replicated=["arr"])
    def impl2(a, arr, c=1):
        return a + c + arr.sum()

    A = np.arange(6)
    _test_equal(impl2(1, A, 3), 19)
    _test_equal(impl2(1, arr=A, c=3), 19)

    # Test BodoSQLContext if bodosql installed in test environment
    try:
        import bodosql
    except ImportError:
        return

    @bodo.jit(spawn=True)
    def impl3(a, bc, b):
        df = bc.sql('select * from "source"')
        return df

    df1 = pd.DataFrame({"id": [1, 1], "dep": ["software", "hr"]})
    df2 = pd.DataFrame({"id": [1, 2], "dep": ["finance", "hardware"]})
    bc = bodosql.context.BodoSQLContext({"target": df1, "source": df2})
    _test_equal(impl3(1, bc, 4), df2)


def test_args_tuple_list_dict():
    """Make sure nested distributed data arguments work for spawn functions properly"""

    @bodo.jit(spawn=True)
    def impl(A):
        return A

    n = 11
    df = pd.DataFrame({"a1": np.arange(n), "a23": np.ones(n)})
    arg = (3, df)
    _test_equal(impl(arg), arg)
    arg = [df, df]
    _test_equal(impl(arg), arg)
    arg = {"k1": df, "k23": df}
    _test_equal(impl(arg), arg)


def test_dist_false():
    """Make sure distributed=False disables spawn"""

    @bodo.jit(spawn=True, distributed=False)
    def f(A):
        return A

    assert not isinstance(f, SubmitDispatcher)


def test_results_deleted_after_collection(datapath):
    """Test that results are deleted from workers after collection"""
    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.parquet")

    @bodo.jit(spawn=True)
    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH)
        return df

    df = impl()
    assert isinstance(df, BodoDataFrame)
    assert isinstance(df._mgr, (LazyBlockManager, LazyArrayManager))
    res_id = df._mgr._md_result_id
    assert res_id is not None
    collect_func = df._mgr._collect_func
    assert collect_func is not None
    df._mgr._collect()

    assert collect_func(res_id) is None


def test_spawn_type_register():
    """test bodo.register_type() support in spawn mode"""
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)
    bodo.register_type("my_type1", df_type1)

    def impl():
        with bodo.objmode(df="my_type1"):
            df = pd.DataFrame({"A": [1, 2, 5]})
        return df

    check_func(
        impl,
        (),
        is_out_distributed=False,
        additional_compiler_arguments={"replicated": ["df"]},
    )


@pytest.mark.no_cover
def test_spawn_atexit_delete_result():
    """Tests that results in the user program are deleted properly upon exit,
    even after spawner has been destroyed"""

    additional_envs = {
        "BODO_NUM_WORKERS": str(1),
    }

    cmd = [sys.executable, "-u", "-m", "spawn_exit"]
    # get directory of test python executable
    cwd = os.path.dirname(os.path.realpath(__file__))

    try:
        subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=dict(os.environ, **additional_envs),
            cwd=cwd,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(e.output)
        raise


def test_destroy_spawn_delete():
    """Tests that it is safe to get a distributed result, destroy spawner,
    create a new global spawner and delete the result.
    """

    @bodo.jit(spawn=True)
    def get_bodo_df(df):
        return df

    df = pd.DataFrame({"A": [1, 2, 3, 4, 5] * 100})

    # create a distributed result
    bodo_df = get_bodo_df(df)

    destroy_spawner()

    # create a new global spawner
    _ = get_bodo_df(df)

    del bodo_df
