import os
import subprocess
import sys
import time

import numba  # noqa TID253
import numpy as np
import pandas as pd
import psutil
import pytest

import bodo
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.managers import LazyBlockManager
from bodo.spawn.spawner import SpawnDispatcher, destroy_spawner, get_num_workers
from bodo.tests.utils import (
    _test_equal,
    check_func,
    pytest_spawn_mode,
    temp_env_override,
)

pytestmark = pytest_spawn_mode

VALUE = 1


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
        with bodo.ir.object_mode.no_warning_objmode:
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

    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.pq")

    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH, dtype_backend="pyarrow")
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
    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.pq")

    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH, dtype_backend="pyarrow")
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
        with bodo.ir.object_mode.no_warning_objmode(ret_val="int64"):
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
        with bodo.ir.object_mode.no_warning_objmode(ret_val="unicode_type"):
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
        import bodosql.compiler  # isort:skip # noqa

        bodo.spawn.utils.import_bodosql_compiler_on_workers()

        @bodo.jit(spawn=True)
        def impl3(a, bc, b):
            df = bc.sql('select * from "source"')
            return df

        df1 = pd.DataFrame({"id": [1, 1], "dep": ["software", "hr"]})
        df2 = pd.DataFrame({"id": [1, 2], "dep": ["finance", "hardware"]})
        bc = bodosql.context.BodoSQLContext({"target": df1, "source": df2})
        _test_equal(impl3(1, bc, 4), df2)
    except ImportError:
        return
    finally:
        # Make sure bodosql isn't in modules since checked in test_no_bodosql_import
        sys.modules.pop("bodosql", None)


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


def test_args_empty_tuple():
    @bodo.jit(spawn=True)
    def impl(A):
        return A

    arg = ()
    _test_equal(impl(arg), arg)


def test_dist_false():
    """Make sure distributed=False disables spawn"""

    @bodo.jit(spawn=True, distributed=False)
    def f(A):
        return A

    assert not isinstance(f, SpawnDispatcher)


def test_results_deleted_after_collection(datapath):
    """Test that results are deleted from workers after collection"""
    CUSTOMER_TABLE_PATH = datapath("tpch-test_data/parquet/customer.pq")

    @bodo.jit(spawn=True)
    def impl():
        df = pd.read_parquet(CUSTOMER_TABLE_PATH, dtype_backend="pyarrow")
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
    """test bodo.types.register_type() support in spawn mode"""
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)
    bodo.types.register_type("my_type1", df_type1)

    def impl():
        with numba.objmode(df="my_type1"):
            df = pd.DataFrame({"A": [1, 2, 5]})
        return df

    check_func(
        impl,
        (),
        is_out_distributed=False,
        additional_compiler_arguments={"replicated": ["df"]},
    )


@pytest.mark.no_cover
@pytest.mark.skip(reason="Testing atexit behavior is flakey.")
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


@pytest.mark.skip(reason="Testing atexit behavior is flakey.")
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


def test_spawn_globals_objmode():
    """Tests that referencing global variables works
    in object mode
    """

    @bodo.jit(spawn=True)
    def f():
        with bodo.ir.object_mode.no_warning_objmode(val="int64"):
            val = VALUE
        return val

    assert f() == VALUE


def test_spawn_input():
    """
    Tests that using input after spawn mode doesn't fail
    """
    sub = subprocess.Popen(
        [
            f"{sys.executable}",
            "-c",
            "'import bodo; bodo.jit(spawn=True)(lambda x: x)(1); input()'",
        ],
        shell=True,
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    sub.communicate(b"\n")
    assert sub.returncode == 0


@pytest.mark.skip("TODO [BSE-5141]: Fix flakey test on CI.")
def test_spawn_jupyter_worker_output_redirect():
    """
    Make sure redirectiing worker output works in Jupyter on Windows
    """
    with temp_env_override(
        {
            "BODO_NUM_WORKERS": "1",
            "BODO_OUTPUT_REDIRECT_TEST": "1",
        }
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import bodo; bodo.jit(spawn=True)(lambda: print("Hello from worker"))()',
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "Hello from worker"
        assert result.stderr == ""


def test_spawn_process_on_nodes():
    """
    Test that spawn_process works correctly on workers.
    This is a basic test to ensure that the spawn_process function can be called
    and that it returns the expected result.
    """

    # Test IO doesn't break spawn_process
    proc = bodo.spawn_process_on_nodes(["python", "-c", "print(0)"])
    bodo.stop_process_on_nodes(proc)
    assert proc._rank_to_pid is not None, "Process should have a rank to PID mapping"
    assert isinstance(proc._rank_to_pid, dict), (
        "Rank to PID mapping should be a dictionary"
    )
    assert len(proc._rank_to_pid) == 1, (
        "There should be one process spawned on the worker since we test on one node"
    )
    time.sleep(1)  # Give some time for the process to close
    for rank, pid in proc._rank_to_pid.items():
        # Check that the process was stopped on the worker
        assert not psutil.pid_exists(pid), (
            f"Process on rank {rank} with PID {pid} still exists"
        )

    # Test we can stop a process that is running
    proc = bodo.spawn_process_on_nodes(["python", "-c", "import time; sleep(10)"])
    # Ensure the process is running
    assert all(psutil.pid_exists(pid) for pid in proc._rank_to_pid.values()), (
        "Process should be running on all workers"
    )
    bodo.stop_process_on_nodes(proc)
    time.sleep(1)  # Give some time for the process to close
    for rank, pid in proc._rank_to_pid.items():
        # Check that the process was stopped on the worker
        assert not psutil.pid_exists(pid), (
            f"Process on rank {rank} with PID {pid} still exists"
        )
