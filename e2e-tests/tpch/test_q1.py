import os
import shutil

from utils.utils import run_cmd


def test_q1():
    """
    Test TPCH Q1 using Spawn mode.
    """
    pytest_working_dir = os.getcwd()

    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))

        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        try:
            n_workers = 4
            execute_q1(n_workers, with_mpiexec=False)
        except:
            print(f"Failed without mpiexec ({n_workers} workers).")
            raise

        try:
            n_workers = 16
            execute_q1(n_workers, with_mpiexec=True)
        except:
            print(f"Failed without mpiexec ({n_workers} workers).")
            raise

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)


def execute_q1(n_workers: int, with_mpiexec: bool = False):
    """
    Execute Q1 in Spawn Mode.

    Args:
        with_mpiexec (bool, optional): Whether the parent process
            should be started with mpiexec. Defaults to False.
    """

    cmd = ["mpiexec", "-n", "1"] if with_mpiexec else []
    cmd += [
        "python",
        "-u",
        "-W",
        "ignore",
        "q1.py",
    ]
    run_cmd(cmd, additional_envs={"BODO_NUM_WORKERS": str(n_workers)})
