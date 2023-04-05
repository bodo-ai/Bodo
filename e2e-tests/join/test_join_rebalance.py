import os
import shutil

from utils.utils import run_cmd


def test_join_rebalance():
    # remove __pycache__ (numba stores cache in there)
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))

        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        num_processes = 12
        cmd = [
            "mpiexec",
            "-n",
            str(num_processes),
            "python",
            "-u",
            "join_rebalance.py",
        ]
        run_cmd(cmd)
    finally:
        os.chdir(pytest_working_dir)
