import os
import shutil

from utils.utils import run_cmd


def test_snowflake_read():

    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)
        snowflake_read()

        # Run again on cached code
        snowflake_read(True)

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)


def snowflake_read(require_cache=False):
    # Run on all 36 cores
    num_processes = 36
    cmd = [
        "mpiexec",
        "-n",
        str(num_processes),
        "-prepend-rank",
        "python",
        "-u",
        "snowflake_read.py",
    ]
    if require_cache:
        cmd.append("--require_cache")
    run_cmd(cmd)
