import os
import shutil
from uuid import uuid4

from utils.utils import run_cmd


def test_iceberg_basic_df():
    num_processes = 36

    table_name = f"types_table_{str(uuid4())[:8]}"
    cmd = [
        "mpiexec",
        "-n",
        str(num_processes),
        "-prepend-rank",
        "python",
        "-u",
        "-W",
        "ignore",
        "basic_table.py",
        table_name,
    ]

    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)
        run_cmd(cmd)

        # Run again on cached code
        cmd.append("--require_cache")
        run_cmd(cmd)

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)
