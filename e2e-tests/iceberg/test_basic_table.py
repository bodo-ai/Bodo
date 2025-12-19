import os
import shutil
from uuid import uuid4

import pytest
from utils.utils import run_cmd


@pytest.mark.skip(reason="TODO [BSE-4605]: update Iceberg tests.")
def test_iceberg_basic_df():
    num_processes = 4
    timeout = 300

    table_name = f"types_table_{str(uuid4())[:8]}"
    cmd = [
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
        run_cmd(cmd, timeout=timeout, additional_envs={"BODO_NUM_WORKERS": "1"})

        # Run again on cached code
        cmd.append("--require_cache")
        run_cmd(
            cmd,
            timeout=timeout,
            additional_envs={"BODO_NUM_WORKERS": str(num_processes)},
        )

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)
