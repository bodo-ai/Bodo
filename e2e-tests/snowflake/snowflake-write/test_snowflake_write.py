import os
import shutil

import pytest
from utils.utils import run_cmd


@pytest.mark.parametrize(
    "sf_username",
    [
        pytest.param(1, id="aws"),
        pytest.param(
            2,
            marks=pytest.mark.skip(reason="[BSE-4457] Fix hang on Nightly CI."),
            id="azure",
        ),
    ],
)
@pytest.mark.parametrize("use_put_method", [False, True])
def test_snowflake_write(sf_username, use_put_method):
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)
        snowflake_write(sf_username, use_put_method)

        # Run again on cached code
        snowflake_write(sf_username, use_put_method, True)

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)


def snowflake_write(sf_username, use_put_method, require_cache=False):
    # Run on all 36 cores.
    num_processes = 36
    cmd = [
        "python",
        "-u",
        "-W",
        "ignore",
        "snowflake_write.py",
        str(sf_username),
    ]
    if use_put_method:
        cmd.append("--use_put_method")
    if require_cache:
        cmd.append("--require_cache")
    run_cmd(cmd, additional_envs={"BODO_NUM_WORKERS": str(num_processes)})
