import os
import sys
from contextlib import contextmanager, nullcontext
from tempfile import TemporaryDirectory

import pytest
from utils.utils import update_env_vars

e2e_tests_base_dir = os.path.dirname(__file__)
if e2e_tests_base_dir not in sys.path:
    sys.path.append(e2e_tests_base_dir)
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] += os.pathsep + e2e_tests_base_dir
else:
    os.environ["PYTHONPATH"] = e2e_tests_base_dir


@contextmanager
def spill_on_unpin():
    """
    Sets the environment variables to set up spill on unpin.
    It sets up a temporary spill directory (with best effort
    clean up afterwards) and sets the BufferPool spill
    configuration to point to it.
    At this point, we assign a spill disk with 100GiB.
    """

    old_env_state = {}
    tmp_dir = TemporaryDirectory()
    try:
        old_env_state = update_env_vars(
            {
                "BODO_BUFFER_POOL_SPILL_ON_UNPIN": "1",
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_dir.name),
                # E2E CI instance used on AWS Nightly comes with 824GB disk space
                # https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "100",
            }
        )
        yield
    finally:
        # Restore env
        update_env_vars(old_env_state)
        # Force cleanup of temporary dir
        del tmp_dir


@contextmanager
def move_on_unpin():
    """
    Sets the environment variables to set up move on unpin.
    It sets up a temporary spill directory (with best effort
    clean up afterwards) and sets the BufferPool spill
    configuration to point to it.
    At this point, we assign a spill disk with 100GiB.
    """
    old_env_state = {}
    tmp_dir = TemporaryDirectory()
    try:
        old_env_state = update_env_vars(
            {
                "BODO_BUFFER_POOL_MOVE_ON_UNPIN": "1",
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_dir.name),
                # E2E instance used on AWS Nightly comes with 824GB disk space
                # https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "100",
            }
        )
        yield
    finally:
        # Restore env
        update_env_vars(old_env_state)
        # Force cleanup of temporary dir
        del tmp_dir


@pytest.fixture(
    params=[
        pytest.param(nullcontext, id="default_unpin"),
        pytest.param(spill_on_unpin, id="spill_on_unpin"),
        pytest.param(move_on_unpin, id="move_on_unpin"),
    ]
)
def unpin_behavior(request):
    """
    Utility fixture to test with the three
    different unpin behaviors:
     - DEFAULT (i.e. just mark frame as unpinned)
     - SPILL
     - MOVE

    Returns:
        Context Manager that the test should
        use to run the test.
    """
    return request.param
