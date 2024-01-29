import os
import sys
from contextlib import contextmanager, nullcontext
from tempfile import TemporaryDirectory

import boto3
import pytest
from utils.utils import update_env_vars

e2e_tests_base_dir = os.path.dirname(__file__)
if e2e_tests_base_dir not in sys.path:
    sys.path.append(e2e_tests_base_dir)
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] += os.pathsep + e2e_tests_base_dir
else:
    os.environ["PYTHONPATH"] = e2e_tests_base_dir


# Our Codebuild project does have a service role that has access
# to the required S3 buckets, etc. The way this usually works is that
# during S3 read, for instance, Arrow goes to the instance metadata
# service and gets the required token to make the S3 API calls.
# However, Codebuild seems to have some throttling at the token
# generation level when all ranks request them, leading some ranks
# to end up without a valid token. To circumvent this, we retrieve
# a set of temporary AWS credentials once before every test, so
# that all ranks can read them directly from the environment
# and don't run into issues due to throttling. In our tests
# we usually call mpiexec in a subprocess, which should inherit
# these environment variables. The reason this needed to be a
# pytest fixture is that if we were to assume the role in
# `run_nightly_tests.sh` once, it only allows us to get credentials
# that are valid for 1hr (limitations of role chaining:
# https://aws.amazon.com/premiumsupport/knowledge-center/iam-role-chaining-limit/).
# We can get the token directly from ECS Metadata service
# (like https://stroobants.dev/use-metadata-iam-role-in-your-docker-build.html
# suggests) which are supposed to be valid for 6hrs by default. However, in case of
# Codebuild, it seems like they're only valid for 1hr.
# See more experiments and details here: https://bodo.atlassian.net/browse/BE-3538.
@pytest.fixture(autouse=True)
def assume_iam_role():
    """
    A function level fixture that assumes a role with permissions
    to read the required S3 buckets and other resources, sets the
    appropriate environment variables, and then restores them at
    the end of the test.
    """
    old_creds = {}
    try:
        # Create STS client
        sts_client = boto3.client("sts")
        # Get the role arn to assume from an environment variable.
        # If environment variable is not set, use the default role.
        role_arn = "arn:aws:iam::427443013497:role/BodoEngineNightlyRole"
        # Get temporary credentials. These should be valid for 1hr
        # by default, which should be long enough for any one test.
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn, RoleSessionName="BodoEngineNightlySession"
        )
        credentials = assumed_role_object["Credentials"]
        # Update env with temp credentials.
        old_creds = update_env_vars(
            {
                "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
                "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"],
                "AWS_SESSION_TOKEN": credentials["SessionToken"],
            }
        )
        yield
    finally:
        # Restore environment
        update_env_vars(old_creds)


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
                # CodeBuild instance used on AWS Nightly comes with 824GB disk space
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
                # CodeBuild instance used on AWS Nightly comes with 824GB disk space
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
