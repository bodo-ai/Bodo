import os
import pytest
from bodosdk.models import JobStatus, ClusterStatus
from integration_tests.common import (
    create_cluster_helper,
    remove_cluster,
    create_job_helper,
    remove_job,
)

TEST_DIR_GIT = "./integration_tests/test_files/"


@pytest.mark.parametrize(
    "cluster_name, instance_type",
    [
        ("e2e_test_new_cluster_new_image_aws_1", "c5.xlarge"),
        ("e2e_test_new_cluster_new_image_aws_2", "m5.2xlarge"),
        ("e2e_test_new_cluster_new_image_azure_2", "Standard_D8as_v4"),
        # ("e2e_test_new_cluster_new_image_azure_2", "Standard_D32as_v4") # Commenting due to quota issues
    ],
)
def test_cluster_creation(
    bodo_client, bodo_version, img_id, cluster_description, cluster_name, instance_type
):
    """
    Tests cluster creation on the platform for the new image_id/bodoVersion being released.
    In order to add more tests, specify the cluster name and instance_type in the parametrization above as a tuple.
    bodo_client, bodo_version, img_id, cluster_description are defined as fixtures.

    Takes in 2 arguments as parameters(as parameterized above):
    :param cluster_name: dummy name for the cluster being created
    :param instance_type: the size of the cluster being created
    """
    if os.environ["CLOUD_PROVIDER"].lower() not in cluster_name:
        pytest.skip("Skipping test: Cluster type not available on this workspace.")
    cluster_metadata = create_cluster_helper(
        bodo_client,
        bodo_version,
        img_id,
        cluster_name,
        instance_type,
        cluster_description,
    )
    assert (
        cluster_metadata.status == ClusterStatus.RUNNING
    ), "The Cluster is not Running, check logs on DEV to see what went wrong!"
    remove_cluster(bodo_client, cluster_metadata.uuid)


@pytest.mark.parametrize(
    "job_name, script, instance_type, workers, input_var, workspace_path, instance_role",
    [
        (
            "new_image_test_nyc-taxi-daily-pickup",
            os.path.join(TEST_DIR_GIT, "common/nyc_taxi/daily_pickup.py"),
            "c5.2xlarge"
            if os.environ["CLOUD_PROVIDER"] == "AWS"
            else "Standard_D8as_v4",
            2,
            {},
            None,
            False,
        ),
        (
            "new_image_test_nyc-taxi-daily-pickup",
            os.path.join(TEST_DIR_GIT, "common/nyc_taxi/daily_pickup.py"),
            None,
            2,
            {},
            None,
            False,
        ),
        (
            "new_image_test_beer_review",
            os.path.join(TEST_DIR_GIT, "common/beer_review/beer_reviews.py"),
            "c5.2xlarge"
            if os.environ["CLOUD_PROVIDER"] == "AWS"
            else "Standard_D8as_v4",
            2,
            {},
            None,
            False,
        ),
        (
            "new_image_test_iceberg_read_aws",
            os.path.join(TEST_DIR_GIT + "aws/iceberg/iceberg_read.py"),
            "c5.2xlarge",
            2,
            {},
            None,
            True,
        ),
        (
            "new_image_test_bodosql",
            os.path.join(TEST_DIR_GIT, "common/bodosql_public.py"),
            "c5.2xlarge"
            if os.environ["CLOUD_PROVIDER"] == "AWS"
            else "Standard_D8as_v4",
            2,
            {},
            None,
            False,
        ),
        (
            "new_image_test_tpch_queries",
            os.path.join(TEST_DIR_GIT, "common/tpch_queries.py"),
            "c5.2xlarge"
            if os.environ["CLOUD_PROVIDER"] == "AWS"
            else "Standard_D8as_v4",
            2,
            {},
            None,
            False,
        ),
        (
            "new_image_test_snowflake_bodosql",
            os.path.join(TEST_DIR_GIT, "aws/snowflake/snowflake_read_write.py"),
            "c5.2xlarge",
            2,
            {},
            None,
            False,
        ),
    ],
)
def test_job(
    bodo_client,
    img_id,
    create_cluster,
    create_instance_role,
    github_branch,
    job_name,
    script,
    instance_type,
    workers,
    input_var,
    workspace_path,
    instance_role,
):
    """
    How to add more test documented here: https://bodo.atlassian.net/wiki/spaces/BP/pages/1185218561/Integration+tests+for+images+with+platform+on+dev
    Tests if various script files and bodo functions run as expected on the platform for the new image_id/bodoVersion being released,
    in the form of job submissions.
    Depending on the input params, either creates a new cluster for the job, or uses a uuid of an already running cluster.
    This function uses pulls files from github repo(test_files under integration tests)
    In order to add more tests, add in a parametrization in the order mentioned below
    Takes in the following paramters as a tuple entry to the function parametrization:
    :param job_name: dummy name for the job being created
    :param script: the python file to be executed as part of the job
    :param path: the path in the shared file system where the script resides
    :param instance_type: If specified, creates and submits a job to a new cluster of this instance_type.
                    If None, uses an already running cluster
    :param input_vars: Specifies any environment variables required for the script file execution. Usually empty.
    :param workspace_path: Default None. Incase using the platform file system as source, the workspace path for the directory in which the script resides.
    :param instance_role: Boolean. Indicates if the job requires the use of an instance role
    """
    job_metadata = None

    if "common" not in script and os.environ["CLOUD_PROVIDER"].lower() not in script:
        pytest.skip("Skipping test: Test unavailable on this workspace!")

    # check if instance types are compatible with the cloud provider
    # skip if None instance type since we already assign the right one
    if instance_type is not None:
        if (os.environ["CLOUD_PROVIDER"] == "AWS" and "large" not in instance_type) or (
            os.environ["CLOUD_PROVIDER"] == "AZURE" and "Standard" not in instance_type
        ):
            assert False, "Wrong instance type, Please use a compatible instance type!"

    role_uuid = None
    if instance_role:
        role_uuid = create_instance_role

    if instance_type:
        job_metadata = create_job_helper(
            job_name,
            script,
            input_var,
            bodo_client,
            role_uuid,
            workspace_path,
            img_id=img_id,
            instance_type=instance_type,
            workers=workers,
            branch=github_branch,
        )
    else:
        job_metadata = create_job_helper(
            job_name,
            script,
            input_var,
            bodo_client,
            create_instance_role,
            workspace_path,
            cluster_uuid=create_cluster,
            branch=github_branch,
        )

    assert (
        job_metadata.status == JobStatus.FINISHED
    ), "Job failed, Check the logs on Dev to see what went wrong"
    # Ideally, we dont remove the job incase of a failure
    # since we need to investigate what happened manually by checking logs on dev.
    remove_job(bodo_client, job_metadata.uuid)
