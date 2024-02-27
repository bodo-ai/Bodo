from os import environ
from bodosdk.models import (
    ClusterDefinition,
    JobDefinition,
    JobCluster,
    JobStatus,
    GitRepoSource,
    WorkspaceSource,
    JobClusterDefinition,
    ClusterStatus,
    CreateRoleDefinition,
    InstanceRole,
)
from time import time, sleep
from uuid import UUID
import os


def create_cluster_helper(
    client, bodo_version, img_id, cluster_name, instance_type, description
):
    """
    Creates a barebone form of cluster with minimum configuration and the lowest value on the paramter
    Reference link: https://pypi.org/project/bodosdk/#create-cluster
    The most important input params are bodo_version and the img_id, which are extracted from the previous
    step of the workflow
    """
    cluster_uuid = None
    cluster_definition = ClusterDefinition(
        name=cluster_name,
        instance_type=instance_type,
        workers_quantity=2,
        auto_shutdown=360,
        auto_pause=240,
        image_id=img_id,
        bodo_version=bodo_version,
        description=description,
    )
    result_create = client.cluster.create(cluster_definition)
    cluster_uuid = result_create.uuid
    cluster_details = client.cluster.get(cluster_uuid)
    # Keep sleeping until we see a success or a failure
    tries = 1
    while True:
        sleep(120 - time() % 120)  # run every 2 minutes ... you can change that
        cluster_details = client.cluster.get(cluster_uuid)
        if (
            cluster_details.status == ClusterStatus.RUNNING
            or cluster_details.status == ClusterStatus.FAILED
            or tries == 50
        ):
            break
        else:
            tries += 1

    return cluster_details


def remove_cluster(client, uuid):
    # removes the cluster with the passed in uuid
    client.cluster.remove(uuid)


def create_job_helper(
    job_name,
    script,
    input_var,
    client,
    instance_role_uuid,
    workspace_path=None,
    img_id=None,
    instance_type=None,
    workers=2,
    cluster_uuid=None,
    branch="",
):
    """
    Create a cluster with the specified name, image, description and instance type.
    Auto-pause and auto-shutdown are set to 4hrs and 6hrs respectively for automatic cleanup
    in case of a failure.
    Submit a job with the provided parameters using the Bodo SDK.
    Reference Link: https://pypi.org/project/bodosdk/#create-job
    """

    role_uuid = instance_role_uuid
    if img_id:
        cluster_obj = JobClusterDefinition(
            instance_type=instance_type,
            accelerated_networking=False,
            image_id=img_id,
            workers_quantity=workers,
            instance_role_uuid=role_uuid,
        )
    else:
        cluster_obj = JobCluster(uuid=cluster_uuid)

    job_source = create_source(workspace_path, branch=branch)

    # Creates a job definition with name, cluster and parameters required for job execution.
    job_definition = JobDefinition(
        name=job_name,
        args=script,
        source_config=job_source,
        cluster_object=cluster_obj,
        variables=input_var,
        timeout=2 * 60,
        retries=0,
        retries_delay=0,
        retry_on_timeout=False,
    )
    result_create = client.job.create(job_definition)
    job_uuid = result_create.uuid
    tries = 1
    # Keep sleeping until we see a success or a failure
    while True:
        sleep(120 - time() % 120)  # run every 2 minutes ... you can change that
        job_details = client.job.get(job_uuid)
        if (
            job_details.status == JobStatus.FINISHED
            or job_details.status == JobStatus.FAILED
            or tries == 50
        ):
            break
        else:
            tries += 1

    return job_details


def remove_job(client, uuid):
    """
    Removes the cluster with the passed in uuid. Note that it only sends a
    deletion request, but doesn't wait for it to finish.
    """
    client.job.remove(uuid)


def create_source(path=None, branch=""):
    """
    Creates a source object representing where the job script is located and
    adds in required params for required for creation of the source, if any.
    """
    if path:
        return WorkspaceSource(path=path)
    else:
        return GitRepoSource(
            repo_url=os.environ["GITHUB_REPO"],
            username=os.environ["GITHUB_USER"],
            token=os.environ["GITHUB_TOKEN"],
            reference=branch,
        )


def create_instance_role_def():
    """
    Creates and returns an instance role object with the aws arn of the role required to access various input
    data buckets used in the testing process
    """
    role_definition = CreateRoleDefinition(
        name="e2e-test-role",
        description="Instance role for e2e image testing",
        data=InstanceRole(
            role_arn="arn:aws:iam::427443013497:role/e2e-image-test-role"
        ),
    )
    return role_definition
