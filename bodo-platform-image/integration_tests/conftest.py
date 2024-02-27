import os
import pytest
import logging
import uuid
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from integration_tests.common import (
    create_cluster_helper,
    remove_cluster,
    create_instance_role_def,
)


@pytest.fixture(scope="session")
def bodo_client():
    """
    Initializes a bodo-client BodoSDK) to manage clusters and submit jobs.
    """
    try:
        logging.info("Getting bodo client")
        assert "WORKSPACE_CLIENT_ID" in os.environ
        assert "WORKSPACE_SECRET_KEY" in os.environ
        assert "BACKEND_API_URL" in os.environ
        assert "BACKEND_AUTH_URL" in os.environ
        keys = WorkspaceKeys(
            client_id=os.environ["WORKSPACE_CLIENT_ID"],
            secret_key=os.environ["WORKSPACE_SECRET_KEY"],
        )
        return get_bodo_client(
            keys,
            api_url=os.environ["BACKEND_API_URL"],
            auth_url=os.environ["BACKEND_AUTH_URL"],
        )
    except Exception as e:
        logging.error(f"Exception while initializing bodo-client: {str(e)}")
        raise e


@pytest.fixture(scope="session")
def create_cluster(
    bodo_client,
    img_id,
    bodo_version,
    default_cluster_name,
    default_cluster_type,
    cluster_description,
):
    """
    Calls the helper function to create a cluster based on the passed in image_id and bodo_version and other metadata
    """
    cluster_metadata = create_cluster_helper(
        bodo_client,
        bodo_version,
        img_id,
        default_cluster_name,
        default_cluster_type,
        cluster_description,
    )
    try:
        yield cluster_metadata.uuid
    finally:
        # Delete the cluster to avoid lingering cloud resources.
        remove_cluster(bodo_client, cluster_metadata.uuid)


@pytest.fixture(scope="session")
def img_id():
    """
    Return the worker image ami id specified as input to the workflow
    """
    return os.environ["IMAGE_ID"]


@pytest.fixture(scope="session")
def bodo_version():
    """
    Return the image name/version specified as input to the workflow
    """
    return os.environ["IMAGE_NAME"]


@pytest.fixture(scope="session")
def cluster_description():
    """
    Generate a generic description for cluster creation.
    """
    return "cluster created for e2e image test"


@pytest.fixture(scope="session")
def default_cluster_type():
    """
    Generate a default cluster instance type.
    """
    if os.environ["CLOUD_PROVIDER"] == "AWS":
        return "c5.2xlarge"
    else:
        return "Standard_D8as_v4"


@pytest.fixture(scope="session")
def default_cluster_name():
    """
    Generate a generic unique name for the cluster.
    """
    return f"new_cluster_for_e2e_image_test-{str(uuid.uuid4())}"


@pytest.fixture(scope="session")
def create_instance_role(bodo_client):
    """
    Creates an instance role to grant necessary role to assume in order to access the right bucket
    """
    if os.environ["CLOUD_PROVIDER"] == "AWS":
        new_role = create_instance_role_def()
        create_response = bodo_client.instance_role.create(new_role)
        try:
            yield create_response.uuid
        finally:
            # Delete Instance role to avoid any issues
            bodo_client.instance_role.remove(
                create_response.uuid
            )  # cleanup role created
    else:
        yield None


@pytest.fixture(scope="session")
def github_branch():
    return os.environ["SOURCE_BRANCH"]
