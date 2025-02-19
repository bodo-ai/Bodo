import os
import random
import subprocess
import time
from importlib import util as importlib_util

import boto3
import pytest

from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.utils import temp_env_override

WRITE_TABLES = [
    "BOOL_BINARY_TABLE",
    "DT_TSZ_TABLE",
    "TZ_AWARE_TABLE",
    "DTYPE_LIST_TABLE",
    "NUMERIC_TABLE",
    "STRING_TABLE",
    "LIST_TABLE",
    "STRUCT_TABLE",
    "OPTIONAL_TABLE",
    # TODO Needs investigation.
    pytest.param(
        "MAP_TABLE",
        marks=pytest.mark.skip(
            reason="Results in runtime error that's consistent with to_parquet."
        ),
    ),
    pytest.param(
        "DECIMALS_TABLE",
        marks=pytest.mark.skip(
            reason="We don't support custom precisions and scale at the moment."
        ),
    ),
    pytest.param(
        "DECIMALS_LIST_TABLE",
        marks=pytest.mark.skip(
            reason="We don't support custom precisions and scale at the moment."
        ),
    ),
    "DICT_ENCODED_STRING_TABLE",
]


@pytest.fixture(params=WRITE_TABLES)
def simple_dataframe(request):
    return (
        request.param,
        f"SIMPLE_{request.param}",
        SIMPLE_TABLES_MAP[f"SIMPLE_{request.param}"][0],
    )


@pytest.fixture(scope="session")
def polaris_server():
    """
    Install Polaris if it is not already installed.
    Start Polaris server if it is not already running.
    Returns the host port, and credentials of the server.
    """
    # Check if Polaris server is already running
    try:
        subprocess.run(["docker", "inspect", "polaris-server-unittests"], check=True)
        # Kill it if it is running
        subprocess.run(["docker", "stop", "polaris-server-unittests"], check=True)
        subprocess.run(["docker", "rm", "polaris-server-unittests"], check=True)
    except subprocess.CalledProcessError:
        # Polaris server is not running, ignore the error
        pass

    health_check_args = [
        "--health-cmd",
        "curl http://localhost:8182/healthcheck",
        "--health-interval",
        "2s",
        "--health-retries",
        "5",
        "--health-timeout",
        "10s",
    ]
    env_args = [
        "-e",
        "quarkus.otel.sdk.disabled=true",
        "-e",
        "POLARIS_BOOTSTRAP_CREDENTIALS=default-realm,root,s3cr3t",
        "-e",
        "polaris.realm-context.realms=default-realm",
        "-e",
        f"AWS_REGION={os.environ.get('AWS_REGION', 'us-east-2')}",
    ]
    # Use boto to get credentials from all possible sources
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials.access_key is not None:
        env_args += ["-e", f"AWS_ACCESS_KEY_ID={credentials.access_key}"]
    if credentials.secret_key is not None:
        env_args += [
            "-e",
            f"AWS_SECRET_ACCESS_KEY={credentials.secret_key}",
        ]
    if credentials.token is not None:
        env_args += ["-e", f"AWS_SESSION_TOKEN={credentials.token}"]

    # Start Polaris server
    # Once Polaris publishes their own docker image, we can use that instead of ours
    # https://github.com/apache/polaris/issues/152
    subprocess.run(
        ["docker", "run", "-d"]
        + health_check_args
        + env_args
        + [
            "-p",
            "8181:8181",
            "-p",
            "8282:8282",
            "--name",
            "polaris-server-unittests",
            "public.ecr.aws/k7f6m2y1/bodo/polaris-unittests:latest",
        ],
        check=True,
    )
    # Wait for Polaris server to start
    while (
        "healthy"
        not in subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=polaris-server-unittests",
                "--format",
                "{{.Status}}",
            ],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .lower()
    ):
        time.sleep(1)
    yield "localhost", 8181, "root", "s3cr3t"

    # Stop Polaris server
    subprocess.run(["docker", "stop", "polaris-server-unittests"], check=True)
    subprocess.run(["docker", "rm", "polaris-server-unittests"], check=True)


@pytest.fixture(scope="session")
def polaris_package():
    """
    Ensure that the polaris client is installed
    """
    # Install the polaris client if it is not already installed
    # This is a temporary solution until the polaris client is published
    # https://github.com/apache/polaris/issues/19
    if importlib_util.find_spec("polaris") is None:
        subprocess.run(["python", "-m", "ensurepip"])
        subprocess.run(
            [
                "pip",
                "install",
                "git+https://github.com/apache/polaris.git#subdirectory=regtests/client/python",
            ]
        )


@pytest.fixture(scope="session")
def polaris_token(polaris_server, polaris_package):
    """
    Fixture to get a polaris access token
    """
    from polaris.catalog.api.iceberg_o_auth2_api import IcebergOAuth2API
    from polaris.catalog.api_client import ApiClient as CatalogApiClient
    from polaris.catalog.api_client import (
        Configuration as CatalogApiClientConfiguration,
    )

    host, port, user, password = polaris_server

    client = CatalogApiClient(
        CatalogApiClientConfiguration(
            username=user, password=password, host=f"http://{host}:{port}/api/catalog"
        )
    )
    _, _, user, password = polaris_server
    oauth_api = IcebergOAuth2API(client)
    token = oauth_api.get_token(
        scope="PRINCIPAL_ROLE:ALL",
        client_id=user,
        client_secret=password,
        grant_type="client_credentials",
        _headers={"realm": "default-realm"},
    )
    return token


@pytest.fixture(scope="session")
def aws_polaris_warehouse(polaris_token, polaris_server, polaris_package):
    """
    Configure an S3 warehouse in the polaris server
    """
    from polaris.catalog import ApiClient as CatalogApiClient
    from polaris.catalog import CreateNamespaceRequest, IcebergCatalogAPI
    from polaris.management import (
        ApiClient as ManagementApiClient,
    )
    from polaris.management import (
        AwsStorageConfigInfo,
        Catalog,
        Configuration,
        CreateCatalogRequest,
        PolarisDefaultApi,
    )

    host, port, _, _ = polaris_server

    management_client = ManagementApiClient(
        Configuration(
            access_token=polaris_token.access_token,
            host=f"http://{host}:{port}/api/management/v1",
        )
    )
    root_client = PolarisDefaultApi(management_client)
    storage_conf = AwsStorageConfigInfo(
        role_arn="arn:aws:iam::427443013497:role/Polaris-Unittests", storage_type="S3"
    )
    catalog_name = "aws_polaris_warehouse"
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))
    catalog = Catalog(
        name=catalog_name,
        type="INTERNAL",
        properties={"default-base-location": f"s3://polaris-unittests/{suffix}"},
        storage_config_info=storage_conf,
    )

    root_client.create_catalog(
        create_catalog_request=CreateCatalogRequest(catalog=catalog)
    )

    catalog_client = CatalogApiClient(
        Configuration(
            access_token=polaris_token.access_token,
            host=f"http://{host}:{port}/api/catalog",
        )
    )
    catalog_api = IcebergCatalogAPI(catalog_client)
    catalog_api.create_namespace(
        prefix=catalog_name,
        create_namespace_request=CreateNamespaceRequest(namespace=["CI"]),
    )

    return catalog_name


@pytest.fixture
def polaris_connection(request, polaris_server, aws_polaris_warehouse):
    """
    Fixture to create a connection to the polaris warehouse.
    Returns the catalog url, warehouse name, and credential.
    """
    assert request.node.get_closest_marker("polaris") is not None, (
        "polaris marker not set"
    )
    # Unset the AWS credentials to avoid using them
    # to confirm that the tests are getting aws credentials from polaris
    with temp_env_override(
        {
            "AWS_ACCESS_KEY_ID": None,
            "AWS_SECRET_ACCESS_KEY": None,
            "AWS_SESSION_TOKEN": None,
        }
    ):
        host, port, user, password = polaris_server
        url = f"http://{host}:{port}/api/catalog"
        yield url, aws_polaris_warehouse, f"{user}:{password}"
