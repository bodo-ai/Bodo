import os
import random
import subprocess
import time

import boto3
import pytest

from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.utils import temp_env_override
from bodo.utils.utils import run_rank0

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

    @run_rank0
    def create():
        # Check if Polaris server is already running
        try:
            subprocess.run(
                ["docker", "inspect", "polaris-server-unittests"], check=True
            )
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
        # Use boto to get credentials from all possible sources
        session = boto3.Session()
        credentials = session.get_credentials()
        env_args = [
            "-e",
            "quarkus.otel.sdk.disabled=true",
            "-e",
            "POLARIS_BOOTSTRAP_CREDENTIALS=default-realm,root,s3cr3t",
            "-e",
            "polaris.realm-context.realms=default-realm",
            "-e",
            f"AWS_REGION={session.region_name if session.region_name else 'us-east-2'}",
        ]
        if credentials.access_key is not None:
            env_args += ["-e", f"AWS_ACCESS_KEY_ID={credentials.access_key}"]
        if credentials.secret_key is not None:
            env_args += [
                "-e",
                f"AWS_SECRET_ACCESS_KEY={credentials.secret_key}",
            ]
        if credentials.token is not None:
            env_args += ["-e", f"AWS_SESSION_TOKEN={credentials.token}"]
        # Add Azure credentials
        if "AZURE_CLIENT_ID" in os.environ:
            env_args += ["-e", f"AZURE_CLIENT_ID={os.environ['AZURE_CLIENT_ID']}"]
        if "AZURE_CLIENT_SECRET" in os.environ:
            env_args += [
                "-e",
                f"AZURE_CLIENT_SECRET={os.environ['AZURE_CLIENT_SECRET']}",
            ]
        if "AZURE_TENANT_ID" in os.environ:
            env_args += [
                "-e",
                f"AZURE_TENANT_ID={os.environ.get('AZURE_TENANT_ID', '72f988bf-86f1-41af-91ab-2d7cd011db47')}",
            ]
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

    create()
    yield "localhost", 8181, "root", "s3cr3t"

    @run_rank0
    def cleanup():
        # Stop Polaris server
        subprocess.run(["docker", "stop", "polaris-server-unittests"], check=True)
        subprocess.run(["docker", "rm", "polaris-server-unittests"], check=True)

    cleanup()


@pytest.fixture(scope="session")
def polaris_package():
    """
    Ensure that the polaris client is installed
    """

    # Install the polaris client if it is not already installed
    # This is a temporary solution until the polaris client is published
    # https://github.com/apache/polaris/issues/19
    @run_rank0
    def ensure_polaris_client():
        try:
            subprocess.run(["pip", "show", "polaris.management"], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(
                [
                    "pip",
                    "install",
                    "git+https://github.com/apache/polaris.git#subdirectory=regtests/client/python",
                ]
            )

    ensure_polaris_client()


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

    @run_rank0
    def get_token():
        _, _, user, password = polaris_server
        client = CatalogApiClient(
            CatalogApiClientConfiguration(
                username=user,
                password=password,
                host=f"http://{host}:{port}/api/catalog",
            )
        )
        oauth_api = IcebergOAuth2API(client)
        token = oauth_api.get_token(
            scope="PRINCIPAL_ROLE:ALL",
            client_id=user,
            client_secret=password,
            grant_type="client_credentials",
            _headers={"realm": "default-realm"},
        )
        return token

    return get_token()


@pytest.fixture(scope="session")
def aws_polaris_warehouse(polaris_token, polaris_server, polaris_package):
    """
    Configure an S3 warehouse in the polaris server
    """
    from polaris.catalog import ApiClient as CatalogApiClient
    from polaris.catalog import CreateNamespaceRequest, IcebergCatalogAPI
    from polaris.management import (
        AddGrantRequest,
        AwsStorageConfigInfo,
        Catalog,
        CatalogGrant,
        CatalogPrivilege,
        Configuration,
        CreateCatalogRequest,
        PolarisDefaultApi,
    )
    from polaris.management import (
        ApiClient as ManagementApiClient,
    )

    host, port, _, _ = polaris_server

    @run_rank0
    def create_aws_warehouse():
        management_client = ManagementApiClient(
            Configuration(
                access_token=polaris_token.access_token,
                host=f"http://{host}:{port}/api/management/v1",
            )
        )
        root_client = PolarisDefaultApi(management_client)
        storage_conf = AwsStorageConfigInfo(
            role_arn="arn:aws:iam::427443013497:role/Polaris-Unittests",
            storage_type="S3",
            region="us-east-2",
        )
        catalog_name = "aws-polaris-warehouse"
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
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_CONTENT
                )
            ),
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
        catalog_api.create_namespace(
            prefix=catalog_name,
            create_namespace_request=CreateNamespaceRequest(namespace=["default"]),
        )

        return catalog_name

    return create_aws_warehouse()


@pytest.fixture(scope="session")
def azure_polaris_warehouse(polaris_token, polaris_server, polaris_package):
    """
    Configure an Azure warehouse in the polaris server
    """
    from polaris.catalog import ApiClient as CatalogApiClient
    from polaris.catalog import CreateNamespaceRequest, IcebergCatalogAPI
    from polaris.management import (
        AddGrantRequest,
        AzureStorageConfigInfo,
        Catalog,
        CatalogGrant,
        CatalogPrivilege,
        Configuration,
        CreateCatalogRequest,
        PolarisDefaultApi,
    )
    from polaris.management import (
        ApiClient as ManagementApiClient,
    )

    host, port, _, _ = polaris_server

    @run_rank0
    def create_azure_warehouse():
        management_client = ManagementApiClient(
            Configuration(
                access_token=polaris_token.access_token,
                host=f"http://{host}:{port}/api/management/v1",
            )
        )
        root_client = PolarisDefaultApi(management_client)
        storage_conf = AzureStorageConfigInfo(
            storage_type="AZURE",
            tenant_id=os.environ.get(
                "AZURE_TENANT_ID", "ac373ae0-dc77-4cbb-bbb7-deddcf6133b3"
            ),
            multiTenantAppName="",
        )

        catalog_name = "azure-polaris-warehouse"
        suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))
        catalog = Catalog(
            name=catalog_name,
            type="INTERNAL",
            properties={
                "default-base-location": f"abfs://polaris-unittests@{os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/{suffix}"
            },
            storage_config_info=storage_conf,
        )

        root_client.create_catalog(
            create_catalog_request=CreateCatalogRequest(catalog=catalog)
        )
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_CONTENT
                )
            ),
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
        catalog_api.create_namespace(
            prefix=catalog_name,
            create_namespace_request=CreateNamespaceRequest(namespace=["default"]),
        )

        return catalog_name

    return create_azure_warehouse()


@pytest.fixture(params=["aws-polaris-warehouse", "azure-polaris-warehouse"])
def polaris_connection(
    request, polaris_server, aws_polaris_warehouse, azure_polaris_warehouse
):
    """
    Fixture to create a connection to the polaris warehouse.
    Returns the catalog url, warehouse name, and credential.
    """
    assert request.node.get_closest_marker("polaris") is not None, (
        "polaris marker not set"
    )
    host, port, user, password = polaris_server
    url = f"http://{host}:{port}/api/catalog"
    if request.param == "aws-polaris-warehouse":
        # Unset the AWS credentials to avoid using them
        # to confirm that the tests are getting aws credentials from polaris
        with temp_env_override(
            {
                "AWS_ACCESS_KEY_ID": None,
                "AWS_SECRET_ACCESS_KEY": None,
                "AWS_SESSION_TOKEN": None,
            }
        ):
            yield url, aws_polaris_warehouse, f"{user}:{password}"
    elif request.param == "azure-polaris-warehouse":
        # Unset the Azure credentials to avoid using them
        # to confirm that the tests are getting azure credentials from polaris
        with temp_env_override(
            {
                # Pyiceberg doesn't support azure vended credentials, it will in 0.9
                #        "AZURE_STORAGE_ACCOUNT_NAME": None,
                #        "AZURE_STORAGE_ACCOUNT_KEY": None,
                #        "AZURE_CLIENT_ID": None,
                #        "AZURE_CLIENT_SECRET": None,
                #        "AZURE_TENANT_ID": None
                f"PYICEBERG_CATALOG__{azure_polaris_warehouse}__ADLS__ACCOUNT_NAME": os.environ.get(
                    "AZURE_STORAGE_ACCOUNT_NAME", "bodosficebergazue2"
                ),
                f"PYICEBERG_CATALOG__{azure_polaris_warehouse}__ADLS__ACCOUNT_KEY": os.environ.get(
                    "AZURE_STORAGE_ACCOUNT_KEY"
                ),
                f"PYICEBERG_CATALOG__{azure_polaris_warehouse}__ADLS__CLIENT_ID": os.environ.get(
                    "AZURE_CLIENT_ID"
                ),
                f"PYICEBERG_CATALOG__{azure_polaris_warehouse}__ADLS__CLIENT_SECRET": os.environ.get(
                    "AZURE_CLIENT_SECRET"
                ),
                f"PYICEBERG_CATALOG__{azure_polaris_warehouse}__ADLS__TENANT_ID": os.environ.get(
                    "AZURE_TENANT_ID"
                ),
            }
        ):
            yield url, azure_polaris_warehouse, f"{user}:{password}"
    else:
        raise ValueError(f"Unknown polaris warehouse: {request.param}")
