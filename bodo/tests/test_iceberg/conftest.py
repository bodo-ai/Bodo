import os
import random
import subprocess
import time

import boto3
import pytest
from mpi4py import MPI

import bodo
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

    If it fails to start locally even with docker running try
    enabling "Allow the default Docker socket to be used" in
    advanced settings of Docker Desktop.
    """
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    # Can't use run_rank0 because containers aren't pickelable
    err = None
    if bodo.get_rank() == 0:
        try:
            # Use boto to get credentials from all possible sources
            session = boto3.Session()
            credentials = session.get_credentials()
            env = {
                "quarkus.otel.sdk.disabled": "true",
                "POLARIS_BOOTSTRAP_CREDENTIALS": "default-realm,root,s3cr3t",
                "polaris.realm-context.realms": "default-realm",
                "AWS_REGION": "us-east-2",
            }
            if credentials.access_key is not None:
                env["AWS_ACCESS_KEY_ID"] = credentials.access_key
            if credentials.secret_key is not None:
                env["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
            if credentials.token is not None:
                env["AWS_SESSION_TOKEN"] = credentials.token
            # Add Azure credentials
            if "AZURE_CLIENT_ID" in os.environ:
                env["AZURE_CLIENT_ID"] = os.environ["AZURE_CLIENT_ID"]
            if "AZURE_CLIENT_SECRET" in os.environ:
                env["AZURE_CLIENT_SECRET"] = os.environ["AZURE_CLIENT_SECRET"]
            env["AZURE_TENANT_ID"] = os.environ.get(
                "AZURE_TENANT_ID", "ac373ae0-dc77-4cbb-bbb7-deddcf6133b3"
            )

            n_retries = 5
            for i in range(n_retries):
                try:
                    polaris = (
                        DockerContainer(
                            "public.ecr.aws/k7f6m2y1/bodo/polaris-unittests:latest"
                        )
                        .with_bind_ports(8181, 8181)
                        .with_bind_ports(8282, 8182)
                        .with_name("polaris-server-unittests")
                    )
                    for key, value in env.items():
                        polaris.with_env(key, value)
                    wait_for_logs(polaris.start(), "Listening on")
                    time.sleep(2**i)
                    break
                except Exception as e:
                    if i == n_retries - 1:
                        raise e
                    continue
        except Exception as e:
            err = e
    err = MPI.COMM_WORLD.bcast(err, root=0)
    if err is not None:
        raise err

    yield "localhost", 8181, "root", "s3cr3t"
    if bodo.get_rank() == 0:
        try:
            polaris.stop()
        except Exception as e:
            err = e

    err = MPI.COMM_WORLD.bcast(err, root=0)
    if err is not None:
        raise err


@pytest.fixture(scope="session")
def polaris_package():
    """
    Ensure that the polaris client is installed
    """
    from bodo.spawn.utils import run_rank0

    # Install the polaris client if it is not already installed
    # This is a temporary solution until the polaris client is published
    # https://github.com/apache/polaris/issues/19
    @run_rank0
    def ensure_polaris_client():
        try:
            subprocess.run(["pip", "show", "polaris"], check=True)
        except subprocess.CalledProcessError:
            raise ValueError(
                "Polaris client is not installed. Please install it manually."
                "You can install it by running: pip install git+https://github.com/apache/polaris.git@release/1.0.x#subdirectory=client/python",
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

    from bodo.spawn.utils import run_rank0

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

    from bodo.spawn.utils import run_rank0

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
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_ACCESS
                )
            ),
        )
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_METADATA
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

    from bodo.spawn.utils import run_rank0

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
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_ACCESS
                )
            ),
        )
        root_client.add_grant_to_catalog_role(
            catalog_name,
            "catalog_admin",
            AddGrantRequest(
                grant=CatalogGrant(
                    type="catalog", privilege=CatalogPrivilege.CATALOG_MANAGE_METADATA
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
        yield url, aws_polaris_warehouse, f"{user}:{password}"
    elif request.param == "azure-polaris-warehouse":
        # Unset the Azure credentials to avoid using them
        # to confirm that the tests are getting azure credentials from polaris
        with temp_env_override(
            {
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


# For cases where we can't used parameterized fixuteres like the ddl test harness
@pytest.fixture
def aws_polaris_connection(polaris_server, aws_polaris_warehouse):
    host, port, user, password = polaris_server
    url = f"http://{host}:{port}/api/catalog"
    with temp_env_override(
        {
            "AWS_ACCESS_KEY_ID": None,
            "AWS_SECRET_ACCESS_KEY": None,
            "AWS_SESSION_TOKEN": None,
        }
    ):
        yield url, aws_polaris_warehouse, f"{user}:{password}"
