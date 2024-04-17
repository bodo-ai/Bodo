from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from mpi4py import MPI
import boto3
import json
from bodo_platform_utils.config import WORKSPACE_UUID, REGION, CLOUD_PROVIDER, DEFAULT_SECRET_GROUP
from bodo_platform_utils.utils_types import CloudProvider


def _get_ssm_parameter(parameter_name):
    ssm_client = boto3.client("ssm", REGION)
    parameters = ssm_client.get_parameters(Names=[parameter_name], WithDecryption=True)
    params = parameters.get("Parameters", None)
    if params is None:
        raise ValueError("Could not get the data. Please check the name and try again")

    if len(params) == 0:
        raise ValueError("Could not get the data. Please check the name and try again")

    value = params[0].get("Value", None)
    if value is None:
        raise ValueError("No Data Found")

    return json.loads(value)


def _get_keyvault_secret(secret_name, keyvault_name: str = None, is_default=True):
    credentials = DefaultAzureCredential()
    if is_default:
        keyvault_name = f"bodoaikv-{WORKSPACE_UUID[:12]}"

    if not keyvault_name:
        raise ValueError("Provide Azure keyvault name")

    url = f"https://{keyvault_name}.vault.azure.net/"
    secret_client = SecretClient(vault_url=url, credential=credentials)
    secret = secret_client.get_secret(secret_name)
    return json.loads(secret.value)


# This is used for Secrets
def get(name, secret_group=DEFAULT_SECRET_GROUP, _parallel=True):
    """
    :param name: Name of the Secret
    :param secret_group: Name of the Secret Group the secret belongs to
    :param _parallel: Defaults to True
    :return: JSON object containing the data
    """

    if CLOUD_PROVIDER == CloudProvider.AWS:
        parameter_name = f"/bodo/workspaces/{WORKSPACE_UUID}/secret-groups/{secret_group}/{name}"
        if _parallel:
            return _get_ssm_parameter_parallel(parameter_name)

        return _get_ssm_parameter(parameter_name)

    elif CLOUD_PROVIDER == CloudProvider.AZURE:
        if _parallel:
            return _get_keyvault_secret_parallel(name)

        return _get_keyvault_secret(name)
    else:
        raise ValueError(f"Unrecognized Cloud Provider: {CLOUD_PROVIDER}")


def _get_ssm_parameter_parallel(name):
    comm = MPI.COMM_WORLD
    data_or_e = None
    if comm.Get_rank() == 0:
        try:
            data_or_e = _get_ssm_parameter(name)
        except Exception as e:
            data_or_e = e

    data_or_e = comm.bcast(data_or_e)

    if isinstance(data_or_e, Exception):
        raise data_or_e
    data = data_or_e
    return data


def _get_keyvault_secret_parallel(name):
    comm = MPI.COMM_WORLD
    data_or_e = None
    if comm.Get_rank() == 0:
        try:
            data_or_e = _get_keyvault_secret(name)
        except Exception as e:
            data_or_e = e

    data_or_e = comm.bcast(data_or_e)

    if isinstance(data_or_e, Exception):
        raise data_or_e
    data = data_or_e
    return data
