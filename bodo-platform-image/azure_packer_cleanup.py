import os
import requests
import json
import argparse
import adal
import dateutil
from datetime import datetime, timezone, timedelta
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient


def is_older_than_x_days(group_details, x):
    creation_date = dateutil.parser.parse(group_details["createdTime"])
    timedelta_since_creation: timedelta = datetime.now(timezone.utc) - creation_date
    return timedelta_since_creation.days > x


# Delete identified packer resource groups that are stale
def delete_resource_groups(resource_groups_to_remove):
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    credential = DefaultAzureCredential()
    client = ResourceManagementClient(credential, subscription_id)
    for group in resource_groups_to_remove:
        print(f"Deleting resource group {group}")
        # Begin delete does not force deletion by defauly we need the force_deletion_types to ensure these are deleted:
        # https://github.com/Azure/azure-sdk-for-python/blob/1867fbb98e9bb2a3efd9f8849bd08da5f091e6c8/sdk/resources/azure-mgmt-resource/azure/mgmt/resource/resources/v2021_04_01/operations/_operations.py#L9884
        delete_async_operation = client.resource_groups.begin_delete(
            group,
            force_deletion_types="Microsoft.Compute/virtualMachines,Microsoft.Compute/virtualMachineScaleSets",
        )
        delete_async_operation.wait()


def get_adal_login_token():
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]
    tenant_id = os.environ["AZURE_TENANT_ID"]
    authority_url = "https://login.microsoftonline.com/" + tenant_id
    resource = "https://management.azure.com/"
    context = adal.AuthenticationContext(authority_url)
    token = context.acquire_token_with_client_credentials(
        resource, client_id, client_secret
    )
    return token


# This function filters out stale packer resource groups that are active to be deleted based on age (older than 1 day)
def find_resource_groups_to_remove():
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_groups_to_be_deleted = []
    token = get_adal_login_token()
    headers = {
        "Authorization": "Bearer " + token["accessToken"],
        "Content-Type": "application/json",
    }
    # Unable to use AZURE SDK as it does not provide creation time: https://stackoverflow.com/questions/58707068/deleting-all-resources-in-an-azure-resource-group-with-age-more-than-x-days/58830232#58830232
    url = (
        "https://management.azure.com/subscriptions/"
        + subscription_id
        + "/resourcegroups?api-version=2019-08-01&%24expand=createdTime"
    )

    res = requests.get(url, headers=headers)
    resource_groups = res.json()["value"]
    for resource_group in resource_groups:
        if "pkr-Resource-Group" in resource_group["name"]:
            # Every image build takes 1.5 hours on average.
            # Therefore there should no never be any stale resources older than 2 hours.
            # Just to be safe we delete the old ones we give it 24 hours.
            if is_older_than_x_days(resource_group, 1):
                resource_groups_to_be_deleted.append(resource_group["name"])
    return resource_groups_to_be_deleted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="resource_groups_to_remove.json",
        help="Output file. Default: resource_groups_to_remove.json",
    )
    parser.add_argument("--delete", dest="delete", action="store_true")
    parser.add_argument("--no-delete", dest="delete", action="store_false")
    parser.set_defaults(delete=False)
    args = parser.parse_args()

    assert "AZURE_SUBSCRIPTION_ID" in os.environ
    assert "AZURE_CLIENT_ID" in os.environ
    assert "AZURE_CLIENT_SECRET" in os.environ
    assert "AZURE_TENANT_ID" in os.environ

    resource_groups_to_remove = find_resource_groups_to_remove()
    print(f"Writing information about resource groups to remove to {args.output}")
    with open(args.output, "w") as f:
        json.dump(resource_groups_to_remove, f)

    if args.delete:
        print("Deleting resource groups...")
        delete_resource_groups(resource_groups_to_remove)
    else:
        print("Skipping deleting step...")
