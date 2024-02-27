import os
from datetime import datetime, timedelta
import re
import json
import argparse
import time

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient


def is_older_than_x_days(img_details, x):
    # Azure does not provide creation date of image, this is the current makeshift calculation using Bodo Version
    img_name = img_details.name
    img_bodo_version = re.search("([\d]{4}\.[\d]+)", img_name).group(0)
    creation_date = datetime.strptime(img_bodo_version, "%Y.%m")
    timedelta_since_creation: timedelta = datetime.now() - creation_date
    return timedelta_since_creation.days > x


def is_release_image(img_details):
    if not hasattr(img_details, "tags"):
        return False
    tags = img_details.tags
    if "VMISha" not in tags.keys():
        # Some older images do not have tags
        img_name = img_details.name
        release_version = re.search("(_v\d+)", img_name)
        if release_version:
            return True
        return False
    vmi_sha = tags["VMISha"]
    if vmi_sha.startswith("20"):
        return True
    return False


def is_master_image(img_details):
    if not hasattr(img_details, "tags"):
        return False
    tags = img_details.tags
    if "VMISha" not in tags.keys():
        # Some older images do not have tags
        if "master" in img_details.name:
            return True
        return False
    vmi_sha = tags["VMISha"]
    if vmi_sha.startswith("master-"):
        return True
    return False


def find_vmis_to_remove():
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group_name = os.environ["AZURE_IMAGES_RESOURCE_GROUP"]
    credential = DefaultAzureCredential()
    compute_client = ComputeManagementClient(credential, subscription_id)
    images = compute_client.images.list_by_resource_group(resource_group_name)

    # Keep if: release or (master and <1yr old) or (<7 months)
    # => Remove if: !release and !(master and <1yr old) and !(<7 months)
    images_to_remove = []
    for img in images:
        if (
            not is_release_image(img)
            and not (is_master_image(img) and not is_older_than_x_days(img, 365))
            and is_older_than_x_days(img, 210)
        ):
            images_to_remove.append(img.name)
    return images_to_remove


def delete_gallery_image_versions(
    compute_client, resource_group_name, gallery_name, gallery_image_name
):
    # Gallery Image version is a nested part of gallery image, gallery images cannot be deleted without deleting version first
    gallery_image_versions = (
        compute_client.gallery_image_versions.list_by_gallery_image(
            resource_group_name, gallery_name, gallery_image_name
        )
    )
    for version in gallery_image_versions:
        try:
            compute_client.gallery_image_versions.begin_delete(
                resource_group_name, gallery_name, gallery_image_name, version.name
            )
            # result.wait()
        except Exception as e:
            print(e)


def delete_images(images_to_remove):
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    gallery_name = "BodoImages"
    credential = DefaultAzureCredential()
    compute_client = ComputeManagementClient(credential, subscription_id)
    resource_group_name = os.environ["AZURE_IMAGES_RESOURCE_GROUP"]
    print(f"Number of VMIs to delete: {len(images_to_remove)}")

    for vmi_image_name in images_to_remove:
        print(f"Deleting {vmi_image_name} versions")
        delete_gallery_image_versions(
            compute_client, resource_group_name, gallery_name, vmi_image_name
        )

    # Sleep to ensure all version deletion has completed. Azure does not allow you to delete gallery_images without completely deleting all of its version
    time.sleep(100)

    for vmi_image_name in images_to_remove:
        try:
            print(f"Deleting {vmi_image_name} gallery image")
            compute_client.gallery_images.begin_delete(
                resource_group_name, gallery_name, vmi_image_name
            )
            compute_client.images.begin_delete(resource_group_name, vmi_image_name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="vmis_to_remove.json",
        help="Output file. Default: vmis_to_remove.json",
    )
    parser.add_argument("--delete", dest="delete", action="store_true")
    parser.add_argument("--no-delete", dest="delete", action="store_false")
    parser.set_defaults(delete=False)
    args = parser.parse_args()

    assert "AZURE_SUBSCRIPTION_ID" in os.environ
    assert "AZURE_CLIENT_ID" in os.environ
    assert "AZURE_CLIENT_SECRET" in os.environ
    assert "AZURE_TENANT_ID" in os.environ
    assert "AZURE_IMAGES_RESOURCE_GROUP" in os.environ

    images_to_remove = find_vmis_to_remove()
    print(f"Writing information about images to remove to {args.output}")
    with open(args.output, "w") as f:
        json.dump(images_to_remove, f)

    if args.delete:
        print("Deleting images...")
        delete_images(images_to_remove)
    else:
        print("Skipping deleting step...")
