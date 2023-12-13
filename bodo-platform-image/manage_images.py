"""
What does this new script do? 
- Create and then manage image metadata to the Database
- Manage can either be adding entry to DB or marking an existing entry in DB to unsupported

What does it create? 
- Reads image metadata for every region in azure and aws created as json to be shared
- Contains information about image name/version, image id, region and cloud provider

How does the script work?
Has 4 modes
- Read from JSON and share: read image metadata from build_and_publish step and share with database
- Read from ENV and share: Incase no metadata, reads ENV variables and constructs new json image data
- Read from ENV and deregister: Reads env variables, constructs new json image data and sends request to mark
                                the database entry for the image version to False
- Update Image name or mark an image as unsupported from CI Pipeline.
"""
import argparse
import json
import logging
import os

import requests

from scripts.common import login


def manage_image(header, be_url, img_data, share=False):
    # The DB is configured to default to null. Since we dont use Pipes for boolean conversion on this endpoint
    # Using None is safer since it gets automarked as false.
    backend_endpoint = os.path.join(be_url, "api", "image", "bodoImage")
    try:
        if share:
            response = requests.post(backend_endpoint, json=img_data, headers=header)
        else:
            response = requests.put(backend_endpoint, json=img_data, headers=header)
    except requests.exceptions.RequestException as e:
        print("bodoImage call to the backend service failed")
        print("Response received from backend: " + str(e))
        raise e

    print("Successfully triggered backend service...")
    print("Response received from backend: " + str(response))


def create_from_env(bodo_version, image_region, image_id, cloud_provider):
    return [{
        "bodoVersion": bodo_version,
        "supported": "true",
        "imagePrefix": "null",
        "imageGitSha": "null",
        "region": image_region,
        "workerImageId": image_id,
        "imageCloudProvider": (cloud_provider.upper())
    }]


def create_from_json():
    with open('img_share_requests.json', 'r') as f:
        img_data = json.load(f)
    return img_data


def create_for_deregister(bodo_version, cloud_provider):
    return {
        'bodoVersion': bodo_version,
        'imagePrefix': "null",
        'supported': None,
        'imageCloudProvider': cloud_provider
    }


def update_image_entry(bodo_version, updated_bodo_version, supported):
    if not supported:
        return {
            'bodoVersion': bodo_version,
            'supported': supported
        }

    return {
        'bodoVersion': bodo_version,
        'updatedBodoVersion': updated_bodo_version
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_from_env", action="store_true", default=False)
    parser.add_argument("--deregister", action="store_true", default=False)
    args = parser.parse_args()

    assert "AUTH_SERVICE_URL" in os.environ
    assert "BOT_PLATFORM_USERNAME" in os.environ
    assert "BOT_PLATFORM_PASSWORD" in os.environ
    assert "BACKEND_SERVICE_URL" in os.environ

    logging.info(f"Log into the platform using {os.environ["BOT_PLATFORM_USERNAME"]} account and get an access token...")
    try:
        access_token = login(os.environ["AUTH_SERVICE_URL"], os.environ["BOT_PLATFORM_USERNAME"],
                             os.environ["BOT_PLATFORM_PASSWORD"])
    except Exception as e:
        logging.info("Login Failed")
        logging.info("Received Response: " + str(e))
        exit(1)

    logging.info("Login successful...")

    logging.info("Triggering bodoImage endpoint on backend with the access token and image entries...")
    header = {'Authorization': 'Bearer ' + access_token}

    bodo_version = os.environ.get('IMAGE_NAME', None)
    cloud_provider = os.environ.get('CLOUD_PROVIDER', None)
    be_url = os.environ.get("BACKEND_SERVICE_URL", None)
    image_region = os.environ.get("IMAGE_REGION", None)
    image_id = os.environ.get("IMAGE_ID", None)
    img_data = None

    update = os.environ.get("UPDATE_IMAGE", "false") == "true"
    updated_bodo_version = os.environ.get("UPDATED_IMAGE_NAME", None)

    logging.info("Update Image: ", update)

    # Create the data to be shared
    if args.deregister or update:
        if update:
            is_supported = False if args.deregister else True
            # Based on args.deregister, we only update the image name or make supported as False
            img_data = update_image_entry(bodo_version, updated_bodo_version, is_supported)
        else:
            img_data = create_for_deregister(bodo_version, cloud_provider)

        # To update image or de-share an image, we will be using the same endpoint
        manage_image(header, be_url, img_data)
    else:
        # Adding new images to DB.
        if args.generate_from_env:
            img_data = create_from_env(bodo_version, image_region, image_id, cloud_provider)
        else:
            img_data = create_from_json()

        # Sending request to backend to add entries to the database
        manage_image(header, be_url, img_data, share=True)
