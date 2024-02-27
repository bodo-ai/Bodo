import boto3
import dateutil
from datetime import datetime, timezone, timedelta
import json
import argparse


def is_older_than_x_days(img_details, x):
    creation_date = dateutil.parser.parse(img_details["CreationDate"])
    timedelta_since_creation: timedelta = datetime.now(timezone.utc) - creation_date
    return timedelta_since_creation.days > x


def is_release_image(img_details):
    if "Tags" not in img_details:
        return False
    for tag_dict in img_details["Tags"]:
        if tag_dict["Key"] == "AMISha" and tag_dict["Value"].startswith(
            "20"
        ):  # changing it for the bodo ami repo merge
            return True
    return False


def is_master_image(img_details):
    if "Tags" not in img_details:
        return False
    for tag_dict in img_details["Tags"]:
        if tag_dict["Key"] == "AMISha" and tag_dict["Value"].startswith("master-"):
            return True
    return False


def find_amis_to_remove(region):
    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_images(Owners=["self"])
    images = response["Images"]
    print(f"Total number of Images: {len(images)}")

    # Keep if: release or (master and <1yr old) or (<60days old)
    # => Remove if: !release and !(master and <1yr old) and !(<60days old)

    should_be_removed = (
        lambda img: (not is_release_image(img))
        and (not (is_master_image(img) and not is_older_than_x_days(img, 365)))
        and (is_older_than_x_days(img, 60))
    )

    images_to_remove = list(filter(should_be_removed, images))
    print(f"Number of images to remove: {len(images_to_remove)}")

    return images_to_remove


def deregister_images(region, images_to_remove):
    ec2 = boto3.client("ec2", region_name=region)
    ami_ids = list(map(lambda x: x["ImageId"], images_to_remove))
    print(f"Number of AMIs to deregister: {len(ami_ids)}")

    for ami_id in ami_ids:
        print(f"Deregistering {ami_id}")
        ec2.deregister_image(ImageId=ami_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-2",
        help="Region to find the AMIs in. Default: us-east-2",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="amis_to_remove.json",
        help="Output file. Default: amis_to_remove.json",
    )
    parser.add_argument("--deregister", dest="deregister", action="store_true")
    parser.add_argument("--no-deregister", dest="deregister", action="store_false")
    parser.set_defaults(deregister=False)
    args = parser.parse_args()

    print(f"AWS Region: {args.region}")
    images_to_remove = find_amis_to_remove(args.region)

    print(f"Writing information about images to remove to {args.output}")
    with open(args.output, "w") as f:
        json.dump(images_to_remove, f)

    if args.deregister:
        print("Deregistering images...")
        deregister_images(args.region, images_to_remove)
    else:
        print("Skipping deregistration step...")
