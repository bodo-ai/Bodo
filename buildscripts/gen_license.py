import array
import datetime
import subprocess

# See bodo/libs/_distributed.cpp for more information on license

# The private key can be generated like this:
# > openssl genpkey -algorithm RSA -out license_private.key -pkeyopt rsa_keygen_bits:2048
#
# The public key can be obtained from the private key like this:
# > openssl rsa -pubout -in license_private.key -out license_public.key
#
# NOTE that the public key corresponding to the private key must be embedded in
# the code in _distributed.cpp

REGULAR_LIC_TYPE = 0
PLATFORM_LIC_TYPE_AWS = 1
PLATFORM_LIC_TYPE_AZURE = 2
# AWS instance ID is 19 characters:
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/resource-ids.html
EC2_INSTANCE_ID_LEN = 19
# Azure VM Unique ID is 36 characters:
# https://azure.microsoft.com/en-us/blog/accessing-and-using-azure-vm-unique-id/
AZURE_INSTANCE_ID_LEN = 36


def generate_license(max_cores, year, month, day, private_key_path):
    # license consists of max_cores, year, month, day as C integers
    msg = array.array("i", [max_cores, year, month, day])

    # sign the license with openssl, get the signature back
    sign_cmd = ["openssl", "dgst", "-sha256", "-sign", private_key_path]
    signature = subprocess.check_output(sign_cmd, input=msg.tobytes())

    # put total size of everything in bytes at start of license
    msg.insert(0, 0)
    total_len = (msg.itemsize * len(msg)) + len(signature)
    msg[0] = (REGULAR_LIC_TYPE << 16) + total_len
    # append signature to license content
    msg = msg.tobytes() + signature

    # encode the whole license content to Base64 (ASCII)
    # we use -A flag so the file doesn't have line breaks (makes it easier to
    # put in an environment variable)
    b64_encode_cmd = ["openssl", "base64", "-A", "-out", "bodo.lic"]
    subprocess.run(b64_encode_cmd, input=msg)


def generate_license_platform(cloud_type, instance_id, private_key_path):
    # license consists of the instance ID
    if isinstance(instance_id, str):
        instance_id = instance_id.encode()  # convert to bytes
    assert isinstance(instance_id, bytes)
    if cloud_type == "aws":
        assert len(instance_id) == EC2_INSTANCE_ID_LEN, "Unexpected instance ID length"
        PLATFORM_LIC_TYPE = PLATFORM_LIC_TYPE_AWS
    elif cloud_type == "azure":
        assert (
            len(instance_id) == AZURE_INSTANCE_ID_LEN
        ), "Unexpected instance ID length"
        PLATFORM_LIC_TYPE = PLATFORM_LIC_TYPE_AZURE
    else:
        raise ValueError(f"Unrecognized cloud type {cloud_type}")

    # sign the license with openssl, get the signature back
    sign_cmd = ["openssl", "dgst", "-sha256", "-sign", private_key_path]
    signature = subprocess.check_output(sign_cmd, input=instance_id)

    # put total size of everything (in bytes) at start of license
    header = array.array("i", [0])
    total_len = header.itemsize + len(instance_id) + len(signature)
    header[0] = (PLATFORM_LIC_TYPE << 16) + total_len
    msg = header.tobytes() + instance_id + signature

    # encode the whole license content to Base64 (ASCII)
    # we use -A flag so the file doesn't have line breaks (makes it easier to
    # put in an environment variable)
    b64_encode_cmd = ["openssl", "base64", "-A", "-out", "bodo.lic"]
    subprocess.run(b64_encode_cmd, input=msg)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser(description="Generate Bodo license file")
    p.add_argument(
        "--ec2-instance-id",
        required=False,
        help="Generate a license for this EC2 instance",
    )
    p.add_argument(
        "--azure-instance-id",
        required=False,
        help="Generate a license for this Azure instance",
    )
    p.add_argument(
        "--max-cores", required=False, metavar="count", help="Max cores for license"
    )
    p.add_argument(
        "--expires",
        required=False,
        metavar="date",
        help="Expiration date in YYYY-MM-DD",
    )
    p.add_argument(
        "--trial-days",
        required=False,
        metavar="days",
        help="Set expiration num days from today",
    )
    p.add_argument(
        "--private-key",
        required=True,
        metavar="path",
        help="Path to private key to sign license",
    )
    args = p.parse_args()
    if args.ec2_instance_id is not None:
        generate_license_platform("aws", args.ec2_instance_id, args.private_key)
        print("Generated license for EC2 Instance ID", args.ec2_instance_id)
    elif args.azure_instance_id is not None:
        generate_license_platform("azure", args.azure_instance_id, args.private_key)
        print("Generated license for Azure Instance ID", args.azure_instance_id)
    else:
        if args.expires is not None:
            year, month, day = [int(val) for val in args.expires.split("-")]
        elif args.trial_days is not None:
            today = datetime.date.today()
            expiration_date = today + datetime.timedelta(days=int(args.trial_days))
            year = expiration_date.year
            month = expiration_date.month
            day = expiration_date.day
        else:
            print("Need to provide expiration date or trial days")
            p.print_usage()
            exit(1)

        generate_license(int(args.max_cores), year, month, day, args.private_key)

        print(
            "Generated license file for",
            args.max_cores,
            "cores, expiring on {}-{}-{}".format(year, month, day),
        )
