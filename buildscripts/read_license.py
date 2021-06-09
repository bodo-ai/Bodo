import array
import subprocess
import sys

# See gen_license.py and bodo/libs/_distributed.cpp for more information on license

REGULAR_LIC_TYPE = 0
PLATFORM_LIC_TYPE_AWS = 1
PLATFORM_LIC_TYPE_AZURE = 2
HEADER_LEN_MASK = 0xFFFF  # license file length is in second half of header
# AWS instance id is 19 characters:
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/resource-ids.html
# Azure VM Unique ID is 36 characters:
# https://azure.microsoft.com/en-us/blog/accessing-and-using-azure-vm-unique-id/
instance_id_len = {PLATFORM_LIC_TYPE_AWS: 19, PLATFORM_LIC_TYPE_AZURE: 36}
cloud_type = {PLATFORM_LIC_TYPE_AWS: "AWS", PLATFORM_LIC_TYPE_AZURE: "Azure"}


def read_license(license_fname):
    """Return license info"""
    b64_decode_cmd = ["openssl", "base64", "-A", "-d", "-in", license_fname]
    msg = subprocess.check_output(b64_decode_cmd)

    header_arr = array.array("i", msg[:4])  # read header as integer
    header = header_arr[0]
    total_len = header & HEADER_LEN_MASK
    license_type = header >> 16

    if license_type == REGULAR_LIC_TYPE:
        max_cores, year, month, day = array.array("i", msg)[1:5]
        print(
            "License for", max_cores, "cores. Expires {}-{}-{}".format(year, month, day)
        )
    elif license_type in {PLATFORM_LIC_TYPE_AWS, PLATFORM_LIC_TYPE_AZURE}:
        license_id = msg[
            header_arr.itemsize : header_arr.itemsize + instance_id_len[license_type]
        ]
        print(f"License for {cloud_type[license_type]} instance", license_id.decode())
    else:
        print("License type {} not recognized".format(license_type))


read_license(sys.argv[1])
