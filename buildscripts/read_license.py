import sys
import subprocess
import array


def read_license(license_fname):
    """ Return license info (max cores and expiration date) from the given
        license """
    b64_decode_cmd = ["openssl", "base64", "-A", "-d", "-in", license_fname]
    msg = subprocess.check_output(b64_decode_cmd)
    max_cores, year, month, day = array.array('i', msg)[1:5]
    print("License for", max_cores, "cores. Expires {}-{}-{}".format(year, month, day))


read_license(sys.argv[1])
