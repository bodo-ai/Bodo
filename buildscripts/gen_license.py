import subprocess
import array
import datetime

# The private key can be generated like this:
# > openssl genpkey -algorithm RSA -out license_private.key -pkeyopt rsa_keygen_bits:2048
#
# The public key can be obtained from the private key like this:
# > openssl rsa -pubout -in license_private.key -out license_public.key
#
# NOTE that the public key corresponding to the private key must be embedded in
# the code in _distributed.cpp


def generate_license(max_cores, year, month, day, private_key_path):
    # license consists of max_cores, year, month, day as C integers
    msg = array.array("i", [max_cores, year, month, day])

    # sign the license with openssl, get the signature back
    sign_cmd = ["openssl", "dgst", "-sha256", "-sign", private_key_path]
    signature = subprocess.check_output(sign_cmd, input=msg.tobytes())

    # put total size of everything in bytes at start of license
    msg.insert(0, 0)
    msg[0] = (msg.itemsize * len(msg)) + len(signature)
    # append signature to license content
    msg = msg.tobytes() + signature

    # encode the whole license content to Base64 (ASCII)
    # we use -A flag so the file doesn't have line breaks (makes it easier to
    # put in an environment variable)
    b64_encode_cmd = ["openssl", "base64", "-A", "-out", "bodo.lic"]
    subprocess.run(b64_encode_cmd, input=msg)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser(description="Generate Bodo license file")
    p.add_argument(
        "--max-cores", required=True, metavar="count", help="Max cores for license"
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
