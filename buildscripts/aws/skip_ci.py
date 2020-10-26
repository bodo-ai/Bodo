import subprocess


def skip_ci():
    """Function that returns if CI should be skipped based upon the differences
    between this branch in master. This needs to be called in each stage of CI.
    """
    res = subprocess.run(["git", "diff", "--name-only", "master"], capture_output=True)
    files = res.stdout.decode("utf-8").strip().split("\n")
    for filename in files:
        if (
            "/" not in filename
            or filename.startswith("aws_scripts/")
            or filename.startswith("bodo/")
            or filename.startswith("buildscripts/")
        ):
            return False

    return True
