import argparse
import subprocess


def run_commands(commands):
    # Runs commands (all with no arguments) if CI should run
    if run_ci():
        # TODO: Support commands with arguments (none exist yet)
        for command in commands:
            subprocess.run([command], check=True)
    else:
        # If we are suppose to run unittests we need to generate an empty artifact
        if "buildscripts/aws/run_unittests.sh" in commands:
            f = open(".coverage", "w")
            f.close()


def run_ci():
    """Function that returns if CI should be skipped based upon the differences
    between this branch in develop. This needs to be called in each stage of CI.
    """
    res = subprocess.run(["git", "diff", "--name-only", "develop"], capture_output=True)
    files = res.stdout.decode("utf-8").strip().split("\n")
    for filename in files:
        if filename.startswith("bodo/docs"):
            continue
        if (
            "/" not in filename
            or filename.startswith("aws_scripts/")
            or filename.startswith("bodo/")
            or (
                filename.startswith("buildscripts/")
                and not filename.startswith("buildscripts/azure/")
            )
            or filename.startswith("iceberg/")
        ):
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commands", type=str, nargs="+", help="Commands to run for CI")
    args = parser.parse_args()
    run_commands(args.commands)
