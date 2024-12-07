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
    """
    Function that returns if CI should run based upon the differences
    between this branch and main. This needs to be called in each stage of CI.
    """
    res = subprocess.run(["git", "diff", "--name-only", "main"], capture_output=True)
    files = res.stdout.decode("utf-8").strip().split("\n")
    for filename in files:
        if filename.startswith("bodo/docs"):
            continue
        if (
            # We should run BodoSQL CI if there's been changes in Bodo
            # or in the BodoSQL sub module
            "/" not in filename
            # BodoSQL changes
            or filename.startswith("BodoSQL/bodosql/")
            or filename.startswith("BodoSQL/buildscripts/")
            or filename.startswith("BodoSQL/calcite_sql/")
            # Bodo and/or Iceberg changes
            or filename.startswith("aws_scripts/")
            or filename.startswith("bodo/")
            or (
                filename.startswith("buildscripts/")
                and not filename.startswith("buildscripts/bodo/azure/")
                and not filename.startswith("buildscripts/bodosql/azure/")
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
