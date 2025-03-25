"""
File used to run tests on CI.
"""

import os
import shutil
import subprocess
import sys

if __name__ == "__main__":
    # first arg is the name of the testing pipeline
    pipeline_name = sys.argv[1]

    # second arg is the number of processes to run the tests with
    num_processes = int(sys.argv[2])

    # the third is the directory of the caching tests
    cache_test_dir = sys.argv[3]

    # Pipeline name is only used when testing on Azure
    use_run_name = "AGENT_NAME" in os.environ

    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(cache_test_dir))
        shutil.rmtree("__pycache__", ignore_errors=True)
    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)

    pytest_cmd_not_cached_flag = [
        "pytest",
        "-s",
        "-v",
        "-p",
        "no:faulthandler",
        cache_test_dir,
        "--is_cached",
        "n",
    ]

    # run tests with pytest
    cmd = ["mpiexec", "-n", str(num_processes)] + pytest_cmd_not_cached_flag

    print("Running", " ".join(cmd))
    p = subprocess.Popen(cmd, shell=False)
    rc = p.wait()
    failed_tests = False
    if rc not in (0, 5):  # pytest returns error code 5 when no tests found
        failed_tests = True

    pytest_cmd_yes_cached_flag = [
        "pytest",
        "-s",
        "-v",
        "-p",
        "no:faulthandler",
        cache_test_dir,
        "--is_cached",
        "y",
    ]
    if use_run_name:
        pytest_cmd_yes_cached_flag.append(
            f"--test-run-title={pipeline_name}",
        )
    # run tests with pytest
    cmd = [
        "mpiexec",
        "-prepend-rank",
        "-n",
        str(num_processes),
    ] + pytest_cmd_yes_cached_flag

    print("Running", " ".join(cmd))
    p = subprocess.Popen(cmd, shell=False)
    rc = p.wait()
    if rc not in (0, 5):  # pytest returns error code 5 when no tests found
        failed_tests = True

    if failed_tests:
        exit(1)
