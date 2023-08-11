# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File used to run caching tests on CI.
"""
import os
import shutil
import subprocess
import sys

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
    # change directory to cache location
    # NOTE:
    os.chdir(cache_test_dir)
    shutil.rmtree("__pycache__", ignore_errors=True)
finally:
    # make sure all state is restored even in the case of exceptions
    os.chdir(pytest_working_dir)

# Remove NUMBA_CACHE_DIR if it is set to enable local testing
# This env variable sets the cache location, which will violate
# our caching assumptions.
if "NUMBA_CACHE_DIR" in os.environ:
    del os.environ["NUMBA_CACHE_DIR"]

pytest_cmd_not_cached_flag = [
    "pytest",
    "-s",
    "-v",
    cache_test_dir,
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
    cache_test_dir,
    "--is_cached",
]
if use_run_name:
    pytest_cmd_yes_cached_flag.append(
        f"--test-run-title={pipeline_name}",
    )
cmd = ["mpiexec", "-n", str(num_processes)] + pytest_cmd_yes_cached_flag
print("Running", " ".join(cmd))
p = subprocess.Popen(cmd, shell=False)
rc = p.wait()
if rc not in (0, 5):  # pytest returns error code 5 when no tests found
    failed_tests = True

if failed_tests:
    exit(1)
