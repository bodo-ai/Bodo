# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File used to run tests on CI.
"""
import os
import shutil
import subprocess
import sys

# first arg is the number of processes to run the tests with
num_processes = int(sys.argv[1])

# the second is the directory of the caching tests
cache_test_dir = sys.argv[2]

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
# run tests with pytest
if num_processes == 1:
    cmd = pytest_cmd_yes_cached_flag
else:
    cmd = ["mpiexec", "-n", str(num_processes)] + pytest_cmd_yes_cached_flag

print("Running", " ".join(cmd))
p = subprocess.Popen(cmd, shell=False)
rc = p.wait()
if rc not in (0, 5):  # pytest returns error code 5 when no tests found
    failed_tests = True

if failed_tests:
    exit(1)
