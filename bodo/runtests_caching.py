"""
File used to run caching tests on CI.
"""

import os
import shutil
import subprocess
import sys

from numba.misc.appdirs import AppDirs


def recursive_rmdir(start_dir, to_remove):
    """
    Find all directories with a given name in a directory tree and remove them.
    Args:
        start_dir: the root of the directory tree to search in
        to_remove: the directory name to be removed when one is found
    """
    for dirpath, dirnames, _ in os.walk(start_dir, topdown=False):
        for dirname in dirnames:
            if dirname == to_remove:
                dir_to_remove = os.path.join(dirpath, dirname)
                shutil.rmtree(dir_to_remove, ignore_errors=True)


# first arg is the name of the testing pipeline
pipeline_name = sys.argv[1]

# second arg is the number of processes to run the tests with
num_processes = int(sys.argv[2])

# the third is the directory of the caching tests
cache_test_dir = sys.argv[3]

# The directory this file resides in
bodo_dir = os.path.dirname(os.path.abspath(__file__))

# Pipeline name is only used when testing on Azure
use_run_name = "AGENT_NAME" in os.environ

# String-generated bodo functions are cached in the directory
# as defined in Numba (which is currently ~/.cache/bodo) with
# .bodo_strfunc_cache appended.
appdirs = AppDirs(appname="bodo", appauthor=False)
cache_path = os.path.join(appdirs.user_cache_dir, ".bodo_strfunc_cache")
# Remove the string-generated cache directory to make sure the tests
# recreate it.
shutil.rmtree(cache_path, ignore_errors=True)

pytest_working_dir = os.getcwd()
try:
    # change directory to cache location
    # NOTE:
    os.chdir(bodo_dir)
    recursive_rmdir(bodo_dir, "__pycache__")
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

# First invocation of the tests done at this point.
if not os.path.isdir(cache_path):
    print(f"FAILED: Bodo string-generated cache directory {cache_path} does not exist.")
    failed_tests = True
elif not any(os.listdir(cache_path)):
    print(f"FAILED: Bodo string-generated cache directory {cache_path} is empty.")
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
