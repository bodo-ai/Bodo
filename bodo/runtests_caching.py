"""
File used to run caching tests on CI.
"""

import multiprocessing
import os
import shutil
import subprocess
import sys

import pandas as pd
from numba.misc.appdirs import AppDirs

import bodo


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


def recursive_count_dir(start_dir, to_count):
    """
    Find all directories with a given name in a directory tree and count them.
    Args:
        start_dir: the root of the directory tree to search in
        to_count: the directory name to be counted when one is found
    """
    ret = 0
    for dirpath, dirnames, _ in os.walk(start_dir, topdown=False):
        for dirname in dirnames:
            if dirname == to_count:
                ret += 1
    return ret


def test_internal_caching(S):
    """This test is here because if we put it in a regular pytest file then you
    have no control over the order in which it is run.  You could check the
    output of the function there but that isn't what we are trying to test
    but instead making sure that an internal function is cached properly.
    Again, if you put that check in a pytest file then that file could be
    run in whatever order and some other test might create that internal
    function which invalidates the purpose of this test.  This test included
    here has to be the first thing run after clearing the internal caches.
    """
    return S.dt.year


def run_test_internal_caching(ic_queue, first_time):
    """
    Run test_internal_caching and check if the cache hits and misses are appropriate.
    """
    S = pd.date_range("2020-01-01", periods=100).to_series()
    # Do bodo.jit here to avoid potential problems of pickling dispatchers.
    bodo_jit_func = bodo.jit(spawn=False, distributed=False)(test_internal_caching)
    # Run the test on the example dataframe above.
    bodo_jit_func(S)
    ret = 0
    # We know the above test uses bodo.libs.int_arr_ext.alloc_int_array.
    bodo_func = bodo.libs.int_arr_ext.alloc_int_array
    sig = bodo_func.signatures[0]
    if first_time:
        # First time run make sure it was a cache miss.
        if bodo_func._cache_hits[sig] != 0:
            ret += 1
        if bodo_func._cache_misses[sig] != 1:
            ret += 2
    else:
        # Second time run make sure it was a cache hit.
        if bodo_func._cache_hits[sig] != 1:
            ret += 4
        if bodo_func._cache_misses[sig] != 0:
            ret += 8
    # Return the result to the parent process.
    ic_queue.put(ret)


if __name__ == "__main__":
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
    # .strfunc_cache appended.
    appdirs = AppDirs(appname="bodo", appauthor=False)
    cache_path = os.path.join(appdirs.user_cache_dir, ".strfunc_cache")
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

    ic_queue = multiprocessing.Queue()
    # Run a function in another process once to prime some internal caches.
    print("Running internal caching test the first time.")
    ic_process = multiprocessing.Process(
        target=run_test_internal_caching, args=(ic_queue, True)
    )
    ic_process.start()
    ic_process.join()
    # Get the return code of run_test_internal_caching.
    ic_res = ic_queue.get()
    if ic_res != 0:
        print(f"FAILED: Bodo internal caching first run code {ic_res}.")
        failed_tests = True
    # Run the same function again in yet another process and check if the disk cache
    # was used.
    print("Running internal caching test the second time.")
    ic_process = multiprocessing.Process(
        target=run_test_internal_caching, args=(ic_queue, False)
    )
    ic_process.start()
    ic_process.join()
    # Get the return code of run_test_internal_caching.
    ic_res = ic_queue.get()
    if ic_res != 0:
        print(f"FAILED: Bodo internal caching second run code {ic_res}.")
        failed_tests = True

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
        print(
            f"FAILED: Bodo string-generated cache directory {cache_path} does not exist."
        )
        failed_tests = True
    elif not any(os.listdir(cache_path)):
        print(f"FAILED: Bodo string-generated cache directory {cache_path} is empty.")
        failed_tests = True

    pycache_count = recursive_count_dir(bodo_dir, "__pycache__")
    if pycache_count <= 1:
        print("FAILED: Bodo internal functions cache directories not created.")
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
