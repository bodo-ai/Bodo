import sys
import os
import subprocess
import re
import json

# first arg is the number of processes to run the tests with
num_processes = int(sys.argv[1])
# all other args go to pytest
pytest_args = sys.argv[2:]

# If in AWS Codebuild partition tests
if "CODEBUILD_BUILD_ID" in os.environ:
    import buildscripts.aws.select_timing_from_logs
    
    # Load the logfile for splitting tests
    result = subprocess.call(
        [
            "python",
            "buildscripts/aws/download_s3paths_with_prefix.py",
            "bodo-pr-testing-logs",
            "splitting_logs/latestlog",
        ]
    )
    if result != 0:
        raise Exception(
            "buildscripts/aws/download_s3_prefixes.py fails trying to download log file."
        )
    if not os.path.exists("splitting_logs/latestlog"):
        raise Exception("Log file download unsuccessful, exiting with failure.")
    # Select the markers that may be needed.
    marker_groups = buildscripts.aws.select_timing_from_logs.generate_marker_groups(
        "splitting_logs/latestlog", int(os.environ["NUMBER_GROUPS_SPLIT"])
    )

    # Generate the marker file.
    with open("bodo/pytest.ini", "a") as f:
        indent = " " * 4
        for marker in set(marker_groups.values()):
            print(
                "{0}{1}: Group {1} for running distributed tests\n".format(
                    indent, marker, file=f
                )
            )

    with open("testtiming.json", "w") as f:
        json.dump(marker_groups, f)

# run pytest with --collect-only to find Python modules containing tests
# (this doesn't execute any tests)
try:
    output = subprocess.check_output(["pytest"] + pytest_args + ["--collect-only"])
except subprocess.CalledProcessError as e:
    if e.returncode == 5:  # pytest returns error code 5 when no tests found
        exit()  # nothing to do
    else:
        raise e

# get the list of test modules (test file names) to run
pytest_module_regexp = re.compile("<Module (\S+.py)>")
modules = []
for l in output.decode().split("\n"):
    m = pytest_module_regexp.search(l)
    if m:
        modules.append(m.group(1))

# The '--cov-report=' option passed to pytest means that we want pytest-cov to
# generate coverage files (".coverage" files used by `coverage` to generate
# reports)
codecov = "--cov-report=" in pytest_args  # codecov requested
if codecov:
    # remove coverage files from pytest execution above
    subprocess.run(["coverage", "erase"])
    if not os.path.exists("cov_files"):
        # temporary directory to store coverage file of each test module
        os.makedirs("cov_files")

# run each test module in a separate process to avoid out-of-memory issues
# due to leaks
tests_failed = False
for i, m in enumerate(modules):
    # run tests only of module m

    # this environment variable causes pytest to mark tests in module m
    # with "single_mod" mark (see bodo/tests/conftest.py)
    os.environ["BODO_TEST_PYTEST_MOD"] = m

    # modify pytest_args to add "-m single_mod"
    mod_pytest_args = list(pytest_args)
    try:
        mark_arg_idx = pytest_args.index("-m")
        mod_pytest_args[mark_arg_idx + 1] += " and single_mod"
    except ValueError:
        mod_pytest_args += ["-m", "single_mod"]
    # run tests with pytest
    if num_processes == 1:
        cmd = ["pytest"] + mod_pytest_args
    else:
        cmd = ["mpiexec", "-n", str(num_processes), "pytest"] + mod_pytest_args
    print("Running", " ".join(cmd))
    p = subprocess.Popen(cmd, shell=False)
    rc = p.wait()
    if rc not in (0, 5):  # pytest returns error code 5 when no tests found
        # raise RuntimeError("An error occurred when running the command " + str(cmd))
        tests_failed = True
        continue  # continue with rest of the tests
    if codecov:
        assert os.path.isfile(".coverage"), "Coverage file was not created"
        # rename coverage file and move aside to avoid conflicts with pytest-cov
        os.rename(".coverage", "./cov_files/.coverage." + str(i))


if tests_failed:
    exit(1)


if codecov:
    # combine all the coverage files
    subprocess.run(["coverage", "combine", "cov_files"])
