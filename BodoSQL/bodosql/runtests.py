"""
File used to run tests on CI.
"""

import os
import re
import subprocess
import sys

# first arg is the name of the testing pipeline
pipeline_name = sys.argv[1]
# second arg is the number of processes to run the tests with
num_processes = int(sys.argv[2])
# all other args go to pytest
pytest_args = sys.argv[3:]
# Get File-Level Timeout Info from Environment Variable (in seconds)
file_timeout = int(os.environ.get("BODO_RUNTESTS_TIMEOUT", 7200))

# Pipeline name is only used when testing on Azure
use_run_name = "AGENT_NAME" in os.environ

# run pytest with --collect-only to find Python modules containing tests
# (this doesn't execute any tests)
try:
    output = subprocess.check_output(["pytest"] + pytest_args + ["--collect-only"])
except subprocess.CalledProcessError as e:
    if e.returncode == 5:  # pytest returns error code 5 when no tests found
        exit()  # nothing to do
    else:
        print(e.output.decode())
        raise e

# get the list of test modules (test file names) to run
# excluding the test files located in the caching tests directory
pytest_module_regexp = re.compile(
    r"<(?:Module|CppTestFile) ((?!tests/caching_tests/)\S+.(?:py|cpp))>"
)
modules = []
for line in output.decode().split("\n"):
    m = pytest_module_regexp.search(line)
    if m:
        filename = m.group(1).split("/")[-1]
        modules.append(filename)
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
    # run tests with mpiexec + pytest always. If you just use
    # pytest then out of memory won't tell you which test failed.
    cmd = [
        "mpiexec",
        "-prepend-rank",
        "-n",
        str(num_processes),
        "pytest",
        "-Wignore",
        # junitxml generates test report file that can be displayed by CodeBuild website
        # use PYTEST_MARKER and module name to generate a unique filename for each group of tests as identified
        # by markers and test filename.
        f"--junitxml=pytest-report-{m.split('.')[0]}-{os.environ['PYTEST_MARKER'].replace(' ', '-')}.xml",
    ]
    if use_run_name:
        cmd.append(
            f"--test-run-title={pipeline_name}",
        )
    cmd += mod_pytest_args
    print(f"Running: {' '.join(cmd)} with module {m}")
    p = subprocess.Popen(cmd, shell=False)
    rc = p.wait(timeout=file_timeout)

    if rc not in (0, 5):  # pytest returns error code 5 when no tests found
        # raise RuntimeError("An error occurred when running the command " + str(cmd))
        print(
            f"Failure exit code encountered while running: {' '.join(cmd)} with module {m}\nExit code found was: {rc}"
        )
        tests_failed = True
        continue  # continue with rest of the tests

if tests_failed:
    exit(1)
