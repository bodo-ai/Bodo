import argparse
import os
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run a testing pipeline with pytest args."
    )
    parser.add_argument(
        "pipeline_name",
        help="Name of the testing pipeline",
    )
    parser.add_argument(
        "num_processes",
        type=int,
        help="Number of processes to run the tests with",
    )
    parser.add_argument(
        "--mode",
        choices=["COMPILER", "SPAWN", "DF_LIB"],
        default="COMPILER",
        help="Optional mode to run the pipeline",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Arguments to forward to pytest",
    )
    args = parser.parse_args()

    mode = args.mode
    pipeline_name = args.pipeline_name
    num_processes = args.num_processes
    pytest_args = args.pytest_args

    # expand ignore args
    new_pytest_args = []
    for arg in pytest_args:
        if arg.startswith("--ignore"):
            new_pytest_args += arg.split()
        else:
            new_pytest_args.append(arg)
    pytest_args = new_pytest_args

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
    # omit caching tests, as those will be run in a separate pipeline
    pytest_module_regexp = re.compile(
        r"<(?:Module|CppTestFile) ((?!tests/caching_tests/)\S+.(?:py|cpp))>"
    )
    all_modules = []

    for l in output.decode().split("\n"):
        m = pytest_module_regexp.search(l)
        if m:
            filename = m.group(1).split("/")[-1]
            all_modules.append(filename)

    # We don't run HDFS tests on CI, so exclude them.
    modules = list(set(all_modules) - {"test_hdfs.py"})

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

        # run tests with mpiexec + pytest always. If you just use
        # pytest then out of memory won't tell you which test failed.
        if mode == "COMPILER":
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
                f"--junitxml=pytest-report-{m.split('.')[0]}-{os.environ.get('PYTEST_MARKER', '').replace(' ', '-')}.xml",
            ]
        else:
            cmd = [
                "env",
                f"BODO_NUM_WORKERS={num_processes}",
                "pytest",
                "-Wignore",
            ]

        if use_run_name:
            cmd.append(
                f"--test-run-title={pipeline_name}",
            )
        cmd += mod_pytest_args
        print("Running", " ".join(cmd))
        p = subprocess.Popen(cmd, shell=False)
        rc = p.wait(timeout=file_timeout)

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


if __name__ == "__main__":
    main()
