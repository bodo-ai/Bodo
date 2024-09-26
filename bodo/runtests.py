import json
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

logfile_name = "splitting_logs/logfile-07-18-22.txt"

# If in AWS Codebuild partition tests
if "CODEBUILD_BUILD_ID" in os.environ:
    # Make sure buildscripts is in the import path
    module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
    repo_dir = os.path.abspath(os.path.join(module_dir, os.pardir))
    assert module_dir in sys.path
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    import buildscripts.aws.select_timing_from_logs

    # Load the logfile for splitting tests
    result = subprocess.call(
        [
            "python",
            "buildscripts/aws/download_s3paths_with_prefix.py",
            "bodo-pr-testing-logs",
            logfile_name,
        ]
    )
    if result != 0:
        raise Exception(
            "buildscripts/aws/download_s3_prefixes.py fails trying to download log file."
        )
    if not os.path.exists(logfile_name):
        raise Exception("Log file download unsuccessful, exiting with failure.")
    # Select the markers that may be needed.
    marker_groups = buildscripts.aws.select_timing_from_logs.generate_marker_groups(
        logfile_name, int(os.environ["NUMBER_GROUPS_SPLIT"])
    )

    # Generate the marker file.
    with open("bodo/pytest.ini", "a") as f:
        indent = " " * 4
        for marker in set(marker_groups.values()):
            print(
                "{0}{1}: Group {1} for running distributed tests\n".format(
                    indent, marker
                ),
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

weekly_modules = [
    "test_pyspark_api.py",
    "test_csr_matrix.py",
    "test_dl.py",
    "test_gcs.py",
    "test_json.py",
    "test_matplotlib.py",
    "test_ml.py",
    "test_xgb.py",
    "test_sklearn_part1.py",
    "test_sklearn_part2.py",
    "test_sklearn_part3.py",
    "test_sklearn_part4.py",
    "test_sklearn_part5.py",
    "test_sklearn_linear_model.py",
    "test_sklearn_errorchecking.py",
    "test_sklearn_cluster_ensemble.py",
    "test_sklearn_feature_extraction_text.py",
    "test_spark_sql_str.py",
    "test_spark_sql_array.py",
    "test_spark_sql_date.py",
    "test_spark_sql_numeric.py",
    "test_spark_sql_map.py",
]

# We don't run HDFS tests on CI, so exclude them.
modules = list(set(all_modules) - set(weekly_modules) - set(["test_hdfs.py"]))


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
        f"--junitxml=pytest-report-{m.split('.')[0]}-{os.environ['PYTEST_MARKER'].replace(' ','-')}.xml",
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
