import os
import re
import shutil

from utils.utils import run_cmd

# NOTE: mnist data was downloaded and converted to numpy with generate_mnist_data.py
# script in this folder, and uploaded to our S3 bucket

BUCKET_NAME = "bodotest-customer-data"
BUCKET_REGION = "us-west-1"


def process_output(output):
    regexp = re.compile(r"Accuracy is (\S+)")
    for l in output.splitlines():
        m = regexp.match(l)
        if m:
            return float(m.group(1))
    raise RuntimeError("process_output: result not found")


def dl_test_run(test_file, data_path, accuracy_threshold):
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        os.environ["AWS_DEFAULT_REGION"] = BUCKET_REGION

        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        cmd = ["mpiexec", "-n", "4", "python", "-u", test_file, data_path]
        accuracy = process_output(run_cmd(cmd))
        assert accuracy >= accuracy_threshold

        # run again, this time from cache
        cmd.append("True")  # tell script to make sure we load from cache, or fail
        accuracy = process_output(run_cmd(cmd))
        assert accuracy >= accuracy_threshold
    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)
        if "AWS_DEFAULT_REGION" in os.environ:
            os.environ.pop("AWS_DEFAULT_REGION")


def test_pytorch_mnist():
    path = "s3://{}/dl-mnist".format(BUCKET_NAME)
    dl_test_run("pytorch_mnist.py", path, 0.9)


def test_tf_mnist():
    path = "s3://{}/dl-mnist".format(BUCKET_NAME)
    dl_test_run("tensorflow_mnist.py", path, 0.9)
