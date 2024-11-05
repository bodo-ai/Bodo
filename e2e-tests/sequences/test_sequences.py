import json
import os
import random
import shutil
import string

from utils.utils import run_cmd


def test_sequences():
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)
        validate_sequences_output()

        # Run again on cached code
        validate_sequences_output(True)

    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)


def validate_sequences_output(require_cache=False):
    inputpath = "s3://bodotest-customer-data/sequences/input/input_sample.txt"
    outputpath = "s3://bodotest-customer-data/sequences/output/output_result.txt"
    chosentype = "0"
    maxseqlength = "4194304"
    # run on 36 cores
    num_processes = 36
    random_str = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    checksum_loc = f"checksum_out_{random_str}.json"
    cmd = [
        "mpiexec",
        "-n",
        str(num_processes),
        "python",
        "-u",
        "-W",
        "ignore",
        "sequences.py",
        "-i",
        inputpath,
        "-o",
        outputpath,
        "--chosentype",
        chosentype,
        "--maxseqlength",
        maxseqlength,
        "--checksum_loc",
        f"{checksum_loc}",
    ]
    if require_cache:
        cmd.append("--require_cache")
    # Precomputed checksum to verify that the Bodo output doesn't change.
    expected_checksum = 17458202297301487
    run_cmd(cmd)
    with open(checksum_loc) as f:
        out_checksum = json.load(f)["checksum"]
    assert out_checksum == expected_checksum, "Checksum doesn't match"
