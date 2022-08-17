import os
import re
import shutil

from utils.utils import run_cmd


def process_output(output):
    regexp = re.compile("checksum (\S+)")
    for l in output.splitlines():
        m = regexp.match(l)
        if m:
            return float(m.group(1))
    raise RuntimeError("process_output: result not found")


def tpcx_bb_helper(test_file_name, good_result):
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))

        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        # First run with 1 process and generate the cache
        cmd = ["python", "-u", test_file_name]
        result = process_output(run_cmd(cmd))
        assert result == good_result

        for num_processes in (2, 3, 4):
            cmd = [
                "mpiexec",
                "-n",
                str(num_processes),
                "python",
                "-u",
                test_file_name,
                "True",  # tell script to make sure we load from cache, or fail
            ]
            result = process_output(run_cmd(cmd))
            assert result == good_result
    finally:
        os.chdir(pytest_working_dir)


def test_tpcx_bb_csv():
    good_result = 3380326.0
    tpcx_bb_helper("TPCxBB_q26.py", good_result)


def test_tpcx_bb_parquet():
    good_result = 3380326.0
    tpcx_bb_helper("TPCxBB_q26_pq.py", good_result)
