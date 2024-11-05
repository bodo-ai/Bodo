import os
import shutil

import pandas as pd
from utils.utils import run_cmd

import bodo

BUCKET_NAME = "s3://bodotest-customer-data/search_grid"


def test_search(tmp_path):
    # Get latest AWS creds for any pandas S3 operation.
    # The env vars should be populated by the assume_iam_role
    # fixture which is used on all tests by default.
    # This is required since Pandas/Botocore may keep using
    # an old "session" whose token has expired. There's no
    # good way to ask it to reset its credentials, so it's
    # safer to set the credentials explicitly using
    # storage_options.
    s3_storage_options = {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "token": os.environ["AWS_SESSION_TOKEN"],
    }
    s3_oracle = BUCKET_NAME + "/oracle_df_rec3.csv"
    pytest_working_dir = os.getcwd()
    try:
        # change to directory of this file
        os.chdir(os.path.dirname(__file__))
        # remove __pycache__ (numba stores cache in there)
        shutil.rmtree("__pycache__", ignore_errors=True)

        # get the oracle
        oracle_df = pd.read_csv(s3_oracle, storage_options=s3_storage_options)
        oracle_sorted = oracle_df.sort_values(list(oracle_df.columns)).reset_index(
            drop=True
        )
        oracle_sorted["PRICE_LOCK"] = oracle_sorted.PRICE_LOCK.astype("boolean")

        # run on 36 cores
        num_processes = 36
        results_file = os.path.join(tmp_path, f"df_rec{str(num_processes)}.csv")
        cmd = [
            "mpiexec",
            "-n",
            str(num_processes),
            "python",
            "-u",
            "search.py",
            BUCKET_NAME,
            results_file,
        ]
        run_cmd(cmd)
        validate_results(results_file, oracle_sorted)
    finally:
        # make sure all state is restored even in the case of exceptions
        os.chdir(pytest_working_dir)


def validate_results(bodo_result, oracle):
    df2 = bodo.jit(lambda: pd.read_csv(bodo_result))()
    # convert pyarrow string data to regular object arrays to avoid dtype errors
    for i in range(len(df2.columns)):
        if df2.dtypes.iloc[i] == pd.StringDtype("pyarrow"):
            df2.iloc[:, i] = df2.iloc[:, i].values.to_numpy()
    df22 = df2.sort_values(list(df2.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        oracle, df22, check_column_type=False, check_dtype=False
    )
