import os
import shutil

import pytest
from utils.utils import run_cmd, temp_env_override


@pytest.mark.parametrize(
    "input_schema, ftype, expected_out_len, expected_checksum_lower, expected_checksum_upper, mem_size_mib, num_ranks",
    [
        pytest.param(
            "tpch_sf10",
            "agg",
            700_000,
            # Actual is 9270394310204
            9270394310200,
            9270394310210,
            None,
            4,
            id="sf10_agg_basic_np4",
        ),
        pytest.param(
            "tpch_sf10",
            "acc",
            700_000,
            # Actual is 3507973254066
            3507973254050,
            3507973254100,
            None,
            18,
            id="sf10_acc_basic_np18",
        ),
        pytest.param(
            "tpch_sf10",
            "drop_duplicates",
            700_000,
            # Actual is 3500126700000
            3500126699950,
            3500126700050,
            None,
            8,
            id="sf10_drop_duplicates_basic_np8",
        ),
    ],
)
def test_stream_groupby(
    unpin_behavior,
    input_schema,
    ftype,
    expected_out_len,
    expected_checksum_lower,
    expected_checksum_upper,
    mem_size_mib,
    num_ranks,
):
    with (
        unpin_behavior(),
        temp_env_override(
            {
                "BODO_BUFFER_POOL_WORKER_MEMORY_SIZE_MiB": str(mem_size_mib)
                if (mem_size_mib is not None)
                else None,
                "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
            }
        ),
    ):
        pytest_working_dir = os.getcwd()
        try:
            # change to directory of this file
            os.chdir(os.path.dirname(__file__))
            # remove __pycache__ (numba stores cache in there)
            shutil.rmtree("__pycache__", ignore_errors=True)
            stream_groupby(
                ftype,
                input_schema,
                expected_out_len,
                expected_checksum_lower,
                expected_checksum_upper,
                False,
                num_ranks,
            )

            # TODO Run again on cached code.
            # This doesn't work yet since we always
            # write to a different table, which means
            # it always re-compiles.
            # stream_groupby(
            #     input_schema,
            #     expected_out_len,
            #     expected_checksum_lower,
            #     expected_checksum_upper,
            #     True,
            # )

        finally:
            # make sure all state is restored even in the case of exceptions
            os.chdir(pytest_working_dir)


def stream_groupby(
    ftype,
    input_schema,
    expected_out_len,
    expected_checksum_lower,
    expected_checksum_upper,
    require_cache=False,
    num_processes=4,
):
    cmd = [
        "python",
        "-u",
        "stream_groupby.py",
        "--ftype",
        ftype,
        "--input_schema",
        input_schema,
        "--expected_out_len",
        str(expected_out_len),
        "--expected_checksum_lower",
        str(expected_checksum_lower),
        "--expected_checksum_upper",
        str(expected_checksum_upper),
    ]
    if require_cache:
        cmd.append("--require_cache")
    run_cmd(cmd, additional_envs={"BODO_NUM_WORKERS": str(num_processes)})
