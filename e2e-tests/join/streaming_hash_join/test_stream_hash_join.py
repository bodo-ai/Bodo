import os
import shutil

import pytest
from utils.utils import run_cmd, temp_env_override


@pytest.mark.skip(reason="TODO [BSE-4589]: Fix hanging tests on CI.")
@pytest.mark.parametrize(
    "input_schema, expected_out_len, expected_checksum_lower, expected_checksum_upper, mem_size_mib, num_ranks",
    [
        pytest.param(
            "tpch_sf1",
            6001215,
            1868530620296400,
            1868530620296430,
            None,
            4,
            id="sf1_basic_np4",
        ),
        pytest.param(
            "tpch_sf10",
            59986052,
            184673545992169650,
            184673545992169700,
            None,
            36,
            id="sf10_basic_np36",
        ),
    ],
)
def test_hash_join(
    unpin_behavior,
    input_schema,
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
                "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": str(mem_size_mib)
                if (mem_size_mib is not None)
                else None,
                "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
            }
        ),
    ):
        pytest_working_dir = os.getcwd()
        try:
            # change to directory of this file
            os.chdir(os.path.dirname(__file__))
            # remove __pycache__ (numba stores cache in there)
            shutil.rmtree("__pycache__", ignore_errors=True)
            hash_join(
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
            # hash_join(
            #     input_schema,
            #     expected_out_len,
            #     expected_checksum_lower,
            #     expected_checksum_upper,
            #     True,
            # )

        finally:
            # make sure all state is restored even in the case of exceptions
            os.chdir(pytest_working_dir)


def hash_join(
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
        "stream_hash_join.py",
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
