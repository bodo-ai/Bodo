import numpy as np
import pandas as pd

import bodo
from bodo.tests.utils import _gather_output, _test_equal

"""
Test that _bodo_rebalance_output_if_skewed produces a correct output
and shuffles the output to have "even" final outputs.

Note we need to manually test the distributions because otherwise we likely won't have significant skew.
In addition this needs to run on at least 5 ranks.
"""


@bodo.jit(distributed=["df1", "df2"])
def impl1(df1, df2):
    return df1.merge(df2, how="cross", _bodo_rebalance_output_if_skewed=True)


@bodo.jit(distributed=["df1", "df2"])
def impl2(df1, df2):
    return df1.merge(
        df2,
        how="inner",
        left_on="B1",
        right_on="B2",
        _bodo_rebalance_output_if_skewed=True,
    )


if __name__ == "__main__":
    # Initial DataFrames for the final output.
    df1 = pd.DataFrame({"A1": np.arange(1000), "B1": np.arange(1000) % 10})
    df2 = pd.DataFrame({"A2": np.arange(100), "B2": np.arange(100) % 5})

    # Generate the local chunk of data for testing cross join.
    # Place the majority of data on rank 0.
    rank_zero_size = len(df1) - (bodo.get_size() - 1)
    df1_start = 0 if bodo.get_rank() == 0 else rank_zero_size + (bodo.get_rank() - 1)
    df1_len = rank_zero_size if bodo.get_rank() == 0 else 1
    df1_chunk = df1[df1_start : df1_start + df1_len].reset_index(drop=True)

    # Evenly divide df2
    df2_start, df2_len = bodo.libs.distributed_api.get_start_count(len(df2))
    df2_chunk = df2[df2_start : df2_start + df2_len].reset_index(drop=True)

    # Call the bodo function.
    res = impl1(df1_chunk, df2_chunk)

    # Verify that the output is evenly distributed.
    local_len = bodo.libs.distributed_api.get_node_portion(
        len(df1) * len(df2), bodo.get_size(), bodo.get_rank()
    )
    assert len(res) == local_len, "Output is not evenly distributed."

    # Verify that the output is correct.
    py_output = df1.merge(df2, how="cross")
    total_bodo_output = _gather_output(res)
    if bodo.get_rank() == 0:
        _test_equal(
            total_bodo_output,
            py_output,
            sort_output=True,
            reset_index=True,
            check_dtype=False,
        )

    # Generate the local chunk of data for testing inner join.

    # Prune df1 to just keys 0.
    df1 = df1[df1.B1 == 0].reset_index(drop=True)
    # Evenly divide df1. We will have imbalance due to hashing
    df1_start, df1_len = bodo.libs.distributed_api.get_start_count(len(df1))
    df1_chunk = df1[df1_start : df1_start + df1_len].reset_index(drop=True)

    # Evenly divide df2
    df2_start, df2_len = bodo.libs.distributed_api.get_start_count(len(df2))
    df2_chunk = df2[df2_start : df2_start + df2_len].reset_index(drop=True)

    # Call the bodo function.
    res = impl2(df1_chunk, df2_chunk)

    # Verify that the output is evenly distributed.
    local_len = bodo.libs.distributed_api.get_node_portion(
        len(df1) * len(df2) // 5, bodo.get_size(), bodo.get_rank()
    )
    assert len(res) == local_len, "Output is not evenly distributed."

    # Verify that the output is correct.
    py_output = df1.merge(df2, how="inner", left_on="B1", right_on="B2")
    total_bodo_output = _gather_output(res)
    if bodo.get_rank() == 0:
        _test_equal(
            total_bodo_output,
            py_output,
            sort_output=True,
            reset_index=True,
            check_dtype=False,
        )

    bodo.barrier()
