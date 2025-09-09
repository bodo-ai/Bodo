"""
Unittests for DataFrames
"""

import random

import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.tests.dataframe_common
from bodo.tests.dataframe_common import *  # noqa
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_array_struct_list_int,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    pytest_pandas,
)

pytestmark = pytest_pandas


def test_dataframe_sample_number(memory_leak_check):
    """Checking the random routine is especially difficult to do.
    We can mostly only check incidental information about the code"""

    def f(df):
        return df.sample(n=4, replace=False).size

    bodo_f = bodo.jit(all_args_distributed_block=True, all_returns_distributed=False)(f)
    n = 10
    df = pd.DataFrame({"A": list(range(n))})
    py_output = f(df)
    df_loc = _get_dist_arg(df)
    assert bodo_f(df_loc) == py_output


@pytest.mark.slow
def test_dataframe_sample_uniform(memory_leak_check):
    """Checking the random routine, this time with uniform input"""

    def f1(df):
        return df.sample(n=4, replace=False)

    def f2(df):
        return df.sample(frac=0.5, replace=False)

    n = 10
    df = pd.DataFrame({"A": [1 for _ in range(n)]})
    check_func(f1, (df,), reset_index=True)
    check_func(f2, (df,), reset_index=True)


@pytest.mark.slow
def test_dataframe_sample_empty_ranks(memory_leak_check):
    """Checking the random routine with some empty ranks, to verify indexing"""

    @bodo.jit(distributed=["df"])
    def f1(df):
        return df.sample(n=4, replace=False)

    @bodo.jit(distributed=["df"])
    def f2(df):
        return df.sample(frac=0.5, replace=False)

    n = 10
    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": [1 for _ in range(n)]}, dtype="Int64")
    else:
        df = pd.DataFrame({"A": []}, dtype="Int64")

    bodo_output = bodo.allgatherv(f1(df))
    py_output = pd.DataFrame({"A": [1 for _ in range(4)]}, dtype="Int64")
    bodo_output.reset_index(inplace=True, drop=True)
    py_output.reset_index(inplace=True, drop=True)
    pd.testing.assert_frame_equal(bodo_output, py_output, check_dtype=False)

    bodo_output = bodo.allgatherv(f2(df))
    py_output = pd.DataFrame({"A": [1 for _ in range(int(10 * 0.5))]}, dtype="Int64")
    bodo_output.reset_index(inplace=True, drop=True)
    py_output.reset_index(inplace=True, drop=True)
    pd.testing.assert_frame_equal(bodo_output, py_output, check_dtype=False)


@pytest.mark.slow
def test_dataframe_sample_sorted(memory_leak_check):
    """Checking the random routine. Since we use sorted and the number of entries is equal to
    the number of sampled rows, after sorting the output becomes deterministic, that is independent
    of the random number generated"""

    def f(df, n):
        return df.sample(n=n, replace=False)

    n = 10
    df = pd.DataFrame({"A": list(range(n))})
    check_func(f, (df, n), reset_index=True, sort_output=True)


@pytest.mark.slow
def test_dataframe_sample_index(memory_leak_check):
    """Checking that the index passed coherently to the A entry."""

    def f(df):
        return df.sample(5)

    df = pd.DataFrame({"A": list(range(20))})
    bodo_f = bodo.jit(all_args_distributed_block=False, all_returns_distributed=False)(
        f
    )
    df_ret = bodo_f(df)
    S = df_ret.index == df_ret["A"]
    assert S.all()


# TODO: fix leak and add memory_leak_check
@pytest.mark.slow
def test_dataframe_sample_nested_datastructures():
    """The sample function relies on allgather operations that deserve to be tested"""
    from bodo.tests.utils_jit import get_start_end

    def check_gather_operation(df):
        siz = df.size

        def f(df, m):
            return df.sample(n=m, replace=False).size

        py_output = f(df, siz)
        start, end = get_start_end(len(df))
        df_loc = df.iloc[start:end]
        bodo_f = bodo.jit(
            all_args_distributed_block=True, all_returns_distributed=False
        )(f)
        df_ret = bodo_f(df_loc, siz)
        assert df_ret == py_output

    n = 10
    random.seed(1)
    df1 = pd.DataFrame({"B": gen_random_arrow_array_struct_int(10, n)})
    df2 = pd.DataFrame({"B": gen_random_arrow_array_struct_list_int(10, n)})
    df3 = pd.DataFrame({"B": gen_random_arrow_list_list_int(1, 0.1, n)})
    df4 = pd.DataFrame({"B": gen_random_arrow_struct_struct(10, n)})
    check_gather_operation(df1)
    check_gather_operation(df2)
    check_gather_operation(df3)
    check_gather_operation(df4)


@pytest.mark.slow
def test_dataframe_sample_frac_one_replace_false():
    def test_impl1(df):
        return df.sample(frac=1, replace=False)

    def test_impl2(df):
        return df.sample(frac=1.0)

    def test_impl3(df, num):
        return df.sample(n=num)

    df = pd.DataFrame(
        {"A": np.arange(10), "B": 1.5 + np.arange(10)},
        index=[f"i{i}" for i in range(10)],
    )
    n = len(df)

    check_func(test_impl1, (df,), sort_output=True)
    check_func(test_impl2, (df,), sort_output=True)
    check_func(
        test_impl3,
        (
            df,
            n,
        ),
        sort_output=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("items_per_rank, niters", [(5, 10000)])
@pytest.mark.parametrize("distributed", [True, False])
def test_dataframe_sample_distribution(
    items_per_rank, niters, frac, distributed, memory_leak_check
):
    """Test that dataframe sample returns uniformly sampled results"""

    def impl(df, nsamp, random_state):
        return df.sample(n=nsamp, random_state=random_state)

    nitems = items_per_rank * bodo.get_size()
    nsamp = int(round(nitems * frac))
    data = np.arange(nitems)

    if distributed:
        jit_impl = bodo.jit(distributed=["df"])(impl)
        df = pd.DataFrame({"A": _get_dist_arg(data)})
        bodo_sample_n = bodo.allgatherv(np.array(jit_impl(df, nsamp, 0)["A"]))
        bodo_sample_all = bodo.allgatherv(np.array(jit_impl(df, len(data), 0)["A"]))
    else:
        jit_impl = bodo.jit()(impl)
        df = pd.DataFrame({"A": data})
        bodo_sample_n = np.array(jit_impl(df, nsamp, 0)["A"])
        bodo_sample_all = np.array(jit_impl(df, len(data), 0)["A"])

    # Check that number of samples is correct
    assert bodo_sample_n.shape[0] == min(nsamp, data.shape[0])

    # Check that samples are a subset of `df` with no repetitions
    bodo_sample_n_elts = set(bodo_sample_n)
    bodo_sample_all_elts = set(bodo_sample_all)
    assert len(bodo_sample_n_elts) == nsamp
    assert bodo_sample_n_elts.issubset(bodo_sample_all_elts)
    assert len(bodo_sample_all_elts) == len(data)
    assert bodo_sample_all_elts.issubset(set(data))

    # Check that output is close to a uniform distribution on each rank.
    # Note that distributed df.sample keeps all data local.
    if nsamp > 0:
        if distributed:
            # Ignore sample order in the distributed case. As each node samples
            # locally, values will not shift too much from their starting node
            # output_freqs[j] indicates the number of times that index j is
            # sampled at any position.
            nlocal = df["A"].shape[0]
            all_nlocal = bodo.allgatherv(np.array([nlocal], dtype=np.int64))
            local_offset = all_nlocal.cumsum()[bodo.get_rank()] - all_nlocal[0]

            output_freqs = np.zeros((nlocal,), dtype=np.int64)
            for i in range(niters):
                output = np.array(jit_impl(df, nsamp, i)["A"])
                for j in range(output.shape[0]):
                    output_freqs[output[j] - local_offset] += 1

            expected_freq = nsamp * niters / nitems

        else:
            # Consider sample order in the sequential case. The index sampled
            # at each position should be uniform.
            # output_freqs[i, j] indicates the number of times that index j is
            # sampled at position i.
            output_freqs = np.zeros((nsamp, nitems), dtype=np.int64)
            for i in range(niters):
                output = np.array(jit_impl(df, nsamp, i)["A"])
                for j in range(output.shape[0]):
                    output_freqs[j, output[j]] += 1

            expected_freq = niters / nitems

        assert np.all(3 / 4 * expected_freq < output_freqs)
        assert np.all(output_freqs < 4 / 3 * expected_freq)
