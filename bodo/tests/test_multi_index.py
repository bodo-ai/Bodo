import math

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_from_product_tuple():
    def impl():
        numbers = [0, 1, 2]
        colors = ["green", "purple"]
        return pd.MultiIndex.from_product((numbers, colors))

    check_func(impl, [], dist_test=False)


@pytest.mark.slow
def test_from_product_complicated_iterables():
    def impl():
        iterables = (
            [1, 10, 4, 5, 2],
            [2, 3, 25, 8, 9],
            [79, 25, 5, 10, -3],
            [3, 4, 4, 2, 90],
        )
        return pd.MultiIndex.from_product(iterables)

    check_func(impl, [], dist_test=False)


@pytest.mark.slow
def test_from_product_tuple_names():
    def impl():
        numbers = [0, 1, 2]
        colors = ["green", "purple"]
        names = ("a", "b")
        return pd.MultiIndex.from_product((numbers, colors), names=names)

    check_func(impl, [], dist_test=False)


@pytest.mark.slow
def test_from_product_tuple_names_different_lengths():
    def impl():
        numbers = [0, 1, 2]
        colors = ["green", "purple"]
        names = ("a",)
        return pd.MultiIndex.from_product((numbers, colors), names=names)

    message = "iterables and names must be of the same length"
    with pytest.raises(BodoError, match=message):
        bodo.jit(impl, distributed=False)()


@pytest.mark.slow
def test_from_product_sortorder_defined():
    def impl():
        numbers = [0, 1, 2]
        colors = ["green", "purple"]
        sortorder = 1
        return pd.MultiIndex.from_product((numbers, colors), sortorder=sortorder)

    message = "sortorder parameter only supports default value None"
    with pytest.raises(BodoError, match=message):
        bodo.jit(impl, distributed=False)()


def test_multi_index_head(memory_leak_check):
    """
    [BE-2273]. Test that df.head works as expected with multi-index
    DataFrames.
    """

    def impl(df):
        new_df = df.groupby(["A", "B"]).apply(lambda x: 1)
        res = new_df.head(10)
        return res

    df = pd.DataFrame(
        {
            "A": [i for i in range(10)] * 70,
            "B": [j for j in range(7)] * 100,
            "C": np.arange(700),
        }
    )
    # Reset the index because the groupby means order
    # won't be maintained
    check_func(impl, (df,), reset_index=True)


def test_multi_nbytes(memory_leak_check):
    """
    Tests nbytes works as expected with a MultiIndex.
    """

    def impl(index):
        return index.nbytes

    arrays = [[1, 1, 2, 2] * 10, ["red", "blue", "red", "blue"] * 10]
    index = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
    py_out = (
        (40 * 8) + (40 // 8) + (41 * 8) + (40 * 3.5)
    )  # int data is 40 * 8, string data is 5 bytes null bitmap + 41 * 8 for offset + 170 for string data
    check_func(impl, (index,), py_output=py_out, only_seq=True)
    # Same data is distributed on multiple ranks. As a result int is unchanged, but
    # string needs to account for null bitmap changes and offset changes.
    # 1 extra offset integer on extra ranks
    extra_offset_bytes = (bodo.get_size() - 1) * 8
    # Null bits depends on the bitmap padding
    extra_null_bitmap = (math.ceil(40 / (8 * bodo.get_size())) * bodo.get_size()) - 5
    py_out = py_out + extra_offset_bytes + extra_null_bitmap
    check_func(impl, (index,), py_output=py_out, only_1DVar=True)
