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
