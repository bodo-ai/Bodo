import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


@pytest.mark.slow
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
