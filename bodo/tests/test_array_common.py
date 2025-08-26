import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
)

"""
Test for common array utilities that should be shared
across all arrays.
"""


@pytest.mark.parametrize(
    "lst, arr_type",
    [
        ([1, 2, 3, 5] * 3, bodo.types.IntegerArrayType(bodo.types.int32)),
        ([1.1, 1.2, 3.1, 4.5] * 3, bodo.types.FloatingArrayType(bodo.types.float64)),
        (["a", "b", "a", "b", "c"] * 3, bodo.types.string_array_type),
        (["a", "b", "a", "b", "c"] * 3, bodo.types.dict_str_arr_type),
    ],
)
def test_list_to_array(lst, arr_type, memory_leak_check):
    def impl(lst):
        return bodo.utils.conversion.list_to_array(lst, arr_type)

    py_output = pd.array(lst)
    check_func(impl, (lst,), py_output=py_output, check_dtype=False)


def test_int_list_with_null_to_array(memory_leak_check):
    """
    A separate test for null because None can only be provided in literal lists.
    """
    lst = [1, 2, 3, None, 5, 1]
    arr_type = bodo.types.IntegerArrayType(bodo.types.int32)

    def impl():
        return bodo.utils.conversion.list_to_array([1, 2, 3, None, 5, 1], arr_type)

    py_output = pd.array(lst)
    check_func(impl, (), py_output=py_output, check_dtype=False)


@pytest.mark.parametrize(
    "arr_type",
    [
        bodo.types.string_array_type,
        bodo.types.dict_str_arr_type,
    ],
)
def test_str_list_with_null_to_array(arr_type, memory_leak_check):
    """
    A separate test for null because None can only be provided in literal lists.
    """
    lst = ["a", "b", None, "b", "c", "c"]

    def impl():
        return bodo.utils.conversion.list_to_array(
            ["a", "b", None, "b", "c", "c"], arr_type
        )

    py_output = pd.array(lst)
    check_func(impl, (), py_output=py_output, check_dtype=False)
