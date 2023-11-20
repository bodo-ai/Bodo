# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for array of map values.
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.map_arr_ext import contains_map_array
from bodo.tests.utils import box_as_map, check_func


@pytest.fixture(
    params=[
        # simple types to handle in C
        np.array(
            [
                {1: 1.4, 2: 3.1},
                {7: -1.2},
                None,
                {11: 3.4, 21: 3.1, 9: 8.1},
                {4: 9.4, 6: 4.1},
                {7: -1.2},
                {},
            ]
        ),
        # nested type
        np.array(
            [
                {1: [3, 1, None], 2: [2, 1]},
                {3: [5], 7: None},
                None,
                {4: [9, 2], 6: [8, 1]},
                {7: [2]},
                {11: [2, -1]},
                {1: [-1]},
                {},
                {21: None, 9: []},
            ]
        ),
    ]
)
def map_arr_value(request):
    return request.param


# there is a memory leak probably due to the decref issue in to_arr_obj_if_list_obj()
# TODO: fix leak and enable test
@pytest.mark.slow
def test_unbox(map_arr_value):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (map_arr_value,))
    check_func(impl2, (map_arr_value,))


@pytest.mark.slow
def test_dtype(map_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (map_arr_value,))


@pytest.mark.slow
def test_nbytes(memory_leak_check):
    """Test MapArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    map_value = np.array(
        [
            {1: 1.4, 2: 3.1},
            {7: -1.2},
            None,
            {11: 3.4, 21: 3.1, 9: 8.1},
            {4: 9.4, 6: 4.1},
            {7: -1.2},
            {},
            {8: 3.3, 5: 6.3},
        ]
    )
    check_func(impl, (map_value,), py_output=253, only_seq=True)
    py_out = 240 + 11 * bodo.get_size()
    if bodo.get_size() == 1:
        py_out += 2
    check_func(impl, (map_value,), py_output=py_out, only_1DVar=True)


def test_map_apply_simple(memory_leak_check):
    """
    Test a simple Series.apply on a map array.
    """

    def impl(df):
        return df["A"].apply(lambda x: x)

    df = pd.DataFrame(
        {"A": [{1: 2, 4: 10, 15: 71, 33: 36, 141: 21, 4214: 2, -1: 0, 0: 0, 5: 2}] * 10}
    )
    check_func(impl, (df,))


def test_map_apply(memory_leak_check):
    """
    Test creating a MapArray from Series.apply.
    This is very similar to what is needed in a customer
    use case.
    """

    def impl(df, keys):
        return df["master_column"].apply(
            lambda row: {x: y for x, y in zip(keys, row.split(","))}
        )

    df1 = pd.DataFrame(
        {"master_column": [",".join([str(i) for i in np.arange(10)])] * 15}
    )
    keys1 = [str(i + 1) for i in np.arange(10)]
    check_func(impl, (df1, keys1))

    df2 = pd.DataFrame(
        {"master_column": [",".join([str(i) for i in np.arange(6000)])] * 15}
    )
    keys2 = [str(i + 1) for i in np.arange(6000)]
    check_func(impl, (df2, keys2))


def test_getitem_bool(map_arr_value):
    """
    Tests using a boolean getitem to select map array values.
    """

    def impl(map_arr, idx):
        return map_arr[idx]

    # Generate the index
    idx = pd.array([True, None, False] * len(map_arr_value))[: len(map_arr_value)]
    check_func(
        impl,
        (map_arr_value, idx),
        py_output=map_arr_value[idx.fillna(False).to_numpy(np.bool_)],
    )


def test_getitem_slice(map_arr_value):
    def test_impl(A, idx):
        return A[idx]

    idx = slice(1, 4)
    check_func(test_impl, (map_arr_value, idx), dist_test=False)


@pytest.mark.parametrize(
    "arr,answer",
    [
        pytest.param(
            bodo.MapArrayType(bodo.int64, bodo.float64), True, id="simple_map_array"
        ),
        pytest.param(bodo.IntegerArrayType(bodo.int64), False, id="simple_false"),
        pytest.param(
            bodo.ArrayItemArrayType(
                bodo.MapArrayType(
                    bodo.IntegerArrayType(bodo.int64),
                    bodo.FloatingArrayType(bodo.float64),
                )
            ),
            True,
            id="map_inside_array",
        ),
        pytest.param(
            bodo.ArrayItemArrayType(
                bodo.ArrayItemArrayType(bodo.IntegerArrayType(bodo.int64))
            ),
            False,
            id="array_false",
        ),
        pytest.param(
            bodo.StructArrayType(
                (
                    bodo.IntegerArrayType(bodo.int64),
                    bodo.MapArrayType(
                        bodo.IntegerArrayType(bodo.int64),
                        bodo.FloatingArrayType(bodo.float64),
                    ),
                ),
                ("ints", "map"),
            ),
            True,
            id="map_inside_struct",
        ),
        pytest.param(
            bodo.StructArrayType(
                (
                    bodo.IntegerArrayType(bodo.int64),
                    bodo.ArrayItemArrayType(bodo.IntegerArrayType(bodo.int64)),
                ),
                ("ints", "array"),
            ),
            False,
            id="struct_false",
        ),
    ],
)
def test_contains_map_array(arr, answer):
    assert contains_map_array(arr) == answer
