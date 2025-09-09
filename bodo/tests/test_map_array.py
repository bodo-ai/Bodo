"""Tests for array of map values."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, get_num_test_workers


@pytest.fixture(
    params=[
        # simple types to handle in C
        pd.array(
            [
                {1: 1.4, 2: 3.1},
                {7: -1.2},
                None,
                {11: 3.4, 21: 3.1, 9: 8.1},
                {4: 9.4, 6: 4.1},
                {7: -1.2},
                {},
            ],
            pd.ArrowDtype(pa.map_(pa.int32(), pa.float32())),
        ),
        # nested type
        pd.array(
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
            ],
            pd.ArrowDtype(pa.map_(pa.int32(), pa.list_(pa.int32()))),
        ),
    ]
)
def map_arr_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(map_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (map_arr_value,))
    check_func(impl2, (map_arr_value,))


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
    check_func(impl, (map_value,), py_output=255, only_seq=True)
    n_pes = get_num_test_workers()
    py_out = 240 + 12 * n_pes
    if n_pes == 1:
        py_out += 3
    check_func(impl, (map_value,), py_output=py_out, only_1DVar=True)


def test_map_apply_simple(memory_leak_check):
    """
    Test a simple Series.apply on a map array.
    """

    def impl(df):
        return df["A"].apply(lambda x: x)

    df = pd.DataFrame(
        {
            "A": pd.Series(
                [{1: 2, 4: 10, 15: 71, 33: 36, 141: 21, 4214: 2, -1: 0, 0: 0, 5: 2}]
                * 10,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.int64())),
            )
        }
    )
    check_func(impl, (df,))


def test_map_apply(memory_leak_check):
    """
    Test creating a MapArray from Series.apply.
    This is very similar to what is needed in a customer
    use case.
    """

    def impl(df, keys):
        return df["master_column"].apply(lambda row: dict(zip(keys, row.split(","))))

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


def test_getitem_int(map_arr_value):
    """
    Tests using a int getitem to select map array values.
    """

    def impl(map_arr, idx):
        return map_arr[idx]

    idx = 1
    check_func(
        impl, (map_arr_value, idx), py_output=dict(map_arr_value[idx]), only_seq=True
    )


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
            bodo.types.MapArrayType(bodo.types.int64, bodo.types.float64),
            True,
            id="simple_map_array",
        ),
        pytest.param(
            bodo.types.IntegerArrayType(bodo.types.int64), False, id="simple_false"
        ),
        pytest.param(
            bodo.types.ArrayItemArrayType(
                bodo.types.MapArrayType(
                    bodo.types.IntegerArrayType(bodo.types.int64),
                    bodo.types.FloatingArrayType(bodo.types.float64),
                )
            ),
            True,
            id="map_inside_array",
        ),
        pytest.param(
            bodo.types.ArrayItemArrayType(
                bodo.types.ArrayItemArrayType(
                    bodo.types.IntegerArrayType(bodo.types.int64)
                )
            ),
            False,
            id="array_false",
        ),
        pytest.param(
            bodo.types.StructArrayType(
                (
                    bodo.types.IntegerArrayType(bodo.types.int64),
                    bodo.types.MapArrayType(
                        bodo.types.IntegerArrayType(bodo.types.int64),
                        bodo.types.FloatingArrayType(bodo.types.float64),
                    ),
                ),
                ("ints", "map"),
            ),
            True,
            id="map_inside_struct",
        ),
        pytest.param(
            bodo.types.StructArrayType(
                (
                    bodo.types.IntegerArrayType(bodo.types.int64),
                    bodo.types.ArrayItemArrayType(
                        bodo.types.IntegerArrayType(bodo.types.int64)
                    ),
                ),
                ("ints", "array"),
            ),
            False,
            id="struct_false",
        ),
    ],
)
def test_contains_map_array(arr, answer):
    from bodo.libs.map_arr_ext import contains_map_array

    assert contains_map_array(arr) == answer
