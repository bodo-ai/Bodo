# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for array of struct values.
"""

import numpy as np
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        # heterogeneous values
        np.array(
            [
                {"X": 1, "Y": 3.1},
                {"X": 2, "Y": 1.1},
                None,
                {"X": -1, "Y": 7.8},
                {"X": 3, "Y": 4.0},
                {"X": -3, "Y": -1.2},
                {"X": None, "Y": 9.0},
            ]
        ),
        # homogeneous values
        np.array(
            [
                {"X": 1, "Y": 3},
                {"X": 2, "Y": 1},
                None,
                {"X": -1, "Y": -1},
                {"X": 3, "Y": 4},
                {"X": -3, "Y": -1},
                {"X": 5, "Y": 9},
            ]
        ),
        pytest.param(
            np.array(
                [
                    {
                        "X": "AB",
                        "Y": [1.1, 2.2],
                        "Z": [[1], None, [3, None]],
                        "W": {"A": 1, "B": "A"},
                    },
                    {
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                        "W": {"A": 1, "B": "ABC"},
                    },
                    None,
                    {
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                        "W": {"A": 1, "B": ""},
                    },
                    {
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                        "W": {"A": 1, "B": "AA"},
                    },
                    {
                        "X": "LMMM",
                        "Y": [9.0, 1.2, 3.1],
                        "Z": [[10, 11], [11, 0, -3, -5]],
                        "W": {"A": 1, "B": "DFG"},
                    },
                ]
                * 2
                + [
                    # putting structs with NA fields last since typeof() cannot handle NAs
                    # in struct fields yet. TODO: support
                    {"X": "", "Y": None, "Z": [], "W": None},
                    {"X": None, "Y": [], "Z": None, "W": {"A": 11, "B": None}},
                ]
            ),
            marks=pytest.mark.slow,
        ),
        # nested struct data with homogeneous values
        pytest.param(
            np.array(
                [
                    {"X": {"A1": 10, "A2": 2}, "Y": {"B1": -11, "B2": 4}},
                    {"X": {"A1": -19, "A2": 5}, "Y": {"B1": 5, "B2": 19}},
                    {"X": {"A1": -5, "A2": -9}, "Y": {"B1": -15, "B2": 13}},
                    {"X": {"A1": -12, "A2": 2}, "Y": {"B1": 14, "B2": 2}},
                    {"X": {"A1": 17, "A2": -12}, "Y": {"B1": 14, "B2": -18}},
                    {"X": {"A1": 17, "A2": 10}, "Y": {"B1": -13, "B2": 18}},
                ]
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def struct_arr_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(struct_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (struct_arr_value,))
    check_func(impl2, (struct_arr_value,))


def test_getitem_int(struct_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    check_func(test_impl, (struct_arr_value, 1), dist_test=False)
    check_func(test_impl, (struct_arr_value, -1), dist_test=False)


def test_getitem_bool(struct_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(0)
    ind = np.random.ranf(len(struct_arr_value)) < 0.2
    check_func(test_impl, (struct_arr_value, ind), dist_test=False)


def test_getitem_slice(struct_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    check_func(test_impl, (struct_arr_value, ind), dist_test=False)


@pytest.mark.smoke
def test_rec_getitem(struct_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]["Y"]

    i = 1
    check_func(test_impl, (struct_arr_value, i), dist_test=False)


@pytest.mark.smoke
def test_rec_setitem(memory_leak_check):
    def test_impl(A, i, val):
        r = A[i]
        r["Y"] = val
        return r

    i = 1
    val = 8
    struct_arr_value = np.array(
        [
            {"X": 1, "Y": 3.1},
            {"X": 2, "Y": 1.1},
            None,
            {"X": -1, "Y": -1.1},
            {"X": 3, "Y": 4.0},
            {"X": -3, "Y": -1.2},
            {"X": 5, "Y": 9.0},
        ]
    )
    assert bodo.jit(test_impl)(struct_arr_value, i, val) == test_impl(
        struct_arr_value, i, val
    )


"""
def test_setitem_none_int(struct_arr_value, memory_leak_check):
    def test_impl(A, i):
        A[i] = None
        return A

    i = 1
    check_func(test_impl, (struct_arr_value.copy(), i), copy_input=True)
"""


@pytest.mark.slow
def test_ndim(struct_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    assert bodo.jit(test_impl)(struct_arr_value) == test_impl(struct_arr_value)


@pytest.mark.slow
def test_shape(struct_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    assert bodo.jit(test_impl)(struct_arr_value) == test_impl(struct_arr_value)


@pytest.mark.slow
def test_dtype(memory_leak_check):
    def test_impl(A):
        return A.dtype

    A = np.array(
        [
            {"X": 1, "Y": 3.1},
            {"X": 2, "Y": 1.1},
            None,
            {"X": -1, "Y": 7.8},
            {"X": 3, "Y": 4.0},
            {"X": -3, "Y": -1.2},
            {"X": None, "Y": 9.0},
        ]
    )
    assert bodo.jit(test_impl)(A) == test_impl(A)


@pytest.mark.slow
def test_copy(struct_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (struct_arr_value,))


@pytest.mark.slow
def test_len(struct_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (struct_arr_value,))


@pytest.mark.slow
def test_struct_len(struct_arr_value, memory_leak_check):
    def test_impl(A, i):
        if A[i] is None:
            return 0
        else:
            return len(A[i])

    check_func(test_impl, (struct_arr_value, 1), dist_test=False)


@pytest.mark.slow
def test_setitem_optional_int(memory_leak_check):
    def test_impl(A, i, flag):
        if flag:
            x = None
        else:
            x = A[0]
        A[i] = x
        return A

    A = np.array(
        [
            {"X": 1, "Y": 3.1},
            {"X": 2, "Y": 1.1},
            None,
            {"X": -1, "Y": 7.8},
            {"X": 3, "Y": 4.0},
            {"X": -3, "Y": -1.2},
            {"X": None, "Y": 9.0},
        ]
    )

    i = 1
    check_func(test_impl, (A, i, False), copy_input=True, dist_test=False)
    check_func(test_impl, (A, i, True), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_setitem_none_int(memory_leak_check):
    def test_impl(A, i):
        A[i] = None
        return A

    A = np.array(
        [
            {"X": 1, "Y": 3.1},
            {"X": 2, "Y": 1.1},
            None,
            {"X": -1, "Y": 7.8},
            {"X": 3, "Y": 4.0},
            {"X": -3, "Y": -1.2},
            {"X": None, "Y": 9.0},
        ]
    )

    i = 1
    check_func(test_impl, (A, i), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_nbytes(memory_leak_check):
    """Test StructArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    struct_value = np.array(
        [
            {"X": 1, "Y": 3},
            {"X": 2, "Y": 1},
            None,
            {"X": -1, "Y": -1},
            {"X": 3, "Y": 4},
            {"X": -3, "Y": -1},
            {"X": 5, "Y": 9},
        ]
    )
    check_func(impl, (struct_value,), py_output=115, only_seq=True)
    py_out = 112 + 3 * bodo.get_size()
    check_func(impl, (struct_value,), py_output=py_out, only_1DVar=True)
