# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for scipy.sparse.csr_matrix data structure
"""

import numpy as np
import pytest
import scipy.sparse

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        # array([[1, 0, 2],
        #        [0, 0, 3],
        #        [4, 5, 6]])
        # csr_matrix((data, (row, col)))
        scipy.sparse.csr_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            )
        ),
    ]
)
def csr_matrix_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(csr_matrix_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (csr_matrix_value,))
    check_func(impl2, (csr_matrix_value,))


@pytest.mark.slow
def test_getitem_slice(csr_matrix_value, memory_leak_check):
    """test basic slice getitem for csr matrix"""

    def test_impl(A):
        return A[:, 1:]

    check_func(test_impl, (csr_matrix_value,), dist_test=False)


def test_getitem_1darray(csr_matrix_value, memory_leak_check):
    """test basic 1darray getitem for csr matrix"""

    def test_impl(A):
        idxs = np.array([0, 2], dtype=np.int32)
        return A[idxs]

    check_func(test_impl, (csr_matrix_value,), dist_test=False)


@pytest.mark.slow
def test_len(csr_matrix_value, memory_leak_check):
    """test len(A) for CSR matrix"""

    def test_impl(A):
        return len(A)

    # scipy.sparse.csr_matrix doesn't provide len() but we support it for consistency
    assert bodo.jit(test_impl)(csr_matrix_value) == csr_matrix_value.shape[0]


@pytest.mark.slow
def test_ndim(csr_matrix_value, memory_leak_check):
    """test A.ndim for CSR matrix"""

    def test_impl(A):
        return A.ndim

    assert bodo.jit(test_impl)(csr_matrix_value) == test_impl(csr_matrix_value)


@pytest.mark.slow
def test_copy(csr_matrix_value, memory_leak_check):
    """test A.copy() for CSR matrix"""

    def test_impl(A):
        return A.copy()

    check_func(test_impl, (csr_matrix_value,))
