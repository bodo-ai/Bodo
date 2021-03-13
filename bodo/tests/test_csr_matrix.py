# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Tests for scipy.sparse.csr_matrix data structure
"""

import numpy as np
import pytest
import scipy.sparse

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
    # check_func(impl2, (csr_matrix_value,))
