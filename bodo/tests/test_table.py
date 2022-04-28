# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Test Bodo's Table data type
"""


import numpy as np
import pytest

from bodo.hiframes.table import Table
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        Table(
            [
                np.ones(10),
                np.arange(10),
                np.array(["AB"] * 10),
                np.ones(10) * 3,
                np.arange(10) + 1,
                np.arange(10) + 2,
                np.array(["A B C"] * 10),
            ]
        ),
    ]
)
def table_value(request):
    return request.param


def test_unbox(table_value, memory_leak_check):
    # just unbox
    def impl(t_arg):
        return True

    # unbox and box
    def impl2(t_arg):
        return t_arg

    check_func(impl, (table_value,), only_seq=True)
    check_func(impl2, (table_value,), only_seq=True)


def test_constant_lowering(table_value, memory_leak_check):
    """Test constant lowering for TableType"""

    def test_impl():
        return table_value

    check_func(test_impl, (), only_seq=True)
