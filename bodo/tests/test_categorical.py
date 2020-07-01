# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Tests for pd.CategoricalDtype/pd.Categorical  functionality
"""
import pandas as pd
import numpy as np
import bodo
import pytest
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "dtype",
    [
        pd.CategoricalDtype(["AA", "B", "CC"]),
        pd.CategoricalDtype(["CC", "AA", "B"], True),
        pd.CategoricalDtype([3, 2, 1, 4]),
    ],
)
def test_unbox_dtype(dtype, memory_leak_check):
    # just unbox
    def impl(dtype):
        return True

    check_func(impl, (dtype,))

    # unbox and box
    def impl2(dtype):
        return dtype

    check_func(impl2, (dtype,))
