# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    check_func,
)

np.random.seed(0)


def test_unary_ufunc():
    ufunc = np.invert
    def test_impl(A):
        return ufunc(A.values)

    A = pd.Series([False, True, True, False, False])
    check_func(test_impl, (A,))
