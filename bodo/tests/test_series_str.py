import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


def test_len():
    def test_impl(S):
        return S.str.len()

    S = pd.Series([' bbCD ', 'ABC', ' mCDm ', np.nan, 'abcffcc', '', 'A'])
    check_func(test_impl, (S,), check_dtype=False)
