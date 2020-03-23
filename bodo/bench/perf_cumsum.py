# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Measure performance of various operations that uses the unordered_map/unordered_set
"""
import pandas as pd
import numpy as np
import bodo
import random
import time
import string
import pytest
from bodo.tests.utils import check_timing_func


def test_median_large_random_numpy():
    def get_random_array(n, sizlen):
        elist = []
        for i in range(n):
            eval = random.randint(1, sizlen)
            if eval == 1:
                eval = None
            elist.append(eval)
        return np.array(elist, dtype=np.float64)

    def impl1(df):
        A = df.groupby("A")["B"].cumsum()
        return A

    random.seed(5)
    nb = 10000000
    df1 = pd.DataFrame({"A": get_random_array(nb, 10), "B": get_random_array(nb, 100)})
    check_timing_func(impl1, (df1,))
