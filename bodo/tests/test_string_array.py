# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


def test_string_dtype():
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.StringDtype(),))

    # constructor
    def impl2():
        return pd.StringDtype()

    check_func(impl2, ())
