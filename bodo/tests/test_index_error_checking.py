# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Tests for pd.Index error checking
"""

import pandas as pd
import pytest
import bodo
from bodo.utils.typing import BodoError

def test_object_dtype(memory_leak_check):
    """
    Test that providing object as the dtype raises a reasonable
    error.
    """
    def impl():
        a =  pd.Index(['a', 'b', 'c'], dtype='object')
        return a[0]

    with pytest.raises(
        BodoError,
        match="pd.Index\\(\\) object 'dtype' is not specific enough for typing. Please provide a more exact type \\(e.g. str\\)."
    ):
        bodo.jit(impl)()
