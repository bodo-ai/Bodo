# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Unittests for objmode blocks
"""

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_type_register():
    """test bodo.register_type() including error checking"""
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)
    bodo.register_type("my_type1", df_type1)

    def impl():
        with bodo.objmode(df="my_type1"):
            df = pd.DataFrame({"A": [1, 2, 5]})
        return df

    check_func(impl, ())

    # error checking
    with pytest.raises(BodoError, match="type name 'my_type1' already exists"):
        bodo.register_type("my_type1", df_type1)
    with pytest.raises(BodoError, match="type name should be a string"):
        bodo.register_type(3, df_type1)
    with pytest.raises(BodoError, match="type value should be a valid data type"):
        bodo.register_type("mt", 3)
