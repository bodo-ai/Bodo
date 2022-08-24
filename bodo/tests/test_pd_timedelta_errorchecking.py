# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test File for pd_timedelta types. Checks that proper error messages are raised.
"""
import re

import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_timedelta_classmethod_err():
    """picking an arbitrary class method to test. If one of them works,
    all of them should"""

    def impl():
        return pd.Timedelta.resolution

    err_msg = re.escape("pandas.Timedelta.resolution not yet supported")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timedelta_classmethod_local_import_err():
    """picking an arbitrary class method to test the local import case. If one of them works,
    all of them should"""
    from pandas import Timedelta

    def impl():
        return Timedelta.resolution

    err_msg = re.escape("pandas.Timedelta.resolution not yet supported")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_attr_err():
    """picking an atribute to test. If one of them works, all of them should"""

    def impl():
        return pd.Timedelta(1000000).is_populated

    err_msg = ".*" + re.escape("pandas.Timedelta.is_populated not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_method_err():
    """picking an method to test. If one of them works, all of them should"""

    def impl():
        return pd.Timedelta(100000).isoformat()

    err_msg = ".*" + re.escape("pandas.Timedelta.isoformat() not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()
