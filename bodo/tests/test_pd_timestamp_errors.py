# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Tests for pd.Timestamp error checking
"""


import re

import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_timestamp_classmethod_err():
    def impl():
        return pd.Timestamp.max

    err_msg = re.escape("pandas.Timestamp.max not supported yet")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_classmethod_local_import_err():
    from pandas import Timestamp

    def impl():
        return Timestamp.max

    err_msg = re.escape("pandas.Timestamp.max not supported yet")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_attr_err():
    def impl():
        return pd.Timestamp("2021-12-08").tz

    err_msg = ".*" + re.escape("pandas.Timestamp.tz not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_method_err():
    def impl():
        return pd.Timestamp("2021-12-08").time()

    err_msg = ".*" + re.escape("pandas.Timestamp.time() not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()
