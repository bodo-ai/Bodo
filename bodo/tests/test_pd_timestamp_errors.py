# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for pd.Timestamp error checking
"""


import re

import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_timestamp_classmethod_err(memory_leak_check):
    def impl():
        return pd.Timestamp.max

    err_msg = re.escape("pandas.Timestamp.max not supported yet")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_classmethod_local_import_err(memory_leak_check):
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
def test_timestamp_attr_err(memory_leak_check):
    def impl():
        return pd.Timestamp("2021-12-08").tz

    err_msg = ".*" + re.escape("pandas.Timestamp.tz not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_timestamp_method_err(memory_leak_check):
    def impl():
        return pd.Timestamp("2021-12-08").time()

    err_msg = ".*" + re.escape("pandas.Timestamp.time() not supported yet") + ".*"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_timestamp_day_name_unsupported(memory_leak_check):
    """
    Test unsupported arguments for Timestamp.day_name
    """

    def impl():
        return pd.Timestamp("2020-03-14T15:32:52.192548651").day_name(
            locale="en_US.utf8"
        )

    err_msg = "locale parameter only supports default value None"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()
