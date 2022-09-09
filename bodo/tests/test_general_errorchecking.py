import re

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


def test_undefined_variable():
    message = "name 'undefined_variable' is not defined"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: pd.read_csv(undefined_variable))()


@pytest.mark.slow
def test_fn_return_type_error():
    def test_impl(n):
        if n > 10:
            return "hello world"
        else:
            return False

    message = r"Unable to unify the following function return types.*"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)(3)


@pytest.mark.slow
def test_bcast_scalar_type_error():
    def test_impl():
        return bodo.libs.distributed_api.bcast_scalar(b"I'm a bytes val")

    message = r"bcast_scalar requires an argument of type Integer, Float, datetime64ns, timedelta64ns, string, None, or Bool. Found type.*"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)()


def test_date_range_unsupported(memory_leak_check):
    """
    Tests an unsupported arguement for pd.date_range()
    """

    def test_impl():
        return pd.date_range(start="1/1/2018", periods=5, tz="Asia/Tokyo")

    message = "tz parameter only supports default value None"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)()

    def test_impl2():
        return pd.date_range(start="1/1/2018")

    message = "exactly three must be specified"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl2)()


def test_np_sort_unsupported(memory_leak_check):
    """
    Tests an unsupported arguement for np.sort()
    """

    def test_impl(arr):
        return np.sort(arr, kind="heapsort")

    message = "kind parameter only supports default value None"
    arr = pd.array(np.arange(100), dtype="Int64")
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)(arr)


def test_pd_datetime_unsupported(memory_leak_check):
    """
    Tests an unsupported argument for pd.to_datetime()
    """

    def test_impl():
        return pd.to_datetime("2022-09-02", errors="ignore")

    message = (
        ".*"
        + re.escape(
            "pd.to_datetime(): errors parameter only supports default value raise"
        )
        + ".*"
    )
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)()
