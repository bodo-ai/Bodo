import pytest
import pandas as pd
import datetime

import bodo
from bodo.utils.typing import BodoError


def test_add_timestamp_unsupported(memory_leak_check):
    """
    Test adding timestamp to datetime_date_type and verify bodo error raised.
    """

    def impl(a, b):
        return a + b

    ts = pd.Timestamp(
        year=2018,
        month=4,
        day=1,
        hour=5,
        minute=3,
        microsecond=12100,
        second=2,
        nanosecond=42,
    )
    datetime_date = datetime.date(2012, 1, 2)
    with pytest.raises(BodoError, match="operator not supported for data types"):
        bodo.jit(impl)(ts, datetime_date)


def test_sub_timedelta_unsupported(memory_leak_check):
    """
    Test subtracting timedelta and datetime.date and verify bodo error raised.
    """

    def impl(a, b):
        return a - b

    td = pd.Timedelta(232142)
    datetime_date = datetime.date(2012, 1, 2)
    with pytest.raises(BodoError, match="operator not supported for data types"):
        bodo.jit(impl)(td, datetime_date)
