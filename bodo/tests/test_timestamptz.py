import pandas as pd

import bodo


def test_timestamptz_array_creation(memory_leak_check):
    """Test creation of TimestampTZ array"""

    # TODO(aneesh) test setting nulls after implementing setna/is_na
    @bodo.jit
    def f():
        arr = bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(5)
        arr[0] = bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100)
        arr[1] = bodo.TimestampTZ(pd.Timestamp("2022-12-31 12:59:59"), 200)
        arr[2] = bodo.TimestampTZ(pd.Timestamp("2024-01-01 00:00:00"), 300)
        return arr

    arr = f()
    assert arr[0].utc_timestamp == pd.Timestamp("2021-01-02 03:04:05")
    assert arr[0].offset_minutes == 100
    assert arr[1].utc_timestamp == pd.Timestamp("2022-12-31 12:59:59")
    assert arr[1].offset_minutes == 200
    assert arr[2].utc_timestamp == pd.Timestamp("2024-01-01 00:00:00")
    assert arr[2].offset_minutes == 300


def test_timestamptz_boxing_unboxing(memory_leak_check):
    """Test boxing and unboxing of TimestampTZ scalar"""

    @bodo.jit
    def f(v):
        return v

    x = f(bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100))
    assert x.utc_timestamp == pd.Timestamp("2021-01-02 03:04:05")
    assert x.offset_minutes == 100


def test_timestamptz_array_boxing_unboxing(memory_leak_check):
    """Test boxing and unboxing of TimestampTZ array"""
    arr = pd.Series(
        [
            bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100),
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
        ]
    )

    @bodo.jit
    def f(arr):
        return arr

    x = f(arr)
    assert x[0].utc_timestamp == pd.Timestamp("2021-01-02 03:04:05")
    assert x[0].offset_minutes == 100
    assert x[1].utc_timestamp == pd.Timestamp("2022-02-03 04:05:06")
    assert x[1].offset_minutes == 200
