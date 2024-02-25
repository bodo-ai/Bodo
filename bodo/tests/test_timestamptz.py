import numpy as np
import pandas as pd

import bodo
from bodo.tests.utils import _get_dist_arg, check_func


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


# Tests for core distributed api operations


def test_bcast_scalar(memory_leak_check):
    """Test that a scalar is broadcasted correctly"""
    expected = bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100)
    if bodo.get_rank() == 0:
        x = expected
    else:
        x = bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:05:13"), 100)
    result = bodo.libs.distributed_api.bcast_scalar(x)
    # Test the values exactly to ensure no conversion occurs.
    assert result.utc_timestamp == expected.utc_timestamp
    assert result.offset_minutes == expected.offset_minutes


def test_scatterv(memory_leak_check):
    """Test that scatterv works correctly on a given array"""
    arr = np.array(
        [
            bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100),
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
            bodo.TimestampTZ(pd.Timestamp("2023-03-04 05:06:07"), 300),
            bodo.TimestampTZ(pd.Timestamp("2024-04-05 06:07:08"), 400),
            None,
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
        ]
    )
    scattered_arr = bodo.libs.distributed_api.scatterv(arr)
    np.testing.assert_array_equal(
        scattered_arr, _get_dist_arg(arr, False, False, False)
    )


def test_gatherv(memory_leak_check):
    """Test that gatherv works correctly on a given array"""
    expected = np.array(
        [
            bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100),
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
            bodo.TimestampTZ(pd.Timestamp("2023-03-04 05:06:07"), 300),
            bodo.TimestampTZ(pd.Timestamp("2024-04-05 06:07:08"), 400),
            None,
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
        ]
    )
    section = _get_dist_arg(expected, False, False, False)
    gathered_array = bodo.libs.distributed_api.gatherv(section)
    if bodo.get_rank() == 0:
        np.testing.assert_array_equal(expected, gathered_array)


def test_distributed_getitem(memory_leak_check):
    """Test that getitem works correctly on a distributed array"""

    def impl(arr):
        return arr[0]

    arr = np.array(
        [
            bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100),
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
            bodo.TimestampTZ(pd.Timestamp("2023-03-04 05:06:07"), 300),
            bodo.TimestampTZ(pd.Timestamp("2024-04-05 06:07:08"), 400),
            None,
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
        ]
    )
    py_output = arr[0]
    check_func(impl, (arr,), py_output=py_output)


def test_distributed_scalar_optional_getitem(memory_leak_check):
    """Test that getitem works correctly on a distributed array"""

    def impl(arr, i):
        return bodo.utils.indexing.scalar_optional_getitem(arr, i)

    arr = np.array(
        [
            bodo.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100),
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
            bodo.TimestampTZ(pd.Timestamp("2023-03-04 05:06:07"), 300),
            bodo.TimestampTZ(pd.Timestamp("2024-04-05 06:07:08"), 400),
            None,
            bodo.TimestampTZ(pd.Timestamp("2022-02-03 04:05:06"), 200),
        ]
    )
    for i in range(len(arr)):
        py_output = arr[i]
        check_func(impl, (arr, i), py_output=py_output)
