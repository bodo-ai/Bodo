"""Test Spark parity functions that are in objmode """
import zlib

import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.mark.slow
def test_zlib_crc32():
    """Test zlib.crc32"""

    def test_impl1():
        return zlib.crc32(b"str123")

    check_func(test_impl1, (), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize("str_val", ["ABC", "1234", "abcdE!FG"])
def test_zlib_crc32_str(str_val):
    """Test zlib.crc32 with byte strings"""

    def test_impl(byte_vals):
        return zlib.crc32(byte_vals)

    check_func(test_impl, (str_val.encode("utf-8"),), dist_test=False)
