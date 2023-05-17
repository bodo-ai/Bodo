import re

import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param("MyNameis", id="unchanged"),
        pytest.param(":rewre. 342 ", id="some escapes"),
        pytest.param("MyNameisØ", id="2 byte string unchanged"),
        pytest.param("\\) 342 Ø", id="2 byte string changed"),
        pytest.param("MyNameis₤", id="4 byte string unchanged"),
        pytest.param("{ ( 342₤", id="4 byte string changed"),
        pytest.param(
            "screen printed men’s shirt",
            id="[BSE-320] 2 byte string with several escapes",
        ),
        pytest.param(
            "screen printed men₤s shirt",
            id="[BSE-320] 4 byte string with several escapes",
        ),
    ]
)
def escape_patterns(request):
    """Returns the patterns used to test re.escape"""
    return request.param


def test_re_escape(escape_patterns, memory_leak_check):
    """
    Tests re.escape matches Python with a variety of input patterns.
    """

    def impl(arg):
        return re.escape(arg)

    check_func(impl, (escape_patterns,))


def test_re_escape_length(escape_patterns, memory_leak_check):
    """
    Tests the expected length with re.escape. Prompted by
    [BSE-320] which was caused by a bug in re.escape's expected length.
    """

    def impl(arg):
        return bodo.libs.str_ext.re_escape_len(arg)

    check_func(impl, (escape_patterns,), py_output=len(re.escape(escape_patterns)))
