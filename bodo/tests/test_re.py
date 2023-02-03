import re

import pytest

from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("MyNameis", id="unchanged"),
        pytest.param(":rewre. 342 ", id="some escapes"),
        pytest.param("MyNameisØ", id="2 byte string unchanged"),
        pytest.param("\\) 342 Ø", id="2 byte string changed"),
        pytest.param("MyNameis₤", id="4 byte string unchanged"),
        pytest.param("{ ( 342₤", id="4 byte string changed"),
    ],
)
def test_re_escape(pattern, memory_leak_check):
    """
    Tests re.escape matches Python with a variety of input patterns.
    """

    def impl(arg):
        return re.escape(arg)

    check_func(impl, (pattern,))
