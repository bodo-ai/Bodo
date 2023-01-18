# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL like functions
"""
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


def test_convert_sql_pattern_to_python_compile_time(memory_leak_check):
    """
    Tests that convert_sql_pattern_to_python_compile_time will properly convert patterns and
    detect potential optimizations. This checks both the boolean values and returned pattern.

    Note: This only tests converting patterns that do not escape any characters.
    """
    func = (
        bodo.libs.bodosql_like_array_kernels.convert_sql_pattern_to_python_compile_time
    )
    # Escape is not used in these tests.
    escape = ""
    # Test % and _ in the middle
    assert func("hei%kje_0", escape, False) == ("^hei.*kje.0$", True, True, True, False)
    # Test just a % in the middle
    assert func("lfew%342432%3242", escape, False) == (
        "^lfew.*342432.*3242$",
        True,
        True,
        True,
        False,
    )

    # Test all %
    assert func("%%%%", escape, False) == ("", False, False, False, True)

    # Test a leading and trailing %
    assert func("%%rewREwr%%", escape, False) == ("rewREwr", False, False, False, False)

    # Test a leading and trailing % with % in the middle
    assert func("%%rewr%ewr%%", escape, False) == (
        "rewr.*ewr",
        True,
        False,
        False,
        False,
    )

    # Test a leading and trailing % with _ in the middle
    assert func("%%rewr_ewr%%", escape, False) == (
        "rewr.ewr",
        True,
        False,
        False,
        False,
    )

    # Test just a leading %
    assert func("%%rewrewr", escape, False) == ("rewrewr", False, False, True, False)

    # Test just a leading % with % in the middle
    assert func("%%rewr%ewr", escape, False) == ("rewr.*ewr$", True, False, True, False)

    # Test just a leading % with _ in the middle
    assert func("%%rewr_ewr", escape, False) == ("rewr.ewr$", True, False, True, False)

    # Test just a trailing %
    assert func("rewrewr%", escape, False) == ("rewrewr", False, True, False, False)

    # Test just a trailing % with % in the middle
    assert func("r%ewrewr%", escape, False) == ("^r.*ewrewr", True, True, False, False)

    # Test just a trailing % with _ in the middle
    assert func("r_Ewrewr%", escape, False) == ("^r.Ewrewr", True, True, False, False)

    # Test no wildcards
    assert func("r3243", escape, False) == ("r3243", False, True, True, False)

    # Test only _ as a wild card
    assert func("er2r_", escape, False) == ("^er2r.$", True, True, True, False)

    # Test all _
    assert func("___", escape, False) == ("^...$", True, True, True, False)

    # empty string
    assert func("", escape, False) == ("", False, True, True, False)

    # Check for a required escape character with regex
    assert func("r3%2.43", escape, False) == ("^r3.*2\\.43$", True, True, True, False)

    # Check for a required escape character without regex
    assert func("r32.43", escape, False) == ("r32.43", False, True, True, False)

    # Check that the pattern will be converted to lower case.
    assert func("FEW232ncew\\Bef", escape, True) == (
        "few232ncew\\bef",
        False,
        True,
        True,
        False,
    )


def test_convert_sql_pattern_to_python_with_escapes_compile_time(memory_leak_check):
    """
    Tests that convert_sql_pattern_to_python_compile_time will properly convert patterns and
    detect potential optimizations with escapes. This checks both the boolean values and returned pattern.

    Note: This only tests converting patterns that have escape characters and therefore may convert
    wildcards to regular characters.
    """
    func = (
        bodo.libs.bodosql_like_array_kernels.convert_sql_pattern_to_python_compile_time
    )
    # Basic wildcard escape support
    assert func("r_rew%ke^%3", "^", False) == ("^r.rew.*ke%3$", True, True, True, False)

    # Escape without wildcard
    assert func("rewrfw^rewr", "^", False) == ("rewrfw^rewr", False, True, True, False)

    # Escape at the end with wildcard
    assert func("rewrfw\\rewr\\_", "\\", False) == (
        "rewrfw\\rewr_",
        False,
        True,
        True,
        False,
    )
    # Escape at the end without wildcard
    assert func("rewrfw\\_rewr\\", "\\", False) == (
        "rewrfw_rewr\\",
        False,
        True,
        True,
        False,
    )
    # Escape at the start with wildcard
    assert func("\\%_%%%%%%", "\\", False) == ("^%.", True, True, False, False)
    # Escape at the start without wildcard
    assert func("^ffewf_%", "^", False) == ("^\\^ffewf.", True, True, False, False)
    # All %, except one is escaped
    assert func("%%%%%%^%%%%%", "^", False) == ("%", False, False, False, False)
    # Only escaped wildcard
    assert func("^%", "^", False) == ("%", False, True, True, False)
    # Start + escaped wildcard
    assert func("%%%%^%fe", "^", False) == ("%fe", False, False, True, False)
    # End + escaped wildcard
    assert func("fe^%%%%%%", "^", False) == ("fe%", False, True, False, False)
    # Test that the escape only matches on the correct case even case insensitive
    assert func("Fa%A%", "A", True) == ("^fa.*%$", True, True, True, False)


@pytest.mark.parametrize(
    "is_case_insensitive",
    [
        False,
        True,
    ],
)
def test_like_constant_pattern_escape(is_case_insensitive, memory_leak_check):
    """
    Test all of the various paths for LIKE/ILIKE from the like_kernel with a constant
    pattern and escape value.
    """
    # Tests without escape
    def test_impl1(arr):
        # Test directly equality
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "helLo", "", is_case_insensitive
        )

    def test_impl2(arr):
        # Test ends with
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "%Lo", "", is_case_insensitive
        )

    def test_impl3(arr):
        # Test starts with
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "H%", "", is_case_insensitive
        )

    def test_impl4(arr):
        # Test contains
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "%elL%", "", is_case_insensitive
        )

    def test_impl5(arr):
        # Test a regex
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "he_Lo", "", is_case_insensitive
        )

    # Same tests but with escape
    def test_impl6(arr):
        # Test directly equality
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "he^%Lo", "^", is_case_insensitive
        )

    def test_impl7(arr):
        # Test ends with
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "%>%Lo", ">", is_case_insensitive
        )

    def test_impl8(arr):
        # Test starts with
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "He<%%", "<", is_case_insensitive
        )

    def test_impl9(arr):
        # Test contains
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "%E&%l%", "&", is_case_insensitive
        )

    def test_impl10(arr):
        # Test a regex
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "he^__Lo", "^", is_case_insensitive
        )

    S = pd.Series(
        [
            "hello",
            None,
            "heLLo",
            "helLo",
            "World",
            "he^lo",
            "hel^lO",
            "he_lLO",
            "he_xLO",
            "HELLO",
            "HE%LO",
            "E&lo",
            "HELLOOOOOOO",
            "human",
            None,
            "H",
            "E",
            "L",
            "L",
            "O",
            "^",
            "_",
            "%",
        ]
    )
    arr = S.values
    # We hardcode each of the answer to avoid an overly complicated expected output
    # that just matches the Python code.
    if is_case_insensitive:
        answer = [
            True,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl1, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            True,
            None,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl2, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            True,
            None,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            None,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            None,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl3, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            True,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl4, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            True,
            None,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl5, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl6, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl7, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl8, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl9, (arr,), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    else:
        answer = [
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    check_func(test_impl10, (arr,), py_output=pd.array(answer))

    # Test once with a scalar
    scalar = arr[3]
    check_func(test_impl1, (scalar,), py_output=True)
