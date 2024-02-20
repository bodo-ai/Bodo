# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL like functions
"""
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


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


@pytest.fixture(params=[False, True])
def is_case_insensitive(request):
    return request.param


@pytest.fixture(
    params=[
        # Note: This array has length 24
        pd.Series(
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
                None,
            ]
        ).values
    ]
)
def like_arr(request):
    return request.param


def test_like_constant_pattern_escape(is_case_insensitive, like_arr, memory_leak_check):
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

    arr = like_arr
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
        ]

    check_func(test_impl10, (arr,), py_output=pd.array(answer))

    # Test once with a scalar
    scalar = arr[3]
    check_func(test_impl1, (scalar,), py_output=True)


def test_like_scalar_pattern_escape(is_case_insensitive, like_arr, memory_leak_check):
    """
    Test all of the various paths for LIKE/ILIKE from the like_kernel with a scalar
    pattern and escape value. These test the same tests as test_like_constant_pattern_escape
    but the pattern and escape aren't always constant.
    """

    def test_impl1(arr, pattern):
        # Test with only a pattern
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, pattern, "", is_case_insensitive
        )

    def test_impl2(arr, escape):
        # Test with only an escape
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "he^%Lo", escape, is_case_insensitive
        )

    def test_impl3(arr, pattern, escape):
        # Test with a pattern and escape
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, pattern, escape, is_case_insensitive
        )

    arr = like_arr
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
            None,
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
            None,
        ]

    # Test the equality case
    check_func(test_impl1, (arr, "helLo"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the endswith case
    check_func(test_impl1, (arr, "%Lo"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the startswith case
    check_func(test_impl1, (arr, "H%"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the contains case.
    check_func(test_impl1, (arr, "%elL%"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the regex case
    check_func(test_impl1, (arr, "he_Lo"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test once with just an escape to ensure its tested.
    # Tests the equality case
    check_func(test_impl2, (arr, "^"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the endswith case
    check_func(test_impl3, (arr, "%>%Lo", ">"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the startswith case
    check_func(test_impl3, (arr, "He<%%", "<"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the contains case
    check_func(test_impl3, (arr, "%E&%l%", "&"), py_output=pd.array(answer))

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
            None,
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
            None,
        ]

    # Test the regex case
    check_func(test_impl3, (arr, "he^__Lo", "^"), py_output=pd.array(answer))

    # Test once with a scalar
    scalar = arr[3]
    check_func(test_impl1, (scalar, "helLo"), py_output=True)


def test_like_arr_pattern_escape(is_case_insensitive, like_arr, memory_leak_check):
    """
    Test for LIKE/ILIKE from the like_kernel where either
    the pattern or the escape is an array. This mixes several types
    of patterns together.
    """
    arr = like_arr
    pattern_arr = pd.Series(
        [
            None,
            "helLo",
            None,
            "he^__Lo",
            "%E&%l%",
            "%>%Lo",
            "He<%%",
            "he^%Lo",
            "he_Lo",
            "%elL%",
            "H%",
            "%lO",
        ]
        * 2
    ).values
    escape_arr = pd.Series(["", "&", None, "^"] * 6).values

    def test_impl1(arr, pattern):
        # Test with only a pattern
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, pattern, "", is_case_insensitive
        )

    def test_impl2(arr, escape):
        # Test with only an escape
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, "he^%Lo", escape, is_case_insensitive
        )

    def test_impl3(arr, pattern, escape):
        # Test with a pattern and escape
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arr, pattern, escape, is_case_insensitive
        )

    if is_case_insensitive:
        answer = [
            None,
            None,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            None,
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
            None,
        ]
    else:
        answer = [
            None,
            None,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            None,
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
            None,
        ]

    check_func(test_impl1, (arr, pattern_arr), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            False,
            None,
            None,
            False,
            False,
            True,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            None,
        ]
    else:
        answer = [
            False,
            None,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            None,
        ]

    check_func(test_impl2, (arr, escape_arr), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            None,
            None,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            True,
            None,
            True,
            None,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            None,
        ]
    else:
        answer = [
            None,
            None,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            None,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            None,
            None,
        ]

    check_func(test_impl3, (arr, pattern_arr, escape_arr), py_output=pd.array(answer))

    if is_case_insensitive:
        answer = [
            None,
            True,
            None,
            False,
            False,
            False,
            None,
            False,
            True,
            True,
            None,
            True,
            None,
            True,
            None,
            False,
            False,
            False,
            None,
            False,
            True,
            True,
            None,
            True,
        ]
    else:
        answer = [
            None,
            True,
            None,
            False,
            False,
            False,
            None,
            False,
            True,
            True,
            None,
            False,
            None,
            True,
            None,
            False,
            False,
            False,
            None,
            False,
            True,
            True,
            None,
            False,
        ]
    # Test once with a scalar
    scalar = arr[3]
    check_func(
        test_impl3, (scalar, pattern_arr, escape_arr), py_output=pd.array(answer)
    )


def test_like_kernel_optional(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodo.libs.bodosql_array_kernels.like_kernel(
            arg0, arg1, arg2, False
        ), bodo.libs.bodosql_array_kernels.like_kernel(arg0, arg1, arg2, True)

    arg = "maximum"
    pattern = "%Um"
    escape = "^"
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                answer = (False, True) if flag0 and flag1 and flag2 else (None, None)
                check_func(
                    impl,
                    (arg, pattern, escape, flag0, flag1, flag2),
                    py_output=answer,
                )
