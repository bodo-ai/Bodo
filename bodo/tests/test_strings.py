# Copyright (C) 2022 Bodo Inc. All rights reserved.
# -*- coding: utf-8 -*-

import gc
import glob
import os
import re
import unittest

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.str_arr_ext import str_arr_from_sequence
from bodo.tests.utils import (
    check_func,
    gen_nonascii_list,
    generate_comparison_ops_func,
)
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_int_hex(memory_leak_check):
    def test_impl(int_val):
        return hex(int_val)

    check_func(test_impl, (999,))
    check_func(test_impl, (0,))
    check_func(test_impl, (11,))
    check_func(test_impl, (16,))
    check_func(test_impl, (256,))
    check_func(test_impl, (-1313,))
    check_func(test_impl, (256,))


@pytest.mark.slow
def test_string_float64_cast(memory_leak_check):
    def test_impl(_str):
        return np.float64(_str)

    check_func(test_impl, ("12.2",))


@pytest.mark.parametrize(
    "cast_func",
    [
        pytest.param(np.int64, marks=pytest.mark.slow),
        np.uint64,
        pytest.param(np.int32, marks=pytest.mark.slow),
        pytest.param(np.uint32, marks=pytest.mark.slow),
        pytest.param(np.int16, marks=pytest.mark.slow),
        pytest.param(np.uint16, marks=pytest.mark.slow),
        pytest.param(np.int8, marks=pytest.mark.slow),
        pytest.param(np.uint8, marks=pytest.mark.slow),
    ],
)
def test_string_integer_cast(cast_func, memory_leak_check):
    def test_impl(_str):
        return cast_func(_str)

    check_func(test_impl, ("13",))


@pytest.mark.slow
def test_float_str_0_cast(memory_leak_check):
    def test_impl(_float):
        return str(_float)

    check_func(test_impl, (0.0,))


@pytest.mark.slow
def test_float_str_lt1_cast(memory_leak_check):
    def test_impl(_float):
        return str(_float)

    # Always choose 6 decimal places because decimal places will differ
    # in Python
    check_func(test_impl, (0.013952,))


@pytest.mark.slow
def test_float_str_abslt1_cast(memory_leak_check):
    def test_impl(_float):
        return str(_float)

    # Always choose 6 decimal places because decimal places will differ
    # in Python
    check_func(test_impl, (-0.035312,))


@pytest.mark.slow
def test_string_float32_cast(memory_leak_check):
    def test_impl(_str):
        return np.float32(_str)

    check_func(test_impl, ("12.2",))


@pytest.mark.slow
def test_string_base10_cast(memory_leak_check):
    def test_impl(_str):
        return int(_str, base=10)

    check_func(test_impl, ("12",))


@pytest.mark.slow
def test_string_base2_cast(memory_leak_check):
    def test_impl(_str):
        return int(_str, base=2)

    check_func(test_impl, ("10",))


@pytest.mark.slow
def test_string_base16_cast(memory_leak_check):
    def test_impl(_str):
        return int(_str, base=16)

    check_func(test_impl, ("a2432c",))


def test_cmp_binary_op(cmp_op, memory_leak_check):
    func = generate_comparison_ops_func(cmp_op)

    A1 = pd.array(["A", np.nan, "CC", "DD", np.nan, "ABC"])
    A2 = pd.array(["A", np.nan, "CCD", "AADD", "DAA", "ABCE"])
    check_func(func, (A1, A2))
    check_func(func, (A1, "DD"))
    check_func(func, ("CCD", A2))


def test_f_strings():
    """test f-string support, which requires bytecode handling"""
    # requires formatting (FORMAT_VALUE) and concatenation (BUILD_STRINGS)
    def impl1(a):
        return f"AA_{a+3}_B"

    # does not require concatenation
    def impl2(a):
        return f"{a+2}"

    # includes format spec
    def impl3(a):
        return f"{a:0.0%}"

    check_func(impl1, (3,))
    check_func(impl2, (4,))
    check_func(impl3, (12,))


def test_get_list_string(memory_leak_check):
    """In this test bodo returns the array [] when pandas returns NaN in that case.
    We are forced to implement things in this way since type stability is required for
    numba"""

    def test_impl(df1):
        value = df1["A"].iat[1]
        return value

    df1 = pd.DataFrame({"A": [["A"], np.nan, ["AB", "CD"]]})
    np.testing.assert_array_equal(bodo.jit(test_impl)(df1), [])


@pytest.mark.parametrize(
    "ind",
    [
        np.array([True, True, False, False, False, True]),
        np.array([0, 1, 3, 4]),
        slice(1, 5),
    ],
)
def test_string_array_getitem_na(ind, memory_leak_check):
    def impl(S, index):
        return S.iloc[index]

    bodo_func = bodo.jit(impl)
    S = pd.Series((["A", np.nan, "CC", "DD"] + gen_nonascii_list(2)))
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind), check_dtype=False)
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind), check_dtype=False)


##########################  Test re support  ##########################


@pytest.fixture(params=(["AB", "A_B", "A_B_C"] + gen_nonascii_list(3)))
def test_in_str(request, memory_leak_check):
    return request.param


def _assert_match_equal(m1, m2):
    """make sure Match objects are equal since equality doesn't work for them"""
    assert (m1 is None and m2 is None) or m1.span() == m2.span()


def test_re_search(test_in_str, memory_leak_check):
    """make sure re.search returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return re.search(pat, in_str)

    pat = "_"
    py_out = test_impl(pat, test_in_str)
    bodo_out = bodo.jit(test_impl)(pat, test_in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


def test_re_flags(memory_leak_check):
    """make sure re flags like re.I work properly"""

    # flags are passed in
    def impl1(pat, in_str, flags):
        return re.search(pat, in_str, flags)

    # flags are globals
    my_flags = re.I | re.S

    def impl2(pat, in_str):
        return re.search(pat, in_str, my_flags)

    # flags are created inside JIT
    def impl3(pat, in_str):
        return re.search(pat, in_str, re.I | re.S)

    args = ("ABC", "abc", re.I)
    _assert_match_equal(impl1(*args), bodo.jit(impl1)(*args))
    args = ("AB.C", "ab\nc", re.I | re.S)
    _assert_match_equal(impl1(*args), bodo.jit(impl1)(*args))

    args = ("AB.C", "ab\nc")
    _assert_match_equal(impl2(*args), bodo.jit(impl2)(*args))
    _assert_match_equal(impl3(*args), bodo.jit(impl3)(*args))


def test_re_match_cast_bool(test_in_str, memory_leak_check):
    """make sure re.search() output can behave like None in conditionals"""

    def test_impl(pat, in_str):
        m = re.search(pat, in_str)
        if m:
            return 1
        return 0

    pat = "_"
    assert test_impl(pat, test_in_str) == bodo.jit(test_impl)(pat, test_in_str)


def test_re_match_check_none(test_in_str, memory_leak_check):
    """make sure re.Match object can be checked for None"""

    def test_impl(pat, in_str):
        m = re.search(pat, in_str)
        if m is None:
            return 1
        return 0

    pat = "_"
    assert test_impl(pat, test_in_str) == bodo.jit(test_impl)(pat, test_in_str)


def test_re_pat_search(test_in_str, memory_leak_check):
    """make sure Pattern.search returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return pat.search(in_str)

    pat = re.compile(r"_")
    py_out = test_impl(pat, test_in_str)
    bodo_out = bodo.jit(test_impl)(pat, test_in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "A_B", "AB_C"])
def test_re_match(in_str, memory_leak_check):
    """make sure re.match returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return re.match(pat, in_str)

    pat = "AB"
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "AB_", "A_B_C"])
def test_re_pat_match(in_str, memory_leak_check):
    """make sure Pattern.match returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return pat.match(in_str)

    pat = re.compile(r"AB")
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "A_B", "AB_C"])
def test_re_fullmatch(in_str, memory_leak_check):
    """make sure re.fullmatch returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return re.fullmatch(pat, in_str)

    pat = "AB"
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "AB_", "A_B_C"])
def test_re_pat_fullmatch(in_str, memory_leak_check):
    """make sure Pattern.fullmatch returns None or a proper re.Match"""

    def test_impl(pat, in_str):
        return pat.fullmatch(in_str)

    pat = re.compile(r"AB")
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


def test_re_split(memory_leak_check):
    """make sure re.split returns proper output (list of strings)"""

    def test_impl(pat, in_str):
        return re.split(pat, in_str)

    pat = r"\W+"
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


def test_pat_split(memory_leak_check):
    """make sure Pattern.split returns proper output (list of strings)"""

    def test_impl(pat, in_str):
        return re.split(pat, in_str)

    pat = re.compile(r"\W+")
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


# TODO(ehsan): investigate adding memory_leak_check, could be Numba's memory leak for
# exceptions
def test_re_findall():
    """make sure re.findall returns proper output (list of strings)"""

    def test_impl(pat, in_str):
        return re.findall(pat, in_str)

    pat = r"\w+"
    in_str = "Words, words, words."
    check_func(test_impl, (pat, in_str), only_seq=True)

    # an error should be raised with multiple groups if pattern is not constant
    pat = r"(\w+).*(\d+)"
    in_str = "ww 132"
    with pytest.raises(BodoError, match="pattern string should be constant"):
        bodo.jit(test_impl)(pat, in_str)

    # constant multi-group pattern should work
    def test_impl2(in_str):
        return re.findall(r"(\w+).*(\d+)", in_str)

    check_func(test_impl2, (in_str,), only_seq=True)


def test_pat_findall(memory_leak_check):
    """make sure Pattern.findall returns proper output (list of strings)"""

    def test_impl(pat, in_str):
        return pat.findall(in_str)

    pat = re.compile(r"\w+")
    in_str = "Words, words, words."
    check_func(test_impl, (pat, in_str), only_seq=True)

    # an error should be raised with multiple groups if pattern is not constant
    def test_impl2(pat, in_str):
        return pat.findall(in_str)

    pat = re.compile(r"(\w+).*(\d+)")
    in_str = "ww 132"
    with pytest.raises(BodoError, match="pattern string should be constant"):
        bodo.jit(test_impl2)(pat, in_str)

    def test_impl3(in_str):
        pat = re.compile(r"(\w+).*(\d+)")
        return pat.findall(in_str)

    in_str = "ww 132"
    check_func(test_impl3, (in_str,), only_seq=True)

    def test_impl4(in_str):
        pat = re.compile(r"(\w+).*")
        return pat.findall(in_str)

    in_str = "ww 132"
    check_func(test_impl4, (in_str,), only_seq=True)


def test_re_sub(memory_leak_check):
    """make sure re.sub returns proper output (a string)"""

    def test_impl(pat, repl, in_str):
        return re.sub(pat, repl, in_str)

    pat = r"\w+"
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_pat_sub(memory_leak_check):
    """make sure Pattern.sub returns proper output (a string)"""

    def test_impl(pat, repl, in_str):
        return pat.sub(repl, in_str)

    pat = re.compile(r"ab*")
    repl = "ff"
    in_str = "aabbcc"
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_re_subn(memory_leak_check):
    """make sure re.subn returns proper output (a string and integer)"""

    def test_impl(pat, repl, in_str):
        return re.subn(pat, repl, in_str)

    pat = r"\w+"
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_pat_subn(memory_leak_check):
    """make sure Pattern.subn returns proper output (a string and integer)"""

    def test_impl(pat, repl, in_str):
        return pat.subn(repl, in_str)

    pat = re.compile(r"\w+")
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_re_escape(memory_leak_check):
    """make sure re.escape returns proper output (a string)"""

    def test_impl(pat):
        return re.escape(pat)

    pat = "http://www.python.org"
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_re_purge(memory_leak_check):
    """make sure re.purge call works (can't see internal cache of re to fully test)"""

    def test_impl():
        return re.purge()

    bodo.jit(test_impl)()


def test_pat_flags(memory_leak_check):
    """test Pattern.flags"""

    def test_impl(pat):
        return pat.flags

    pat = re.compile(r"AA", flags=re.IGNORECASE)
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_pat_groups(memory_leak_check):
    """test Pattern.groups"""

    def test_impl(pat):
        return pat.groups

    pat = re.compile(r"(AA) (\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_pat_groupindex(memory_leak_check):
    """test Pattern.groupindex. Python returns mappingproxy object but Bodo returns
    a Numba TypedDict
    """

    def test_impl(pat):
        return pat.groupindex

    pat = re.compile(r"(?P<first_name>\w+) (?P<last_name>\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert dict(py_out) == dict(bodo_out)


def test_pat_pattern(memory_leak_check):
    """test Pattern.pattern"""

    def test_impl(pat):
        return pat.pattern

    pat = re.compile(r"(AA) (\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_match_expand(memory_leak_check):
    """test Match.expand()"""

    def test_impl(m):
        return m.expand(r"\1 WW \2")

    pat = re.compile(r"(\w+) (\w+)")
    m = pat.search("words words etc")
    py_out = test_impl(m)
    bodo_out = bodo.jit(test_impl)(m)
    assert py_out == bodo_out


def test_match_group(memory_leak_check):
    """test Match.group(), the output is a string or tuple of strings"""

    def test_impl_zero(m):
        return m.group()

    def test_impl_one(m, a):
        return m.group(a)

    def test_impl_two(m, a, b):
        return m.group(a, b)

    def test_impl_three(m, a, b, c):
        return m.group(a, b, c)

    pat = re.compile(r"(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl_zero(m) == bodo.jit(test_impl_zero)(m)
    assert test_impl_one(m, "A") == bodo.jit(test_impl_one)(m, "A")
    assert test_impl_two(m, "A", 3) == bodo.jit(test_impl_two)(m, "A", 3)
    assert test_impl_three(m, 2, "A", 3) == bodo.jit(test_impl_three)(m, 2, "A", 3)

    # test Match.group() when an output should be a None
    pat = re.compile(r"(\w+)? (\w+) (\w+)")
    m = pat.search(" words word")

    assert test_impl_one(m, 1) == bodo.jit(test_impl_one)(m, 1)
    assert test_impl_two(m, 1, 2) == bodo.jit(test_impl_two)(m, 1, 2)


def test_match_getitem(memory_leak_check):
    """test Match[g], which is shortcut for Match.group(g)"""

    def test_impl(m, a):
        return m[a]

    pat = re.compile(r"(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl(m, "A") == bodo.jit(test_impl)(m, "A")


def test_match_groups(memory_leak_check):
    """test Match.groups(). Python returns a tuple but we return a list since length
    of tuple is not known in advance.
    """

    def test_impl(m):
        return m.groups()

    pat = re.compile(r"(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert list(test_impl(m)) == bodo.jit(test_impl)(m)

    # test Match.group() when an output should be a None
    pat = re.compile(r"(\w+)? (\w+) (\w+)")
    m = pat.search(" words word")
    assert list(test_impl(m)) == bodo.jit(test_impl)(m)


def test_match_groupdict(memory_leak_check):
    """test Match.groupdict(), which returns a dictionary of named groups"""

    def test_impl(m):
        return m.groupdict()

    pat = re.compile(r"(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl(m) == bodo.jit(test_impl)(m)

    # test Match.groupdict() when an output should be a None
    pat = re.compile(r"(?P<A>\w+)? (?P<B>\w+) (?P<C>\w+)")
    m = pat.search(" words word")
    with pytest.raises(BodoError, match="does not support default=None"):
        bodo.jit(test_impl)(m)


def test_match_start(memory_leak_check):
    """test Match.start()"""

    def test_impl(m, g):
        return m.start(g)

    m = re.search(r"(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_end(memory_leak_check):
    """test Match.end()"""

    def test_impl(m, g):
        return m.end(g)

    m = re.search(r"(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_span(memory_leak_check):
    """test Match.span()"""

    def test_impl(m, g):
        return m.span(g)

    m = re.search(r"(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_pos(memory_leak_check):
    """test Match.pos attribute"""

    def test_impl(m):
        return m.pos

    pat = re.compile(r"(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("  words words etc", 2)

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_endpos(memory_leak_check):
    """test Match.endpos attribute"""

    def test_impl(m):
        return m.endpos

    pat = re.compile(r"(?P<A>\w+) (\w+)")
    m = pat.search("  words words etc bcd bcd", 2, 14)

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_lastindex(memory_leak_check):
    """test Match.lastindex attribute"""

    def test_impl(m):
        return m.lastindex

    pat = re.compile(r"(?P<A>\w+) (\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)

    # no group match, should return None
    pat = re.compile(r"\w+ (\d+)?")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_lastgroup(memory_leak_check):
    """test Match.lastgroup attribute"""

    def test_impl(m):
        return m.lastgroup

    pat = re.compile(r"(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)

    # no group match, should return None
    pat = re.compile(r"\w+ (\d+)?")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_re(memory_leak_check):
    """test Match.re attribute"""

    def test_impl(m):
        return m.re

    pat = re.compile(r"(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_string(memory_leak_check):
    """test Match.string attribute"""

    def test_impl(m):
        return m.string

    pat = re.compile(r"(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_binary_format(memory_leak_check):
    """tests format string on binary strings"""

    def test_impl(string, val):
        return string.format(val)

    check_func(test_impl, ("{0:b}", 121311))
    check_func(
        test_impl,
        (
            "{0:b}",
            -1,
        ),
    )


@pytest.mark.slow
def test_binary_format_literal(memory_leak_check):
    """tests format string on binary strings"""

    def test_impl(val):
        return "{0:b}".format(val)

    check_func(test_impl, (121311,))
    check_func(test_impl, (-1,))


def test_str_max(memory_leak_check):
    """tests max between two string literals"""

    def test_impl():
        return max("hello", "world")

    check_func(test_impl, ())


def test_str_min(memory_leak_check):
    """tests min between two string literals"""

    def test_impl():
        return min("hello", "world")

    check_func(test_impl, ())


@pytest.mark.slow
def test_format_numbered_args(memory_leak_check):
    """tests format string with numbered args"""

    def test_impl():
        return "I like {1} and {0}".format("Java", "Python")

    check_func(test_impl, ())


def test_format_kwargs(memory_leak_check):
    """tests format string with kwargs"""

    def test_impl():
        return "{name} is the {job} of {company}".format(
            name="Ehsan", job="CTO", company="Bodo.ai"
        )

    check_func(test_impl, ())


@pytest.mark.slow
def test_format_args_kwargs(memory_leak_check):
    """tests format string with args and kwargs"""

    def test_impl():
        return "{1} is the {job} of {0}".format("Bodo.ai", "Ehsan", job="CTO")

    check_func(test_impl, ())


@pytest.mark.parametrize(
    "impl",
    [
        lambda x: np.int64(x),
        lambda x: np.uint32(x),
        lambda x: np.float32(x),
        lambda x: np.float64(x),
    ],
)
def test_invalid_runtime_conversion(impl, memory_leak_check):
    str_val = "a"
    error_msg = "invalid string to .* conversion"
    with pytest.raises(RuntimeError, match=error_msg):
        bodo.jit(impl)(str_val)


@pytest.mark.slow
class TestString(unittest.TestCase):
    def test_pass_return(self):
        def test_impl(_str):
            return _str

        bodo_func = bodo.jit(test_impl)
        # pass single string and return
        arg = "test_str"
        self.assertEqual(bodo_func(arg), test_impl(arg))
        # pass string list and return
        arg = ["test_str1", "test_str2"]
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_const(self):
        def test_impl():
            return "test_str"

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_str2str(self):
        str2str_methods = [
            "capitalize",
            "casefold",
            "lower",
            "lstrip",
            "rstrip",
            "strip",
            "swapcase",
            "title",
            "upper",
        ]
        for method in str2str_methods:
            func_text = "def test_impl(_str):\n"
            func_text += "  return _str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            # XXX: \t support pending Numba #4188
            # arg = ' \tbbCD\t '
            arg = " bbCD "
            self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_strip_args(self):
        strip_methods = [
            "lstrip",
            "rstrip",
            "strip",
        ]
        for method in strip_methods:
            loc_vars = {}
            globs = {}
            func_text1 = "def test_impl1(_str):\n"
            func_text1 += "  return _str.{}(' ')\n".format(method)
            exec(func_text1, globs, loc_vars)
            test_impl1 = loc_vars["test_impl1"]

            loc_vars = {}
            func_text2 = "def test_impl2(_str):\n"
            func_text2 += "  return _str.{}('\\n')\n".format(method)
            exec(func_text2, globs, loc_vars)
            test_impl2 = loc_vars["test_impl2"]

            arg1 = " bbCD "
            arg2 = "\n \tbbCD\t \n"
            check_func(test_impl1, (arg1,))
            check_func(test_impl2, (arg2,))

    def test_str2bool(self):
        str2bool_methods = [
            "isalnum",
            "isalpha",
            "isdigit",
            "isspace",
            "islower",
            "isupper",
            "istitle",
            "isnumeric",
            "isdecimal",
        ]
        for method in str2bool_methods:
            func_text = "def test_impl(_str):\n"
            func_text += "  return _str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars["test_impl"]
            bodo_func = bodo.jit(test_impl)
            args = ["11", "aa", "AA", " ", "Hi There"]
            for arg in args:
                self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_equality(self):
        def test_impl(_str):
            return _str == "test_str"

        bodo_func = bodo.jit(test_impl)
        arg = "test_str"
        self.assertEqual(bodo_func(arg), test_impl(arg))

        def test_impl(_str):
            return _str != "test_str"

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_concat(self):
        def test_impl(_str):
            return _str + "test_str" + "À È Ì"

        bodo_func = bodo.jit(test_impl)
        arg = "a_"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_split(self):
        def test_impl(_str):
            return _str.split("/")

        bodo_func = bodo.jit(test_impl)
        arg = "aa/bb/cc"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_replace(self):
        def test_impl(_str):
            return _str.replace("/", ";")

        bodo_func = bodo.jit(test_impl)
        arg = "aa/bb/cc"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_rfind(self):
        def test_impl(_str):
            return _str.rfind("/", 2)

        bodo_func = bodo.jit(test_impl)
        arg = "aa/bb/cc"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_getitem_int(self):
        def test_impl(_str):
            return _str[3]

        bodo_func = bodo.jit(test_impl)
        arg = "aa/bb/cc"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_int_cast(self):
        def test_impl(_str):
            return int(_str)

        bodo_func = bodo.jit(test_impl)
        arg = "12"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_float_cast(self):
        def test_impl(_str):
            return float(_str)

        bodo_func = bodo.jit(test_impl)
        arg = "12.2"
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_str_cast(self):
        def test_impl(a):
            return str(a)

        bodo_func = bodo.jit(test_impl)
        for arg in [np.int32(45), np.uint32(45), 43, np.float32(1.4), 4.5]:
            py_res = test_impl(arg)
            h_res = bodo_func(arg)
            # XXX: use startswith since bodo output can have extra characters
            self.assertTrue(h_res.startswith(py_res))

    # def test_str_findall_count(self):
    #     def bodo_test_impl(_str):
    #         p = re.compile('ab*')
    #         return str_findall_count(p, _str)
    #     def test_impl(_str):
    #         p = re.compile('ab*')
    #         return len(p.findall(_str))
    #     bodo_func = bodo.jit(bodo_test_impl)
    #     arg = 'abaabbcc'
    #     self.assertEqual(bodo_func(arg), len(test_impl(arg)))

    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return str_arr_from_sequence(["ABC", "BB", "CDEF"])

        bodo_func = bodo.jit(test_impl)
        self.assertTrue(np.array_equal(bodo_func(), ["ABC", "BB", "CDEF"]))

    def test_string_array_shape(self):
        def test_impl():
            return str_arr_from_sequence(["ABC", "BB", "CDEF"]).shape

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), (3,))

    def test_string_array_comp(self):
        def test_impl():
            A = str_arr_from_sequence(["ABC", "BB", "CDEF"])
            B = A == "ABC"
            return B.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), 1)

    def test_string_series(self):
        def test_impl(ds):
            rs = ds == "one"
            return ds, rs

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({"A": [1, 2, 3] * 33, "B": ["one", "two", "three"] * 33})
        ds, rs = bodo_func(df.B)
        gc.collect()
        self.assertTrue(isinstance(ds, pd.Series) and isinstance(rs, pd.Series))
        self.assertTrue(
            ds[0] == "one" and ds[2] == "three" and rs[0] == True and rs[2] == False
        )

    def test_string_array_bool_getitem(self):
        def test_impl():
            A = str_arr_from_sequence(["ABC", "BB", "CDEF"])
            B = A == "ABC"
            C = A[B]
            return len(C) == 1 and C[0] == "ABC"

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), True)

    def test_string_NA_box(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            return df.five

        bodo_func = bodo.jit(distributed=False)(test_impl)
        # XXX just checking isna() since Pandas uses None in this case
        # instead of nan for some reason
        np.testing.assert_array_equal(bodo_func().isna(), test_impl().isna())

    # test utf8 decode
    def test_decode_empty1(self):
        def test_impl(S):
            return S[0]

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([""])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_single_ascii_char1(self):
        def test_impl(S):
            return S[0]

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["A"])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_ascii1(self):
        def test_impl(S):
            return S[0]

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["Abc12", "bcd", "345"])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_unicode1(self):
        def test_impl(S):
            return S[0], S[1], S[2]

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_unicode2(self):
        # test strings that start with ascii
        def test_impl(S):
            return S[0], S[1], S[2]

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["abc¡Y tú quién te crees?", "dd2🐍⚡", "22 大处着眼，小处着手。"])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_encode_unicode1(self):
        def test_impl():
            return pd.Series(["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"])

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_unicode_decode_2byte_kind(self):
        """Test decoding element from string array in UTF8 to 2-byte kind Unicode"""

        def test_impl(A):
            return bodo.libs.str_arr_ext.get_utf8_size(A[0])

        # a string array with a 2-byte kind Unicode
        A = np.array(["Õ"], dtype=object)
        # length of element after encoding back to UTF8 should be 2
        assert bodo.jit(test_impl)(A) == 2

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        A = np.array(["AA", "B"])
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_glob(self):
        def test_impl1():
            return glob.glob("*py")

        def test_impl2():
            return glob.glob("**/*py", recursive=True)

        bodo_func = bodo.jit(test_impl1)
        self.assertEqual(bodo_func(), test_impl1())

        bodo_func = bodo.jit(test_impl2)
        self.assertEqual(bodo_func(), test_impl2())


if __name__ == "__main__":
    unittest.main()
