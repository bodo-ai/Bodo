# Copyright (C) 2019 Bodo Inc. All rights reserved.
# -*- coding: utf-8 -*-

import unittest
import os
import numba
import operator
import bodo
import numpy as np
import pandas as pd
import glob
import gc
import re
import pyarrow.parquet as pq
from bodo.libs.str_arr_ext import StringArray
from bodo.libs.str_ext import unicode_to_std_str, std_str_to_unicode, str_findall_count
from bodo.tests.utils import check_func
import pytest


@pytest.mark.parametrize(
    "op", (operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt)
)
def test_cmp_binary_op(op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += "  return A {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1 = np.array(["A", np.nan, "CC", "DD", np.nan, "ABC"], dtype="O")
    A2 = np.array(["A", np.nan, "CCD", "AADD", "DAA", "ABCE"], dtype="O")

    # # >, >=, <, <= not supported in numpy between np.nan and str
    if op_str not in ("==", "!=="):
        A1[4] = "AF"

    check_func(test_impl, (A1, A2))

    # >, >=, <, <= not supported in numpy between array and str
    if op_str in ("==", "!=="):
        check_func(test_impl, (A1, "DD"))
        check_func(test_impl, ("CCD", A2))


@pytest.mark.parametrize(
    "ind",
    [
        np.array([True, True, False, False, False, True]),
        np.array([0, 1, 3, 4]),
        slice(1, 5),
    ],
)
def test_string_array_getitem_na(ind):
    def impl(S, index):
        return S.iloc[index]

    bodo_func = bodo.jit(impl)
    S = pd.Series(["A", np.nan, "CC", "DD", np.nan, "ABC"])
    # ind = slice(0, 3)
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind))
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind))


##########################  Test re support  ##########################


@pytest.fixture(
    params= ["AB", "A_B", "A_B_C"]
)
def test_in_str(request):
    return request.param


def test_re_search(test_in_str):
    """make sure re.search returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return re.search(pat, in_str)

    pat = "_"
    py_out = test_impl(pat, test_in_str)
    bodo_out = bodo.jit(test_impl)(pat, test_in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


def test_re_match_cast_bool(test_in_str):
    """make sure re.search() output can behave like None in conditionals
    """

    def test_impl(pat, in_str):
        m = re.search(pat, in_str)
        if m:
            return 1
        return 0

    pat = "_"
    assert test_impl(pat, test_in_str) == bodo.jit(test_impl)(pat, test_in_str)


def test_re_match_check_none(test_in_str):
    """make sure re.Match object can be checked for None
    """

    def test_impl(pat, in_str):
        m = re.search(pat, in_str)
        if m is None:
            return 1
        return 0

    pat = "_"
    assert test_impl(pat, test_in_str) == bodo.jit(test_impl)(pat, test_in_str)


def test_re_pat_search(test_in_str):
    """make sure Pattern.search returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return pat.search(in_str)

    pat = re.compile("_")
    py_out = test_impl(pat, test_in_str)
    bodo_out = bodo.jit(test_impl)(pat, test_in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "A_B", "AB_C"])
def test_re_match(in_str):
    """make sure re.match returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return re.match(pat, in_str)

    pat = "AB"
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "AB_", "A_B_C"])
def test_re_pat_match(in_str):
    """make sure Pattern.match returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return pat.match(in_str)

    pat = re.compile("AB")
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "A_B", "AB_C"])
def test_re_fullmatch(in_str):
    """make sure re.fullmatch returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return re.fullmatch(pat, in_str)

    pat = "AB"
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


@pytest.mark.parametrize("in_str", ["AB", "AB_", "A_B_C"])
def test_re_pat_fullmatch(in_str):
    """make sure Pattern.fullmatch returns None or a proper re.Match
    """

    def test_impl(pat, in_str):
        return pat.fullmatch(in_str)

    pat = re.compile("AB")
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    # output is None or re.Match
    # just testing span of re.Match should be enough
    assert (py_out is None and bodo_out is None) or py_out.span() == bodo_out.span()


def test_re_split():
    """make sure re.split returns proper output (list of strings)
    """

    def test_impl(pat, in_str):
        return re.split(pat, in_str)

    pat = r"\W+"
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


def test_pat_split():
    """make sure Pattern.split returns proper output (list of strings)
    """

    def test_impl(pat, in_str):
        return re.split(pat, in_str)

    pat = re.compile(r"\W+")
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


def test_re_findall():
    """make sure re.findall returns proper output (list of strings)
    """

    def test_impl(pat, in_str):
        return re.findall(pat, in_str)

    pat = r"\w+"
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


def test_pat_findall():
    """make sure Pattern.findall returns proper output (list of strings)
    """

    def test_impl(pat, in_str):
        return pat.findall(in_str)

    pat = re.compile(r"\w+")
    in_str = "Words, words, words."
    py_out = test_impl(pat, in_str)
    bodo_out = bodo.jit(test_impl)(pat, in_str)
    assert py_out == bodo_out


def test_re_sub():
    """make sure re.sub returns proper output (a string)
    """

    def test_impl(pat, repl, in_str):
        return re.sub(pat, repl, in_str)

    pat = r"\w+"
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_pat_sub():
    """make sure Pattern.sub returns proper output (a string)
    """

    def test_impl(pat, repl, in_str):
        return pat.sub(repl, in_str)

    pat = re.compile(r"ab*")
    repl = "ff"
    in_str = "aabbcc"
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_re_subn():
    """make sure re.subn returns proper output (a string and integer)
    """

    def test_impl(pat, repl, in_str):
        return re.subn(pat, repl, in_str)

    pat = r"\w+"
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_pat_subn():
    """make sure Pattern.subn returns proper output (a string and integer)
    """

    def test_impl(pat, repl, in_str):
        return pat.subn(repl, in_str)

    pat = re.compile(r"\w+")
    repl = "PP"
    in_str = "Words, words, words."
    py_out = test_impl(pat, repl, in_str)
    bodo_out = bodo.jit(test_impl)(pat, repl, in_str)
    assert py_out == bodo_out


def test_re_escape():
    """make sure re.escape returns proper output (a string)
    """

    def test_impl(pat):
        return re.escape(pat)

    pat = "http://www.python.org"
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_re_purge():
    """make sure re.purge call works (can't see internal cache of re to fully test)
    """

    def test_impl():
        return re.purge()

    bodo.jit(test_impl)()


def test_pat_flags():
    """test Pattern.flags
    """

    def test_impl(pat):
        return pat.flags

    pat = re.compile("AA", flags=re.IGNORECASE)
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_pat_groups():
    """test Pattern.groups
    """

    def test_impl(pat):
        return pat.groups

    pat = re.compile("(AA) (\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_pat_groupindex():
    """test Pattern.groupindex. Python returns mappingproxy object but Bodo returns
    a Numba TypedDict
    """

    def test_impl(pat):
        return pat.groupindex

    pat = re.compile(r"(?P<first_name>\w+) (?P<last_name>\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert dict(py_out) == dict(bodo_out)


def test_pat_pattern():
    """test Pattern.pattern
    """

    def test_impl(pat):
        return pat.pattern

    pat = re.compile("(AA) (\w+)")
    py_out = test_impl(pat)
    bodo_out = bodo.jit(test_impl)(pat)
    assert py_out == bodo_out


def test_match_expand():
    """test Match.expand()
    """

    def test_impl(m):
        return m.expand(r"\1 WW \2")

    pat = re.compile("(\w+) (\w+)")
    m = pat.search("words words etc")
    py_out = test_impl(m)
    bodo_out = bodo.jit(test_impl)(m)
    assert py_out == bodo_out


def test_match_group():
    """test Match.group(), the output is a string or tuple of strings
    """

    def test_impl_zero(m):
        return m.group()

    def test_impl_one(m, a):
        return m.group(a)

    def test_impl_two(m, a, b):
        return m.group(a, b)

    def test_impl_three(m, a, b, c):
        return m.group(a, b, c)

    pat = re.compile("(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl_zero(m) == bodo.jit(test_impl_zero)(m)
    assert test_impl_one(m, "A") == bodo.jit(test_impl_one)(m, "A")
    assert test_impl_two(m, "A", 3) == bodo.jit(test_impl_two)(m, "A", 3)
    assert test_impl_three(m, 2, "A", 3) == bodo.jit(test_impl_three)(m, 2, "A", 3)


def test_match_getitem():
    """test Match[g], which is shortcut for Match.group(g)
    """

    def test_impl(m, a):
        return m[a]

    pat = re.compile("(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl(m, "A") == bodo.jit(test_impl)(m, "A")


def test_match_groups():
    """test Match.groups(). Python returns a tuple but we return a list since length
    of tuple is not known in advance.
    """

    def test_impl(m):
        return m.groups()

    pat = re.compile("(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert list(test_impl(m)) == bodo.jit(test_impl)(m)


def test_match_groupdict():
    """test Match.groupdict(), which returns a dictionary of named groups
    """

    def test_impl(m):
        return m.groupdict()

    pat = re.compile("(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("words words etc")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_start():
    """test Match.start()
    """

    def test_impl(m, g):
        return m.start(g)

    m = re.search("(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_end():
    """test Match.end()
    """

    def test_impl(m, g):
        return m.end(g)

    m = re.search("(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_span():
    """test Match.span()
    """

    def test_impl(m, g):
        return m.span(g)

    m = re.search("(?P<A>\w+) (\w+) (\w+)", "words words etc")
    g = 2

    assert test_impl(m, g) == bodo.jit(test_impl)(m, g)


def test_match_pos():
    """test Match.pos attribute
    """

    def test_impl(m):
        return m.pos

    pat = re.compile("(?P<A>\w+) (\w+) (\w+)")
    m = pat.search("  words words etc", 2)

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_endpos():
    """test Match.endpos attribute
    """

    def test_impl(m):
        return m.endpos

    pat = re.compile("(?P<A>\w+) (\w+)")
    m = pat.search("  words words etc bcd bcd", 2, 14)

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_lastindex():
    """test Match.lastindex attribute
    """

    def test_impl(m):
        return m.lastindex

    pat = re.compile("(?P<A>\w+) (\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_lastgroup():
    """test Match.lastgroup attribute
    """

    def test_impl(m):
        return m.lastgroup

    pat = re.compile("(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_re():
    """test Match.re attribute
    """

    def test_impl(m):
        return m.re

    pat = re.compile("(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


def test_match_string():
    """test Match.string attribute
    """

    def test_impl(m):
        return m.string

    pat = re.compile("(?P<A>\w+) (?P<BB>\w+)")
    m = pat.search("  words words etc bcd bcd")

    assert test_impl(m) == bodo.jit(test_impl)(m)


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
            return _str + "test_str"

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
        for arg in [np.int32(45), 43, np.float32(1.4), 4.5]:
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
            return StringArray(["ABC", "BB", "CDEF"])

        bodo_func = bodo.jit(test_impl)
        self.assertTrue(np.array_equal(bodo_func(), ["ABC", "BB", "CDEF"]))

    def test_string_array_shape(self):
        def test_impl():
            return StringArray(["ABC", "BB", "CDEF"]).shape

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), (3,))

    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(["ABC", "BB", "CDEF"])
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
            A = StringArray(["ABC", "BB", "CDEF"])
            B = A == "ABC"
            C = A[B]
            return len(C) == 1 and C[0] == "ABC"

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), True)

    def test_string_NA_box(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            return df.five

        bodo_func = bodo.jit(test_impl)
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
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        A = np.array(["AA", "B"])
        self.assertEqual(bodo_func(A), test_impl(A))

    @unittest.skip("TODO: support glob")    
    def test_glob(self):
        def test_impl():
            glob.glob("*py")

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())


if __name__ == "__main__":
    unittest.main()
