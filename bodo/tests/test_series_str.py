# Copyright (C) 2022 Bodo Inc. All rights reserved.
import itertools

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import _test_equal, check_func


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    "ABCDD,OSAJD",
                    "a1b2d314f,sdf234",
                    "22!@#,$@#$",
                    np.nan,
                    "A,C,V,B,B",
                    "AA",
                    "",
                ]
                * 2,
                [4, 3, 5, 1, 0, -3, 2, -5, 6, 10, -2, 7, -1, -4],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    np.nan,
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                ]
                * 2,
                [4, 3, 5, 1, 0, -3, 2, -5, 6, 10, -2, 7, -1, -4],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    np.nan,
                    "こんにち,は世界",
                    "大处着眼，小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                    "한국,가,고싶다ㅠ",
                ]
                * 2,
                [4, 3, 5, 1, 0, -3, 2, -5, 6, 10, -2, 7, -1, -4],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "😀🐍,⚡😅😂",
                    "🌶🍔,🏈💔💑💕",
                    "𠁆𠁪,𠀓𠄩𠆶",
                    np.nan,
                    "🏈,💔,𠄩,😅",
                    "🠂,🠋🢇🄐,🞧",
                    "🢇🄐,🏈𠆶💑😅",
                ]
                * 2,
                [4, 3, 5, 1, 0, -3, 2, -5, 6, 10, -2, 7, -1, -4],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pd.Series(
            [
                "A",
                " bbCD",
                " mCDm",
                "C,ABB, D",
                "B,B,CC",
                "ABBD",
                "ABCDD,OSAJD",
                "a1b2d314f,sdf234",
                "C,ABB,D",
                "¿abc¡Y tú, quién te cre\t\tes?",
                "오늘도 피츠버그의 날씨는 매\t우, 구림",
                np.nan,
                "🏈,💔,𠄩,😅",
                "大处着眼，小处着手。",
                "🠂,🠋🢇🄐,🞧",
                "россия очень, холодная страна",
                "",
                " ",
            ],
            [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            name="A",
        ),
    ]
)
def test_unicode(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    "AOSJD,OSAJD",
                    "a1b2d314f,sdf234",
                    "22!@#,$@#$",
                    "A,C,V,B,B",
                    "HELLO, WORLD",
                    "aAbB,ABC",
                ],
                [4, 3, 1, 0, 2, 5],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                ],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    "こんにち,は世界",
                    "大处着眼, 小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                    "한국,가,고싶다ㅠ",
                ],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧", "🢇🄐,🏈𠆶💑😅"],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pd.Series(
            [
                "A",
                " bbCD",
                " mCDm",
                "C,ABB, D",
                "B,B,CC",
                "ABBD",
                "ABCDD,OSAJD",
                "a1b2d314f,sdf234",
                "C,ABB,D",
                "¿abc¡Y tú, quién te cre\t\tes?",
                "오늘도 피츠버그의 날씨는 매\t우, 구림",
                "🏈,💔,𠄩,😅",
                "🠂,🠋🢇🄐,🞧",
                "россия очень, холодная страна",
                " ",
                "",
            ],
            [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            name="A",
        ),
    ]
)
def test_unicode_no_nan(request):
    return request.param


@pytest.mark.slow
def test_len(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.len()

    check_func(test_impl, (test_unicode,), check_dtype=False, check_typing_issues=False)


# TODO: Add memory_leak_check when bugs are resolved.
@pytest.mark.slow
def test_split(test_unicode_no_nan):
    def test_impl(S):
        return S.str.split(",")

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    check_func(test_impl, (test_unicode_no_nan,))


# TODO: Add memory_leak_check when bugs are resolved.
@pytest.mark.slow
def test_split_empty(test_unicode_no_nan):
    def test_impl(S):
        return S.str.split("")

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    check_func(test_impl, (test_unicode_no_nan,))


# TODO: Add memory_leak_check when bugs are resolved.
@pytest.mark.slow
def test_split_n():
    def test_impl(S):
        return S.str.split(",", n=1)

    S = pd.Series(
        ["ab,cde,erfe,s,e,qrq,", "no commmas here", "Only 1 comma at the end,"] * 5
    )

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    check_func(test_impl, (S,))

    S = pd.Series(["°, ∞, ‰,", "± × ÷ √", "♩ ♪ ♫ ♬ ♭ ♮ ♯,"] * 5)

    check_func(test_impl, (S,))


# TODO: Add memory_leak_check when bugs are resolved.
@pytest.mark.slow
def test_split_regex():
    def test_impl(S):
        return S.str.split("a|b")

    # Check that this will actually be interpretted as a regex
    S = pd.Series(
        [
            "abababbabababba",
            "a|ba|ba|ba|bb|aa|b",
            "here is only an a",
            "Only a b here",
            "No A or B",
        ]
        * 5
    )

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    check_func(test_impl, (S,))

    S = pd.Series(
        [
            "ƗØƀƗØƀ",
            "ƗØ|ƀƗ|Øƀ|ƗØ|ƀƗ|Øƀ",
            "Ɨ Ɨ Ɨ Ɨ Ɨ Ɨ Ɨ Ɨ",
            "ƀ Ɨ ƀ Ɨ ƀ Ɨ ƀ Ɨ",
            "Ǎ Ě Ǐ Ǎ Ě Ǐ",
        ]
        * 5
    )

    check_func(test_impl, (S,))


def test_series_str_split_explode(memory_leak_check):
    """test split() and explode() combination"""

    def test_impl1(df):
        return df.A.str.split().explode()

    # split view case
    def test_impl2(df):
        return df.A.str.split(",").explode()

    df = pd.DataFrame(
        {"A": pd.array(["A B C", "A", "D E", "A N C E Q  R#R##R#RR F", np.nan] * 5)}
    )
    check_func(test_impl1, (df,))
    df = pd.DataFrame(
        {"A": pd.array(["A,B,C", "A", "D,E", "", "A,N,C,E,Q  R#R##R#RR,F", np.nan] * 5)}
    )
    check_func(test_impl2, (df,))

    df = pd.DataFrame(
        {"A": pd.array(["Ȩ Ç Ḑ", "ẞ", "Ő Ű", "Å Ů ẘ ẙ Q Ð#Ð##Ð#ÐÐ F", np.nan] * 5)}
    )

    check_func(test_impl1, (df,))

    df = pd.DataFrame(
        {"A": pd.array(["Ȩ,Ç,Ḑ", "ẞ", "Ő,Ű", "", "Å,Ů,ẘ,ẙ,Q Ð#Ð##Ð#ÐÐ,F", np.nan] * 5)}
    )

    check_func(test_impl2, (df,))


# TODO: Add memory_leak_check when bugs are resolved.
@pytest.mark.slow
def test_split_no_regex():
    def test_impl1(S):
        # Check that . will not be viewed as a regex in splitview
        return S.str.split(".")

    def test_impl2(S, pat):
        # Check that . will not be viewed as a regex not in splitview
        return S.str.split(pat)

    # check that . match only the . character and not any character
    S = pd.Series(
        ["No dots here", "...............", ",.,.,,,,.,.,.,.,.,.", "Only 1 . here"] * 5
    )

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S, "."))


def test_repeat(test_unicode_no_nan):
    def test_impl(S):
        return S.str.repeat(3)

    check_func(test_impl, (test_unicode_no_nan,))


@pytest.mark.slow
def test_repeat_arr(test_unicode_no_nan):
    def test_impl(S, arr):
        return S.str.repeat(arr)

    arr = np.array(range(len(test_unicode_no_nan)))
    check_func(test_impl, (test_unicode_no_nan, arr))


@pytest.mark.slow
def test_repeat_const_list():
    # Only test on value where the list length works
    def test_impl(S):
        return S.str.repeat([1, 4, 1, 7, 9, 2, 5, 2, 1, 8, 2, 1, 2, 3, 1, 4])

    S = pd.Series(
        [
            "A",
            " bbCD",
            " mCDm",
            "C,ABB, D",
            "B,B,CC",
            "ABBD",
            "ABCDD,OSAJD",
            "a1b2d314f,sdf234",
            "C,ABB,D",
            "¿abc¡Y tú, quién te cre\t\tes?",
            "오늘도 피츠버그의 날씨는 매\t우, 구림",
            "🏈,💔,𠄩,😅",
            "🠂,🠋🢇🄐,🞧",
            "россия очень, холодная страна",
            " ",
            "",
        ],
        [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        name="A",
    )

    check_func(test_impl, (S,), dist_test=False)


def test_get(memory_leak_check):
    def test_impl(S):
        B = S.str.split(",")
        return B.str.get(1)

    # TODO: support and test NA
    # TODO: support distributed
    S = pd.Series(
        [
            "A,B",
            " bb,CD",
            " mCD,m",
            "C,ABB, D",
            "B,B,CC",
            "AB,BD",
            "ABCDD,OSAJD",
            "a1b2d314f,sdf234",
            "C,ABB,D",
            "¿abc¡Y tú, quién te cre\t\tes?",
            "오늘도 피츠버그의 날씨는 매\t우, 구림",
            "🏈,💔,𠄩,😅",
            "🠂,🠋🢇🄐,🞧",
            "россия очень, холодная страна",
        ],
        [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12],
        name="A",
    )
    # check_func(test_impl, (S,))
    pd.testing.assert_series_equal(
        bodo.jit(test_impl)(S), test_impl(S), check_dtype=False
    )


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([[1, 3, None], None, [2, 4], [2], [], [5, -3, 1, 6]]),
        pytest.param(
            pd.Series(
                [
                    [[[1, 2], [3]], [[2, None]]],
                    [[[3], [], [1, None, 4]]],
                    None,
                    [[[4, 5, 6], []], [[1]], [[1, 2]]],
                    [],
                    [[[], [1]], None, [[1, 4]], []],
                ]
                * 2
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: nested string test when old list(str) type is removed
    ],
)
def test_get_array_item(S, memory_leak_check):
    """Tests Series.str.get() support for non-string arrays like array(item)."""

    def test_impl(S):
        return S.str.get(1)

    check_func(test_impl, (S,), check_dtype=False)


@pytest.mark.slow
def test_replace_regex(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.replace("AB*", "EE", regex=True)

    def test_impl2(S):
        return S.str.replace("피츠*", "뉴욕의", regex=True)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_replace_noregex(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.replace("AB", "EE", regex=False)

    def test_impl2(S):
        return S.str.replace("피츠버그의", "뉴욕의", regex=False)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.parametrize("case", [True, False])
def test_match(test_unicode, case, memory_leak_check):
    def test_impl(S):
        return S.str.match("AB*", case=case)

    def test_impl1(S):
        return S.str.match("AB", case=case)

    def test_impl2(S):
        return S.str.match("피츠버*", case=case)

    def test_impl3(S):
        return S.str.match("피츠버", case=case)

    def test_impl4(S):
        return S.str.match("ab*", case=case)

    def test_impl5(S):
        return S.str.match("ab", case=case)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl1, (test_unicode,))
    check_func(test_impl2, (test_unicode,))
    check_func(test_impl3, (test_unicode,))
    check_func(test_impl4, (test_unicode,))
    check_func(test_impl5, (test_unicode,))


@pytest.mark.slow
@pytest.mark.parametrize("case", [True, False])
def test_contains_regex(test_unicode, case, memory_leak_check):
    def test_impl(S):
        return S.str.contains("AB*", regex=True, case=case)

    def test_impl2(S):
        return S.str.contains("피츠버*", regex=True, case=case)

    def test_impl3(S):
        return S.str.contains("ab*", regex=True, case=case)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))
    check_func(test_impl3, (test_unicode,))


@pytest.mark.parametrize("sep", [None, "__"])
def test_cat(test_unicode, sep, memory_leak_check):
    """test Series.str.cat() with dataframe input"""

    def impl(S, df, sep):
        return S.str.cat(df, sep)

    df = pd.DataFrame({"A": test_unicode, "B": ["ABC"] * len(test_unicode)})
    check_func(impl, (test_unicode, df, sep))


@pytest.mark.slow
@pytest.mark.parametrize("case", [True, False])
def test_re_syntax(case, memory_leak_check):

    # Test special characters and quantifiers
    def test_impl(S):
        return S.str.contains(r"[a-z]+", regex=True, case=case)

    # Test special characters and quantifiers
    def test_impl2(S):
        return S.str.contains(r"^a...s$", regex=True, case=case)

    # Test groups
    def test_impl3(S):
        return S.str.contains(r"(a|b|c)xz", regex=True, case=case)

    # Test (?P<name>...)
    def test_impl4(S):
        return S.str.contains(r"(?P<w1>\w+),(?P<w2>\w+)", regex=True, case=case)

    # ()\number
    def test_impl5(S):
        return S.str.contains(r"(\w+),\1", regex=True, case=case)

    # (?=...): a positive lookahead assertion.
    def test_impl6(S):
        return S.str.contains(r"AB(?=[a-z])", regex=True, case=case)

    # (?!...):  a negative lookahead assertion.
    def test_impl7(S):
        return S.str.contains(r"xz(?![a-z])", regex=True, case=case)

    # (?<=...):  a positive lookbehind assertion.
    def test_impl8(S):
        return S.str.contains(r"(?<=ax)z", regex=True, case=case)

    # (?<!...) a negative lookbehind assertion.
    def test_impl9(S):
        return S.str.contains(r"(?<!foo)bar", regex=True, case=case)

    # Special sequences
    # \A
    def test_impl10(S):
        return S.str.contains(r"\Afoo", regex=True, case=case)

    # {m,n} repititions
    def test_impl11(S):
        return S.str.contains(r"a{3,5}?", regex=True, case=case)

    # (?:<regex>): non-capturing group
    def test_impl12(S):
        return S.str.contains(r"(\w+),(?:\w+),(\w+)", regex=True, case=case)

    # (?P=name) Matches the contents of a previously captured named group
    def test_impl13(S):
        return S.str.contains(r"(?P<word>\w+),(?P=word)", regex=True, case=case)

    # ---------Following tests falls back to Python objmode------------

    # Test unsupported patterns (?aiLmux)
    # ?a = ASCII-only matching
    def test_impl_a(S):
        return S.str.contains(r"(?a)^foo", regex=True, case=case)

    # ?i = ignore case
    def test_impl_i(S):
        return S.str.contains(r"(?i)^bar", regex=True, case=case)

    # ?m = multi-line
    def test_impl_m(S):
        return S.str.contains(r"(?m)^bar", regex=True, case=case)

    # ?s = dot matches all (including newline)
    def test_impl_s(S):
        return S.str.contains(r"(?s)foo.bar", regex=True, case=case)

    # ?u = dot matches all (including newline)
    # This exists for backward compatibility but is redundant
    # See https://docs.python.org/3/library/re.html#re.ASCII
    def test_impl_u(S):
        return S.str.contains(r"(?u:\w+)", regex=True, case=case)

    # ?x = verbose. Whitespace within the pattern is ignored
    def test_impl_x(S):
        return S.str.contains(
            r"""(?x)\d  +
                                .""",
            regex=True,
            case=case,
        )

    def test_impl_comment(S):
        return S.str.contains(r"bar(?#This is a comment).*", regex=True, case=case)

    # Test escape with unsupported patterns
    def test_impl14(S):
        return S.str.contains(r"\(?i", regex=True, case=case)

    # Test flags
    import re

    flag = re.M.value

    def test_impl15(S):
        return S.str.contains(r"foo*", regex=True, case=case, flags=flag)

    # ---------End of tests for Python objmode--------------------------

    # Test C++ pattern regex pattern as literal by Python
    def test_impl16(S):
        return S.str.contains(r"[:alnum:]", regex=True, case=case, flags=flag)

    def test_impl17(S):
        return S.str.contains(r"[[:digit:]]", regex=True, case=case, flags=flag)

    S = pd.Series(
        [
            "ABCDD,OSAJD",
            "a1b2d314f,sdf234",
            "22!@#,$@#$",
            np.nan,
            "A,C,V,B,B",
            "ABcd",
            "",
            "axz1s",
            "foo,foo",
            "foo\nbar",
            "foobar",
            "[:alnum:]",
            "[[:digit:]]",
            "(?i",
            "aaa111",
            "BAR@#4",
            "١٢٣",
        ]
        * 2,
    )
    check_func(test_impl, (S,))
    check_func(test_impl2, (S,))
    check_func(test_impl3, (S,))
    check_func(test_impl4, (S,))
    check_func(test_impl5, (S,))
    check_func(test_impl6, (S,))
    check_func(test_impl7, (S,))
    check_func(test_impl8, (S,))
    check_func(test_impl9, (S,))
    check_func(test_impl10, (S,))
    check_func(test_impl11, (S,))
    check_func(test_impl12, (S,))
    check_func(test_impl13, (S,))

    check_func(test_impl_a, (S,))
    check_func(test_impl_i, (S,))
    check_func(test_impl_m, (S,))
    check_func(test_impl_u, (S,))
    check_func(test_impl_x, (S,))
    check_func(test_impl_s, (S,))
    check_func(test_impl_comment, (S,))

    check_func(test_impl14, (S,))
    check_func(test_impl15, (S,))
    check_func(test_impl16, (S,))
    check_func(test_impl17, (S,))


@pytest.mark.parametrize("case", [True, False])
def test_contains_noregex(test_unicode, case, memory_leak_check):
    def test_impl(S):
        return S.str.contains("AB", regex=False, case=case)

    def test_impl2(S):
        return S.str.contains("피츠버그", regex=False, case=case)

    def test_impl3(S):
        return S.str.contains("ab", regex=False, case=case)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))
    check_func(test_impl3, (test_unicode,))


def test_extract(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd])(?P<C>\d+)")

    def test_impl2(S):
        return S.str.extract(r"(?P<BBB>[아])(?P<C>\d+)")

    check_func(test_impl, (test_unicode,), check_typing_issues=False)
    check_func(test_impl2, (test_unicode,), check_typing_issues=False)


def test_extract_const(memory_leak_check):
    """make sure 'pat' argument can be forced to a constant value"""

    def test_impl(S, pat):
        return S.str.extract(pat)

    S = pd.Series(["a1", "b2", "c3"])
    p = r"(\w)(\d)"
    check_func(test_impl, (S, p), only_seq=True)


def test_extract_noexpand(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd]+)\d+", expand=False)

    def test_impl2(S):
        return S.str.extract(r"(?P<BBB>[아])(?P<C>\d+)", expand=False)

    # when regex group has no name, Series name should be used
    def test_impl_noname(S):
        return S.str.extract(r"([abd]+)\d+", expand=False)

    check_func(test_impl, (test_unicode,), check_typing_issues=False)
    check_func(test_impl_noname, (test_unicode,), check_typing_issues=False)
    check_func(test_impl2, (test_unicode,), check_typing_issues=False)


# TODO: Add memory_leak_check when problem are resolved.
def test_extractall():
    """Test Series.str.extractall() with various input cases"""
    # ascii input with non-string index, single named group
    def test_impl1(S):
        return S.str.extractall(r"(?P<BBB>[abd]+)\d+")

    S = pd.Series(
        ["a1b1", "b1", np.nan, "a2", "c2", "ddd", "dd4d1", "d22c2"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    check_func(test_impl1, (S,))

    # unicode input with string index, multiple unnamed group
    def test_impl2(S):
        return S.str.extractall(r"([чен]+)\d+([ст]+)\d+")

    S2 = pd.Series(
        ["чьь1т33", "ьнн2с222", "странаст2", np.nan, "ьнне33ст3"] * 2,
        ["е3", "не3", "н2с2", "AA", "C"] * 2,
    )
    check_func(test_impl2, (S2,))


@pytest.mark.slow
def test_count_noflag(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.count("A")

    def test_impl2(S):
        return S.str.count("피츠")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


@pytest.mark.slow
def test_count_flag(test_unicode, memory_leak_check):
    import re

    # TODO: the flag does not work inside numba
    flag = re.IGNORECASE.value

    def test_impl(S):
        return S.str.count("A", flag)

    def test_impl2(S):
        return S.str.count("피츠", flag)

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


@pytest.mark.smoke
def test_find(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.find("AB")

    def test_impl2(S):
        return S.str.find("🍔")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


@pytest.mark.slow
def test_find_start_end(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.find("AB", start=3, end=10)

    def test_impl2(S):
        return S.str.find("AB", start=1, end=5)

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


@pytest.mark.slow
def test_rfind(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.rfind("AB")

    def test_impl2(S):
        return S.str.rfind("дн")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


@pytest.mark.parametrize(
    "S, sub, start, end",
    [
        (
            pd.Series(
                ["alpha", "beta", "alphabet", "patatasbravas", np.nan, "houseofcards"]
            ),
            "a",
            0,
            10,
        ),
        (
            pd.Series(["alpha", "beta", "alphabet", "patatasbravas", "emeralds"]),
            "a",
            2,
            6,
        ),
        (
            pd.Series(
                ["bagel", np.nan, "gelatin", "gelato", "angelfish", "evangelist"]
            ),
            "gel",
            0,
            10,
        ),
    ],
)
@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_rindex(S, sub, start, end, method, memory_leak_check):
    func_text = (
        "def test_impl1(S, sub):\n"
        f"    return S.str.{method}(sub)\n"
        "def test_impl2(S, sub, start):\n"
        f"    return S.str.{method}(sub, start=start)\n"
        "def test_impl3(S, sub, end):\n"
        f"    return S.str.{method}(sub, end=end)\n"
        "def test_impl4(S, sub, start, end):\n"
        f"    return S.str.{method}(sub, start, end)\n"
    )
    local_vars = {}
    exec(func_text, {}, local_vars)
    test_impl1 = local_vars["test_impl1"]
    test_impl2 = local_vars["test_impl2"]
    test_impl3 = local_vars["test_impl3"]
    test_impl4 = local_vars["test_impl4"]
    check_func(test_impl1, (S, sub), check_dtype=False)
    check_func(test_impl2, (S, sub, start), check_dtype=False)
    check_func(test_impl3, (S, sub, end), check_dtype=False)
    check_func(test_impl4, (S, sub, start, end), check_dtype=False)


@pytest.mark.slow
def test_pad_fill_fast(test_unicode, memory_leak_check):
    # this function increases coverage for not slow test suite
    def test_impl1(S):
        return S.str.center(1, "필")

    def test_impl2(S):
        return S.str.rjust(1, "d")

    def test_impl3(S):
        return S.str.ljust(1, "a")

    def test_impl4(S):
        return S.str.pad(1, "left", "🍔")

    def test_impl5(S):
        return S.str.zfill(1)

    check_func(test_impl1, (test_unicode,))
    check_func(test_impl2, (test_unicode,))
    check_func(test_impl3, (test_unicode,))
    check_func(test_impl4, (test_unicode,))
    check_func(test_impl5, (test_unicode,))


@pytest.mark.slow
def test_center(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.center(5, "*")

    def test_impl2(S):
        return S.str.center(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_ljust(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.ljust(5, "*")

    def test_impl2(S):
        return S.str.ljust(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_rjust(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.rjust(5, "*")

    def test_impl2(S):
        return S.str.rjust(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_pad(test_unicode, memory_leak_check):
    def test_impl_default(S):
        return S.str.pad(5)

    def test_impl_left(S):
        return S.str.pad(5, "left", "*")

    def test_impl_right(S):
        return S.str.pad(5, "right", "*")

    def test_impl_both(S):
        return S.str.pad(5, "both", "*")

    def test_impl_both2(S):
        return S.str.pad(5, "both", "🍔")

    check_func(test_impl_default, (test_unicode,))
    check_func(test_impl_left, (test_unicode,))
    check_func(test_impl_right, (test_unicode,))
    check_func(test_impl_both, (test_unicode,))
    check_func(test_impl_both2, (test_unicode,))


@pytest.mark.slow
def test_zfill(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.zfill(10)

    check_func(test_impl, (test_unicode,))


def test_slice(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.slice(step=2)

    check_func(test_impl, (test_unicode,))


def test_startswith(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.startswith("AB")

    def test_impl2(S):
        return S.str.startswith("테스팅")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_endswith(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.endswith("AB")

    def test_impl2(S):
        return S.str.endswith("테스팅")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_isupper(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.isupper()

    check_func(test_impl, (test_unicode,))


@pytest.mark.parametrize("ind", [slice(2), 2])
def test_getitem(ind, test_unicode, memory_leak_check):
    def test_impl(S, ind):
        return S.str[ind]

    check_func(test_impl, (test_unicode, ind))


def test_slice_replace(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.str.slice_replace()

    check_func(test_impl, (test_unicode,))


@pytest.mark.slow
def test_slice_replace_repl(test_unicode, memory_leak_check):
    def test_impl(S, repl):
        return S.str.slice_replace(repl=repl)

    check_func(test_impl, (test_unicode, "bodo.ai"))


@pytest.mark.slow
def test_slice_replace_start(test_unicode, memory_leak_check):
    def test_impl(S, start):
        return S.str.slice_replace(start=start)

    check_func(test_impl, (test_unicode, 5))


@pytest.mark.slow
def test_slice_replace_stop(test_unicode, memory_leak_check):
    def test_impl(S, stop):
        return S.str.slice_replace(stop=stop)

    check_func(test_impl, (test_unicode, 3))


def test_slice_replace_all_args(test_unicode, memory_leak_check):
    def test_impl(S, start, stop, repl):
        return S.str.slice_replace(start, stop, repl)

    check_func(test_impl, (test_unicode, 5, 8, "피츠버그"))


def test_add_series(test_unicode, memory_leak_check):
    def test_impl(S1, S2):
        return S1.add(S2, fill_value="🍔")

    S2 = test_unicode.map(lambda x: np.nan if pd.isna(x) else x[::-1])
    # dict arr unboxing sets nulls to None to avoid PyArrow issues but None causes
    # issues with Series.add. Setting back to np.nan here:
    test_unicode = test_unicode.map(lambda x: np.nan if pd.isna(x) else x)
    check_func(test_impl, (test_unicode, S2))


def test_add_scalar(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.add("hello")

    check_func(test_impl, (test_unicode,))


def test_mul_scalar(test_unicode, memory_leak_check):
    def test_impl(S):
        return S.mul(4)

    check_func(test_impl, (test_unicode,))


##############  list of string array tests  #################


@pytest.fixture(
    params=[
        pytest.param(
            np.array([["a", "bc"], ["a"], ["aaa", "b", "cc"]] * 2, dtype=object),
            marks=pytest.mark.slow,
        ),
        # empty strings, empty lists, NA
        pytest.param(
            np.array(
                [["a", "bc"], ["a"], [], ["aaa", "", "cc"], [""], np.nan] * 2,
                dtype=object,
            ),
            marks=pytest.mark.slow,
        ),
        # large array
        np.array(
            [
                ["a", "bc"],
                ["a"],
                [],
                ["aaa", "", "cc"],
                [""],
                np.nan,
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                ],
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    "こんにち,は世界",
                    "大处着眼，小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                ],
                ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧"],
            ]
            * 1000,
            dtype=object,
        ),
    ]
)
def list_str_arr_value(request):
    return request.param


@pytest.mark.slow
def test_list_str_arr_unbox(list_str_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (list_str_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (list_str_arr_value,))


@pytest.mark.smoke
def test_getitem_int(list_str_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    bodo_func = bodo.jit(test_impl)
    i = 2
    np.testing.assert_array_equal(
        bodo_func(list_str_arr_value, i), test_impl(list_str_arr_value, i)
    )


def test_getitem_bool(list_str_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    np.random.seed(0)
    ind = np.random.ranf(len(list_str_arr_value)) < 0.2
    # TODO: parallel test
    _test_equal(bodo_func(list_str_arr_value, ind), test_impl(list_str_arr_value, ind))


def test_getitem_slice(list_str_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = slice(1, 4)
    # TODO: parallel test
    _test_equal(bodo_func(list_str_arr_value, ind), test_impl(list_str_arr_value, ind))


@pytest.mark.slow
def test_copy(list_str_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    _test_equal(bodo.jit(test_impl)(list_str_arr_value), list_str_arr_value)


def test_flatten1(test_unicode_no_nan, memory_leak_check):
    """tests flattening array of string lists after split call when split view
    optimization is applied
    """

    def impl(S):
        A = S.str.split(",")
        return pd.Series(list(itertools.chain(*A)))

    check_func(impl, (test_unicode_no_nan,))


def test_flatten2(test_unicode_no_nan, memory_leak_check):
    """tests flattening array of string lists after split call when split view
    optimization is not applied
    """

    def impl(S):
        A = S.str.split()
        return pd.Series(list(itertools.chain(*A)))

    check_func(impl, (test_unicode_no_nan,))


@pytest.mark.slow
def test_flatten3(test_unicode_no_nan, memory_leak_check):
    """tests flattening array without the "list" call"""

    def impl(S):
        A = S.str.split(",")
        return pd.Series(itertools.chain(*A))

    check_func(impl, (test_unicode_no_nan,))


def test_flatten4(test_unicode_no_nan, memory_leak_check):
    """tests flattening array with "from_iterable" """

    def impl(S):
        A = S.str.split(",")
        res = pd.Series(itertools.chain.from_iterable(A))
        return res

    check_func(impl, (test_unicode_no_nan,))


def test_join(memory_leak_check):
    """test the functionality of bodo's join with NaN"""

    def test_impl(S):
        return S.str.join("-")

    S = pd.Series(
        [
            ["ABCDD,OSAJD", "a1b2d314f,sdf234", "22!@#,$@#$", "A,C,V,B,B", ""],
            [
                "¿abc¡Y tú, quién te crees?",
                "ÕÕÕú¡úú,úũ¿ééé",
                "россия очень, холодная страна",
                "مرحبا, العالم ، هذا هو بودو",
                "Γειά σου ,Κόσμε",
            ],
            [
                "아1, 오늘 저녁은 뭐먹지",
                "나,는 유,니,코,드 테스팅 중",
                "こんにち,は世界",
                "大处着眼，小处着手。",
                "오늘도 피츠버그의 날씨는 매우, 구림",
            ],
            np.nan,
            ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧"],
        ]
    )
    check_func(test_impl, (S,))


def test_split_non_ascii(memory_leak_check):
    """
    Tests that Series.str.split with a non-ascii
    single character allows subsequent operations.
    """

    def test_impl(S):
        return S.str.split("è").apply(lambda x: len(x))

    S = pd.Series(["afeèefwfewèqr3", "fefè3", "33r3"] * 10)
    check_func(test_impl, (S,))


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_setitem_unichar_arr(memory_leak_check):
    """test Series setitem when the string array comes from Numpy
    UnicodeSeq Arrays"""

    def test_impl(S, idx, val):
        S[idx] = val
        return S

    S = pd.Series(
        [
            "A,B",
            " bb,CD",
            " mCD,m",
            "C,ABB, D",
            "B,B,CC",
            "AB,BD",
            "ABCDD,OSAJD",
            "a1b2d314f,sdf234",
            "C,ABB,D",
            "¿abc¡Y tú, quién te cre\t\tes?",
            "오늘도 피츠버그의 날씨는 매\t우, 구림",
            "🏈,💔,𠄩,😅",
            "🠂,🠋🢇🄐,🞧",
            "россия очень, холодная страна",
        ]
    )
    arr = np.array(["AA", "BB"])
    bool_idx = [True, True] + [False] * (len(S) - 2)
    for idx in (bool_idx, np.array(bool_idx)):
        check_func(test_impl, (S, idx, arr), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_join_string(test_unicode, memory_leak_check):
    """test the functionality of bodo's join with just a string"""

    def test_impl(test_unicode):
        return test_unicode.str.join("-")

    def test_impl2(test_unicode):
        return test_unicode.str.join("*****************")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_join_splitview(test_unicode_no_nan, memory_leak_check):
    """test the functionality of bodo's join with split view type as an input"""

    def test_impl(S):
        B = S.str.split(",")
        return B.str.join("-")

    check_func(test_impl, (test_unicode_no_nan,))


@pytest.mark.slow
def test_join_splitview_nan_entry(memory_leak_check):
    """test the functionality of bodo's join with split view type as an input"""

    def test_impl(S):
        B = S.str.split(",")
        return B.str.join("-")

    S = pd.Series(["ABCDD,OSAJD", "a1b2d314f,sdf234", np.nan], [4, 3, 1], name="A")
    check_func(test_impl, (S,), check_typing_issues=False)
