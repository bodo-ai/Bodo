# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL regexp functions
"""


import re

import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.fixture
def regexp_strings():
    return pd.Series(
        [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "theta",
            "Alpha Beta",
            "Alpha Beta Gamma",
            "Alpha Beta Gamma Delta",
            None,
            "Alphabet soup is so very very delicious\njust like an apple or anchovies or a slice of pie",
            "I bought 14 lemons, 110 blueberries, 1\n0 oranges, 015 apples, 2010 limes and 1024 grapefruits",
            "I am happy,\nI am not sad.\nI am feeling wonderful!",
            None,
            "15-112: Python\n15-122: C\n15-150: SML\n15-213: C\n15-210: SML\n15-251: N/A\n15-312: SML",
            "<alphabet> <soup> <is> <very> <delicious>",
            "*[A]* [*E*] [*I*] *O* *UY*",
            "1121123211234321123454321234567654321123456787654321",
            None,
            "~\n\x14\x1e(",
            "Abcd0123",
            "alphabet_SOUP_42",
            "AlPhAbEt",
            "Al\nPh\nIAbEt",
            "\x04\x0b\x0c\x06\r\x1c\x1f\x19\x7f",
            "15309",
            "!#%Zdx}~",
            "fun",
            "WHEE",
            "\n\t    \nz \v   \v\fy x\r",
            "<...>!!!" " \t\n",
            ' "$(2<}',
            "A1F096",
            "alphabet soup is delicious",
            "#YAY!",
            "ABC,\nIt's easy as 123.\nAs simple as Do re mi.",
            None,
            "¿abc¡Y tú, quién te crees?",
            "ÕÕÕú¡úúúũ¿ééé",
            "россия очень, холодная страна",
            "مرحبا, العالم ، هذا هو بودو",
            "Γειά σου Κόσμε",
            "Español es agradable escuchar",
            "아1, 오늘 저녁은 뭐먹지",
            "나,는 유,니,코,드 테스팅 중",
            "こんにち,は世界",
            "😀🐍⚡😅😂",
            "🌶🍔🏈💔💑💕",
            "🏈💔𠄩😅",
            "🢇🄐,🏈𠆶💑😅",
            None,
            "123 4 56789 01 2345 678 9012",
            None,
            "It was the best of times, it was the worst of times.",
            "In    the   string   the   extra   spaces\rare   redundant.",
            "A thespian theater is nearby.",
            "The quick brown fox jumps over the lazy dog.",
            None,
            "Last Friday I saw a spotted striped blue worm shake hands with a legless lizard.\tImagine his surprise when he discovered that the safe was full of pudding.",
            "Be careful with that butter knife.\rIt was a slippery slope and he was willing to slide all the way to the deepest depths.\fThe fence was confused about whether it was supposed to keep things in or keep things out.",
            "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I will see if the book has arrived.",
            "The snow-covered path was no help in finding his way out of the back-country. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
            "The stranger officiates the meal. I received a heavy fine but it failed to crush my spirit.\nToday arrived with a crash of my car through the garage door.",
            "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in the middle of July. The furnace repairman indicated the heating system was acting as an air conditioner.",
        ]
    )


@pytest.fixture(
    params=[
        # Basic test: vowel, some # of letters, ends with a
        pytest.param((r"[aeiou][a-z]*a", "", 1), id="vowelLetterStarA_noflags_1"),
        # Adding i flag => does not care about capitalization
        pytest.param(
            (r"[aeiou][a-z ]*", "i", 1),
            id="vowelLetterStarA_ignore_1",
        ),
        # Adding c & e flags => shouldn't do anything
        pytest.param(
            (r"\b[A-Z]*a\b", "ce", 2),
            id="wordCapitalStarA_noflags_2",
            marks=pytest.mark.slow,
        ),
        # Adding c & e & i flags => the i flag wins because it comes after the c
        pytest.param(
            (r"\b([A-Z]*)a\b", "cei", 2),
            id="wordCapitalStarA_ignore_2",
            marks=pytest.mark.slow,
        ),
        # Adding i & c flags => the c flag wins because it comes after the i
        pytest.param((r"\bs.*y\b", "ic", 1), id="wordSDotStarY_noflags_1"),
        # Adding c, i & m flags => the i flag wins because it comes after the c,
        # and the m flag allows linebreaks to interact with anchor expressions
        # (except for REGEXP_LIKE)
        pytest.param((r"^I.*", "icim", 1), id="anchorStart_multilineIgnore_1"),
        # Adding m flag => allows linebreaks to interact with anchor expressions
        # (except for REGEXP_LIKE)
        pytest.param(
            (r"^I.*[\.!]$", "m", 15),
            id="anchorBoth_multiline_15",
            marks=pytest.mark.slow,
        ),
        # Adding m flag => allows linebreaks to interact with anchor expressions
        # (except for REGEXP_LIKE)
        pytest.param(
            (r".*[[:punct:]]$", "m", 1),
            id="anchorRight_multiline_1",
            marks=pytest.mark.slow,
        ),
        # Adding m flag => allows linebreaks to interact with anchor expressions
        pytest.param(
            (r"[125678]{3,6}", "m", 10),
            id="numericRange_multiline_10",
            marks=pytest.mark.slow,
        ),
        # Adding s flag => allows the . symbol to include newline characters
        pytest.param((r"(1.0)|(5.1)", "s", 1), id="1dot0or5dot1_dotall_1"),
        # Adding s flag => allows the . symbol to include newline characters
        pytest.param(
            (r"\b\d.*o\w*\b", "s", 1), id="numOWord_dotall_1", marks=pytest.mark.slow
        ),
        # Adding s flag => allows the . symbol to include newline characters
        pytest.param(
            (r"^I.*\.$", "", 1), id="anchors_noflags_1", marks=pytest.mark.slow
        ),
        # Adding s flag => allows the . symbol to include newline characters
        pytest.param(
            (r"^I.*\.$", "s", 1), id="anchors_dotall_1", marks=pytest.mark.slow
        ),
        # No flags, just testing the behavior of POSIX character classes
        pytest.param((r"[[:ascii:]]{10,30}", "", 1), id="posixFlagsA_noflags_1"),
        # No flags, just testing the behavior of POSIX character classes
        pytest.param(
            (r"[[:cntrl:]]{1,2}", "", 1),
            id="posixFlagsB_noflags_1",
            marks=pytest.mark.slow,
        ),
        # No flags, just testing the behavior of POSIX character classes
        pytest.param(
            (r"[[:digit:][:upper:]]{5,30}", "", pd.Series(list(range(1, 9)) * 8)),
            id="posixFlagsC_noflags_vector",
            marks=pytest.mark.slow,
        ),
        # No flags, just testing the behavior of POSIX character classes
        pytest.param(
            (r"[[:xdigit:]]{1,10}", "", 1),
            id="posixFlagsD_noflags_1",
            marks=pytest.mark.slow,
        ),
        # No flags, just testing the behavior of POSIX character classes
        pytest.param(
            (r"[[:space:]]", "", pd.Series(list(range(1, 9)) * 8)),
            id="posixFlagsE_noflags_vector",
            marks=pytest.mark.slow,
        ),
    ]
)
def regexp_like_count_args(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (r"[[:punct:]]", "", "", pd.Series(list(range(1, 17)) * 4), 0),
            id="punctuation_noflags_delete_vector_all",
        ),
        pytest.param(
            (r"\bthe", "i", "", 1, 1),
            id="the_ignore_delete_1_1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"\bthe \w+", "i", "the xxx", 15, 0),
            id="theWord_ignorecase_hideWord_15_all",
        ),
        pytest.param(
            (r"[^[:ascii:]]", "", "*", 1, pd.Series([0, 1, 0, 2, 0, 3, 0, 4] * 8)),
            id="notAscii_noflags_hideWord_1_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"(\w+)-(\w+)", "", "HYPHEN", 1, 1), id="hyphenatedWord_noflags_HYPHEN_1_1"
        ),
        pytest.param(
            (r"[\d-]", "", "NUMBER", 1, 4),
            id="numberHyphen_noflags_NUMBER_1_4",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"\b(\w+) (\w+) (\w+)", "", "3WORDS", 10, 0),
            id="threeWords_noflags_3WORDS_10_all",
            marks=pytest.mark.slow,
        ),
    ]
)
def regexp_replace_args(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "i", 1, 1, 0, 1),
            id="theTwoWords_ignore_1_1_start_1",
        ),
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "i", 1, 1, 1, 1),
            id="theTwoWords_ignore_1_1_end_1",
        ),
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "i", 1, 1, pd.Series([0, 1] * 32), 1),
            id="theTwoWords_ignore_1_2_vector_1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "i", 10, 1, 1, 1),
            id="theTwoWords_ignore_10_1_end_1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"\bthe\s+(\w+)", "i", 1, 2, 0, 1),
            id="theWord_ignoreExtract_1_1_start_1",
        ),
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "ie", 1, 1, 1, pd.Series([1, 2] * 32)),
            id="theTwoWords_ignoreExtract_1_1_end_vector",
        ),
        pytest.param(
            (r"\bthe\s+(\w+)\s+(\w+)", "ie", 1, 1, 0, 2),
            id="theTwoWords_ignoreExtract_1_1_start_2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"(\d+)(.)(\d+)", "se", 10, 1, 0, 3),
            id="numberAnyNumber_dotallExtract_10_1_start_3",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"[^[:ascii:]]", "", 5, 3, 1, 1),
            id="notAsciiPlus_noflags_5_3_end_1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (r"\W+(\w+)$", "me", 1, 1, 0, 1), id="lastWord_multilineExtract_1_1_start_1"
        ),
    ]
)
def regexp_substr_instr_args(request):
    return request.param


def snowflake_to_python_re(pattern, flags):
    """Transforms a snowflake pattern string and flag string into a Python
    regexp string and flag bitvector using this mapping:
    https://github.com/micromatch/posix-character-classes

    Currently, errors are caused when a null terminator is inside of the
    embedded stirng literals, so [:ascii:] and [:word:] start at character 1
    instead of character 0."""
    pattern = bodo.libs.bodosql_array_kernels.posix_to_re(pattern)
    flags = bodo.libs.bodosql_array_kernels.make_flag_bitvector(flags)
    return pattern, flags


@pytest.mark.parametrize(
    "args",
    [
        pytest.param((("", ""), ("", 0)), id="empty"),
        pytest.param(((r"\w+", "e"), ("\w+", 0)), id="no_effect"),
        pytest.param(
            ((r"[[:alpha:]]{3}\d", "i"), ("[A-Za-z]{3}\d", re.I)),
            id="alpha_ignorecase",
        ),
        pytest.param(
            (
                (r"[[:ascii:]]{1,2}[[:punct:]]", "ce"),
                ("[\x01-\x7f]{1,2}[\]\[!\"#$%&'()*+,./:;<=>?@\^_`{|}~-]", 0),
            ),
            id="ascii_capitalize",
        ),
        pytest.param(
            ((r"[^[:word:]][[:blank:]]", "ci"), ("[^A-Za-z0-9_][ \t]", re.I)),
            id="word_blank_capitalize_capital_ignore",
        ),
        pytest.param(
            (
                (r"[[:lower:]][[:digit:][:upper:]]*", "icm"),
                ("[a-z][0-9A-Z]*", re.M),
            ),
            id="lower_digit_upper_ignore_capital_multiline",
        ),
        pytest.param(
            (
                (r"^[[:xdigit:]]+.[[:space:]]+[[:alnum:]]", "ems"),
                ("^[A-Fa-f0-9]+.[ \t\r\n\v\f]+[A-Za-z0-9]", re.M | re.S),
            ),
            id="xdigit_space_alnum_dotall_multiline",
        ),
        pytest.param(
            (
                (r"[[:graph:]][[:cntrl:]][[:print:]]", "csaebcidcifsmgie"),
                ("[\x21-\x7e][\x01-\x1F\x7f][\x20-\x7e]", re.M | re.S | re.I),
            ),
            id="graph_ctrl_print_allflags_plus_garbage",
        ),
    ],
)
def test_pattern_flag_conversions(args):
    """Verify that the conversion oracles correctly convert POSIX ERE character
    classes to the equivalent form, and flag strings to the correct bitvector"""
    args, answers = args
    patternAnswer, flagAnswer = answers
    outputs = snowflake_to_python_re(*args)
    assert outputs == (
        patternAnswer,
        flagAnswer,
    ), "Did not correctly convert the pattern/flag strings"


def test_regexp_count(regexp_strings, regexp_like_count_args):
    def impl(arr, pattern, position, flags):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_count(arr, pattern, position, flags)
        )

    # Simulates REGEXP_COUNT on a single row
    def regexp_count_scalar_fn(elem, pattern, position, flags):
        if pd.isna(elem) or pd.isna(pattern) or pd.isna(position) or pd.isna(flags):
            return None
        else:
            pattern, flags = snowflake_to_python_re(pattern, flags)
            return len(re.findall(pattern, elem[position - 1 :], flags=flags))

    pattern, flags, position = regexp_like_count_args
    args = (regexp_strings, pattern, position, flags)
    regexp_count_answer = vectorized_sol(args, regexp_count_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=regexp_count_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_regexp_instr(regexp_strings, regexp_substr_instr_args):
    def impl(arr, pattern, position, occurrence, option, flags, group):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_instr(
                arr, pattern, position, occurrence, option, flags, group
            )
        )

    # Simulates REGEXP_INSTR on a single row
    def regexp_instr_scalar_fn(
        elem, pattern, position, occurrence, option, flags, group
    ):
        if (
            pd.isna(elem)
            or pd.isna(pattern)
            or pd.isna(position)
            or pd.isna(occurrence)
            or pd.isna(option)
            or pd.isna(flags)
            or pd.isna(group)
        ):
            return None
        else:
            extract = "e" in flags
            pattern, flags = snowflake_to_python_re(pattern, flags)
            offset = position
            for j in range(occurrence):
                match = re.search(pattern, elem[position - 1 :], flags=flags)
                if match is None:
                    return 0
                start, end = match.span()
                if j == occurrence - 1:
                    if extract:
                        start, end = match.span(group)
                    if option == 0:
                        return offset + start
                    else:
                        return offset + end
                else:
                    elem = elem[end:]
                    offset += end

    pattern, flags, position, occurrence, option, group = regexp_substr_instr_args
    args = (regexp_strings, pattern, position, occurrence, option, flags, group)
    regexp_instr_answer = vectorized_sol(args, regexp_instr_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=regexp_instr_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_regexp_like(regexp_strings, regexp_like_count_args):
    def impl(arr, pattern, flags):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_like(arr, pattern, flags)
        )

    # Simulates REGEXP_LIKE on a single row
    def regexp_like_scalar_fn(elem, pattern, flags):
        if pd.isna(elem) or pd.isna(pattern) or pd.isna(flags):
            return None
        else:
            pattern, flags = snowflake_to_python_re(pattern, flags)
            return re.fullmatch(pattern, elem, flags=flags) is not None

    pattern, flags, _ = regexp_like_count_args
    args = (regexp_strings, pattern, flags)
    regexp_like_answer = vectorized_sol(args, regexp_like_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=regexp_like_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_regexp_replace(regexp_strings, regexp_replace_args):
    def impl(arr, pattern, replacement, position, occurrence, flags):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_replace(
                arr, pattern, replacement, position, occurrence, flags
            )
        )

    # Simulates REGEXP_REPLACE on a single row
    def regexp_replace_scalar_fn(
        elem, pattern, replacement, position, occurrence, flags
    ):
        if (
            pd.isna(elem)
            or pd.isna(pattern)
            or pd.isna(replacement)
            or pd.isna(position)
            or pd.isna(occurrence)
            or pd.isna(flags)
        ):
            return None
        else:
            pattern, flags = snowflake_to_python_re(pattern, flags)
            if occurrence == 0:
                return elem[: position - 1] + re.sub(
                    pattern, replacement, elem[position - 1 :], flags=flags
                )
            else:
                result = elem[: position - 1]
                elem = elem[position - 1 :]
                for i in range(occurrence - 1):
                    match = re.search(pattern, elem, flags=flags)
                    if match is None:
                        return result + elem
                    _, end = match.span()
                    result += elem[:end]
                    elem = elem[end:]
                result += re.sub(pattern, replacement, elem, count=1, flags=flags)
                return result

    pattern, flags, replacement, position, occurrence = regexp_replace_args
    args = (regexp_strings, pattern, replacement, position, occurrence, flags)
    regexp_replace_answer = vectorized_sol(args, regexp_replace_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=regexp_replace_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_regexp_substr(regexp_strings, regexp_substr_instr_args):
    def impl(arr, pattern, position, occurrence, flags, group):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_substr(
                arr, pattern, position, occurrence, flags, group
            )
        )

    # Simulates REGEXP_SUBSTR on a single row
    def regexp_substr_scalar_fn(elem, pattern, position, occurrence, flags, group):
        if (
            pd.isna(elem)
            or pd.isna(pattern)
            or pd.isna(position)
            or pd.isna(occurrence)
            or pd.isna(flags)
            or pd.isna(group)
        ):
            return None
        else:
            extract = "e" in flags
            pattern, flags = snowflake_to_python_re(pattern, flags)
            elem = elem[position - 1 :]
            if extract:
                matches = re.findall(pattern, elem, flags=flags)
                if len(matches) < occurrence:
                    return None
                match = matches[occurrence - 1]
                if isinstance(match, str):
                    return match
                return match[group - 1]
            else:
                for j in range(occurrence):
                    match = re.search(pattern, elem, flags=flags)
                    if match is None:
                        return None
                    start, end = match.span()
                    if j == occurrence - 1:
                        return elem[start:end]
                    else:
                        elem = elem[end:]

    pattern, flags, position, occurrence, _, group = regexp_substr_instr_args
    args = (regexp_strings, pattern, position, occurrence, flags, group)
    regexp_substr_answer = vectorized_sol(args, regexp_substr_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=regexp_substr_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                r"\b(c\w+)\b.*?\b(c\w+)",
                r"<<\2-\1>>",
                0,
                pd.Series(
                    [
                        "Be <<confused-careful>> about whether it was supposed to keep things in or keep things out.",
                        "They decided to plant an orchard of <<candy-cotton>>. His get rich quick scheme was to grow a <<Check-cactus>> back tomorrow; I will see if the book has arrived.",
                        "The snow-<<country-covered>>. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
                        "The stranger officiates the meal. I received a heavy fine but it failed to <<crash-crush>> of my car through the garage door.",
                        "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked <<conditioner-Christmas>>.",
                        "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
                        "Never underestimate the willingness of the greedy to throw you under the bus. Jim liked driving around town with his hazard lights on. If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.",
                        "We accidentally <<circles-created>> in hopes of <<calculator-coming>> had a history, it would be more embarrassing than my browser history.",
                    ]
                ),
            ),
            id="contraction_pattern_replace_all_swapped",
        ),
        pytest.param(
            (
                r"\bthe\W+([ds]\w+)\b.*?\b([ds]\w+)",
                r"the \1 \2",
                0,
                pd.Series(
                    [
                        "Be careful with that butter knife. It was a slippery slope and he was willing to slide all the way to the deepest depths. The fence was confused about whether it was supposed to keep things in or keep things out.",
                        "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I will see if the book has arrived.",
                        "the snow should be obeyed by all except herself. There are no heroes in a punk rock band.",
                        "the stranger spirit. Today arrived with a crash of my car through the garage door.",
                        "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in the middle of July. The furnace repairman indicated the heating system was acting as an air conditioner.",
                        "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
                        "Never underestimate the willingness of the greedy to throw you under the bus. Jim liked driving around town with his hazard lights on. If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.",
                        "We accidentally created a new universe. the doll spun around in circles in hopes of coming alive. If my calculator had a history, it would be more embarrassing than my browser history.",
                    ]
                ),
            ),
            id="contraction_pattern_replace_all_same_order",
        ),
        pytest.param(
            (
                r"\bI\W+(\w+)\W+(\w+)",
                r"I [\1 \2]",
                0,
                pd.Series(
                    [
                        "Be careful with that butter knife. It was a slippery slope and he was willing to slide all the way to the deepest depths. The fence was confused about whether it was supposed to keep things in or keep things out.",
                        "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I [will see] if the book has arrived.",
                        "The snow-covered path was no help in finding his way out of the back-country. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
                        "The stranger officiates the meal. I [received a] heavy fine but it failed to crush my spirit. Today arrived with a crash of my car through the garage door.",
                        "I [would have] gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in the middle of July. The furnace repairman indicated the heating system was acting as an air conditioner.",
                        "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
                        "Never underestimate the willingness of the greedy to throw you under the bus. Jim liked driving around town with his hazard lights on. If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.",
                        "We accidentally created a new universe. The doll spun around in circles in hopes of coming alive. If my calculator had a history, it would be more embarrassing than my browser history.",
                    ]
                ),
            ),
            id="words_replace_all_in_brackets",
        ),
        pytest.param(
            (
                r"\b(\w+)+-(\w+)",
                r"\2-\1",
                0,
                pd.Series(
                    [
                        "Be careful with that butter knife. It was a slippery slope and he was willing to slide all the way to the deepest depths. The fence was confused about whether it was supposed to keep things in or keep things out.",
                        "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I will see if the book has arrived.",
                        "The covered-snow path was no help in finding his way out of the country-back. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
                        "The stranger officiates the meal. I received a heavy fine but it failed to crush my spirit. Today arrived with a crash of my car through the garage door.",
                        "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in the middle of July. The furnace repairman indicated the heating system was acting as an air conditioner.",
                        "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
                        "Never underestimate the willingness of the greedy to throw you under the bus. Jim liked driving around town with his hazard lights on. If eating egg-three omelets causes gain-weight, budgie eggs are a good substitute.",
                        "We accidentally created a new universe. The doll spun around in circles in hopes of coming alive. If my calculator had a history, it would be more embarrassing than my browser history.",
                    ]
                ),
            ),
            id="hyphenated_replace_all_swap_words",
        ),
        pytest.param(
            (
                r"\bthe\W+(\w+)\W+(\w+)",
                r"(THE \1 \2)",
                3,
                pd.Series(
                    [
                        "Be careful with that butter knife. It was a slippery slope and he was willing to slide all the way to the deepest depths. (THE fence was) confused about whether it was supposed to keep things in or keep things out.",
                        "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I will see if the book has arrived.",
                        "The snow-covered path was no help in finding his way out of the back-country. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
                        "The stranger officiates the meal. I received a heavy fine but it failed to crush my spirit. Today arrived with a crash of my car through (THE garage door).",
                        "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in (THE middle of) July. The furnace repairman indicated the heating system was acting as an air conditioner.",
                        "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
                        "Never underestimate the willingness of the greedy to throw you under (THE bus Jim) liked driving around town with his hazard lights on. If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.",
                        "We accidentally created a new universe. The doll spun around in circles in hopes of coming alive. If my calculator had a history, it would be more embarrassing than my browser history.",
                    ]
                ),
            ),
            id="words_replace_3",
        ),
    ],
)
def test_regexp_replace_backreferences(args):
    """Tests REGEXP_REPLACE with backreferences"""

    def impl(arr, pattern, replacement, position, occurrence, flags):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.regexp_replace(
                arr, pattern, replacement, position, occurrence, flags
            )
        )

    arr = pd.Series(
        [
            "Be careful with that butter knife. It was a slippery slope and he was willing to slide all the way to the deepest depths. The fence was confused about whether it was supposed to keep things in or keep things out.",
            "They decided to plant an orchard of cotton candy. His get rich quick scheme was to grow a cactus farm. Check back tomorrow; I will see if the book has arrived.",
            "The snow-covered path was no help in finding his way out of the back-country. Karen believed all traffic laws should be obeyed by all except herself. There are no heroes in a punk rock band.",
            "The stranger officiates the meal. I received a heavy fine but it failed to crush my spirit. Today arrived with a crash of my car through the garage door.",
            "I would have gotten the promotion, but my attendance wasn't good enough. He was the type of guy who liked Christmas lights on his house in the middle of July. The furnace repairman indicated the heating system was acting as an air conditioner.",
            "Peanuts don't grow on trees, but cashews do. The murder hornet was disappointed by the preconceived ideas people had of him. Getting up at dawn is for the birds.",
            "Never underestimate the willingness of the greedy to throw you under the bus. Jim liked driving around town with his hazard lights on. If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.",
            "We accidentally created a new universe. The doll spun around in circles in hopes of coming alive. If my calculator had a history, it would be more embarrassing than my browser history.",
        ]
    )
    position, flags = 1, "i"
    pattern, replacement, occurrence, answer = args
    check_func(
        impl,
        (arr, pattern, replacement, position, occurrence, flags),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("like", "", True), id="like_empty"),
        pytest.param(("like", "hello", False), id="like_nonempty"),
        pytest.param(("count", "", 0), id="count_empty"),
        pytest.param(("count", "hello", 0), id="count_nonempty"),
        pytest.param(("replace", "", ""), id="replace_empty"),
        pytest.param(("replace", "hello", "hello"), id="replace_nonempty"),
        pytest.param(("substr", "", None), id="substr_empty"),
        pytest.param(("substr", "hello", None), id="substr_nonempty"),
        pytest.param(("instr", "", 0), id="instr_empty"),
        pytest.param(("instr", "hello", 0), id="instr_nonempty"),
    ],
)
def test_regexp_empty_pattern(args):
    """Tests REGEXP functions with empty pattern"""

    def impl1(source):
        return bodo.libs.bodosql_array_kernels.regexp_like(source, "", "")

    def impl2(source):
        return bodo.libs.bodosql_array_kernels.regexp_count(source, "", 1, "")

    def impl3(source):
        return bodo.libs.bodosql_array_kernels.regexp_replace(source, "", "*", 1, 0, "")

    def impl4(source):
        return bodo.libs.bodosql_array_kernels.regexp_substr(source, "", 1, 1, "", 1)

    def impl5(source):
        return bodo.libs.bodosql_array_kernels.regexp_instr(source, "", 1, 1, 0, "", 1)

    func, source, answer = args
    impl = {
        "like": impl1,
        "count": impl2,
        "replace": impl3,
        "substr": impl4,
        "instr": impl5,
    }[func]
    check_func(
        impl,
        (source,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.skip(
    "The following are examples where the current REGEXP functions diverge from the POSIX semantics"
)
@pytest.mark.parametrize(
    "args",
    [
        # POSIX would match on the 5s instead of the 3s, thus allowing it to
        # capture more of the x's
        pytest.param(
            ("xxxxxxxxxx", r"(xxx|xxxxx)*", 1, 1, "", 1, "xxxxxxxxxx"),
            id="matches_3x_instead_of_5x",
        ),
        # POSIX would match on the longer of the two choices if they both match,
        # Python matches on whichever one matches first (left-to-right)
        pytest.param(
            ("PrefixSuffix", r"(Prefix|PrefixSuffix)", 1, 1, "", 1, "PrefixSuffix"),
            id="matches_shorter_choice",
        ),
        # POSIX supports equivalence classes, but Python does not
        pytest.param(
            ("aàâba", r"[=a=]*", 1, 1, "", 1, "aàâ"), id="uses_equivalence_classes"
        ),
    ],
)
def test_regexp_posix_unsupported(args):
    """
    Snowflake SQL Regexp specifications:
        https://docs.snowflake.com/en/sql-reference/functions-regexp.html#label-regexp-general-usage-notes
    Wiki page on POSIX ERE spec:
        https://en.wikipedia.org/wiki/Regular_expression#POSIX_basic_and_extended
    Python docs on re module (leftmost-earliest nature of "|" noted here):
        https://docs.python.org/3/library/re.html
    Other docs on POSIX ERE (leftmost-longest nature of "|" noted here):
        https://www.regular-expressions.info/posix.html
    """

    def impl(arr, pattern, position, occurrence, flags, group):
        return bodo.libs.bodosql_array_kernels.regexp_substr(
            arr, pattern, position, occurrence, flags, group
        )

    arr, pattern, position, occurrence, flags, group, answer = args
    check_func(
        impl,
        (arr, pattern, position, occurrence, flags, group),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "pattern,answer",
    [
        pytest.param(
            r"\b(\w+)\b.*\1",
            "alpha beta gamma delta alpha",
            id="backreferences_single",
        ),
        pytest.param(
            r"\b(\w)(.)\w+\1.*\2",
            "alpha beta gamma delta alphabet soup is del",
            id="backreferences_multiple",
        ),
        pytest.param(r".*?alpha", "epsilon alpha", id="lazy"),
        pytest.param(r"\A\w+", "epsilon", id="A"),
        pytest.param(r"\w+\Z", "delicious", id="Z"),
        pytest.param(r"(?i)[AEIOU]\w+", "epsilon", id="question_flags"),
        pytest.param(
            r"(?<= )\w{6}\w+(?= )", "alphabet", id="question_lookahead_lookbehind"
        ),
        pytest.param(r"(?<![an]) \w+", " soup", id="question_lookbehind_negative"),
    ],
)
def test_regexp_python_only(pattern, answer):
    """
    Tests regular expression syntactic forms that are supported by Python re,
    but not by Snowflake. If our underlying engine for regular expressions is
    updated, these tests may no longer be supported.
    """

    def impl(arr, pattern, position, occurrence, flags, group):
        return bodo.libs.bodosql_array_kernels.regexp_substr(
            arr, pattern, position, occurrence, flags, group
        )

    check_func(
        impl,
        (
            "epsilon alpha beta gamma delta alphabet soup is delicious",
            pattern,
            1,
            1,
            "",
            1,
        ),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        # These tests are only looking at scalars
        dist_test=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("func", ["like", "count", "replace", "substr", "instr"])
def test_option_regexp(func):
    def impl1(optFlag):
        source = (
            "The dragon and the wolf fought side by side in the war"
            if optFlag
            else None
        )
        pattern = r"(the)\s(\S+)"
        flag = "ie"
        return bodo.libs.bodosql_array_kernels.regexp_like(source, pattern, flag)

    def impl2(optFlag):
        source = "The dragon and the wolf fought side by side in the war"
        pos = 1 if optFlag else None
        pattern = r"(the) (\w+)"
        flag = "ie"
        return bodo.libs.bodosql_array_kernels.regexp_count(source, pattern, pos, flag)

    def impl3(optFlag):
        source = (
            "The dragon and the wolf fought side by side in the war"
            if optFlag
            else None
        )
        pos = 1 if optFlag else None
        occur = 2
        pattern = r"(\Bhe) (\w+)"
        replacement = r"\1 [\2]" if optFlag else None
        flag = "ie"
        return bodo.libs.bodosql_array_kernels.regexp_replace(
            source, pattern, replacement, pos, occur, flag
        )

    def impl4(optFlag):
        source = "The dragon and the wolf fought side by side in the war"
        pos = 1 if optFlag else None
        occur = 2 if optFlag else None
        pattern = r"(the) (\D\w+)"
        flag = "ie"
        group = 2 if optFlag else None
        return bodo.libs.bodosql_array_kernels.regexp_substr(
            source, pattern, pos, occur, flag, group
        )

    def impl5(optFlag):
        source = "The dragon and the wolf fought side by side in the war"
        pos = 1
        occur = 2
        opt = 0 if optFlag else None
        pattern = r"(the) (\w+)"
        flag = "ie"
        group = 2 if optFlag else None
        return bodo.libs.bodosql_array_kernels.regexp_instr(
            source, pattern, pos, occur, opt, flag, group
        )

    for optFlag in [True, False]:
        impl, answer = {
            "like": (impl1, False),
            "count": (impl2, 3),
            "replace": (
                impl3,
                "The dragon and the [wolf] fought side by side in the war",
            ),
            "substr": (impl4, "wolf"),
            "instr": (impl5, 20),
        }[func]
        check_func(
            impl,
            (optFlag,),
            py_output=answer if optFlag else None,
        )


@pytest.mark.parametrize("in_series", [True, False])
@pytest.mark.parametrize(
    "func,args",
    [
        pytest.param("count", (" ", " ", -3, ""), id="count_posneg"),
        pytest.param("replace", (" ", " ", "", 0, 0, ""), id="replace_poszero"),
        pytest.param("replace", (" ", " ", "", 4, -1, ""), id="replace_occneg"),
        pytest.param("substr", (" ", " ", -1, 1, "", 1), id="substr_posneg"),
        pytest.param("substr", (" ", " ", 4, 0, "", 1), id="substr_occzero"),
        pytest.param("substr", ("  ", "(.)(.)", 4, 0, "e", 3), id="substr_groupthree"),
        pytest.param("instr", (" ", " ", 0, 1, 0, "", 1), id="instr_poszero"),
        pytest.param("instr", (" ", " ", 1, 0, 0, "", 1), id="instr_occzero"),
        pytest.param("instr", (" ", " ", 1, 1, -1, "", 1), id="instr_optneg"),
        pytest.param("instr", (" ", " ", 1, 1, 2, "", 1), id="instr_opttwo"),
        pytest.param("instr", (" ", " ", 1, 1, 0, "e", 0), id="instr_groupzero"),
    ],
)
def test_regexp_error(func, args, in_series):
    """
    Tests cases that cause error messages:
    - non-positive positions/occurrences (0 allowed for REGEXP_REPLACE)
    - option is not 0 or 1
    - group_num is not a number in the valid range (only matters if extracting)
    - all of the above with and without series
    """

    def impl1(arr, pattern, position, flags):
        return bodo.libs.bodosql_array_kernels.regexp_count(
            arr, pattern, position, flags
        )

    def impl2(arr, pattern, replacement, position, occurrence, flags):
        return bodo.libs.bodosql_array_kernels.regexp_replace(
            arr, pattern, replacement, position, occurrence, flags
        )

    def impl3(arr, pattern, position, occurrence, flags, group):
        return bodo.libs.bodosql_array_kernels.regexp_substr(
            arr, pattern, position, occurrence, flags, group
        )

    def impl4(arr, pattern, position, occurrence, opt, flags, group):
        return bodo.libs.bodosql_array_kernels.regexp_instr(
            arr, pattern, position, occurrence, opt, flags, group
        )

    impl = {
        "count": impl1,
        "replace": impl2,
        "substr": impl3,
        "instr": impl4,
    }[func]

    if in_series:
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                new_args.append(arg)
            else:
                new_args.append(pd.Series([arg] * 5))
        args = tuple(new_args)

    with pytest.raises(ValueError):
        check_func(impl, args)
