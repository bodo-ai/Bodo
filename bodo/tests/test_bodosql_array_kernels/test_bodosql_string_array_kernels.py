# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL string functions
"""


from builtins import round as py_round

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import (
    check_func,
    gen_nonascii_list,
    pytest_slow_unless_codegen,
)

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (pd.Series(["alpha", "beta", "zeta", "pi", "epsilon"])),
                "e",
            ),
        ),
        pytest.param(
            (
                (pd.Series(["", "zenith", "zebra", "PI", "fooze"])),
                "ze",
            ),
        ),
        pytest.param(
            (
                (pd.Series([b"00000", b"", b"**", b"918B*a", b""])),
                b"",
            ),
        ),
    ],
)
def test_contains(args, memory_leak_check):
    def impl(arr, pattern):
        return pd.Series(bodo.libs.bodosql_array_kernels.contains(arr, pattern))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, (pd.Series, np.ndarray)) for arg in args):
        impl = lambda arr, pattern: bodo.libs.bodosql_array_kernels.contains(
            arr, pattern
        )

    # Simulates CONTAINS on a single row
    def contains_scalar_fn(elem, pattern):
        if pd.isna(elem) or pd.isna(pattern):
            return False
        else:
            return pattern in elem

    arr, pattern = args
    contains_answer = vectorized_sol((arr, pattern), contains_scalar_fn, object)
    check_func(
        impl,
        (arr, pattern),
        py_output=contains_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_contains_optional(memory_leak_check):
    """Test the optional code path for contains."""

    def impl(A, pattern, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = pattern if flag1 else None
        return bodo.libs.bodosql_array_kernels.contains(arg0, arg1)

    A = "mean"
    pattern = "e"
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            py_output = True if flag0 and flag1 else None
            check_func(
                impl,
                (A, pattern, flag0, flag1),
                py_output=py_output,
            )


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(
            pd.Series(pd.array([65, 100, 110, 0, 33])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_char(n, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.char(arr))

    # avoid Series conversion for scalar output
    if not isinstance(n, pd.Series):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.char(arr)

    # Simulates CHAR on a single row
    def char_scalar_fn(elem):
        if pd.isna(elem) or elem < 0 or elem > 127:
            return None
        else:
            return chr(elem)

    chr_answer = vectorized_sol((n,), char_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (n,),
        py_output=chr_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pd.Series(
                        [
                            "book",
                            "book",
                            "book",
                            "book",
                            "alpha",
                            "alpha",
                            "kitten",
                            "alpha",
                            "alphabet soup is delicious",
                            "Be careful with the butter knife.",
                            None,
                            "A",
                            None,
                            "Hello! Goodbye!",
                            "Hello!",
                        ]
                    ),
                    pd.Series(
                        [
                            "books",
                            "book",
                            "boo",
                            "boot",
                            "beta",
                            "gamma",
                            "sitting",
                            "alphabet",
                            "alpha beta gamma delta",
                            "The careful fence was utterly confused.",
                            "B",
                            None,
                            None,
                            "Goodbye!",
                            "Hello! Goodbye!",
                        ]
                    ),
                ),
                pd.Series(
                    [1, 0, 1, 1, 4, 4, 3, 3, 15, 19, None, None, None, 7, 9],
                    dtype=pd.UInt16Dtype(),
                ),
            ),
            id="vector_vector_no_max",
        ),
        pytest.param(
            (
                (
                    pd.Series(
                        [
                            "the world of coca-cola is a museum to the company",
                            "i'd like to buy the world a coke and keep it company it's the real thing",
                            "i'd like to buy the world a home and furnish it with love grow apple trees and honey bees and snow white turtle doves",
                            "i'd like to teach the world to sing in perfect harmony i'd like to buy the world a coke and keep it company that's the real thing",
                            "",
                            "id love to buy the world a pepsi and keep it warm it is really sad",
                            "i'd  buy the world a coke and like to keep it company it's the real thing",
                        ]
                    ),
                    "i'd like to buy the world a coke and keep it company it's the real thing",
                ),
                pd.Series([48, 0, 65, 58, 72, 25, 15], dtype=pd.UInt16Dtype()),
            ),
            id="vector_scalar_no_max",
        ),
        pytest.param(
            (
                (
                    pd.Series(
                        [
                            "",
                            "disappointment",
                            None,
                            "corruption",
                            "jazz",
                            "admonition",
                            "revival",
                            "correspondence",
                            "infrastructure",
                            "ventriloquizing",
                            "municipalizing",
                            "station",
                            "blackjack",
                            "crackerjacks",
                            "recommend",
                            "recommend",
                            "commend",
                            None,
                            "accommodate",
                            "dependable",
                            "precedented",
                            "commendation",
                            "recommendations",
                            "réccömΣnd@t10n",
                            "noitadnemmocer",
                            "nonmetameric",
                            "coordinate",
                            "denominator",
                            "intercommunication",
                            "gravitation",
                            "redifferentiation",
                            "redistribution",
                            "recognitions",
                        ]
                    ),
                    "recommendation",
                    10,
                ),
                pd.Series(
                    [
                        10,
                        10,
                        None,
                        8,
                        10,
                        8,
                        10,
                        10,
                        10,
                        10,
                        10,
                        9,
                        10,
                        10,
                        5,
                        5,
                        7,
                        None,
                        7,
                        9,
                        9,
                        2,
                        1,
                        7,
                        10,
                        10,
                        10,
                        7,
                        7,
                        9,
                        8,
                        8,
                        6,
                    ],
                    dtype=pd.UInt16Dtype(),
                ),
            ),
            id="vector_scalar_with_scalar_max",
        ),
        pytest.param(
            (
                (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Sociis natoque penatibus et magnis dis parturient montes. Dolor morbi non arcu risus quis varius quam quisque. Sed libero enim sed faucibus. Erat pellentesque adipiscing commodo elit at imperdiet dui accumsan. Nisi porta lorem mollis aliquam ut porttitor. Parturient montes nascetur ridiculus mus mauris vitae. Quis eleifend quam adipiscing vitae proin sagittis. Fusce id velit ut tortor pretium viverra suspendisse potenti nullam. Sit amet mattis vulputate enim nulla aliquet porttitor lacus. Quis imperdiet massa tincidunt nunc pulvinar sapien. Duis tristique sollicitudin nibh sit amet commodo nulla facilisi nullam.",
                    "Tortor id aliquet lectus proin nibh nisl condimentum id venenatis. Morbi tristique senectus et netus et. Mollis nunc sed id semper risus in. Tristique et egestas quis ipsum. Vel facilisis volutpat est velit egestas dui id ornare arcu. Consequat nisl vel pretium lectus. Ultricies leo integer malesuada nunc vel risus commodo viverra maecenas. Sed vulputate mi sit amet mauris commodo quis imperdiet. Amet dictum sit amet justo donec enim diam vulputate. Facilisi etiam dignissim diam quis enim lobortis scelerisque fermentum dui. Rhoncus urna neque viverra justo nec ultrices dui. Fermentum et sollicitudin ac orci phasellus egestas tellus. Donec pretium vulputate sapien nec. Nunc mattis enim ut tellus elementum sagittis vitae. Commodo viverra maecenas accumsan lacus vel facilisis volutpat est velit. Fringilla ut morbi tincidunt augue interdum. Nunc sed augue lacus viverra vitae congue.",
                    pd.Series([100, 300, 500, 700, 900, None], dtype=pd.UInt16Dtype()),
                ),
                pd.Series([100, 300, 500, 640, 640, None], dtype=pd.UInt16Dtype()),
            ),
            id="scalar_scalar_with_vector_max",
        ),
        pytest.param(
            (
                (
                    pd.Series(
                        [
                            None,
                            "A",
                            None,
                            "book",
                            "book",
                            "book",
                            "book",
                            "alpha",
                            "alpha",
                            "kitten",
                            "alpha",
                            "alphabet soup is delicious",
                            "Be careful with the butter knife.",
                        ]
                    ),
                    pd.Series(
                        [
                            "B",
                            None,
                            None,
                            "books",
                            "book",
                            "boo",
                            "boot",
                            "beta",
                            "gamma",
                            "sitting",
                            "alphabet",
                            "alpha beta gamma delta",
                            "The careful fence was utterly confused.",
                        ]
                    ),
                    3,
                ),
                pd.Series(
                    [None, None, None, 1, 0, 1, 1, 3, 3, 3, 3, 3, 3],
                    dtype=pd.UInt16Dtype(),
                ),
            ),
            id="vector_vector_with_scalar_max",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (("", ""), 0), id="all_scalar_no_max_empty", marks=pytest.mark.slow
        ),
        pytest.param(
            (
                (
                    "eylsquxvzltzdjakokuchnkaftepittjleqalnerroggysvpwplhchmuvvkaknfclvuskzvpeyqhchuhpddqiaqpndwenlehmxohtyrohedpyeqnxmncpfbimyltuabapamvenenayvwwaszukswevziqjvdlvpypvtdnhtwnuarxeytrotryuoblrebfmksolndgvtlzvqqrmyxxciogvqehrndlbczloupugfaxbtrufnaqwjxizomvhwnhibfxhowguyntthvhxatzvvefsrmycljhrwmisnzsavipdceibrmlxurswirrixhbscypcrpbzvcjadqinhtitpazbmagudkhvggeqnczeteeoqbkiuqkrkuptqcotngyymipwogofxdlghvnbfbnglnltjwurrzkzujcaaxknerwxvrifnvlehuahqmnepmazemupvzadxmigjhrbchsjhkwnnronkuvhbjqrxvgmszlagrcaxahauckxpzcfzdkqqrtehvcjogmbrtjjbemdnevygmwgpbbhertaywzuvowtnpspukqfkidjcufdxdwqpvqygprcmtoqqbjpurpzxqkeyxdafaojsixxqijtukwshscgmeisrxmpfqtgplcdkmlofjrgffwqdykrybwhjzbkzxnalpkrhuorsqsbwxddczjbdhdmtspytpluvyaaftujsrsdilefromyoyuzrwascywcdhsgjwbgdhclladwnpgpyvracukvvfnkcnixqkfchxwagtbbgzjcnwslfdpvojdinkvqpthfqcvfqzhrutbxngsfuccfzsyzfcyrqdcktqfrlrtrwmrxrvrgzjivqftbddgxvacnmgjfkbuhxlanoxpodtzvlxjxlmlpbssogoboawmcgddmrhwjkxvmazmyhaoarpdflsnghfmhapkgawfgtozqheedhzcpzzgkylpaoduyfhkcrsfgbwwjatbanwgfzqibxmvfpm",
                    "rcjorpssgddcwbuwktpfghklbtkotktjeyyhnrakgnmzuasgsfcvuatcwfsayibwiencbwigkgeowvmwtintxvuigqxnmjpqabiwqmcuothpsqqrkyjcxydbtlzlrfkfaaiquapmfeeaixuluxgjfciqttkqknpemrkxjygdjygwsklyzzuannpjemtuiketxhbebaujorcnbupvflluzbuphavtwiahitubvlljhgkipboskqateqhiaqxpzyceafqjzmenuwzyaywoktecdkgjllvlmeqiqwaeeeoxqqpwwlefpctoddwyxujduwrsspejgsijijapvwiwmspjcbuzznzvtlwfgotipbiaglsggzvvaloxmptiwtrhfpkcsfnupgljizkltzplypcwybiixxcqqhwfxqxnhtlpasegpjlqzjqnggdkafknbetrwqiudftotojpkfjymwlryogcsajwnnhrentiuypyyfbhsldcbxnepkslqslojphugvdjzffgjhqtuowmfxwfaoltiaqkhqbanqrzygacjisspqtrmmbiadqdglevjorfpxovqyqnmkzszkldwznojynegljoikwodklssjfozqyxfrnnijmzhbhgilzabwumzvsrjnsqmtanmfsixqhdunpwnlgztajnmepkhficlyeyuzswlhelhrboecofupivplvwchzlijzlmimmiditxiyegygfigshmaqabumugnliuqnwluqutwwdjrjzisjxqiozquivpifplvidlegtoqptgqaeperpnxcxzalxoymvetwizvtipdnnixncdedkldrbkrbkcpsoouvzndelfxkthtytrerlipyejxjbxagbaqphpijdqsryjenkfcquqevmefktrbxdtqthvlvzaslojfqwwgcpxxlyqfqhzbjvkscrjjsvzcmsaailqpvznqryyvrmfxsuwfrqgbeuhqhmsnxkikinhclcdfvv",
                ),
                884,
            ),
            id="all_scalar_no_max_large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (("batman", "superman", 10), 5),
            id="all_scalar_with_max",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (("yay", None), None), id="all_scalar_with_null", marks=pytest.mark.slow
        ),
    ],
)
def test_editdistance(args, memory_leak_check):
    """Answers calculated with https://planetcalc.com/1721/"""

    def impl1(s, t):
        return pd.Series(bodo.libs.bodosql_array_kernels.editdistance_no_max(s, t))

    def impl2(s, t, maxDist):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.editdistance_with_max(s, t, maxDist)
        )

    args, answer = args

    # avoid Series conversion for scalar output
    if not isinstance(answer, pd.Series):
        impl1 = lambda s, t: bodo.libs.bodosql_array_kernels.editdistance_no_max(s, t)
        impl2 = (
            lambda s, t, maxDist: bodo.libs.bodosql_array_kernels.editdistance_with_max(
                s, t, maxDist
            )
        )

    if len(args) == 2:
        check_func(impl1, args, py_output=answer, check_dtype=False, reset_index=True)
    elif len(args) == 3:
        check_func(impl2, args, py_output=answer, check_dtype=False, reset_index=True)


@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        "jake",
                        "jake",
                        "amy",
                        "amy",
                        "Gute nacht",
                        "Ich weiß nicht",
                        "Ich weiß nicht",
                        "Snowflake",
                        "Snowflake",
                        "",
                        "",
                        "scissors",
                        "scissors",
                        "alpha beta",
                    ]
                ),
                pd.Series(
                    [
                        "JOE",
                        "john",  # 50 on Snowflake, 55 with the wiki algorithm
                        "mary",
                        "jean",
                        "Ich weis nicht",
                        "Ich wei? nicht",
                        "Ich weiss nicht",
                        "Oracle",
                        "Snowflake",
                        "Snowflake",
                        "",
                        None,
                        "scolding",  # 58 on Snowflake, 66 with the wiki algorithm
                        "Alphabet Soup",
                    ]
                ),
            ),
            pd.Series(
                [75, 50, 80, 0, 56, 97, 95, 61, 100, 0, 0, None, 58, 87],
                dtype=pd.Int8Dtype(),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        "The quick",
                        None,
                        "brown",
                        "The lazy green ferret leaps around the zippy wolf",  # 65 on Snowflake, 79 with the wiki algorithm
                        "god yzal eht revo spmuj xof nworb kciuq ehT",
                        "lazy dog",
                        "lazy",
                        "Quick the Fox brown Over jumps dog Lazy the",
                        "hte quick brown fox jumps over hte lazy dog",
                    ]
                ),
                "The quick brown fox jumps over the lazy dog",
            ),
            pd.Series(
                [84, None, 70, 65, 66, 43, 0, 86, 98],
                dtype=pd.Int8Dtype(),
            ),
            id="vector_scalar",
        ),
        pytest.param(
            (
                "pseudopseudohypoparathyroidism",
                "supercalifragilisticexpialidocious",
            ),
            56,
            id="all_scalar",
        ),
    ],
)
def test_jarowinkler_similarity(args, answer, memory_leak_check):
    """
    Answers calculated via https://tilores.io/jaro-winkler-distance-algorithm-online-tool
    (accounting for the fact that the online calculator is case-sensitive by
    re-writing all strings as lowercase before calculating their refsol)
    """

    def impl(s, t):
        return pd.Series(bodo.libs.bodosql_array_kernels.jarowinkler_similarity(s, t))

    # avoid Series conversion for scalar output
    if not isinstance(answer, pd.Series):
        impl = lambda s, t: bodo.libs.bodosql_array_kernels.jarowinkler_similarity(s, t)

    check_func(impl, args, py_output=answer, check_dtype=False, reset_index=True)
    check_func(impl, args[::-1], py_output=answer, check_dtype=False, reset_index=True)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                np.array(
                    [
                        15.112345,
                        1234567890,
                        np.NAN,
                        17,
                        -13.6413,
                        1.2345,
                        12345678910111213.141516171819,
                    ]
                ),
                pd.Series(pd.array([3, 4, 6, None, 0, -1, 5])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                12345678.123456789,
                pd.Series(pd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param((-426472, 2), id="all_scalar_not_null"),
        pytest.param((None, 5), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_format(args, memory_leak_check):
    def impl(arr, places):
        return pd.Series(bodo.libs.bodosql_array_kernels.format(arr, places))

    # Simulates FORMAT on a single row
    def format_scalar_fn(elem, places):
        if pd.isna(elem) or pd.isna(places):
            return None
        elif places <= 0:
            return "{:,}".format(py_round(elem))
        else:
            return (f"{{:,.{places}f}}").format(elem)

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, places: bodo.libs.bodosql_array_kernels.format(arr, places)

    arr, places = args
    format_answer = vectorized_sol((arr, places), format_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (arr, places),
        py_output=format_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pd.Series(
                        pd.array(
                            ["aLpHaBET Soup iS DeLicious"] * 3
                            + ["alpha beta gamma delta epsilon"] * 3
                        )
                    ),
                    pd.Series(pd.array([" ", "", "aeiou"] * 2)),
                ),
                pd.Series(
                    pd.array(
                        [
                            "Alphabet Soup Is Delicious",
                            "Alphabet soup is delicious",
                            "ALphaBet soUP iS deLiCiOUS",
                            "Alpha Beta Gamma Delta Epsilon",
                            "Alpha beta gamma delta epsilon",
                            "ALpha beTa gaMma deLta ePsiLoN",
                        ]
                    )
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                (
                    pd.Series(
                        pd.array(
                            [
                                "sansa,arya+gendry,\nrob,jon,bran,rickon",
                                "cersei+jaime,\ntyrion",
                                "daenerys+daario,missandei+grey_worm,\njorah,selmy",
                                "\nrobert,stannis,renly+loras",
                                None,
                            ]
                        )
                    ),
                    " \n\t.,;~!@#$%^&*()-+_=",
                ),
                pd.Series(
                    pd.array(
                        [
                            "Sansa,Arya+Gendry,\nRob,Jon,Bran,Rickon",
                            "Cersei+Jaime,\nTyrion",
                            "Daenerys+Daario,Missandei+Grey_Worm,\nJorah,Selmy",
                            "\nRobert,Stannis,Renly+Loras",
                            None,
                        ]
                    )
                ),
            ),
            id="vector_scalar",
        ),
        pytest.param(
            (("a+b=c,a+d=e,b-c=d", "+-=,"), "A+B=C,A+D=E,B-C=D"),
            id="all_scalar",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_initcap(args, memory_leak_check):
    def impl(arr, delim):
        return pd.Series(bodo.libs.bodosql_array_kernels.initcap(arr, delim))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, delim: bodo.libs.bodosql_array_kernels.initcap(arr, delim)

    args, answer = args
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(
            pd.Series(
                [" r 32r23 ", "   ", "3r", "", "R#2r3", "🟥🟧🟨🟩🟦🟪 foo 🟥🟧🟨🟩🟦🟪"] * 2
                + [None] * 4
            ).values,
            id="vector",
        ),
        pytest.param(
            "  fewfew   ",
            id="scalar",
        ),
    ],
)
def test_string_one_arg_fns(arg, memory_leak_check):
    """
    Tests for the BodoSQL array kernel for length, upper, lower, trim,
    ltrim, and rtrim. We test theses together because they use
    the same infrastructure.
    """

    def impl1(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.length(arr))

    def impl2(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.upper(arr))

    def impl3(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.lower(arr))

    # avoid Series conversion for scalar output
    if isinstance(arg, str):
        impl1 = lambda arr: bodo.libs.bodosql_array_kernels.length(arr)
        impl2 = lambda arr: bodo.libs.bodosql_array_kernels.upper(arr)
        impl3 = lambda arr: bodo.libs.bodosql_array_kernels.lower(arr)

    def length_fn(val):
        if pd.isna(val):
            return None
        else:
            return len(val)

    length_fn = lambda val: None if pd.isna(val) else len(val)
    upper_fn = lambda val: None if pd.isna(val) else val.upper()
    lower_fn = lambda val: None if pd.isna(val) else val.lower()

    answer1 = vectorized_sol((arg,), length_fn, object)
    answer2 = vectorized_sol((arg,), upper_fn, object)
    answer3 = vectorized_sol((arg,), lower_fn, object)
    check_func(
        impl1,
        (arg,),
        py_output=answer1,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        (arg,),
        py_output=answer2,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl3,
        (arg,),
        py_output=answer3,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "inputs, answer",
    [
        pytest.param(("bodo.ai", 1, 3, ""), "o.ai", id="delete_start"),
        pytest.param(("bodo.ai", 5, 3, ""), "bodo", id="delete_end"),
        pytest.param(("bodo.ai", 1, 0, ""), "bodo.ai", id="delete_none"),
        pytest.param(("bodo.ai", 1, 7, ""), "", id="delete_all"),
        pytest.param(("bodo", 1, 4, "ai"), "ai", id="replace_1"),
        pytest.param(("bodo", 3, 2, "ai"), "boai", id="replace_2"),
        pytest.param(("bodo", 1, 0, "ai"), "aibodo", id="append_start"),
        pytest.param(("bodo", 3, 0, "ai"), "boaido", id="append_middle"),
        pytest.param(("bodo", 5, 2, "ai"), "bodoai", id="append_end"),
        pytest.param((b"bar", 1, 0, b"foo"), b"foobar", id="append_end_binary"),
        pytest.param(
            (b"The quick brown fox", 11, 5, b"red"),
            b"The quick red fox",
            id="delete_start_binary",
        ),
        pytest.param(
            (b"the fast fox", 5, 4, b"foo"), b"the foo fox", id="replace_binary"
        ),
    ],
)
def test_insert_scalar(inputs, answer, memory_leak_check):
    impl = lambda source, pos, len, inject: bodo.libs.bodosql_array_kernels.insert(
        source, pos, len, inject
    )
    check_func(
        impl,
        inputs,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(("bodo.ai", 0, 3, ""), id="string_error_1"),
        pytest.param(("bodo.ai", 1, -3, ""), id="string_error_2"),
        pytest.param(("bodo.ai", -1, -3, ""), id="string_error_3"),
        pytest.param((b"the fast fox", 0, 4, b"foo"), id="binary_error_1"),
        pytest.param((b"the fast fox", 5, -1, b"foo"), id="binary_error_2"),
        pytest.param((b"the fast fox", -1, 2, b"foo"), id="binary_error_3"),
    ],
)
def test_insert_scalar_error(inputs):
    impl = lambda source, pos, len, inject: bodo.libs.bodosql_array_kernels.insert(
        source, pos, len, inject
    )
    source, pos, len, inject = inputs
    if pos < 1:
        with pytest.raises(AssertionError, match="<pos> argument must be at least 1!"):
            bodo.jit(impl)(source, pos, len, inject)
    elif len < 0:
        with pytest.raises(AssertionError, match="<len> argument must be at least 0!"):
            bodo.jit(impl)(source, pos, len, inject)


@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            (
                pd.Series(["alphabet"] * 4 + ["soup"] * 4),
                pd.Series([1, 2, 3, 4] * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, 2, 0, 3] * 2, dtype=pd.Int32Dtype()),
                pd.Series(["X", "X", "****", "****"] * 2),
            ),
            pd.Series(
                [
                    "Xalphabet",
                    "aXhabet",
                    "al****phabet",
                    "alp****et",
                    "Xsoup",
                    "sXp",
                    "so****up",
                    "sou****",
                ]
            ),
            id="all_vector_string",
        ),
        pytest.param(
            (
                pd.Series([b"the quick fox"] * 5),
                pd.Series([5, 5, None, 5, 5], dtype=pd.Int32Dtype()),
                pd.Series([6, 6, 6, 0, 0], dtype=pd.Int32Dtype()),
                pd.Series([b"fast ", b"fast and ", b"fast", b"fast and ", b"fast"]),
            ),
            pd.Series(
                [
                    b"the fast fox",
                    b"the fast and fox",
                    None,
                    b"the fast and quick fox",
                    b"the fastquick fox",
                ]
            ),
            id="all_vector_binary",
        ),
        pytest.param(
            (
                "123456789",
                pd.Series([1, 3, 5, 7, 9, None] * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, 1, 2, 3] * 3, dtype=pd.Int32Dtype()),
                "",
            ),
            pd.Series(
                [
                    "123456789",
                    "12456789",
                    "1234789",
                    "123456",
                    "123456789",
                    None,
                    "3456789",
                    "126789",
                    "123456789",
                    "12345689",
                    "12345678",
                    None,
                ]
            ),
            id="scalar_vector_vector_scalar_string",
        ),
        pytest.param(
            (
                b"The quick brown fox jumps over the lazy dog",
                11,
                5,
                pd.Series([b"red", b"orange", b"yellow", b"green", b"blue", None]),
            ),
            pd.Series(
                [
                    b"The quick red fox jumps over the lazy dog",
                    b"The quick orange fox jumps over the lazy dog",
                    b"The quick yellow fox jumps over the lazy dog",
                    b"The quick green fox jumps over the lazy dog",
                    b"The quick blue fox jumps over the lazy dog",
                    None,
                ]
            ),
            id="scalar_scalar_scalar_vector_binary",
        ),
    ],
)
def test_insert(args, answer, memory_leak_check):
    def impl(source, pos, len, inject):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.insert(source, pos, len, inject)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda source, pos, len, inject: bodo.libs.bodosql_array_kernels.insert(
            source, pos, len, inject
        )

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["alpha", "beta", "gamma", None, "epsilon"])),
                pd.Series(pd.array(["a", "b", "c", "t", "n"])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "alphabet soup is delicious",
                pd.Series(pd.array([" ", "ici", "x", "i", None])),
            ),
            id="scalar_vector",
        ),
        pytest.param(
            ("The quick brown fox jumps over the lazy dog", "x"),
            id="all_scalar",
        ),
    ],
)
def test_instr(args, memory_leak_check):
    def impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.instr(arr0, arr1))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr0, arr1: bodo.libs.bodosql_array_kernels.instr(arr0, arr1)

    # Simulates INSTR on a single row
    def instr_scalar_fn(elem, target):
        if pd.isna(elem) or pd.isna(target):
            return None
        else:
            return elem.find(target) + 1

    instr_answer = vectorized_sol(args, instr_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=instr_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                pd.Series([1, -4, 3, 14, 5, 0]),
            ),
            id="all_vector_no_null",
        ),
        pytest.param(
            (
                pd.Series(pd.array(["AAAAA", "BBBBB", "CCCCC", None] * 3)),
                pd.Series(pd.array([2, 4, None] * 4)),
            ),
            id="all_vector_some_null",
        ),
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                4,
            ),
            id="vector_string_scalar_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta"]),
                pd.Series([1, -4, 3, 14, 5, 0]),
            ),
            id="binary_vector_no_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta"]),
                4,
            ),
            id="binary_vector_scalar_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([b"alpha", b"beta", None, b"delta", b"epsilon", None] * 4),
                4,
            ),
            id="binary_vector_some_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                pd.Series(pd.array(list(range(-2, 11)))),
            ),
            id="scalar_string_vector_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                6,
            ),
            id="all_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                None,
            ),
            id="scalar_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(gen_nonascii_list(6)),
                None,
            ),
            id="nonascii_vector_null",
        ),
    ],
)
def test_left_right(args, memory_leak_check):
    def impl1(arr, n_chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.left(arr, n_chars))

    def impl2(arr, n_chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.right(arr, n_chars))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl1 = lambda arr, n_chars: bodo.libs.bodosql_array_kernels.left(arr, n_chars)
        impl2 = lambda arr, n_chars: bodo.libs.bodosql_array_kernels.right(arr, n_chars)

    # Simulates LEFT on a single row
    def left_scalar_fn(elem, n_chars):
        arr_is_string = is_valid_string_arg(elem)
        empty_char = "" if arr_is_string else b""
        if pd.isna(elem) or pd.isna(n_chars):
            return None
        elif n_chars <= 0:
            return empty_char
        else:
            return elem[:n_chars]

    # Simulates RIGHT on a single row
    def right_scalar_fn(elem, n_chars):
        arr_is_string = is_valid_string_arg(elem)
        empty_char = "" if arr_is_string else b""
        if pd.isna(elem) or pd.isna(n_chars):
            return None
        elif n_chars <= 0:
            return empty_char
        else:
            return elem[-n_chars:]

    arr, n_chars = args
    left_answer = vectorized_sol((arr, n_chars), left_scalar_fn, object)
    right_answer = vectorized_sol((arr, n_chars), right_scalar_fn, object)
    check_func(
        impl1,
        (arr, n_chars),
        py_output=left_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        (arr, n_chars),
        py_output=right_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.array(["alpha", "beta", "gamma", "delta", "epsilon"]),
                pd.array([2, 4, 8, 16, 32]),
                pd.array(["_", "_", "_", "AB", "123"]),
            ),
            id="all_vector_no_null",
        ),
        pytest.param(
            (
                pd.array([None, "words", "words", "words", "words", "words"]),
                pd.array([16, None, 16, 0, -5, 16]),
                pd.array(["_", "_", None, "_", "_", ""]),
            ),
            id="all_vector_with_null",
        ),
        pytest.param(
            (
                np.array(
                    [b"", b"abc", b"c", b"ccdefg", b"abcde", b"poiu", b"abc"], object
                ),
                20,
                b"_",
            ),
            id="binary_vector",
        ),
        pytest.param(
            (
                pd.Series([b"", b"abc", b"c", b"ccdefg", b"abcde", b"poiu", None]),
                20,
                b"abc",
            ),
            marks=pytest.mark.slow,
            id="binary_vector_with_null",
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, "_"),
            id="vector_scalar_scalar_A",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 0, "_"),
            id="vector_scalar_scalar_B",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, ""),
            id="vector_sscalar_scalar_C",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), None, "_"),
            id="vector_null_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, None),
            id="vector_scalar_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("words", 20, "0123456789"), id="all_scalar_no_null", marks=pytest.mark.slow
        ),
        pytest.param(
            (None, 20, "0123456789"), id="all_scalar_with_null", marks=pytest.mark.slow
        ),
        pytest.param(
            ("words", pd.array([2, 4, 8, 16, 32]), "0123456789"),
            id="scalar_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, 20, pd.array(["A", "B", "C", "D", "E"])),
            id="null_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "words",
                30,
                pd.array(["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "", None]),
            ),
            id="scalar_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "words",
                pd.array([-10, 0, 10, 20, 30]),
                pd.array([" ", " ", " ", "", None]),
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((None, None, None), id="all_null", marks=pytest.mark.slow),
        pytest.param(
            (
                pd.array(["A", "B", "C", "D", "E"]),
                pd.Series([2, 4, 6, 8, 10]),
                pd.Series(["_"] * 5),
            ),
            id="series_test",
        ),
    ],
)
def test_lpad_rpad(args, memory_leak_check):
    def impl1(arr, length, lpad_string):
        return pd.Series(bodo.libs.bodosql_array_kernels.lpad(arr, length, lpad_string))

    def impl2(arr, length, rpad_string):
        return pd.Series(bodo.libs.bodosql_array_kernels.rpad(arr, length, rpad_string))

    # avoid Series conversion for scalar output
    if all(
        not isinstance(arg, (pd.Series, pd.core.arrays.ExtensionArray, np.ndarray))
        for arg in args
    ):
        impl1 = lambda arr, length, lpad_string: bodo.libs.bodosql_array_kernels.lpad(
            arr, length, lpad_string
        )
        impl2 = lambda arr, length, rpad_string: bodo.libs.bodosql_array_kernels.rpad(
            arr, length, rpad_string
        )

    # Simulates LPAD on a single element
    def lpad_scalar_fn(elem, length, pad):
        if pd.isna(elem) or pd.isna(length) or pd.isna(pad):
            return None
        elif pad == "":
            return elem
        elif length <= 0:
            return ""
        elif len(elem) > length:
            return elem[:length]
        else:
            return (pad * length)[: length - len(elem)] + elem

    # Simulates RPAD on a single element
    def rpad_scalar_fn(elem, length, pad):
        if pd.isna(elem) or pd.isna(length) or pd.isna(pad):
            return None
        elif pad == "":
            return elem
        elif length <= 0:
            return ""
        elif len(elem) > length:
            return elem[:length]
        else:
            return elem + (pad * length)[: length - len(elem)]

    arr, length, pad_string = args
    lpad_answer = vectorized_sol((arr, length, pad_string), lpad_scalar_fn, object)
    rpad_answer = vectorized_sol((arr, length, pad_string), rpad_scalar_fn, object)
    check_func(
        impl1,
        (arr, length, pad_string),
        py_output=lpad_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        (arr, length, pad_string),
        py_output=rpad_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "s",
    [
        pytest.param(
            pd.Series(pd.array(["alphabet", "ɲɳ", "Ʃ=sigma", "", " yay "])),
            id="vector",
        ),
        pytest.param(
            "Apple",
            id="scalar",
        ),
    ],
)
def test_ord_ascii(s, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.ord_ascii(arr))

    # avoid Series conversion for scalar output
    if not isinstance(s, pd.Series):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.ord_ascii(arr)

    # Simulates ORD/ASCII on a single row
    def ord_ascii_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elif len(elem) == 0:
            return 0
        else:
            return ord(elem[0])

    ord_answer = vectorized_sol((s,), ord_ascii_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        (s,),
        py_output=ord_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            (
                pd.Series(["a", None, "l", "a", "D", "e"] * 3),
                pd.Series(["alpha", "beta", None, "gamma", "delta", "epsilon"] * 3),
                pd.Series([1, 1, 1, 3, 1, None] * 3, dtype=pd.Int32Dtype()),
            ),
            pd.Series([1, None, None, 5, 0, None] * 3, dtype=pd.Int32Dtype()),
            id="all_vector_string",
        ),
        pytest.param(
            (
                pd.Series([b"ab", None, b"is", b"vo", b"i", b"", b"!"]),
                pd.Series(
                    [b"alphabet", b"soup", b"is", b"very", b"delicious", None, b"yum!"]
                ),
                pd.Series([5, None, 1, 0, 6, 1, 4], dtype=pd.Int32Dtype()),
            ),
            pd.Series([5, None, 1, 0, 6, None, 4], dtype=pd.Int32Dtype()),
            id="all_vector_binary",
        ),
        pytest.param(
            (
                " ",
                "alphabet soup is very delicious",
                pd.Series([1, 5, 10, None, 15, 20, 25], dtype=pd.Int32Dtype()),
            ),
            pd.Series([9, 9, 14, None, 17, 22, 0], dtype=pd.Int32Dtype()),
            id="scalar_scalar_vector_string",
        ),
        pytest.param(
            (
                pd.Series([b" ", b"so", b"very", None, b"!"]),
                b"alphabet soup is so very very delicious!",
                1,
            ),
            pd.Series([9, 10, 21, None, 40], dtype=pd.Int32Dtype()),
            id="vector_scalar_scalar_binary",
        ),
        pytest.param(("a", "darkness and light", 5), 10, id="all_scalar_string"),
        pytest.param(
            (b"i", b"rainbow", 1),
            3,
            id="all_scalar_binary",
        ),
    ],
)
def test_position(args, answer, memory_leak_check):
    def impl(substr, source, start):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.position(substr, source, start)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda substr, source, start: bodo.libs.bodosql_array_kernels.position(
            substr, source, start
        )

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["A", "BCD", "EFGH🐍", None, "I", "J"])),
                pd.Series(pd.array([2, 6, -1, 3, None, 3])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (pd.Series(pd.array(["", "A✓", "BC", "DEF", "GHIJ", None])), 10),
            id="vector_scalar",
        ),
        pytest.param(
            ("Ʃ = alphabet", pd.Series(pd.array([-5, 0, 1, 5, 2]))),
            id="scalar_vector",
        ),
        pytest.param(("racecars!", 4), id="all_scalar_no_null"),
        pytest.param((None, None), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_repeat(args, memory_leak_check):
    def impl(arr, repeats):
        return pd.Series(bodo.libs.bodosql_array_kernels.repeat(arr, repeats))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, repeats: bodo.libs.bodosql_array_kernels.repeat(arr, repeats)

    # Simulates REPEAT on a single row
    def repeat_scalar_fn(elem, repeats):
        if pd.isna(elem) or pd.isna(repeats):
            return None
        else:
            return elem * repeats

    strings_binary, numbers = args
    repeat_answer = vectorized_sol((strings_binary, numbers), repeat_scalar_fn, object)
    check_func(
        impl,
        (strings_binary, numbers),
        py_output=repeat_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["alphabet", "s🟦o🟦u🟦p", "is", "delicious", None])),
                pd.Series(pd.array(["a", "", "4", "ic", " "])),
                pd.Series(pd.array(["_", "X", "5", "", "."])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "i'd like to buy",
                            "the world a coke",
                            "and",
                            None,
                            "keep it company",
                        ]
                    )
                ),
                pd.Series(pd.array(["i", " ", "", "$", None])),
                "🟩",
            ),
            id="vector_vector_scalar",
        ),
        pytest.param(
            (
                pd.Series(pd.array(["oohlala", "books", "oooo", "ooo", "ooohooooh"])),
                "oo",
                pd.Series(pd.array(["", "OO", "*", "#O#", "!"])),
            ),
            id="vector_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "♪♪♪ I'd like to teach the world to sing ♫♫♫",
                " ",
                pd.Series(pd.array(["_", "  ", "", ".", None])),
            ),
            id="scalar_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is so very delicious", "so", "SO"), id="all_scalar_no_null"
        ),
        pytest.param(
            ("Alpha", None, "Beta"), id="all_scalar_with_null", marks=pytest.mark.slow
        ),
    ],
)
def test_replace(args, memory_leak_check):
    def impl(arr, to_replace, replace_with):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.replace(arr, to_replace, replace_with)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, to_replace, replace_with: bodo.libs.bodosql_array_kernels.replace(
            arr, to_replace, replace_with
        )

    # Simulates REPLACE on a single row
    def replace_scalar_fn(elem, to_replace, replace_with):
        if pd.isna(elem) or pd.isna(to_replace) or pd.isna(replace_with):
            return None
        elif to_replace == "":
            return elem
        else:
            return elem.replace(to_replace, replace_with)

    replace_answer = vectorized_sol(args, replace_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        args,
        py_output=replace_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "strings_binary",
    [
        pytest.param(
            pd.Series(pd.array(["A", "BƬCD", "EFGH", None, "I", "J✖"])),
            id="vector",
        ),
        pytest.param("racecarsƟ", id="scalar"),
        pytest.param(
            pd.Series(pd.array(gen_nonascii_list(6))),
            id="nonascii_vector",
        ),
        pytest.param(gen_nonascii_list(1)[0], id="scalar"),
        pytest.param(
            pd.Series([b"abcdef", b"12345", b"AAAA", b"zzzzz", b"z", b"1"]),
            id="binary_vector",
        ),
        pytest.param(
            np.array([b"a", b"abc", b"c", b"ccdefg", b"abcde", b"poiu", None], object),
            id="binary_vector_null",
        ),
    ],
)
def test_reverse(strings_binary, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.reverse(arr))

    # avoid Series conversion for scalar output
    if not isinstance(strings_binary, (pd.Series, np.ndarray)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.reverse(arr)

    # Simulates REVERSE on a single row
    def reverse_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem[::-1]

    reverse_answer = vectorized_sol((strings_binary,), reverse_scalar_fn, object)
    check_func(
        impl,
        (strings_binary,),
        py_output=reverse_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "strings, answer",
    [
        pytest.param(
            pd.Series(
                ["   ", None, "dog", "", None, "alphabet soup   ", "   alphabet soup"]
            ),
            pd.Series([0, None, 3, 0, None, 13, 16], dtype=pd.Int32Dtype()),
            id="vector",
        ),
        pytest.param(" The quick fox jumped over the lazy dog. \n   ", 42, id="scalar"),
    ],
)
def test_rtrimmed_length(strings, answer, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.rtrimmed_length(arr))

    # avoid Series conversion for scalar output
    if not isinstance(strings, (pd.Series, np.ndarray)):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.rtrimmed_length(arr)

    check_func(
        impl,
        (strings,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param(
            pd.Series(pd.array([2, 6, -1, 3, None, 3])),
            id="vector",
        ),
        pytest.param(
            4,
            id="scalar",
        ),
    ],
)
def test_space(numbers, memory_leak_check):
    def impl(n_chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.space(n_chars))

    # avoid Series conversion for scalar output
    if not isinstance(numbers, pd.Series):
        impl = lambda n_chars: bodo.libs.bodosql_array_kernels.space(n_chars)

    # Simulates SPACE on a single row
    def space_scalar_fn(n_chars):
        if pd.isna(n_chars):
            return None
        else:
            return " " * n_chars

    space_answer = vectorized_sol((numbers,), space_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (numbers,),
        py_output=space_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pd.Series(
                        pd.array(
                            ["alpha beta gamma"] * 4
                            + ["oh what a beautiful morning"] * 4
                            + ["aaeaaeieaaeioiea"] * 4
                        )
                    ),
                    pd.Series(pd.array([" ", " ", "a", "a"] * 3)),
                    pd.Series(pd.array([1, 3] * 6)),
                ),
                pd.Series(
                    pd.array(
                        [
                            "alpha",
                            "gamma",
                            "",
                            " bet",
                            "oh",
                            "a",
                            "oh wh",
                            " be",
                            "aaeaaeieaaeioiea",
                            "",
                            "",
                            "e",
                        ]
                    )
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                (
                    "oh what a beautiful morning",
                    " ",
                    pd.Series(
                        pd.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, None])
                    ),
                ),
                pd.Series(
                    pd.array(
                        [
                            "",
                            "oh",
                            "what",
                            "a",
                            "beautiful",
                            "morning",
                            "oh",
                            "oh",
                            "what",
                            "a",
                            "beautiful",
                            "morning",
                            "",
                            None,
                        ]
                    )
                ),
            ),
            id="scalar_scalar_vector",
        ),
        pytest.param(
            (
                (
                    "alphabet soup is delicious",
                    pd.Series(pd.array(["", " ", "ou", " is ", "yay"] * 2)),
                    pd.Series([1] * 5 + [2] * 5),
                ),
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is delicious",
                            "alphabet",
                            "alphabet s",
                            "alphabet soup",
                            "alphabet soup is delicious",
                            "",
                            "soup",
                            "p is delici",
                            "delicious",
                            "",
                        ]
                    )
                ),
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (("255.136.64.224", ".", 3), "64"), id="all_scalar", marks=pytest.mark.slow
        ),
    ],
)
def test_split_part(args, memory_leak_check):
    def impl(source, delim, target):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.split_part(source, delim, target)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda source, delim, target: bodo.libs.bodosql_array_kernels.split_part(
            source, delim, target
        )

    args, answer = args
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        ["alphabet soup is delicious", "", "alpha", "beta", "usa", None]
                        * 6
                    )
                ),
                pd.Series(pd.array(["alphabet", "a", "", "bet", "us", None])).repeat(6),
            ),
            id="all_vector_string",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            b"12345",
                            b"",
                            b"123",
                            b"345",
                            b"35",
                            b"54321",
                            b"45123",
                            b" ",
                            b"1",
                            None,
                        ]
                        * 10
                    )
                ),
                pd.Series(
                    pd.array(
                        [
                            b"1",
                            b"12",
                            b"123",
                            b"1234",
                            b"12345",
                            b"2345",
                            b"345",
                            b"45",
                            b"5",
                            None,
                        ]
                    )
                ).repeat(10),
            ),
            id="all_vector_binary",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "the quick fox",
                            "The party",
                            "dropped the ball",
                            None,
                            "the hero",
                            "make the",
                        ]
                    )
                ),
                "the",
            ),
            id="vector_scalar_string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                b"the quick fox",
                pd.Series(
                    pd.array(
                        [b"the quick fox", b"the", b" ", b"The", None, b"xof", b"quick"]
                    )
                ),
            ),
            id="scalar_vector_binary",
            marks=pytest.mark.slow,
        ),
        pytest.param(("12-345", "1"), id="all_scalar_good_string"),
        pytest.param(
            ("12-345", "45"), id="all_scalar_bad_string", marks=pytest.mark.slow
        ),
        pytest.param((b"bookshelf", b"books"), id="all_binary_good_string"),
        pytest.param(
            (b"book", b"books"), id="all_binary_bad_string", marks=pytest.mark.slow
        ),
    ],
)
@pytest.mark.parametrize("startswith", [True, False])
def test_startswith_endswith(args, startswith, memory_leak_check):
    def startswith_impl(source, prefix):
        return pd.Series(bodo.libs.bodosql_array_kernels.startswith(source, prefix))

    # Avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        startswith_impl = (
            lambda source, prefix: bodo.libs.bodosql_array_kernels.startswith(
                source, prefix
            )
        )

    def endswith_impl(source, prefix):
        return pd.Series(bodo.libs.bodosql_array_kernels.endswith(source, prefix))

    # Avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        endswith_impl = lambda source, prefix: bodo.libs.bodosql_array_kernels.endswith(
            source, prefix
        )

    # Simulates STARTSWITH on a single row
    def startswith_scalar_fn(source, prefix):
        if pd.isna(source) or pd.isna(prefix):
            return None
        else:
            return source.startswith(prefix)

    # Simulates ENDSWITH on a single row
    def endswith_scalar_fn(source, prefix):
        if pd.isna(source) or pd.isna(prefix):
            return None
        else:
            return source.endswith(prefix)

    if startswith:
        impl = startswith_impl
        scalar_fn = startswith_scalar_fn
    else:
        impl = endswith_impl
        scalar_fn = endswith_scalar_fn

    answer = vectorized_sol(args, scalar_fn, pd.BooleanDtype())
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["ABC", "25", "X", None, "A"])),
                pd.Series(pd.array(["abc", "123", "X", "B", None])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (pd.Series(pd.array(["ABC", "ACB", "ABZ", "AZB", "ACE", "ACX"])), "ACE"),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(("alphabet", "soup"), id="all_scalar"),
    ],
)
def test_strcmp(args, memory_leak_check):
    def impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.strcmp(arr0, arr1))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr0, arr1: bodo.libs.bodosql_array_kernels.strcmp(arr0, arr1)

    # Simulates STRCMP on a single row
    def strcmp_scalar_fn(arr0, arr1):
        if pd.isna(arr0) or pd.isna(arr1):
            return None
        else:
            return -1 if arr0 < arr1 else (1 if arr0 > arr1 else 0)

    strcmp_answer = vectorized_sol(args, strcmp_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=strcmp_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [""] * 4 + ["110,112,122,150\n210,213,251\n451"] * 4 + [None]
                    )
                ),
                pd.Series(pd.array(["", "", ",\n ", ",\n "] * 2 + ["a"])),
                pd.Series(pd.array([1, 5] * 4 + [2])),
            ),
            pd.Series(
                pd.array(
                    [
                        None,
                        None,
                        None,
                        None,
                        "110,112,122,150\n210,213,251\n451",
                        None,
                        "110",
                        "210",
                        None,
                    ]
                )
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "Odysseus,Achilles,Diomedes,Ajax,Agamemnon\nParis,Hector,Helen,Aeneas",
                "\n,",
                pd.Series(pd.array(list(range(-2, 11)))),
            ),
            pd.Series(
                pd.array(
                    [
                        None,
                        None,
                        None,
                        "Odysseus",
                        "Achilles",
                        "Diomedes",
                        "Ajax",
                        "Agamemnon",
                        "Paris",
                        "Hector",
                        "Helen",
                        "Aeneas",
                        None,
                    ]
                )
            ),
            id="scalar_scalar_vector",
        ),
        pytest.param(
            (
                "The quick brown fox jumps over the lazy dog",
                pd.Series(pd.array(["aeiou"] * 5 + [" "] * 5)),
                pd.Series([1, 2, 4, 8, 16] * 2),
            ),
            pd.Series(
                pd.array(
                    [
                        "Th",
                        " q",
                        "wn f",
                        "r th",
                        None,
                        "The",
                        "quick",
                        "fox",
                        "lazy",
                        None,
                    ]
                )
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("aaeaaeieaaeioiea", "a", 3),
            "eioie",
            id="all_scalar",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_strtok(args, answer, memory_leak_check):
    def impl(source, delim, target):
        return pd.Series(bodo.libs.bodosql_array_kernels.strtok(source, delim, target))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda source, delim, target: bodo.libs.bodosql_array_kernels.strtok(
            source, delim, target
        )

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(
            (
                "Odysseus,Achilles,Diomedes,Ajax,Agamemnon\nParis,Hector,Helen,Aeneas",
                pd.Series(["\n,", "", None]),
            ),
            pd.Series(
                [
                    [
                        "Odysseus",
                        "Achilles",
                        "Diomedes",
                        "Ajax",
                        "Agamemnon",
                        "Paris",
                        "Hector",
                        "Helen",
                        "Aeneas",
                    ],
                    [
                        "Odysseus,Achilles,Diomedes,Ajax,Agamemnon\nParis,Hector,Helen,Aeneas"
                    ],
                    None,
                ],
            ),
            id="vector_vector",
        ),
        pytest.param(
            (
                "",
                " ",
            ),
            pd.array([], dtype="string[pyarrow]"),
            id="empty_string",
        ),
    ],
)
def test_strtok_to_array(args, expected, memory_leak_check):
    def impl(source, delim):
        return pd.Series(bodo.libs.bodosql_array_kernels.strtok_to_array(source, delim))

    # avoid Series conversion for scalar output
    if not any(isinstance(arg, pd.Series) for arg in args):
        impl = lambda source, delim: bodo.libs.bodosql_array_kernels.strtok_to_array(
            source, delim
        )

    check_func(
        impl,
        args,
        py_output=expected,
        check_dtype=False,
        reset_index=True,
        dist_test=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is 🟥🟧🟨🟩🟦🟪",
                            "so very very delicious",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array([5, -5, 3, -8, 10, 20, 1])),
                pd.Series(pd.array([10, 5, 12, 4, 2, 5, -1])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo 🟥🟧🟨🟩🟦🟪",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array([0, 1, -2, 4, -8, 16, -32])),
                5,
            ),
            id="scalar_vector_mix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is 🟥🟧🟨🟩🟦🟪 so very delicious", 10, 8),
            id="all_scalar_no_null",
        ),
        pytest.param(
            ("alphabet soup is so very delicious", None, 8),
            id="all_scalar_some_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array([b"a", b"abc", b"cd", b"ccdefg", b"abcde", b"poiu"], object),
                1,
                3,
            ),
            id="binary_vector",
        ),
        pytest.param(
            (pd.Series([b"", b"abc", b"c", b"ccdefg", b"abcde", b"poiu", None]), 10, 8),
            id="binary_vector_with_null",
        ),
    ],
)
def test_substring(args, memory_leak_check):
    def impl(arr, start, length):
        return pd.Series(bodo.libs.bodosql_array_kernels.substring(arr, start, length))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, (pd.Series, np.ndarray)) for arg in args):
        impl = lambda arr, start, length: bodo.libs.bodosql_array_kernels.substring(
            arr, start, length
        )

    # Simulates SUBSTRING on a single row
    def substring_scalar_fn(elem, start, length):
        if pd.isna(elem) or pd.isna(start) or pd.isna(length):
            return None
        elif length <= 0:
            return ""
        elif start < 0 and start + length >= 0:
            return elem[start:]
        else:
            if start > 0:
                start -= 1
            return elem[start : start + length]

    arr, start, length = args
    substring_answer = vectorized_sol((arr, start, length), substring_scalar_fn, object)
    check_func(
        impl,
        (arr, start, length),
        py_output=substring_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_substring_suffix(memory_leak_check):
    """Test substring_suffix kernel"""
    arr = pd.Series(
        pd.array(
            [
                "alphabet soup is 🟥🟧🟨🟩🟦🟪",
                "so very very delicious",
                "aaeaaeieaaeioiea",
                "alpha beta gamma delta epsilon",
                None,
                "foo",
                "bar",
            ]
        )
    )
    start = pd.Series(pd.array([5, -5, 3, -8, 10, 20, 1]))

    def impl(arr, start):
        return pd.Series(bodo.libs.bodosql_array_kernels.substring_suffix(arr, start))

    # Simulates SUBSTRING on a single row
    def substring_scalar_fn(elem, start):
        if pd.isna(elem) or pd.isna(start):
            return None
        elif start > 0:
            start -= 1
        return elem[start:]

    substring_answer = vectorized_sol((arr, start), substring_scalar_fn, object)
    check_func(
        impl,
        (arr, start),
        py_output=substring_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious 🟥🟧🟨🟩🟦🟪",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array(["a", "b", "e", " ", " ", "o", "r"])),
                pd.Series(pd.array([1, 4, 3, 0, 1, -1, None])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious 🟥🟧🟨🟩🟦🟪",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                " ",
                pd.Series(pd.array([1, 2, -1, 4, 5, 1, 0])),
            ),
            id="scalar_vector_mix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is so very delicious", "o", 3),
            id="all_scalar_no_null",
        ),
        pytest.param(
            ("alphabet soup is so very delicious", None, 3),
            id="all_scalar_some_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_substring_index(args, memory_leak_check):
    def impl(arr, delimiter, occurrences):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.substring_index(arr, delimiter, occurrences)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, delimiter, occurrences: bodo.libs.bodosql_array_kernels.substring_index(
            arr, delimiter, occurrences
        )

    # Simulates SUBSTRING_INDEX on a single row
    def substring_index_scalar_fn(elem, delimiter, occurrences):
        if pd.isna(elem) or pd.isna(delimiter) or pd.isna(occurrences):
            return None
        elif delimiter == "" or occurrences == 0:
            return ""
        elif occurrences >= 0:
            return delimiter.join(elem.split(delimiter)[:occurrences])
        else:
            return delimiter.join(elem.split(delimiter)[occurrences:])

    arr, delimiter, occurrences = args
    substring_index_answer = vectorized_sol(
        (arr, delimiter, occurrences), substring_index_scalar_fn, pd.StringDtype()
    )
    check_func(
        impl,
        (arr, delimiter, occurrences),
        py_output=substring_index_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param(
            pd.Series(pd.array([2, 6, -1, 3, None, 3])),
            id="vector",
        ),
        pytest.param(
            4,
            id="scalar",
        ),
    ],
)
def test_space(numbers, memory_leak_check):
    def impl(n_chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.space(n_chars))

    # avoid Series conversion for scalar output
    if not isinstance(numbers, pd.Series):
        impl = lambda n_chars: bodo.libs.bodosql_array_kernels.space(n_chars)

    # Simulates SPACE on a single row
    def space_scalar_fn(n_chars):
        if pd.isna(n_chars):
            return None
        else:
            return " " * n_chars

    space_answer = vectorized_sol((numbers,), space_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (numbers,),
        py_output=space_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pd.Series(
                        pd.array(
                            ["alphabet soup is delicious"] * 4
                            + ["Where others saw order, I saw a straitjacket."] * 4
                        )
                    ),
                    pd.Series(
                        pd.array(
                            [
                                " aeiou,.",
                                " aeiou,.",
                                "abcdefghijklmnopqrstuvwxyz",
                                "abcdefghijklmnopqrstuvwxyz",
                            ]
                            * 2
                        )
                    ),
                    pd.Series(pd.array(["_AEIOU", "zebracdfghijklmnopqstuvwxy"] * 4)),
                ),
                pd.Series(
                    pd.array(
                        [
                            "AlphAbEt_sOUp_Is_dElIcIOUs",
                            "elphebbtzsacpzrszdblrcracs",
                            "__AO   IOE",
                            "zjnfzeas qmtn gq rajgbgmtq",
                            "WhErE_OthErs_sAw_OrdEr_I_sAw_A_strAItjAckEt",
                            "WhbrbzathbrszsewzardbrdzIzsewzezstrertjeckbtf",
                            "WOO O _ IO, I _ _ __EO.",
                            "Wfapa msfapq qzv mprap, I qzv z qspzgshzbias.",
                        ]
                    )
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                (
                    "what do we say to the god of death?",
                    " aeioubcdfghjklmnpqrstvwxyz",
                    pd.Series(
                        pd.array(
                            [
                                " aeiou",
                                " aeiou********************",
                                "_AEIOUbcdfghjklmnpqrstvwxyz",
                                "",
                                None,
                            ]
                        )
                    ),
                ),
                pd.Series(
                    pd.array(
                        [
                            "a o e a o e o o ea?",
                            "**a* *o *e *a* *o **e *o* o* *ea**?",
                            "whAt_dO_wE_sAy_tO_thE_gOd_Of_dEAth?",
                            "?",
                            None,
                        ]
                    )
                ),
            ),
            id="scalar_scalar_vector",
        ),
        pytest.param(
            (("WE ATTACK AT DAWN", "ACDEKTNW ", "acdektnw"), "weattackatdawn"),
            id="all_scalar",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_translate(args, memory_leak_check):
    def impl(arr, source, target):
        return pd.Series(bodo.libs.bodosql_array_kernels.translate(arr, source, target))

    args, answer = args

    # avoid Series conversion for scalar output
    if not isinstance(answer, pd.Series):
        impl = lambda arr, source, target: bodo.libs.bodosql_array_kernels.translate(
            arr, source, target
        )

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_char_ord_ascii(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.ord_ascii(arg0),
            bodo.libs.bodosql_array_kernels.char(arg1),
        )

    A, B = "A", 97
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            a0 = 65 if flag0 else None
            a1 = "a" if flag1 else None
            check_func(impl, (A, B, flag0, flag1), py_output=(a0, a1), dist_test=False)


@pytest.mark.slow
def test_option_format(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.format(arg0, arg1)

    A, B = 12345678910.111213, 4
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = "12,345,678,910.1112" if flag0 and flag1 else None
            check_func(impl, (A, B, flag0, flag1), py_output=answer, dist_test=False)


@pytest.mark.slow
def test_option_left_right(memory_leak_check):
    def impl1(scale1, scale2, flag1, flag2):
        arr = scale1 if flag1 else None
        n_chars = scale2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.left(arr, n_chars)

    def impl2(scale1, scale2, flag1, flag2):
        arr = scale1 if flag1 else None
        n_chars = scale2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.right(arr, n_chars)

    scale1, scale2 = "alphabet soup", 10
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1 and flag2:
                answer1 = "alphabet s"
                answer2 = "habet soup"
            else:
                answer1 = None
                answer2 = None
            check_func(
                impl1,
                (scale1, scale2, flag1, flag2),
                py_output=answer1,
                check_dtype=False,
                dist_test=False,
            )
            check_func(
                impl2,
                (scale1, scale2, flag1, flag2),
                py_output=answer2,
                check_dtype=False,
                dist_test=False,
            )


@pytest.mark.slow
def test_option_lpad_rpad(memory_leak_check):
    def impl1(arr, length, lpad_string, flag1, flag2):
        B = length if flag1 else None
        C = lpad_string if flag2 else None
        return bodo.libs.bodosql_array_kernels.lpad(arr, B, C)

    def impl2(val, length, lpad_string, flag1, flag2, flag3):
        A = val if flag1 else None
        B = length if flag2 else None
        C = lpad_string if flag3 else None
        return bodo.libs.bodosql_array_kernels.rpad(A, B, C)

    arr, length, pad_string = pd.array(["A", "B", "C", "D", "E"]), 3, " "
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1 and flag2:
                answer = pd.array(["  A", "  B", "  C", "  D", "  E"])
            else:
                answer = pd.array([None] * 5, dtype=pd.StringDtype())
            check_func(
                impl1,
                (arr, length, pad_string, flag1, flag2),
                py_output=answer,
                check_dtype=False,
                dist_test=False,
            )

    val, length, pad_string = "alpha", 10, "01"
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            for flag3 in [True, False]:
                if flag1 and flag2 and flag3:
                    answer = "alpha01010"
                else:
                    answer = None
                check_func(
                    impl2,
                    (val, length, pad_string, flag1, flag2, flag3),
                    py_output=answer,
                    dist_test=False,
                )


@pytest.mark.slow
def test_option_reverse_repeat_replace_space(memory_leak_check):
    def impl(A, B, C, D, flag0, flag1, flag2, flag3):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        arg3 = D if flag3 else None
        return (
            bodo.libs.bodosql_array_kernels.reverse(arg0),
            bodo.libs.bodosql_array_kernels.replace(arg0, arg1, arg2),
            bodo.libs.bodosql_array_kernels.repeat(arg2, arg3),
            bodo.libs.bodosql_array_kernels.space(arg3),
        )

    A, B, C, D = "alphabet soup", "a", "_", 4
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    a0 = "puos tebahpla" if flag0 else None
                    a1 = "_lph_bet soup" if flag0 and flag1 and flag2 else None
                    a2 = "____" if flag2 and flag3 else None
                    a3 = "    " if flag3 else None
                    check_func(
                        impl,
                        (A, B, C, D, flag0, flag1, flag2, flag3),
                        py_output=(a0, a1, a2, a3),
                        dist_test=False,
                    )


@pytest.mark.slow
def test_option_startswith_endswith_insert_position(memory_leak_check):
    def impl(A, B, C, D, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C
        arg3 = D
        return (
            bodo.libs.bodosql_array_kernels.startswith(arg0, arg1),
            bodo.libs.bodosql_array_kernels.endswith(arg0, arg1),
            bodo.libs.bodosql_array_kernels.insert(arg0, arg2, arg3, arg1),
            bodo.libs.bodosql_array_kernels.position(arg1, arg0, arg2),
        )

    A, B, C, D = "The night is dark and full of terrors.", "terrors.", 14, 100
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                (False, True, "The night is terrors.", 31)
                if flag0 and flag1
                else (None, None, None, None)
            )
            check_func(
                impl,
                (A, B, C, D, flag0, flag1),
                py_output=answer,
                dist_test=False,
            )


@pytest.mark.slow
def test_strcmp_instr_option(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.strcmp(arg0, arg1),
            bodo.libs.bodosql_array_kernels.instr(arg0, arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (1, 0) if flag0 and flag1 else None
            check_func(
                impl, ("a", "Z", flag0, flag1), py_output=answer, dist_test=False
            )


@pytest.mark.slow
def test_option_strtok_split_part(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return (
            bodo.libs.bodosql_array_kernels.split_part(arg0, arg1, arg2),
            bodo.libs.bodosql_array_kernels.strtok(arg0, arg1, arg2),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                answer = ("b", "c") if flag0 and flag1 and flag2 else None
                check_func(
                    impl,
                    ("a  b  c", " ", 3, flag0, flag1, flag2),
                    py_output=answer,
                    dist_test=False,
                )


@pytest.mark.slow
def test_option_strtok_to_array(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.strtok_to_array(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                pd.array(["a", "b", "c"], dtype="string[pyarrow]")
                if flag0 and flag1
                else None
            )
            check_func(
                impl,
                ("a  b  c", " ", flag0, flag1),
                py_output=answer,
                dist_test=False,
            )


@pytest.mark.slow
def test_option_substring(memory_leak_check):
    def impl(A, B, C, D, E, flag0, flag1, flag2, flag3, flag4):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        arg3 = D if flag3 else None
        arg4 = E if flag4 else None
        return (
            bodo.libs.bodosql_array_kernels.substring(arg0, arg1, arg2),
            bodo.libs.bodosql_array_kernels.substring_index(arg0, arg3, arg4),
        )

    A, B, C, D, E = "alpha beta gamma", 7, 4, " ", 1
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    for flag4 in [True, False]:
                        a0 = "beta" if flag0 and flag1 and flag2 else None
                        a1 = "alpha" if flag0 and flag3 and flag4 else None
                        check_func(
                            impl,
                            (A, B, C, D, E, flag0, flag1, flag2, flag3, flag4),
                            py_output=(a0, a1),
                            dist_test=False,
                        )


@pytest.mark.slow
def test_option_translate_initcap(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return (
            bodo.libs.bodosql_array_kernels.initcap(arg0, arg1),
            bodo.libs.bodosql_array_kernels.translate(arg0, arg1, arg2),
        )

    A, B, C = "The night is dark and full of terrors.", " .", "_"
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                a0 = (
                    "The Night Is Dark And Full Of Terrors."
                    if flag0 and flag1
                    else None
                )
                a1 = (
                    "The_night_is_dark_and_full_of_terrors"
                    if flag0 and flag1 and flag2
                    else None
                )
                check_func(
                    impl,
                    (A, B, C, flag0, flag1, flag2),
                    py_output=(a0, a1),
                    dist_test=False,
                )


@pytest.mark.slow
def test_option_string_one_arg_fns(memory_leak_check):
    """
    Tests optional support for length, upper, lower.
    """

    def impl(A, flag):
        arg = A if flag else None
        return (
            bodo.libs.bodosql_array_kernels.length(arg),
            bodo.libs.bodosql_array_kernels.upper(arg),
            bodo.libs.bodosql_array_kernels.lower(arg),
        )

    str_val = "    The night is dark and full of terrors.     "
    for flag in [True, False]:
        a0 = len(str_val) if flag else None
        a1 = str_val.upper() if flag else None
        a2 = str_val.lower() if flag else None
        check_func(
            impl,
            (str_val, flag),
            py_output=(a0, a1, a2),
        )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ("  ABC   ", " "),
            id="scalar_without_optional_characters",
        ),
        pytest.param(
            ("    *-ABC-*-   ", "*-"),
            id="scalar_with_optional_characters",
        ),
        pytest.param(
            (
                pd.Series(["asdfghj", "--++++--", "mnvcxzm", "   lkjdfg   "] * 4),
                pd.Series(["a", "-+", "mn", " "] * 4),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    ["asdfghja", "jhskdjfh", "fdsa", "  a  ", "   abcdefa   "] * 4
                ),
                "a",
            ),
            id="vector_scalar_mix",
        ),
    ],
)
def test_trim_ltrim_rtrim(args, memory_leak_check):
    """
    Tests BodoSQL array kernels for TRIM, LTRIM, RTRIM.
    """
    output_is_scalar = all(not isinstance(arg, pd.Series) for arg in args)
    src, chars = args

    # impl for trim
    def trim_impl(src, chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.trim(src, chars))

    if output_is_scalar:
        trim_impl = lambda src, chars: bodo.libs.bodosql_array_kernels.trim(src, chars)

    # impl for ltrim
    def ltrim_impl(src, chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.ltrim(src, chars))

    if output_is_scalar:
        ltrim_impl = lambda src, chars: bodo.libs.bodosql_array_kernels.ltrim(
            src, chars
        )

    # impl for rtrim
    def rtrim_impl(src, chars):
        return pd.Series(bodo.libs.bodosql_array_kernels.rtrim(src, chars))

    if output_is_scalar:
        rtrim_impl = lambda src, chars: bodo.libs.bodosql_array_kernels.rtrim(
            src, chars
        )

    def trim_scalar_fn(src, chars):
        return src.strip(chars)

    def ltrim_scalar_fn(src, chars):
        return src.lstrip(chars)

    def rtrim_scalar_fn(src, chars):
        return src.rstrip(chars)

    src, chars = args

    impls = (trim_impl, ltrim_impl, rtrim_impl)
    answers = (
        vectorized_sol((src, chars), trim_scalar_fn, pd.StringDtype()),
        vectorized_sol((src, chars), ltrim_scalar_fn, pd.StringDtype()),
        vectorized_sol((src, chars), rtrim_scalar_fn, pd.StringDtype()),
    )

    for impl, answer in zip(impls, answers):
        check_func(
            impl,
            (src, chars),
            py_output=answer,
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "string, separator",
    [
        pytest.param(
            "127.0.0.1",
            ".",
            id="all_scalar",
        ),
        pytest.param(
            pd.Series(["*as*14*xv*", None, "43qvtwe", "*****", "*.*N*D*E"] * 4),
            "*",
            id="vector_scalar",
        ),
        pytest.param(
            pd.Series(["20230203", "", "124as", None, "ababababc"] * 4),
            pd.Series(["20", "abc", None, "!@#%^&", "ba"] * 4),
            id="all_vector",
        ),
        pytest.param(
            "8.8.8.8",
            pd.Series(["8", ".", ".8", "", None] * 4),
            id="scalar_vector",
        ),
        pytest.param(
            "empty separator",
            "",
            id="empty separator",
        ),
    ],
)
def test_split(string, separator, memory_leak_check):
    is_out_distributed = True

    def impl(string, separator):
        return pd.Series(bodo.libs.bodosql_array_kernels.split(string, separator))

    if not isinstance(string, pd.Series) and not isinstance(separator, pd.Series):
        is_out_distributed = False
        impl = lambda string, separator: bodo.libs.bodosql_array_kernels.split(
            string, separator
        )

    def scalar_fn(string, separator):
        if pd.isna(string) or pd.isna(separator):
            return None
        if separator == "":
            return pd.array([string])
        return pd.array(string.split(separator))

    answer = vectorized_sol(
        (string, separator), scalar_fn, pd.ArrowDtype(pa.large_list(pa.large_string()))
    )
    check_func(
        impl,
        (string, separator),
        py_output=answer,
        check_dtype=False,
        is_out_distributed=is_out_distributed,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(0, id="lower"),
        pytest.param(1, id="upper", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "string, answer",
    [
        pytest.param(
            "Snowflake",
            "536e6f77666c616b65",
            id="scalar-ascii",
        ),
        pytest.param(
            "Snake 🐍 Infinity ∞",
            "536e616b6520f09f908d20496e66696e69747920e2889e",
            id="scalar-non_ascii",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(["", None, "\x01\x02\x42\x6c", "A", "@Ѐࠀ䀀"]),
            pd.Series(["", None, "0102426c", "41", "40d080e0a080e48080"]),
            id="vector-mixed",
        ),
    ],
)
def test_hex_encode_decode(string, case, answer, memory_leak_check):
    # Pass in case as a global
    if isinstance(string, pd.Series):

        def impl_enc(string):
            return pd.Series(bodo.libs.bodosql_array_kernels.hex_encode(string, case))

        def impl_dec_str(string):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.hex_decode_string(string, True)
            )

        def impl_dec_bin(string):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.hex_decode_bin(string, False)
            )

        answer = answer.str.lower() if case == 0 else answer.str.upper()

    else:
        impl_enc = lambda string: bodo.libs.bodosql_array_kernels.hex_encode(
            string, case
        )
        impl_dec_str = lambda string: bodo.libs.bodosql_array_kernels.hex_decode_string(
            string, False
        )
        impl_dec_bin = lambda string: bodo.libs.bodosql_array_kernels.hex_decode_binary(
            string, True
        )
        answer = answer.lower() if case == 0 else answer.upper()

    check_func(
        impl_enc,
        (string,),
        py_output=answer,
        check_dtype=False,
    )
    check_func(
        impl_dec_str,
        (answer,),
        py_output=string,
        check_dtype=False,
    )
    # For ascii strings, check again but with the input converted from strings to binary
    if isinstance(string, str) and string.isascii():
        check_func(
            impl_enc,
            (bytes(string, encoding="utf-8"),),
            py_output=answer,
            check_dtype=False,
        )
        check_func(
            impl_dec_bin,
            (answer,),
            py_output=bytes(string, encoding="utf-8"),
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "string, max_line_length, alphabet, answer",
    [
        pytest.param(
            "Snowflake",
            0,
            "+/=",
            "U25vd2ZsYWtl",
            id="scalar-ascii-no_limit-defaults-no_pad",
        ),
        pytest.param(
            "Snowflake",
            2,
            "+/=",
            "U2\n5v\nd2\nZs\nYW\ntl",
            id="scalar-ascii-small_limit-defaults-no_pad",
        ),
        pytest.param(
            "HELLO",
            0,
            "+/=",
            "SEVMTE8=",
            id="scalar-ascii-no_limit-defaults-one_pad",
        ),
        pytest.param(
            "Snowflake",
            0,
            "+/=",
            "U25vd2ZsYWtl",
            id="scalar-ascii-multiline-defaults-no_pad",
        ),
        pytest.param(
            "Snowflake ❄❄❄ Snowman ☃☃☃",
            32,
            "$",
            "U25vd2ZsYWtlIOKdhOKdhOKdhCBTbm93\nbWFuIOKYg$KYg$KYgw==",
            id="scalar-nonascii-multiline-alternative_chars",
        ),
        pytest.param(
            "🐍",
            0,
            "",
            "8J+QjQ==",
            id="scalar-nonascii-single_char",
        ),
        pytest.param(
            pd.Series(
                [
                    "",
                    "S",
                    "Sn",
                    "Sno",
                    "Snow",
                    "Snowf",
                    "Snowfl",
                    "Snowfla",
                    "Snowflak",
                    "Snowflake",
                ]
                * 4
            ),
            3,
            "@?*",
            pd.Series(
                [
                    "",
                    "Uw*\n*",
                    "U24\n*",
                    "U25\nv",
                    "U25\nvdw\n**",
                    "U25\nvd2\nY*",
                    "U25\nvd2\nZs",
                    "U25\nvd2\nZsY\nQ**",
                    "U25\nvd2\nZsY\nWs*",
                    "U25\nvd2\nZsY\nWtl",
                ]
                * 4
            ),
            id="vector-limit-alternative",
        ),
    ],
)
def test_base64_encode_decode(
    string, max_line_length, alphabet, answer, memory_leak_check
):
    # Pass in max_line_length and alphabet as globals
    if isinstance(string, pd.Series):

        def impl_enc(string):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.base64_encode(
                    string, max_line_length, alphabet
                )
            )

        def impl_dec_str(string):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.base64_decode_string(string, alphabet)
            )

        def impl_dec_bin(string):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.base64_decode_binary(string, alphabet)
            )

    else:
        impl_enc = lambda string: bodo.libs.bodosql_array_kernels.base64_encode(
            string, max_line_length, alphabet
        )
        impl_dec_str = (
            lambda string: bodo.libs.bodosql_array_kernels.base64_decode_string(
                string, alphabet
            )
        )
        impl_dec_bin = (
            lambda string: bodo.libs.bodosql_array_kernels.base64_decode_binary(
                string, alphabet
            )
        )

    check_func(
        impl_enc,
        (string,),
        py_output=answer,
        check_dtype=False,
    )

    check_func(
        impl_dec_str,
        (answer,),
        py_output=string,
        check_dtype=False,
    )

    # For ascii strings, check again but with the input converted from strings to binary
    if isinstance(string, str) and string.isascii():
        check_func(
            impl_enc,
            (bytes(string, encoding="utf-8"),),
            py_output=answer,
            check_dtype=False,
        )
        check_func(
            impl_dec_bin,
            (answer,),
            py_output=bytes(string, encoding="utf-8"),
            check_dtype=False,
        )
