import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import gen_nonascii_list


@pytest.fixture
def bodosql_string_fn_testing_df():
    """fixture used for testing string functions that have a variety of different inputs for each argument"""
    data = {
        "POSITIVE_INTS": pd.Series([0, 1, 2, 3, 4, 5, 6, 7] * 2, dtype=np.int8),
        "MIXED_INTS": pd.Series([0, -7, 8, -9, 10, -11, 12, 13] * 2),
        "MIXED_FLOATS": pd.Series(
            [
                0.0,
                0.01232,
                -0.12,
                123.21,
                -12345.0,
                1234567890.123456,
                0.0980000002,
                1.23,
            ]
            * 2
        ),
        "STRINGS": pd.Series(
            [
                "\n\t     hello world     \n\t",
                "h e l l o w o r l d",
                "",
                "e",
                "l",
                "o",
                ' " hello "." \\ world " ',
                '"',
                "\\ . \" ' ",
                "'",
                "h.e.l.l.o.w.o.r.l.d",
                ".",
                'h"e"l"l"o"w"o"r"l"d',
                "\\",
                " ",
                "\t HELLO WORLD\t ",
            ],
        ),
        "STRINGS_NULL_1": pd.Series(
            [
                "alpha",
                "beta",
                None,
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                None,
                None,
                "lambda",
                "mu",
                "nu",
                "xi",
                "omicron",
                None,
            ]
        ),
        "STRINGS_NULL_2": pd.Series(
            [
                " ",
                " ",
                " ",
                "_",
                "_",
                "_",
                "AB",
                "",
                "AB",
                "12345",
                "12345",
                "12345",
                None,
                None,
                None,
                None,
            ]
        ),
        "STRINGS_NONASCII_1": pd.Series(gen_nonascii_list(16)),
        "MIXED_INTS_NULL": pd.Series(
            pd.array(
                [
                    4,
                    10,
                    5,
                    -1,
                    20,
                    32,
                    None,
                    10,
                    None,
                    5,
                    21,
                    22,
                    23,
                    None,
                    25,
                    None,
                ],
                dtype=pd.Int32Dtype(),
            )
        ),
    }
    return {"TABLE1": pd.DataFrame(data)}


BODOSQL_TO_PYSPARK_FN_MAP = {
    "ORD": "ASCII",
    "INSTR": "LOCATE",
    "FORMAT": "FORMAT_NUMBER",
}


@pytest.fixture(
    params=[
        ("CONCAT", ["strings", "strings"], ("'A'", "'B'")),
        pytest.param(
            (
                "CONCAT",
                ["strings", "strings", "strings", "strings"],
                ("'A'", "'B'"),
            ),
            marks=pytest.mark.slow,
        ),
    ]
    +
    # string functions that take one string arg and return a string
    [
        pytest.param((x, ["strings"], ("'A'", "'B'")), marks=pytest.mark.slow)
        for x in [
            "LCASE",
            "UCASE",
            "LOWER",
            "UPPER",
        ]
    ]
    +
    # string functions that take one string arg, and return a number
    [
        (x, ["strings"], ("1", "2"))
        for x in ["CHARACTER_LENGTH", "CHAR_LENGTH", "LENGTH"]
    ]
)
def string_fn_info(request):
    """fixture that returns information used to test string functions
    First argument is function name, second is an equivalent spark function name,
    the third is a list of arguments to use with the function
    The fourth argument is tuple of two possible return values for the function, which
    are used while checking scalar cases
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param("'h%o'", marks=pytest.mark.slow),
        "'%el%'",
        pytest.param("'%'", marks=pytest.mark.slow),
        "'h____'",
        pytest.param("''", marks=pytest.mark.slow),
    ]
)
def regex_string(request):
    """fixture that returns a variety of regex strings to be used for like testing"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("('h%o %olo')", marks=pytest.mark.slow),
        pytest.param("('%Jo%oe%','T%e')", marks=pytest.mark.slow),
        pytest.param("('%Jo%oe%','J%e')", marks=pytest.mark.slow),
        pytest.param("('%Jo%oe%','J%n')", marks=pytest.mark.slow),
        pytest.param("('%J%h%^_do%', 'J%^%wn')", marks=pytest.mark.slow),
        pytest.param("('%Jo%oe%','T%')", marks=pytest.mark.slow),
        pytest.param("('%jo%oe%','j%E')", marks=pytest.mark.slow),
        pytest.param("('%JO%oE%','J%N')", marks=pytest.mark.slow),
        pytest.param("('%J%h%^_dO%', 'J%^%Wn')", marks=pytest.mark.slow),
    ]
)
def regex_strings(request):
    """fixture that returns tuples of regex strings to be used for like any/all testing"""
    return request.param


@pytest.fixture(params=[pytest.param("like", marks=pytest.mark.slow), "not like"])
def like_expression(request):
    """returns 'like' or 'not like'"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("like any", marks=pytest.mark.slow),
        pytest.param("like all", marks=pytest.mark.slow),
    ]
)
def like_any_all_expression(request):
    """returns 'like any' or 'like all'"""
    return request.param


@pytest.fixture(
    params=[
        ".*",
        pytest.param("^ello", marks=pytest.mark.slow),
        pytest.param("^^.*", marks=pytest.mark.slow),
    ]
)
def pythonic_regex(request):
    """fixture that returns a variety of pythonic regex strings to be used for like testing
    currently causing problems, see BS-109"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("f l a m i n g o", marks=pytest.mark.slow),
        '"e"-"l"-"l"-"o"',
        pytest.param(
            "__hippopoto__monstroses__quipped__aliophobia__", marks=pytest.mark.slow
        ),
        pytest.param("", marks=pytest.mark.slow),
    ]
)
def string_constants(request):
    """fixture that returns a variety of string constants to be used for like testing"""
    return request.param
