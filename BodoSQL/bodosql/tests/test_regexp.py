# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Test correctness of SQL regex functions on BodoSQL
"""

import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def regexp_strings_df():
    return {
        "table1": pd.DataFrame(
            {
                "A": [
                    "She opened up her third bottle of wine of the night.",
                    "The teens wondered what was kept in the red shed on the far edge of the school grounds.",
                    "I like to leave work after my eight-hour tea-break.",
                    "He played the game as if his life depended on it and the truth was that it did.",
                    "The snow-covered path was no help in finding his way out of the back-country.",
                    None,
                ],
                "B": [1, 2, 3, 1, 2, 3],
            }
        )
    }


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGEXP_LIKE(A, '.*The.*') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, False, False, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="simple_pattern-no_flags",
        ),
        pytest.param(
            (
                "SELECT A RLIKE '.*The.*' FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, False, False, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="simple_pattern-no_flags-alias-1",
        ),
        pytest.param(
            (
                "SELECT A REGEXP '.*The.*' FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, False, False, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="simple_pattern-no_flags-alias-2",
        ),
        pytest.param(
            (
                "SELECT A RLIKE '.*\W+o\w*.*' FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, True, True, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="medium_pattern-no_flags-alias-1",
        ),
        pytest.param(
            (
                "SELECT A REGEXP '.*\W+o\w*.*' FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, True, True, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="medium_pattern-no_flags-alias-2",
        ),
        pytest.param(
            (
                "SELECT RLIKE(A, '.*The.*') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, False, False, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="simple_pattern-no_flags-alias3",
        ),
        pytest.param(
            (
                "SELECT REGEXP_LIKE(A, '.*The.*', 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True, True, False, True, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="simple_patterns-ignore_case",
        ),
        pytest.param(
            (
                "SELECT REGEXP_LIKE(A, 'the.*', 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, False, False, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="medium_pattern-ignore_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGEXP_LIKE(A, '.*\W+o\w*.*') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True, True, False, True, True, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="medium_pattern-no_flags",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN REGEXP_LIKE(A, '.*\w+-\w+.*', 'i') THEN 'Y' ELSE 'N' END FROM table1",
                pd.DataFrame({0: pd.Series(["N", "N", "Y", "N", "Y", "N"])}),
            ),
            id="CASE-medium_pattern-ignore_case",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regexp_like(regexp_strings_df, args, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        regexp_strings_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGEXP_COUNT(A, 'The') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0, 1, 0, 0, 1, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="simple_pattern-no_pos-no_flags",
        ),
        pytest.param(
            (
                "SELECT REGEXP_COUNT(A, 'The', 1, 'i') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([1, 4, 0, 2, 2, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="simple_pattern-1-ignore_case",
        ),
        pytest.param(
            (
                "SELECT REGEXP_COUNT(A, '\W+o\w*', 36) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([1, 2, 0, 1, 2, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="medium_pattern-36-no_flags",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN REGEXP_COUNT(A, '\w+-\w+', 1, 'i') > 0 THEN ':)' ELSE ':(' END FROM table1",
                pd.DataFrame({0: pd.Series([":(", ":(", ":)", ":(", ":)", ":("])}),
            ),
            id="CASE-medium_pattern-1-ignore_case",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regexp_count(regexp_strings_df, args, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        regexp_strings_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGEXP_REPLACE(A, 'the ') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened up her third bottle of wine of night.",
                                "The teens wondered what was kept in red shed on far edge of school grounds.",
                                "I like to leave work after my eight-hour tea-break.",
                                "He played game as if his life depended on it and truth was that it did.",
                                "The snow-covered path was no help in finding his way out of back-country.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="simple_pattern-no_replace-no_pos-no_occur-no_flags",
        ),
        pytest.param(
            (
                "SELECT REGEXP_REPLACE(A, 'the ', 'THE ', 1, 0, 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened up her third bottle of wine of THE night.",
                                "THE teens wondered what was kept in THE red shed on THE far edge of THE school grounds.",
                                "I like to leave work after my eight-hour tea-break.",
                                "He played THE game as if his life depended on it and THE truth was that it did.",
                                "THE snow-covered path was no help in finding his way out of THE back-country.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="simple_pattern-simple_replace-1-0-ignore_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGEXP_REPLACE(A, 'the \w+', 'the ???', 1, 2, 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened up her third bottle of wine of the night.",
                                "The teens wondered what was kept in the ??? shed on the far edge of the school grounds.",
                                "I like to leave work after my eight-hour tea-break.",
                                "He played the game as if his life depended on it and the ??? was that it did.",
                                "The snow-covered path was no help in finding his way out of the ???-country.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="medium_pattern-simple_replace-1-2-ignore_case",
        ),
        pytest.param(
            (
                "SELECT REGEXP_REPLACE(A, 'the \w+', 'the ***', 5, 0, 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened up her third bottle of wine of the ***.",
                                "The teens wondered what was kept in the *** shed on the *** edge of the *** grounds.",
                                "I like to leave work after my eight-hour tea-break.",
                                "He played the *** as if his life depended on it and the *** was that it did.",
                                "The snow-covered path was no help in finding his way out of the ***-country.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="medium_pattern-simple_replace-5-0-ignore_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGEXP_REPLACE(A, 'the (\w+)', 'the (\\\\1)', 1, 1, 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened up her third bottle of wine of the (night).",
                                "the (teens) wondered what was kept in the red shed on the far edge of the school grounds.",
                                "I like to leave work after my eight-hour tea-break.",
                                "He played the (game) as if his life depended on it and the truth was that it did.",
                                "the (snow)-covered path was no help in finding his way out of the back-country.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="one_group-medium_replace-1-1-ignore_case",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, '-') > 0 THEN REGEXP_REPLACE(A, '\W+([[:alpha:]]+)-([[:alpha:]]+)', ' \\\\\\\\2-\\\\\\\\1') ELSE REGEXP_REPLACE(A, '(\W+)(\w+) (\w+) (\w+)', '\\\\\\\\1[\\\\\\\\4,\\\\\\\\3,\\\\\\\\2]', 6) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "She opened [third,her,up] [wine,of,bottle] [night,the,of].",
                                "The teens [was,what,wondered] [the,in,kept] [on,shed,red] [edge,far,the] [school,the,of] grounds.",
                                "I like to leave work after my hour-eight break-tea.",
                                "He played [as,game,the] [life,his,if] [it,on,depended] [truth,the,and] [it,that,was] did.",
                                "The covered-snow path was no help in finding his way out of the country-back.",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="CASE-multiple_groups-medium_replace-06-00-no_flags",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regexp_replace(regexp_strings_df, args, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        regexp_strings_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, 'the \w+', 1, 2) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([None, "the far", None, "the truth", None, None])}
                ),
            ),
            id="simple_pattern-1-2-no_flags-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, 'the \w+', 1, 1, 'i') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "the night",
                                "The teens",
                                None,
                                "the game",
                                "The snow",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="simple_pattern-1-1-ignore_case-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, '(\w*-\w*)|(the .* the)') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                None,
                                "the red shed on the far edge of the",
                                "eight-hour",
                                "the game as if his life depended on it and the",
                                "snow-covered",
                                None,
                            ],
                        )
                    }
                ),
            ),
            id="medium_pattern-no_pos-no_occur-no_flags-no_group",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, 'the (\w+)', 1, 1, 'ie') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            ["night", "teens", None, "game", "snow", None],
                        )
                    }
                ),
            ),
            id="simple_pattern-1-1-ignore_case_extract-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, '(\w+)-(\w+)', 1, 2, '', 2) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [None, None, "break", None, "country", None],
                        )
                    }
                ),
            ),
            id="medium_pattern-1-2-no_flags-2",
        ),
        pytest.param(
            (
                "SELECT REGEXP_SUBSTR(A, '\W+(\w+) (\w+) (\w+)', 6, 1, 'e', B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            ["up", "what", "work", "the", "path", None],
                        )
                    }
                ),
            ),
            id="medium_pattern-6-1-extract-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, 'of') = 0 THEN REGEXP_SUBSTR(A, '(\w+)\W+(\w+)\.', 31, 1, 'e', 2) ELSE REGEXP_SUBSTR(A, '(\w+) of (\w+)', 31, 1, 'e', 1) END FROM table1",
                pd.DataFrame(
                    {0: pd.Series(["wine", "edge", "break", "did", "out", None])}
                ),
            ),
            id="CASE-medium_pattern-31-1-extract-12",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regexp_substr(regexp_strings_df, args, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        regexp_strings_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, 'the \w+', 1, 2) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0, 53, 0, 54, 0, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="simple_pattern-1-2-no_option-no_flags-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, 'the \w+', 1, 1, 0, 'i') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([43, 1, 0, 11, 1, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="simple_pattern-1-1-0-ignore_case-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, '(\w*-\w*)|(the .* the)') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0, 37, 31, 11, 5, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="medium_pattern-no_pos-no_occur-no_option-no_flags-no_group",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, 'the (\w+)', 1, 1, 0, 'ie') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([47, 5, 0, 15, 5, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="simple_pattern-1-1-0-ignore_case_extract-no_group",
        ),
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, '(\w+)-(\w+)', 1, 2, 1, '', 2) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0, 0, 51, 0, 77, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="medium_pattern-1-2-1-no_flags-2",
        ),
        pytest.param(
            (
                "SELECT REGEXP_INSTR(A, '\W+(\w+) (\w+) (\w+)', 6, 1, 0, 'e', B) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([12, 20, 17, 11, 18, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="medium_pattern-6-1-0-extract-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, 'of') = 0 THEN REGEXP_INSTR(A, '(\w+)\W+(\w+)\.', 31, 1, 0, 'e', 2) ELSE REGEXP_INSTR(A, '(\w+) of (\w+)', 31, 1, 1, 'e', 1) END FROM table1",
                pd.DataFrame(
                    {0: pd.Series([39, 65, 46, 76, 57, None], dtype=pd.Int32Dtype())}
                ),
            ),
            id="CASE-medium_pattern-31-1-01-extract-12",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regexp_instr(regexp_strings_df, args, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        regexp_strings_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )
