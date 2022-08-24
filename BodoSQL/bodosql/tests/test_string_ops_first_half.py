# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL string operation queries on BodoSQL
"""

import pytest
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query


def test_like(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    tests that like works for a variety of different possible regex strings
    """
    check_query(
        f"select A from table1 where A {like_expression} {regex_string}",
        bodosql_string_types,
        spark_info,
    )


@pytest.mark.slow
def test_like_scalar(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    tests that like works for a variety of different possible regex strings
    """
    check_query(
        f"select case when A {like_expression} {regex_string} then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_like_with_logical_operators(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    test that like behaves well with logical operators
    """
    check_query(
        f"select A from table1 where A {like_expression} {regex_string} and B like {regex_string}",
        bodosql_string_types,
        spark_info,
    )
    check_query(
        f"select B from table1 where A {like_expression} {regex_string} or B like {regex_string}",
        bodosql_string_types,
        spark_info,
    )


def test_like_cols(
    basic_df, regex_string, spark_info, like_expression, memory_leak_check
):
    """tests that like is working in the column case"""
    check_query(
        f"select A from table1 where C {like_expression} {regex_string} or B {like_expression} {regex_string}",
        basic_df,
        spark_info,
        check_dtype=False,  # need this for case where the select retuns empty table
    )


@pytest.mark.slow
def test_like_constants(
    basic_df,
    regex_string,
    string_constants,
    spark_info,
    like_expression,
    memory_leak_check,
):
    """
    tests that like works on constant strings
    """
    query = f"select A from table1 where '{string_constants}' {like_expression} {regex_string}"
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_nested_upper_lower(bodosql_string_types, spark_info):
    """
    Tests that lower/upper calls nest properly
    """
    check_query(
        f"select lower(upper(lower(upper(A)))) from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def test_upper_lower_scalars(basic_df, string_constants, spark_info, memory_leak_check):
    """
    Tests that lower/upper calls work on scalar values
    """
    """
    "select A, upper('{string_constants}'), lower('{string_constants}') from table1" causes an issue, so for now,
    I'm just doing it as two seperate queries
    """
    query = f"select A, upper('{string_constants}') from table1"

    query2 = f"select A, lower('{string_constants}') from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_upper_lower_scalars_nested(
    basic_df, string_constants, spark_info, memory_leak_check
):
    """
    Tests that nested lower/upper calls work on scalar values
    """
    query = f"select A, upper(lower(upper('{string_constants}'))) from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_upper_lower_like_constants(
    basic_df,
    regex_string,
    string_constants,
    spark_info,
    like_expression,
    # memory_leak_check, Seems to be leaking memory sporatically, see [BS-534]
):
    """
    Tests that lower/upper works on string constants
    """
    check_query(
        f"select A from table1 where upper('{string_constants}') {like_expression} upper({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )
    check_query(
        f"select A from table1 where lower('{string_constants}') {like_expression} upper({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )
    check_query(
        f"select A from table1 where upper('{string_constants}') {like_expression} lower({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_pythonic_regex(
    bodosql_string_types,
    pythonic_regex,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that pythonic regex is working as inteded
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '{pythonic_regex}'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_wildcardless_like(pandas_code)


@pytest.mark.slow
def test_all_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex that is all %% is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)


@pytest.mark.slow
def test_all_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex that is all %% is correct
    """
    check_query(
        f"select case when A {like_expression} '%%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_leading_percent(
    bodosql_string_types,
    spark_info,
    like_expression,  # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex starting with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%o'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} '%.o'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_percent_like(pandas_code)


@pytest.mark.slow
def test_leading_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,  # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex starting with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%.o' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_trailing_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex ending with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} 'h%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_end_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} 'h.%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_end_percent_like(pandas_code)


@pytest.mark.slow
def test_trailing_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex ending with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%.o' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_both_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # memory_leak_check Seems to be failing memory leak check intermitently, see BS-534
):
    """
    checks that a regex starting and ending with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%e%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} '%e.%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)


@pytest.mark.slow
def test_both_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,
    # memory_leak_check Seems to be failing memory leak check intermitently, see BS-534
):
    """
    checks that a regex starting and ending with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%e%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        f"select case when A {like_expression} '%e.%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def check_wildcardless_like(pandas_code):
    """
    Checks that given pandas_code doesn't contain any contains
    code because the regular expression didn't contain any
    SQL wildcards.
    """
    assert ".str.contains" not in pandas_code


def check_start_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses endswith
    because the regular expression only included % at the beginning.
    """
    assert ".str.endswith" in pandas_code


def check_end_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses startswith
    because the regular expression only included % at the beginning.
    """
    assert ".str.startswith" in pandas_code


def check_start_and_end_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses regex=False
    because the regular expression only included % at the beginning
    and end.
    """
    assert "regex=False" in pandas_code


@pytest.mark.slow
def test_utf_scalar(spark_info):
    check_query(
        "select 'ǖǘǚǜ'",
        {},
        spark_info,
        check_names=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT TRANSLATE(A, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "",
                                None,
                                "we've had vicious kings and idiot kings, but i don't know if we've ever been cursed with a vicious idiot for a king.",
                                "the next time i have an idea like that, punch me in the face.",
                                "that's what i do. i drink and i know things.",
                                "an unhappy wife is a wine merchant's best friend.",
                            ]
                        )
                    }
                ),
            ),
            id="vector_scalar_scalar_upper_to_lower",
        ),
        pytest.param(
            (
                "SELECT TRANSLATE(A, ' ,.'';:!?', '_') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "",
                                None,
                                "Weve_had_vicious_kings_and_idiot_kings_but_I_dont_know_if_weve_ever_been_cursed_with_a_vicious_idiot_for_a_king",
                                "The_next_time_I_have_an_idea_like_that_punch_me_in_the_face",
                                "Thats_what_I_do_I_drink_and_I_know_things",
                                "An_unhappy_wife_is_a_wine_merchants_best_friend",
                            ]
                        )
                    }
                ),
            ),
            id="vector_scalar_scalar_remove_punct_transform_space",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN LENGTH(A) < 10 THEN 'xxx' ELSE TRANSLATE(A, 'abcdefghijklmnopqrstuvwxyz', 'silverabcdfghjkmnopqtuwxyz') END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "xxx",
                                None,
                                "We'ue bsv uclcktp fcjap sjv cvckq fcjap, itq I vkj'q fjkw cr we'ue eueo ieej ltopev wcqb s uclcktp cvckq rko s fcja.",
                                "Tbe jexq qche I bsue sj cves gcfe qbsq, mtjlb he cj qbe rsle.",
                                "Tbsq'p wbsq I vk. I vocjf sjv I fjkw qbcjap.",
                                "Aj tjbsmmy wcre cp s wcje heolbsjq'p iepq rocejv.",
                            ]
                        )
                    }
                ),
            ),
            id="vector_scalar_scalar_subst_cipher_case",
        ),
    ],
)
def test_translate(args, spark_info, memory_leak_check):
    query, answer = args
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": [
                    "",
                    None,
                    "We've had vicious kings and idiot kings, but I don't know if we've ever been cursed with a vicious idiot for a king.",
                    "The next time I have an idea like that, punch me in the face.",
                    "That's what I do. I drink and I know things.",
                    "An unhappy wife is a wine merchant's best friend.",
                ]
            }
        )
    }
    check_query(
        query,
        ctx,
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
                "SELECT INITCAP(A) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "",
                                None,
                                "Alphabet Soup Is Delicious",
                                ' Yay\tAb\nCd\rEf\fGh\u000bIj!Kl?Mn@Op"Qr^St#Uv$Wx&Yz~Ab_Cd,Ef.Gh:Ij;Kl+Mn-Op*Qr%St/Uv|Wx\\Yz[Ab]Cd(Ef)Gh{Ij}Kl<Mn>Op1qr¢stπuv',
                                "Alpha,Beta,Gamma,Delta,Epsilon\nDo,Re,Mi,Fa,So,La,Ti,Do",
                                "Run-Of-The-Mill",
                            ]
                        )
                    }
                ),
            ),
            id="vector_default",
        ),
        pytest.param(
            (
                "SELECT INITCAP(A, ' ,') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "",
                                None,
                                "Alphabet Soup Is Delicious",
                                ' Yay\tab\ncd\ref\fgh\u000bij!kl?mn@op"qr^st#uv$wx&yz~ab_cd,Ef.gh:ij;kl+mn-op*qr%st/uv|wx\\yz[ab]cd(ef)gh{ij}kl<mn>op1qr¢stπuv',
                                "Alpha,Beta,Gamma,Delta,Epsilon\ndo,Re,Mi,Fa,So,La,Ti,Do",
                                "Run-of-the-mill",
                            ]
                        )
                    }
                ),
            ),
            id="vector_space_comma",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, ',') > 0 AND INSTR(A, ',') < 10 THEN 'xxx' ELSE INITCAP(A, '') END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "",
                                None,
                                "Alphabet soup is delicious",
                                ' yay\tab\ncd\ref\fgh\u000bij!kl?mn@op"qr^st#uv$wx&yz~ab_cd,ef.gh:ij;kl+mn-op*qr%st/uv|wx\\yz[ab]cd(ef)gh{ij}kl<mn>op1qr¢stπuv',
                                "xxx",
                                "Run-of-the-mill",
                            ]
                        )
                    }
                ),
            ),
            id="vector_empty_case",
        ),
    ],
)
def test_initcap(args, spark_info, memory_leak_check):
    query, answer = args
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": [
                    "",
                    None,
                    "alphabet SOUP is DELICIOUS",
                    ' yay\tab\ncd\ref\fgh\u000bij!kl?mn@op"qr^st#uv$wx&yz~ab_cd,ef.gh:ij;kl+mn-op*qr%st/uv|wx\\yz[ab]cd(ef)gh{ij}kl<mn>op1qr¢stπuv',
                    "alpha,beta,gamma,delta,epsilon\nDO,RE,MI,FA,SO,LA,TI,DO",
                    "Run-of-the-mill",
                ]
            }
        )
    }
    check_query(
        query,
        ctx,
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
                "SELECT EDITDISTANCE(A, 'pokerface') FROM table1",
                pd.DataFrame(
                    {0: pd.Series([7, None, 4, 6, 5, 1], dtype=pd.Int32Dtype())}
                ),
            ),
            id="scalar_vector_no_case_no_max",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN A IS NULL THEN -1 ELSE EDITDISTANCE(A, 'pokerface') END FROM table1",
                pd.DataFrame(
                    {0: pd.Series([7, -1, 4, 6, 5, 1], dtype=pd.Int32Dtype())}
                ),
            ),
            id="scalar_vector_with_case_no_max",
        ),
        pytest.param(
            (
                "SELECT EDITDISTANCE(A, 'pokerface', 5) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([5, None, 4, 5, 5, 1], dtype=pd.Int32Dtype())}
                ),
            ),
            id="scalar_vector_no_case_with_max",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN A IS NULL THEN -1 ELSE EDITDISTANCE(A, 'pokerface', 5) END FROM table1",
                pd.DataFrame(
                    {0: pd.Series([5, -1, 4, 5, 5, 1], dtype=pd.Int32Dtype())}
                ),
            ),
            id="scalar_vector_with_case_with_max",
        ),
    ],
)
def test_editdistance(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {"A": ["blackjack", None, "poker", "procedure", "disgrace", "poker face"]}
        )
    }
    query, answer = args
    check_query(
        query,
        ctx,
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
                "SELECT SPLIT_PART(A, ' ', 1) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            ["alphabet", "aaeaaeieaaeioiea", "alpha", "", "a", None]
                        )
                    }
                ),
            ),
            id="vector_space_1",
        ),
        pytest.param(
            (
                "SELECT SPLIT_PART(A, 'a', 2) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "lph",
                                "",
                                "lph",
                                "",
                                "  b     c  d e        f  g     h  i        j ",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="vector_a_2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT SPLIT_PART(A, '  ', -1) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "alphabet soup is delicious",
                                "aaeaaeieaaeioiea",
                                "delta epsilon",
                                "",
                                "j ",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="vector_doublespace_-1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT SPLIT_PART(A, RIGHT(A, 1), 1 + (LENGTH(A) % 6)) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                " deliciou",
                                "eie",
                                "",
                                "",
                                "",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="vector_lastchar_vector",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, '  ') > 0 THEN SPLIT_PART(A, '  ', 3) ELSE SPLIT_PART(A, 'e', -2) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "t soup is d",
                                "ioi",
                                "delta epsilon",
                                "",
                                "",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="case",
        ),
    ],
)
def test_split_part(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": [
                    "alphabet soup is delicious",
                    "aaeaaeieaaeioiea",
                    "alpha  beta gamma  delta epsilon",
                    "",
                    "a  b     c  d e        f  g     h  i        j ",
                    None,
                ]
            }
        )
    }
    query, answer = args
    check_query(
        query,
        ctx,
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
                "SELECT STRTOK(A) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                "alphabet",
                                "aaeaaeieaaeioiea",
                                "A.BCD.E.FGH.I.JKLMN.O.PQRST.U.VWXYZ",
                                "415-555-1234,",
                                "a",
                                None,
                            ]
                        )
                    }
                ),
            ),
            id="vector_default_default",
        ),
        pytest.param(
            (
                "SELECT STRTOK(A, ' .,-') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            ["alphabet", "aaeaaeieaaeioiea", "A", "415", "a", None]
                        )
                    }
                ),
            ),
            id="vector_symbols_default",
        ),
        pytest.param(
            (
                "SELECT STRTOK(A, ' ', 3) FROM table1",
                pd.DataFrame(
                    {0: pd.Series(["is", None, None, "937-555-3456", "c", None])}
                ),
            ),
            id="vector_space_5",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN INSTR(A, 'a') + INSTR(A, ' ') > 0 THEN STRTOK(A, 'a ', 1) ELSE 'xxx' END FROM table1",
                pd.DataFrame(
                    {0: pd.Series(["lph", "e", "xxx", "415-555-1234,", "b", "xxx"])}
                ),
            ),
            id="vector_aspace_1",
        ),
    ],
)
def test_strtok(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": [
                    "alphabet soup is delicious",
                    "aaeaaeieaaeioiea",
                    "A.BCD.E.FGH.I.JKLMN.O.PQRST.U.VWXYZ",
                    "415-555-1234, 412-555-2345, 937-555-3456",
                    "a  b     c  d e        f  g     h  i        j ",
                    None,
                ]
            }
        )
    }
    query, answer = args
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )
