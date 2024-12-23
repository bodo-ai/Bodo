"""
Test correctness of SQL case queries on BodoSQL
"""

import pandas as pd
import pytest
from numba.core import ir
from numba.core.ir_utils import find_callname, guard

import bodo
import bodosql
from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.utils import (
    DistTestPipeline,
    SeriesOptTestPipeline,
    dist_IR_contains,
    gen_nonascii_list,
    pytest_mark_one_rank,
    temp_config_override,
)
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        # TODO: Float literals (Spark outputs decimal type, not float)
        pytest.param(
            (1.3, -3124.2, 0.0, 314.1), id="float_literals", marks=pytest.mark.slow
        ),
        # Integer literals
        pytest.param((341, -3, 0, 3443), id="integer_literals", marks=pytest.mark.slow),
        # Boolean literals
        (True, False, False, True),
        # String literals
        pytest.param(
            ("'hello'", "'world'", "'goodbye'", "'spark'"),
            id="string_literals",
            marks=pytest.mark.slow,
        ),
        # TODO: Timestamp Literals (Cannot properly compare with Spark)
        pytest.param(
            (
                "TIMESTAMP '1997-01-31 09:26:50.124'",
                "TIMESTAMP '2021-05-31 00:00:00.00'",
                "TIMESTAMP '2021-04-28 00:40:00.00'",
                "TIMESTAMP '2021-04-29'",
            ),
            id="timestamp_literals",
        ),
        # TODO: Interval Literals (Cannot convert to Pandas in Spark)
        # (
        #     "INTERVAL '1' year",
        #     "INTERVAL '1' day",
        #     "INTERVAL '-4' hours",
        #     "INTERVAL '3' months",
        # ),
        # Binary literals. These are hex byte values starting with X''
        pytest.param(
            (
                f"X'{b'hello'.hex()}'",
                f"X'{b'world'.hex()}'",
                f"X'{b'goodbye'.hex()}'",
                f"X'{b'spark'.hex()}'",
            ),
            marks=pytest.mark.skip("[BE-3304] Support Bytes literals"),
            id="binary_literals",
        ),
    ]
)
def case_literals(request):
    """
    Fixture of possible literal choices for generating
    case code. There are 4 values to support possible nested cases.
    """
    return request.param


@pytest.mark.slow
def test_case_agg(bodosql_string_types, spark_info, memory_leak_check):
    """
    Test using an aggregate function on the output of a case statement.
    """
    query = """
        SELECT SUM(CASE
          WHEN A = 'hello'
          THEN 1
          ELSE 0
         END) FROM table1"""
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )

    # make sure scalar_optional_getitem generated by case is handled properly in 1D_Var
    # parfors and there is no distributed exscan
    @bodo.jit(pipeline_class=DistTestPipeline, all_args_distributed_varlength=True)
    def bodo_func(bc, query):
        return bc.sql(query)

    bodo_func(bodosql.BodoSQLContext(bodosql_string_types), query)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "exscan")


@pytest.mark.slow
def test_boolean_literal_case(basic_df, spark_info, memory_leak_check):
    """
    Tests the behavior of case when the possible results are boolean literals.
    """
    query1 = "Select B, Case WHEN A >= 2 THEN True ELSE True END as CASERES FROM table1"
    query2 = (
        "Select B, Case WHEN A >= 2 THEN True ELSE False END as CASERES FROM table1"
    )
    query3 = (
        "Select B, Case WHEN A >= 2 THEN False ELSE True END as CASERES FROM table1"
    )
    query4 = (
        "Select B, Case WHEN A >= 2 THEN False ELSE False END as CASERES FROM table1"
    )
    check_query(query1, basic_df, spark_info, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_dtype=False)
    check_query(query3, basic_df, spark_info, check_dtype=False)
    check_query(query4, basic_df, spark_info, check_dtype=False)


def test_case_literals(basic_df, case_literals, spark_info, memory_leak_check):
    """
    Test a case statement with each possible literal return type.
    """
    query = f"Select B, Case WHEN A >= 2 THEN {case_literals[0]} ELSE {case_literals[1]} END as CASERES FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CASERES"],
    )


@pytest.mark.slow
def test_case_literals_multiple_when(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with multiple whens and each possible literal return type.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CASERES FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CASERES"],
    )


@pytest.mark.slow
def test_case_literals_groupby(basic_df, case_literals, spark_info, memory_leak_check):
    """
    Test a case statement with each possible literal return type in a groupby.
    """
    query = f"Select B, Case WHEN A >= 2 THEN {case_literals[0]} ELSE {case_literals[1]} END as CASERES FROM table1 Group By A, B"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CASERES"],
    )


@pytest.mark.slow
def test_case_literals_multiple_when_groupby(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with each possible literal return type in a groupby.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CASERES FROM table1 Group By A, B"

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CASERES"],
    )


def test_case_literals_nonascii(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement with non-ASCII literals.
    """
    case_literals = gen_nonascii_list(4)

    query = f"Select B, Case WHEN A = 1 THEN '{case_literals[0]}' WHEN A = 2 THEN '{case_literals[1]}' WHEN B > 6 THEN '{case_literals[2]}' ELSE '{case_literals[3]}' END as CASERES FROM table1 Group By A, B"

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CASERES"],
    )


@pytest.mark.slow
def test_case_agg_groupby(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement with an aggregate function applied to each group.
    """
    query1 = "Select Case WHEN A >= 2 THEN Sum(B) ELSE 0 END as CASERES FROM table1 Group By A"
    query2 = "Select Case WHEN A >= 2 THEN Count(B) ELSE 0 END as CASERES FROM table1 Group By A"
    check_query(query1, basic_df, spark_info, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_case_no_else_clause_literals(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement that doesn't have an else clause whose values are scalars
    """
    query = f"Select Case WHEN A >= 2 THEN {case_literals[0]} WHEN A = 1 THEN {case_literals[1]} END as CASERES FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_decimal=["CASERES"],
    )


def test_case_no_else_clause_columns(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement that doesn't have an else clause whose values are columns
    """
    query = "Select Case WHEN A >= 2 THEN A WHEN A < 0 THEN B END FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


@pytest.mark.slow
def test_tz_aware_case_null(representative_tz, memory_leak_check):
    """
    Tests a case statement using a column + NULL on tz-aware timestamp
    data.
    """
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                "2022/1/1", periods=30, freq="6D5H", tz=representative_tz
            ),
            "B": [True, False, True, True, False] * 6,
        }
    )
    query = "Select Case WHEN B THEN A END as output FROM table1"
    ctx = {"TABLE1": df}
    expected_output = pd.DataFrame({"OUTPUT": df["A"].copy()})
    expected_output[~df.B] = None
    check_query(
        query, ctx, None, expected_output=expected_output, session_tz=representative_tz
    )


def test_case_no_inlining(basic_df, spark_info, memory_leak_check):
    """
    Test a case that makes sure the no inlining path works as expected.
    """
    # Save the old threshold and set it to 1 so that the case statement
    # is not inlined.
    with temp_config_override("COMPLEX_CASE_THRESHOLD", 1):
        query = "Select B, Case WHEN A = 1 THEN 1 WHEN A = 2 THEN 2 WHEN B > 6 THEN 3 ELSE NULL END as CASERES FROM table1"
        check_query(
            query,
            basic_df,
            spark_info,
            check_dtype=False,
        )

        # Add a check that the case statement is not inlined.
        @bodo.jit(pipeline_class=SeriesOptTestPipeline)
        def bodo_func(bc, query):
            return bc.sql(query)

        bodo_func(bodosql.BodoSQLContext(basic_df), query)
        fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        # Inspect the IR for a bodosql_case_kernel call
        found = False
        for block in fir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign):
                    fdef = guard(find_callname, fir, stmt.value)
                    found = found or (
                        fdef is not None and fdef[0] == "bodosql_case_kernel"
                    )
        assert found, "bodosql_case_kernel not found in IR, case statement was inlined"


@pytest.mark.timeout(600)
@pytest.mark.slow
# This test is very slow when NP > 1, so we disable it.
@pytest_mark_one_rank
def test_case_indent_limit(memory_leak_check):
    """
    Test of case statement bug noted in BSE-853.
    This demonstrates a Python limitation where if a
    case statement is naively decomposed into if/else
    it hits a max indentation limit.
    """
    query = """
    select
        case
            when endswith(A, ' a') or startswith(A, 'a ') or A like '% abortion %' then 'a'
            when endswith(A, ' ab') or startswith(A, 'ab ') or A like '% ab %' then 'ab'
            when endswith(A, ' ad') or startswith(A, 'ad ') or A like '% ad %' then 'ad'
            when endswith(A, ' al') or startswith(A, 'al ') or A like '% al %' then 'al'
            when endswith(A, ' al2') or startswith(A, 'al2 ') or A like '% al2 %' then 'al2'
            when endswith(A, ' am') or startswith(A, 'am ') or A like '% am %' then 'am'
            when endswith(A, ' an') or startswith(A, 'an ') or A like '% an %' then 'an'
            when endswith(A, ' an2') or startswith(A, 'an2 ') or A like '% an2 %' then 'an2'
            when endswith(A, ' ar') or startswith(A, 'ar ') or A like '% ar %' then 'ar'
            when endswith(A, ' as') or startswith(A, 'as ') or A like '% as %' then 'as'
            when endswith(A, ' as2') or startswith(A, 'as2 ') or A like '% as2 %' then 'as2'
            when endswith(A, ' ay') or startswith(A, 'ay ') or A like '% ay %' then 'ay'
            when endswith(A, ' ba') or startswith(A, 'ba ') or A like '% ba %' then 'ba'
            when endswith(A, ' ba2') or startswith(A, 'ba2 ') or A like '% ba2 %' then 'ba2'
            when endswith(A, ' ba3') or startswith(A, 'ba3 ') or A like '% ba3 %' then 'ba3'
            when endswith(A, ' ba4') or startswith(A, 'ba4 ') or A like '% ba4 %' then 'ba4'
            when endswith(A, ' ba5') or startswith(A, 'ba5 ') or A like '% ba5 %' then 'ba5'
            when endswith(A, ' bd') or startswith(A, 'bd ') or A like '% bd %' then 'bd'
            when endswith(A, ' be') or startswith(A, 'be ') or A like '% be %' then 'be'
            when endswith(A, ' be2') or startswith(A, 'be2 ') or A like '% be2 %' then 'be2'
            when endswith(A, ' be3') or startswith(A, 'be3 ') or A like '% be3 %' then 'be3'
            when endswith(A, ' be4') or startswith(A, 'be4 ') or A like '% be4 %' then 'be4'
            when endswith(A, ' be5') or startswith(A, 'be5 ') or A like '% be5 %' then 'be5'
            when endswith(A, ' bi') or startswith(A, 'bi ') or A like '% bi %' then 'bi'
            when endswith(A, ' bi2') or startswith(A, 'bi2 ') or A like '% bi2 %' then 'bi2'
            when endswith(A, ' bl') or startswith(A, 'bl ') or A like '% bl %' then 'bl'
            when endswith(A, ' bm') or startswith(A, 'bm ') or A like '% bm %' then 'bm'
            when endswith(A, ' bo') or startswith(A, 'bo ') or A like '% bo %' then 'bo'
            when endswith(A, ' bo2') or startswith(A, 'bo2 ') or A like '% bo2 %' then 'bo2'
            when endswith(A, ' bo3') or startswith(A, 'bo3 ') or A like '% bo3 %' then 'bo3'
            when endswith(A, ' bo4') or startswith(A, 'bo4 ') or A like '% bo4 %' then 'bo4'
            when endswith(A, ' bo5') or startswith(A, 'bo5 ') or A like '% bo5 %' then 'bo5'
            when endswith(A, ' bo6') or startswith(A, 'bo6 ') or A like '% bo6 %' then 'bo6'
            when endswith(A, ' br') or startswith(A, 'br ') or A like '% br %' then 'br'
            when endswith(A, ' br2') or startswith(A, 'br2 ') or A like '% br2 %' then 'br2'
            when endswith(A, ' br3') or startswith(A, 'br3 ') or A like '% br3 %' then 'br3'
            when endswith(A, ' br4') or startswith(A, 'br4 ') or A like '% br4 %' then 'br4'
            when endswith(A, ' br5') or startswith(A, 'br5 ') or A like '% br5 %' then 'br5'
            when endswith(A, ' bu') or startswith(A, 'bu ') or A like '% bu %' then 'bu'
            when endswith(A, ' bu2') or startswith(A, 'bu2 ') or A like '% bu2 %' then 'bu2'
            when endswith(A, ' bu3') or startswith(A, 'bu3 ') or A like '% bu3 %' then 'bu3'
            when endswith(A, ' bu4') or startswith(A, 'bu4 ') or A like '% bu4 %' then 'bu4'
            when endswith(A, ' ca') or startswith(A, 'ca ') or A like '% ca %' then 'ca'
            when endswith(A, ' ca2') or startswith(A, 'ca2 ') or A like '% ca2 %' then 'ca2'
            when endswith(A, ' ca3') or startswith(A, 'ca3 ') or A like '% ca3 %' then 'ca3'
            when endswith(A, ' ca4') or startswith(A, 'ca4 ') or A like '% ca4 %' then 'ca4'
            when endswith(A, ' ca5') or startswith(A, 'ca5 ') or A like '% ca5 %' then 'ca5'
            when endswith(A, ' ca6') or startswith(A, 'ca6 ') or A like '% ca6 %' then 'ca6'
            when endswith(A, ' ca7') or startswith(A, 'ca7 ') or A like '% ca7 %' then 'ca7'
            when endswith(A, ' ca8') or startswith(A, 'ca8 ') or A like '% ca8 %' then 'ca8'
            when endswith(A, ' cb') or startswith(A, 'cb ') or A like '% cb %' then 'cb'
            when endswith(A, ' ce') or startswith(A, 'ce ') or A like '% ce %' then 'ce'
            when endswith(A, ' ch') or startswith(A, 'ch ') or A like '% ch %' then 'ch'
            when endswith(A, ' ch2') or startswith(A, 'ch2 ') or A like '% ch2 %' then 'ch2'
            when endswith(A, ' ch3') or startswith(A, 'ch3 ') or A like '% ch3 %' then 'ch3'
            when endswith(A, ' cl') or startswith(A, 'cl ') or A like '% cl %' then 'cl'
            when endswith(A, ' cl2') or startswith(A, 'cl2 ') or A like '% cl2 %' then 'cl2'
            when endswith(A, ' cl3') or startswith(A, 'cl3 ') or A like '% cl3 %' then 'cl3'
            when endswith(A, ' co') or startswith(A, 'co ') or A like '% co %' then 'co'
            when endswith(A, ' co2') or startswith(A, 'co2 ') or A like '% co2 %' then 'co2'
            when endswith(A, ' co3') or startswith(A, 'co3 ') or A like '% co3 %' then 'co3'
            when endswith(A, ' co4') or startswith(A, 'co4 ') or A like '% co4 %' then 'co4'
            when endswith(A, ' co5') or startswith(A, 'co5 ') or A like '% co5 %' then 'co5'
            when endswith(A, ' co6') or startswith(A, 'co6 ') or A like '% co6 %' then 'co6'
            when endswith(A, ' co7') or startswith(A, 'co7 ') or A like '% co7 %' then 'co7'
            when endswith(A, ' co8') or startswith(A, 'co8 ') or A like '% co8 %' then 'co8'
            when endswith(A, ' co9') or startswith(A, 'co9 ') or A like '% co9 %' then 'co9'
            when endswith(A, ' cr') or startswith(A, 'cr ') or A like '% cr %' then 'cr'
            when endswith(A, ' cu') or startswith(A, 'cu ') or A like '% cu %' then 'cu'
            when endswith(A, ' cu2') or startswith(A, 'cu2 ') or A like '% cu2 %' then 'cu2'
            when endswith(A, ' da') or startswith(A, 'da ') or A like '% da %' then 'da'
            when endswith(A, ' da2') or startswith(A, 'da2 ') or A like '% da2 %' then 'da2'
            when endswith(A, ' de') or startswith(A, 'de ') or A like '% de %' then 'de'
            when endswith(A, ' de2') or startswith(A, 'de2 ') or A like '% de2 %' then 'de2'
            when endswith(A, ' de3') or startswith(A, 'de3 ') or A like '% de3 %' then 'de3'
            when endswith(A, ' di') or startswith(A, 'di ') or A like '% di %' then 'di'
            when endswith(A, ' di2') or startswith(A, 'di2 ') or A like '% di2 %' then 'di2'
            when endswith(A, ' di3') or startswith(A, 'di3 ') or A like '% di3 %' then 'di3'
            when endswith(A, ' di4') or startswith(A, 'di4 ') or A like '% di4 %' then 'di4'
            when endswith(A, ' dk') or startswith(A, 'dk ') or A like '% dk %' then 'dk'
            when endswith(A, ' do') or startswith(A, 'do ') or A like '% do %' then 'do'
            when endswith(A, ' do2') or startswith(A, 'do2 ') or A like '% do2 %' then 'do2'
            when endswith(A, ' do3') or startswith(A, 'do3 ') or A like '% do3 %' then 'do3'
            when endswith(A, ' do4') or startswith(A, 'do4 ') or A like '% do4 %' then 'do4'
            when endswith(A, ' dr') or startswith(A, 'dr ') or A like '% dr %' then 'dr'
            when endswith(A, ' du') or startswith(A, 'du ') or A like '% du %' then 'du'
            when endswith(A, ' dy') or startswith(A, 'dy ') or A like '% dy %' then 'dy'
            when endswith(A, ' el') or startswith(A, 'el ') or A like '% el %' then 'el'
            when endswith(A, ' el2') or startswith(A, 'el2 ') or A like '% el2 %' then 'el2'
            when endswith(A, ' er') or startswith(A, 'er ') or A like '% er %' then 'er'
            when endswith(A, ' er2') or startswith(A, 'er2 ') or A like '% er2 %' then 'er2'
            when endswith(A, ' er3') or startswith(A, 'er3 ') or A like '% er3 %' then 'er3'
            when endswith(A, ' er4') or startswith(A, 'er4 ') or A like '% er4 %' then 'er4'
            when endswith(A, ' es') or startswith(A, 'es ') or A like '% es %' then 'es'
            when endswith(A, ' et') or startswith(A, 'et ') or A like '% et %' then 'et'
            when endswith(A, ' f') or startswith(A, 'f ') or A like '% f %' then 'f'
            when endswith(A, ' f2') or startswith(A, 'f2 ') or A like '% f2 %' then 'f2'
            when endswith(A, ' f3') or startswith(A, 'f3 ') or A like '% f3 %' then 'f3'
            else null
        end as sensitive_word
    from table1
    """
    df = pd.DataFrame({"A": ["er ", "a ", " el2    ", " f2 "] * 5})
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({"SENSITIVE_WORD": ["er", "a", "el2", "f2"] * 5})
    check_query(query, ctx, None, expected_output=py_output)


def test_nested_case(memory_leak_check):
    """Tests when a case statement is nested
    within another case statement.
    """
    query = """
    Select CASE
        WHEN A = 3
                then (CASE
                    WHEN B > 5
                        then 1
                    Else 0
                end)
                +
                (CASE
                    WHEN C < 3
                        then 1
                    Else 0
                end)
        ELSE -1
        end as res
    from table1
    """
    df = pd.DataFrame(
        {"A": [3, 3, 3, 3, 4] * 3, "B": [1, 6, 1, 6, 1] * 3, "C": [2, 4, 4, 2, 4] * 3}
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({"RES": [1, 1, 0, 2, -1] * 3})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


def test_null_array_to_timezone_aware(memory_leak_check):
    """Test CASE + casting NULL to Timezone-aware type"""
    df = pd.DataFrame({"A": [pd.Timestamp("1999-12-15 11:03:40", tz="Asia/Dubai")] * 5})
    ctx = {"TABLE1": df}

    query = """
        select case when t is not null then t else A end as out1
        from table1
        cross join (select null as t)
        """
    check_query(
        query, ctx, None, expected_output=df, check_names=False, session_tz="Asia/Dubai"
    )


def test_case_proper_character_escape(memory_leak_check):
    """Tests that the case statement properly handles escaping characters"""

    df = pd.DataFrame(
        {
            "PRODUCT_ID": [
                None,
                '""',
                "   ",
                "12345",
                " 12345 ",
                "",
            ]
            * 5
        }
    )
    ctx = {"TABLE1": df}

    expected_out = df = pd.DataFrame(
        {
            "PRODUCT_ID": [
                None,
                None,
                None,
                "12345",
                " 12345 ",
                None,
            ]
            * 5
        }
    )

    query = """
    SELECT
        CASE WHEN PRODUCT_ID='""' OR REPLACE(PRODUCT_ID,' ','')='' THEN NULL ELSE PRODUCT_ID END AS PRODUCT_ID
    FROM table1
    """

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_out,
        check_names=False,
    )


def test_case_proper_character_escape_regex(memory_leak_check):
    """Tests that the case statement properly handles escaping regex characters if needed"""

    orig_web_url_list = [
        None,
        "https://bodo.atlassian.net/jira/software/c/projects/BSE/boards/25?assignee=60a3bbae47ba02006f1f8fbb&selectedIssue=BSE-2523",
        "https://bodo.atlassian.net/jira/software/c/projects/BSE/boards/25?assignee=60a3bbae47ba02006f1f8fbb&selectedIssue=BSE-2583",
        "https://www.merriam-webster.com/dictionary/test",
        " 12345 ",
        "",
        "BSE-2523",
    ] * 5

    df = pd.DataFrame({"WEB_URL": orig_web_url_list})
    ctx = {"TABLE1": df}

    expected_out = df = pd.DataFrame(
        {
            "WEB_URL": orig_web_url_list,
            "EXTRACTED_VALUE": [
                None,
                "2523",
                "2583",
                None,
                None,
                None,
                None,
            ]
            * 5,
        }
    )

    query = """
    SELECT
        WEB_URL,
        CASE WHEN STARTSWITH(WEB_URL, 'https://bodo.atlassian.net/') THEN REGEXP_SUBSTR(WEB_URL, 'selectedIssue=BSE-(\\d+)', 1::BIGINT, 1::BIGINT, 'e', 1::BIGINT) ELSE NULL END AS PRODUCT_ID
    FROM table1
    """

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_out,
        check_names=False,
    )


def test_case_in(memory_leak_check):
    """
    Test that IN works properly inside of a case statement.
    """
    query = "select case when A IN (2, 3, 4) THEN 1 ELSE 0 END as OUTPUT from TABLE1"
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    ctx = {"TABLE1": df}
    expected_output = pd.DataFrame({"OUTPUT": [0, 1, 1, 1, 0]})
    check_query(query, ctx, None, expected_output=expected_output, check_dtype=False)
