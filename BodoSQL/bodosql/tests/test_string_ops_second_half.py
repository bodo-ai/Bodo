import uuid

import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    gen_nonascii_list,
    pytest_mark_one_rank,
    pytest_slow_unless_codegen,
)
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "select A from table1 where contains(A, 'a')",
                pd.DataFrame({"A": ["alpha", "beta", "zeta"]}),
            ),
        ),
        pytest.param(
            (
                "select B from table1 where contains(B, 'bet')",
                pd.DataFrame({"B": ["beta"]}),
            ),
        ),
        pytest.param(
            (
                "select C from table1 where contains(C, 'Â Ê Î')",
                pd.DataFrame({"C": ["Â Ê Î"]}),
            ),
        ),
        pytest.param(
            (
                "select A, B, C from table1 where contains(C, 'aaaaaaaaa')",
                pd.DataFrame({"A": [], "B": [], "C": []}).astype(object),
            ),
        ),
        pytest.param(
            ("select C, D from table1 where contains(D, X'616263')", pd.DataFrame()),
            marks=pytest.mark.skip(
                "[BE-3304]: Add support for binary literals in BodoSQL"
            ),
        ),
    ],
)
@pytest.mark.slow
def test_contains(args, spark_info, memory_leak_check):
    df = pd.DataFrame(
        {
            "A": ["alpha", "beta", "zeta", "pi", "epsilon"],
            "B": ["", "beta", "zebra", "PI", "foo"],
            "C": gen_nonascii_list(5),
            "D": [b"abc", b"def", b"", b"000000", b"123"],
        }
    )

    query, expected_output = args
    check_query(
        query,
        {"TABLE1": df},
        spark_info,
        check_names=False,
        expected_output=expected_output,
    )


def test_concat_operator_cols(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat operator is working for columns"""
    query = "select A || B || 'scalar' || C from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_concat_operator_scalars(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat operator is working for scalar values"""
    query = (
        "select CASE WHEN A > 'A' THEN B || ' case1' ELSE C || ' case2' END from table1"
    )
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def test_concat_fn_cols(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat function is working for columns"""
    query = "select CONCAT(A, B, 'scalar', C) from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def test_concat_fn_single_arg(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat function is working for a single argument"""
    query = "select CONCAT(A) from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_concat_fn_scalars(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat function is working for scalar values"""
    query = "select CASE WHEN A > 'A' THEN CONCAT(B, ' case1') ELSE CONCAT(C, ' case2') END from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def test_concat_ws_single_arg(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat_ws function is working for a single argument"""
    query = "select CONCAT_WS('_', A) from table1"
    spark_query = "select CONCAT(A) from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_concat_ws_cols(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat_ws function is working for columns"""
    query = "select CONCAT_WS('_', A, B, C), CONCAT_WS(A, B) from table1"
    spark_query = "select CONCAT(A, '_', B, '_', C), B from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_concat_ws_scalars(bodosql_string_types, spark_info, memory_leak_check):
    """Checks that the concat_ws function is working for scalar values"""
    query = "select CASE WHEN A > 'A' THEN CONCAT_WS(' case1 ', B, C, A) ELSE CONCAT_WS(A,B,C) END from table1"
    spark_query = "select CASE WHEN A > 'A' THEN CONCAT(B, ' case1 ', C, ' case1 ', A) ELSE CONCAT(B, A, C) END from table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_concat_ws_single_arg_binary(
    bodosql_binary_types, spark_info, memory_leak_check
):
    """Checks that the concat_ws function is working for a single argument"""
    query = "select CONCAT_WS(B, A) from table1"
    spark_query = "select CONCAT(A) from table1"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_concat_cols_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """Checks that the concat_ws function is working for columns"""
    query = (
        "select CONCAT(B, C) as A0, CONCAT_WS(A, B, C) as A1, A || B as A2 from table1"
    )
    spark_query = (
        "select CONCAT(B, C) as A0, CONCAT(B, A, C) as A1, A || B as A2 from table1"
    )
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
        convert_columns_bytearray=["A0", "A1", "A2"],
    )


@pytest.mark.slow
def test_concat_ws_scalars_binary(bodosql_binary_types, memory_leak_check):
    """Checks that the concat_ws function is working for scalar values"""
    query = "SELECT CASE WHEN A IS NOT NULL THEN CONCAT_WS(TO_BINARY('2c'), C, A) ELSE CONCAT_WS(A, B, C) END AS A0 FROM table1"
    df = bodosql_binary_types["TABLE1"]
    # If "A IS NOT NULL" is false, then CONCAT_WS(A, B, C) will also be null, so we can just use CONCAT_WS(TO_BINARY('2c'), C, A)
    answer = df["C"] + b"2c" + df["A"]
    answer = answer.where(answer.notnull(), None).astype(
        pd.ArrowDtype(pa.large_binary())
    )
    check_query(
        query,
        bodosql_binary_types,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"A0": answer}),
    )


def test_string_fns_cols(
    spark_info, bodosql_string_fn_testing_df, string_fn_info, memory_leak_check
):
    """tests that the specified string functions work on columns"""
    bodo_fn_name = string_fn_info[0]
    arglistString = ", ".join(string_fn_info[1])
    bodo_fn_call = f"{bodo_fn_name}({arglistString})"

    query = f"SELECT {bodo_fn_call} FROM table1"

    if bodo_fn_name in BODOSQL_TO_PYSPARK_FN_MAP:
        spark_fn_name = BODOSQL_TO_PYSPARK_FN_MAP[bodo_fn_name]
        spark_fn_call = f"{spark_fn_name}({arglistString})"
        spark_query = f"SELECT {spark_fn_call} FROM table1"
    else:
        spark_query = None

    check_query(
        query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_string_fns_scalars(
    spark_info, bodosql_string_fn_testing_df, string_fn_info, memory_leak_check
):
    """tests that the specified string functions work on Scalars"""
    bodo_fn_name = string_fn_info[0]
    arglistString = ", ".join(string_fn_info[1])
    bodo_fn_call = f"{bodo_fn_name}({arglistString})"
    retval_1 = string_fn_info[2][0]
    retval_2 = string_fn_info[2][1]

    query = f"SELECT CASE WHEN {bodo_fn_call} = {retval_1} THEN {retval_2} ELSE {bodo_fn_call} END FROM table1"
    if bodo_fn_name in BODOSQL_TO_PYSPARK_FN_MAP:
        spark_fn_name = BODOSQL_TO_PYSPARK_FN_MAP[bodo_fn_name]
        spark_fn_call = f"{spark_fn_name}({arglistString})"
        spark_query = f"SELECT CASE WHEN {spark_fn_call} = {retval_1} THEN {retval_2} ELSE {spark_fn_call} END FROM table1"
    else:
        spark_query = None

    check_query(
        query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def mk_broadcasted_string_queries():
    """Makes a list of params related to the new broadcasted string function array
    kernels. Constructed from a list of tuples where the first element
    is the query itself (where the cols come from bodosql_string_fn_testing_df),
    the second element is the name of the test case, and the third element
    is a boolean indicated whether the test is slow.

    Also, each query type is tagged with a skipif attached to the corresponding
    Bodo mini-release version required for the array kernel to exist. The
    query types are inferred by the naming scheme of the ids: "TYPE_other_info"
    """
    queries = [
        (
            "SELECT LPAD(strings_null_1, mixed_ints_null) from table1",
            "LPAD-all_vector-2args",
            True,
        ),
        (
            "SELECT LPAD(strings_null_1, 20) from table1",
            "LPAD-scalar_int-2args",
            True,
        ),
        ("SELECT LPAD('A', 25) from table1", "LPAD-all_scalar-2args", True),
        (
            "SELECT RPAD(strings_null_1, mixed_ints_null) from table1",
            "RPAD-all_vector-2args",
            True,
        ),
        (
            "SELECT RPAD(strings_null_1, 25) from table1",
            "RPAD-scalar_int-2args",
            True,
        ),
        (
            "SELECT RPAD('words', 25) from table1",
            "RPAD-two_scalar-2args",
            True,
        ),
        (
            "SELECT LPAD(strings_null_1, mixed_ints_null, strings_null_2) from table1",
            "LPAD-all_vector",
            False,
        ),
        (
            "SELECT LPAD(strings_null_1, mixed_ints_null, ' ') from table1",
            "LPAD-scalar_str",
            False,
        ),
        (
            "SELECT LPAD(strings_null_1, 20, strings_null_2) from table1",
            "LPAD-scalar_int",
            True,
        ),
        ("SELECT LPAD('A', 25, ' ') from table1", "LPAD-all_scalar", False),
        (
            "SELECT RPAD(strings_null_1, mixed_ints_null, strings_null_2) from table1",
            "RPAD-all_vector",
            False,
        ),
        (
            "SELECT RPAD(strings_null_1, mixed_ints_null, 'ABC') from table1",
            "RPAD-scalar_str",
            True,
        ),
        (
            "SELECT RPAD(strings_null_1, 25, strings_null_2) from table1",
            "RPAD-scalar_int",
            True,
        ),
        (
            "SELECT RPAD('words', 25, strings_null_2) from table1",
            "RPAD-two_scalar",
            True,
        ),
        ("SELECT RPAD('B', 20, '_$*') from table1", "RPAD-all_scalar", True),
        (
            "SELECT LEFT(strings_null_1, positive_ints) from table1",
            "LEFT_all_vector",
            False,
        ),
        ("SELECT LEFT(strings_null_1, 3) from table1", "LEFT_scalar_int", False),
        (
            "SELECT LEFT('Alphabet Soup Is Delicious!!!', mixed_ints) from table1",
            "LEFT_scalar_str",
            True,
        ),
        ("SELECT LEFT('anagrams are cool', 10) from table1", "LEFT_all_scalar", False),
        (
            "SELECT RIGHT(strings_null_1, positive_ints) from table1",
            "RIGHT_all_vector",
            False,
        ),
        ("SELECT RIGHT(strings_null_1, 3) from table1", "RIGHT_scalar_int", True),
        (
            "SELECT RIGHT('Alphabet Soup Is Delicious!!!', mixed_ints) from table1",
            "RIGHT_scalar_str",
            False,
        ),
        ("SELECT RIGHT('anagrams are cool', 10) from table1", "RIGHT_all_scalar", True),
        ("SELECT ORD(strings_null_1) from table1", "ORD_ASCII_vector", False),
        ("SELECT ORD('Hello!') from table1", "ORD_ASCII_scalar", False),
        ("SELECT CHAR(positive_ints) from table1", "CHAR_vector", False),
        ("SELECT CHAR(42) from table1", "CHAR_scalar", False),
        ("SELECT CHR(positive_ints) from table1", "CHR_vector", False),
        ("SELECT CHR(42) from table1", "CHR_scalar", False),
        (
            "SELECT REPEAT(strings_null_1, mixed_ints_null) from table1",
            "REPEAT_all_vector",
            False,
        ),
        ("SELECT REPEAT('AB', positive_ints) from table1", "REPEAT_scalar_str", True),
        ("SELECT REPEAT(strings_null_1, 2) from table1", "REPEAT_scalar_int", True),
        ("SELECT REPEAT('alphabet', 3) from table1", "REPEAT_all_scalar", False),
        ("SELECT REVERSE(strings_null_1) from table1", "REVERSE_vector", False),
        (
            "SELECT REVERSE('I drive a racecar to work!') from table1",
            "REVERSE_scalar",
            False,
        ),
        (
            "SELECT REPLACE(strings_null_1, LEFT(strings_null_1, 1), strings_null_2) from table1",
            "REPLACE_all_vector",
            False,
        ),
        (
            "SELECT REPLACE(strings_null_1, 'a', strings_null_2) from table1",
            "REPLACE_scalar_str_1",
            True,
        ),
        (
            "SELECT REPLACE(strings, '  ', '*') from table1",
            "REPLACE_scalar_str_2",
            True,
        ),
        (
            "SELECT REPLACE('alphabetagamma', 'a', '_') from table1",
            "REPLACE_all_scalar",
            False,
        ),
        ("SELECT SPACE(mixed_ints_null) from table1", "SPACE_vector", False),
        ("SELECT SPACE(10) from table1", "SPACE_scalar", False),
        (
            "SELECT INSTR(strings, strings_null_2) from table1",
            "INSTR_all_vector",
            False,
        ),
        ("SELECT INSTR(strings_null_1, 'a') from table1", "INSTR_vector_scalar", False),
        (
            "SELECT INSTR('alphabet soup is delicious!', ' ') from table1",
            "INSTR_all_scalar",
            False,
        ),
    ]
    # Add bodo release version dependencies to this dictionary whenever
    # implementing a new kernel, i.e. "INSTR": bodo_version_older(2022, 6, 2)
    dependencies = {}
    result = []
    for query, tag, slow in queries:
        name = tag[: tag.find("_")]
        marks = ()
        if slow:
            marks += (pytest.mark.slow,)
        if name in dependencies:
            marks += (
                pytest.mark.skipif(
                    dependencies[name],
                    reason=f"Cannot test {name} until next mini release",
                ),
            )
        param = pytest.param(query, id=tag, marks=marks)
        result.append(param)
    return result


@pytest.fixture(params=mk_broadcasted_string_queries())
def broadcasted_string_query(request):
    return request.param


def test_string_fns_scalar_vector(
    broadcasted_string_query,
    spark_info,
    bodosql_string_fn_testing_df,
    memory_leak_check,
):
    spark_query = broadcasted_string_query
    for func in BODOSQL_TO_PYSPARK_FN_MAP:
        spark_query = spark_query.replace(func, BODOSQL_TO_PYSPARK_FN_MAP[func])
        # The equivalent function for INSTR is LOCATE, but the arguments
        # are taken in opposite order
        if func == "INSTR" and "INSTR" in broadcasted_string_query:
            lhs, rhs = spark_query.split("LOCATE(")
            args, rhs = rhs.split(") from")
            arg0, arg1 = args.split(",")
            spark_query = f"{lhs} LOCATE({arg1}, {arg0}) FROM {rhs}"

    check_query(
        broadcasted_string_query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        equivalent_spark_query=spark_query,
        sort_output=False,
    )


def test_string_fns_scalar_vector_case(
    broadcasted_string_query,
    spark_info,
    bodosql_string_fn_testing_df,
    memory_leak_check,
):
    lhs, _ = broadcasted_string_query.split(" from ")
    lhs = lhs[7:]
    broadcasted_string_query = (
        f"SELECT CASE WHEN positive_ints < 0 THEN NULL ELSE {lhs} END from table1"
    )

    spark_query = broadcasted_string_query
    for func in BODOSQL_TO_PYSPARK_FN_MAP:
        spark_query = spark_query.replace(func, BODOSQL_TO_PYSPARK_FN_MAP[func])
        # The equivalent function for INSTR is LOCATE, but the arguments
        # are taken in opposite order
        if func == "INSTR" and "INSTR" in broadcasted_string_query:
            lhs, rhs = spark_query.split("LOCATE(")
            args, rhs = rhs.split(") END from")
            arg0, arg1 = args.split(",")
            spark_query = f"{lhs} LOCATE({arg1}, {arg0}) END from {rhs}"

    check_query(
        broadcasted_string_query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        equivalent_spark_query=spark_query,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ("SELECT STRCMP(A, B) FROM T", pd.DataFrame({0: [1, 0, 1, 1, -1]})),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT STRCMP(A, 'epsilon') FROM T",
                pd.DataFrame({0: [-1, -1, 1, 1, 0]}),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("SELECT STRCMP('whimsy', 'whimsical') FROM T", pd.DataFrame({0: [1] * 5})),
            id="all_scalar",
        ),
    ],
)
def test_strcmp(args, spark_info, memory_leak_check):
    """Spark doesn't support a BigInt for this function we use an expected output."""

    df = pd.DataFrame(
        {
            "A": ["alpha", "beta", "zeta", "pi", "epsilon"],
            "B": ["", "beta", "zebra", "PI", "foo"],
        }
    )

    query, expected_output = args
    check_query(
        query,
        {"T": df},
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        sort_output=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            ("SELECT RTRIMMED_LENGTH((A)) FROM table1"),
            id="vector-no_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("SELECT CASE WHEN RTRIMMED_LENGTH((A)) > 0 THEN 1 ELSE 0 END FROM table1"),
            id="vector-case",
        ),
        pytest.param(
            (
                "SELECT RTRIMMED_LENGTH(('   Alphabet  Soup Is Delicious   ')) FROM table1"
            ),
            id="scalar-no_case",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN RTRIMMED_LENGTH(('   Alphabet  Soup Is \\tDelicious   ')) > 0 THEN 1 ELSE 0 END FROM table1"
            ),
            id="scalar-case",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_rtrimmed_length(query, spark_info, memory_leak_check):
    whitespace = " " * 8
    chars = "a\tcdef\nh"
    # Generate a column of strings with every combination of 8 characters
    # being space vs non-space
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": [
                    "".join(
                        whitespace[j] if (i >> j) % 2 else chars[j]
                        for j in range(len(chars))
                    )
                    for i in range(256)
                ]
                + [None]
            }
        )
    }

    spark_query = query.replace("RTRIMMED_LENGTH(", "LENGTH(RTRIM")

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        sort_output=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ("SELECT STRCMP(A, B) FROM T", pd.DataFrame({0: [-1, -1, -1, -1, -1]})),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT STRCMP(A, 'epsilon') FROM T",
                pd.DataFrame({0: [1, 1, 1, 1, 1]}),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("SELECT STRCMP('whimsy', 'whimsical') FROM T", pd.DataFrame({0: [1] * 5})),
            id="all_scalar",
        ),
    ],
)
def test_strcmp_nonascii(args, spark_info, memory_leak_check):
    """Spark doesn't support a BigInt for this function we use an expected output."""

    df = pd.DataFrame(
        {
            "A": gen_nonascii_list(5),
            "B": gen_nonascii_list(10)[5:],
        }
    )

    query, expected_output = args
    check_query(
        query,
        {"T": df},
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        sort_output=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT FORMAT(mixed_floats, positive_ints) FROM table1",
                pd.DataFrame(
                    [
                        "0",
                        "0.0",
                        "-0.12",
                        "123.210",
                        "-12,345.0000",
                        "1,234,567,890.12346",
                        "0.098000",
                        "1.2300000",
                    ]
                    * 2
                ),
            ),
            id="FORMAT_all_vector",
        ),
        pytest.param(
            (
                "SELECT FORMAT(12345.6789, 2) FROM table1",
                pd.DataFrame(["12,345.68"] * 16),
            ),
            id="FORMAT_all_scalar",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_format(args, spark_info, bodosql_string_fn_testing_df, memory_leak_check):
    query, refsol = args

    check_query(
        query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_nullable_bodosql=False,
        expected_output=refsol,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT SUBSTRING(source FROM start_pos FOR length) from table1",
            id="SUBSTRING_all_vector",
        ),
        pytest.param(
            "SELECT SUBSTR(source, start_pos, 3) from table1",
            id="SUBSTRING_scalar_int_1A",
        ),
        pytest.param(
            "SELECT MID(source, -2, length) from table1",
            id="SUBSTRING_scalar_int_1B",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT SUBSTRING(source, -5, 3) from table1",
            id="SUBSTRING_scalar_int_2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT SUBSTR('alphabet soup is delicious', start_pos, length) from table1",
            id="SUBSTRING_scalar_str",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT MID('alphabet soup is delicious', 9, 4) from table1",
            id="SUBSTRING_all_scalar",
        ),
        pytest.param(
            "SELECT SUBSTRING_INDEX(source, delim, occur) from table1",
            id="SUBSTRING_INDEX_all_vector",
        ),
        pytest.param(
            "SELECT SUBSTRING_INDEX(source, 'a', occur) from table1",
            id="SUBSTRING_INDEX_scalar_str",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT SUBSTRING_INDEX(source, ' ', 2) from table1",
            id="SUBSTRING_INDEX_scalar_str_scalar_int",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT SUBSTRING_INDEX(source, delim, 3) from table1",
            id="SUBSTRING_INDEX_scalar_int",
        ),
        pytest.param(
            "SELECT SUBSTRING_INDEX('alpha,beta,gamma,delta,epsilon', ',', 3) from table1",
            id="SUBSTRING_INDEX_all_scalar",
        ),
    ],
)
def test_substring(query, spark_info, memory_leak_check):
    subst_df = pd.DataFrame(
        {
            "SOURCE": pd.Series(
                pd.array(
                    [
                        "a bc def ghij",
                        "kl mnopq r",
                        "st uv wx yz",
                        "a e i o u y",
                        "alphabet",
                        "soup",
                        None,
                        "",
                        "Ɨ Ø ƀ",
                        "ǖ ǘ ǚ ǜ",
                        "± × ÷ √",
                        "Ŋ ŋ",
                    ]
                )
            ),
            "START_POS": pd.Series(pd.array([-8, -4, -2, 0, 2, 4, 8, 16])),
            "LENGTH": pd.Series(pd.array([3, 7, 2, 1, -1, 1, 0, None])),
            "DELIM": pd.Series(pd.array([" ", " ", "", "a", "a", "a", "--", "--"])),
            "OCCUR": pd.Series(pd.array([2, 0, 2, 4, 2, -1, 0, None])),
        }
    )
    spark_query = query.replace("MID", "SUBSTR")
    check_query(
        query,
        {"TABLE1": subst_df},
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize("func", ["SUBSTR", "SUBSTRING"])
def test_substring_suffix(func, spark_info, memory_leak_check):
    """Test SUBSTR/SUBSTRING with 2 arguments only where length is optional"""
    query = f"SELECT {func}(S, I) FROM table1"
    df = {
        "TABLE1": pd.DataFrame(
            {
                "S": pd.Series(
                    [
                        "a bc def ghij",
                        "kl mnopq r",
                        "st uv wx yz",
                        "a e i o u y",
                        "alphabet",
                        "soup",
                        None,
                        "",
                        "Ɨ Ø ƀ",
                        "ǖ ǘ ǚ ǜ",
                        "± × ÷ √",
                        "Ŋ ŋ",
                    ]
                ),
                "I": pd.Series(
                    [None, -1, 1, -10, 5, 0, 2, 6, None, 10, -5, 4],
                    dtype=pd.Int32Dtype(),
                ),
            }
        )
    }
    check_query(
        query,
        df,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


def test_length(bodosql_string_types, spark_info, memory_leak_check):
    query = "SELECT LENGTH(A) as OUT1 FROM table1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )

    query1 = "SELECT LEN(A) as OUT1 FROM table1"
    spark_query1 = "SELECT LENGTH(A) as OUT1 FROM table1"
    check_query(
        query1,
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        equivalent_spark_query=spark_query1,
    )


def test_length_binary(bodosql_binary_types, spark_info, memory_leak_check):
    query = "SELECT LENGTH(A) as OUT1 FROM table1"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )

    query1 = "SELECT LEN(A) as OUT1 FROM table1"
    spark_query1 = "SELECT LENGTH(A) as OUT1 FROM table1"
    check_query(
        query1,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        equivalent_spark_query=spark_query1,
    )


@pytest.mark.slow
def test_reverse_binary(bodosql_binary_types, spark_info, memory_leak_check):
    query = "SELECT REVERSE(A) as OUT1 FROM table1"
    expected_output1 = pd.DataFrame(
        {
            "OUT1": [b"cba", b"c", None, b"gfedcc"] * 3,
        }
    )
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=expected_output1,
    )


@pytest.mark.slow
def test_substring_binary(bodosql_binary_types, spark_info, memory_leak_check):
    query = "SELECT SUBSTR(A, 2, 3) as OUT1 FROM table1"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        convert_columns_bytearray=["OUT1"],
    )

    query1 = "SELECT SUBSTRING(A, 2, 3) as OUT1 FROM table1"
    check_query(
        query1,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        convert_columns_string=["OUT1"],
    )

    query2 = (
        "SELECT A, REVERSE(A) as OUT1, SUBSTRING(REVERSE(A), 2, 3) as OUT2 FROM table1"
    )
    check_query(
        query2,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        convert_columns_string=["OUT1", "OUT2"],
    )


@pytest.mark.slow
def test_left_right_binary(bodosql_binary_types, spark_info, memory_leak_check):
    query1 = "SELECT LEFT(B,3) as OUT1, RIGHT(B,3) as OUT2 FROM table1"
    query2 = "SELECT LEFT(A,10) as OUT1, RIGHT(C,10) as OUT2 FROM table1"

    expected_output1 = pd.DataFrame(
        {
            "OUT1": [bytes(3), b"abc", b"iho", None] * 3,
            "OUT2": [bytes(3), b"cde", b"324", None] * 3,
        }
    )
    expected_output2 = pd.DataFrame(
        {
            "OUT1": [b"abc", b"c", None, b"ccdefg"] * 3,
            "OUT2": [None, b"poiu", b"fewfqqqqq", b"3f3"] * 3,
        }
    )
    check_query(
        query1,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=expected_output1,
    )
    check_query(
        query2,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=expected_output2,
    )


def test_lpad_rpad_binary(bodosql_binary_types, spark_info, memory_leak_check):
    query1 = "SELECT LEFT(B,3) as OUT1, RPAD(A, 6, LEFT(B, 3)) as OUT2 FROM table1"
    query2 = "SELECT RIGHT(B,3) as OUT1, LPAD(A, 6, RIGHT(B, 3)) as OUT2 FROM table1"

    expected_output1 = pd.DataFrame(
        {
            "OUT1": [bytes(3), b"abc", b"iho", None] * 3,
            "OUT2": [b"abc" + bytes(3), b"cabcab", None, None] * 3,
        }
    )
    expected_output2 = pd.DataFrame(
        {
            "OUT1": [bytes(3), b"cde", b"324", None] * 3,
            "OUT2": [bytes(3) + b"abc", b"cdecdc", None, None] * 3,
        }
    )

    check_query(
        query1,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=expected_output1,
    )
    check_query(
        query2,
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=expected_output2,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(True, id="CASE", marks=pytest.mark.slow),
        pytest.param(False, id="NO_CASE"),
    ],
)
@pytest.mark.parametrize(
    "startswith, table, answer",
    [
        pytest.param(
            True,
            "STR_TABLE",
            pd.DataFrame({0: [False] * 3 + [True] * 6}),
            id="startswith-strings",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            True,
            "BIN_TABLE",
            pd.DataFrame({0: [False] * 3 + [True] * 6}),
            id="startswith-binary",
        ),
        pytest.param(
            False,
            "STR_TABLE",
            pd.DataFrame(
                {0: [False, True, False, True, False, True, True, False, False]}
            ),
            id="endswith-strings",
        ),
        pytest.param(
            False,
            "BIN_TABLE",
            pd.DataFrame(
                {0: [False, True, False, True, False, True, True, False, False]}
            ),
            id="endswith-binary",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_startswith_endswith(
    startswith, table, answer, use_case, spark_info, memory_leak_check
):
    if use_case:
        query = f"SELECT CASE WHEN {'STARTSWITH' if startswith else 'ENDSWITH'}(A, B) THEN 1 ELSE 0 END FROM {table}"
        answer = pd.DataFrame({0: [int(b) for b in answer[0]]})
    else:
        query = (
            f"SELECT {'STARTSWITH' if startswith else 'ENDSWITH'}(A, B) FROM {table}"
        )
    ctx = {
        "STR_TABLE": pd.DataFrame(
            {
                "A": ["alpha", "alphabet", "alpha beta"] * 3,
                "B": ["bet"] * 3 + ["a"] * 3 + ["alpha"] * 3,
            }
        ),
        "BIN_TABLE": pd.DataFrame(
            {
                "A": [b"alpha", b"alphabet", b"alpha beta"] * 3,
                "B": [b"bet"] * 3 + [b"a"] * 3 + [b"alpha"] * 3,
            }
        ),
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(True, id="CASE", marks=pytest.mark.slow),
        pytest.param(False, id="NO_CASE"),
    ],
)
@pytest.mark.parametrize(
    "calculation, table, answer",
    [
        pytest.param(
            "INSERT(A, B, C, D)",
            "STR_TABLE",
            pd.DataFrame(
                {
                    0: [
                        "the orange fox",
                        "the yellow fox",
                        "the green fox",
                        "the red and blue fox",
                        "the purple and red fox",
                    ]
                    * 2
                }
            ),
            id="replace-strings",
        ),
        pytest.param(
            "INSERT(A, E, F, G)",
            "STR_TABLE",
            pd.DataFrame(
                {
                    0: [
                        "the ",
                        None,
                        "the spotted red fox",
                        "the red fox jumped",
                        None,
                        "I saw the red fox",
                        "red fox",
                        "the fox",
                        None,
                        "the red dog",
                    ],
                }
            ),
            id="inject_delete-strings",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "INSERT(A, B, C, D)",
            "BIN_TABLE",
            pd.DataFrame(
                {
                    0: [
                        b"the orange fox",
                        b"the yellow fox",
                        b"the green fox",
                        b"the red and blue fox",
                        b"the purple and red fox",
                    ]
                    * 2
                }
            ),
            id="replace-binary",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "INSERT(A, E, F, G)",
            "BIN_TABLE",
            pd.DataFrame(
                {
                    0: [
                        b"the ",
                        None,
                        b"the spotted red fox",
                        b"the red fox jumped",
                        None,
                        b"I saw the red fox",
                        b"red fox",
                        b"the fox",
                        None,
                        b"the red dog",
                    ],
                }
            ),
            id="inject_delete-binary",
            marks=pytest.mark.skip("TODO: fix for Pandas 3"),
        ),
    ],
)
def test_insert(calculation, table, answer, case, spark_info, memory_leak_check):
    if case:
        query = f"SELECT CASE WHEN {calculation} IS NULL THEN -1 ELSE LENGTH({calculation}) END FROM {table}"
        answer = pd.DataFrame(
            {0: [-1 if pd.isna(res) else len(res) for res in answer[0]]}
        )
    else:
        query = f"SELECT {calculation} FROM {table}"
    ctx = {
        "STR_TABLE": pd.DataFrame(
            {
                "A": ["the red fox"] * 10,
                "B": [5, 5, 5, 9, 5] * 2,
                "C": [3, 3, 3, 0, 0] * 2,
                "D": ["orange", "yellow", "green", "and blue ", "purple and "] * 2,
                "E": pd.Series(
                    [5, 5, 5, 30, None, 1, 1, 4, 8, 9], dtype=pd.Int32Dtype()
                ),
                "F": pd.Series(
                    [100, None, 0, 0, 1, 0, 4, 4, 5, 3], dtype=pd.Int32Dtype()
                ),
                "G": [
                    "",
                    "yay",
                    "spotted ",
                    " jumped",
                    "foo",
                    "I saw ",
                    "",
                    "",
                    None,
                    "dog",
                ],
            }
        ),
        "BIN_TABLE": pd.DataFrame(
            {
                "A": [b"the red fox"] * 10,
                "B": [5, 5, 5, 9, 5] * 2,
                "C": [3, 3, 3, 0, 0] * 2,
                "D": [b"orange", b"yellow", b"green", b"and blue ", b"purple and "] * 2,
                "E": pd.Series(
                    [5, 5, 5, 30, None, 1, 1, 4, 8, 9], dtype=pd.Int32Dtype()
                ),
                "F": pd.Series(
                    [100, None, 0, 0, 1, 0, 4, 4, 5, 3], dtype=pd.Int32Dtype()
                ),
                "G": [
                    b"",
                    b"yay",
                    b"spotted ",
                    b" jumped",
                    b"foo",
                    b"I saw ",
                    b"",
                    b"",
                    None,
                    b"dog",
                ],
            }
        ),
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="NO_CASE"),
        pytest.param(True, id="CASE", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "table",
    [
        pytest.param("str_table", id="strings"),
        pytest.param(
            "bin_table",
            id="binary",
            marks=pytest.mark.skip("[BE-3717] Support binary find with 3 args"),
        ),
    ],
)
@pytest.mark.parametrize(
    "calculation",
    [
        pytest.param(
            "POSITION(A, B)",
            id="position_normal-2_args",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "POSITION(A IN B)",
            id="position_in-2_args",
        ),
        pytest.param(
            "POSITION(A, B, C)",
            id="position_normal-3_args",
        ),
        pytest.param(
            "CHARINDEX(A, B)",
            id="charindex-2_args",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "CHARINDEX(A, B, C)",
            id="charindex-3_args",
        ),
    ],
)
def test_position(calculation, table, case, spark_info, memory_leak_check):
    if case:
        query = f"SELECT CASE WHEN {calculation} IS NULL THEN -1 ELSE {calculation} END FROM {table}"
    else:
        query = f"SELECT {calculation} FROM {table}"

    # Spark has POSITION but not CHARINDEX, so change the function name for it
    spark_query = query
    spark_query = spark_query.replace("CHARINDEX", "POSITION")

    # For some reason, Spark's POSITION handles nulls weirdly when there are 3
    # arguments so manually outputing null is required
    spark_query = spark_query.replace(
        "POSITION(A, B, C)",
        "CASE WHEN A IS NULL OR B IS NULL OR C IS NULL THEN NULL ELSE POSITION(A, B, C) END",
    )

    ctx = {
        "STR_TABLE": pd.DataFrame(
            {
                "A": [None] + (["a "] * 3 + [" "] * 3 + ["t"] * 3) * 3,
                "B": [None]
                + ["alphabet"] * 9
                + ["alpha beta gamma delta"] * 9
                + ["the quick fox jumped over the lazy dog"] * 9,
                "C": pd.Series([None] + [1, 5, 9] * 9, dtype=pd.Int32Dtype()),
            }
        ),
        "BIN_TABLE": pd.DataFrame(
            {
                "A": [None] + ([b"a "] * 3 + [b" "] * 3 + [b"t"] * 3) * 3,
                "B": [None]
                + [b"alphabet"] * 9
                + [b"alpha beta gamma delta"] * 9
                + [b"the quick fox jumped over the lazy dog"] * 9,
                "C": pd.Series([None] + [1, 5, 9] * 9, dtype=pd.Int32Dtype()),
            }
        ),
    }

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_replace_two_args_scalar(memory_leak_check):
    """Test REPLACE works correctly with two scalar inputs"""
    query = "SELECT REPLACE('string', 'in')"
    check_query(
        query,
        {},
        None,
        check_names=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame({"A": ["strg"]}),
    )


@pytest.mark.slow
def test_replace_two_args_column(memory_leak_check):
    """Test REPLACE works correctly with two column inputs"""
    query = "SELECT REPLACE(A, B) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": ["abcabcabc", None, "kbykujdt", "no replace", "zzyyxxxzy"] * 4,
                "B": ["abc", "oiu", None, "none", "zy"] * 4,
            }
        )
    }
    expected_output = pd.DataFrame({"A": ["", None, None, "no replace", "zyxxx"] * 4})
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "query, output",
    [
        pytest.param(
            "SELECT SHA2('Default digest size is 256')",
            "dac91c24b4686ce55713b01c5a21fff1cbce74db3a3e3feee36231471b13f96c",
            id="default_digest_size",
        ),
        pytest.param(
            "SELECT SHA2_HEX('Test SHA224 result', 224)",
            "44d92c6ef8c8c89f721f8a4b63ab59caa3f267cfb2c111b576b7f221",
            id="224",
        ),
        pytest.param(
            "SELECT SHA2('Two arguments SHA256 test case', 256)",
            "2e898ab336709daee16584ea08d8e75c14502d169e16584243b76ee6ddfaf5ca",
            id="256",
        ),
        pytest.param(
            "SELECT SHA2_HEX('fkghvjjgiuglj', 384)",
            "2e472a1acd0f33c7707fdfcfc3dea65169e2962a0fb0e62c5d59c67b08314e58c5fe9698e37c8e1360174c8db4dbf6b4",
            id="384",
        ),
        pytest.param(
            "SELECT SHA2('*&IUYHKJB^TUYFD', 512)",
            "581f65968e1cdaffe0603f394efcb33ae91c37c1d8da85b4491c24e476cdcd2d"
            "50a219c5f6f46eaf753d32e79b58e2cd6776d12f6c18f6d43745b4a4eadab379",
            id="512",
        ),
    ],
)
def test_sha2_scalars(query, output, memory_leak_check):
    """Test SHA2 and SHA2_HEX work correctly with scalar inputs"""
    check_query(
        query,
        {},
        None,
        check_names=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame({"A": pd.Series([output])}),
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT SHA2(A, 256) FROM table1",
            id="256",
        ),
        pytest.param(
            "SELECT CASE WHEN A IS NULL THEN NULL ELSE SHA2_HEX(A, 256) END FROM table1",
            id="256_case",
        ),
        pytest.param(
            "SELECT SHA2(B, 256) FROM table1",
            id="256_binary",
        ),
    ],
)
def test_sha2_columns(query, memory_leak_check):
    """Test SHA2 and SHA2_HEX work correctly with column inputs"""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": ["abcabcabc", None, "kbykujdt", "no replace", "zzyyxxxzy"] * 4,
                "B": [b"abcabcabc", None, b"kbykujdt", b"no replace", b"zzyyxxxzy"] * 4,
            }
        )
    }
    output = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    "76b99ab4be8521d78b19bcff7d1078aabeb477bd134f404094c92cd39f051c3e",
                    None,
                    "90babc4e5405e215a4bbdfaf13def1687ef0f00ad152705250890400b7a097a3",
                    "0dc95e29e0583513cb4a75409bcdf9cee72eb647d0ee7f43fa47832b4efd4c23",
                    "b3af10a334d34a4c6dee44530efd39a1b2564b443d5e617d209cb8921c642f7e",
                ]
                * 4,
            )
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "query, output",
    [
        pytest.param(
            "SELECT MD5('String to be MD5 encrypted')",
            "6f8c044a4e850a2710dfb65fc77e9665",
            id="MD5",
        ),
        pytest.param(
            "SELECT MD5_HEX('Test MD5_HEX')",
            "cb3c5742480f5e49edd014f103ac2679",
            id="MD5_HEX",
        ),
    ],
)
def test_md5_scalars(query, output, memory_leak_check):
    """Test MD5 and MD5_HEX work correctly with scalar inputs"""
    check_query(
        query,
        {},
        None,
        check_names=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame({"A": pd.Series([output])}),
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT MD5(A) FROM table1",
            id="no_case",
        ),
        pytest.param(
            "SELECT CASE WHEN A IS NULL THEN NULL ELSE MD5(A) END FROM table1",
            id="with_case",
        ),
        pytest.param(
            "SELECT MD5(B) FROM table1",
            id="binary",
        ),
    ],
)
def test_md5_columns(query, memory_leak_check):
    """Test MD5 and MD5_HEX work correctly with column inputs"""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": ["bcdbcdbcd", "CVMDKAJDS", None, "no replace", ":@#E?><"] * 4,
                "B": [b"bcdbcdbcd", b"CVMDKAJDS", None, b"no replace", b":@#E?><"] * 4,
            }
        )
    }
    output = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    "b533ce74b4f2b0dfde46e0f8bb35e4c9",
                    "52b6fbd8a16092d4e96bd9af00e8a7bf",
                    None,
                    "ea168808cd7e976473706fd1ec902b6f",
                    "e68c0610abfc1dd0f5fde151e0c7ee35",
                ]
                * 4,
            )
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "col_fmt",
    [
        pytest.param("S", id="string"),
        pytest.param("B", id="binary", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "query_fmt, uppercase",
    [
        pytest.param(
            "SELECT HEX_ENCODE({0}) FROM table1",
            True,
            id="default_uppercase-no_case",
        ),
        pytest.param(
            "SELECT HEX_ENCODE({0}, 1) FROM table1",
            True,
            id="manual_uppercase-no_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT CASE WHEN D THEN HEX_ENCODE({0}, 0) ELSE NULL END FROM table1",
            False,
            id="manual_lowercase-with_case",
        ),
    ],
)
def test_hex_encode_decode(query_fmt, uppercase, col_fmt, memory_leak_check):
    """Tests HEX_ENCODE and the various HEX_DECODE functions with and without CASE statements"""
    enc_query = query_fmt.format(col_fmt)
    if "CASE" in query_fmt:
        dec_str_query = (
            "SELECT CASE WHEN D THEN HEX_DECODE_STRING(H) ELSE NULL END FROM table1"
        )
        dec_bin_query = (
            "SELECT CASE WHEN D THEN HEX_DECODE_BINARY(H) ELSE NULL END FROM table1"
        )
    else:
        dec_str_query = "SELECT HEX_DECODE_STRING(H) FROM table1"
        dec_bin_query = "SELECT HEX_DECODE_BINARY(H) FROM table1"
    s = "Alphabet Soup"
    b = b"Alphabet Soup"
    h = "416c70686162657420536f7570"
    if uppercase:
        h = h.upper()
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "D": [True] * 14,
                "S": [None if i % 5 == 2 else s[:i] for i in range(14)],
                "B": [None if i % 5 == 2 else b[:i] for i in range(14)],
                "H": [None if i % 5 == 2 else h[: 2 * i] for i in range(14)],
            }
        )
    }
    check_query(
        enc_query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame({0: ctx["TABLE1"]["H"]}),
    )
    check_query(
        dec_str_query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame({0: ctx["TABLE1"]["S"]}),
    )
    check_query(
        dec_bin_query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame({0: ctx["TABLE1"]["B"]}),
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="only_jit_seq=True disables spawn testing so pytest.raises fails",
)
@pytest_mark_one_rank
@pytest.mark.parametrize(
    "func",
    [
        pytest.param("HEX_DECODE_STRING"),
        pytest.param("TRY_HEX_DECODE_STRING"),
        pytest.param("HEX_DECODE_BINARY", marks=pytest.mark.slow),
        pytest.param("TRY_HEX_DECODE_BINARY", marks=pytest.mark.slow),
    ],
)
def test_hex_decode_error(func):
    """Tests the HEX_DECODE family of functions with invalid strings to see
    how well they handle decoding errors"""

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "S": pd.Series(
                    [
                        "",
                        "4",  # Odd number of characters
                        "46",
                        "466",  # Odd number of characters
                        "466f",
                        "466f6",  # Odd number of characters
                        "466f6f",
                        "ABCDEG",  # Non-hex character: 'G'
                        "",
                        "12#4",  # Non-hex character: '#'
                        "",
                        "F🐍",  # Non-hex character: '🐍'
                        "",
                    ]
                ),
            }
        )
    }

    query = f"SELECT {func}(S) FROM table1"
    result = pd.Series(
        ["", None, "F", None, "Fo", None, "Foo", None, "", None, "", None, ""]
    )

    if func.startswith("TRY"):
        if func.endswith("BINARY"):
            result = result.apply(
                lambda x: None if pd.isna(x) else bytes(x, encoding="utf-8")
            )
        check_query(
            query,
            ctx,
            None,
            check_names=False,
            expected_output=pd.DataFrame({0: result}),
            only_jit_seq=True,
        )
    else:
        with pytest.raises(
            ValueError,
            match=f"{func} failed due to malformed string input",
        ):
            check_query(
                query,
                ctx,
                None,
                check_names=False,
                # Pointless output, but must be set
                expected_output=pd.DataFrame(),
                only_jit_seq=True,
            )


@pytest.mark.parametrize(
    "col_fmt",
    [
        pytest.param("S", id="string"),
        pytest.param("B", id="binary", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "query_fmt, answer",
    [
        pytest.param(
            "SELECT BASE64_ENCODE({0}) FROM table1",
            pd.Series(
                [
                    "Uw==",
                    "U24=",
                    "U25v",
                    None,
                    "U25vdw==",
                    "U25vd2Y=",
                    "U25vd2Zs",
                    None,
                    "U25vd2ZsYQ==",
                    "U25vd2ZsYWs=",
                    "U25vd2ZsYWtl",
                    "QUI/Q0Q+RUYvR0g8",
                ]
            ),
            id="encode-no_extra_args-no_case",
        ),
        pytest.param(
            "SELECT BASE64_ENCODE({0}, 5) FROM table1",
            pd.Series(
                [
                    "Uw==",
                    "U24=",
                    "U25v",
                    None,
                    "U25vd\nw==",
                    "U25vd\n2Y=",
                    "U25vd\n2Zs",
                    None,
                    "U25vd\n2ZsYQ\n==",
                    "U25vd\n2ZsYW\ns=",
                    "U25vd\n2ZsYW\ntl",
                    "QUI/Q\n0Q+RU\nYvR0g\n8",
                ]
            ),
            id="encode-limit_5-no_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT BASE64_ENCODE({0}, 0, '!@ ') FROM table1",
            pd.Series(
                [
                    "Uw  ",
                    "U24 ",
                    "U25v",
                    None,
                    "U25vdw  ",
                    "U25vd2Y ",
                    "U25vd2Zs",
                    None,
                    "U25vd2ZsYQ  ",
                    "U25vd2ZsYWs ",
                    "U25vd2ZsYWtl",
                    "QUI@Q0Q!RUYvR0g8",
                ]
            ),
            id="encode-no_limit-replace_alphabet-no_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT CASE WHEN S IS NULL THEN '' ELSE BASE64_ENCODE({0}) END FROM table1",
            pd.Series(
                [
                    "Uw==",
                    "U24=",
                    "U25v",
                    "",
                    "U25vdw==",
                    "U25vd2Y=",
                    "U25vd2Zs",
                    "",
                    "U25vd2ZsYQ==",
                    "U25vd2ZsYWs=",
                    "U25vd2ZsYWtl",
                    "QUI/Q0Q+RUYvR0g8",
                ]
            ),
            id="encode-no_extra_args-with_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT BASE64_DECODE_STRING(BASE64_ENCODE({0})) FROM table1",
            pd.Series(
                [
                    "S",
                    "Sn",
                    "Sno",
                    None,
                    "Snow",
                    "Snowf",
                    "Snowfl",
                    None,
                    "Snowfla",
                    "Snowflak",
                    "Snowflake",
                    "AB?CD>EF/GH<",
                ]
            ),
            id="decode_string-no_extra_args-no_case",
        ),
        pytest.param(
            "SELECT BASE64_DECODE_STRING(BASE64_ENCODE({0}, 3, '().'), '().') FROM table1",
            pd.Series(
                [
                    "S",
                    "Sn",
                    "Sno",
                    None,
                    "Snow",
                    "Snowf",
                    "Snowfl",
                    None,
                    "Snowfla",
                    "Snowflak",
                    "Snowflake",
                    "AB?CD>EF/GH<",
                ]
            ),
            id="decode_string-limit_3-replace_alphabet-no_case",
        ),
        pytest.param(
            "SELECT BASE64_DECODE_BINARY(BASE64_ENCODE({0}, 3, '().'), '().') FROM table1",
            pd.Series(
                [
                    b"S",
                    b"Sn",
                    b"Sno",
                    None,
                    b"Snow",
                    b"Snowf",
                    b"Snowfl",
                    None,
                    b"Snowfla",
                    b"Snowflak",
                    b"Snowflake",
                    b"AB?CD>EF/GH<",
                ]
            ),
            id="decode_binary-limit_3-replace_alphabet-no_case",
        ),
    ],
)
def test_base64_encode_decode(query_fmt, answer, col_fmt, memory_leak_check):
    """Tests BASE64_ENCODE and BASE64_DECODE_STRING with and without CASE statements"""
    # Note: "AB?CD>EF/GH<" is important as it forces the usage of the 62 & 63 characters
    query = query_fmt.format(col_fmt)
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "S": [
                    "S",
                    "Sn",
                    "Sno",
                    None,
                    "Snow",
                    "Snowf",
                    "Snowfl",
                    None,
                    "Snowfla",
                    "Snowflak",
                    "Snowflake",
                    "AB?CD>EF/GH<",
                ],
                "B": [
                    b"S",
                    b"Sn",
                    b"Sno",
                    None,
                    b"Snow",
                    b"Snowf",
                    b"Snowfl",
                    None,
                    b"Snowfla",
                    b"Snowflak",
                    b"Snowflake",
                    b"AB?CD>EF/GH<",
                ],
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=pd.DataFrame({0: answer}),
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="only_jit_seq=True disables spawn testing so pytest.raises fails",
)
@pytest_mark_one_rank
@pytest.mark.parametrize(
    "func",
    [
        pytest.param("BASE64_DECODE_STRING"),
        pytest.param("TRY_BASE64_DECODE_STRING"),
        pytest.param("BASE64_DECODE_BINARY", marks=pytest.mark.slow),
        pytest.param("TRY_BASE64_DECODE_BINARY", marks=pytest.mark.slow),
    ],
)
def test_base64_decode_error(func):
    """Tests the BASE64_DECODE family of functions with invalid strings to see
    how well they handle decoding errors"""

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "S": pd.Series(
                    [
                        "",
                        "Uw==",
                        "U24=",
                        "U25v",
                        None,
                        "U25vdw==",
                        "U25vd2Y=",
                        "U25vd2Zs",
                        "U25vd2ZsYQ==",
                        "U25vd2ZsYWs=",
                        "U25vd2ZsYWtl",
                        "QUI/Q0Q+RUYvR0g8",
                    ]
                ),
                "T": pd.Series(
                    [
                        "",
                        "U",
                        "Uw",
                        "Uwe",
                        "Uw==",
                        "U25vd",
                        "U25vdw",
                        "U25vdw=",
                        "U25vdw==",
                        "",
                        "",
                        "",
                    ]
                ),
                "U": pd.Series(
                    ["U25v", "====", "U25v", "U===", "U25v", "=U5vd2Y="] * 2
                ),
            }
        )
    }

    # First type of invalid situation: the padding character is wrong
    query_pad_error = f"SELECT {func}(S, '+/.') FROM table1"
    original_strings_pad_error = pd.Series(
        [
            "",
            None,
            None,
            "Sno",
            None,
            None,
            None,
            "Snowfl",
            None,
            None,
            "Snowflake",
            "AB?CD>EF/GH<",
        ]
    )

    # Second type of invalid situation: the index 62 character is wrong
    query_62_error = f"SELECT {func}(S, '%') FROM table1"
    original_strings_62_error = pd.Series(
        [
            "",
            "S",
            "Sn",
            "Sno",
            None,
            "Snow",
            "Snowf",
            "Snowfl",
            "Snowfla",
            "Snowflak",
            "Snowflake",
            None,
        ]
    )

    # Third type of invalid situation: not all strings are 4 characters
    query_length_error = f"SELECT {func}(T) FROM table1"
    original_strings_length_error = pd.Series(
        [
            "",
            None,
            None,
            None,
            "S",
            None,
            None,
            None,
            "Snow",
            "",
            "",
            "",
        ],
    )

    # Fourth type of invalid situation: invalid padding characters
    query_pad_location_error = f"SELECT {func}(U) FROM table1"
    original_strings_pad_location_error = pd.Series(["Sno", None] * 6)

    combinations = [
        (query_pad_error, original_strings_pad_error),
        (query_62_error, original_strings_62_error),
        (query_length_error, original_strings_length_error),
        (query_pad_location_error, original_strings_pad_location_error),
    ]

    for query, answer in combinations:
        if func.startswith("TRY"):
            if func.endswith("BINARY"):
                answer = answer.apply(
                    lambda x: None if pd.isna(x) else bytes(x, encoding="utf-8")
                )
            check_query(
                query,
                ctx,
                None,
                check_names=False,
                expected_output=pd.DataFrame({0: answer}),
                only_jit_seq=True,
            )
        else:
            with pytest.raises(
                ValueError,
                match=f"{func} failed due to malformed string input",
            ):
                check_query(
                    query,
                    ctx,
                    None,
                    check_names=False,
                    # Pointless output, but must be set
                    expected_output=pd.DataFrame(),
                    only_jit_seq=True,
                )


@pytest.mark.slow
@pytest.mark.parametrize("func", ["LPAD", "RPAD"])
def test_binary_pad_2args_errorchecking(func, memory_leak_check):
    """
    Test error message is thrown when
    LPAD/RPAD is used with binary data and 2 arguments.
    """

    query = f"SELECT {func}(A, len) FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, dtype=object),
                "LEN": pd.Series(
                    [None, -1, 10, 0] * 3,
                    dtype=pd.Int32Dtype(),
                ),
            }
        )
    }
    with pytest.raises(
        Exception,
        match=f".*Cannot apply '{func.upper()}' to arguments of type '{func.upper()}\\(<VARBINARY\\(.*\\)>, <INTEGER>\\).*'",
    ):
        check_query(
            query,
            ctx,
            None,
            # Pointless output, but must be set
            expected_output=pd.DataFrame(),
        )


def test_jarowinkler_similarity(datapath, memory_leak_check):
    """
    Tests the correctness of the function JAROWINKLER_SIMILARITY on a larger dataset
    generated via Snowflake.

    The data was produced from the following Snowflake query:

    SELECT
        translate(p1.p_name, 'aeiou', '') as A,
        Initcap(translate(p2.p_name, 'aiu', '.')) as B,
        JAROWINKLER_SIMILARITY(A, B) AS J
    FROM TPCH_SF1.part p1, TPCH_SF1.part p2
    WHERE
        p1.p_brand = p2.p_brand
        AND p1.p_size = p2.p_size
        AND p1.p_container = p2.p_container
        AND CEIL(p1.p_retailprice) % 4 = 0
        AND CEIL(p2.p_retailprice) % 5 = 0

    The produced csv data has 49,766 rows in the following format:

    sddl zr stl mgnt drk,Ch.Rtrese Bl.Nched Or.Nge Vory Nd.N,51
    pr mtllc lmnd snn sddl,Per Met.Llc .Lmond Senn. S.Ddle,91
    pr mtllc lmnd snn sddl,Cre.M Goldenrod Pff S.Ddle Bl.Ck,60
    """

    jw_data = pd.read_csv(datapath("jaro_winkler_data.csv"))
    ctx = {"TABLE1": jw_data[["A", "B"]]}
    query = "SELECT A, B, JAROWINKLER_SIMILARITY(A, B) FROM table1"

    check_query(
        query, ctx, None, check_dtype=False, check_names=False, expected_output=jw_data
    )


def test_uuid_string_niladic(memory_leak_check):
    """Check that every UUID is unique"""

    query = (
        "SELECT COUNT(DISTINCT uuid) from (SELECT UUID_STRING() as uuid from table1)"
    )
    ctx = {"TABLE1": pd.DataFrame({"a": pd.Series([0] * 10)})}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({0: [10]}),
        is_out_distributed=False,
    )


def test_uuid_string_with_arguments(memory_leak_check):
    """Check that every UUID generated matches python's uuid5"""

    query = "SELECT UUID_STRING(NS, N) as uuid from table1"

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "NS": pd.Series(
                    [
                        str(x)
                        for x in [
                            uuid.NAMESPACE_DNS,
                            uuid.NAMESPACE_URL,
                            uuid.NAMESPACE_OID,
                            uuid.NAMESPACE_X500,
                            uuid.NAMESPACE_DNS,
                            uuid.NAMESPACE_DNS,
                            uuid.NAMESPACE_DNS,
                        ]
                    ]
                ),
                "N": pd.Series(["foo", "bar", "baz", "qux", "foo", "foo0", "foo1"]),
            }
        )
    }

    answer = pd.Series(
        [
            str(uuid.uuid5(uuid.UUID(ns), n))
            for (ns, n) in zip(ctx["TABLE1"]["NS"], ctx["TABLE1"]["N"])
        ]
    )
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({0: answer}),
    )
