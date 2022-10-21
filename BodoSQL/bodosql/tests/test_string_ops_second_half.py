import pandas as pd
import pytest
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query

from bodo.tests.utils import gen_nonascii_list


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
        {"table1": df},
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
    # Technically, in MYSQL, it's valid to pass only one arg to Concat
    # However, defining a function that takes at least 1 string arguement seems to
    # cause validateQuery in the RelationalAlgebraGenerator to throw an index out of
    # bounds error and I don't think calling Concat on one string is a common use case
    query = "select CONCAT(A, B, 'scalar', C) from table1"
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

    # Trim fn's not supported on columns. see BE-965
    if bodo_fn_name in {"LTRIM", "RTRIM", "TRIM"}:
        return

    check_query(
        query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_names=False,
        check_dtype=False,
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

    Also, each query type is tagged with a skipif attatched to the corresponding
    Bodo mini-release version required for the array kernel to exist. The
    query types are inferred by the naming scheme of the ids: "TYPE_other_info"
    """
    queries = [
        (
            "SELECT LPAD(strings_null_1, mixed_ints_null, strings_null_2) from table1",
            "LPAD_all_vector",
            False,
        ),
        (
            "SELECT LPAD(strings_null_1, mixed_ints_null, ' ') from table1",
            "LPAD_scalar_str",
            False,
        ),
        (
            "SELECT LPAD(strings_null_1, 20, strings_null_2) from table1",
            "LPAD_scalar_int",
            True,
        ),
        ("SELECT LPAD('A', 25, ' ') from table1", "LPAD_all_scalar", False),
        (
            "SELECT RPAD(strings_null_1, mixed_ints_null, strings_null_2) from table1",
            "RPAD_all_vector",
            False,
        ),
        (
            "SELECT RPAD(strings_null_1, mixed_ints_null, 'ABC') from table1",
            "RPAD_scalar_str",
            True,
        ),
        (
            "SELECT RPAD(strings_null_1, 25, strings_null_2) from table1",
            "RPAD_scalar_int",
            True,
        ),
        (
            "SELECT RPAD('words', 25, strings_null_2) from table1",
            "RPAD_two_scalar",
            True,
        ),
        ("SELECT RPAD('B', 20, '_$*') from table1", "RPAD_all_scalar", True),
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
        ),
    ],
)
def test_rtrimmed_length(query, spark_info, memory_leak_check):
    whitespace = " " * 8
    chars = "a\tcdef\nh"
    # Generate a column of strings with every combination of 8 characters
    # being space vs non-space
    ctx = {
        "table1": pd.DataFrame(
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
        ),
    ],
)
def test_format(args, spark_info, bodosql_string_fn_testing_df):
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
            "source": pd.Series(
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
            "start_pos": pd.Series(pd.array([-8, -4, -2, 0, 2, 4, 8, 16])),
            "length": pd.Series(pd.array([3, 7, 2, 1, -1, 1, 0, None])),
            "delim": pd.Series(pd.array([" ", " ", "", "a", "a", "a", "--", "--"])),
            "occur": pd.Series(pd.array([2, 0, 2, 4, 2, -1, 0, None])),
        }
    )
    spark_query = query.replace("MID", "SUBSTR")
    check_query(
        query,
        {"table1": subst_df},
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
        equivalent_spark_query=spark_query,
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
            "str_table",
            pd.DataFrame({0: [False] * 3 + [True] * 6}),
            id="startswith-strings",
        ),
        pytest.param(
            True,
            "bin_table",
            pd.DataFrame({0: [False] * 3 + [True] * 6}),
            id="startswith-binary",
        ),
        pytest.param(
            False,
            "str_table",
            pd.DataFrame(
                {0: [False, True, False, True, False, True, True, False, False]}
            ),
            id="endswith-strings",
        ),
        pytest.param(
            False,
            "bin_table",
            pd.DataFrame(
                {0: [False, True, False, True, False, True, True, False, False]}
            ),
            id="endswith-binary",
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
        "str_table": pd.DataFrame(
            {
                "A": ["alpha", "alphabet", "alpha beta"] * 3,
                "B": ["bet"] * 3 + ["a"] * 3 + ["alpha"] * 3,
            }
        ),
        "bin_table": pd.DataFrame(
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


# TODO: test with negatives once that behavior is properly defined ([BE-3719])
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
            "str_table",
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
            "str_table",
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
        ),
        pytest.param(
            "INSERT(A, B, C, D)",
            "bin_table",
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
        ),
        pytest.param(
            "INSERT(A, E, F, G)",
            "bin_table",
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
        "str_table": pd.DataFrame(
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
        "bin_table": pd.DataFrame(
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
            marks=pytest.mark.skip("[BE-3718] Support POSITION without 'IN' syntax"),
        ),
        pytest.param(
            "POSITION(A IN B)",
            id="position_in-2_args",
        ),
        pytest.param(
            "POSITION(A, B, C)",
            id="position_normal-3_args",
            marks=pytest.mark.skip("[BE-3718] Support POSITION without 'IN' syntax"),
        ),
        pytest.param(
            "CHARINDEX(A, B)",
            id="charindex-2_args",
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
        "str_table": pd.DataFrame(
            {
                "A": [None] + (["a "] * 3 + [" "] * 3 + ["t"] * 3) * 3,
                "B": [None]
                + ["alphabet"] * 9
                + ["alpha beta gamma delta"] * 9
                + ["the quick fox jumped over the lazy dog"] * 9,
                "C": pd.Series([None] + [1, 5, 9] * 9, dtype=pd.Int32Dtype()),
            }
        ),
        "bin_table": pd.DataFrame(
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
