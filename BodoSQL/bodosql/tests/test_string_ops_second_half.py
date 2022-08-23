import pandas as pd
import pytest
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query


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
