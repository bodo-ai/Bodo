"""
Test correctness of SQL comparison operations on BodoSQL
"""

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodo.types import Time
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param("BETWEEN", id="between"),
        pytest.param("NOT BETWEEN", marks=pytest.mark.slow, id="not_between"),
    ]
)
def between_clause(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            ([None, 1, 2, 4, 8, 127, 128, 255, 0], pd.UInt8Dtype()), id="uint8"
        ),
        pytest.param(([None, 1, 2, 4, 8, 127, -128, -1, 0], pd.Int8Dtype()), id="int8"),
        pytest.param(
            ([None, 1, 2, 65535, 32767, 127, 128, 255, 0], pd.UInt16Dtype()),
            id="uint16",
        ),
        pytest.param(
            ([None, 1, 2, -32768, 32767, 127, -128, -1, 0], pd.Int16Dtype()), id="int16"
        ),
        pytest.param(
            ([None, 1, 7, 4, 8, 4294967295, 1234567890, 255, 0], pd.UInt32Dtype()),
            id="uint32",
        ),
        pytest.param(
            ([None, 1, 2, 13, 8, 2147483647, -2147483647, -1, 0], pd.Int32Dtype()),
            id="int32",
        ),
        pytest.param(
            (
                [
                    None,
                    9,
                    256,
                    65535,
                    32767,
                    184467440515,
                    9223368477807,
                    255,
                    3,
                ],
                pd.UInt64Dtype(),
            ),
            id="uint64",
        ),
        pytest.param(
            (
                [
                    None,
                    1,
                    4,
                    -32768,
                    32767,
                    9223378575807,
                    -9223372035808,
                    -1,
                    -100,
                ],
                pd.Int64Dtype(),
            ),
            id="int64",
        ),
        pytest.param(
            (
                [
                    None,
                    "",
                    "alpha",
                    "alphabet",
                    "beta",
                    "ZEBRA",
                    "40",
                    "123",
                    "ÁÖ¬⅐♫",
                    " ",
                    "\t",
                ],
                None,
            ),
            id="strings",
        ),
        pytest.param(
            (
                [
                    None,
                    b"",
                    b"alpha",
                    b"alphabet",
                    b"beta",
                    b"ZEBRA",
                    b"40",
                    b"123",
                    b"zebra ",
                    b"\t",
                ],
                pd.ArrowDtype(pa.large_binary()),
            ),
            id="binary",
        ),
        pytest.param(
            (
                [
                    None,
                    datetime.date(1999, 12, 31),
                    datetime.date(2019, 7, 4),
                    datetime.date(2022, 2, 6),
                    datetime.date(2022, 3, 4),
                    datetime.date(2022, 3, 14),
                ],
                pd.ArrowDtype(pa.date32()),
            ),
            id="date",
        ),
        pytest.param(
            (
                [
                    None,
                    pd.Timestamp("2000-1-1"),
                    pd.Timestamp("2010-8-3"),
                    pd.Timestamp("2023-1-1"),
                    pd.Timestamp("2023-4-1"),
                    pd.Timestamp("2000-1-2"),
                ],
                pd.ArrowDtype(pa.timestamp("ns")),
            ),
            id="timestamp",
        ),
    ]
)
def comparison_df(request):
    """Creates a DataFrame from a list of distinct values of a certain type with
    two columns ensuring that the span of the rows is equivalent to a
    cartesian product of the original list with itself. I.e.:
    [42, 16, -1] would produce the following DataFrame:

         A   B
     0  42  42
     1  16  42
     2  -1  42
     3  42  16
     4  16  16
     5  -1  16
     6  42  -1
     7  16  -1
     8  -1  -1

     This creats an ideal table for testing comparison operators.
    """
    data, dtype = request.param
    A = pd.Series(data * len(data), dtype=dtype)
    b = []
    for elem in data:
        b += [elem] * len(data)
    B = pd.Series(b, dtype=dtype)
    return {"TABLE1": pd.DataFrame({"A": A, "B": B})}


@pytest.fixture(
    params=[
        pytest.param(("=", False), id="=-no_case"),
        pytest.param(("<>", False), id="<>-no_case"),
        pytest.param(("!=", False), id="!=-no_case", marks=pytest.mark.slow),
        pytest.param(("<=", False), id="<=-no_case"),
        pytest.param((">=", False), id=">=-no_case", marks=pytest.mark.slow),
        pytest.param(("<", False), id="<-no_case", marks=pytest.mark.slow),
        pytest.param((">", False), id=">-no_case"),
        pytest.param(("<=>", False), id="<=>-no_case"),
        pytest.param(("=", True), id="=-with_case"),
        pytest.param(("<>", True), id="<>-with_case", marks=pytest.mark.slow),
        pytest.param(("!=", True), id="!=-with_case"),
        pytest.param(("<=", True), id="<=-with_case", marks=pytest.mark.slow),
        pytest.param((">=", True), id=">=-with_case", marks=pytest.mark.slow),
        pytest.param(("<", True), id="<-with_case", marks=pytest.mark.slow),
        pytest.param((">", True), id=">-with_case", marks=pytest.mark.slow),
        pytest.param(("<=>", True), id="<=>-with_case"),
    ],
)
def comparison_query_args(request):
    return request.param


def test_comparison_operators_within_table(
    comparison_df,
    comparison_query_args,
    memory_leak_check,
):
    cmp_op, use_case = comparison_query_args
    if use_case:
        query = f"SELECT A, B, \
                        CASE WHEN (A {cmp_op} B) IS NULL THEN 'N' \
                            WHEN (A {cmp_op} B) then 'T' \
                                ELSE 'F' END FROM table1"
    else:
        query = f"SELECT A, B, A {cmp_op} B FROM table1"
    check_query(
        query,
        comparison_df,
        None,
        check_dtype=False,
        check_names=False,
        use_duckdb=True,
    )


@pytest.fixture
def time_comparison_args(comparison_query_args):
    cmp_op, use_case = comparison_query_args
    data = [
        None,
        Time(0, 0, 0, precision=9),
        Time(12, 30, 15, nanosecond=13, precision=9),
        Time(12, 45, 0, precision=9),
        Time(5, 58, 1, microsecond=999, precision=9),
        Time(5, 58, 30, precision=9),
        Time(5, 58, 1, millisecond=1, precision=9),
        Time(22, 14, 20, nanosecond=67, precision=9),
        Time(22, 14, 20, precision=9),
    ]
    A = pd.Series(data * 9)
    b = []
    for elem in data:
        b += [elem] * 9
    B = pd.Series(b)
    ctx = {"TABLE1": pd.DataFrame({"A": A, "B": B})}
    row_funcs = {
        "=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] == x.iloc[1],
        "<>": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] != x.iloc[1],
        "!=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] != x.iloc[1],
        "<": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] < x.iloc[1],
        "<=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] <= x.iloc[1],
        ">": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] > x.iloc[1],
        ">=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0] >= x.iloc[1],
        "<=>": lambda x: True
        if pd.isna(x.iloc[0]) and pd.isna(x.iloc[1])
        else (
            False
            if pd.isna(x.iloc[1]) or pd.isna(x.iloc[1])
            else x.iloc[0] == x.iloc[1]
        ),
    }
    answer = ctx["TABLE1"].apply(row_funcs[cmp_op], axis=1)
    return cmp_op, use_case, ctx, answer


def test_time_comparison_operators_within_table(
    time_comparison_args, memory_leak_check
):
    cmp_op, use_case, ctx, answer = time_comparison_args
    if use_case:
        if cmp_op == "<=>":
            query = f"SELECT A, B, \
                        CASE WHEN (A {cmp_op} B) then 'T' \
                                ELSE 'F' END FROM table1"
        else:
            query = f"SELECT A, B, \
                        CASE WHEN (A {cmp_op} B) IS NULL THEN 'N' \
                            WHEN (A {cmp_op} B) then 'T' \
                                ELSE 'F' END FROM table1"
        answer = answer.apply(lambda x: "N" if pd.isna(x) else ("T" if x else "F"))
        expected_output = pd.DataFrame(
            {"A": ctx["TABLE1"].A, "B": ctx["TABLE1"].B, "C": answer}
        )
    else:
        query = f"SELECT A, B, A {cmp_op} B FROM table1"
        expected_output = pd.DataFrame(
            {"A": ctx["TABLE1"].A, "B": ctx["TABLE1"].B, "C": answer}
        )
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
    )


def test_comparison_operators_interval_within_table(
    bodosql_interval_types, comparison_ops, memory_leak_check
):
    """
    Tests that the basic comparison operators work with Timedelta data within the same table
    """
    query = f"""
        SELECT
            A, C
        FROM
            table1
        WHERE
            A {comparison_ops} C
        """
    # NOTE: this assumes that the input data doesn't require comparing two nulls in <=>
    pd_op = (
        "=="
        if comparison_ops in ("=", "<=>")
        else "!="
        if comparison_ops == "<>"
        else comparison_ops
    )
    expected_output = bodosql_interval_types["TABLE1"].query(f"A {pd_op} C")[["A", "C"]]
    check_query(
        query,
        bodosql_interval_types,
        None,
        check_dtype=False,
        convert_columns_timedelta=["A", "C"],
        expected_output=expected_output,
    )


def test_comparison_operators_between_tables(
    bodosql_numeric_types, comparison_ops, spark_info, memory_leak_check
):
    """
    Tests that the basic comparison operators work when comparing data between two numeric tables of the same type
    """
    if comparison_ops != "=":
        # TODO: Add support for cross-join from general-join cond
        return
    query = f"""
        SELECT
            table1.A, table2.B
        FROM
            table1, table2
        WHERE
            table1.A {comparison_ops} table2.B
        """
    new_context = {
        "TABLE1": bodosql_numeric_types["TABLE1"],
        "TABLE2": bodosql_numeric_types["TABLE1"],
    }
    check_query(query, new_context, spark_info, check_dtype=False)


def test_comparison_operators_decimal(comparison_ops, spark_info, memory_leak_check):
    """Test comparison for decimal values"""

    query = f"""
        SELECT
            A {comparison_ops} B
        FROM
            table1
        """
    context = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.array([1, 4, 0, None, 3, None], "Int64"),
                "B": np.array(
                    [
                        Decimal("0.0"),
                        Decimal("-5.1"),
                        Decimal("1"),
                        None,
                        Decimal("7"),
                        Decimal("-1.71"),
                    ]
                ),
            }
        ),
    }
    check_query(query, context, spark_info, check_dtype=False, check_names=False)
    query = f"""
        SELECT
            B {comparison_ops} A
        FROM
            table1
        """
    check_query(query, context, spark_info, check_dtype=False, check_names=False)
    query = f"""SELECT CASE WHEN A {comparison_ops} B THEN 1 ELSE 0 END FROM table1
    """
    check_query(query, context, spark_info, check_dtype=False, check_names=False)


def test_where_and(join_dataframes, spark_info, memory_leak_check):
    """
    Tests an and expression within a where clause.
    """
    # For join DataFrames, A and B must share a common type across both tables

    if isinstance(join_dataframes["TABLE1"]["A"].values[0], bytes):
        pytest.skip(
            "No support for binary literals: https://bodo.atlassian.net/browse/BE-3304"
        )

    elif isinstance(join_dataframes["TABLE1"]["A"].values[0], str):
        assert isinstance(join_dataframes["TABLE2"]["A"].values[0], str)
        scalar_val1 = "'" + join_dataframes["TABLE1"]["A"].values[0] + "'"
        scalar_val2 = "'" + join_dataframes["TABLE2"]["A"].values[0] + "'"
    else:
        scalar_val1, scalar_val2 = (3, 4)

    query = f"""SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
               WHERE
                 (t1.A = {scalar_val1}
                  and t2.A = {scalar_val2})
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
    )


def test_where_or(join_dataframes, spark_info, memory_leak_check):
    """
    Tests an or expression within a where clause.
    """
    # For join DataFrames, A and B must share a common type across both tables
    if isinstance(join_dataframes["TABLE1"]["A"].values[0], bytes):
        pytest.skip(
            "No support for binary literals: https://bodo.atlassian.net/browse/BE-3304"
        )
    elif isinstance(join_dataframes["TABLE1"]["A"].values[0], str):
        assert isinstance(join_dataframes["TABLE2"]["A"].values[0], str)
        scalar_val1 = "'" + join_dataframes["TABLE1"]["A"].values[0] + "'"
        scalar_val2 = "'" + join_dataframes["TABLE2"]["A"].values[-1] + "'"
    else:
        scalar_val1, scalar_val2 = (3, 4)

    if any(
        isinstance(x, pd.core.arrays.integer.IntegerDtype)
        for x in join_dataframes["TABLE1"].dtypes
    ):
        check_dtype = False
    else:
        check_dtype = True
    query = f"""SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
               WHERE
                 (t1.A = {scalar_val1}
                  or t2.A = {scalar_val2})
    """
    check_query(
        query, join_dataframes, spark_info, check_names=False, check_dtype=check_dtype
    )


def test_between_date(spark_info, between_clause, memory_leak_check):
    query = f"""SELECT A {between_clause} DATE '1995-01-01'
                 AND DATE '1996-12-31' FROM table1"""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": [
                    np.datetime64("1996-12-31"),
                    np.datetime64("1995-01-01"),
                    np.datetime64("1996-01-01"),
                    np.datetime64("1997-01-01"),
                ]
                * 3,
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_between_interval(bodosql_interval_types, between_clause, memory_leak_check):
    """
    tests that between works for interval values
    """
    query = f"""
        SELECT
            A, B, C
        FROM
            table1
        WHERE
            table1.A {between_clause} Interval 1 SECOND AND Interval 1 DAY
    """
    df = bodosql_interval_types["TABLE1"]
    filter_rows = (df.A > datetime.timedelta(seconds=1)) & (
        df.A < datetime.timedelta(days=1)
    )
    if between_clause == "NOT BETWEEN":
        filter_rows = ~filter_rows
    expected_output = df[filter_rows]

    check_query(
        query,
        bodosql_interval_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_between_int(
    bodosql_numeric_types, between_clause, spark_info, memory_leak_check
):
    """
    tests that between works for integer values
    """
    query = f"""
        SELECT
            table1.A
        FROM
            table1
        WHERE
            table1.A {between_clause} 1 AND 3
    """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
    )


def test_between_str(
    bodosql_string_types, between_clause, spark_info, memory_leak_check
):
    """
    tests that between works for string values
    """
    query = f"""
        SELECT
            *
        FROM
            table1
        WHERE
            table1.A {between_clause} 'a' AND 'z'
    """

    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


@pytest.fixture
def date_datetime64_comparison_args(comparison_query_args):
    cmp_op, use_case = comparison_query_args
    data = [
        None,
        datetime.date(1999, 12, 31),
        datetime.date(2019, 7, 4),
        datetime.date(2022, 2, 6),
        datetime.date(2022, 3, 4),
        datetime.date(2022, 3, 14),
    ]
    A = pd.Series(data * 6)
    b = []
    for elem in data:
        b += [np.datetime64(elem)] * 6
    B = pd.Series(b)
    if use_case:
        ctx = {"TABLE1": pd.DataFrame({"B": B, "A": A})}
    else:
        ctx = {"TABLE1": pd.DataFrame({"A": A, "B": B})}
    row_funcs = {
        "=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) == pd.Timestamp(x.iloc[1]),
        "<>": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) != pd.Timestamp(x.iloc[1]),
        "!=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) != pd.Timestamp(x.iloc[1]),
        "<": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) < pd.Timestamp(x.iloc[1]),
        "<=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) <= pd.Timestamp(x.iloc[1]),
        ">": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) > pd.Timestamp(x.iloc[1]),
        ">=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else pd.Timestamp(x.iloc[0]) >= pd.Timestamp(x.iloc[1]),
        "<=>": lambda x: True
        if pd.isna(x.iloc[0]) and pd.isna(x.iloc[1])
        else (
            False
            if pd.isna(x.iloc[1]) or pd.isna(x.iloc[1])
            else pd.Timestamp(x.iloc[0]) == pd.Timestamp(x.iloc[1])
        ),
    }
    answer = ctx["TABLE1"].apply(row_funcs[cmp_op], axis=1)
    return cmp_op, use_case, ctx, answer


def test_date_compare_datetime64(date_datetime64_comparison_args, memory_leak_check):
    """
    Checks that comparison operator works correctly between datetime.date
    objects and np.datetime64 objects
    """
    cmp_op, use_case, ctx, answer = date_datetime64_comparison_args
    if cmp_op == "<=>":
        # <=> operator requires that both sides must be of the same type
        return
    if use_case:
        query = f"SELECT CASE WHEN (B {cmp_op} A) IS NULL THEN NULL ELSE B {cmp_op} A END as OUTPUT from table1"
    else:
        query = f"SELECT A {cmp_op} B as OUTPUT from table1"
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"OUTPUT": answer}),
    )


@pytest.fixture
def tz_aware_tz_naive_comparison_args(comparison_query_args):
    cmp_op, use_case = comparison_query_args
    data = [
        None,
        pd.Timestamp("2010-01-17", tz="US/Pacific"),
        pd.Timestamp("2011-02-26 03:36:01", tz="US/Pacific"),
        pd.Timestamp("2012-05-09 16:43:16.123456", tz="US/Pacific"),
        pd.Timestamp("2013-10-22 02:32:21.987654321", tz="US/Pacific"),
        pd.Timestamp("2010-02-03 01:15:12.501000", tz="US/Pacific"),
    ]
    A = pd.Series(data * 6)
    b = []
    for elem in data:
        b += [pd.Timestamp(elem).tz_localize(None)] * 6
    B = pd.Series(b)
    if use_case:
        ctx = {"TABLE1": pd.DataFrame({"B": B, "A": A})}
    else:
        ctx = {"TABLE1": pd.DataFrame({"A": A, "B": B})}
    row_funcs = {
        "=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) == x.iloc[1].tz_localize(None),
        "<>": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) != x.iloc[1].tz_localize(None),
        "!=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) != x.iloc[1].tz_localize(None),
        "<": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) < x.iloc[1].tz_localize(None),
        "<=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) <= x.iloc[1].tz_localize(None),
        ">": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) > x.iloc[1].tz_localize(None),
        ">=": lambda x: None
        if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
        else x.iloc[0].tz_localize(None) >= x.iloc[1].tz_localize(None),
        "<=>": lambda x: True
        if pd.isna(x.iloc[0]) and pd.isna(x.iloc[1])
        else (
            False
            if pd.isna(x.iloc[0]) or pd.isna(x.iloc[1])
            else x.iloc[0].tz_localize(None) == x.iloc[1].tz_localize(None)
        ),
    }
    answer = ctx["TABLE1"].apply(row_funcs[cmp_op], axis=1)
    return cmp_op, use_case, ctx, answer, "US/Pacific"


def test_tz_aware_compare_tz_naive(
    tz_aware_tz_naive_comparison_args, memory_leak_check
):
    """
    Checks that comparison operator works correctly between tz_aware
    timestamps and tz_naive timestamps
    """
    cmp_op, use_case, ctx, answer, session_tz = tz_aware_tz_naive_comparison_args
    if cmp_op == "<=>":
        # <=> operator requires that both sides must be of the same type
        return
    if use_case:
        query = f"SELECT CASE WHEN (B {cmp_op} A) IS NULL THEN NULL ELSE B {cmp_op} A END as OUTPUT from table1"
    else:
        query = f"SELECT A {cmp_op} B as OUTPUT from table1"
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"OUTPUT": answer}),
        session_tz=session_tz,
    )
