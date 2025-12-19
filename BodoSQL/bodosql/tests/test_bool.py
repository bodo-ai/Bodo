"""
Test correctness of SQL boolean functions on BodoSQL
"""

import datetime

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodo.types import Time
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": pd.Series(
                            [1, -2, 3, 0, 0, 0, None, None, None], dtype=pd.Int32Dtype()
                        ),
                        "B": pd.Series(
                            [1, 0, None, -2, 0, None, 3, 0, None], dtype=pd.Int32Dtype()
                        ),
                    }
                )
            },
            id="int32",
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": pd.Series(
                            [42.0] * 3 + [0.0] * 3 + [None] * 3, dtype="Float64"
                        ),
                        "B": pd.Series([-13.1, 0.0, None] * 3, dtype="Float64"),
                    }
                )
            },
            id="float",
            marks=pytest.mark.slow,
        ),
    ]
)
def numeric_truthy_df(request):
    return request.param


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BOOLAND(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True, False, None, False, False, False, None, False, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT BOOLAND(A, 0.0) FROM table1",
                pd.DataFrame({0: pd.Series([False] * 9, dtype=pd.BooleanDtype())}),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN A IS NULL AND B IS NULL THEN FALSE ELSE BOOLAND(A, B) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                True,
                                False,
                                None,
                                False,
                                False,
                                False,
                                None,
                                False,
                                False,
                            ],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_scalar_with_case",
        ),
    ],
)
def test_booland(args, numeric_truthy_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        numeric_truthy_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BOOLOR(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True, True, True, True, False, None, True, None, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT BOOLOR(A, 0.0) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True] * 3 + [False] * 3 + [None] * 3,
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN A IS NULL AND B IS NULL THEN FALSE ELSE BOOLOR(A, B) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True, True, True, True, False, None, None, True, False],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_scalar_with_case",
        ),
    ],
)
def test_boolor(args, numeric_truthy_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        numeric_truthy_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BOOLXOR(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, None, True, False, None, None, None, None],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT BOOLXOR(A, 0.0) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [True] * 3 + [False] * 3 + [None] * 3,
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN A IS NULL AND B IS NULL THEN FALSE ELSE BOOLXOR(A, B) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, True, None, True, False, None, None, None, False],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_scalar_with_case",
        ),
    ],
)
def test_boolxor(args, numeric_truthy_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        numeric_truthy_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BOOLNOT(A) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False] * 3 + [True] * 3 + [None] * 3,
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "SELECT BOOLNOT(64) FROM table1",
                pd.DataFrame({0: pd.Series([False] * 9, dtype=pd.BooleanDtype())}),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN B IS NULL THEN FALSE ELSE BOOLNOT(A) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [False, False, False, True, True, False, None, None, False],
                            dtype=pd.BooleanDtype(),
                        )
                    }
                ),
            ),
            id="all_scalar_with_case",
        ),
    ],
)
def test_boolnot(args, numeric_truthy_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        numeric_truthy_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "dtype, val1, val2, use_symbol",
    [
        pytest.param(str, "A", "B", True, id="string"),
        pytest.param(str, b"yay", b"boo", False, id="binary", marks=pytest.mark.slow),
        pytest.param(pd.Int32Dtype(), -1, 500, True, id="int32"),
        pytest.param(
            None,
            pd.Timestamp("2023-1-30 6:00:00"),
            pd.Timestamp("2024-2-29 9:30:00"),
            False,
            id="timestamp",
        ),
        pytest.param(None, True, False, True, id="bool", marks=pytest.mark.slow),
        pytest.param(
            None,
            Time(12, 30, 0),
            Time(16, 59, 30, nanosecond=999999999),
            False,
            id="time",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            None,
            datetime.date(2000, 1, 1),
            datetime.date(2023, 12, 31),
            True,
            id="date",
        ),
        pytest.param(None, [1, 2, 3], [], False, id="integer_array"),
        pytest.param(
            None,
            {"A": 0, "B": "foo", "C": True},
            {"A": 0, "B": "bar", "C": True},
            False,
            id="struct",
        ),
    ],
)
def test_equal_null(dtype, val1, val2, use_case, use_symbol, memory_leak_check):
    """Tests EQUAL_NULL on multiple dtypes by parametrizing the values used
    from each dtype.

    If use_symbol is true, uses <=> instead of EQUAL_NULL.
    If use_case is true, makes the call inside of a case statement.
    """
    A = pd.Series([val1, val2, None] * 9, dtype=dtype)
    B = pd.Series(
        [val1] * 3
        + [val2] * 3
        + [None] * 3
        + [val1] * 3
        + [val2] * 3
        + [None] * 3
        + [val1] * 3
        + [val2] * 3
        + [None] * 3,
        dtype=dtype,
    )
    result = pd.Series([True, False, False, False, True, False, False, False, True] * 3)
    if use_case:
        if use_symbol:
            query = "SELECT CASE WHEN A <=> B THEN 'T' ELSE 'F' END AS are_equal FROM TABLE1"
        else:
            query = "SELECT CASE WHEN EQUAL_NULL(A, B) THEN 'T' ELSE 'F' END AS are_equal FROM TABLE1"
        result = result.astype(str).str[0]
    else:
        if use_symbol:
            query = "SELECT A <=> B AS are_equal FROM TABLE1"
        else:
            query = "SELECT EQUAL_NULL(A, B) AS are_equal FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"A": A, "B": B})}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"ARE_EQUAL": result}),
    )
