# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL boolean functions on BodoSQL
"""

import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param(
            {
                "table1": pd.DataFrame(
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
                "table1": pd.DataFrame(
                    {
                        "A": pd.Series([42.0] * 3 + [0.0] * 3 + [None] * 3),
                        "B": pd.Series([-13.1, 0.0, None] * 3),
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
    "args",
    [
        pytest.param(
            (
                "SELECT EQUAL_NULL(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                True,
                                False,
                                False,
                                False,
                                False,
                                True,
                                False,
                                False,
                                True,
                                False,
                            ]
                        )
                    }
                ),
            ),
            id="all_vector_string",
        ),
        pytest.param(
            (
                "SELECT EQUAL_NULL(A, 'A') OR EQUAL_NULL(A, 'B') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                True,
                                True,
                                False,
                                False,
                                True,
                                True,
                                False,
                                False,
                                False,
                                False,
                            ]
                        )
                    }
                ),
            ),
            id="vector_scalar_string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN B = '' THEN TRUE ELSE EQUAL_NULL(B, NULLIF(B, B)) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                False,
                                True,
                                False,
                                False,
                                False,
                                False,
                                False,
                                True,
                                True,
                                True,
                            ]
                        )
                    }
                ),
            ),
            id="case_string",
        ),
        pytest.param(
            (
                "SELECT EQUAL_NULL(C, D) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                                True,
                                False,
                                False,
                            ]
                        )
                    }
                ),
            ),
            id="all_vector_int",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN C IS NULL OR C = 1 THEN TRUE ELSE EQUAL_NULL(C, D) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                                True,
                                True,
                                True,
                            ]
                        )
                    }
                ),
            ),
            id="all_vector_int",
        ),
    ],
)
def test_equal_null(args, spark_info, memory_leak_check):
    query, answer = args
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": ["A", "B", "", None, "A", "B", "C", "D", None, None],
                "B": ["A", "", "B", "A", "C", "B", "A", None, None, ""],
                "C": pd.Series(
                    [1, 2, 3, 4, 5, 5, None, 3, None, 1], dtype=pd.Int32Dtype()
                ),
                "D": pd.Series(
                    [1, 2, 3, 4, 5, 1, None, 3, 4, None], dtype=pd.Int32Dtype()
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )
