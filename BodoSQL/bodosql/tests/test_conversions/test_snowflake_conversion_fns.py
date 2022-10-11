# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL Snowflake date-related conversion functions"""


import bodosql
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query
from pandas.api.types import is_bool_dtype, is_float_dtype

from bodo.tests.bodosql_array_kernel_tests.test_bodosql_snowflake_conversion_array_kernels import (
    _dates,
    _dates_nans,
    _times,
    _times_nans,
    str_to_bool,
)
from bodo.utils.typing import BodoError

valid_bool_params = [
    pytest.param(
        pd.DataFrame(
            {
                "a": [
                    "y",
                    "yes",
                    "t",
                    "true",
                    "on",
                    "1",
                    "n",
                    "no",
                    None,
                    "f",
                    "false",
                    "off",
                    "0",
                ]
                * 3
            }
        ),
        id="valid_to_boolean_strings",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "a": [
                    "Y",
                    "yEs",
                    "T",
                    "TRuE",
                    "on",
                    None,
                    "1",
                    "n",
                    "NO",
                    "f",
                    "FALSE",
                    "OFF",
                    "0",
                ]
                * 3
            }
        ),
        id="valid_to_boolean_strings_mixed_case",
    ),
    pytest.param(
        pd.DataFrame({"a": pd.Series([1, 0, 2, 0, None, -2, -400] * 3, dtype="Int64")}),
        id="valid_to_boolean_ints",
    ),
    pytest.param(
        pd.DataFrame({"a": [1.1, 0.0, None, 0.1, 0, -2, -400] * 3}),
        id="valid_to_boolean_floats",
    ),
]

invalid_bool_params = [
    pytest.param(
        pd.DataFrame({"a": pd.Series(["t", "a", "b", "y", None, "f"] * 3)}),
        id="invalid_to_boolean_strings",
    ),
    pytest.param(
        pd.DataFrame({"a": [1.1, 0.0, np.inf, 0.1, np.nan, -2, -400] * 3}),
        id="invalid_to_boolean_floats",
    ),
]


@pytest.fixture(params=valid_bool_params)
def to_boolean_valid_test_dfs(request):
    return request.param


@pytest.fixture(params=invalid_bool_params)
def to_boolean_invalid_test_dfs(request):
    return request.param


@pytest.fixture(params=valid_bool_params + invalid_bool_params)
def to_boolean_all_test_dfs(request):
    return request.param


def test_to_boolean_valid_cols(
    spark_info, to_boolean_valid_test_dfs, memory_leak_check
):
    df = to_boolean_valid_test_dfs
    query = f"SELECT TO_BOOLEAN(a) FROM table1"
    ctx = {"table1": df}
    arr = df[df.columns[0]]
    if arr.apply(type).eq(str).any():
        py_output = arr.apply(str_to_bool)
    else:
        py_output = arr.apply(lambda x: np.nan if pd.isna(x) else bool(x))
    py_output = pd.DataFrame({"a": py_output.astype("boolean")})
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=py_output,
    )


def test_to_boolean_invalid_cols(spark_info, to_boolean_invalid_test_dfs):
    df = to_boolean_invalid_test_dfs
    query = f"SELECT TO_BOOLEAN(a) FROM table1"
    ctx = {"table1": df}
    arr = df[df.columns[0]]
    is_str = arr.apply(type).eq(str).any()
    bc = bodosql.BodoSQLContext(ctx)
    if is_str:
        with pytest.raises(ValueError, match="string must be one of"):
            bc.sql(query)
    else:
        with pytest.raises(
            ValueError, match="value must be a valid numeric expression"
        ):
            bc.sql(query)


def to_boolean_equiv(arr):
    if arr.apply(type).eq(str).any():
        return arr.apply(str_to_bool)
    else:
        return arr.apply(lambda x: np.nan if pd.isna(x) or np.isinf(x) else bool(x))


def test_to_boolean_scalars(spark_info, memory_leak_check):
    df = pd.DataFrame(
        {
            "a": [
                "y",
                "yes",
                "t",
                "true",
                "on",
                "1",
                "n",
                "no",
                "f",
                "false",
                "off",
                "0",
            ]
            * 3,
            "b": [1.1, 0.0, 1.0, 0.1, 0, -2, -400, 1.1, 0.0, 1.0, 0.1, 0] * 3,
            "c": np.tile([0, 1], 18),
        }
    )
    ctx = {"table1": df}
    query = "SELECT CASE WHEN TO_BOOLEAN(c) THEN TO_BOOLEAN(a) ELSE TO_BOOLEAN(b) END FROM table1"
    py_output = (
        df["a"].apply(str_to_bool).where(df["c"].astype(bool), df["b"].astype(bool))
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"a": py_output}),
    )


def test_try_to_boolean_cols(spark_info, to_boolean_all_test_dfs, memory_leak_check):
    df = to_boolean_all_test_dfs
    query = f"SELECT TRY_TO_BOOLEAN(a) FROM table1"
    ctx = {"table1": df}
    arr = df[df.columns[0]]
    is_str = arr.apply(type).eq(str).any()
    if is_str:
        py_output = arr.apply(str_to_bool)
    else:
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) or np.isinf(x) else bool(x)
        )
    py_output = pd.DataFrame({"a": py_output})
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=py_output,
    )


def test_try_to_boolean_scalars(spark_info):
    df = pd.DataFrame(
        {
            "a": [
                "y",
                "yes",
                "t",
                "true",
                "on",
                "1",
                "n",
                "no",
                "f",
                "false",
                "off",
                "0",
            ]
            * 3,
            "b": [1.1, 0.0, 1.0, 0.1, 0, -2, -400, 1.1, 0.0, 1.0, 0.1, 0] * 3,
            "c": np.tile([0, 1], 18),
        }
    )
    ctx = {"table1": df}
    query = "SELECT CASE WHEN TRY_TO_BOOLEAN(c) THEN TRY_TO_BOOLEAN(a) ELSE TRY_TO_BOOLEAN(b) END FROM table1"
    py_output = to_boolean_equiv(df["a"]).where(
        df["c"].astype(bool), to_boolean_equiv(df["b"])
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"a": py_output}),
    )


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame({"a": [1, 2, 3, 4, 5]}),
            id="int",
        ),
        pytest.param(
            pd.DataFrame({"a": pd.Series([1, 2, None, 4, 5], dtype="Int64")}),
            id="int_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.1, 2.2, 3.3, 4.4, 5.5]}),
            id="float",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.1, 2.2, None, 4.4, np.inf]}),
            id="float_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"a": [True, False, True, False, True]}),
            id="bool",
        ),
        pytest.param(
            pd.DataFrame(
                {"a": pd.Series([True, False, None, False, True], dtype="boolean")}
            ),
            id="bool_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"a": _dates[:5]}),
            id="date",
        ),
        pytest.param(
            pd.DataFrame({"a": _dates_nans[:5]}),
            id="date_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"a": _times[:5]}),
            id="times",
        ),
        pytest.param(
            pd.DataFrame({"a": _times_nans[:5]}),
            id="times_with_nulls",
        ),
    ]
)
def to_char_test_dfs(request):
    return request.param


@pytest.mark.parametrize(
    "func",
    [
        pytest.param("TO_CHAR"),
        pytest.param("TO_VARCHAR", marks=pytest.mark.slow),
    ],
)
def test_to_char_cols(spark_info, to_char_test_dfs, func, memory_leak_check):
    df = to_char_test_dfs
    query = f"SELECT {func}(a) FROM table1"
    ctx = {"table1": df}
    arr = df["a"]
    if is_float_dtype(arr):
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) else "inf" if np.isnan(x) else f"{x:.6f}"
        )
    elif is_bool_dtype(arr):
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) else ("true" if x else "false")
        )
    else:
        py_output = arr.apply(lambda x: np.nan if pd.isna(x) else str(x))
    py_output = pd.DataFrame({"a": py_output})
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "func",
    [
        "TO_CHAR",
        pytest.param("TO_VARCHAR", marks=pytest.mark.slow),
    ],
)
def test_to_char_scalars(spark_info, func):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 3,
            "b": [1.1, 2.2, np.nan, 4.4, 5.5] * 3,
            "c": [True, False, True, False, True] * 3,
            "d": pd.date_range("20130101", periods=15, freq="D"),
        }
    )
    ctx = {"table1": df}
    query = f"SELECT CASE WHEN c THEN {func}(a + b) ELSE {func}(d) END FROM table1"
    py_output = (
        (df["a"] + df["b"])
        .apply(lambda x: np.nan if pd.isna(x) else "inf" if np.isnan(x) else f"{x:.6f}")
        .where(df["c"], df["d"].apply(lambda x: np.nan if pd.isna(x) else str(x)))
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"a": py_output}),
    )


def test_to_char_error():
    """Test that we raise an error when we try to use a format string with TO_CHAR."""
    ctx = {
        "table1": pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5] * 3,
            }
        )
    }
    query = f"SELECT TO_CHAR(a, 'YYYY-MM-DD') FROM table1"
    bc = bodosql.BodoSQLContext(ctx)
    with pytest.raises(
        BodoError,
        match="format string for TO_CHAR not yet supported",
    ):
        bc.sql(query)
