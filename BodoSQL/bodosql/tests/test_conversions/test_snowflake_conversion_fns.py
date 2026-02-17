"""Test Bodo's array kernel utilities for BodoSQL Snowflake date-related conversion functions"""

import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.api.types import is_bool_dtype, is_float_dtype

import bodosql
from bodo.types import Time
from bodosql.tests.test_kernels.test_snowflake_conversion_array_kernels import (
    _dates,
    _dates_nans,
    _timestamps,
    _timestamps_nans,
    str_to_bool,
)
from bodosql.tests.utils import check_query

valid_bool_params = [
    pytest.param(
        pd.DataFrame(
            {
                "A": [
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
                    # Extra Padding
                    "\n\t   n",
                    " \t\t  true\n  ",
                ]
                * 3
            }
        ),
        id="valid_to_boolean_strings",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "A": [
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
        pd.DataFrame({"A": pd.Series([1, 0, 2, 0, None, -2, -400] * 3, dtype="Int64")}),
        id="valid_to_boolean_ints",
    ),
    pytest.param(
        pd.DataFrame({"A": [1.1, 0.0, None, 0.1, 0, -2, -400] * 3}),
        id="valid_to_boolean_floats",
    ),
]

invalid_bool_params = [
    pytest.param(
        pd.DataFrame({"A": pd.Series(["t", "a", "b", "y", None, "f"] * 3)}),
        id="invalid_to_boolean_strings",
    ),
    pytest.param(
        pd.DataFrame({"A": pd.Series(["\t\tt", None, "\nf\talse"] * 3)}),
        id="invalid_to_boolean_strings",
    ),
    pytest.param(
        pd.DataFrame({"A": [1.1, 0.0, np.inf, 0.1, np.nan, -2, -400] * 3}),
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
    query = "SELECT TO_BOOLEAN(a) FROM table1"
    ctx = {"TABLE1": df}
    arr = df[df.columns[0]]
    if arr.apply(type).eq(str).any():
        py_output = arr.apply(str_to_bool)
    else:
        py_output = arr.apply(lambda x: np.nan if pd.isna(x) else bool(x))
    py_output = pd.DataFrame({"A": py_output.astype("boolean")})
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
    query = "SELECT TO_BOOLEAN(a) FROM table1"
    ctx = {"TABLE1": df}
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
            "A": [
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
            "B": [1.1, 0.0, 1.0, 0.1, 0, -2, -400, 1.1, 0.0, 1.0, 0.1, 0] * 3,
            "C": np.tile([0, 1], 18),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT CASE WHEN TO_BOOLEAN(c) THEN TO_BOOLEAN(a) ELSE TO_BOOLEAN(b) END FROM table1"
    py_output = (
        df["A"].apply(str_to_bool).where(df["C"].astype(bool), df["B"].astype(bool))
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": py_output}),
    )


def test_try_to_boolean_cols(spark_info, to_boolean_all_test_dfs, memory_leak_check):
    df = to_boolean_all_test_dfs
    query = "SELECT TRY_TO_BOOLEAN(a) FROM table1"
    ctx = {"TABLE1": df}
    arr = df[df.columns[0]]
    is_str = arr.apply(type).eq(str).any()
    if is_str:
        py_output = arr.apply(str_to_bool)
    else:
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) or np.isinf(x) else bool(x)
        )
    py_output = pd.DataFrame({"A": py_output.astype(pd.ArrowDtype(pa.bool_()))})
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
            "A": [
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
            "B": [1.1, 0.0, 1.0, 0.1, 0, -2, -400, 1.1, 0.0, 1.0, 0.1, 0] * 3,
            "C": np.tile([0, 1], 18),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT CASE WHEN TRY_TO_BOOLEAN(c) THEN TRY_TO_BOOLEAN(a) ELSE TRY_TO_BOOLEAN(b) END FROM table1"
    py_output = to_boolean_equiv(df["A"]).where(
        df["C"].astype(bool), to_boolean_equiv(df["B"])
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": py_output}),
    )


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
            id="int",
        ),
        pytest.param(
            pd.DataFrame({"A": pd.Series([1, 2, None, 4, 5], dtype="Int64")}),
            id="int_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"A": [1.1, 2.2, 3.3, 4.4, 5.5]}),
            id="float",
        ),
        pytest.param(
            pd.DataFrame({"A": [1.1, 2.2, None, 4.4, np.inf]}),
            id="float_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"A": [True, False, True, False, True]}),
            id="bool",
        ),
        pytest.param(
            pd.DataFrame(
                {"A": pd.Series([True, False, None, False, True], dtype="boolean")}
            ),
            id="bool_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"A": _dates[:]}),
            id="date",
        ),
        pytest.param(
            pd.DataFrame({"A": _dates_nans[:]}),
            id="date_with_nulls",
        ),
        pytest.param(
            pd.DataFrame({"A": _timestamps[:]}),
            id="timestamp",
        ),
        pytest.param(
            pd.DataFrame({"A": _timestamps_nans[:]}),
            id="timestamp_with_nulls",
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
    ctx = {"TABLE1": df}
    arr = df["A"]
    if is_float_dtype(arr):
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) else "inf" if np.isnan(x) else f"{x:.6f}"
        )
    elif is_bool_dtype(arr):
        py_output = arr.apply(
            lambda x: np.nan if pd.isna(x) else ("true" if x else "false")
        )
    else:
        py_output = (
            arr.astype(str)
            .replace("<NA>", None)
            .replace("nan", None)
            .replace("NaT", None)
        )
    py_output = pd.DataFrame({"A": py_output})
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
            "A": [1, 2, 3, 4, 5] * 3,
            "B": [1.1, 2.2, np.nan, 4.4, 5.5] * 3,
            "C": [True, False, True, False, True] * 3,
            "D": pd.date_range("20130101", periods=15, freq="D", unit="ns"),
        }
    )
    ctx = {"TABLE1": df}
    query = f"SELECT CASE WHEN c THEN {func}(a + b) ELSE {func}(d) END FROM table1"
    py_output = (
        (df["A"] + df["B"])
        .apply(lambda x: np.nan if pd.isna(x) else "inf" if np.isnan(x) else f"{x:.6f}")
        .where(df["C"], df["D"].apply(lambda x: np.nan if pd.isna(x) else str(x)))
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": py_output}),
    )


@pytest.mark.tz_aware
def test_tz_aware_datetime_to_char(tz_aware_df, memory_leak_check):
    """simplest test for TO_CHAR on timezone aware data"""
    query = "SELECT TO_CHAR(A) as A from table1"

    expected_output = pd.DataFrame({"A": tz_aware_df["TABLE1"]["A"].astype(str)})
    check_query(
        query,
        tz_aware_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_timestamp_to_char(memory_leak_check):
    """simplest test for TO_CHAR on timezone-naive timestamps"""
    query = "SELECT TO_CHAR(A) as A from table1"

    dt_series = pd.date_range(
        "2022/1/1", periods=30, freq="6D5h15min45s", unit="ns"
    ).to_series()
    df = pd.DataFrame({"A": dt_series})
    expected_output = pd.DataFrame({"A": dt_series.dt.strftime("%Y-%m-%d %X%z")})

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_date_to_char(memory_leak_check):
    """simplest test for TO_CHAR on date inputs"""
    query = "SELECT TO_CHAR(A) as A from table1"

    dt_series = pd.Series(
        [
            None
            if i % 7 == 2
            else datetime.date(
                2023 - (i**2) % 25, 1 + ((i**2) % 48) // 4, 1 + (i**5) % 28
            )
            for i in range(30)
        ]
    )
    df = pd.DataFrame({"A": dt_series})
    expected_output = pd.DataFrame(
        {
            "A": pd.DatetimeIndex(dt_series)
            .to_series(index=pd.RangeIndex(30))
            .dt.strftime("%Y-%m-%d")
        }
    )

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_char_datetime_format_str(memory_leak_check):
    """test for TO_CHAR with a datetime format string"""
    datetime_string_series = pd.Series(
        [
            pd.Timestamp(1990, 2, 7, 12, 12, 12),
            pd.Timestamp(2000, 8, 17, 22, 12, 22),
            pd.Timestamp(1965, 3, 3, 2, 12, 42),
            pd.Timestamp(2010, 12, 25, 4, 12, 52),
        ]
        * 4,
        dtype="datetime64[ns]",
    )
    sql_format_str = "MMMM DD, YYYY HH24:MI:SS"
    query = f"SELECT TO_CHAR(A, '{sql_format_str}') as A from table1"
    df = pd.DataFrame({"A": datetime_string_series})
    expected_output = pd.DataFrame(
        {
            "A": pd.DatetimeIndex(datetime_string_series)
            .to_series(index=pd.RangeIndex(16))
            .dt.strftime("%B %d, %Y %H:%M:%S")
        }
    )

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(
            True,
            id="with_case",
        ),
    ],
)
def test_time_to_char(use_case, memory_leak_check):
    """Test TO_CHAR on TIME data using the default format: 'HH:MM:SS'"""
    if use_case:
        query = "SELECT TO_CHAR(T) from table1"
    else:
        query = (
            "SELECT CASE WHEN HOUR(T) = 0 THEN 'x' ELSE TO_VARCHAR(T) END from table1"
        )

    T = pd.Series(
        [
            Time(0, 1, 2),
            None,
            Time(12, 30, 0),
            Time(1, 45, 59, nanosecond=999999999),
            Time(23, 59, 5, microsecond=250),
        ]
    )
    if use_case:
        S = pd.Series(["00:01:02", None, "12:30:00", "01:45:59", "23:59:05"])
    else:
        S = pd.Series(["x", None, "12:30:00", "01:45:59", "23:59:05"])

    ctx = {"TABLE1": pd.DataFrame({"T": T})}
    expected_output = pd.DataFrame({0: S})

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    ["68656c6c6f", "416C70686162657420536F7570", None, "01", "AbCdEf"]
                ),
                pd.Series(
                    [
                        b"hello",
                        b"Alphabet Soup",
                        None,
                        b"\x01",
                        b"\xab\xcd\xef",
                    ]
                ),
            ),
            id="hex_strings",
        ),
        pytest.param(
            (
                pd.Series([b"123", b"alpha beta", None, b"soup", b"7Fff"]),
                pd.Series([b"123", b"alpha beta", None, b"soup", b"7Fff"]),
            ),
            id="binary_copy",
        ),
    ]
)
def binary_cast_data(request):
    return request.param


# TODO ([BE-4344]): implement and test to_binary with other formats
@pytest.mark.parametrize(
    "calculation",
    [
        pytest.param("TO_BINARY(S)", id="to_binary-no_case"),
        pytest.param(
            "TRY_TO_BINARY(S)", id="try-to_binary-no_case", marks=pytest.mark.slow
        ),
        pytest.param(
            "CASE WHEN LENGTH(S) > 0 THEN TO_BINARY(S) END",
            id="to_binary-with_case",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "CASE WHEN LENGTH(S) > 0 THEN TRY_TO_BINARY(S) END",
            id="try_to_binary-with_case",
        ),
    ],
)
def test_to_binary(calculation, binary_cast_data, memory_leak_check):
    data, answer = binary_cast_data
    query = f"SELECT {calculation} AS B FROM table1"
    ctx = {"TABLE1": pd.DataFrame({"S": data})}
    expected_output = pd.DataFrame({"B": answer})
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        convert_columns_bytearray=["B"],
        expected_output=expected_output,
    )


def test_try_to_binary_invalid(memory_leak_check):
    """Verifies that TRY_TO_BINARY outputs NULL on malformed cases
    (non-hex characters or odd number of characters)"""
    query = "SELECT TRY_TO_BINARY(S) AS B FROM table1"
    data = pd.Series(["AB", "ABC", "ABCD", "ABCDE", "GHI", "10", "101", "10 A"])
    answer = pd.Series([b"\xab", None, b"\xab\xcd", None, None, b"\x10", None, None])
    ctx = {"TABLE1": pd.DataFrame({"S": data})}
    expected_output = pd.DataFrame({"B": answer})
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        convert_columns_bytearray=["B"],
        expected_output=expected_output,
    )


def test_to_binary_error():
    """Verifies that TO_BINARY raises an exception on malformed cases
    (non-hex characters or odd number of characters)"""
    query = "SELECT TO_BINARY(S) FROM table1"
    data = pd.Series(["AB", "ABC", "ABCD", "ABCDE", "GHI", "10", "101", "10 A"])
    ctx = {"TABLE1": pd.DataFrame({"S": data})}
    with pytest.raises(ValueError):
        bc = bodosql.BodoSQLContext(ctx)
        bc.sql(query)


valid_double_params = [
    pytest.param(
        pd.DataFrame(
            {
                "A": [
                    ".1",
                    ".1e5",
                    ".1e+5",
                    ".1e-5",
                    "15",
                    "15E09",
                    "0.",
                    "1.34",
                    "1.34e-7",
                    "+.1",
                    "+.37E4",
                    "-28",
                    "-90e0",
                    "+2.64",
                    "-2.64e-07",
                    "123.456e+010",
                    "nan",
                    "NaN",
                    "-nAn",
                    "inf",
                    "INF",
                    "iNf",
                    "-Inf",
                    "-InFiNiTy",
                    "+Infinity",
                    # Extra Left and Right Padding
                    "     12.4",
                    "  -0.5\t\n",
                    "\r\n\r\n0.\r\n",
                ]
            }
        ),
        id="valid_to_double_strings",
    ),
    pytest.param(
        pd.DataFrame({"A": pd.Series([1, 0, 2, 0, None, -2, -400], dtype="Int64")}),
        id="valid_to_double_ints",
    ),
    pytest.param(
        pd.DataFrame({"A": [1.1, 0.0, None, 0.1, 0, -2, -400, np.inf, np.nan]}),
        id="valid_to_double_floats",
    ),
]


invalid_double_params = [
    pytest.param(
        pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        "baked beans",
                        "nane",
                        ".",
                        "+",
                        "-.",
                        "2e",
                        "2e-",
                        "123.4.56",
                        "4-7",
                        "12-",
                        "",
                        # Internal Padding
                        " - 0.73 ",
                    ]
                )
            }
        ),
        id="invalid_to_double_strings",
    ),
]


@pytest.fixture(params=valid_double_params)
def to_double_valid_test_dfs(request):
    return request.param


@pytest.fixture(params=invalid_double_params)
def to_double_invalid_test_dfs(request):
    return request.param


@pytest.fixture(params=valid_double_params + invalid_double_params)
def to_double_all_test_dfs(request):
    return request.param


def to_double_equiv(arr):
    def _conv_to_double(x):
        """Converts an array of values to double"""
        if pd.isna(x):
            return None
        try:
            return str(np.float64(x))
        except Exception:
            return None

    out_arr = arr.apply(_conv_to_double)
    # Workaround Int NA being passed as NaN
    if arr.dtype == pd.Int64Dtype():
        out_arr = out_arr.map(lambda x: None if x == "nan" else x)
    # Workaround float64 NA ambiguity
    if arr.dtype == np.float64:
        out_arr = arr.map(lambda x: str(x))
    # Make sure NaNs are preserved correctly in Arrow conversion
    out_arr = pa.compute.cast(
        out_arr.astype(pd.ArrowDtype(pa.large_string())).array._pa_array,
        pa.float64(),
    )
    return pd.Series(out_arr, dtype=pd.ArrowDtype(pa.float64()))


def test_to_double_valid_cols(spark_info, to_double_valid_test_dfs, memory_leak_check):
    df = to_double_valid_test_dfs
    query = "SELECT TO_DOUBLE(a) FROM table1"
    ctx = {"TABLE1": df}
    arr = df[df.columns[0]]
    py_output = pd.DataFrame({"A": to_double_equiv(arr)})
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=py_output,
    )


def test_to_double_invalid_cols(spark_info, to_double_invalid_test_dfs):
    df = to_double_invalid_test_dfs
    query = "SELECT TO_DOUBLE(a) FROM table1"
    ctx = {"TABLE1": df}
    arr = df[df.columns[0]]
    bc = bodosql.BodoSQLContext(ctx)
    is_str = arr.apply(type).eq(str).any()
    if is_str:
        with pytest.raises(
            ValueError, match="string must be a valid numeric expression"
        ):
            bc.sql(query)
    else:
        with pytest.raises(
            ValueError, match="value must be a valid numeric expression"
        ):
            bc.sql(query)


def test_to_double_scalars(spark_info, memory_leak_check):
    df = pd.DataFrame(
        {
            "A": [
                ".1",
                ".1e5",
                ".1e+5",
                ".1e-5",
                "15",
                "15E09",
                "1.34",
                "1.34e-7",
                "+.1",
                "+.37E4",
                "-28",
                "-90e0",
                "+2.64",
                "-2.64e-07",
            ]
            * 3,
            "B": [
                1.1,
                0.0,
                1.0,
                0.1,
                0,
                -2,
                -400,
                1.1,
                0.0,
                1.0,
                0.1,
                0,
                np.nan,
                np.inf,
            ]
            * 3,
            "C": np.tile([0, 1], 21),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT CASE WHEN TO_DOUBLE(c) = 1.0 THEN TO_DOUBLE(a) ELSE TO_DOUBLE(b) END FROM table1"
    py_output = to_double_equiv(df["A"]).where(
        df["C"].astype(bool), to_double_equiv(df["B"])
    )
    # Avoid NaN vs NA mismatch
    py_output = py_output.map(lambda x: pd.NA if pd.isna(x) else x).astype(
        pd.ArrowDtype(pa.float64())
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": py_output}),
    )


def test_try_to_double_cols(spark_info, to_double_all_test_dfs, memory_leak_check):
    df = to_double_all_test_dfs
    query = "SELECT TRY_TO_DOUBLE(a) FROM table1"
    ctx = {"TABLE1": df}
    arr = df[df.columns[0]]
    py_output = pd.DataFrame({"A": to_double_equiv(arr)})
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=py_output,
    )


def test_try_to_double_scalars(spark_info):
    df = pd.DataFrame(
        {
            "A": [
                ".1",
                ".1e5",
                ".1e+5",
                ".1e-5",
                "15",
                "15E09",
                "1.34",
                "1.34e-7",
                "+.1",
                "+.37E4",
                "-28",
                "-90e0",
                "+2.64",
                "-2.64e-07",
            ]
            * 3,
            "B": [
                1.1,
                0.0,
                1.0,
                0.1,
                0,
                -2,
                -400,
                1.1,
                0.0,
                1.0,
                0.1,
                0,
                np.nan,
                np.inf,
            ]
            * 3,
            "C": np.tile([0, 1], 21),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT CASE WHEN TRY_TO_DOUBLE(c) = 1.0 THEN TRY_TO_DOUBLE(a) ELSE TRY_TO_DOUBLE(b) END FROM table1"
    py_output = to_double_equiv(df["A"]).where(
        df["C"].astype(bool), to_double_equiv(df["B"])
    )
    # Avoid NaN vs NA mismatch
    py_output = py_output.map(lambda x: pd.NA if pd.isna(x) else x).astype(
        pd.ArrowDtype(pa.float64())
    )
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": py_output}),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_BOOLEAN",
        "TRY_TO_BOOLEAN",
    ],
)
def test_to_boolean_optional_invalid_str(fn_name):
    """Test TRY_TO_BOOLEAN and TO_BOOLEAN in optional argument case.
    Note because of how the IN operator works the integer values are cast
    to boolean, not the other way around. This is verified in SF."""
    query = f"""SELECT case when {fn_name}(A) in (0, 1, 2, 3, 4, 5)
            then 'USA' else 'international' end as origin_zip_type
            FROM table1 """
    df = pd.DataFrame(
        {
            "A": [
                "a",
                "yes",
                "T",
                "TRUE",
                "on",
                "1",
                "n",
                "NO",
                "f",
                "FALSE",
                "OFF",
                "0",
            ]
        }
    )
    expected_output = pd.DataFrame(
        {
            "ORIGIN_ZIP_TYPE": [
                "international",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
                "USA",
            ]
        }
    )
    ctx = {"TABLE1": df}
    if "TRY" in fn_name:
        check_query(
            query,
            ctx,
            None,
            expected_output=expected_output,
        )
    else:
        with pytest.raises(ValueError, match="invalid value for boolean conversion"):
            bc = bodosql.BodoSQLContext({"TABLE1": df})

            bc.sql(query)


@pytest.mark.slow
@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_DOUBLE",
        "TRY_TO_DOUBLE",
    ],
)
def test_to_double_optional_invalid_str(fn_name):
    """Test TRY_TO_DOUBLE and TO_DOUBLE in optional argument case"""
    query = f"""SELECT case when {fn_name}(A) > -1
            then 'USA' else 'international' end as origin_zip_type
            FROM table1 """
    df = pd.DataFrame(
        {
            "A": [
                "+2.64",
                "baked beans",
                "-2.64e-07",
                "+",
                "123.456e+010",
                "nan",
                "nane",
                ".",
            ]
        }
    )
    expected_output = pd.DataFrame(
        {
            "ORIGIN_ZIP_TYPE": [
                "USA",
                "USA",
                "USA",
                "international",
                "international",
                "international",
                "international",
                "international",
            ]
        }
    )
    ctx = {"TABLE1": df}
    if "TRY" in fn_name:
        check_query(
            query,
            ctx,
            None,
            expected_output=expected_output,
        )
    else:
        with pytest.raises(ValueError, match="invalid value for double conversion"):
            bc = bodosql.BodoSQLContext({"TABLE1": df})

            bc.sql(query)
