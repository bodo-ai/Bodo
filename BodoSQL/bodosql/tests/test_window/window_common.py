import math
import os

import numpy as np
import pandas as pd
import pytest

# Helper environment variable to allow for testing locally, while avoiding
# memory issues on CI
testing_locally = os.environ.get("BODOSQL_TESTING_LOCALLY", False)


def count_window_applies(pandas_code, num_windows, funcs):
    """Verifies that the generated code has the correct number of closures
    for window functions. If fusion is working correctly, then pandas_code
    should have the same number of closures as the argument provided.

    Note: currently does not check for fusion if SUM is one of the functions
    being tested, since SUM fusion is not currently working. Will hopefully
    be fixed by [BE-3962]"""

    if "SUM" not in funcs:
        assert (
            pandas_code.count("def __bodo_dummy___sql_windowed_apply_fn") == num_windows
        )


@pytest.fixture(
    params=[
        "RESPECT NULLS",
        pytest.param(
            "IGNORE NULLS",
            marks=pytest.mark.skip("https://bodo.atlassian.net/browse/BE-3583"),
        ),
        pytest.param("", id="empty_str", marks=pytest.mark.slow),
    ]
)
def null_respect_string(request):
    """Returns the null behavior string, for use with LEAD/LAG and First/Last/Nth value"""
    return request.param


window_col_size = 200
uint8_col = pd.Series(
    [None if (i**2) % 10 == 1 else (i**2) % 10 for i in range(window_col_size)],
    dtype=pd.UInt8Dtype(),
)
int32_col = pd.Series(
    [
        max((i**2) % 10, (i**2) % 11, (i**2) % 12) * (-1) ** i
        for i in range(window_col_size)
    ],
    dtype=pd.Int32Dtype(),
)
int64_col = pd.Series(
    [
        None
        if i % 7 == 4 or round(i * 8 / window_col_size) == 6
        else ((-1) ** (round(math.tan(i)) % 2)) * (2 ** ((i**2) % 60)) - 1
        for i in range(window_col_size)
    ],
    dtype=pd.Int64Dtype(),
)

# The float cases don't contain nulls (for now) because we treat the data as
# NaN, which causes PySpark to calculate differently versus nulls
float64_col = pd.Series([math.tan(i) for i in range(window_col_size)])
boolean_col = pd.Series(
    [None if math.tan(i) < 0 else math.sin(i) > -0.1 for i in range(window_col_size)],
    dtype=pd.BooleanDtype(),
)
string_col = pd.Series(
    [
        None
        if math.sin(i) < -0.8
        else "".join(
            "AEIOU!🐍\n∞π"[round(math.tan(i) + j) % 10] for j in range(4 ** (i % 4))
        )
        for i in range(window_col_size)
    ]
)
binary_col = pd.Series(
    [
        None
        if math.cos(i) < -0.8
        else bytes(str(round(1 / (0.001 + math.tan(i) ** 2))), encoding="utf-8")
        for i in range(window_col_size)
    ]
)
datetime64_col = pd.Series(
    [
        None
        if 1 / math.cos(i) < -1
        else np.datetime64(f"201{(i**2)%10}-{1+(i**3)%12:02}-{1+(i**4)%15:02}")
        for i in range(window_col_size)
    ]
)


def col_to_window_df(cols):
    """Takes in a dictionary mapping column names to columns and produces a DataFrame
    containing those column as well as 4 others that can be used for window
    partitioning / ordering"""
    n = len(list(cols.values())[0])
    return {
        "table1": pd.DataFrame(
            {
                "W1": [round(i * 8 / n) for i in range(n)],
                "W2": [i % 7 for i in range(n)],
                "W3": [round(math.tan(i)) for i in range(n)],
                "W4": [i for i in range(n)],
                **cols,
            }
        )
    }


@pytest.fixture
def uint8_window_df():
    """Returns a DataFrame for window function testing using only uint8 data"""
    return col_to_window_df({"A": uint8_col})


@pytest.fixture(
    params=[
        pytest.param(uint8_col, id="uint8", marks=pytest.mark.slow),
        pytest.param(int32_col, id="int32", marks=pytest.mark.slow),
        pytest.param(int64_col, id="int64"),
        pytest.param(float64_col, id="float64", marks=pytest.mark.slow),
    ]
)
def numeric_types_window_df(request):
    """Returns a DataFrame for window function testing for several numeric types"""
    return col_to_window_df({"A": request.param})


@pytest.fixture
def all_numeric_window_df(request):
    """Same as numeric_types_window_df except htat it returns all the columns at once"""
    return col_to_window_df(
        {
            "U8": uint8_col,
            "I32": int32_col,
            "I64": int64_col,
            "F64": float64_col,
        }
    )


@pytest.fixture
def all_numeric_window_col_names(request):
    """Returns the data column names from all_numeric_window_df"""
    return {
        "U8": "200",
        "I32": "-12345",
        "I64": "-987654321",
        "F64": "2.718281828",
    }


@pytest.fixture(
    params=[
        pytest.param(uint8_col, id="uint8", marks=pytest.mark.slow),
        pytest.param(
            int32_col,
            id="int32",
        ),
        pytest.param(
            int64_col,
            id="int64",
        ),
        pytest.param(float64_col, id="float64", marks=pytest.mark.slow),
        pytest.param(boolean_col, id="boolean", marks=pytest.mark.slow),
        pytest.param(string_col, id="string"),
        pytest.param(binary_col, id="binary", marks=pytest.mark.slow),
        pytest.param(
            datetime64_col,
            id="datetime64",
        ),
    ]
)
def all_types_window_df(request):
    """Returns a DataFrame for window function testing for all major types"""
    return col_to_window_df({"A": request.param})


@pytest.fixture
def all_window_df():
    """Same as all_types_window_df, but returns them all in the same DataFrame."""
    return col_to_window_df(
        {
            "U8": uint8_col,
            "I32": int32_col,
            "I64": int64_col,
            "F64": float64_col,
            "BO": boolean_col,
            "ST": string_col,
            "BI": binary_col,
            "DT": datetime64_col,
            # [BE-3891] add Timedelta tests
            # [BE-3892] add Time tests
            # [BE-4032] add Timezone-aware tests
        }
    )


@pytest.fixture
def all_window_col_names():
    """Returns the data column names from all_window_df, each mapped to a
    string of a constant value of that type.

    This allows test functions to iterate across the names of the data columns
    in all_window_df, and also to extract a literal value corresponding to
    each of those column types (for LEAD/LAG testing)."""
    return {
        "U8": "255",
        "I32": "-1",
        "I64": "0",
        "F64": "3.1415926",
        "BO": "False",
        "ST": "'default'",
        "BI": "X'7fff'",
        "DT": "TIMESTAMP '2022-02-18'",
    }


@pytest.fixture(
    params=[
        pytest.param(
            (
                [
                    "PARTITION BY W2 ORDER BY W4 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
                    "PARTITION BY W2 ORDER BY W4 ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
                    "PARTITION BY W2 ORDER BY W4 ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING",
                ],
                1,
            ),
            id="prefix-suffix-rolling_5-fused",
        ),
        pytest.param(
            (
                [
                    "PARTITION BY W1 ORDER BY W4 ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING",
                    "PARTITION BY W1 ORDER BY W3 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING",
                    "PARTITION BY W2 % 5, W3 % 5 ORDER BY W4 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
                ],
                3,
            ),
            id="10_before-exclusive_suffix-whole_window-not_fused",
            marks=pytest.mark.slow,
        ),
    ]
)
def window_frames(request):
    """
    Returns a tuple of two things:
    - A list of window specifications that can be used for a window function call.
    - An integer indicating how many distinct groupby-apply calls are required
      for the list of windows after window fusion
    """
    return request.param
