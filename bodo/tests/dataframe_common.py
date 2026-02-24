import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import gen_nonascii_list

# TODO: other possible df types like dt64, td64, ...
df_value_params = [
    # int and float columns
    pytest.param(
        pd.DataFrame(
            {
                "A": [1, 8, 4, 11, -3],
                "B": [1.1, np.nan, 4.2, 3.1, -1.3],
                "C": [True, False, False, True, True],
            }
        ),
        marks=pytest.mark.slow,
        id="numeric_df",
    ),
    # Categorical columns
    pytest.param(
        pd.DataFrame(
            {
                "A": pd.Series(["AA", "BB", "", "AA", None], dtype="category"),
                "B": pd.Series([1, 2, 5, None, 5], dtype="category"),
                "C": pd.concat(
                    [
                        pd.Series(
                            pd.date_range(
                                start="2/1/2015", end="2/24/2021", periods=4, unit="ns"
                            )
                        ),
                        pd.Series(data=[None], index=[4]),
                    ]
                )
                .astype("datetime64[ns]")
                .astype("category"),
                "D": pd.concat(
                    [
                        pd.Series(
                            pd.timedelta_range(start="1 day", periods=4, unit="ns")
                        ),
                        pd.Series(data=[None], index=[4]),
                    ]
                )
                .astype("timedelta64[ns]")
                .astype("category"),
            }
        ),
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled,
            reason="[BSE-4764] Support conversion from categorical dtypes.",
        ),
        id="categorical_df",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "A": pd.array([1, 8, 4, 10, 3] * 2, dtype="Int32"),
                2: [1.1, np.nan, 4.2, 3.1, -1.3] * 2,
                "C": pd.array([True, False, False, None, True] * 2, dtype="boolean"),
            },
            ["A", "BA", "", "DD", "C", "e2", "#4", "32", "ec", "#43"],
        ),
        id="string_index",
    ),
    # uint8, float32 dtypes
    pytest.param(
        pd.DataFrame(
            {
                3: np.array([1, 8, 4, 0, 3], dtype=np.uint8),
                1: np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
            }
        ),
        marks=[
            pytest.mark.slow,
            pytest.mark.skipif(
                bodo.test_dataframe_library_enabled,
                reason="[BSE-4781] Integer column name support.",
            ),
        ],
        id="uint8_df",
    ),
    # string and int columns, float index
    pytest.param(
        pd.DataFrame(
            {
                "A": ["AA", None, "", "D", "GG", "FF"],
                "B": [1, 8, 4, -1, 2, 10],
                "C": gen_nonascii_list(6),
            },
            [-2.1, 0.1, 1.1, 7.1, 9.0, 7.7],
        ),
        marks=pytest.mark.slow,
        id="float_index",
    ),
    # range index
    pytest.param(
        pd.DataFrame(
            {"A": [1, 8, 4, 1, -2] * 3, "B": ["A", "B", "CG", "ACDE", "C"] * 3},
            range(0, 5 * 3, 1),
        ),
        marks=pytest.mark.slow,
        id="range_index",
    ),
    pytest.param(
        # TODO: parallel range index with start != 0 and stop != 1
        # int index
        pd.DataFrame(
            {"A": [1, 8, 4, 1, -3] * 2, "B": ["A", "B", "CG", "ACDE", "C"] * 2},
            [-2, 1, 3, 5, 9, -3, -5, 0, 4, 7],
        ),
        id="int_index",
    ),
    # string index
    pytest.param(
        pd.DataFrame({"A": [1, 2, 3, -1, 4]}, ["A", "BA", "", "DD", "C"]),
        marks=[
            pytest.mark.slow,
        ],
        id="string_index",
    ),
    # datetime column
    pytest.param(
        pd.DataFrame(
            {
                "A": pd.date_range(
                    start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
                ).astype("datetime64[ns]")
            }
        ),
        id="datetime_df",
    ),
    # datetime index
    pytest.param(
        pd.DataFrame(
            {"A": [3, 5, 1, -1, 4]},
            pd.date_range(
                start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
            ).astype("datetime64[ns]"),
        ),
        marks=pytest.mark.slow,
        id="datetime_index",
    ),
    # Binary column
    pytest.param(
        pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        b"",
                        b"abc",
                        b"c",
                        None,
                        b"ccdefg",
                        b"abcde",
                        b"poiu",
                        bytes(3),
                    ]
                    * 2
                )
            },
        ),
        id="binary_df",
        marks=pytest.mark.slow,
    ),
]


@pytest.fixture(params=df_value_params)
def df_value(request):
    return request.param


@pytest.fixture(
    params=[
        # int
        pytest.param(pd.DataFrame({"A": [1, 8, 4, 11, -3]}), marks=pytest.mark.slow),
        # int and float columns
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, 11, -3], 2: [1.1, np.nan, 4.2, 3.1, -1.1]}),
            marks=pytest.mark.slow,
        ),
        # uint8, float32 dtypes
        pd.DataFrame(
            {
                55: np.array([1, 8, 4, 0, 2], dtype=np.uint8),
                -3: np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
            }
        ),
        # pd.DataFrame({'A': np.array([1, 8, 4, 0], dtype=np.uint8),
        # }),
        # int column, float index
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, -1, 3]}, [-2.1, 0.1, 1.1, 7.1, 9.0]),
            marks=pytest.mark.slow,
        ),
        # range index
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, 1, -2]}, range(0, 5, 1)),
            marks=pytest.mark.slow,
        ),
        # datetime column
        pd.DataFrame(
            {
                "A": pd.date_range(
                    start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
                )
            }
        ),
        # datetime index
        pytest.param(
            pd.DataFrame(
                {"A": [3, 5, 1, -1, 2]},
                pd.date_range(
                    start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
                ),
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: timedelta
    ]
)
def numeric_df_value(request):
    return request.param


@pytest.fixture(
    params=[
        # column name overlaps with pandas function
        pd.DataFrame({"product": ["a", "b", "c", "d", "e", "f"]}),
        pd.DataFrame(
            {"product": ["a", "b", "c", "d", "e", "f"], "keys": [1, 2, 3, 4, 5, 6]}
        ),
    ]
)
def column_name_df_value(request):
    return request.param


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "a": [1, 2] * 20,
                "b": [True, False] * 20,
                "c": [1.0, 2.0] * 20,
                "d": pd.array([1.0, 2.0] * 20, "Float64"),
            }
        ),
    ]
)
def select_dtypes_df(request):
    return request.param


@pytest.fixture(
    params=[
        # array-like
        pytest.param([2, 3, 5], marks=pytest.mark.slow),
        pytest.param([2.1, 3.2, np.nan, 5.4], marks=pytest.mark.slow),
        pytest.param(["A", "C", "AB"], marks=pytest.mark.slow),
        # int array, no NA sentinel value
        pytest.param(np.array([2, 3, 5, -1, -4, 9]), marks=pytest.mark.slow),
        # float array with np.nan
        pytest.param(np.array([2.9, np.nan, 1.4, -1.1, -4.2]), marks=pytest.mark.slow),
        pd.Series([2.1, 5.3, np.nan, -1.0, -3.7], [3, 5, 6, -2, 4], name="C"),
        pytest.param(
            pd.Index([10, 12, 14, 17, 19], dtype="Int64", name="A"),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.RangeIndex(5), marks=pytest.mark.slow),
        # dataframe
        pd.DataFrame(
            {"A": ["AA", None, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
            [1.1, -2.1, 7.1, 0.1, 3.1],
        ),
        # scalars
        3,
        pytest.param(1.3, marks=pytest.mark.slow),
        np.nan,
        "ABC",
        None,
        np.datetime64("NaT"),
        pytest.param(np.timedelta64("NaT"), marks=pytest.mark.slow),
    ]
)
def na_test_obj(request):
    return request.param
