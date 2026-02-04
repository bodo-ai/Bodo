"""
Common import file for Series test fixtures
"""

import datetime
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo


@dataclass
class SeriesReplace:
    series: pd.Series
    to_replace: (int, str)
    value: (int, str)


@dataclass
class WhereNullable:
    series: pd.Series
    cond: pd.array
    other: pd.Series

    def __iter__(self):
        return iter([self.series, self.cond, self.other])

    def __len__(self):
        return 3


GLOBAL_VAL = 2


# using length of 5 arrays to enable testing on 3 ranks (2, 2, 1 distribution)
# zero length chunks on any rank can cause issues, TODO: fix
# TODO: other possible Series types like Categorical, dt64, td64, ...
series_val_params = [
    pytest.param(
        pd.Series(
            [
                Decimal("1.6"),
                Decimal("-0.2"),
                Decimal("44.2"),
                None,
                Decimal("0"),
            ]
            * 2,
            dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
        ),
        id="decimal",
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled, reason="[BSE 4789] Decimal support."
        ),
    ),
    pytest.param(pd.Series([1, 8, 4, 11, -3]), marks=pytest.mark.slow, id="integer"),
    pytest.param(
        pd.Series([True, False, False, True, True]),
        marks=pytest.mark.slow,
        id="boolean_nona",
    ),  # bool array without NA
    pytest.param(
        pd.Series([True, False, False, None, True] * 2, dtype="boolean"),
        id="boolean_withna",
    ),  # bool array with NA
    pytest.param(
        pd.Series([1, 8, 4, 0, 3], dtype=np.uint8),
        marks=pytest.mark.slow,
        id="uint8",
    ),
    pytest.param(pd.Series([1, 8, 4, 10, 3], dtype="Int32"), id="series_val5"),
    pytest.param(
        pd.Series([1, 8, 4, -1, 2], name="ACD"),
        marks=pytest.mark.slow,
        id="int32",
    ),
    pytest.param(
        pd.Series([1, 8, 4, 1, -3], [3, 7, 9, 2, 1]),
        marks=pytest.mark.slow,
        id="integer_with_index",
    ),
    pytest.param(
        pd.Series(
            [1.1, np.nan, 4.2, 3.1, -3.5], [3, 7, 9, 2, 1], name="float_with_index"
        ),
    ),
    pytest.param(
        pd.Series([1, 2, 3, -1, 6], ["A", "BA", "", "DD", "GGG"]),
        id="integer_with_string_index",
    ),
    pytest.param(
        pd.Series(["A", "B", "CDD", "AA", "GGG"]),
        marks=pytest.mark.slow,
        id="string",
    ),  # TODO: string with Null (np.testing fails)
    pytest.param(
        pd.Series(["A", "B", "CG", "ACDE", "C"], [4, 7, 0, 1, -2]),
        id="string_with_integer_index",
    ),
    pytest.param(
        pd.Series(
            pd.date_range(start="2018-04-24", end="2018-04-29", periods=5, unit="ns")
        ).astype("datetime64[ns]"),
        id="timestamp",
    ),
    pytest.param(
        pd.Series(
            pd.date_range(
                start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
            ).date
        ),
        id="date",
    ),
    pytest.param(
        pd.Series(
            [
                datetime.timedelta(3, 3, 3),
                datetime.timedelta(2, 2, 2),
                datetime.timedelta(1, 1, 1),
                None,
                datetime.timedelta(5, 5, 5),
            ]
        ).astype("timedelta64[ns]"),
        id="timedelta",
    ),
    pytest.param(
        pd.Series(
            [3, 5, 1, -1, 2],
            pd.date_range(
                start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
            ).astype("datetime64[ns]"),
        ),
        marks=pytest.mark.slow,
        id="integer_with_timestamp_index",
    ),
    pytest.param(
        pd.Series(
            [
                ["a", "bc", "éè", "日本人"],
                ["a", ";∞¥₤€"],
                ["aaa", "b", "cc", "~=[]()%+{}@;’"],
                None,
                ["xx", "yy", "#!$_&-"],
            ],
            dtype=pd.ArrowDtype(pa.list_(pa.large_string())),
        ),
        id="string_non_unicode",
    ),
    pytest.param(
        pd.Series(
            [[1, 2], [3], [5, 4, 6], None, [-1, 3, 4]],
            dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ),
        id="integer_array",
    ),
    pytest.param(
        pd.Series(["AA", "BB", "", "AA", None, "AA"] * 2, dtype="category"),
        id="categorical_string",
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled,
            reason="[BSE-4804] Categorical dtypes support.",
        ),
    ),
    pytest.param(
        pd.Series(pd.Categorical([1, 2, 5, None, 2] * 2, ordered=True)),
        id="categorical_integer_ordered",
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled,
            reason="[BSE-4804] Categorical dtypes support.",
        ),
    ),
    pytest.param(
        pd.concat(
            [
                pd.Series(
                    pd.date_range(
                        start="1/1/2018", end="1/10/2018", periods=9, unit="ns"
                    )
                ),
                pd.Series([None]),
            ]
        )
        .astype("datetime64[ns]")
        .astype("category"),
        id="categorical_timestamp",
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled,
            reason="[BSE-4804] Categorical dtypes support.",
        ),
    ),
    pytest.param(
        pd.concat(
            [
                pd.Series(pd.timedelta_range(start="1 day", periods=9, unit="ns")),
                pd.Series([None]),
            ]
        )
        .astype("timedelta64[ns]")
        .astype(pd.CategoricalDtype(ordered=True)),
        id="categorical_timestamp_ordered",
        marks=pytest.mark.skipif(
            bodo.test_dataframe_library_enabled,
            reason="[BSE-4804] Categorical dtypes support.",
        ),
    ),
    pytest.param(
        pd.Series(
            [b"", b"abc", b"c", None, b"ccdefg", b"abcde", b"poiu", bytes(3)] * 2
        ),
        id="binary",
    ),
]


@pytest.fixture(params=series_val_params)
def series_val(request):
    return request.param


# TODO: timedelta, period, tuple, etc.
@pytest.fixture(
    params=[
        pytest.param(pd.Series([1, 8, 4, 11, -3]), marks=pytest.mark.slow),
        pd.Series([1.1, np.nan, 4.1, 1.4, -2.1]),
        pytest.param(
            pd.Series([1, 8, 4, 10, 3], dtype=np.uint8), marks=pytest.mark.slow
        ),
        pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32"),
        pytest.param(
            pd.Series([1, 8, 4, -1, 2], [3, 7, 9, 2, 1], name="AAC"),
            marks=pytest.mark.slow,
        ),
        pd.Series(
            pd.date_range(
                start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
            ).astype("datetime64[ns]")
        ),
    ]
)
def numeric_series_val(request):
    return request.param


@pytest.fixture(
    params=[
        pd.Series([np.nan, -1.0, -1.0, 0.0, 78.0]),
        pd.Series([1.0, 2.0, 3.0, 42.3]),
        pd.Series([1, 2, 3, 42]),
        pytest.param(
            pd.Series([1, 2]),
            marks=pytest.mark.slow,
        ),
    ]
)
def series_stat(request):
    return request.param
