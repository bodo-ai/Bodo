# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Common import file for Series test fixtures
"""
import datetime
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
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


GLOBAL_VAL = 2


nullable_float_marker = pytest.mark.skipif(
    not bodo.libs.float_arr_ext._use_nullable_float,
    reason="nullable float not fully supported yet",
)


# using length of 5 arrays to enable testing on 3 ranks (2, 2, 1 distribution)
# zero length chunks on any rank can cause issues, TODO: fix
# TODO: other possible Series types like Categorical, dt64, td64, ...
@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    Decimal("1.6"),
                    Decimal("-0.2"),
                    Decimal("44.2"),
                    np.nan,
                    Decimal("0"),
                ]
                * 2
            ),
            id="series_val0",
        ),
        pytest.param(
            pd.Series([1, 8, 4, 11, -3]), marks=pytest.mark.slow, id="series_val1"
        ),
        pytest.param(
            pd.Series([True, False, False, True, True]),
            marks=pytest.mark.slow,
            id="series_val2",
        ),  # bool array without NA
        pytest.param(
            pd.Series([True, False, False, np.nan, True] * 2), id="series_val3"
        ),  # bool array with NA
        pytest.param(
            pd.Series([1, 8, 4, 0, 3], dtype=np.uint8),
            marks=pytest.mark.slow,
            id="series_val4",
        ),
        pytest.param(pd.Series([1, 8, 4, 10, 3], dtype="Int32"), id="series_val5"),
        pytest.param(
            pd.Series([1, 8, 4, -1, 2], name="ACD"),
            marks=pytest.mark.slow,
            id="series_val6",
        ),
        pytest.param(
            pd.Series([1, 8, 4, 1, -3], [3, 7, 9, 2, 1]),
            marks=pytest.mark.slow,
            id="series_val7",
        ),
        pytest.param(
            pd.Series(
                [1.1, np.nan, 4.2, 3.1, -3.5], [3, 7, 9, 2, 1], name="series_val8"
            ),
        ),
        pytest.param(
            pd.Series([1, 2, 3, -1, 6], ["A", "BA", "", "DD", "GGG"]), id="series_val9"
        ),
        pytest.param(
            pd.Series(["A", "B", "CDD", "AA", "GGG"]),
            marks=pytest.mark.slow,
            id="series_val10",
        ),  # TODO: string with Null (np.testing fails)
        pytest.param(
            pd.Series(["A", "B", "CG", "ACDE", "C"], [4, 7, 0, 1, -2]),
            id="series_val11",
        ),
        pytest.param(
            pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
            id="series_val12",
        ),
        pytest.param(
            pd.Series(
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date
            ),
            id="series_val13",
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
            ),
            id="series_val14",
        ),
        pytest.param(
            pd.Series(
                [3, 5, 1, -1, 2],
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            ),
            marks=pytest.mark.slow,
            id="series_val15",
        ),
        pytest.param(
            pd.Series(
                [
                    ["a", "bc", "éè", "日本人"],
                    ["a", ";∞¥₤€"],
                    ["aaa", "b", "cc", "~=[]()%+{}@;’"],
                    None,
                    ["xx", "yy", "#!$_&-"],
                ]
            ),
            id="series_val16",
        ),
        pytest.param(
            pd.Series([[1, 2], [3], [5, 4, 6], None, [-1, 3, 4]]),
            id="series_val17",
        ),
        pytest.param(
            pd.Series(["AA", "BB", "", "AA", None, "AA"] * 2, dtype="category"),
            id="series_val18",
        ),
        pytest.param(
            pd.Series(pd.Categorical([1, 2, 5, None, 2] * 2, ordered=True)),
            id="series_val19",
        ),
        pytest.param(
            pd.concat(
                [
                    pd.Series(
                        pd.date_range(start="1/1/2018", end="1/10/2018", periods=9)
                    ),
                    pd.Series([None]),
                ]
            ).astype("category"),
            id="series_val20",
        ),
        pytest.param(
            pd.concat(
                [
                    pd.Series(pd.timedelta_range(start="1 day", periods=9)),
                    pd.Series([None]),
                ]
            ).astype(pd.CategoricalDtype(ordered=True)),
            id="series_val21",
        ),
        pytest.param(
            pd.Series(
                [b"", b"abc", b"c", np.nan, b"ccdefg", b"abcde", b"poiu", bytes(3)] * 2
            ),
            id="series_val22",
        ),
    ]
)
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
        pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
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
