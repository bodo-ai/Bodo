# Copyright (C) 2023 Bodo Inc. All rights reserved.
import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo import Time
from bodo.tests.utils import check_func, nullable_float_arr_maker


@pytest.fixture(
    params=[
        pytest.param(np.arange(16, dtype=np.uint8), id="numpy_uint8"),
        pytest.param(
            pd.Series(range(-20, 21), dtype=pd.Int32Dtype()),
            id="nullable_int32-no_nulls",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [None if i % 3 == 0 else i**2 for i in range(25)],
                dtype=pd.Int32Dtype(),
            ),
            id="nullable_int32-with_nulls",
        ),
        pytest.param(
            pd.Series(
                [None] * 50,
                dtype=pd.Int32Dtype(),
            ),
            id="nullable_int64-all_nulls",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    None if np.cos(i) < -0.4 else int(10 / np.tan(i + 1)) ** 3
                    for i in range(10000)
                ],
                dtype=pd.Int64Dtype(),
            ),
            id="nullable_int64-large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            nullable_float_arr_maker(
                [i / 10 for i in range(25)],
                [3, 9, 15, 18],
                [1, 2, 4, 6, 21, 23],
            ),
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [[None, True, False][i % 3] for i in range(16)], dtype=pd.BooleanDtype()
            ),
            id="boolean",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [None if i % 5 == 2 else Decimal(f"{i}.{i}{i}") for i in range(20)]
            ),
            id="decimal",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    None if i % 3 == 0 else datetime.date.fromordinal(737500 + i**2)
                    for i in range(40)
                ]
            ),
            id="date",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
        pytest.param(
            pd.Series(
                [None if i % 7 < 2 else Time(nanosecond=3**i) for i in range(45)]
            ),
            id="time",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 5 == 2
                    else pd.Timestamp("2020-1-1") + pd.Timedelta(minutes=2**i)
                    for i in range(23)
                ]
            ),
            id="timestamp-naive",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 6 > 3
                    else pd.Timestamp("2021-4-1", tz="US/Pacific")
                    + pd.Timedelta(seconds=2**i)
                    for i in range(30)
                ]
            ),
            id="timestamp-tz",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 2 == 0
                    else ("" if i % 7 == 1 else f"{chr(65+i%10)}{i%7}")
                    for i in range(26)
                ]
            ),
            id="string",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    None
                    if i % 3 == 1
                    else bytes(str(i % 5) * (i % 3), encoding="UTF-8")
                    for i in range(24)
                ]
            ),
            id="binary",
            marks=pytest.mark.skip(
                reason="[BSE-1042] Support ARRAY_AGG on other dtypes"
            ),
        ),
    ]
)
def array_agg_data(request):
    """
    Creates a DataFrame with 3 columns to be used for testing array_agg:

    - keys: the keys to group by.
    - ordering: the data to sort by within each group.
    - data: the values that are to be placed in an array within each group after sorting.

    Currently only tests numeric data (decimals, integers, floats, booleans, both nullable and numpy).

    The distribution of data between various keys, as well as the values of the ordering column,
    are set up so that they will vary with different lengths of input data.
    """
    data = request.param
    keys = [
        "AABAABCBAABCDCBA"[int(10 * np.tan(i + len(data))) % 16]
        for i in range(len(data))
    ]
    ordering = [np.tan(i + len(data)) for i in range(len(data))]
    return pd.DataFrame({"key": keys, "order": ordering, "data": data}).sort_values(
        by=["key"]
    )


def array_agg_func(group):
    col = group.sort_values(by=("order"), ascending=(True,), na_position="last")["data"]
    col = col.dropna()
    return col.values


def test_array_agg_single_order(array_agg_data, memory_leak_check):
    """
    Tests ARRAY_AGG with a single ordering clause (no DISTINCT keyword).

    .sort_values() must be used to ensure correct ordering even when there are
    multiple ranks since check_func with sort_output=True is not supported
    on array data.
    """

    def impl(df):
        return (
            df.groupby(["key"], as_index=False, dropna=False)
            .agg(
                res=bodo.utils.utils.ExtendedNamedAgg(
                    column="data",
                    aggfunc="array_agg",
                    additional_args=(("order",), (True,), ("last",)),
                )
            )
            .sort_values(by=["key"])
        )

    keys = array_agg_data["key"].drop_duplicates()
    answers = []
    for key in keys:
        group = array_agg_data[array_agg_data["key"] == key]
        group_answer = array_agg_func(group)
        answers.append(group_answer)
    answer = pd.DataFrame({"key": keys, "res": answers})
    check_func(
        impl,
        (array_agg_data,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
    )


def test_array_agg_multiple_orders(memory_leak_check):
    """
    Tests ARRAY_AGG with multiple ordering clauses (no DISTINCT keyword).

    .sort_values() must be used to ensure correct ordering even when there are
    multiple ranks since check_func with sort_output=True is not supported
    on array data.
    """

    def impl(df):
        return (
            df.groupby(["key"], as_index=False, dropna=False)
            .agg(
                res=bodo.utils.utils.ExtendedNamedAgg(
                    column="data",
                    aggfunc="array_agg",
                    additional_args=(
                        (
                            "order_1",
                            "order_2",
                            "order_3",
                        ),
                        (True, False, False),
                        ("last", "first", "last"),
                    ),
                )
            )
            .sort_values(by=["key"])
        )

    df = pd.DataFrame(
        {
            "key": list("ABAA") * 10,
            "order_1": pd.Series(
                [[None, -1, 3, 0, None, 0, 3][i % 7] for i in range(40)],
                dtype=pd.Int32Dtype(),
            ),
            "order_2": ["A", "B", None, "A", None] * 8,
            "order_3": list(range(40)),
            "data": [i**2 for i in range(40)],
        },
    )
    answer = pd.DataFrame(
        {
            "key": ["A", "B"],
            "res": [
                [
                    484,
                    1296,
                    225,
                    64,
                    576,
                    361,
                    144,
                    961,
                    676,
                    1444,
                    100,
                    9,
                    1156,
                    729,
                    4,
                    256,
                    36,
                    900,
                    529,
                    400,
                    1521,
                    1024,
                    196,
                    49,
                    16,
                    121,
                    1225,
                    784,
                    324,
                    0,
                ],
                [841, 1, 289, 1089, 25, 1369, 81, 169, 441, 625],
            ],
        }
    )
    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
    )
