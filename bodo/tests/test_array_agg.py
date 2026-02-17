import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, nullable_float_arr_maker


@pytest.fixture(
    params=[
        pytest.param(
            (np.arange(16, dtype=np.uint8) % 5, pa.large_list(pa.int8())),
            id="numpy_uint8",
        ),
        pytest.param(
            (
                pd.Series(range(-20, 21), dtype=pd.Int32Dtype()) % 5,
                pa.large_list(pa.int32()),
            ),
            id="nullable_int32-no_nulls",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [None if i % 3 == 0 else i**2 for i in range(25)],
                    dtype=pd.Int32Dtype(),
                ),
                pa.large_list(pa.int32()),
            ),
            id="nullable_int32-with_nulls",
        ),
        pytest.param(
            (
                pd.Series(
                    [None] * 50,
                    dtype=pd.Int64Dtype(),
                ),
                pa.large_list(pa.int64()),
            ),
            id="nullable_int64-all_nulls",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if np.cos(i) < -0.4 else int(10 / np.tan(i + 1)) ** 3
                        for i in range(10000)
                    ],
                    dtype=pd.Int64Dtype(),
                ),
                pa.large_list(pa.int64()),
            ),
            id="nullable_int64-large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            lambda: (
                nullable_float_arr_maker(
                    [i / 10 for i in range(25)],
                    [3, 9, 15, 18],
                    [-1],
                ),
                pa.large_list(pa.float64()),
            ),
            id="float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [[None, True, False][i % 3] for i in range(16)],
                    dtype=pd.BooleanDtype(),
                ),
                pa.large_list(pa.bool_()),
            ),
            id="boolean",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [None if i % 5 == 2 else Decimal(f"{i}.{i}{i}") for i in range(20)]
                ),
                pa.large_list(pa.decimal128(38, 18)),
            ),
            id="decimal",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if i % 3 == 0 else datetime.date.fromordinal(737500 + i**2)
                        for i in range(40)
                    ]
                ),
                pa.large_list(pa.date32()),
            ),
            id="date",
        ),
        pytest.param(
            lambda: (
                pd.Series(
                    [
                        None if i % 7 < 2 else bodo.types.Time(microsecond=3**i)
                        for i in range(45)
                    ]
                ),
                pa.large_list(pa.time64("ns")),
            ),
            id="time",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 5 == 2
                        else pd.Timestamp("2020-1-1") + pd.Timedelta(minutes=2**i)
                        for i in range(23)
                    ]
                ),
                pa.large_list(pa.timestamp("ns")),
            ),
            id="timestamp_ntz",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 6 > 3
                        else pd.Timestamp("2021-4-1", tz="US/Pacific")
                        + pd.Timedelta(seconds=2**i)
                        for i in range(30)
                    ]
                ),
                pa.large_list(pa.timestamp("ns", "US/Pacific")),
            ),
            id="timestamp_ltz",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 2 == 0
                        else ("" if i % 7 == 1 else f"{chr(65 + i % 10)}{i % 7}")
                        for i in range(26)
                    ]
                ),
                pa.large_list(pa.large_string()),
            ),
            id="string-ascii-with_nulls",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        "".join(
                            [
                                "AIU46♬🐍💚㗨⅛€∞"[(i + j) % 12]
                                for j in range(np.int64(np.abs(2.5 + np.tan(i))))
                            ]
                        )
                        for i in range(100)
                    ]
                ),
                pa.large_list(pa.large_string()),
            ),
            id="string-non_ascii-no_nulls",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 3 == 1
                        else bytes(str(i % 5) * (i % 3), encoding="UTF-8")
                        for i in range(24)
                    ]
                ),
                pa.large_list(pa.binary()),
            ),
            id="binary",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if i % 5 == 4 else [1, 2, None, 3, 4, 5][: i % 6]
                        for i in range(25)
                    ],
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
                pa.large_list(pa.large_list(pa.int64())),
            ),
            id="array_integer",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 10 == 7
                        else [
                            ("AABAABCBAABCDCBA" * 2)[i + j : i + j + 5]
                            for j in range(i % 6)
                        ]
                        for i in range(500)
                    ],
                    dtype=pd.ArrowDtype(pa.large_list(pa.string())),
                ),
                pa.large_list(pa.large_list(pa.string())),
            ),
            id="array_string",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 11 == 10
                        else {
                            "id": None if i % 10 == 7 else (i % 25) * 3,
                            "tags": None
                            if i % 9 == 8
                            else ["A", None, "E", "I", "O", "U"][: i % 7],
                        }
                        for i in range(400)
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("id", pa.int8()),
                                pa.field("tags", pa.large_list(pa.string())),
                            ]
                        )
                    ),
                ),
                pa.large_list(
                    pa.struct(
                        [
                            pa.field("id", pa.int8()),
                            pa.field("tags", pa.large_list(pa.string())),
                        ]
                    )
                ),
            ),
            id="struct",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 10 == 0
                        else dict(
                            zip(
                                [chr(65 + i % 9 + j) for j in range(i % 7)],
                                [
                                    None if (i + j) % 10 == 9 else i**j
                                    for j in range(i % 7)
                                ],
                            )
                        )
                        for i in range(450)
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
                pa.large_list(pa.map_(pa.string(), pa.int64())),
            ),
            id="map",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if i % 41 == 0
                        else {
                            "loc": {
                                str(j % 10): None
                                if j % 7 == 6
                                else {
                                    str(k % 10): None if j == k else (j + k) % 20
                                    for k in range(-j, j, i)
                                }
                                for j in range(i, i + (i**2) % 4)
                            },
                            "atr": [
                                None
                                if (i + j) % 5 == 2
                                else {
                                    "name": chr(65 + (i + j) % 6),
                                    "value": chr(65 + (j - i) % 5),
                                }
                                for j in range(i % 7)
                            ],
                        }
                        for i in range(450)
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field(
                                    "loc",
                                    pa.map_(
                                        pa.string(), pa.map_(pa.string(), pa.int64())
                                    ),
                                ),
                                pa.field(
                                    "atr",
                                    pa.large_list(
                                        pa.struct(
                                            [
                                                pa.field("name", pa.string()),
                                                pa.field("value", pa.string()),
                                            ]
                                        )
                                    ),
                                ),
                            ]
                        )
                    ),
                ),
                pa.large_list(
                    pa.struct(
                        [
                            pa.field(
                                "loc",
                                pa.map_(pa.string(), pa.map_(pa.string(), pa.int64())),
                            ),
                            pa.field(
                                "atr",
                                pa.large_list(
                                    pa.struct(
                                        [
                                            pa.field("name", pa.string()),
                                            pa.field("value", pa.string()),
                                        ]
                                    )
                                ),
                            ),
                        ]
                    )
                ),
            ),
            id="multi_nested",
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
    # Import compiler for bodo.types
    import bodo.decorators  # noqa

    val = request.param
    if callable(val):
        val = val()

    data, array_dtype = val
    keys = [
        "AABAABCBAABCDCBA"[int(10 * np.tan(i + len(data))) % 16]
        for i in range(len(data))
    ]
    ordering = [np.tan(i + len(data)) for i in range(len(data))]
    return pd.DataFrame({"key": keys, "order": ordering, "data": data}), array_dtype


def array_agg_func(group):
    order = group["order"].set_axis(range(len(group)))
    order = order.sort_values(ascending=True, na_position="last").index
    col = group["data"].iloc[order]
    col = col.dropna()
    return col.array


def test_array_agg_regular(array_agg_data, memory_leak_check):
    """
    Tests ARRAY_AGG with a single ordering clause (no DISTINCT keyword).

    .sort_values() must be used to ensure correct ordering even when there are
    multiple ranks since check_func with sort_output=True is not supported
    on array data.
    """

    def impl(df):
        return df.groupby(["key"], as_index=False, dropna=False).agg(
            res=bodo.utils.utils.ExtendedNamedAgg(
                column="data",
                aggfunc="array_agg",
                additional_args=(("order",), (True,), ("last",)),
            )
        )

    data, array_dtype = array_agg_data

    keys = data["key"].drop_duplicates()
    answers = []
    for key in keys:
        group = data[data["key"] == key]
        group_answer = array_agg_func(group)
        answers.append(group_answer)
    answer = pd.DataFrame(
        {"key": keys, "res": pd.array(answers, dtype=pd.ArrowDtype(array_dtype))}
    )
    check_func(
        impl,
        (data,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        sort_output=True,
        convert_columns_to_pandas=True,
    )


def array_agg_distinct_func(group):
    def _is_nested_pa_dtype(pa_dtype):
        return (
            pa.types.is_list(pa_dtype)
            or pa.types.is_large_list(pa_dtype)
            or pa.types.is_struct(pa_dtype)
            or pa.types.is_map(pa_dtype)
        )

    if isinstance(group["data"].dtype, pd.ArrowDtype) and _is_nested_pa_dtype(
        group["data"].dtype.pyarrow_dtype
    ):
        # Using bodo.jit since its implementation of sorting and dropna
        # works on semi-structured data better than the Python/Pandas version.
        @bodo.jit(distributed=False)
        def bodo_sort(to_order):
            return to_order.sort_values(ascending=True, na_position="last").index

        order = bodo_sort(group["data"].set_axis(range(len(group))))
        sorted_col_without_na = group["data"].iloc[order].dropna()
        # Hacky semi-structured safe implementation of drop_duplicates
        as_str = bodo.tests.utils.convert_non_pandas_columns(
            pd.DataFrame({"A": sorted_col_without_na})
        )["A"]
        as_str[sorted_col_without_na.apply(lambda x: len(x) == 0)] = "<EMPTY>"
        mask = as_str != as_str.shift(1)
        return sorted_col_without_na[mask].array
    else:
        order = group["data"].set_axis(range(len(group)))
        order = order.sort_values(ascending=True, na_position="last").index
        col = group["data"].iloc[order]
        return col.dropna().drop_duplicates().array


def test_array_agg_distinct(array_agg_data, memory_leak_check):
    """
    Tests ARRAY_AGG with the DISTINCT keyword.

    .sort_values() must be used to ensure correct ordering even when there are
    multiple ranks since check_func with sort_output=True is not supported
    on array data.
    """

    def impl(df):
        return df.groupby(["key"], as_index=False, dropna=False).agg(
            res=bodo.utils.utils.ExtendedNamedAgg(
                column="data",
                aggfunc="array_agg_distinct",
                additional_args=(("data",), (True,), ("last",)),
            )
        )

    data, array_dtype = array_agg_data

    keys = data["key"].drop_duplicates()
    answers = []
    for key in keys:
        group = data[data["key"] == key]
        group_answer = array_agg_distinct_func(group)
        answers.append(group_answer)
    answer = pd.DataFrame(
        {"key": keys, "res": pd.array(answers, dtype=pd.ArrowDtype(array_dtype))}
    )
    check_func(
        impl,
        (data,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        sort_output=True,
        convert_columns_to_pandas=True,
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


def test_array_agg_any_value(memory_leak_check):
    """
    Tests ARRAY_AGG followed by a call to ANY_VALUE.
    """

    def impl(df):
        arr_agg_res = df.groupby(["key_a", "key_b"], as_index=False, dropna=False).agg(
            arr_data_i=bodo.utils.utils.ExtendedNamedAgg(
                column="data_i",
                aggfunc="array_agg",
                additional_args=(("order",), (True,), ("last",)),
            ),
            arr_data_s=bodo.utils.utils.ExtendedNamedAgg(
                column="data_s",
                aggfunc="array_agg",
                additional_args=(("order",), (True,), ("last",)),
            ),
        )
        any_val_res = (
            arr_agg_res.groupby(["key_a"], as_index=False, dropna=False)
            .agg(
                res_i=pd.NamedAgg("arr_data_i", aggfunc="first"),
                res_s=pd.NamedAgg("arr_data_s", aggfunc="first"),
            )
            .sort_values(by=["key_a"])
        )
        return any_val_res

    df = pd.DataFrame(
        {
            "key_a": list("BAABAAABACACCAA"),
            "key_b": list("xxzyxyzzzzyyxxy"),
            "order": pd.Series([9, 2, 6, 10, 1, 4, 8, 11, 7, 14, 5, 13, 12, 0, -1]),
            "data_i": pd.Series(
                [None, 2, 0, None, 1, 1, 2, None, 1, -1, 2, -1, -1, 0, 0],
                dtype=pd.Int32Dtype(),
            ),
            "data_s": pd.Series(
                [
                    None,
                    "2.718",
                    "0.0",
                    None,
                    "3.14",
                    "3.14",
                    "2.718",
                    None,
                    "3.14",
                    "-1.5",
                    "2.718",
                    "-1.5",
                    "-1.5",
                    "0.0",
                    "0.0",
                ],
            ),
        },
    )
    answer = pd.DataFrame(
        {
            "key_a": ["A", "B", "C"],
            "res_i": pd.array(
                [[0, 1, 2], [], [-1]], dtype=pd.ArrowDtype(pa.large_list(pa.int64()))
            ),
            "res_s": pd.array(
                [["0.0", "3.14", "2.718"], [], ["-1.5"]],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
        }
    )

    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
    )
