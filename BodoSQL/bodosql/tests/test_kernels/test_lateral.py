import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import check_func
from bodo.utils.typing import ColNamesMetaType, MetaType


def simulate_lateral_flatten_array(
    df, keep_cols, explode_col, output_idx, output_val, output_this, outer
):
    """
    Generates the expected results for a LATERAL FLATTEN operation on an array.

    Args:
        df (pd.DataFrame): the DataFrame whose rows are to be exploded
        keep_cols (Tuple[integer]): the columns that are to be kept during the operation
        explode_col (integer): which column is the array column that is to be exploded
        output_idx (bool): whether to include the indices within the inner array in the output
        output_val (bool): whether to include the values within the inner array in the output
        output_this (bool): whether to include the replicated values of the exploded column
        outer (bool): if true, ensure 1 row is generated even when the explode col is null/empty
    """
    keep_cols_idx = df.columns[list(keep_cols)]
    df_subset = df.loc[:, keep_cols_idx]
    column_to_explode = df.iloc[:, explode_col]
    out_dict = {}
    if output_idx:
        out_dict["idx"] = []
    if output_val:
        out_dict["val"] = []
    if output_this:
        out_dict["this"] = []
    for i in range(len(keep_cols)):
        out_dict[i] = []
    for i in range(len(df)):
        sub_arr = column_to_explode.iloc[i]
        explode_length = 0 if sub_arr is None or sub_arr is pd.NA else len(sub_arr)
        dummy_row = outer and (sub_arr is None or sub_arr is pd.NA or len(sub_arr) == 0)
        if dummy_row:
            explode_length = 1
        if explode_length == 0:
            continue
        for j in range(len(keep_cols)):
            val = df_subset.iloc[i, j]
            for _ in range(explode_length):
                out_dict[j].append(val)
        if output_idx:
            if dummy_row:
                out_dict["idx"].append(pd.NA)
            else:
                out_dict["idx"].extend(list(range(explode_length)))
        if output_val:
            if dummy_row:
                out_dict["val"].append(pd.NA)
            else:
                out_dict["val"].extend(sub_arr)
        if output_this:
            out_dict["this"].extend([sub_arr] * explode_length)
    for i in range(len(keep_cols)):
        out_dict[i] = pd.Series(out_dict[i], dtype=df_subset.iloc[:, i].dtype)
    return pd.DataFrame(out_dict)


def simulate_lateral_flatten_json(
    df, keep_cols, explode_col, output_key, output_val, output_this, outer
):
    """
    Generates the expected results for a LATERAL FLATTEN operation on a JSON object.

    Args:
        df (pd.DataFrame): the DataFrame whose rows are to be exploded
        keep_cols (Tuple[integer]): the columns that are to be kept during the operation
        explode_col (integer): which column is the array column that is to be exploded
        output_key (bool): whether to include the keys from the JSON key-value pairs
        output_val (bool): whether to include the values from the JSON key-value pairs
        output_this (bool): whether to include the replicated values of the exploded column
        outer (bool): if true, ensure 1 row is generated even when the explode col is null/empty
    """
    keep_cols_idx = df.columns[list(keep_cols)]
    df_subset = df.loc[:, keep_cols_idx]
    column_to_explode = df.iloc[:, explode_col]
    out_dict = {}
    if output_key:
        out_dict["key"] = []
    if output_val:
        out_dict["val"] = []
    if output_this:
        out_dict["this"] = []
    for i in range(len(keep_cols)):
        out_dict[i] = []
    for i in range(len(df)):
        json_obj = column_to_explode.iloc[i]
        keys = []
        vals = []
        explode_length = 0
        if json_obj is not None and json_obj is not pd.NA:
            explode_length = len(json_obj)
            json_obj = dict(json_obj)
            for k, v in json_obj.items():
                keys.append(pd.NA if k is None else k)
                vals.append(pd.NA if v is None else v)
        if outer and (json_obj is None or json_obj is pd.NA or len(json_obj) == 0):
            keys.append(pd.NA)
            vals.append(pd.NA)
            explode_length = 1
        if explode_length == 0:
            continue
        for j in range(len(keep_cols)):
            val = df_subset.iloc[i, j]
            for _ in range(explode_length):
                out_dict[j].append(val)
        if output_key:
            out_dict["key"].extend(keys)
        if output_val:
            out_dict["val"].extend(vals)
        if output_this:
            out_dict["this"].extend([json_obj] * explode_length)
    if output_this:
        out_dict["this"] = pd.Series(out_dict["this"], dtype=column_to_explode.dtype)
    for i in range(len(keep_cols)):
        out_dict[i] = pd.Series(out_dict[i], dtype=df_subset.iloc[:, i].dtype)
    return pd.DataFrame(out_dict)


@pytest.mark.parametrize(
    "outer, output_idx, output_val, output_this",
    [
        pytest.param(
            True,
            False,
            False,
            False,
            id="output_nothing-with_outer",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            False, False, True, False, id="output_value", marks=pytest.mark.slow
        ),
        pytest.param(False, True, True, True, id="output_all"),
        pytest.param(True, True, True, True, id="output_all-with_outer"),
    ],
)
@pytest.mark.parametrize(
    "keep_cols",
    [
        pytest.param((0,), id="keep_int", marks=pytest.mark.slow),
        pytest.param((0, 1, 2, 4), id="keep_all_but_string_array"),
    ],
)
@pytest.mark.parametrize(
    "explode_col",
    [
        pytest.param(1, id="explode_integer"),
        pytest.param(3, id="explode_string"),
    ],
)
def test_lateral_flatten_array(
    explode_col,
    keep_cols,
    output_idx,
    output_val,
    output_this,
    outer,
    memory_leak_check,
):
    """
    Tests the lateral_flatten kernel with exploding array columns.
    """

    df = pd.DataFrame(
        {
            "a": pd.Series(range(10), dtype=np.int64),
            "b": pd.Series(
                [
                    [1],
                    None,
                    [4, 5, 6],
                    [],
                    [7],
                    [8, 9],
                    [10, 11, 12],
                    [13],
                    [14, None],
                    [15],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
            "c": "A,BCD,A,FG,HIJKL,,MNOPQR,S,FG,U".split(","),
            "d": pd.Series(
                [
                    ["A", "B"],
                    [],
                    ["CDE", "", "", "F"],
                    [],
                    ["GHI"],
                    ["J", "KL"],
                    ["MNO", "PQRS", "TUVWX"],
                    ["Y"],
                    [],
                    ["Z"],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            "e": pd.Series(
                [1, 2, None, 8, 16, 32, 64, 128, None, 512], dtype=pd.Int32Dtype()
            ),
        }
    )
    answer = simulate_lateral_flatten_array(
        df, keep_cols, explode_col, output_idx, output_val, output_this, outer
    )
    global_1 = MetaType((0, 1, 2, 3, 4))
    global_2 = MetaType(keep_cols)
    global_3 = ColNamesMetaType(tuple(answer.columns))
    global_4 = MetaType((False, False, False, output_idx, output_val, output_this))

    def impl(df1):
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodosql.kernels.lateral.lateral_flatten(
            T1, global_2, explode_col, global_4, outer
        )
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T2), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), index_1, global_3)
        return df2

    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "output_key, output_val, output_this",
    [
        pytest.param(False, True, False, id="output_value", marks=pytest.mark.slow),
        pytest.param(True, True, True, id="output_all"),
    ],
)
@pytest.mark.parametrize(
    "outer, keep_cols",
    [
        pytest.param(False, (2,), id="keep_string"),
        pytest.param(True, (0, 1, 2), id="keep_all-with_outer"),
    ],
)
@pytest.mark.parametrize(
    "explode_col_values, val_type",
    [
        pytest.param(
            [
                {"hex": "c93434", "name": "pomegranate"},
                {"hex": "00a4b4", "name": "peacock"},
                {"hex": "ffd700", "name": "gold"},
                {"hex": "7b6d8d", "name": "ultraviolet"},
                {"hex": None, "name": "midnight"},
            ],
            pa.struct([pa.field("hex", pa.string()), pa.field("name", pa.string())]),
            id="explode_struct-string",
        ),
        pytest.param(
            [
                {"ratings": [5.0, 5.0, 3.1, 5.0], "scores": [96.3]},
                {"ratings": [1.0, 4.4], "scores": [45.1, None]},
                {"ratings": [2.0, 5.0, 2.0], "scores": [88.2, 70.5, 87.5]},
                {"ratings": [3.0, 1.0], "scores": [94.5]},
                {"ratings": None, "scores": [None]},
            ],
            pa.struct(
                [
                    pa.field("ratings", pa.list_(pa.float64())),
                    pa.field("scores", pa.list_(pa.float64())),
                ]
            ),
            id="explode_struct-float_arrays",
        ),
        pytest.param(
            [
                {
                    "states": ["CA", "NV", "WA", "OR"],
                    "cities": ["San Francisco", "Las Vegas", "Seattle"],
                },
                {"states": ["IL", None, "MI"], "cities": ["Chicago", "Detroit"]},
                {
                    "states": ["NY", "PA", "NJ"],
                    "cities": ["New York City", "Pittsburgh", "Newark", "Philadelphia"],
                },
                {
                    "states": ["GA", "SC", "TN", "KY"],
                    "cities": ["Atlanta", "Charleston", "Nashville", "Louisville"],
                },
                {"states": [], "cities": ["Omaha"]},
            ],
            pa.struct(
                [
                    pa.field("states", pa.list_(pa.string())),
                    pa.field("cities", pa.list_(pa.string())),
                ]
            ),
            id="explode_struct-string_arrays",
        ),
        pytest.param(
            [
                {"A": [[0], [], [1, 2]], "B": [[3, 4], [5, 6]]},
                {"A": [[None, 7]], "B": [[8], [None], [9]]},
                {"A": [[10, 11, 12], [13]], "B": [[14], [15, 16], [17], [18, 19]]},
                {"A": [[20]], "B": [[21, 22, None], [23, None, None]]},
                {"A": [[24], [25], [26], [27]], "B": None},
            ],
            pa.struct(
                [
                    pa.field(
                        "A",
                        pa.list_(pa.list_(pa.int64())),
                        pa.field("cities", pa.list_(pa.list_(pa.int64()))),
                    )
                ]
            ),
            id="explode_struct-double_nested_int_arrays",
        ),
        pytest.param(
            [
                {"A": 65, "D": 68, "G": 71, "J": 74},
                {"B": 66, "E": None, "H": 72},
                {"C": 67, "I": 73},
                {"F": 70},
                {},
            ],
            pa.map_(pa.string(), pa.int64()),
            id="explode-map_int",
        ),
        pytest.param(
            [
                {
                    "RPDR_S8": {"name": "BTDQ", "rank": 1, "dates": None},
                    "RPDRAS_S3": {"name": "TM", "rank": 1, "dates": []},
                },
                {
                    "RPDRAS_S2": {
                        "name": "KZ",
                        "rank": 2,
                        "dates": [datetime.date(2023, 1, 1)],
                    },
                    "RPDR_S2": None,
                },
                {"RPDR_S7": {"name": "VC", "rank": 1, "dates": [None]}},
                {
                    "RPDR_S1": {
                        "name": "PC",
                        "rank": 9,
                        "dates": [
                            datetime.date(2023, 7, 4),
                            datetime.date(2023, 10, 15),
                            datetime.date(2023, 4, 1),
                        ],
                    },
                    "RPDR_S12": {"name": "WVD", "rank": 7, "dates": None},
                    "RPDR_S10": {
                        "name": "VVM",
                        "rank": 14,
                        "dates": [datetime.date(1999, 12, 31), None],
                    },
                },
                {},
            ],
            pa.map_(
                pa.string(),
                pa.struct(
                    [
                        pa.field("name", pa.string()),
                        pa.field("rank", pa.int32()),
                        pa.field("dates", pa.large_list(pa.date32())),
                    ]
                ),
            ),
            id="explode_map-struct_string_int_date_list",
        ),
    ],
)
def test_lateral_flatten_json(
    explode_col_values,
    val_type,
    keep_cols,
    output_key,
    output_val,
    output_this,
    outer,
    memory_leak_check,
):
    """
    Tests the lateral_flatten kernel with exploding array columns.
    """

    df = pd.DataFrame(
        {
            "a": pd.Series(range(13), dtype=np.int64),
            "b": pd.Series(
                [
                    explode_col_values[0],
                    explode_col_values[1],
                    None,
                    explode_col_values[2],
                    explode_col_values[3],
                    explode_col_values[2],
                    explode_col_values[4],
                    explode_col_values[1],
                    explode_col_values[0],
                    None,
                    None,
                    explode_col_values[3],
                    explode_col_values[4],
                ],
                dtype=pd.ArrowDtype(val_type),
            ),
            "c": "A,BCD,A,FG,HIJKL,,MNOPQR,S,FG,U,VW,XYZ,".split(","),
        }
    )
    answer = simulate_lateral_flatten_json(
        df, keep_cols, 1, output_key, output_val, output_this, outer
    )
    global_1 = MetaType((0, 1, 2))
    global_2 = MetaType(keep_cols)
    global_3 = ColNamesMetaType(tuple(answer.columns))
    global_4 = MetaType((False, output_key, False, False, output_val, output_this))

    def impl(df1):
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodosql.kernels.lateral.lateral_flatten(T1, global_2, 1, global_4, outer)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T2), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), index_1, global_3)
        return df2

    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        check_names=False,
    )


def test_lateral_flatten_with_array_agg(memory_leak_check):
    """
    Tests mixing the lateral_flatten kernel with array_agg
    in the following manner:
    1. Flatten an array item array of integers to remove any empty
      rows
    2. Call array agg to reconstruct the arrays but without the null entries
    3. Repeat step 1, which will now exclude any rows that originally had
       only nulls
    4. Repeat step 2, which will now only include the rows that originally
       had at least 1 non-null element
    """
    df = pd.DataFrame(
        {
            "a": pd.Series(range(15), dtype=np.int64),
            "b": pd.Series(
                [
                    [1],
                    [2, None, 3],
                    [4, 5, 6, None],
                    [],
                    [None],
                    [7],
                    [8, 9],
                    [None, None],
                    [10, 11, 12],
                    [],
                    [14, 15],
                    [None, None, None],
                    [16],
                    [None, 17],
                    [18],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
        }
    )
    answer = pd.DataFrame(
        {
            "a": pd.Series([0, 1, 2, 5, 6, 8, 10, 12, 13, 14]),
            "b": pd.Series(
                [
                    [1],
                    [2, 3],
                    [4, 5, 6],
                    [7],
                    [8, 9],
                    [10, 11, 12],
                    [14, 15],
                    [16],
                    [17],
                    [18],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
        }
    )
    global_1 = MetaType((0, 1))
    global_2 = MetaType((0,))
    global_3 = ColNamesMetaType(tuple(answer.columns)[::-1])
    global_4 = MetaType((False, False, False, False, True, False))

    def impl(df1):
        # First lateral flatten array
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodosql.kernels.lateral.lateral_flatten(T1, global_2, 1, global_4, False)
        # First array_agg
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T2), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), index_1, global_3)
        df3 = df2.groupby(["a"], as_index=False, dropna=False).agg(
            b=bodo.utils.utils.ExtendedNamedAgg(
                column="b",
                aggfunc="array_agg",
                additional_args=(("b",), (True,), ("last",)),
            )
        )
        # Second lateral flatten array
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df3), (), global_1, 2
        )
        T4 = bodosql.kernels.lateral.lateral_flatten(T3, global_2, 1, global_4, False)
        # Second array_agg
        index_2 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T4), 1, None)
        df4 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T4,), index_2, global_3)
        df5 = df4.groupby(["a"], as_index=False, dropna=False).agg(
            b=bodo.utils.utils.ExtendedNamedAgg(
                column="b",
                aggfunc="array_agg",
                additional_args=(("b",), (True,), ("last",)),
            )
        )
        return df5

    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        check_names=False,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_lateral_streaming(memory_leak_check):
    global_1 = MetaType((0, 1))
    global_2 = MetaType((0,))
    global_3 = MetaType(
        (
            False,
            False,
            False,
            False,
            True,
            False,
        )
    )
    global_4 = ColNamesMetaType(("B_Flat", "A_Rep"))
    batch_size = 3

    def impl(df1):
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            3, bodo.libs.memory_budget.OperatorType.ACCUMULATE_TABLE, 0, 0, -1
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 3
        )
        __bodo_is_last_streaming_output_1 = False
        _iter_1 = 0
        _temp1 = bodo.hiframes.table.local_len(T1)
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(3)
        )
        while not (__bodo_is_last_streaming_output_1):
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            T3 = bodosql.kernels.lateral.lateral_flatten(
                T2, global_2, 1, global_3, False
            )
            __bodo_is_last_streaming_output_1 = (_iter_1 * batch_size) >= _temp1
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, T3
            )
            _iter_1 = _iter_1 + 1
        T4 = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T4), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T4,), index_1, global_4)
        return df2

    df = pd.DataFrame(
        {
            "A": list(range(10)),
            "B": [
                [],
                [1],
                [2, 3],
                [4, 5, 6],
                [],
                [7],
                [8, 9],
                [10, 11, 12, 13, 14],
                [],
                [15],
            ],
        }
    )
    answer = pd.DataFrame(
        {
            "B_Flat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "A_Rep": [1, 2, 2, 3, 3, 3, 5, 6, 6, 7, 7, 7, 7, 7, 9],
        }
    )

    check_func(
        impl,
        (df,),
        py_output=answer,
        reset_index=True,
        check_dtype=False,
        check_names=False,
        sort_output=True,
    )
