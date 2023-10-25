# Copyright (C) 2023 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import ColNamesMetaType, MetaType


def simulate_lateral_flatten(
    df, keep_cols, explode_col, output_idx, output_val, output_this
):
    """
    Generates the expected results for a LATERAL FLATTEN operation on an array.

    Args:
        df (pd.DataFrame): the DataFrame whose rows are to be exploded
        keep_cols (Tuple[integer]): the columns that are to be kept during the operation
        explode_col (integer): which column is the array column that is to be exploded
        output_idx (bool): whether to include the indices within the inner array in the output
        output_val (bool): whether to include the values within the inner array in the output
    """
    keep_cols_idx = df.columns[list(keep_cols)]
    df_subset = df.loc[:, keep_cols_idx]
    column_to_explode = df.iloc[:, explode_col]
    out_dict = {i: [] for i in range(len(keep_cols))}
    if output_idx:
        out_dict["idx"] = []
    if output_val:
        out_dict["val"] = []
    if output_this:
        out_dict["this"] = []
    for i in range(len(df)):
        sub_arr = column_to_explode.iloc[i]
        explode_length = 0 if sub_arr is None else len(sub_arr)
        if explode_length == 0:
            continue
        for j in range(len(keep_cols)):
            val = df_subset.iloc[i, j]
            for _ in range(explode_length):
                out_dict[j].append(val)
        if output_idx:
            out_dict["idx"].extend(list(range(explode_length)))
        if output_val:
            out_dict["val"].extend(sub_arr)
        if output_this:
            out_dict["this"].extend([sub_arr] * len(sub_arr))
    for i in range(len(keep_cols)):
        out_dict[i] = pd.Series(out_dict[i], dtype=df_subset.iloc[:, i].dtype)
    return pd.DataFrame(out_dict)


@pytest.mark.parametrize(
    "output_idx, output_val, output_this",
    [
        pytest.param(False, False, False, id="output_nothing", marks=pytest.mark.slow),
        pytest.param(False, True, False, id="output_value"),
        pytest.param(True, True, True, id="output_all"),
    ],
)
@pytest.mark.parametrize(
    "keep_cols",
    [
        pytest.param((0,), id="keep_int", marks=pytest.mark.slow),
        pytest.param((2,), id="keep_string", marks=pytest.mark.slow),
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
def test_lateral_flatten(
    explode_col, keep_cols, output_idx, output_val, output_this, memory_leak_check
):
    """
    Tests the lateral_flatten kernel
    """

    if (explode_col == 3 and output_this) or (3 in keep_cols):
        pytest.skip(reason="TODO: support arrays of strings in replicated columns")

    df = pd.DataFrame(
        {
            "a": pd.Series(range(10), dtype=np.int64),
            "b": pd.Series(
                [
                    [1],
                    [2, 3],
                    [4, 5, 6],
                    [],
                    [7],
                    [8, 9],
                    [10, 11, 12],
                    [13],
                    [14],
                    [15],
                ]
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
                ]
            ),
            "e": pd.Series(
                [1, 2, None, 8, 16, 32, 64, 128, None, 512], dtype=pd.Int32Dtype()
            ),
        }
    )
    answer = simulate_lateral_flatten(
        df, keep_cols, explode_col, output_idx, output_val, output_this
    )
    global_1 = MetaType((0, 1, 2, 3, 4))
    global_2 = MetaType(keep_cols)
    global_3 = ColNamesMetaType(tuple(answer.columns))
    global_4 = MetaType((False, False, False, output_idx, output_val, output_this))

    def impl(df1):
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodo.libs.lateral.lateral_flatten(T1, global_2, explode_col, global_4)
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
                ]
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
                ]
            ),
        }
    )
    global_1 = MetaType((0, 1))
    global_2 = MetaType((0,))
    global_3 = ColNamesMetaType(tuple(answer.columns))
    global_4 = MetaType((False, False, False, False, True, False))

    def impl(df1):
        # First lateral flatten array
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodo.libs.lateral.lateral_flatten(T1, global_2, 1, global_4)
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
        T4 = bodo.libs.lateral.lateral_flatten(T3, global_2, 1, global_4)
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
    global_4 = ColNamesMetaType(("A_Rep", "B_Flat"))
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
            T3 = bodo.libs.lateral.lateral_flatten(T2, global_2, 1, global_3)
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
        bodo.libs.memory_budget.delete_operator_comptroller()
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
            "A_Rep": [1, 2, 2, 3, 3, 3, 5, 6, 6, 7, 7, 7, 7, 7, 9],
            "B_Flat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
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
