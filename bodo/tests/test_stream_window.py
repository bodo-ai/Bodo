# Copyright (C) 2022 Bodo Inc. All rights reserved.

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func


def gen_simple_window_over_nothing_tests():
    params = []
    int32_nullable_arr = pd.array(
        [None if (i // 2500) % 3 < 2 else (i**3) % 99999 for i in range(10000)],
        dtype=pd.Int32Dtype(),
    )
    decimal_arr = pd.array(
        [
            None if i % 7 == 0 else Decimal(str(i) * 6 + "." + str(i) * 2)
            for i in range(10000)
        ],
        dtype=pd.ArrowDtype(pa.decimal128(32, 8)),
    )
    int32_numpy_arr = pd.array([(i**4) % 99999 for i in range(10000)], dtype=np.int32)
    string_arr = pd.array(
        [None if (i // 2000) % 2 == 0 else str(i**2) for i in range(5000)]
    )
    int32_arr_all_null = pd.array(
        [None for i in range(10000)],
        dtype=pd.Int32Dtype(),
    )
    string_arr_all_null = pd.array(
        [None for i in range(5000)], dtype=pd.ArrowDtype(pa.large_string())
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "min",
            pd.array([8] * len(int32_nullable_arr)),
            id="min-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "max",
            pd.array([99972] * len(int32_nullable_arr)),
            id="max-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "count",
            pd.array([2500] * len(int32_nullable_arr)),
            id="count-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "sum",
            pd.array([124646669] * len(int32_nullable_arr), dtype=pd.Int64Dtype()),
            id="sum-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "min",
            pd.array(
                [Decimal("111111.11000000")] * len(decimal_arr),
                dtype=pd.ArrowDtype(pa.decimal128(32, 8)),
            ),
            id="min-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "max",
            pd.array(
                [Decimal("999999999999999999999999.99999999")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(32, 8)),
            ),
            id="max-decimal",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "sum",
            pd.array(
                [Decimal("4242867611357352697748771451.87305829")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="sum-decimal",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "min",
            pd.array([0] * len(int32_numpy_arr)),
            id="min-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "max",
            pd.array([99976] * len(int32_numpy_arr)),
            id="max-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "count",
            pd.array([10000] * len(int32_numpy_arr)),
            id="count-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "sum",
            pd.array([494581416] * len(int32_numpy_arr), dtype=pd.Int64Dtype()),
            id="sum-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            string_arr, "min", pd.array(["10004569"] * len(string_arr)), id="min-string"
        )
    )
    params.append(
        pytest.param(
            string_arr, "max", pd.array(["9998244"] * len(string_arr)), id="max-string"
        )
    )
    params.append(
        pytest.param(
            string_arr, "count", pd.array([2000] * len(string_arr)), id="count-string"
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "min",
            pd.array([None] * len(int32_arr_all_null)),
            id="min-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "max",
            pd.array([None] * len(int32_arr_all_null)),
            id="max-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "count",
            pd.array([0] * len(int32_arr_all_null)),
            id="count-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "sum",
            pd.array([None] * len(int32_arr_all_null)),
            id="sum-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "min",
            pd.array([None] * len(string_arr_all_null)),
            id="min-string_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "max",
            pd.array([None] * len(string_arr_all_null)),
            id="max-string_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "count",
            pd.array([0] * len(string_arr_all_null)),
            id="count-string_all_null",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "mean",
            pd.array(
                [Decimal("495025972623655664187232.697686741138")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(38, 12)),
            ),
            id="avg-decimal",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "mean",
            pd.array([49858.6676] * len(int32_nullable_arr), pd.Float64Dtype),
            id="avg-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "mean",
            pd.array(
                [None] * len(int32_arr_all_null),
                pd.Float64Dtype,
            ),
            id="avg-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "mean",
            pd.array([49458.1416] * len(int32_numpy_arr), pd.Float64Dtype),
            id="avg-int32_numpy",
        )
    )
    return params


@pytest.mark.parametrize(
    "data, func_name, answer", gen_simple_window_over_nothing_tests()
)
def test_simple_window_over_nothing(data, func_name, answer, memory_leak_check):
    """
    Tests the streaming window code for `F(X) OVER ()`
    """
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(("idx", "win"))
    output_col_order = bodo.utils.typing.MetaType((0, 1))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType((0,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((1,),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            2,
        )
        window_state = bodo.libs.stream_window.init_window_state(
            4001,
            empty_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            2,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.stream_window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.stream_window.window_produce_output_batch(window_state, True)
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.stream_window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        out_table = bodo.hiframes.table.table_subset(
            window_output, output_col_order, False
        )

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    in_df = pd.DataFrame(
        {
            "idx": [str(i) for i in range(len(data))],
            "data": data,
        }
    )
    out_df = pd.DataFrame(
        {
            "idx": [str(i) for i in range(len(data))],
            "win": answer,
        }
    )

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "kept_input_indices",
    [
        pytest.param([], id="keep_nothing"),
        pytest.param([0], id="keep_idx"),
        pytest.param([1], id="keep_input"),
        pytest.param([0, 1], id="keep_both"),
    ],
)
def test_max_over_nothing_different_kept_inputs(kept_input_indices, memory_leak_check):
    """
    Tests keeping or tossing the input/pass-through columns to a window function.
    """
    output_col_names = ["idx", "data", "win"]
    output_col_names = [
        name
        for idx, name in enumerate(output_col_names)
        if idx in kept_input_indices or name == "win"
    ]

    kept_cols = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(output_col_names))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType(tuple(kept_input_indices))
    func_names = bodo.utils.typing.MetaType(("max",))
    func_input_indices = bodo.utils.typing.MetaType(((1,),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            3,
        )
        window_state = bodo.libs.stream_window.init_window_state(
            4001,
            empty_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            2,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.stream_window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.stream_window.window_produce_output_batch(window_state, True)
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.stream_window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    in_df = pd.DataFrame(
        {
            "idx": [str(i) for i in range(1000)],
            "data": pd.array(
                [
                    None if i % 7 == 0 else int(str(i).replace("9", "1"))
                    for i in range(1000)
                ],
                dtype=pd.Int32Dtype(),
            ),
        }
    )
    out_df = pd.DataFrame(
        {
            "idx": [str(i) for i in range(1000)],
            "data": pd.array(
                [
                    None if i % 7 == 0 else int(str(i).replace("9", "1"))
                    for i in range(1000)
                ],
                dtype=pd.Int32Dtype(),
            ),
            "win": pd.array([888 for _ in range(1000)], dtype=pd.Int32Dtype()),
        }
    ).loc[:, output_col_names]

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "func_name, order_keys, in_df, out_df",
    [
        pytest.param(
            "row_number",
            (0,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": range(1, 1001),
                }
            ),
            id="row_number-integer",
        ),
        pytest.param(
            "rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": [i // 10 for i in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + 10 * (i // 10) for i in range(1000)],
                }
            ),
            id="rank-integer",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": [i // 10 for i in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-integer",
        ),
        pytest.param(
            "percent_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": [i // 10 for i in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [(10 * (i // 10)) / 999 for i in range(1000)],
                }
            ),
            id="percent_rank-integer",
        ),
        pytest.param(
            "cume_dist",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": [i // 10 for i in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [((10 * (1 + (i // 10)))) / 1000 for i in range(1000)],
                }
            ),
            id="cume_dist-integer",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": [f"{(i//10):02}" for i in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-strings",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [f"{(i//10):02}" for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
                    ),
                },
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                },
            ),
            id="dense_rank-dictionary",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [[i // 10, None] for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-array_item",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [
                            {"A": i // 10, "B": 7, "C": ["A", "B", "A"]}
                            for i in range(1000)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("A", pa.int32()),
                                    pa.field("B", pa.int32()),
                                    pa.field(
                                        "C",
                                        pa.large_list(
                                            pa.dictionary(pa.int32(), pa.string())
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-struct",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [{"A": i // 10, "B": 7, "C": None} for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int64())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-map",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [
                            [[], None, [{"A": 0}, {"B": 7 * (i // 10), "C": i // 100}]]
                            for i in range(1000)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.large_list(
                                pa.large_list(pa.map_(pa.large_string(), pa.int16()))
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-multi_nested",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "O": pd.array(
                        [
                            bodo.TimestampTZ(
                                pd.Timestamp("2018-10-1")
                                + pd.DateOffset(months=5 * (i // 10)),
                                i,
                            )
                            for i in range(1000)
                        ]
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "OUT": [1 + (i // 10) for i in range(1000)],
                }
            ),
            id="dense_rank-timestamptz",
        ),
    ],
)
def test_partitionless_rank_family(
    func_name, order_keys, in_df, out_df, memory_leak_check
):
    # Randomize the order of the input data
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(in_df))
    in_df = in_df.iloc[perm, :]

    n_inputs = len(in_df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(n_inputs)))
    col_meta = bodo.utils.typing.ColNamesMetaType(("IDX", "OUT"))
    empty_global = bodo.utils.typing.MetaType(())
    order_global = bodo.utils.typing.MetaType(order_keys)
    true_global = bodo.utils.typing.MetaType(tuple([True] * len(order_keys)))
    kept_indices = bodo.utils.typing.MetaType((0,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            n_inputs,
        )
        window_state = bodo.libs.stream_window.init_window_state(
            4001,
            empty_global,
            order_global,
            true_global,
            true_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            n_inputs,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.stream_window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.stream_window.window_produce_output_batch(window_state, True)
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.stream_window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )
