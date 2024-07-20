# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "data, func_name, answer",
    [
        pytest.param(
            pd.array(
                [
                    None if (i // 2500) % 3 < 2 else (i**3) % 99999
                    for i in range(10000)
                ],
                dtype=pd.Int32Dtype(),
            ),
            "max",
            99972,
            id="max-int_nullable",
        ),
        pytest.param(
            pd.array([(i**4) % 99999 for i in range(10000)], dtype=np.int32),
            "max",
            99976,
            id="max-int_numpy",
        ),
        pytest.param(
            pd.array(
                [None if (i // 2000) % 2 == 0 else str(i**2) for i in range(5000)]
            ),
            "max",
            "9998244",
            id="max-string",
        ),
        pytest.param(
            pd.array(
                [None for i in range(10000)],
                dtype=pd.Int32Dtype(),
            ),
            "max",
            None,
            id="max-int_all_null",
        ),
        pytest.param(
            pd.array(
                [None for i in range(5000)], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "max",
            None,
            id="max-string_all_null",
        ),
    ],
)
def test_simple_window_over_nothing(data, func_name, answer, memory_leak_check):
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
            "win": [answer for _ in range(len(data))],
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
