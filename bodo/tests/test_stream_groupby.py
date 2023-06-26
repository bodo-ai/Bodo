import numpy as np
import pandas as pd

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.ir.aggregate import supported_agg_funcs
from bodo.libs.stream_groupby import (
    delete_groupby_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.tests.utils import check_func


def test_groupby_basic(memory_leak_check):
    """
    Tests support for the basic streaming groupby functionality.
    """

    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    ftypes = bodo.utils.typing.MetaType((supported_agg_funcs.index("sum"),))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((1,))

    def test_groupby(df):
        groupby_state = init_groupby_state(keys_inds, ftypes, f_in_offsets, f_in_cols)
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df.iloc[(_temp1 * batch_size) : ((_temp1 + 1) * batch_size)]
            is_last1 = (_temp1 * batch_size) >= len(df)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            groupby_build_consume_batch(groupby_state, table1, is_last1)

        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)

        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    df = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 2, 0],
            "B": [1, 3, 5, 11, 1, 3],
        }
    )
    expected_df = df.groupby("A", as_index=False).sum()

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
