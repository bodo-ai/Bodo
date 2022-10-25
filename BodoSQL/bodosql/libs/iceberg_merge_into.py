"""
Selection of helper functions used in our MERGE_INTO implementation
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import numba
import pandas as pd

import bodo
from bodo.utils.typing import BodoError, ColNamesMetaType, to_nullable_type

# As defined in our Calcite branch
DELETE_ENUM = 0
INSERT_ENUM = 1
UPDATE_ENUM = 2

ROW_ID_COL_NAME = "_bodo_row_id"
MERGE_ACTION_ENUM_COL_NAME = "_merge_into_change"


@bodo.jit
def do_delta_merge_with_target(target_df, delta_df):  # pragma: no cover
    """
    This function takes two dataframes, a target df and a delta_df. It
    then applies the changes found in the delta table to the target df.

    This function is steps 6 through 8 in the overall COW design:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1157529601/MERGE+INTO+Design#Bodo-Design-(COW)

    Args:
        target_df (dataframe): Must contain row_id column with name equal to the ROW_ID_COL_NAME constant
        defined above, and be sorted by said column
        delta_df (dataframe): Must contain all the data rows of the target df, a row_id column,
        and a merge_into_change column both with the constant names equal to the constants
        ROW_ID_COL_NAME and MERGE_ACTION_ENUM_COL_NAME defined above
    """

    # First, split the delta dataframe into the rows to be inserted, and the rows to be modified/deleted
    insert_filter = delta_df[MERGE_ACTION_ENUM_COL_NAME] == INSERT_ENUM
    delta_df_new = delta_df[insert_filter]
    delta_df_changes = delta_df[~insert_filter]

    # Next, we get the row_id boundaries on each rank
    row_id_chunk_bounds = bodo.libs.distributed_api.get_chunk_bounds(
        bodo.utils.conversion.coerce_to_array(target_df[ROW_ID_COL_NAME])
    )

    # Next, we do a parallel sort on the delta df, using the same row_id boundaries as the sorted
    # target dataframe. This ensures that each rank has all the information needed to update its local
    # slice of the target dataframe
    sorted_delta_df_changes = delta_df_changes.sort_values(
        by=ROW_ID_COL_NAME, _bodo_chunk_bounds=row_id_chunk_bounds
    )

    # Finally, update the target dataframe based on the actions stored in the delta df
    target_df_with_updated_and_deleted_rows = merge_sorted_dataframes(
        target_df,
        sorted_delta_df_changes,
    )

    # For copy on write, each rank can just concatenate the new rows to the end of the target table.
    # For MOR, this may be more complicated.
    # TODO: Handle inserts in merge_sorted_dataframes by pre-allocating space in the output arrays
    # in merge_sorted_dataframes. https://bodo.atlassian.net/browse/BE-3793
    delta_df_new = delta_df_new.drop(
        [MERGE_ACTION_ENUM_COL_NAME, ROW_ID_COL_NAME], axis=1
    )
    output_table = pd.concat([target_df_with_updated_and_deleted_rows, delta_df_new])

    return output_table


def delta_table_setitem_common_code(n_out_cols: int, from_target_table=True):
    """
    Helper fn for merge_sorted_dataframes's func text generation. Generates code that sets the
    index 'output_tbl_idx' in each of the output series. The source of the values to use (delta table
    or source table) is specified by the argument `from_target_table`. Example codegen:

      if bodo.libs.array_kernels.isna(target_table_col_0, i):
        bodo.libs.array_kernels.setna(arr0, output_tbl_idx)
      else:
        val = target_table_col_0[i]
        arr0[output_tbl_idx] = val
      if bodo.libs.array_kernels.isna(target_table_col_1, i):
        bodo.libs.array_kernels.setna(arr1, output_tbl_idx)
      else:
        val = target_table_col_1[i]
        arr1[output_tbl_idx] = val
      if bodo.libs.array_kernels.isna(target_table_col_2, i):
        bodo.libs.array_kernels.setna(arr2, output_tbl_idx)
      else:
        val = target_table_col_2[i]
        arr2[output_tbl_idx] = val

    Args:
        n_out_cols (int): The number of output columns to set
        from_target_table (bool, optional): From which table to source the values. Defaults to True.

    Returns:
        str: Func text used to be used within merge_sorted_dataframes
    """
    prefix = "target_table" if from_target_table else "delta_table"
    idx_var = "target_df_index" if from_target_table else "delta_df_index"
    indent = "  " * 3 if from_target_table else "  " * 4
    func_text = ""

    for out_col_idx in range(n_out_cols):
        colname = f"{prefix}_col_{out_col_idx}"
        func_text += f"{indent}if bodo.libs.array_kernels.isna({colname}, {idx_var}):\n"
        func_text += f"{indent}  bodo.libs.array_kernels.setna(arr{out_col_idx}, output_tbl_idx)\n"
        func_text += f"{indent}else:\n"
        func_text += f"{indent}  val = {colname}[{idx_var}]\n"
        func_text += f"{indent}  arr{out_col_idx}[output_tbl_idx] = val\n"

    return func_text


@numba.generated_jit(nopython=True)
def merge_sorted_dataframes(target_df, delta_df):
    """
        Helper function that merges the chunked/sorted target and delta dataframes.
        May throw an error if duplicate row_id's are
        found in the delta table. Example codegen included below:

    def impl(target_df, delta_df):
      target_df_len = len(target_df)
      delta_df_len = len(delta_df)
      num_deletes = (delta_df['_merge_into_change'] == 0).sum()
      target_df_row_id_col = get_dataframe_data(target_df, 3)
      delta_df_row_id_col = get_dataframe_data(delta_df, 3)
      delta_df_merge_into_change_col = get_dataframe_data(delta_df, 4)
      for i in range(1, delta_df_len):
        if delta_df_row_id_col[i-1] == delta_df_row_id_col[i]:
          raise BodoError('Error in MERGE INTO: Found multiple actions to apply to the same row in the target table')
      arr0 = bodo.utils.utils.alloc_type(target_df_len - num_deletes, _arr_typ0, (-1,))
      arr1 = bodo.utils.utils.alloc_type(target_df_len - num_deletes, _arr_typ1, (-1,))
      arr2 = bodo.utils.utils.alloc_type(target_df_len - num_deletes, _arr_typ2, (-1,))
      target_table_col_0 = get_dataframe_data(target_df, 0)
      target_table_col_1 = get_dataframe_data(target_df, 1)
      target_table_col_2 = get_dataframe_data(target_df, 2)
      delta_table_col_0 = get_dataframe_data(delta_df, 0)
      delta_table_col_1 = get_dataframe_data(delta_df, 1)
      delta_table_col_2 = get_dataframe_data(delta_df, 2)
      delta_df_index = 0
      output_tbl_idx = 0
      for target_df_index in range(target_df_len):
        if delta_df_index >= delta_df_len or (target_df_row_id_col[target_df_index] != delta_df_row_id_col[delta_df_index]):
          if bodo.libs.array_kernels.isna(target_table_col_0, target_df_index):
            bodo.libs.array_kernels.setna(arr0, output_tbl_idx)
          else:
            val = target_table_col_0[target_df_index]
            arr0[output_tbl_idx] = val
          if bodo.libs.array_kernels.isna(target_table_col_1, target_df_index):
            bodo.libs.array_kernels.setna(arr1, output_tbl_idx)
          else:
            val = target_table_col_1[target_df_index]
            arr1[output_tbl_idx] = val
          if bodo.libs.array_kernels.isna(target_table_col_2, target_df_index):
            bodo.libs.array_kernels.setna(arr2, output_tbl_idx)
          else:
            val = target_table_col_2[target_df_index]
            arr2[output_tbl_idx] = val
        else:
          if delta_df_merge_into_change_col[delta_df_index] == 0:
            delta_df_index += 1
            continue
          if delta_df_merge_into_change_col[delta_df_index] == 2:
            if bodo.libs.array_kernels.isna(delta_table_col_0, delta_df_index):
              bodo.libs.array_kernels.setna(arr0, output_tbl_idx)
            else:
              val = delta_table_col_0[delta_df_index]
              arr0[output_tbl_idx] = val
            if bodo.libs.array_kernels.isna(delta_table_col_1, delta_df_index):
              bodo.libs.array_kernels.setna(arr1, output_tbl_idx)
            else:
              val = delta_table_col_1[delta_df_index]
              arr1[output_tbl_idx] = val
            if bodo.libs.array_kernels.isna(delta_table_col_2, delta_df_index):
              bodo.libs.array_kernels.setna(arr2, output_tbl_idx)
            else:
              val = delta_table_col_2[delta_df_index]
              arr2[output_tbl_idx] = val
          delta_df_index += 1
        output_tbl_idx += 1
      return bodo.hiframes.pd_dataframe_ext.init_dataframe((arr0, arr1, arr2,), bodo.hiframes.pd_index_ext.init_range_index(0, (target_df_len - num_deletes), 1, None), __col_name_meta_value_delta_merge)


        Args:
            target_df (dataframe): Must contain row_id column, with name equal to the constant
                                    ROW_ID_COL_NAME as defined at the top of this file. Must be
                                    sorted by said row_id column, with the same chunking as the delta_df.
            delta_df (dataframe): Must contain all the data rows of the target df, a row_id, and a
                                  merge_into_change column, with names as defined in the ROW_ID_COL_NAME
                                  and MERGE_ACTION_ENUM_COL_NAME constants.
                                  Must be sorted by row_id column, with the same chunking as the
                                  delta_df. merge_into_change column can only contain
                                  updates and deletes (inserts are handled separately).

        Returns:
            A target_df, with the updates/deletes applied.
    """

    glbls = {}
    target_row_id_col_index = target_df.column_index[ROW_ID_COL_NAME]
    out_arr_types = (
        target_df.data[:target_row_id_col_index]
        + target_df.data[target_row_id_col_index + 1 :]
    )
    out_column_names = (
        target_df.columns[:target_row_id_col_index]
        + target_df.columns[target_row_id_col_index + 1 :]
    )
    n_out_cols = len(out_arr_types)

    delta_id_col_index = delta_df.column_index[ROW_ID_COL_NAME]
    delta_merge_into_change_col_index = delta_df.column_index[
        MERGE_ACTION_ENUM_COL_NAME
    ]

    func_text = f"def impl(target_df, delta_df):\n"
    func_text += "  target_df_len = len(target_df)\n"
    func_text += "  delta_df_len = len(delta_df)\n"
    func_text += f"  num_deletes = (delta_df['{MERGE_ACTION_ENUM_COL_NAME}'] == {DELETE_ENUM}).sum()\n"

    func_text += f"  target_df_row_id_col = get_dataframe_data(target_df, {target_row_id_col_index})\n"
    func_text += (
        f"  delta_df_row_id_col = get_dataframe_data(delta_df, {delta_id_col_index})\n"
    )

    func_text += f"  delta_df_merge_into_change_col = get_dataframe_data(delta_df, {delta_merge_into_change_col_index})\n"

    # NOTE: we need to preemptively iterate over the delta table to verify correctness of the delta table.
    # This is because
    # we may have multiple deletes assigned to the same row, in which case, num_deletes may lead to an
    # incorrect output allocation size, which can in turn, lead to segfaults.

    func_text += "  for i in range(1, delta_df_len):\n"
    func_text += "    if delta_df_row_id_col[i-1] == delta_df_row_id_col[i]:\n"
    func_text += "      raise BodoError('Error in MERGE INTO: Found multiple actions to apply to the same row in the target table')\n"

    # TODO: Support table format: https://bodo.atlassian.net/jira/software/projects/BE/boards/4/backlog?selectedIssue=BE-3792
    for i in range(n_out_cols):
        func_text += f"  arr{i} = bodo.utils.utils.alloc_type(target_df_len - num_deletes, _arr_typ{i}, (-1,))\n"
        glbls[f"_arr_typ{i}"] = to_nullable_type(out_arr_types[i])

    for i in range(len(target_df.data)):
        if i == target_row_id_col_index:
            continue
        func_text += f"  target_table_col_{i} = get_dataframe_data(target_df, {i})\n"

    for i in range(len(delta_df.data)):
        if i in (delta_id_col_index, delta_merge_into_change_col_index):
            continue
        func_text += f"  delta_table_col_{i} = get_dataframe_data(delta_df, {i})\n"

    func_text += "  delta_df_index = 0\n"
    # out table idx != target_df_index, because of delete rows
    func_text += "  output_tbl_idx = 0\n"
    func_text += "  for target_df_index in range(target_df_len):\n"

    # neither of these columns can be NULL, so no need to null check
    func_text += "    if delta_df_index >= delta_df_len or (target_df_row_id_col[target_df_index] != delta_df_row_id_col[delta_df_index]):\n"
    # If we don't have an update/delete for the current row, we copy the values from the input
    # dataframe into the output
    func_text += delta_table_setitem_common_code(n_out_cols, from_target_table=True)
    func_text += "    else:\n"

    func_text += (
        f"      if delta_df_merge_into_change_col[delta_df_index] == {DELETE_ENUM}:\n"
    )
    # For the delete action for the current row, we just omit adding anything to the output columns
    func_text += "        delta_df_index += 1\n"
    # It's ok to have multiple delete actions for the same row, but it's not ok to have a delete and an update
    func_text += "        continue\n"
    # If we have an update for the current row, we copy the values from the delta
    # dataframe into the output dataframe
    func_text += (
        f"      if delta_df_merge_into_change_col[delta_df_index] == {UPDATE_ENUM}:\n"
    )
    func_text += delta_table_setitem_common_code(n_out_cols, from_target_table=False)
    # update the delta df index accordingly
    func_text += "      delta_df_index += 1\n"
    # We can't have an update and any other action targeting the same row
    func_text += "    output_tbl_idx += 1\n"

    data_arrs = ", ".join(f"arr{i}" for i in range(n_out_cols))

    func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_arrs},), bodo.hiframes.pd_index_ext.init_range_index(0, (target_df_len - num_deletes), 1, None), __col_name_meta_value_delta_merge)\n"

    loc_vars = {}
    glbls.update(
        {
            "__col_name_meta_value_delta_merge": ColNamesMetaType(out_column_names),
            "get_dataframe_data": bodo.hiframes.pd_dataframe_ext.get_dataframe_data,
            "bodo": bodo,
            "BodoError": BodoError,
        }
    )
    exec(func_text, glbls, loc_vars)
    f = loc_vars["impl"]
    return f
