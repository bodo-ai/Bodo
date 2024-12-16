"""
Selection of helper functions used in our MERGE_INTO implementation
"""

import numba
import pandas as pd
from numba.core import types

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.utils.typing import BodoError, ColNamesMetaType, to_nullable_type

# As defined in our Calcite branch
DELETE_ENUM = 0
INSERT_ENUM = 1
UPDATE_ENUM = 2

ROW_ID_COL_NAME = "_BODO_ROW_ID"
MERGE_ACTION_ENUM_COL_NAME = "_MERGE_INTO_CHANGE"


# We can't have this function be inlined, as it would break our filter pushdown,
# Since we explicitly check for this function in typing pass
# We don't need to explicitly provide never since never is the default (https://numba.pydata.org/numba-doc/latest/developer/inlining.html)
# But we're doing it here just for future proofing, in case the default ever changes
@bodo.jit(inline="never")
def do_delta_merge_with_target(target_df, delta_df):  # pragma: no cover
    """
    This function takes two DataFrames, a target df and a delta_df. It
    then applies the changes found in the delta table to the target df.

    This function is steps 6 through 8 in the overall COW design:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1157529601/MERGE+INTO+Design#Bodo-Design-(COW)

    Args:
        target_df (DataFrame): Must contain row_id column with name equal to the ROW_ID_COL_NAME constant
            defined above, and be sorted by said column
        delta_df (DataFrame): Must contain all the data rows of the target df, a row_id column,
            and a merge_into_change column both with the constant names equal to the constants
            ROW_ID_COL_NAME and MERGE_ACTION_ENUM_COL_NAME defined above
    """

    # This is required to avoid type conflicts in the later steps. This also handles
    # the case where a column should be converted from string to dict-encoded strings
    # and vice-versa.
    delta_df_casted = delta_df.astype(
        target_df.dtypes, copy=False, _bodo_nan_to_str=False
    )

    # First, split the delta DataFrame into the rows to be inserted, and the rows to be modified/deleted
    insert_filter = delta_df_casted[MERGE_ACTION_ENUM_COL_NAME] == INSERT_ENUM
    delta_df_new = delta_df_casted[insert_filter]
    delta_df_changes = delta_df_casted[~insert_filter]

    # Next, we get the row_id boundaries on each rank
    row_id_chunk_bounds = bodo.libs.distributed_api.get_chunk_bounds(
        bodo.utils.conversion.coerce_to_array(target_df[ROW_ID_COL_NAME])
    )

    # Next, we do a parallel sort on the delta df, using the same row_id boundaries as the sorted
    # target DataFrame. This ensures that each rank has all the information needed to update its local
    # slice of the target DataFrame
    sorted_delta_df_changes = delta_df_changes.sort_values(
        by=ROW_ID_COL_NAME, _bodo_chunk_bounds=row_id_chunk_bounds
    )

    # Finally, update the target DataFrame based on the actions stored in the delta df
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
    output_table = pd.concat((target_df_with_updated_and_deleted_rows, delta_df_new))

    return output_table


def delta_table_setitem_common_code(
    n_out_cols: int, out_arr_types: list[types.Type], from_target_table=True
):
    """
    Helper fn for merge_sorted_dataframes's func text generation. Generates code that sets the
    index 'output_tbl_idx' in each of the output series. The source of the values to use (delta table
    or source table) is specified by the argument `from_target_table`. Example codegen:

      bodo.libs.array_kernels.copy_array_element(arr0, output_tbl_idx, target_table_col_0, i)
      bodo.libs.array_kernels.copy_array_element(arr1, output_tbl_idx, target_table_col_1, i)
      bodo.libs.array_kernels.copy_array_element(arr2, output_tbl_idx, target_table_col_2, i)

      This code includes optimizations for dictionary encoding and strings
      to avoid intermediate allocations

    Args:
        n_out_cols (int): The number of output columns to set
        out_arr_types(List[types.Type]): List of output Array types. Used for optimizing
          how to copy data while avoiding intermediate allocations.
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

        if out_arr_types[out_col_idx] != bodo.dict_str_arr_type:
            # Avoid intermediate allocations for arrays. Note that copy_array_element handles
            # null checking for us
            func_text += f"{indent}bodo.libs.array_kernels.copy_array_element(arr{out_col_idx}, output_tbl_idx, {colname}, {idx_var})\n"
        else:
            # Here we set the just copy an index, possibly adding an offset.

            # Offset used for dictionary encoded arrays.
            dict_offset = f"{prefix}_offset_{out_col_idx}"
            # Indices array
            dict_indices = f"{prefix}_col_indices_{out_col_idx}"

            func_text += (
                f"{indent}if bodo.libs.array_kernels.isna({colname}, {idx_var}):\n"
            )
            func_text += f"{indent}  bodo.libs.array_kernels.setna(arr{out_col_idx}, output_tbl_idx)\n"
            func_text += f"{indent}else:\n"

            if from_target_table:
                # Target table maintains the same indices
                func_text += f"{indent}  val = {dict_indices}[{idx_var}]\n"
            else:
                func_text += (
                    f"{indent}  val = {dict_indices}[{idx_var}] + {dict_offset}\n"
                )
            func_text += f"{indent}  arr{out_col_idx}[output_tbl_idx] = val\n"

    return func_text


@numba.generated_jit(nopython=True)
def merge_sorted_dataframes(target_df: DataFrameType, delta_df: DataFrameType):
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
          bodo.libs.array_kernels.copy_array_element(arr0, output_tbl_idx, target_table_col_0, target_df_index)
          bodo.libs.array_kernels.copy_array_element(arr1, output_tbl_idx, target_table_col_1, target_df_index)
          bodo.libs.array_kernels.copy_array_element(arr2, output_tbl_idx, target_table_col_2, target_df_index)
        else:
          if delta_df_merge_into_change_col[delta_df_index] == 0:
            delta_df_index += 1
            continue
          if delta_df_merge_into_change_col[delta_df_index] == 2:
            bodo.libs.array_kernels.copy_array_element(arr0, output_tbl_idx, delta_table_col_0, delta_df_index)
            bodo.libs.array_kernels.copy_array_element(arr1, output_tbl_idx, delta_table_col_1, delta_df_index)
            bodo.libs.array_kernels.copy_array_element(arr2, output_tbl_idx, delta_table_col_2, delta_df_index)
          delta_df_index += 1
        output_tbl_idx += 1
      return bodo.hiframes.pd_dataframe_ext.init_dataframe((arr0, arr1, arr2,), bodo.hiframes.pd_index_ext.init_range_index(0, (target_df_len - num_deletes), 1, None), __col_name_meta_value_delta_merge)


      Note: This code includes 2 main optimizations:
        1. If the output array is a string array then we avoid intermediate allocations.
        2. If the target_df and delta_df both have a dictionary encoded array then the
           output data will be dictionary encoded, which we do by concatenating the dictionaries.

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
    assert (
        target_df.data is not None
        and target_df.columns is not None
        and delta_df.data is not None
        and delta_df.columns is not None
    )

    glbls = {}
    target_row_id_col_index = target_df.column_index[ROW_ID_COL_NAME]
    out_arr_types = []
    for i in range(len(target_df.data)):
        # We drop the ID column so skip it
        if i != target_row_id_col_index:
            # Only dictionary encode data if both input and output are dictionary encoded.
            if target_df.data[i] == bodo.dict_str_arr_type:
                if delta_df.data[i] == bodo.dict_str_arr_type:
                    out_arr_types.append(bodo.dict_str_arr_type)
                else:
                    out_arr_types.append(bodo.string_array_type)
            else:
                out_arr_types.append(target_df.data[i])

    out_column_names = (
        target_df.columns[:target_row_id_col_index]
        + target_df.columns[target_row_id_col_index + 1 :]
    )
    n_out_cols = len(out_arr_types)

    delta_id_col_index = delta_df.column_index[ROW_ID_COL_NAME]
    delta_merge_into_change_col_index = delta_df.column_index[
        MERGE_ACTION_ENUM_COL_NAME
    ]

    func_text = "def impl(target_df, delta_df):\n"
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
        if out_arr_types[i] == bodo.dict_str_arr_type:
            # If we have a dictionary array we allocate for the indices instead.
            alloc_arr_type = bodo.libs.dict_arr_ext.dict_indices_arr_type
        else:
            alloc_arr_type = to_nullable_type(out_arr_types[i])
        glbls[f"_arr_typ{i}"] = alloc_arr_type

    for i in range(len(target_df.data)):
        if i == target_row_id_col_index:
            continue
        # Make sure the array number is consistent with n_out_cols
        j = i - 1 if i > target_row_id_col_index else i
        func_text += f"  target_table_col_{j} = get_dataframe_data(target_df, {i})\n"

    for i in range(len(delta_df.data)):
        if i in (delta_id_col_index, delta_merge_into_change_col_index):
            continue
        # Make sure the array number is consistent with n_out_cols
        j = i - 1 if i > target_row_id_col_index else i
        func_text += f"  delta_table_col_{j} = get_dataframe_data(delta_df, {i})\n"

    # Allocate any output dictionaries for dictionary encoded columns.
    for i in range(n_out_cols):
        if out_arr_types[i] == bodo.dict_str_arr_type:
            func_text += f"  target_table_col_data_{i} = target_table_col_{i}._data\n"
            func_text += (
                f"  target_table_col_indices_{i} = target_table_col_{i}._indices\n"
            )
            func_text += f"  delta_table_offset_{i} = len(target_table_col_data_{i})\n"
            func_text += f"  delta_table_col_data_{i} = delta_table_col_{i}._data\n"
            func_text += (
                f"  delta_table_col_indices_{i} = delta_table_col_{i}._indices\n"
            )
            func_text += (
                f"  delta_table_col_indices_{i} = delta_table_col_{i}._indices\n"
            )
            # Allocate the dictionary
            func_text += f"  num_strings = len(target_table_col_data_{i}) + len(delta_table_col_data_{i})\n"
            func_text += f"  num_chars = bodo.libs.str_arr_ext.num_total_chars(target_table_col_data_{i}) + bodo.libs.str_arr_ext.num_total_chars(delta_table_col_data_{i})\n"
            func_text += f"  out_dict_data_{i} = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strings, num_chars)\n"
            # Copy the entries
            func_text += f"  for l in range(len(target_table_col_data_{i})):\n"
            func_text += f"    bodo.libs.str_arr_ext.get_str_arr_item_copy(out_dict_data_{i}, l, target_table_col_data_{i}, l)\n"
            func_text += f"  for l in range(len(delta_table_col_data_{i})):\n"
            func_text += f"    bodo.libs.str_arr_ext.get_str_arr_item_copy(out_dict_data_{i}, l + delta_table_offset_{i}, delta_table_col_data_{i}, l)\n"
            # Determine if its global
            func_text += f"  dict_is_global_{i} = target_table_col_{i}._has_global_dictionary and delta_table_col_{i}._has_global_dictionary\n"

    func_text += "  delta_df_index = 0\n"
    # out table idx != target_df_index, because of delete rows
    func_text += "  output_tbl_idx = 0\n"
    func_text += "  for target_df_index in range(target_df_len):\n"

    # neither of these columns can be NULL, so no need to null check
    func_text += "    if delta_df_index >= delta_df_len or (target_df_row_id_col[target_df_index] != delta_df_row_id_col[delta_df_index]):\n"
    # If we don't have an update/delete for the current row, we copy the values from the input
    # dataframe into the output
    func_text += delta_table_setitem_common_code(
        n_out_cols, out_arr_types, from_target_table=True
    )
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
    func_text += delta_table_setitem_common_code(
        n_out_cols, out_arr_types, from_target_table=False
    )
    # update the delta df index accordingly
    func_text += "      delta_df_index += 1\n"
    # We can't have an update and any other action targeting the same row
    func_text += "    output_tbl_idx += 1\n"

    # Create the dictionaries and drop duplicates
    for i in range(n_out_cols):
        if out_arr_types[i] == bodo.dict_str_arr_type:
            # Note: We cannot assume it is unique even if each component were unique.
            func_text += f"  out_dict_arr_{i} = bodo.libs.dict_arr_ext.init_dict_arr(out_dict_data_{i}, arr{i}, dict_is_global_{i}, False, None)\n"
            # Drop any duplicates and update the dictionary
            func_text += f"  arr{i} = bodo.libs.array.drop_duplicates_local_dictionary(out_dict_arr_{i}, False)\n"

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
