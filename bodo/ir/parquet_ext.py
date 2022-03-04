# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the parquet data access"""

import numba
from numba.core import ir, ir_utils, typeinfer, types

import bodo
import bodo.ir.connector
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.transforms import distributed_analysis
from bodo.transforms.table_column_del_pass import (
    get_live_column_nums_block,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)


class ParquetReader(ir.Stmt):
    def __init__(
        self,
        file_name,
        df_out,
        col_names,
        col_indices,
        out_types,
        out_vars,
        loc,
        partition_names,
        # These are the same storage_options that would be passed to pandas
        storage_options,
        index_column_index,
        index_column_type,
        unsupported_columns,
        unsupported_arrow_types,
    ):
        self.connector_typ = "parquet"
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.df_colnames = col_names
        self.col_indices = col_indices
        self.out_types = out_types
        # Original out types + columns are maintained even if columns are pruned.
        # This is maintained in case we need type info for filter pushdown and
        # the column has been eliminated.
        # For example, if our Pandas code was:
        # def ex(filename):
        #     df = pd.read_parquet(filename)
        #     df = df[df.A > 1]
        #     return df[["B", "C"]]
        # Then DCE should remove all columns from df_colnames/out_types except B and C,
        # but we still need to the type of column A to determine if we need to generate
        # a cast inside the arrow filters.
        self.original_out_types = out_types
        self.original_df_colnames = col_names
        self.out_vars = out_vars
        self.loc = loc
        self.partition_names = partition_names
        self.filters = None
        # storage_options passed to pandas during read_parquet
        self.storage_options = storage_options
        self.index_column_index = index_column_index
        self.index_column_type = index_column_type
        # Columns within the output table type that are actually used.
        # These will be updated during optimzations and do not contain
        # the actual columns numbers that should be loaded. For more
        # information see 'pq_remove_dead_column'.
        self.type_usecol_offset = list(range(len(col_indices)))
        # These fields are used to enable compilation if unsupported columns
        # get eliminated.
        self.unsupported_columns = unsupported_columns
        self.unsupported_arrow_types = unsupported_arrow_types

    def __repr__(self):  # pragma: no cover
        # TODO
        return "({}) = ReadParquet({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.df_out,
            self.file_name.name,
            self.df_colnames,
            self.col_indices,
            self.out_types,
            self.original_out_types,
            self.original_df_colnames,
            self.out_vars,
            self.partition_names,
            self.filters,
            self.storage_options,
            self.index_column_index,
            self.index_column_type,
            self.type_usecol_offset,
            self.unsupported_columns,
            self.unsupported_arrow_types,
        )


def remove_dead_pq(
    pq_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """
    Function that eliminates parquet reader variables when they
    are no longer live.
    """
    table_var = pq_node.out_vars[0].name
    index_var = pq_node.out_vars[1].name
    if table_var not in lives and index_var not in lives:
        # If neither the table or index is live, remove the node.
        return None
    elif table_var not in lives:
        # If table isn't live we only want to load the index.
        # To do this we should mark the col_indices as empty
        pq_node.col_indices = []
        pq_node.df_colnames = []
        pq_node.type_usecol_offset = []
    elif index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_index as None
        pq_node.index_column_index = None
        pq_node.index_column_type = types.none
    # TODO: Update the usecols if only 1 of the variables is live.
    return pq_node


def pq_remove_dead_column(pq_node, column_live_map, equiv_vars, typemap):
    """
    Function that tracks which columns to prune from the Parquet node.
    This updates type_usecol_offset which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the actual file columns in during distributed pass.
    """
    # All pq_nodes should have 2 vars
    assert len(pq_node.out_vars) == 2, "invalid ParquetReader node"
    table_var_name = pq_node.out_vars[0].name
    assert isinstance(
        typemap[table_var_name], TableType
    ), "Parquet Node Table must be a TableType"
    # if col_indices == [] then the table is dead and we are only loading
    # the index. See 'remove_dead_pq'
    if pq_node.col_indices:
        # Compute all columns that are live at this statement.
        used_columns, use_all = get_live_column_nums_block(
            column_live_map, equiv_vars, table_var_name
        )
        used_columns = bodo.ir.connector.trim_extra_used_columns(
            used_columns, len(pq_node.col_indices)
        )
        if not use_all and not used_columns:
            # If we see no specific column is need some operations need some
            # column but no specific column. For example:
            # T = read_parquet(table(0, 1, 2, 3))
            # n = len(T)
            #
            # Here we just load column 0. If no columns are actually needed, dead
            # code elimination will remove the entire IR var in 'remove_dead_parquet'.
            #
            used_columns = [0]
        if not use_all and len(used_columns) != len(pq_node.type_usecol_offset):
            # Update the type offset. If an index column its not included in
            # the original table. If we have code like
            #
            # T = read_csv(table(0, 1, 2, 3)) # Assume index column is column 2
            #
            # We type T without the index column as Table(arr0, arr1, arr3).
            # As a result once we apply optimizations, all the column indices
            # will refer to the index within that type, not the original file.
            #
            # i.e. T[2] == arr3
            #
            # This means that used_columns will track the offsets within the type,
            # not the actual column numbers in the file. We keep these offsets separate
            # while finalizing DCE and we will update the file with the actual columns later
            # in distirbuted pass.
            #
            # For more information see:
            # https://bodo.atlassian.net/wiki/spaces/B/pages/921042953/Table+Structure+with+Dead+Columns#User-Provided-Column-Pruning-at-the-Source

            pq_node.type_usecol_offset = used_columns
            # Return that this table was updated
            return True
    return False


def pq_table_column_use(pq_node, block_use_map, equiv_vars, typemap):
    """
    Function to handle any necessary processing for column uses
    with a particular table. ParquetReader defines a table and doesn't
    use any other table, so this does nothing.
    """
    return


numba.parfors.array_analysis.array_analysis_extensions[
    ParquetReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    ParquetReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[ParquetReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[ParquetReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[ParquetReader] = remove_dead_pq
numba.core.analysis.ir_extension_usedefs[
    ParquetReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[
    ParquetReader
] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    ParquetReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    ParquetReader
] = bodo.ir.connector.build_connector_definitions
remove_dead_column_extensions[ParquetReader] = pq_remove_dead_column
ir_extension_table_column_use[ParquetReader] = pq_table_column_use
