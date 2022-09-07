# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""IR node for the parquet data access"""

import numba
from numba.core import ir, ir_utils, typeinfer, types

import bodo
import bodo.ir.connector
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.transforms import distributed_analysis
from bodo.transforms.table_column_del_pass import (
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
        input_file_name_col,
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
        self.out_used_cols = list(range(len(col_indices)))
        # Name of the column where we insert the name of file that the row comes from
        self.input_file_name_col = input_file_name_col
        # These fields are used to enable compilation if unsupported columns
        # get eliminated.
        self.unsupported_columns = unsupported_columns
        self.unsupported_arrow_types = unsupported_arrow_types
        # Is the variable currently alive. This should be replaced with more
        # robust handling in connectors.
        self.is_live_table = True

    def __repr__(self):  # pragma: no cover
        # TODO
        return "({}) = ReadParquet({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
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
            self.out_used_cols,
            self.input_file_name_col,
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
        pq_node.out_used_cols = []
        pq_node.is_live_table = False
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
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the actual file columns in during distributed pass.
    """
    return bodo.ir.connector.base_connector_remove_dead_columns(
        pq_node,
        column_live_map,
        equiv_vars,
        typemap,
        "ParquetReader",
        # col_indices is set to an empty list if the table is dead
        # see 'remove_dead_pq'
        pq_node.col_indices,
        # Parquet can track length without loading any columns.
        require_one_column=False,
    )


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
ir_extension_table_column_use[
    ParquetReader
] = bodo.ir.connector.connector_table_column_use
