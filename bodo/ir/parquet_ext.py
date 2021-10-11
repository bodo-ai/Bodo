# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the parquet data access"""

import numba
from numba.core import ir, ir_utils, typeinfer

import bodo
import bodo.ir.connector
from bodo.transforms import distributed_analysis


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
        partition_names=None,
        # These are the same storage_options that would be passed to pandas
        storage_options=None,
    ):
        self.connector_typ = "parquet"
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.col_names = col_names
        self.col_indices = col_indices
        self.out_types = out_types
        self.out_vars = out_vars
        self.loc = loc
        self.partition_names = partition_names
        self.filters = None
        # storage_options passed to pandas during read_parquet
        self.storage_options = storage_options

    def __repr__(self):  # pragma: no cover
        # TODO
        return "({}) = ReadParquet({}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.df_out,
            self.file_name.name,
            self.col_names,
            self.col_indices,
            self.out_types,
            self.out_vars,
            self.partition_names,
            self.filters,
            self.storage_options,
        )


def remove_dead_pq(
    pq_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # TODO
    new_col_names = []
    new_out_vars = []
    new_out_types = []
    new_col_indices = []

    for i, col_var in enumerate(pq_node.out_vars):
        if col_var.name in lives:
            new_col_names.append(pq_node.col_names[i])
            new_out_vars.append(pq_node.out_vars[i])
            new_out_types.append(pq_node.out_types[i])
            new_col_indices.append(pq_node.col_indices[i])

    pq_node.col_names = new_col_names
    pq_node.out_vars = new_out_vars
    pq_node.out_types = new_out_types
    pq_node.col_indices = new_col_indices

    if len(pq_node.out_vars) == 0:
        return None

    return pq_node


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
