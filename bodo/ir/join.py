# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the join and merge"""
from collections import defaultdict
from typing import Dict, List, Literal, Set, Tuple, Union

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    next_label,
    replace_arg_nodes,
    replace_vars_inner,
    visit_vars_inner,
)
from numba.extending import intrinsic

import bodo
from bodo.hiframes.table import init_table, set_table_len
from bodo.ir.connector import trim_extra_used_columns
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    cpp_table_to_py_table,
    delete_table,
    delete_table_decref_arrays,
    hash_join_table,
    info_from_table,
    info_to_array,
)
from bodo.libs.str_arr_ext import cp_str_list_to_array, to_list_if_immutable_arr
from bodo.libs.timsort import getitem_arr_tup, setitem_arr_tup
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    get_live_column_nums_block,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import (
    INDEX_SENTINEL,
    BodoError,
    dtype_to_array_type,
    find_common_np_dtype,
    is_dtype_nullable,
    is_nullable_type,
    is_str_arr_type,
    to_nullable_type,
)
from bodo.utils.utils import alloc_arr_tup, is_null_pointer

# TODO: it's probably a bad idea for these to be global. Maybe try moving them
# to a context or dispatcher object somehow
# Maps symbol name to cfunc object that implements a general condition function.
# This dict is used only when compiling
join_gen_cond_cfunc = {}
# Maps symbol name to cfunc address (used when compiling and loading from cache)
# When compiling, this is populated in join.py::_gen_general_cond_cfunc
# When loading from cache, this is populated in numba_compat.py::resolve_join_general_cond_funcs
# when the compiled result is loaded from cache
join_gen_cond_cfunc_addr = {}


@intrinsic
def add_join_gen_cond_cfunc_sym(typingctx, func, sym):
    """This "registers" a cfunc that implements a general join condition
    so it can be cached. It does two things:
    - Generate a dummy call to the cfunc to make sure the symbol is not
      discarded during linking
    - Add cfunc library to the library of the Bodo function being compiled
      (necessary for caching so that the cfunc is part of the cached result)
    """

    def codegen(context, builder, signature, args):
        # generate dummy call to the cfunc
        # Assume signature is
        # types.bool_(
        #   types.voidptr,
        #   types.voidptr,
        #   types.voidptr,
        #   types.voidptr,
        #   types.voidptr,
        #   types.voidptr,
        #   types.int64,
        #   types.int64,
        # )
        # See: _gen_general_cond_cfunc
        sig = func.signature
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, sym._literal_value)
        builder.call(
            fn_tp,
            [
                context.get_constant_null(sig.args[0]),
                context.get_constant_null(sig.args[1]),
                context.get_constant_null(sig.args[2]),
                context.get_constant_null(sig.args[3]),
                context.get_constant_null(sig.args[4]),
                context.get_constant_null(sig.args[5]),
                context.get_constant(types.int64, 0),
                context.get_constant(types.int64, 0),
            ],
        )
        # add cfunc library to the library of the Bodo function being compiled.
        context.add_linking_libs([join_gen_cond_cfunc[sym._literal_value]._library])
        return

    return types.none(func, sym), codegen


@numba.jit
def get_join_cond_addr(name):
    """Resolve address of cfunc given by its symbol name"""
    with numba.objmode(addr="int64"):
        # This loads the function pointer at runtime, preventing
        # hardcoding the address into the IR.
        addr = join_gen_cond_cfunc_addr[name]
    return addr


HOW_OPTIONS = Literal["inner", "left", "right", "outer", "asof"]


class Join(ir.Stmt):
    def __init__(
        self,
        left_keys: Union[List[str], str],
        right_keys: Union[List[str], str],
        out_data_vars: List[ir.Var],
        out_df_type: bodo.DataFrameType,
        left_vars: List[ir.Var],
        left_df_type: bodo.DataFrameType,
        right_vars: List[ir.Var],
        right_df_type: bodo.DataFrameType,
        how: HOW_OPTIONS,
        suffix_left: str,
        suffix_right: str,
        loc: ir.Loc,
        is_left: bool,
        is_right: bool,
        is_join: bool,
        left_index: bool,
        right_index: bool,
        indicator_col_num: int,
        is_na_equal: bool,
        gen_cond_expr: str,
    ):
        """
        IR node used to represent join operations. These are produced
        by pd.merge, pd.merge_asof, and DataFrame.join. The inputs
        have the following values.

        Keyword arguments:
        left_keys -- Label or list of labels used as the keys for the left DataFrame.
        right_keys -- Label or list of labels used as the keys for the left DataFrame.
        out_data_vars -- (list[ir.Var | None]) output table and index variables (i.e. [table_var, index_var]).
        out_df_type -- Output DataFrame type for the join. This is used for the column name
                       to index map.
        left_vars -- List[ir.Var] used as the left DataFrame's used arrays.
        left_df_type -- DataFrame type for the left input. This is used for the column name
                        to index map.
        right_vars -- List[ir.Var] used as the right DataFrame's used arrays.
        right_df_type -- DataFrame type for the right input. This is used for the column name
                         to index map.
        how -- String defining the type of merge. Must be one of the above defined
               HOW_OPTIONS.
        suffix_left -- String to append to the column name of the output columns
                       from the left DataFrame if they are also found in the right DataFrame.
                       One exception is that keys with an inner join can share a name and do
                       not need a suffix.
        suffix_right -- String to append to the column name of the output columns
                        from the right DataFrame if they are also found in the left DataFrame.
                        One exception is that keys with an inner join can share a name and do
                        not need a suffix.
        loc -- Location in the source code that contains this join. Used for error messages.
        is_left -- Is this an outer join on the left side?
        is_right -- Is this an outer join on the right side?
        is_join -- Is this produced by DataFrame.join?
        left_index -- Do we use the left DataFrame's index as a key column?
        right_index -- Do we use the right DataFrame's index as a key column?
        indicator_col_num -- Location of the indicator column. -1 if no column exists.
        is_na_equal -- Should NA values be treated as equal when comparing keys?
                       In Pandas this is True, but conforming with SQL behavior
                       this is False.
        gen_cond_expr -- String used to describe the general merge condition. This
                         is used when a more general condition is needed than is
                         provided by equality.
        """
        self.left_keys = left_keys
        self.right_keys = right_keys
        # Create a set for future lookups
        self.left_key_set = set(left_keys)
        self.right_key_set = set(right_keys)
        self.out_data_vars = out_data_vars
        # Store the column names for logging pruned columns.
        self.out_col_names = out_df_type.columns
        self.left_vars = left_vars
        self.right_vars = right_vars
        self.how = how
        self.loc = loc
        self.is_left = is_left
        self.is_right = is_right
        self.is_join = is_join
        self.left_index = left_index
        self.right_index = right_index
        self.indicator_col_num = indicator_col_num
        self.is_na_equal = is_na_equal
        self.gen_cond_expr = gen_cond_expr
        # Columns within the output table type that are actually used.
        # These will be updated during optimzations. For more
        # information see 'join_remove_dead_column'.
        self.n_table_cols = len(self.out_col_names)
        self.out_used_cols = set(range(self.n_table_cols))
        if self.out_data_vars[1] is not None:
            self.out_used_cols.add(self.n_table_cols)

        # Track the indices of dead vars for the left and right
        # inputs.
        self.left_dead_var_inds = set()
        self.right_dead_var_inds = set()

        left_col_names = left_df_type.columns
        right_col_names = right_df_type.columns
        self.left_col_names = left_col_names
        self.right_col_names = right_col_names

        # Create a map for selecting variables
        self.left_var_map = {c: i for i, c in enumerate(left_col_names)}
        self.right_var_map = {c: i for i, c in enumerate(right_col_names)}
        # If INDEX_SENTINEL exists its always the last var
        if INDEX_SENTINEL in self.left_key_set:
            self.left_var_map[INDEX_SENTINEL] = len(left_vars) - 1
        if INDEX_SENTINEL in self.right_key_set:
            self.right_var_map[INDEX_SENTINEL] = len(right_vars) - 1

        if gen_cond_expr:
            # find columns used in general join condition to avoid removing them in rm dead
            # Note: this generates code per key and also is not fully correct. An expression
            # like (left.A)) != (right.B) will look like both A and A) are left key columns
            # based on this check, even though only A) is. Fixing this requires a more detailed
            # refactoring of the parsing.
            self.left_cond_cols = set(
                c for c in left_col_names if f"(left.{c})" in gen_cond_expr
            )
            self.right_cond_cols = set(
                c for c in right_col_names if f"(right.{c})" in gen_cond_expr
            )
        else:
            self.left_cond_cols = set()
            self.right_cond_cols = set()

        # When merging with one key on the index and the other on a column,
        # you can have the data column repeated as both a key and and data.
        # In this situation, since the column is used twice we must generate
        # an extra data column via the input. For an example, please refer to
        # test_merge_index_column.
        extra_data_col_tuple: Tuple[int, Literal["left", "right"], int] = (
            -1,
            "left",
            -1,
        )

        # Compute maps for each input to the output and the
        # output to the inputs.
        comm_keys = self.left_key_set & self.right_key_set
        comm_data = set(left_col_names) & set(right_col_names)
        add_suffix = comm_data - comm_keys
        # Map the output column numbers to the input
        # locations. We use this to avoid repeating
        # the conversion.
        out_to_input_col_map: Dict[int, (Literal["left", "right"], int)] = {}
        # Map each input to the output location.
        left_to_output_map: Dict[int, int] = {}
        right_to_output_map: Dict[int, int] = {}
        for i, c in enumerate(left_col_names):
            if c in add_suffix:
                suffixed_left_name = str(c) + suffix_left
                out_col_num = out_df_type.column_index[suffixed_left_name]
                # If a column is both a data column and a key
                # from left we have an extra column.
                if right_index and not left_index and c in self.left_key_set:
                    extra_out_col_num = out_df_type.column_index[c]
                    extra_data_col_tuple = (extra_out_col_num, "left", i)
            else:
                out_col_num = out_df_type.column_index[c]
            out_to_input_col_map[out_col_num] = ("left", i)
            left_to_output_map[i] = out_col_num

        for i, c in enumerate(right_col_names):
            if c not in comm_keys:
                if c in add_suffix:
                    suffixed_right_name = str(c) + suffix_right
                    out_col_num = out_df_type.column_index[suffixed_right_name]
                    # If a column is both a data column and a key
                    # from left we have an extra column.
                    if left_index and not right_index and c in self.right_key_set:
                        extra_out_col_num = out_df_type.column_index[c]
                        extra_data_col_tuple = (extra_out_col_num, "right", i)
                else:
                    out_col_num = out_df_type.column_index[c]
                out_to_input_col_map[out_col_num] = ("right", i)
                right_to_output_map[i] = out_col_num

        self.out_to_input_col_map = out_to_input_col_map
        self.left_to_output_map = left_to_output_map
        self.right_to_output_map = right_to_output_map
        self.extra_data_col_tuple = extra_data_col_tuple

        if len(out_data_vars) > 1:
            # Compute the source for the index. Note we only
            # need to track the source for a possible output
            # index.
            index_source = "left" if right_index else "right"
            # The index is always stored in the last slot.
            if index_source == "left":
                index_col_num = len(left_vars) - 1
            elif index_source == "right":
                index_col_num = len(right_vars) - 1
        else:
            index_source = None
            index_col_num = -1
        self.index_source = index_source
        self.index_col_num = index_col_num

        # vect_same_key is a vector of boolean containing whether the key have the same
        # name on the left and right. This has impact how they show up in the output:
        # ---If they have the same name then they show up just once (and have no additional
        #   missing entry)
        # ---If they have different name then they show up two times (and can have additional
        #   missing entry)
        vect_same_key = []
        n_keys = len(left_keys)
        for iKey in range(n_keys):
            name_left = left_keys[iKey]
            name_right = right_keys[iKey]
            vect_same_key.append(name_left == name_right)
        self.vect_same_key = vect_same_key

    @property
    def has_live_table_var(self):
        """Does this Join node contain a live output variable
        for the table.

        Returns:
            bool: Table output var exists and is live.
        """
        return self.out_data_vars[0] is not None

    @property
    def has_live_index_var(self):
        """Does this Join node contain a live output variable
        for the index.

        Returns:
            bool: Index output var exists and is live.
        """
        return self.out_data_vars[1] is not None

    def get_table_var(self):
        """Returns the table var for this Join Node.

        Returns:
            ir.Var: The table var.
        """
        return self.out_data_vars[0]

    def get_index_var(self):
        """Returns the index var for this Join Node.

        Returns:
            ir.Var: The index var.
        """
        return self.out_data_vars[1]

    def get_live_left_vars(self):
        """Returns the left variables that are live
        for this join node.

        Returns:
            List[ir.Var]: Currently live variables.
        """
        vars = []
        for i, v in enumerate(self.left_vars):
            if i not in self.left_dead_var_inds:
                vars.append(v)
        return vars

    def get_live_right_vars(self):
        """Returns the right variables that are live
        for this join node.

        Returns:
            List[ir.Var]: Currently live variables.
        """
        vars = []
        for i, v in enumerate(self.right_vars):
            if i not in self.right_dead_var_inds:
                vars.append(v)
        return vars

    def get_live_out_vars(self):
        """Returns the output variables that are live
        for this join node.

        Returns:
            List[ir.Var]: Currently live variables.
        """
        vars = []
        for var in self.out_data_vars:
            if var is not None:
                vars.append(var)
        return vars

    def set_live_left_vars(self, live_data_vars):
        """Sets the new left_vars for the join node based
        on which variables are live. The input only includes
        the live data vars, so this function formats left_vars
        properly.
        """
        left_vars = []
        idx = 0
        for i in range(len(self.left_vars)):
            if i in self.left_dead_var_inds:
                left_vars.append(None)
            else:
                left_vars.append(live_data_vars[idx])
                idx += 1
        self.left_vars = left_vars

    def set_live_right_vars(self, live_data_vars):
        """Sets the new right_vars for the join node based
        on which variables are live. The input only includes
        the live data vars, so this function formats right_vars
        properly.
        """
        right_vars = []
        idx = 0
        for i in range(len(self.right_vars)):
            if i in self.right_dead_var_inds:
                right_vars.append(None)
            else:
                right_vars.append(live_data_vars[idx])
                idx += 1
        self.right_vars = right_vars

    def set_live_out_data_vars(self, live_data_vars):
        """Sets the new out_data_vars for the join node based
        on which variables are live. The input only includes
        the live data vars, so this function formats out_data_vars
        properly.
        """
        out_data_vars = []
        is_live = [self.has_live_table_var, self.has_live_index_var]
        idx = 0
        for i in range(len(self.out_data_vars)):
            if not is_live[i]:
                out_data_vars.append(None)
            else:
                out_data_vars.append(live_data_vars[idx])
                idx += 1
        self.out_data_vars = out_data_vars

    def get_out_table_used_cols(self):
        """Returns the out_used_cols contained in the table.

        Returns:
            Set[int]: Set of column numbers found in the table.
        """
        return {i for i in self.out_used_cols if i < self.n_table_cols}

    def __repr__(self):  # pragma: no cover
        in_col_names = ", ".join([f"{c}" for c in self.left_col_names])
        df_left_str = f"left={{{in_col_names}}}"
        in_col_names = ", ".join([f"{c}" for c in self.right_col_names])
        df_right_str = f"right={{{in_col_names}}}"
        return "join [{}={}]: {}, {}".format(
            self.left_keys, self.right_keys, df_left_str, df_right_str
        )


def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    """
    Array analysis for the variables in the Join IR node. This states that
    all arrays in the input share a dimension and all arrays in the output
    share a dimension.
    """
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.get_live_out_vars()) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    in_vars = join_node.get_live_left_vars()
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    all_shapes = []
    in_vars = list(join_node.get_live_right_vars())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # columns of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for out_var in join_node.get_live_out_vars():
        typ = typemap[out_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, out_var, typ.ndim, None, post)
        equiv_set.insert_equiv(out_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(out_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Join] = join_array_analysis


def join_distributed_analysis(join_node, array_dists):
    """
    Perform distributed analysis for the IR variables
    contained in the Join IR node
    """

    # left and right inputs can have 1D or 1D_Var seperately (q26 case)
    # input columns have same distribution
    left_dist = Distribution.OneD
    right_dist = Distribution.OneD
    for col_var in join_node.get_live_left_vars():
        left_dist = Distribution(min(left_dist.value, array_dists[col_var.name].value))

    for col_var in join_node.get_live_right_vars():
        right_dist = Distribution(
            min(right_dist.value, array_dists[col_var.name].value)
        )

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for out_var in join_node.get_live_out_vars():
        # output dist might not be assigned yet
        if out_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[out_var.name].value)
            )

    # out dist should meet input dist (e.g. REP in causes REP out)
    # output can be stay parallel if any of the inputs is parallel, hence max()
    out_dist1 = Distribution(min(out_dist.value, left_dist.value))
    out_dist2 = Distribution(min(out_dist.value, right_dist.value))
    out_dist = Distribution(max(out_dist1.value, out_dist2.value))
    for out_var in join_node.get_live_out_vars():
        array_dists[out_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        left_dist = out_dist
        right_dist = out_dist

    # assign input distributions
    for col_var in join_node.get_live_left_vars():
        array_dists[col_var.name] = left_dist

    for col_var in join_node.get_live_right_vars():
        array_dists[col_var.name] = right_dist

    return


distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis


def visit_vars_join(join_node, callback, cbdata):
    """
    Visit each variable in the Join IR node.
    """
    # left
    join_node.set_live_left_vars(
        [
            visit_vars_inner(var, callback, cbdata)
            for var in join_node.get_live_left_vars()
        ]
    )
    # right
    join_node.set_live_right_vars(
        [
            visit_vars_inner(var, callback, cbdata)
            for var in join_node.get_live_right_vars()
        ]
    )
    # output
    join_node.set_live_out_data_vars(
        [
            visit_vars_inner(var, callback, cbdata)
            for var in join_node.get_live_out_vars()
        ]
    )


# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join


def remove_dead_join(
    join_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """
    Dead code elimination for the Join IR node. This finds columns that
    are dead data columns in the output and eliminates them from the
    inputs.
    """
    if join_node.has_live_table_var:
        # Columns to delete from out_to_input_col_map
        del_col_nums = []
        table_var = join_node.get_table_var()
        if table_var.name not in lives:
            # Remove the IR var
            join_node.out_data_vars[0] = None
            # If the table is dead remove all table columns.
            join_node.out_used_cols.difference_update(
                join_node.get_out_table_used_cols()
            )

        for col_num in join_node.out_to_input_col_map.keys():
            if col_num in join_node.out_used_cols:
                continue
            # Set the column to delete
            del_col_nums.append(col_num)

            # avoid indicator (that is not in the input)
            if join_node.indicator_col_num == col_num:
                # If _merge is removed, indicator_col_num
                # to -1 to avoid generating the indicator
                # column.
                join_node.indicator_col_num = -1
                continue
            # avoid extra data column (that is not in the input)
            if col_num == join_node.extra_data_col_tuple[0]:
                # If the extra column is removed, avoid generating
                join_node.extra_data_col_tuple = (-1, "left", -1)
                continue
            orig, col_num = join_node.out_to_input_col_map[col_num]
            if orig == "left":
                col_name = join_node.left_col_names[col_num]
                if (
                    col_name not in join_node.left_key_set
                    and col_name not in join_node.left_cond_cols
                ):
                    join_node.left_dead_var_inds.add(col_num)
            elif orig == "right":
                col_name = join_node.right_col_names[col_num]
                if (
                    col_name not in join_node.right_key_set
                    and col_name not in join_node.right_cond_cols
                ):
                    join_node.right_dead_var_inds.add(col_num)

        # Remove dead columns from the dictionary.
        for i in del_col_nums:
            del join_node.out_to_input_col_map[i]

    if join_node.has_live_index_var:
        index_var = join_node.get_index_var()
        if index_var.name not in lives:
            # Remove the IR var
            join_node.out_data_vars[1] = None
            # Update the input
            join_node.out_used_cols.remove(join_node.n_table_cols)
            if join_node.index_source == "left":
                if (
                    INDEX_SENTINEL not in join_node.left_key_set
                    and INDEX_SENTINEL not in join_node.left_cond_cols
                ):
                    join_node.left_dead_var_inds.add(join_node.index_col_num)
            else:
                if (
                    INDEX_SENTINEL not in join_node.right_key_set
                    and INDEX_SENTINEL not in join_node.right_cond_cols
                ):
                    join_node.right_dead_var_inds.add(join_node.index_col_num)
            join_node.index_col_num = -1

    if not (join_node.has_live_table_var or join_node.has_live_index_var):
        # remove empty join node
        return None
    return join_node


ir_utils.remove_dead_extensions[Join] = remove_dead_join


def join_remove_dead_column(join_node, column_live_map, equiv_vars, typemap):
    # Compute the columns that are live for the table
    changed = False
    if join_node.has_live_table_var:
        table_var_name = join_node.get_table_var().name
        used_columns, use_all, cannot_del_cols = get_live_column_nums_block(
            column_live_map, equiv_vars, table_var_name
        )
        if not (use_all or cannot_del_cols):
            used_columns = trim_extra_used_columns(used_columns, join_node.n_table_cols)
            table_used_cols = join_node.get_out_table_used_cols()
            if len(used_columns) != len(table_used_cols):
                # Mark the columns as changed as we may be able to
                # remove dead columns as a result.
                changed = True
                removed_cols = table_used_cols - used_columns
                join_node.out_used_cols.difference_update(removed_cols)
    return changed


remove_dead_column_extensions[Join] = join_remove_dead_column


def join_table_column_use(node, block_use_map, equiv_vars, typemap, table_col_use_map):
    # TODO: Update when the input can be tables
    return


ir_extension_table_column_use[Join] = join_table_column_use


def join_usedefs(join_node, use_set=None, def_set=None):
    """
    Tracks existing variables that are used (inputs) and new
    variables that are defined (output) by the Join IR Node.
    """

    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input columns are used
    use_set.update({v.name for v in join_node.get_live_left_vars()})
    use_set.update({v.name for v in join_node.get_live_right_vars()})

    # output columns are defined
    def_set.update({v.name for v in join_node.get_live_out_vars()})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    """
    Return gen and kill sets for a copy propagation
    data flow analysis. Join doesn't generate any
    copies, it just kills the output columns.
    """
    kill_set = set(v.name for v in join_node.get_live_out_vars())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Join] = get_copies_join


def apply_copies_join(
    join_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """Apply copy propagate in join node by replacing the inputs
    and the outputs."""

    # left
    join_node.set_live_left_vars(
        [replace_vars_inner(var, var_dict) for var in join_node.get_live_left_vars()]
    )
    # right
    join_node.set_live_right_vars(
        [replace_vars_inner(var, var_dict) for var in join_node.get_live_right_vars()]
    )

    # output
    join_node.set_live_out_data_vars(
        [replace_vars_inner(var, var_dict) for var in join_node.get_live_out_vars()]
    )


ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join


def build_join_definitions(join_node, definitions=None):
    """
    Construct definitions for the output variables of the
    join node.
    """
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in join_node.get_live_out_vars():
        definitions[col_var.name].append(join_node)

    return definitions


ir_utils.build_defs_extensions[Join] = build_join_definitions


def join_distributed_run(
    join_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """
    Replace the join IR node with the distributed implementations. This
    is called in distributed_pass and removes the Join from the IR.
    """
    # Add debug info about column pruning and dictionary encoded arrays.
    if bodo.user_logging.get_verbose_level() >= 2:
        join_source = join_node.loc.strformat()
        join_cols = [
            join_node.out_col_names[i]
            for i in sorted(join_node.get_out_table_used_cols())
        ]
        pruning_msg = "Finish column pruning on join node:\n%s\nOutput columns: %s\n"
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            join_source,
            join_cols,
        )

    left_parallel, right_parallel = False, False
    if array_dists is not None:
        left_parallel, right_parallel = _get_table_parallel_flags(
            join_node, array_dists
        )

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    n_keys = len(join_node.left_keys)

    # Get the output table type.
    if join_node.has_live_table_var:
        out_table_type = typemap[join_node.get_table_var().name]
        # Logical location for every cpp_table into the output
        # Python table. Any values that are left as -1 refer to
        # dead columns.
        cpp_table_logical_idx = np.full(len(out_table_type.arr_types), -1, np.int64)
    else:
        out_table_type = types.none
        cpp_table_logical_idx = None
    if join_node.has_live_index_var:
        index_col_type = typemap[join_node.get_index_var().name]
    else:
        index_col_type = types.none

    physical_idx = 0
    # physical index used for the index var if live
    index_physical_index = -1

    # Optional column refer: When doing a merge on column and index, the key
    # is put also in output, so we need one additional column in that case.
    extra_data_col_var = ()
    extra_data_col = join_node.extra_data_col_tuple[0] != -1
    if extra_data_col:
        out_col_num, source, in_col_num = join_node.extra_data_col_tuple
        cpp_table_logical_idx[out_col_num] = physical_idx
        physical_idx += 1
        if source == "left":
            extra_data_col_var = (join_node.left_vars[in_col_num],)
        else:
            extra_data_col_var = (join_node.right_vars[in_col_num],)

    # It is a fairly complex construction
    # ---keys can have same name on left and right.
    # ---keys can be two times or one time in output.
    # ---output keys can be computed from just one column (and so additional NaN may occur)
    #  or from two columns
    # ---keys may be from the index or not.
    #
    # The following rules apply:
    # ---A key that is an index or not behave in the same way. If it is an index key
    #  then its name is the value of INDEX_SENTINEL, so please don't use that one.
    # ---Identity of key is determined by their name, whether they are index or not.
    # ---If a key appears on same name on left and right then both columns are used
    #  and so the name will never have additional NaNs

    # For each key and data column used in the condition function,
    # keep track of it should be live in the output. This is
    # a list of boolean values, one per key/data column.

    # After the lists are populated the final lengths will be
    # len(left_key_in_output) = nkeys + num data_cols used in general cond func
    # len(right_key_in_output) = nkeys - (nshared_keys) + num data_cols used in general cond func
    # where nshared_keys == len(self.left_key_set & self.right_key_set)
    left_key_in_output = []
    right_key_in_output = []

    # Determine the column numbers that are live.
    left_used_key_nums = set()
    right_used_key_nums = set()

    # Generate a map for the general_merge_cond. Here we a
    # define a column by two values. The logical index is its
    # column number in the Pandas DataFrame. In constrast the
    # physical index is the actual location in the C++ table.
    # For example if our DataFrame had columns ["A", "B"], but
    # the C++ layout was ["B", "A"], then the columns would be
    # "A": (logical 0, physical 1), "B": (logical 1, physical 0).
    #
    # We need to track the mapping for each of the inputs because
    # the generated cfuncs for complex joins are written using the
    # column name and needs to be converted to the column number in
    # C++ for the generated code.
    #
    left_logical_physical_map = {}
    right_logical_physical_map = {}
    left_physical_index = 0
    right_physical_index = 0
    # Extract the left column variables for the keys and determine
    # the physical index for each live column.
    left_key_vars = []
    for c in join_node.left_keys:
        in_col_num = join_node.left_var_map[c]
        left_key_vars.append(join_node.left_vars[in_col_num])
        is_live = 1
        if c == INDEX_SENTINEL:
            # If we are joining on the index, we may need to return
            # the index to the output. If so, the output can be either
            # from the keys or the data columns, which we track via
            # the index_source and index_col_num. If the index column
            # exists but is not used in the output it is a dead key
            # by definition.
            if (
                join_node.has_live_index_var
                and join_node.index_source == "left"
                and join_node.index_col_num == in_col_num
            ):
                index_physical_index = physical_idx
                physical_idx += 1
                left_used_key_nums.add(in_col_num)
            else:
                is_live = 0
        else:
            out_col_num = join_node.left_to_output_map[in_col_num]
            if out_col_num not in join_node.out_used_cols:
                is_live = 0
            else:
                left_used_key_nums.add(in_col_num)
                cpp_table_logical_idx[out_col_num] = physical_idx
                physical_idx += 1
        left_logical_physical_map[in_col_num] = left_physical_index
        left_physical_index += 1
        left_key_in_output.append(is_live)
    left_key_vars = tuple(left_key_vars)

    # Extract the left column variables for the non-keys and determine
    # the physical index for each live column.
    left_other_col_vars = []
    for i, n in enumerate(join_node.left_col_names):
        v = join_node.left_vars[i]
        if i not in join_node.left_dead_var_inds and n not in join_node.left_key_set:
            left_other_col_vars.append(v)
            is_live = 1
            out_col_num = join_node.left_to_output_map[i]
            if n in join_node.left_cond_cols:
                # Join conditions need to be tracked like keys
                if out_col_num not in join_node.out_used_cols:
                    is_live = 0
                left_key_in_output.append(is_live)
            if is_live:
                cpp_table_logical_idx[out_col_num] = physical_idx
                physical_idx += 1
            left_logical_physical_map[i] = left_physical_index
            left_physical_index += 1

    # Append the index data column if it exists
    if (
        join_node.has_live_index_var
        and join_node.index_source == "left"
        and index_physical_index == -1
    ):
        # If we are joining on the index, we may need to return
        # the index to the output. If so, the output can be either
        # from the keys or the data columns. Here we determine that
        # the index is a data column because its not a key.
        left_other_col_vars.append(join_node.left_vars[join_node.index_col_num])
        index_physical_index = physical_idx
        physical_idx += 1

    left_other_col_vars = tuple(left_other_col_vars)

    # Extract the right column variables for the keys and determine
    # the physical index for each live column.
    right_key_vars = []
    for i, c in enumerate(join_node.right_keys):
        in_col_num = join_node.right_var_map[c]
        right_key_vars.append(join_node.right_vars[in_col_num])
        if not join_node.vect_same_key[i] and not join_node.is_join:
            is_live = 1
            if c == INDEX_SENTINEL:
                # If we are joining on the index, we may need to return
                # the index to the output. If so, the output can be either
                # from the keys or the data columns, which we track via
                # the index_source and index_col_num. If the index column
                # exists but is not used in the output it is a dead key
                # by definition.
                if (
                    join_node.has_live_index_var
                    and join_node.index_source == "right"
                    and join_node.index_col_num == in_col_num
                ):
                    index_physical_index = physical_idx
                    physical_idx += 1
                    right_used_key_nums.add(in_col_num)
                else:
                    is_live = 0
            else:
                out_col_num = join_node.right_to_output_map[in_col_num]
                if out_col_num not in join_node.out_used_cols:
                    is_live = 0
                else:
                    right_used_key_nums.add(in_col_num)
                    cpp_table_logical_idx[out_col_num] = physical_idx
                    physical_idx += 1
            right_key_in_output.append(is_live)
        right_logical_physical_map[in_col_num] = right_physical_index
        right_physical_index += 1
    right_key_vars = tuple(right_key_vars)

    # Extract the right column variables for the non-keys and determine
    # the physical index for each live column.
    right_other_col_vars = []
    for i, n in enumerate(join_node.right_col_names):
        v = join_node.right_vars[i]
        if i not in join_node.right_dead_var_inds and n not in join_node.right_key_set:
            right_other_col_vars.append(v)
            is_live = 1
            out_col_num = join_node.right_to_output_map[i]
            if n in join_node.right_cond_cols:
                # Join conditions need to be tracked like keys
                if out_col_num not in join_node.out_used_cols:
                    is_live = 0
                right_key_in_output.append(is_live)
            if is_live:
                cpp_table_logical_idx[out_col_num] = physical_idx
                physical_idx += 1
            right_logical_physical_map[i] = right_physical_index
            right_physical_index += 1

    # Append the index data column if it exists
    if (
        join_node.has_live_index_var
        and join_node.index_source == "right"
        and index_physical_index == -1
    ):
        # If we are joining on the index, we may need to return
        # the index to the output. If so, the output can be either
        # from the keys or the data columns. Here we determine that
        # the index is a data column because its not a key.
        right_other_col_vars.append(join_node.right_vars[join_node.index_col_num])
        index_physical_index = physical_idx
        physical_idx += 1

    right_other_col_vars = tuple(right_other_col_vars)

    if join_node.indicator_col_num != -1:
        # There is an indicator column in the output.
        cpp_table_logical_idx[join_node.indicator_col_num] = physical_idx
        physical_idx += 1

    # get column types
    arg_vars = (
        extra_data_col_var
        + left_key_vars
        + right_key_vars
        + left_other_col_vars
        + right_other_col_vars
    )
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

    # arg names of non-key columns
    extra_names = tuple("opti_c" + str(i) for i in range(len(extra_data_col_var)))

    left_other_names = tuple("t1_c" + str(i) for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i) for i in range(len(right_other_col_vars)))
    left_other_types = tuple([typemap[c.name] for c in left_other_col_vars])
    right_other_types = tuple([typemap[c.name] for c in right_other_col_vars])
    left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))
    glbs = {}
    loc = join_node.loc

    func_text = "def f({}{}, {},{}{}{}):\n".format(
        ("{},".format(extra_names[0]) if len(extra_names) == 1 else ""),
        ",".join(left_key_names),
        ",".join(right_key_names),
        ",".join(left_other_names),
        ("," if len(left_other_names) != 0 else ""),
        ",".join(right_other_names),
    )

    left_key_types = tuple(typemap[v.name] for v in left_key_vars)
    right_key_types = tuple(typemap[v.name] for v in right_key_vars)

    # add common key type to globals to use below for type conversion
    for i in range(n_keys):
        glbs[f"key_type_{i}"] = _match_join_key_types(
            left_key_types[i], right_key_types[i], loc
        )

    # Cast the keys to a common dtype for comparison
    func_text += "    t1_keys = ({},)\n".format(
        ", ".join(
            f"bodo.utils.utils.astype({left_key_names[i]}, key_type_{i})"
            for i in range(n_keys)
        )
    )
    func_text += "    t2_keys = ({},)\n".format(
        ", ".join(
            f"bodo.utils.utils.astype({right_key_names[i]}, key_type_{i})"
            for i in range(n_keys)
        )
    )

    # Extract the values as a tuple.
    func_text += "    data_left = ({}{})\n".format(
        ",".join(left_other_names), "," if len(left_other_names) != 0 else ""
    )
    func_text += "    data_right = ({}{})\n".format(
        ",".join(right_other_names), "," if len(right_other_names) != 0 else ""
    )

    # Generate a general join condition function if it exists
    # and determine the data columns it needs.
    general_cond_cfunc, left_col_nums, right_col_nums = _gen_general_cond_cfunc(
        join_node,
        typemap,
        left_logical_physical_map,
        right_logical_physical_map,
    )

    # TODO: Update asof to use table format.
    if join_node.how == "asof":
        if left_parallel or right_parallel:
            assert (
                left_parallel and right_parallel
            ), "pd.merge_asof requires both left and right to be replicated or distributed"
            # only the right key needs to be aligned
            func_text += "    t2_keys, data_right = parallel_asof_comm(t1_keys, t2_keys, data_right)\n"
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_merge_asof(t1_keys, t2_keys, data_left, data_right)\n"
        )
    else:
        func_text += _gen_local_hash_join(
            extra_data_col,
            left_key_types,
            right_key_types,
            left_other_names,
            right_other_names,
            left_other_types,
            right_other_types,
            join_node.vect_same_key,
            left_key_in_output,
            right_key_in_output,
            join_node.is_left,
            join_node.is_right,
            join_node.is_join,
            left_parallel,
            right_parallel,
            glbs,
            out_table_type,
            cpp_table_logical_idx,
            index_col_type,
            index_physical_index,
            join_node.get_out_table_used_cols(),
            left_used_key_nums,
            right_used_key_nums,
            join_node.loc,
            join_node.indicator_col_num != -1,
            join_node.is_na_equal,
            general_cond_cfunc,
            left_col_nums,
            join_node.left_to_output_map,
            right_col_nums,
            join_node.right_to_output_map,
        )
    # TODO: Update asof
    # Set the output variables.
    if join_node.how == "asof":
        for i in range(len(left_other_names)):
            func_text += "    left_{} = out_data_left[{}]\n".format(i, i)
        for i in range(len(right_other_names)):
            func_text += "    right_{} = out_data_right[{}]\n".format(i, i)
        for i in range(n_keys):
            func_text += f"    t1_keys_{i} = out_t1_keys[{i}]\n"
        for i in range(n_keys):
            func_text += f"    t2_keys_{i} = out_t2_keys[{i}]\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars["f"]

    glbs.update(
        {
            "bodo": bodo,
            "np": np,
            "pd": pd,
            "to_list_if_immutable_arr": to_list_if_immutable_arr,
            "cp_str_list_to_array": cp_str_list_to_array,
            "parallel_asof_comm": parallel_asof_comm,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "hash_join_table": hash_join_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
            "add_join_gen_cond_cfunc_sym": add_join_gen_cond_cfunc_sym,
            "get_join_cond_addr": get_join_cond_addr,
            # key_in_output is defined to contain left_table then right_table
            # to match the iteration order in C++
            "key_in_output": np.array(
                left_key_in_output + right_key_in_output, dtype=np.bool_
            ),
            "cpp_table_to_py_table": cpp_table_to_py_table,
            "init_table": init_table,
            "set_table_len": set_table_len,
        }
    )
    if general_cond_cfunc:
        glbs.update({"general_cond_cfunc": general_cond_cfunc})

    f_block = compile_to_numba_ir(
        join_impl,
        glbs,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, arg_vars)

    # Replace the return values with assignments to the output IR
    nodes = f_block.body[:-3]
    if join_node.has_live_index_var:
        nodes[-1].target = join_node.out_data_vars[1]
    if join_node.has_live_table_var:
        nodes[-2].target = join_node.out_data_vars[0]
    assert (
        join_node.has_live_index_var or join_node.has_live_table_var
    ), "At most one of table and index should be dead if the Join IR node is live"
    if not join_node.has_live_index_var:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not join_node.has_live_table_var:
        nodes.pop(-2)
    return nodes


distributed_pass.distributed_run_extensions[Join] = join_distributed_run


def _gen_general_cond_cfunc(
    join_node, typemap, left_logical_physical_map, right_logical_physical_map
):
    """Generate cfunc for general join condition and return its address.
    Return 0 (NULL) if there is no general join condition to evaluate.
    The cfunc takes data pointers of table columns and row indices to access as input
    and returns True or False.
    E.g. left_table=[A_data_ptr, B_data_ptr], right_table=[A_data_ptr, C_data_ptr],
    left_ind=3, right_ind=7

    Args:
        join_node (Join): The join node being used.
        typemap (Dict[str, types.Type]): The type map for determine the code
            to generate for each array
        left_logical_physical_map (Dict[int, int]): Mapping from the logical
            column number of the left table to the physical number of the C++
            table. This is done because of dead columns.
        right_logical_physical_map (Dict[int, int]): Mapping from the logical
            column number of the right table to the physical number of the C++
            table. This is done because of dead columns.

    Returns:
        Tuple[cfunc, List[int], List[int]]: Triple containing the generated
            cfunc and the columns used by each table.
    """
    expr = join_node.gen_cond_expr
    if not expr:
        return None, [], []

    label_suffix = next_label()

    table_getitem_funcs = {
        "bodo": bodo,
        "numba": numba,
        "is_null_pointer": is_null_pointer,
    }
    na_check_name = "NOT_NA"
    func_text = f"def bodo_join_gen_cond{label_suffix}(left_table, right_table, left_data1, right_data1, left_null_bitmap, right_null_bitmap, left_ind, right_ind):\n"
    func_text += "  if is_null_pointer(left_table):\n"
    func_text += "    return False\n"

    expr, func_text, left_col_nums = _replace_column_accesses(
        expr,
        left_logical_physical_map,
        join_node.left_var_map,
        typemap,
        join_node.left_vars,
        table_getitem_funcs,
        func_text,
        "left",
        len(join_node.left_keys),
        na_check_name,
    )
    expr, func_text, right_col_nums = _replace_column_accesses(
        expr,
        right_logical_physical_map,
        join_node.right_var_map,
        typemap,
        join_node.right_vars,
        table_getitem_funcs,
        func_text,
        "right",
        len(join_node.right_keys),
        na_check_name,
    )
    func_text += f"  return {expr}"

    loc_vars = {}
    exec(func_text, table_getitem_funcs, loc_vars)
    cond_func = loc_vars[f"bodo_join_gen_cond{label_suffix}"]

    c_sig = types.bool_(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int64,
    )
    cfunc_cond = numba.cfunc(c_sig, nopython=True)(cond_func)
    # Store the function inside join_gen_cond_cfunc
    join_gen_cond_cfunc[cfunc_cond.native_name] = cfunc_cond
    join_gen_cond_cfunc_addr[cfunc_cond.native_name] = cfunc_cond.address
    return cfunc_cond, left_col_nums, right_col_nums


def _replace_column_accesses(
    expr,
    logical_to_physical_ind,
    name_to_var_map,
    typemap,
    col_vars,
    table_getitem_funcs,
    func_text,
    table_name,
    n_keys,
    na_check_name,
):
    """replace column accesses in join condition expression with an intrinsic that loads
    values from table data pointers.
    For example, left.B is replaced with data_ptrs[1][row_ind]

    This function returns the modified expression, the func_text defining the column
    accesses, and the list of column numbers that are used by the table.
    """
    col_nums = []
    for c, c_ind in name_to_var_map.items():
        cname = f"({table_name}.{c})"
        if cname not in expr:
            continue
        getitem_fname = f"getitem_{table_name}_val_{c_ind}"
        val_varname = f"_bodo_{table_name}_val_{c_ind}"
        array_typ = typemap[col_vars[c_ind].name]
        if is_str_arr_type(array_typ):
            # If we have unicode we pass the table variable which is an array info
            func_text += f"  {val_varname}, {val_varname}_size = {getitem_fname}({table_name}_table, {table_name}_ind)\n"
            # Create proper Python string.
            func_text += f"  {val_varname} = bodo.libs.str_arr_ext.decode_utf8({val_varname}, {val_varname}_size)\n"
        else:
            # If we have a numeric type we just pass the data pointers
            func_text += f"  {val_varname} = {getitem_fname}({table_name}_data1, {table_name}_ind)\n"

        physical_ind = logical_to_physical_ind[c_ind]

        table_getitem_funcs[getitem_fname] = bodo.libs.array._gen_row_access_intrinsic(
            array_typ, physical_ind
        )
        expr = expr.replace(cname, val_varname)

        # We should only require an NA check if the column is also present
        na_cname = f"({na_check_name}.{table_name}.{c})"
        if na_cname in expr:
            na_check_fname = f"nacheck_{table_name}_val_{c_ind}"
            na_val_varname = f"_bodo_isna_{table_name}_val_{c_ind}"
            if (
                isinstance(array_typ, bodo.libs.int_arr_ext.IntegerArrayType)
                or array_typ == bodo.libs.bool_arr_ext.boolean_array
                or is_str_arr_type(array_typ)
            ):
                func_text += f"  {na_val_varname} = {na_check_fname}({table_name}_null_bitmap, {table_name}_ind)\n"
            else:
                func_text += f"  {na_val_varname} = {na_check_fname}({table_name}_data1, {table_name}_ind)\n"

            table_getitem_funcs[
                na_check_fname
            ] = bodo.libs.array._gen_row_na_check_intrinsic(array_typ, physical_ind)
            expr = expr.replace(na_cname, na_val_varname)

        # only append the column if it is not a key
        if c_ind >= n_keys:
            col_nums.append(physical_ind)
    return expr, func_text, col_nums


def _match_join_key_types(t1, t2, loc):
    """make sure join key array types match since required in the C++ join code"""
    if t1 == t2:
        return t1

    # Matching string + dictionary encoded arrays produces
    # a string key.
    if is_str_arr_type(t1) and is_str_arr_type(t2):
        return bodo.string_array_type

    try:
        arr = dtype_to_array_type(find_common_np_dtype([t1, t2]))
        # output should be nullable if any input is nullable
        return (
            to_nullable_type(arr)
            if is_nullable_type(t1) or is_nullable_type(t2)
            else arr
        )
    except Exception:
        raise BodoError(f"Join key types {t1} and {t2} do not match", loc=loc)


def _get_table_parallel_flags(join_node, array_dists):
    """
    Determine if the input Tables are parallel. This verifies
    that if either of the inputs is parallel, the output is
    also parallel.
    """
    par_dists = (
        distributed_pass.Distribution.OneD,
        distributed_pass.Distribution.OneD_Var,
    )

    left_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.get_live_left_vars()
    )
    right_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.get_live_right_vars()
    )
    if not left_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.get_live_left_vars()
        )
    if not right_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.get_live_right_vars()
        )

    if left_parallel or right_parallel:
        assert all(
            array_dists[v.name] in par_dists for v in join_node.get_live_out_vars()
        )

    return left_parallel, right_parallel


def _gen_local_hash_join(
    extra_data_col,
    left_key_types,
    right_key_types,
    left_other_names,
    right_other_names,
    left_other_types,
    right_other_types,
    vect_same_key,
    left_key_in_output,
    right_key_in_output,
    is_left,
    is_right,
    is_join,
    left_parallel,
    right_parallel,
    glbs,
    out_table_type,
    cpp_table_logical_idx,
    index_col_type,
    index_physical_index,
    out_table_used_cols,
    left_used_key_nums,
    right_used_key_nums,
    loc,
    indicator,
    is_na_equal,
    general_cond_cfunc,
    left_col_nums,
    left_to_output_map,
    right_col_nums,
    right_to_output_map,
):
    """
    Generate the code need to compute a hash join in C++
    """
    # In some case the column in output has a type different from the one in input.
    # TODO: Unify those type changes between all cases.
    def needs_typechange(in_type, need_nullable, is_same_key):
        return (
            isinstance(in_type, types.Array)
            and not is_dtype_nullable(in_type.dtype)
            and need_nullable
            and not is_same_key
        )

    # The vect_need_typechange is computed in the python code and is sent to C++.
    # This is a general approach for this kind of combinatorial problem: compute in python
    # preferably to C++. Compute in dataframe_pass.py preferably to the IR node.
    #
    # The vect_need_typechange is for the need to change the type in some cases.
    # Following constraints have to be taken into account:
    # ---For NullableArrayType the output column has the same format as the input column
    # ---For numpy array of float the output column has the same format as the input column
    # ---For numpy array of integer it may happen that we need to add missing entries and so
    #  we change the output type.
    # ---For categorical array data, the input is integer and we do not change the type.
    #   We may have to change this if missing data in categorical gets treated differently.

    # Sets for determining dead outputs for general condition columns
    left_cond_col_nums_set = set(left_col_nums)
    right_cond_col_nums_set = set(right_col_nums)

    # List of columns in the output that need to be cast.
    vect_need_typechange = []
    for i in range(len(left_key_types)):
        if left_key_in_output[i]:
            key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
            vect_need_typechange.append(
                needs_typechange(key_type, is_right, vect_same_key[i])
            )

    # Offset for left and right table inside the key in output
    # lists. Every left key is always included so we only count
    # the left table for data columns.
    left_key_in_output_idx = len(left_key_types)
    right_key_in_output_idx = 0

    # Determine the number of keys in the left table
    left_key_offset = left_key_in_output_idx

    for i in range(len(left_other_names)):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + left_key_offset in left_cond_col_nums_set:
            load_arr = left_key_in_output[left_key_in_output_idx]
            left_key_in_output_idx += 1
        if load_arr:
            vect_need_typechange.append(
                needs_typechange(left_other_types[i], is_right, False)
            )

    for i in range(len(right_key_types)):
        if not vect_same_key[i] and not is_join:
            if right_key_in_output[right_key_in_output_idx]:
                key_type = _match_join_key_types(
                    left_key_types[i], right_key_types[i], loc
                )
                vect_need_typechange.append(needs_typechange(key_type, is_left, False))
            right_key_in_output_idx += 1

    # Determine the number of keys in the right table
    right_key_offset = right_key_in_output_idx

    for i in range(len(right_other_names)):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + right_key_offset in right_cond_col_nums_set:
            load_arr = right_key_in_output[right_key_in_output_idx]
            right_key_in_output_idx += 1
        if load_arr:
            vect_need_typechange.append(
                needs_typechange(right_other_types[i], is_left, False)
            )

    n_keys = len(left_key_types)
    func_text = "    # beginning of _gen_local_hash_join\n"
    eList_l = []
    for i in range(n_keys):
        eList_l.append("t1_keys[{}]".format(i))
    for i in range(len(left_other_names)):
        eList_l.append("data_left[{}]".format(i))
    func_text += "    info_list_total_l = [{}]\n".format(
        ",".join("array_to_info({})".format(a) for a in eList_l)
    )
    func_text += "    table_left = arr_info_list_to_table(info_list_total_l)\n"
    eList_r = []
    for i in range(n_keys):
        eList_r.append("t2_keys[{}]".format(i))
    for i in range(len(right_other_names)):
        eList_r.append("data_right[{}]".format(i))
    func_text += "    info_list_total_r = [{}]\n".format(
        ",".join("array_to_info({})".format(a) for a in eList_r)
    )
    func_text += "    table_right = arr_info_list_to_table(info_list_total_r)\n"
    # Add globals that will be used in the function call.
    glbs["vect_same_key"] = np.array(vect_same_key, dtype=np.int64)
    glbs["vect_need_typechange"] = np.array(vect_need_typechange, dtype=np.int64)
    glbs["left_table_cond_columns"] = np.array(
        left_col_nums if len(left_col_nums) > 0 else [-1], dtype=np.int64
    )
    glbs["right_table_cond_columns"] = np.array(
        right_col_nums if len(right_col_nums) > 0 else [-1], dtype=np.int64
    )
    if general_cond_cfunc:
        func_text += f"    cfunc_cond = add_join_gen_cond_cfunc_sym(general_cond_cfunc, '{general_cond_cfunc.native_name}')\n"
        func_text += (
            f"    cfunc_cond = get_join_cond_addr('{general_cond_cfunc.native_name}')\n"
        )
    else:
        func_text += "    cfunc_cond = 0\n"

    # single-element numpy array to return number of global rows from C++
    func_text += f"    total_rows_np = np.array([0], dtype=np.int64)\n"
    func_text += "    out_table = hash_join_table(table_left, table_right, {}, {}, {}, {}, {}, vect_same_key.ctypes, key_in_output.ctypes, vect_need_typechange.ctypes, {}, {}, {}, {}, {}, {}, cfunc_cond, left_table_cond_columns.ctypes, {}, right_table_cond_columns.ctypes, {}, total_rows_np.ctypes)\n".format(
        left_parallel,
        right_parallel,
        n_keys,
        len(left_other_names),
        len(right_other_names),
        is_left,
        is_right,
        is_join,
        extra_data_col,
        indicator,
        is_na_equal,
        len(left_col_nums),
        len(right_col_nums),
    )
    func_text += "    delete_table(table_left)\n"
    func_text += "    delete_table(table_right)\n"

    if out_table_type == types.none:
        # The table is dead
        func_text += f"    T = None\n"
    else:
        glbs["py_table_type"] = out_table_type
        if not out_table_used_cols:
            # The table isn't dead but there aren't any output columns.
            # We only need a length.
            func_text += "    null_table = init_table(py_table_type, False)\n"
            func_text += "    T = set_table_len(null_table, total_rows_np[0])\n"
        else:
            # The table is live
            func_text += (
                f"    T = cpp_table_to_py_table(out_table, table_idx, py_table_type)\n"
            )
            glbs["table_idx"] = np.array(cpp_table_logical_idx, np.int64)
            cast_map = determine_out_table_cast_map(
                left_key_types,
                right_key_types,
                left_used_key_nums,
                right_used_key_nums,
                left_to_output_map,
                right_to_output_map,
                loc,
            )
            if cast_map:
                func_text += f"    T = bodo.utils.table_utils.table_astype(T, cast_table_type, False, _bodo_nan_to_str=False, used_cols=used_cols)\n"
                # Determine the types that must be loaded.
                pre_cast_array_types = list(out_table_type.arr_types)
                for col_num, typ in cast_map.items():
                    pre_cast_array_types[col_num] = typ
                pre_cast_table_type = bodo.TableType(tuple(pre_cast_array_types))
                # Update the table types
                glbs["py_table_type"] = pre_cast_table_type
                glbs["cast_table_type"] = out_table_type
                glbs["used_cols"] = bodo.utils.typing.MetaType(
                    tuple(out_table_used_cols)
                )
    if index_physical_index != -1:
        func_text += f"    index_var = info_to_array(info_from_table(out_table, {index_physical_index}), index_col_type)\n"
        glbs["index_col_type"] = index_col_type
    else:
        func_text += f"    index_var = None\n"
    if out_table_used_cols or index_physical_index != -1:
        # Only delete the C++ table if it is not a nullptr (0 output columns returns nullptr)
        func_text += "    delete_table(out_table)\n"
    func_text += f"    out_table = T\n"
    func_text += f"    out_index = index_var\n"
    return func_text


def determine_out_table_cast_map(
    left_key_types: List[types.Type],
    right_key_types: List[types.Type],
    left_used_key_nums: Set[int],
    right_used_key_nums: Set[int],
    left_to_output_map: Dict[int, int],
    right_to_output_map: Dict[int, int],
    loc: ir.Loc,
):
    """Determine any columns in the output table keys that were
    cast on the input of the to enable consistent hashing. These
    then need to be cast on output to convert the keys back to
    the correct output type.

    For example with an inner join, if left the key type is int64
    and the right key type is float64 then the input casts the
    left to int64->float64 and the output needs to cast it back,
    float64 -> int64. To do this, we determine the logical column
    numbers in the output table that must be cast and return a
    dictionary that will be used to the correct type information
    when loading the table from C++.

    Args:
        left_key_types (List[types.Type]): Type of the keys in the left table.
        right_key_types (List[types.Type]): Type of the keys in the right table.
        left_used_key_nums (Set[int]): Set of logical column indices in the left table
            that are also live in the output.
        right_used_key_nums (Set[int]): Set of logical column indices in the right table
            that are also live in the output.
        left_to_output_map (Dict[int, int]): Dictionary mapping the logical index
            in the left input to the logical index in the output.
        right_to_output_map (Dict[int, int]): Dictionary mapping the logical index
            in the right input to the logical index in the output.
        loc (ir.Loc): Location in the source code. Used for generating
            error messages.

    Returns:
        Dict[int, types.Type]: Dictionary mapping the logical column number
        in the output table to the correct output type when loading the table.
    """
    cast_map: Dict[int, types.Type] = {}

    # Check the keys for casts.
    n_keys = len(left_key_types)
    key_nums_list = [left_key_types, right_key_types]
    used_key_nums_list = [left_used_key_nums, right_used_key_nums]
    input_output_map_list = [left_to_output_map, right_to_output_map]
    for i in range(len(key_nums_list)):
        key_types = key_nums_list[i]
        used_key_nums = used_key_nums_list[i]
        input_output_map = input_output_map_list[i]
        for j in range(n_keys):
            # Determine if the column is live in the output.
            if j in used_key_nums:
                key_type = _match_join_key_types(
                    left_key_types[j], right_key_types[j], loc
                )
                # Astype is needed when the key had to be cast for the join
                # (e.g. left=int64 and right=float64 casts the left to float64)
                # and we need the key back to the original type in the output.
                if key_type != key_types[j] and key_types[j] != bodo.dict_str_arr_type:
                    # Need to generate an astype for the table.
                    # The CPP Table's return type contains key_types[j]
                    # and we need to convert to key_type.
                    cast_map[input_output_map[j]] = key_type

    return cast_map


@numba.njit
def parallel_asof_comm(left_key_arrs, right_key_arrs, right_data):  # pragma: no cover
    # align the left and right intervals
    # allgather the boundaries of all left intervals and calculate overlap
    # rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    # TODO: multiple keys
    bnd_starts = np.empty(n_pes, left_key_arrs[0].dtype)
    bnd_ends = np.empty(n_pes, left_key_arrs[0].dtype)
    bodo.libs.distributed_api.allgather(bnd_starts, left_key_arrs[0][0])
    bodo.libs.distributed_api.allgather(bnd_ends, left_key_arrs[0][-1])

    send_counts = np.zeros(n_pes, np.int32)
    send_disp = np.zeros(n_pes, np.int32)
    recv_counts = np.zeros(n_pes, np.int32)
    my_start = right_key_arrs[0][0]
    my_end = right_key_arrs[0][-1]

    offset = -1
    i = 0
    # ignore no overlap processors (end of their interval is before current)
    while i < n_pes - 1 and bnd_ends[i] < my_start:
        i += 1
    while i < n_pes and bnd_starts[i] <= my_end:
        offset, count = _count_overlap(right_key_arrs[0], bnd_starts[i], bnd_ends[i])
        # one extra element in case first value is needed for start of boundary
        if offset != 0:
            offset -= 1
            count += 1
        send_counts[i] = count
        send_disp[i] = offset
        i += 1
    # one extra element in case last value is need for start of boundary
    # TODO: see if next processor provides the value
    while i < n_pes:
        send_counts[i] = 1
        send_disp[i] = len(right_key_arrs[0]) - 1
        i += 1

    bodo.libs.distributed_api.alltoall(send_counts, recv_counts, 1)
    n_total_recv = recv_counts.sum()
    out_r_keys = np.empty(n_total_recv, right_key_arrs[0].dtype)
    # TODO: support string
    out_r_data = alloc_arr_tup(n_total_recv, right_data)
    recv_disp = bodo.ir.join.calc_disp(recv_counts)
    bodo.libs.distributed_api.alltoallv(
        right_key_arrs[0], out_r_keys, send_counts, recv_counts, send_disp, recv_disp
    )
    bodo.libs.distributed_api.alltoallv_tup(
        right_data, out_r_data, send_counts, recv_counts, send_disp, recv_disp
    )

    return (out_r_keys,), out_r_data


@numba.njit
def _count_overlap(r_key_arr, start, end):  # pragma: no cover
    # TODO: use binary search
    count = 0
    offset = 0
    j = 0
    while j < len(r_key_arr) and r_key_arr[j] < start:
        offset += 1
        j += 1
    while j < len(r_key_arr) and start <= r_key_arr[j] <= end:
        j += 1
        count += 1
    return offset, count


import llvmlite.binding as ll

from bodo.libs import hdist

ll.add_symbol("c_alltoallv", hdist.c_alltoallv)


@numba.njit
def calc_disp(arr):  # pragma: no cover
    disp = np.empty_like(arr)
    disp[0] = 0
    for i in range(1, len(arr)):
        disp[i] = disp[i - 1] + arr[i - 1]
    return disp


@numba.njit
def local_merge_asof(left_keys, right_keys, data_left, data_right):  # pragma: no cover
    # adapted from pandas/_libs/join_func_helper.pxi
    l_size = len(left_keys[0])
    r_size = len(right_keys[0])

    out_left_keys = alloc_arr_tup(l_size, left_keys)
    out_right_keys = alloc_arr_tup(l_size, right_keys)
    out_data_left = alloc_arr_tup(l_size, data_left)
    out_data_right = alloc_arr_tup(l_size, data_right)

    left_ind = 0
    right_ind = 0

    for left_ind in range(l_size):
        # restart right_ind if it went negative in a previous iteration
        if right_ind < 0:
            right_ind = 0

        # find last position in right whose value is less than left's
        while right_ind < r_size and getitem_arr_tup(
            right_keys, right_ind
        ) <= getitem_arr_tup(left_keys, left_ind):
            right_ind += 1

        right_ind -= 1

        setitem_arr_tup(out_left_keys, left_ind, getitem_arr_tup(left_keys, left_ind))
        # TODO: copy_tup
        setitem_arr_tup(out_data_left, left_ind, getitem_arr_tup(data_left, left_ind))

        if right_ind >= 0:
            setitem_arr_tup(
                out_right_keys, left_ind, getitem_arr_tup(right_keys, right_ind)
            )
            setitem_arr_tup(
                out_data_right, left_ind, getitem_arr_tup(data_right, right_ind)
            )
        else:
            bodo.libs.array_kernels.setna_tup(out_right_keys, left_ind)
            bodo.libs.array_kernels.setna_tup(out_data_right, left_ind)

    return out_left_keys, out_right_keys, out_data_left, out_data_right
