"""IR node for the join and merge"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numba
import numpy as np
import pandas as pd
import pandas.core.computation.expr
import pandas.core.computation.ops
import pandas.core.computation.parsing
import pandas.core.computation.scope
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    guard,
    next_label,
    replace_arg_nodes,
    replace_vars_inner,
    require,
    visit_vars_inner,
)
from numba.extending import intrinsic

import bodo
from bodo.hiframes.table import TableType
from bodo.ir.connector import trim_extra_used_columns
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    cpp_table_to_py_data,
    delete_table,
    hash_join_table,
    interval_join_table,
    nested_loop_join_table,
    py_data_to_cpp_table,
)
from bodo.libs.vendored.timsort import getitem_arr_tup, setitem_arr_tup
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    _compute_table_column_uses,
    get_live_column_nums_block,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import (
    INDEX_SENTINEL,
    BodoError,
    MetaType,
    dtype_to_array_type,
    find_common_np_dtype,
    is_dtype_nullable,
    is_nullable_type,
    is_str_arr_type,
    to_nullable_type,
)
from bodo.utils.utils import alloc_arr_tup, is_null_pointer

if TYPE_CHECKING:  # pragma: no cover
    from bodo.hiframes.pd_dataframe_ext import DataFrameType


# TODO: it's probably a bad idea for these to be global. Maybe try moving them
# to a context or dispatcher object somehow
# Maps symbol name to cfunc object that implements a general condition function.
# This dict is used only when compiling
join_gen_cond_cfunc = {}
# Maps symbol name to cfunc address (used when compiling and loading from cache)
# When compiling, this is populated in join.py::gen_general_cond_cfunc
# When loading from cache, this is populated in numba_compat.py::resolve_join_general_cond_funcs
# when the compiled result is loaded from cache
join_gen_cond_cfunc_addr = {}


@intrinsic(prefer_literal=True)
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
        # See: gen_general_cond_cfunc
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


@numba.njit
def get_join_cond_addr(name):
    """Resolve address of cfunc given by its symbol name"""
    with bodo.no_warning_objmode(addr="int64"):
        # This loads the function pointer at runtime, preventing
        # hardcoding the address into the IR.
        addr = bodo.ir.join.join_gen_cond_cfunc_addr[name]
    return addr


HOW_OPTIONS = Literal["inner", "left", "right", "outer", "asof", "cross"]


class Join(ir.Stmt):
    def __init__(
        self,
        left_keys: list[str] | str,
        right_keys: list[str] | str,
        out_data_vars: list[ir.Var],
        out_df_type: "DataFrameType",
        left_vars: list[ir.Var],
        left_df_type: "DataFrameType",
        right_vars: list[ir.Var],
        right_df_type: "DataFrameType",
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
        rebalance_output_if_skewed: bool,
        gen_cond_expr: str,
        left_len_var: ir.Var,
        right_len_var: ir.Var,
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
        rebalance_output_if_skewed -- Should the output be rebalanced if it is skewed?
        gen_cond_expr -- String used to describe the general merge condition. This
                         is used when a more general condition is needed than is
                         provided by equality.
        """
        self.left_keys = left_keys
        self.right_keys = right_keys
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
        self.rebalance_output_if_skewed = rebalance_output_if_skewed
        self.gen_cond_expr = gen_cond_expr
        self.left_len_var = left_len_var
        self.right_len_var = right_len_var
        # Columns within the output table type that are actually used.
        # These will be updated during optimizations. For more
        # information see 'join_remove_dead_column'.
        self.n_out_table_cols = len(self.out_col_names)
        self.out_used_cols = set(range(self.n_out_table_cols))
        if self.out_data_vars[1] is not None:
            self.out_used_cols.add(self.n_out_table_cols)

        left_col_names: Sequence[str] = left_df_type.columns  # type: ignore
        right_col_names: Sequence[str] = right_df_type.columns  # type: ignore
        self.left_col_names = left_col_names
        self.right_col_names = right_col_names
        self.is_left_table = left_df_type.is_table_format
        self.is_right_table = right_df_type.is_table_format

        self.n_left_table_cols = len(left_col_names) if self.is_left_table else 0
        self.n_right_table_cols = len(right_col_names) if self.is_right_table else 0

        left_index_loc = (
            self.n_left_table_cols if self.is_left_table else len(left_vars) - 1
        )
        right_index_loc = (
            self.n_right_table_cols if self.is_right_table else len(right_vars) - 1
        )

        # Track the indices of dead vars for the left and right
        # inputs. These hold the logical indices for the variables.
        self.left_dead_var_inds = set()
        self.right_dead_var_inds = set()
        if self.left_vars[-1] is None:
            self.left_dead_var_inds.add(left_index_loc)
        if self.right_vars[-1] is None:
            self.right_dead_var_inds.add(right_index_loc)

        # Create a map for selecting variables
        self.left_var_map = {c: i for i, c in enumerate(left_col_names)}
        self.right_var_map = {c: i for i, c in enumerate(right_col_names)}
        # If INDEX_SENTINEL exists its always the last var
        if self.left_vars[-1] is not None:
            self.left_var_map[INDEX_SENTINEL] = left_index_loc
        if self.right_vars[-1] is not None:
            self.right_var_map[INDEX_SENTINEL] = right_index_loc

        # Create a set of keys future lookups
        self.left_key_set = {self.left_var_map[c] for c in left_keys}
        self.right_key_set = {self.right_var_map[c] for c in right_keys}

        if gen_cond_expr:
            # find columns used in general join condition to avoid removing them in rm dead
            # Note: this generates code per key and also is not fully correct. An expression
            # like (left.A)) != (right.B) will look like both A and A) are left key columns
            # based on this check, even though only A) is. Fixing this requires a more detailed
            # refactoring of the parsing.
            self.left_cond_cols = {
                self.left_var_map[c]
                for c in left_col_names
                if f"(left.{c})" in gen_cond_expr
            }
            self.right_cond_cols = {
                self.right_var_map[c]
                for c in right_col_names
                if f"(right.{c})" in gen_cond_expr
            }
        else:
            self.left_cond_cols = set()
            self.right_cond_cols = set()

        # When merging with one key on the index and the other on a column,
        # you can have the data column repeated as both a key and and data.
        # In this situation, since the column is used twice we must generate
        # an extra data column via the input. For an example, please refer to
        # test_merge_index_column.
        extra_data_col_num: int = -1

        # Compute maps for each input to the output and the
        # output to the inputs.
        comm_keys = set(left_keys) & set(right_keys)
        comm_data = set(left_col_names) & set(right_col_names)
        add_suffix = comm_data - comm_keys
        # Map the output column numbers to the input
        # locations. We use this to avoid repeating
        # the conversion.
        out_to_input_col_map: dict[int, (Literal["left", "right"], int)] = {}
        # Map each input to the output location.
        left_to_output_map: dict[int, int] = {}
        right_to_output_map: dict[int, int] = {}
        for i, c in enumerate(left_col_names):
            if c in add_suffix:
                suffixed_left_name = str(c) + suffix_left
                out_col_num = out_df_type.column_index[suffixed_left_name]
                # If a column is both a data column and a key
                # from left we have an extra column.
                if right_index and not left_index and i in self.left_key_set:
                    extra_data_col_num = out_df_type.column_index[c]
                    out_to_input_col_map[extra_data_col_num] = ("left", i)
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
                    # from right we have an extra column.
                    if left_index and not right_index and i in self.right_key_set:
                        extra_data_col_num = out_df_type.column_index[c]
                        out_to_input_col_map[extra_data_col_num] = ("right", i)
                else:
                    out_col_num = out_df_type.column_index[c]
                out_to_input_col_map[out_col_num] = ("right", i)
                right_to_output_map[i] = out_col_num

        if self.left_vars[-1] is not None:
            left_to_output_map[left_index_loc] = self.n_out_table_cols
        if self.right_vars[-1] is not None:
            right_to_output_map[right_index_loc] = self.n_out_table_cols

        self.out_to_input_col_map = out_to_input_col_map
        self.left_to_output_map = left_to_output_map
        self.right_to_output_map = right_to_output_map
        self.extra_data_col_num = extra_data_col_num

        if self.out_data_vars[1] is not None:
            # Compute the source for the index. Note we only
            # need to track the source for a possible output
            # index.
            index_source = "left" if right_index else "right"
            if index_source == "left":
                index_col_num = left_index_loc
            elif index_source == "right":
                index_col_num = right_index_loc
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
        # Store information if we select the optimized point interval
        # join implementation.
        self.point_interval_join_info: tuple[bool, str, str, str] | None = None

    @property
    def has_live_left_table_var(self):
        """Does this Join node contain a left input table
        that is live.

        Returns:
            bool: Left Table var exists and is live.
        """
        return self.is_left_table and self.left_vars[0] is not None

    @property
    def has_live_right_table_var(self):
        """Does this Join node contain a right input table
        that is live.

        Returns:
            bool: Right Table var exists and is live.
        """
        return self.is_right_table and self.right_vars[0] is not None

    @property
    def has_live_out_table_var(self):
        """Does this Join node contain a live output variable
        for the table.

        Returns:
            bool: Table output var exists and is live.
        """
        return self.out_data_vars[0] is not None

    @property
    def has_live_out_index_var(self):
        """Does this Join node contain a live output variable
        for the index.

        Returns:
            bool: Index output var exists and is live.
        """
        return self.out_data_vars[1] is not None

    def get_out_table_var(self):
        """Returns the table var for this Join Node's output.

        Returns:
            ir.Var: The table var.
        """
        return self.out_data_vars[0]

    def get_out_index_var(self):
        """Returns the index var for this Join Node's output.

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
        for var in self.left_vars:
            if var is not None:
                vars.append(var)
        return vars

    def get_live_right_vars(self):
        """Returns the right variables that are live
        for this join node.

        Returns:
            List[ir.Var]: Currently live variables.
        """
        vars = []
        for var in self.right_vars:
            if var is not None:
                vars.append(var)
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
        start = 0
        # Handle the table var
        if self.is_left_table:
            if self.has_live_left_table_var:
                left_vars.append(live_data_vars[idx])
                idx += 1
            else:
                left_vars.append(None)
            # Update the start for other vars
            start = 1
        # Handle array vars
        offset = max(self.n_left_table_cols - 1, 0)
        for i in range(start, len(self.left_vars)):
            if (i + offset) in self.left_dead_var_inds:
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
        start = 0
        # Handle the table var
        if self.is_right_table:
            if self.has_live_right_table_var:
                right_vars.append(live_data_vars[idx])
                idx += 1
            else:
                right_vars.append(None)
            # Update the start for other vars
            start = 1
        # Handle array vars
        offset = max(self.n_right_table_cols - 1, 0)
        for i in range(start, len(self.right_vars)):
            if (i + offset) in self.right_dead_var_inds:
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
        is_live = [self.has_live_out_table_var, self.has_live_out_index_var]
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
        return {i for i in self.out_used_cols if i < self.n_out_table_cols}

    def __repr__(self):  # pragma: no cover
        in_col_names = ", ".join([f"{c}" for c in self.left_col_names])
        df_left_str = f"left={{{in_col_names}}}"
        in_col_names = ", ".join([f"{c}" for c in self.right_col_names])
        df_right_str = f"right={{{in_col_names}}}"
        return (
            f"join [{self.left_keys}={self.right_keys}]: {df_left_str}, {df_right_str}"
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

    # left and right inputs can have 1D or 1D_Var separately (q26 case)
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

    # save distributions in case all input vars are dead (cross join corner case)
    # see _get_table_parallel_flags()
    join_node.left_dist = left_dist
    join_node.right_dist = right_dist


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
    if join_node.how == "cross":
        join_node.left_len_var = visit_vars_inner(
            join_node.left_len_var, callback, cbdata
        )
        join_node.right_len_var = visit_vars_inner(
            join_node.right_len_var, callback, cbdata
        )


# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join


def check_cross_join_coltypes(
    left_col_types: list[types.Type], right_col_types: list[types.Type]
):
    """
    Check the Columns of Cross Join or Interval Join tables to
    make sure that they don't use unsupported column types.
    Currently, we don not support cases where a timedelta
    column is used in the condition.
    """
    for col_type in chain(left_col_types, right_col_types):
        if col_type == bodo.datetime_timedelta_array_type or (
            isinstance(col_type, types.Array) and col_type.dtype == bodo.timedelta64ns
        ):
            raise BodoError(
                "The Timedelta column data type is not supported for Cross Joins or Joins with Inequality Conditions"
            )


def _is_cross_join_len(join_node):
    """Return True if we have a cross join with all output columns dead but output table
    alive. This means only the length of the output table is used (corner case).
    In this case, we need to keep input columns alive to calculate output length.
    Cross join needs special handling since it has no keys that would stay alive.
    See test_merge_cross_len_only.
    """
    return (
        join_node.how == "cross"
        and not join_node.out_used_cols
        and join_node.has_live_out_table_var
        and not join_node.has_live_out_index_var
    )


def remove_dead_join(
    join_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """
    Dead code elimination for the Join IR node. This finds columns that
    are dead data columns in the output and eliminates them from the
    inputs.
    """
    if join_node.has_live_out_table_var:
        # Columns to delete from out_to_input_col_map
        del_col_nums = []
        table_var = join_node.get_out_table_var()
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
            if col_num == join_node.extra_data_col_num:
                # If the extra column is removed, avoid generating
                join_node.extra_data_col_num = -1
                continue
            orig, col_num = join_node.out_to_input_col_map[col_num]
            if orig == "left":
                if (
                    col_num not in join_node.left_key_set
                    and col_num not in join_node.left_cond_cols
                ):
                    join_node.left_dead_var_inds.add(col_num)
                    if not join_node.is_left_table:
                        join_node.left_vars[col_num] = None
            elif orig == "right":
                if (
                    col_num not in join_node.right_key_set
                    and col_num not in join_node.right_cond_cols
                ):
                    join_node.right_dead_var_inds.add(col_num)
                    if not join_node.is_right_table:
                        join_node.right_vars[col_num] = None

        # Remove dead columns from the dictionary.
        for i in del_col_nums:
            del join_node.out_to_input_col_map[i]

        # Check if the left or right table is completely dead
        if join_node.is_left_table:
            all_dead_set = set(range(join_node.n_left_table_cols))
            remove_table = not bool(all_dead_set - join_node.left_dead_var_inds)
            if remove_table:
                join_node.left_vars[0] = None

        if join_node.is_right_table:
            all_dead_set = set(range(join_node.n_right_table_cols))
            remove_table = not bool(all_dead_set - join_node.right_dead_var_inds)
            if remove_table:
                join_node.right_vars[0] = None

    if join_node.has_live_out_index_var:
        index_var = join_node.get_out_index_var()
        if index_var.name not in lives:
            # Remove the IR var
            join_node.out_data_vars[1] = None
            # Update the input
            join_node.out_used_cols.remove(join_node.n_out_table_cols)
            if join_node.index_source == "left":
                if (
                    join_node.index_col_num not in join_node.left_key_set
                    and join_node.index_col_num not in join_node.left_cond_cols
                ):
                    join_node.left_dead_var_inds.add(join_node.index_col_num)
                    # The index is always the last var.
                    join_node.left_vars[-1] = None
            else:
                if (
                    join_node.index_col_num not in join_node.right_key_set
                    and join_node.index_col_num not in join_node.right_cond_cols
                ):
                    join_node.right_dead_var_inds.add(join_node.index_col_num)
                    # The index is always the last var.
                    join_node.right_vars[-1] = None

    if not (join_node.has_live_out_table_var or join_node.has_live_out_index_var):
        # remove empty join node
        return None

    return join_node


ir_utils.remove_dead_extensions[Join] = remove_dead_join


def join_remove_dead_column(join_node, column_live_map, equiv_vars, typemap):
    # Compute the columns that are live for the table
    changed = False
    if join_node.has_live_out_table_var:
        table_var_name = join_node.get_out_table_var().name
        table_key = (table_var_name, None)
        used_columns, use_all, cannot_del_cols = get_live_column_nums_block(
            column_live_map, equiv_vars, table_key
        )
        if not (use_all or cannot_del_cols):
            used_columns = trim_extra_used_columns(
                used_columns, join_node.n_out_table_cols
            )
            table_used_cols = join_node.get_out_table_used_cols()
            if len(used_columns) != len(table_used_cols):
                # If the inputs are not tables, dead column elimination could
                # lead to dead code elimination. If both inputs are tables this
                # node can't directly lead to dead code.
                changed = not (join_node.is_left_table and join_node.is_right_table)
                removed_cols = table_used_cols - used_columns
                join_node.out_used_cols = join_node.out_used_cols - removed_cols
    return changed


remove_dead_column_extensions[Join] = join_remove_dead_column


def join_table_column_use(
    join_node: Join,
    block_use_map: dict[str, tuple[set[int], bool, bool]],
    equiv_vars: dict[str, set[str]],
    typemap: dict[str, types.Type],
    table_col_use_map: dict[int, dict[str, tuple[set[int], bool, bool]]],
):
    """Compute column uses in input tables of Join based on output table's
    uses.

    Args:
        join_node (Join): Join node to process
        block_use_map (Dict[str, Tuple[Set[int], bool, bool]]): column uses for current
             block.
         equiv_vars (Dict[str, Set[str]]): Dictionary
             mapping table variable names to a set of
             other table name aliases.
         typemap (Dict[str, types.Type]): typemap of variables
         table_col_use_map (Dict[int, Dict[str, Tuple[Set[int], bool, bool]]]):
             Dictionary mapping block numbers to a dictionary of table names
             and "column uses". A column use is represented by the triple
                 - used_cols: Set of used column numbers
                 - use_all: Flag for if all columns are used. If True used_cols
                     is garbage
                 - cannot_del_columns: Flag indicate this table is used in
                     an unsupported operation (e.g. passed to a DataFrame)
                     and therefore no columns can be deleted.
    """

    # Only compute uses if there is a table input.
    if not (join_node.is_left_table or join_node.is_right_table):
        return

    # get output's uses
    if join_node.has_live_out_table_var:
        out_table_var = join_node.get_out_table_var()
        out_key = (out_table_var.name, None)
        (
            used_cols,
            use_all,
            cannot_del_cols,
        ) = _compute_table_column_uses(out_key, table_col_use_map, equiv_vars)
    else:
        (used_cols, use_all, cannot_del_cols) = (
            set(),
            False,
            False,
        )

    if join_node.has_live_left_table_var:
        left_table = join_node.left_vars[0].name
        left_key = (left_table, None)

        (
            orig_used_cols,
            orig_use_all,
            orig_cannot_del_cols,
        ) = block_use_map[left_key]

        # skip if input already uses all columns or cannot delete the table
        if not (orig_use_all or orig_cannot_del_cols):
            # Map the used columns from output to left
            left_used_cols = {
                join_node.out_to_input_col_map[i][1]
                for i in used_cols
                if join_node.out_to_input_col_map[i][0] == "left"
            }

            # key columns are always used in join
            left_key_cols = {
                i
                for i in (join_node.left_key_set | join_node.left_cond_cols)
                if i < join_node.n_left_table_cols
            }

            # Update the dead columns for the left
            if not (use_all or cannot_del_cols):
                join_node.left_dead_var_inds |= set(
                    range(join_node.n_left_table_cols)
                ) - (left_used_cols | left_key_cols)

            block_use_map[left_key] = (
                orig_used_cols | left_used_cols | left_key_cols,
                use_all or cannot_del_cols,
                False,
            )

    if join_node.has_live_right_table_var:
        right_table = join_node.right_vars[0].name
        right_key = (right_table, None)

        (
            orig_used_cols,
            orig_use_all,
            orig_cannot_del_cols,
        ) = block_use_map[right_key]

        # skip if input already uses all columns or cannot delete the table
        if not (orig_use_all or orig_cannot_del_cols):
            # Map the used columns from output to right
            right_used_cols = {
                join_node.out_to_input_col_map[i][1]
                for i in used_cols
                if join_node.out_to_input_col_map[i][0] == "right"
            }

            # key columns are always used in join
            right_key_cols = {
                i
                for i in (join_node.right_key_set | join_node.right_cond_cols)
                if i < join_node.n_right_table_cols
            }

            # Update the dead columns for the right
            if not (use_all or cannot_del_cols):
                join_node.right_dead_var_inds |= set(
                    range(join_node.n_right_table_cols)
                ) - (right_used_cols | right_key_cols)

            block_use_map[right_key] = (
                orig_used_cols | right_used_cols | right_key_cols,
                use_all or cannot_del_cols,
                False,
            )


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

    if join_node.how == "cross":
        use_set.add(join_node.left_len_var.name)
        use_set.add(join_node.right_len_var.name)

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    """
    Return gen and kill sets for a copy propagation
    data flow analysis. Join doesn't generate any
    copies, it just kills the output columns.
    """
    kill_set = {v.name for v in join_node.get_live_out_vars()}
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

    if join_node.how == "cross":
        join_node.left_len_var = replace_vars_inner(join_node.left_len_var, var_dict)
        join_node.right_len_var = replace_vars_inner(join_node.right_len_var, var_dict)


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


def _gen_cross_join_len(
    join_node,
    out_table_type,
    typemap,
    calltypes,
    typingctx,
    targetctx,
    left_parallel,
    right_parallel,
):
    """generate join output nodes for cross join corner case where only the length of
    the output table is used.
    Creates a dummy table with the output length assigned as product of input lengths.
    See test_merge_cross_len_only.
    """

    func_text = "def f(left_len, right_len):\n"

    # Bodo passes global lengths, which needs to be converted to local lengths to get
    # the correct size of output chunk
    n_pes = "bodo.libs.distributed_api.get_size()"
    rank = "bodo.libs.distributed_api.get_rank()"

    if left_parallel:
        func_text += f"  left_len = bodo.libs.distributed_api.get_node_portion(left_len, {n_pes}, {rank})\n"
    if right_parallel and not left_parallel:
        func_text += f"  right_len = bodo.libs.distributed_api.get_node_portion(right_len, {n_pes}, {rank})\n"

    func_text += "  n_rows = left_len * right_len\n"
    func_text += "  py_table = init_table(py_table_type, False)\n"
    func_text += "  py_table = set_table_len(py_table, n_rows)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars["f"]

    glbs = {
        "py_table_type": out_table_type,
        "init_table": bodo.hiframes.table.init_table,
        "set_table_len": bodo.hiframes.table.set_table_len,
        "sum_op": np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value),
        "bodo": bodo,
    }

    arg_vars = [join_node.left_len_var, join_node.right_len_var]
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

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
    nodes = f_block.body[:-3]
    nodes[-1].target = join_node.out_data_vars[0]
    return nodes


def _gen_cross_join_repeat(
    join_node,
    out_table_type,
    typemap,
    calltypes,
    typingctx,
    targetctx,
    left_parallel,
    right_parallel,
    left_is_dead,
):
    """generate code for cross join corner cases where all input data from one side
    is dead.
    Replicates the other side's data based on length of the dead side.
    """

    in_vars = join_node.right_vars if left_is_dead else join_node.left_vars
    data_args = ", ".join(
        f"t{i}" for i in range(len(in_vars)) if in_vars[i] is not None
    )

    n_in_data_cols = (
        len(join_node.right_col_names)
        if left_is_dead
        else len(join_node.left_col_names)
    )
    is_in_table = join_node.is_right_table if left_is_dead else join_node.is_left_table
    dead_in_inds = (
        join_node.right_dead_var_inds if left_is_dead else join_node.left_dead_var_inds
    )
    # TODO(ehsan): support repeat on tables directly
    data_cols = [
        f"get_table_data(t0, {i})" if is_in_table else f"t{i}"
        for i in range(n_in_data_cols)
    ]
    table_args = ", ".join(
        f"bodo.libs.array_kernels.repeat_kernel({data_cols[i]}, repeats)"
        if i not in dead_in_inds
        else "None"
        for i in range(n_in_data_cols)
    )

    n_out_cols = len(out_table_type.arr_types)
    col_inds = [
        join_node.out_to_input_col_map.get(i, (-1, -1))[1] for i in range(n_out_cols)
    ]

    # Bodo passes global length. If the dead side is distributed but the other side is
    # not, we need to distribute the dead side's global len to get correct output.
    n_pes = "bodo.libs.distributed_api.get_size()"
    rank = "bodo.libs.distributed_api.get_rank()"
    dead_in_len = "left_len" if left_is_dead else "right_len"
    live_parallel = right_parallel if left_is_dead else left_parallel
    dead_parallel = left_parallel if left_is_dead else right_parallel
    dead_is_dist_live_rep = not live_parallel and dead_parallel

    repeats = (
        f"bodo.libs.distributed_api.get_node_portion({dead_in_len}, {n_pes}, {rank})"
        if dead_is_dist_live_rep
        else dead_in_len
    )

    func_text = f"def f({data_args}, left_len, right_len):\n"
    func_text += f"  repeats = {repeats}\n"
    func_text += f"  out_data = ({table_args},)\n"
    func_text += f"  py_table = logical_table_to_table(out_data, (), col_inds, {n_in_data_cols}, out_table_type, used_cols)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars["f"]

    glbs = {
        "out_table_type": out_table_type,
        "sum_op": np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value),
        "bodo": bodo,
        "used_cols": bodo.utils.typing.MetaType(tuple(join_node.out_used_cols)),
        "col_inds": bodo.utils.typing.MetaType(tuple(col_inds)),
        "logical_table_to_table": bodo.hiframes.table.logical_table_to_table,
        "get_table_data": bodo.hiframes.table.get_table_data,
    }

    arg_vars = [v for v in in_vars if v is not None] + [
        join_node.left_len_var,
        join_node.right_len_var,
    ]
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

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
    nodes = f_block.body[:-3]
    nodes[-1].target = join_node.out_data_vars[0]
    return nodes


def join_distributed_run(
    join_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """
    Replace the join IR node with the distributed implementations. This
    is called in distributed_pass and removes the Join from the IR.
    """
    # Add debug info about column pruning for left, right, and output.
    if bodo.user_logging.get_verbose_level() >= 2:
        join_source = join_node.loc.strformat()

        left_join_cols = [
            join_node.left_col_names[i]
            for i in sorted(
                set(range(len(join_node.left_col_names))) - join_node.left_dead_var_inds
            )
        ]
        pruning_msg = "Finished column elimination on join's left input:\n%s\nLeft input columns: %s\n"
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            join_source,
            left_join_cols,
        )

        right_join_cols = [
            join_node.right_col_names[i]
            for i in sorted(
                set(range(len(join_node.right_col_names)))
                - join_node.right_dead_var_inds
            )
        ]
        pruning_msg = "Finished column elimination on join's right input:\n%s\nRight input columns: %s\n"
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            join_source,
            right_join_cols,
        )

        out_join_cols = [
            join_node.out_col_names[i]
            for i in sorted(join_node.get_out_table_used_cols())
        ]
        pruning_msg = "Finished column pruning on join node:\n%s\nOutput columns: %s\n"
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            join_source,
            out_join_cols,
        )

    left_parallel, right_parallel = False, False
    if array_dists is not None:
        left_parallel, right_parallel = _get_table_parallel_flags(
            join_node, array_dists
        )

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    n_keys = len(join_node.left_keys)

    # Logical location for every cpp_table into the output
    # Python table. Any values that are left as -1 refer to
    # dead columns.
    out_physical_to_logical_list = []
    # Get the output table type.
    if join_node.has_live_out_table_var:
        out_table_type = typemap[join_node.get_out_table_var().name]
    else:
        out_table_type = types.none
    if join_node.has_live_out_index_var:
        index_col_type = typemap[join_node.get_out_index_var().name]
    else:
        index_col_type = types.none

    # handle cross join corner case where only output length is used
    if _is_cross_join_len(join_node):
        return _gen_cross_join_len(
            join_node,
            out_table_type,
            typemap,
            calltypes,
            typingctx,
            targetctx,
            left_parallel,
            right_parallel,
        )
    # handle cross join corner case when left input is dead
    elif join_node.how == "cross" and all(
        i in join_node.left_dead_var_inds for i in range(len(join_node.left_col_names))
    ):
        return _gen_cross_join_repeat(
            join_node,
            out_table_type,
            typemap,
            calltypes,
            typingctx,
            targetctx,
            left_parallel,
            right_parallel,
            True,
        )
    # handle cross join corner case when right input is dead
    elif join_node.how == "cross" and all(
        i in join_node.right_dead_var_inds
        for i in range(len(join_node.right_col_names))
    ):
        return _gen_cross_join_repeat(
            join_node,
            out_table_type,
            typemap,
            calltypes,
            typingctx,
            targetctx,
            left_parallel,
            right_parallel,
            False,
        )

    # Extra column refer: When doing a merge on column and index, the key
    # is put also in output, so we need one additional column in that case.
    if join_node.extra_data_col_num != -1:
        out_physical_to_logical_list.append(join_node.extra_data_col_num)

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

    # Determine the key numbers that are live.
    # Note: This is not the same as the column number of the
    # key and instead is i for join_node.left_keys[i]
    left_used_key_nums = set()
    right_used_key_nums = set()
    # Determine the key input column numbers that are live.
    # These are the actual column numbers for the keys
    left_used_key_col_nums = set()
    right_used_key_col_nums = set()

    # Generate a map for the general_merge_cond. Here we
    # define a column by two values. The logical index is its
    # column number in the Pandas DataFrame. In contrast the
    # physical index is the actual location in the C++ table.
    # For example if our DataFrame had columns ["A", "B"], but
    # the C++ layout was ["B", "A"], then the columns would be
    # "A": (logical 0, physical 1), "B": (logical 1, physical 0).
    #
    # We need to track the mapping for each of the inputs because
    # the generated cfuncs for complex joins are written using the
    # column name and needs to be converted to the column number in
    # C++ for the generated code.
    left_logical_physical_map = {}
    right_logical_physical_map = {}
    left_physical_to_logical_list = []
    right_physical_to_logical_list = []
    left_physical_index = 0
    right_physical_index = 0
    # Extract the left column variables for the keys and determine
    # the physical index for each live column.
    left_key_vars = []
    for key_num, c in enumerate(join_node.left_keys):
        in_col_num = join_node.left_var_map[c]
        # If the inputs are array we need to collect the used vars
        if not join_node.is_left_table:
            left_key_vars.append(join_node.left_vars[in_col_num])
        is_live = 1
        out_col_num = join_node.left_to_output_map[in_col_num]
        if c == INDEX_SENTINEL:
            # If we are joining on the index, we may need to return
            # the index to the output. If so, the output can be either
            # from the keys or the data columns, which we track via
            # the index_source and index_col_num. If the index column
            # exists but is not used in the output it is a dead key
            # by definition.
            if (
                join_node.has_live_out_index_var
                and join_node.index_source == "left"
                and join_node.index_col_num == in_col_num
            ):
                out_physical_to_logical_list.append(out_col_num)
                left_used_key_nums.add(key_num)
                left_used_key_col_nums.add(in_col_num)
            else:
                is_live = 0
        else:
            if out_col_num not in join_node.out_used_cols:
                is_live = 0
            else:
                if in_col_num in left_used_key_col_nums:
                    # If a key is repeated it is only in the output
                    # once.
                    is_live = 0
                else:
                    left_used_key_nums.add(key_num)
                    left_used_key_col_nums.add(in_col_num)
                    out_physical_to_logical_list.append(out_col_num)
        left_physical_to_logical_list.append(in_col_num)
        left_logical_physical_map[in_col_num] = left_physical_index
        left_physical_index += 1
        left_key_in_output.append(is_live)
    left_key_vars = tuple(left_key_vars)

    # Extract the left column variables for the non-keys and determine
    # the physical index for each live column.
    left_other_col_vars = []
    for i in range(len(join_node.left_col_names)):
        if i not in join_node.left_dead_var_inds and i not in join_node.left_key_set:
            # If the inputs are array we need to collect the used vars
            if not join_node.is_left_table:
                v = join_node.left_vars[i]
                left_other_col_vars.append(v)
            is_live_output = 1
            is_live_input = 1
            out_col_num = join_node.left_to_output_map[i]
            if i in join_node.left_cond_cols:
                # Join conditions need to be tracked like keys
                if out_col_num not in join_node.out_used_cols:
                    is_live_output = 0
                left_key_in_output.append(is_live_output)
            elif i in join_node.left_dead_var_inds:
                is_live_output = 0
                is_live_input = 0
            if is_live_output:
                out_physical_to_logical_list.append(out_col_num)
            if is_live_input:
                left_physical_to_logical_list.append(i)
                left_logical_physical_map[i] = left_physical_index
                left_physical_index += 1

    # Append the index data column if it exists
    if (
        join_node.has_live_out_index_var
        and join_node.index_source == "left"
        and join_node.index_col_num not in join_node.left_key_set
    ):
        # If we are joining on the index, we may need to return
        # the index to the output. If so, the output can be either
        # from the keys or the data columns. Here we determine that
        # the index is a data column because its not a key.
        if not join_node.is_left_table:
            left_other_col_vars.append(join_node.left_vars[join_node.index_col_num])
        out_col_num = join_node.left_to_output_map[join_node.index_col_num]
        out_physical_to_logical_list.append(out_col_num)
        left_physical_to_logical_list.append(join_node.index_col_num)

    left_other_col_vars = tuple(left_other_col_vars)

    if join_node.is_left_table:
        # If we have table format input and the
        # table is live, we append a single variable for
        # the table. In the none table format case
        # we added each array that was live in the loop.
        left_other_col_vars = tuple(join_node.get_live_left_vars())

    # Extract the right column variables for the keys and determine
    # the physical index for each live column.
    right_key_vars = []
    for key_num, c in enumerate(join_node.right_keys):
        in_col_num = join_node.right_var_map[c]
        # If the inputs are array we need to collect the used vars
        if not join_node.is_right_table:
            right_key_vars.append(join_node.right_vars[in_col_num])
        if not join_node.vect_same_key[key_num] and not join_node.is_join:
            is_live = 1
            if in_col_num not in join_node.right_to_output_map:
                # This path is taken if the key is a common key but
                # not capture by vect_same_key. See test_merge_repeat_key.
                is_live = 0
            else:
                out_col_num = join_node.right_to_output_map[in_col_num]
                if c == INDEX_SENTINEL:
                    # If we are joining on the index, we may need to return
                    # the index to the output. If so, the output can be either
                    # from the keys or the data columns, which we track via
                    # the index_source and index_col_num. If the index column
                    # exists but is not used in the output it is a dead key
                    # by definition.
                    if (
                        join_node.has_live_out_index_var
                        and join_node.index_source == "right"
                        and join_node.index_col_num == in_col_num
                    ):
                        out_physical_to_logical_list.append(out_col_num)
                        right_used_key_nums.add(key_num)
                        right_used_key_col_nums.add(in_col_num)
                    else:
                        is_live = 0
                else:
                    if out_col_num not in join_node.out_used_cols:
                        is_live = 0
                    else:
                        if in_col_num in right_used_key_col_nums:
                            # If a key is repeated it is only in the output
                            # once.
                            is_live = 0
                        else:
                            right_used_key_nums.add(key_num)
                            right_used_key_col_nums.add(in_col_num)
                            out_physical_to_logical_list.append(out_col_num)
            right_key_in_output.append(is_live)
        right_physical_to_logical_list.append(in_col_num)
        right_logical_physical_map[in_col_num] = right_physical_index
        right_physical_index += 1
    right_key_vars = tuple(right_key_vars)

    # Extract the right column variables for the non-keys and determine
    # the physical index for each live column.
    right_other_col_vars = []
    for i in range(len(join_node.right_col_names)):
        if i not in join_node.right_dead_var_inds and i not in join_node.right_key_set:
            # If the inputs are array we need to collect the used vars
            if not join_node.is_right_table:
                right_other_col_vars.append(join_node.right_vars[i])
            is_live_output = 1
            is_live_input = 1
            out_col_num = join_node.right_to_output_map[i]
            if i in join_node.right_cond_cols:
                # Join conditions need to be tracked like keys
                if out_col_num not in join_node.out_used_cols:
                    is_live_output = 0
                right_key_in_output.append(is_live_output)
            elif i in join_node.right_dead_var_inds:
                is_live_output = 0
                is_live_input = 0
            if is_live_output:
                out_physical_to_logical_list.append(out_col_num)
            if is_live_input:
                right_physical_to_logical_list.append(i)
                right_logical_physical_map[i] = right_physical_index
                right_physical_index += 1

    # Append the index data column if it exists
    if (
        join_node.has_live_out_index_var
        and join_node.index_source == "right"
        and join_node.index_col_num not in join_node.right_key_set
    ):
        # If we are joining on the index, we may need to return
        # the index to the output. If so, the output can be either
        # from the keys or the data columns. Here we determine that
        # the index is a data column because its not a key.
        if not join_node.is_right_table:
            right_other_col_vars.append(join_node.right_vars[join_node.index_col_num])
        out_col_num = join_node.right_to_output_map[join_node.index_col_num]
        out_physical_to_logical_list.append(out_col_num)
        right_physical_to_logical_list.append(join_node.index_col_num)

    right_other_col_vars = tuple(right_other_col_vars)

    if join_node.is_right_table:
        # If we have table format input and the
        # table is live, we append a single variable for
        # the table. In the none table format case
        # we added each array that was live in the loop.
        right_other_col_vars = tuple(join_node.get_live_right_vars())

    if join_node.indicator_col_num != -1:
        # There is an indicator column in the output.
        out_physical_to_logical_list.append(join_node.indicator_col_num)

    # get column types
    arg_vars = (
        left_key_vars + right_key_vars + left_other_col_vars + right_other_col_vars
    )
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

    # arg names of non-key columns
    left_other_names = tuple("t1_c" + str(i) for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i) for i in range(len(right_other_col_vars)))
    if join_node.is_left_table:
        left_key_names = ()
    else:
        left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    if join_node.is_right_table:
        right_key_names = ()
    else:
        right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))
    glbs = {}
    loc = join_node.loc

    func_text = "def f({}):\n".format(
        ",".join(
            left_key_names + right_key_names + left_other_names + right_other_names
        ),
    )

    # If the inputs are tables we either need to find the keys from
    # the table or from the index var.
    if join_node.is_left_table:
        left_key_types = []
        left_other_types = []
        if join_node.has_live_left_table_var:
            table_type = typemap[join_node.left_vars[0].name]
        else:
            table_type = types.none
        for ind in left_physical_to_logical_list:
            if ind < join_node.n_left_table_cols:
                # This should never be reached if the table is dead
                assert join_node.has_live_left_table_var, (
                    "No logical columns should refer to a dead table"
                )
                typ = table_type.arr_types[ind]
            else:
                # The source is an index array.
                typ = typemap[join_node.left_vars[-1].name]
            if ind in join_node.left_key_set:
                left_key_types.append(typ)
            else:
                left_other_types.append(typ)
        left_key_types = tuple(left_key_types)
        left_other_types = tuple(left_other_types)
    else:
        left_key_types = tuple(typemap[v.name] for v in left_key_vars)
        left_other_types = tuple([typemap[c.name] for c in left_other_col_vars])
    if join_node.is_right_table:
        right_key_types = []
        right_other_types = []
        if join_node.has_live_right_table_var:
            table_type = typemap[join_node.right_vars[0].name]
        else:
            table_type = types.none
        for ind in right_physical_to_logical_list:
            if ind < join_node.n_right_table_cols:
                # This should never be reached if the table is dead
                assert join_node.has_live_right_table_var, (
                    "No logical columns should refer to a dead table"
                )
                typ = table_type.arr_types[ind]
            else:
                # The source is an index array.
                typ = typemap[join_node.right_vars[-1].name]
            if ind in join_node.right_key_set:
                right_key_types.append(typ)
            else:
                right_other_types.append(typ)
        right_key_types = tuple(right_key_types)
        right_other_types = tuple(right_other_types)
    else:
        right_key_types = tuple(typemap[v.name] for v in right_key_vars)
        right_other_types = tuple([typemap[c.name] for c in right_other_col_vars])

    # add common key type to globals to use below for type conversion
    matched_key_types = []
    for i in range(n_keys):
        matched_key_type = _match_join_key_types(
            left_key_types[i], right_key_types[i], loc
        )
        glbs[f"key_type_{i}"] = matched_key_type
        matched_key_types.append(matched_key_type)

    # Extract the left data
    if join_node.is_left_table:
        cast_map = determine_table_cast_map(
            matched_key_types,
            left_key_types,
            None,
            # Map the key number to its column number
            {
                i: join_node.left_var_map[key]
                for i, key in enumerate(join_node.left_keys)
            },
            True,
        )
        # We need to generate a cast of the table or index.
        if cast_map:
            table_changed = False
            index_changed = False
            index_typ = None
            if join_node.has_live_left_table_var:
                table_arrs = list(typemap[join_node.left_vars[0].name].arr_types)
            else:
                table_arrs = None
            for col_num, typ in cast_map.items():
                if col_num < join_node.n_left_table_cols:
                    assert join_node.has_live_left_table_var, (
                        "Casting columns for a dead table should not occur"
                    )
                    table_arrs[col_num] = typ
                    table_changed = True
                else:
                    index_typ = typ
                    index_changed = True
            if table_changed:
                # Generate a table astype call
                func_text += f"    {left_other_names[0]} = bodo.utils.table_utils.table_astype({left_other_names[0]}, left_cast_table_type, False, _bodo_nan_to_str=False, used_cols=left_used_cols)\n"
                glbs["left_cast_table_type"] = TableType(tuple(table_arrs))
                glbs["left_used_cols"] = MetaType(
                    tuple(
                        sorted(
                            set(range(join_node.n_left_table_cols))
                            - join_node.left_dead_var_inds
                        )
                    )
                )
            if index_changed:
                # Generate a cast for the index
                func_text += f"    {left_other_names[1]} = bodo.utils.utils.astype({left_other_names[1]}, left_cast_index_type)\n"
                glbs["left_cast_index_type"] = index_typ
    else:
        # Cast the keys to a common dtype for comparison
        # if the types differ.
        func_text += "    t1_keys = ({}{})\n".format(
            ", ".join(
                f"bodo.utils.utils.astype({left_key_names[i]}, key_type_{i})"
                if left_key_types[i] != matched_key_types[i]
                else f"{left_key_names[i]}"
                for i in range(n_keys)
            ),
            "," if n_keys != 0 else "",
        )
        func_text += "    data_left = ({}{})\n".format(
            ",".join(left_other_names), "," if len(left_other_names) != 0 else ""
        )

    # Extract the right data
    if join_node.is_right_table:
        # Map the key number to its column number
        cast_map = determine_table_cast_map(
            matched_key_types,
            right_key_types,
            None,
            {
                i: join_node.right_var_map[key]
                for i, key in enumerate(join_node.right_keys)
            },
            True,
        )
        # We need to generate a cast of the table or index.
        if cast_map:
            table_changed = False
            index_changed = False
            index_typ = None
            if join_node.has_live_right_table_var:
                table_arrs = list(typemap[join_node.right_vars[0].name].arr_types)
            else:
                table_arrs = None
            for col_num, typ in cast_map.items():
                if col_num < join_node.n_right_table_cols:
                    assert join_node.has_live_right_table_var, (
                        "Casting columns for a dead table should not occur"
                    )
                    table_arrs[col_num] = typ
                    table_changed = True
                else:
                    index_typ = typ
                    index_changed = True
            if table_changed:
                # Generate a table astype call
                func_text += f"    {right_other_names[0]} = bodo.utils.table_utils.table_astype({right_other_names[0]}, right_cast_table_type, False, _bodo_nan_to_str=False, used_cols=right_used_cols)\n"
                glbs["right_cast_table_type"] = TableType(tuple(table_arrs))
                glbs["right_used_cols"] = MetaType(
                    tuple(
                        sorted(
                            set(range(join_node.n_right_table_cols))
                            - join_node.right_dead_var_inds
                        )
                    )
                )
            if index_changed:
                # Generate a cast for the index
                func_text += f"    {right_other_names[1]} = bodo.utils.utils.astype({right_other_names[1]}, left_cast_index_type)\n"
                glbs["right_cast_index_type"] = index_typ
    else:
        # Cast the keys to a common dtype for comparison
        # if the types differ.
        func_text += "    t2_keys = ({}{})\n".format(
            ", ".join(
                f"bodo.utils.utils.astype({right_key_names[i]}, key_type_{i})"
                if right_key_types[i] != matched_key_types[i]
                else f"{right_key_names[i]}"
                for i in range(n_keys)
            ),
            "," if n_keys != 0 else "",
        )
        func_text += "    data_right = ({}{})\n".format(
            ",".join(right_other_names), "," if len(right_other_names) != 0 else ""
        )

    # Generate a general join condition function if it exists
    # and determine the data columns it needs.
    general_cond_cfunc, left_col_nums, right_col_nums = gen_general_cond_cfunc(
        typemap,
        left_logical_physical_map,
        right_logical_physical_map,
        join_node.gen_cond_expr,
        join_node.left_var_map,
        join_node.left_vars,
        join_node.left_key_set,
        typemap[join_node.left_vars[0].name]
        if join_node.has_live_left_table_var
        else None,
        join_node.right_var_map,
        join_node.right_vars,
        join_node.right_key_set,
        typemap[join_node.right_vars[0].name]
        if join_node.has_live_right_table_var
        else None,
        # cross join passes batches of input to condition function
        compute_in_batch=not join_node.left_keys,
    )
    # Determine if we have a point in interval join
    join_node.point_interval_join_info: tuple[bool, str, str, str] | None = guard(
        _get_interval_join_info,
        join_node,
        left_col_nums,
        right_col_nums,
        left_other_types,
        right_other_types,
        left_physical_to_logical_list,
        right_physical_to_logical_list,
    )
    if join_node.point_interval_join_info is not None:
        # If we have the point in interval join we can discard the
        # general join condition because the entire implementation
        # will be in C++. While we have compiled the general join condition
        # this will avoid storing it in join_gen_cond_cfunc.
        join_node.gen_cond_expr = ""
        general_cond_cfunc = None

    # TODO: Update asof to use table format.
    if join_node.how == "asof":
        if left_parallel or right_parallel:
            assert left_parallel and right_parallel, (
                "pd.merge_asof requires both left and right to be replicated or distributed"
            )
            # only the right key needs to be aligned
            func_text += "    t2_keys, data_right = parallel_asof_comm(t1_keys, t2_keys, data_right)\n"
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_merge_asof(t1_keys, t2_keys, data_left, data_right)\n"
        )
    else:
        func_text += _gen_join_cpp_call(
            join_node,
            left_key_types,
            right_key_types,
            matched_key_types,
            left_other_names,
            right_other_names,
            left_other_types,
            right_other_types,
            left_key_in_output,
            right_key_in_output,
            left_parallel,
            right_parallel,
            glbs,
            out_physical_to_logical_list,
            out_table_type,
            index_col_type,
            join_node.get_out_table_used_cols(),
            left_used_key_nums,
            right_used_key_nums,
            general_cond_cfunc,
            left_col_nums,
            right_col_nums,
            left_physical_to_logical_list,
            right_physical_to_logical_list,
            left_logical_physical_map,
            right_logical_physical_map,
        )
    # TODO: Update asof
    # Set the output variables.
    if join_node.how == "asof":
        for i in range(len(left_other_names)):
            func_text += f"    left_{i} = out_data_left[{i}]\n"
        for i in range(len(right_other_names)):
            func_text += f"    right_{i} = out_data_right[{i}]\n"
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
            "parallel_asof_comm": parallel_asof_comm,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "nested_loop_join_table": nested_loop_join_table,
            "interval_join_table": interval_join_table,
            "hash_join_table": hash_join_table,
            "delete_table": delete_table,
            "add_join_gen_cond_cfunc_sym": add_join_gen_cond_cfunc_sym,
            "get_join_cond_addr": get_join_cond_addr,
            # key_in_output is defined to contain left_table then right_table
            # to match the iteration order in C++
            "key_in_output": np.array(
                left_key_in_output + right_key_in_output, dtype=np.bool_
            ),
            "py_data_to_cpp_table": py_data_to_cpp_table,
            "cpp_table_to_py_data": cpp_table_to_py_data,
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
    if join_node.has_live_out_index_var:
        nodes[-1].target = join_node.out_data_vars[1]
    if join_node.has_live_out_table_var:
        nodes[-2].target = join_node.out_data_vars[0]
    assert join_node.has_live_out_index_var or join_node.has_live_out_table_var, (
        "At most one of table and index should be dead if the Join IR node is live"
    )
    if not join_node.has_live_out_index_var:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not join_node.has_live_out_table_var:
        nodes.pop(-2)
    return nodes


distributed_pass.distributed_run_extensions[Join] = join_distributed_run


def gen_general_cond_cfunc(
    typemap,
    left_logical_physical_map,
    right_logical_physical_map,
    expr,
    left_var_map,
    left_vars,
    left_key_set,
    left_table_type,
    right_var_map,
    right_vars,
    right_key_set,
    right_table_type,
    compute_in_batch,
):
    """Generate cfunc for general join condition and return its address.
    Return 0 (NULL) if there is no general join condition to evaluate.
    The cfunc takes data pointers of table columns and row indices to access as input
    and returns True or False.
    E.g. left_table=[A_data_ptr, B_data_ptr], right_table=[A_data_ptr, C_data_ptr],
    left_ind=3, right_ind=7

    Args:
        typemap (Optional[Dict[str, types.Type]]): The type map for determining array
            types for condition columns (can be None if table types provided)
        left_logical_physical_map (Dict[int, int]): Mapping from the logical
            column number of the left table to the physical number of the C++
            table. This is done because of dead columns.
        right_logical_physical_map (Dict[int, int]): Mapping from the logical
            column number of the right table to the physical number of the C++
            table. This is done because of dead columns.
        expr (str): general condition expression
        left_var_map (Dict[str, int]): map column name to logical column number
        left_vars (Optional[List[ir.Var]]): column variables for left table, needed if
            left_table_type not provided to determine column array types
        left_key_set (Set[int]): logical column indices for equality keys if any
        left_table_type (types.Type): left table type if in table format
        right_var_map (Dict[str, int]): map column name to logical column number
        right_vars (Optional[List[ir.Var]]): column variables for left table, needed if
            right_table_type not provided to determine column array types
        right_key_set (Set[int]): logical column indices for equality keys if any
        right_table_type (types.Type): right table type if in table format
        compute_in_batch (bool): process batches of input instead of a single row.
            Cross join implementation passes input batches currently.

    Returns:
        Tuple[cfunc, List[int], List[int]]: Triple containing the generated
            cfunc and the columns used by each table.
    """
    if not expr:
        return None, [], []

    label_suffix = next_label()

    table_getitem_funcs = {
        "bodo": bodo,
        "numba": numba,
        "is_null_pointer": is_null_pointer,
        "set_bit_to": bodo.utils.cg_helpers.set_bit_to,
    }
    na_check_name = "NOT_NA"

    ext_args = "left_ind, right_ind"
    if compute_in_batch:
        ext_args = "match_arr, left_block_start, left_block_end, right_block_start, right_block_end"

    func_text = f"def bodo_join_gen_cond{label_suffix}(left_table, right_table, left_data1, right_data1, left_null_bitmap, right_null_bitmap, {ext_args}):\n"
    func_text += "  if is_null_pointer(left_table):\n"
    func_text += f"    return {'' if compute_in_batch else 'False'}\n"

    if compute_in_batch:
        func_text += "  i = 0\n"
        func_text += "  for right_ind in range(right_block_start, right_block_end):\n"
        func_text += "    for left_ind in range(left_block_start, left_block_end):\n"

    indent = "      " if compute_in_batch else "  "
    expr, func_text, left_col_nums = _replace_column_accesses(
        expr,
        left_logical_physical_map,
        left_var_map,
        typemap,
        left_vars,
        table_getitem_funcs,
        func_text,
        "left",
        left_key_set,
        na_check_name,
        left_table_type,
        indent,
    )
    expr, func_text, right_col_nums = _replace_column_accesses(
        expr,
        right_logical_physical_map,
        right_var_map,
        typemap,
        right_vars,
        table_getitem_funcs,
        func_text,
        "right",
        right_key_set,
        na_check_name,
        right_table_type,
        indent,
    )
    # use short-circuit boolean operators to avoid invalid access of NA locations
    # see https://bodo.atlassian.net/browse/BE-4146
    expr = expr.replace(" & ", " and ").replace(" | ", " or ")

    if compute_in_batch:
        func_text += f"      set_bit_to(match_arr, i, {expr})\n"
        func_text += "      i += 1\n"
    else:
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

    if compute_in_batch:
        c_sig = types.void(
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.int64,
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
    key_set,
    na_check_name,
    table_type,
    indent,
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
        if table_type:
            array_typ = table_type.arr_types[c_ind]
        else:
            array_typ = typemap[col_vars[c_ind].name]

        # Not creating intermediate variables for val_varname to avoid invalid access of
        # NA locations (null checks should run before getitems)
        # see https://bodo.atlassian.net/browse/BE-4146
        if is_str_arr_type(array_typ) or array_typ == bodo.binary_array_type:
            # If we have unicode we pass the table variable which is an array info
            val_varname = f"{getitem_fname}({table_name}_table, {table_name}_ind)\n"
        else:
            # If we have a numeric type we just pass the data pointers
            val_varname = f"{getitem_fname}({table_name}_data1, {table_name}_ind)\n"

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
                isinstance(
                    array_typ,
                    (
                        bodo.libs.int_arr_ext.IntegerArrayType,
                        bodo.FloatingArrayType,
                        bodo.TimeArrayType,
                    ),
                )
                or array_typ
                in (
                    bodo.libs.bool_arr_ext.boolean_array_type,
                    bodo.binary_array_type,
                    bodo.datetime_date_array_type,
                )
                or is_str_arr_type(array_typ)
            ):
                func_text += f"{indent}{na_val_varname} = {na_check_fname}({table_name}_null_bitmap, {table_name}_ind)\n"
            else:
                func_text += f"{indent}{na_val_varname} = {na_check_fname}({table_name}_data1, {table_name}_ind)\n"

            table_getitem_funcs[na_check_fname] = (
                bodo.libs.array._gen_row_na_check_intrinsic(array_typ, physical_ind)
            )
            expr = expr.replace(na_cname, na_val_varname)

        # only append the column if it is not an (equality) key
        if c_ind not in key_set:
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
    # use saved distribution if all input vars are dead (cross join corner case)
    if not join_node.get_live_left_vars():
        assert join_node.how == "cross", "cross join expected if left data is dead"
        left_parallel = join_node.left_dist in par_dists

    right_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.get_live_right_vars()
    )
    # use saved distribution if all input vars are dead (cross join corner case)
    if not join_node.get_live_right_vars():
        assert join_node.how == "cross", "cross join expected if right data is dead"
        right_parallel = join_node.right_dist in par_dists

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


def _gen_join_cpp_call(
    join_node: Join,
    left_key_types,
    right_key_types,
    matched_key_types,
    left_other_names,
    right_other_names,
    left_other_types,
    right_other_types,
    left_key_in_output,
    right_key_in_output,
    left_parallel,
    right_parallel,
    glbs,
    out_physical_to_logical_list,
    out_table_type,
    index_col_type,
    out_table_used_cols,
    left_used_key_nums,
    right_used_key_nums,
    general_cond_cfunc,
    left_col_nums,
    right_col_nums,
    left_physical_to_logical_list,
    right_physical_to_logical_list,
    left_logical_physical_map,
    right_logical_physical_map,
):
    """
    Generate the code need to compute a hash join in C++
    """

    # In some case the column in output has a type different from the one in input.
    # TODO: Unify those type changes between all cases.
    def needs_typechange(in_type, need_nullable, is_same_key):
        return (
            isinstance(in_type, types.Array)
            and (
                not is_dtype_nullable(in_type.dtype)
                or isinstance(in_type.dtype, types.Float)
            )
            and need_nullable
            and not is_same_key
        )

    # The use_nullable_arr_type is computed in the python code and is sent to C++.
    # This is a general approach for this kind of combinatorial problem: compute in python
    # preferably to C++. Compute in dataframe_pass.py preferably to the IR node.
    #
    # The use_nullable_arr_type is for the need to change the type in some cases.
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

    vect_same_key = join_node.vect_same_key

    # List of columns in the output that need to be cast.
    use_nullable_arr_type = []
    for i in range(len(left_key_types)):
        if left_key_in_output[i]:
            use_nullable_arr_type.append(
                needs_typechange(
                    matched_key_types[i], join_node.is_right, vect_same_key[i]
                )
            )

    # Offset for left and right table inside the key in output
    # lists. Every left key is always included so we only count
    # the left table for data columns.
    left_key_in_output_idx = len(left_key_types)
    right_key_in_output_idx = 0

    left_data_logical_list = left_physical_to_logical_list[len(left_key_types) :]

    for i, ind in enumerate(left_data_logical_list):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if ind in left_cond_col_nums_set:
            load_arr = left_key_in_output[left_key_in_output_idx]
            left_key_in_output_idx += 1
        if load_arr:
            use_nullable_arr_type.append(
                needs_typechange(left_other_types[i], join_node.is_right, False)
            )

    for i in range(len(right_key_types)):
        if not vect_same_key[i] and not join_node.is_join:
            if right_key_in_output[right_key_in_output_idx]:
                use_nullable_arr_type.append(
                    needs_typechange(matched_key_types[i], join_node.is_left, False)
                )
            right_key_in_output_idx += 1

    right_data_logical_list = right_physical_to_logical_list[len(right_key_types) :]

    for i, ind in enumerate(right_data_logical_list):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if ind in right_cond_col_nums_set:
            load_arr = right_key_in_output[right_key_in_output_idx]
            right_key_in_output_idx += 1
        if load_arr:
            use_nullable_arr_type.append(
                needs_typechange(right_other_types[i], join_node.is_left, False)
            )

    n_keys = len(left_key_types)
    func_text = "    # beginning of _gen_join_cpp_call\n"
    if join_node.is_left_table:
        if join_node.has_live_left_table_var:
            index_left_others = left_other_names[1:]
            table_var = left_other_names[0]
        else:
            index_left_others = left_other_names
            table_var = None
        index_str = (
            "()" if len(index_left_others) == 0 else f"({index_left_others[0]},)"
        )
        func_text += f"    table_left = py_data_to_cpp_table({table_var}, {index_str}, left_in_cols, {join_node.n_left_table_cols})\n"
        glbs["left_in_cols"] = MetaType(tuple(left_physical_to_logical_list))
    else:
        eList_l = []
        for i in range(n_keys):
            eList_l.append(f"t1_keys[{i}]")
        for i in range(len(left_other_names)):
            eList_l.append(f"data_left[{i}]")
        func_text += "    info_list_total_l = [{}]\n".format(
            ",".join(f"array_to_info({a})" for a in eList_l)
        )
        func_text += "    table_left = arr_info_list_to_table(info_list_total_l)\n"
    if join_node.is_right_table:
        if join_node.has_live_right_table_var:
            index_right_others = right_other_names[1:]
            table_var = right_other_names[0]
        else:
            index_right_others = right_other_names
            table_var = None
        index_str = (
            "()" if len(index_right_others) == 0 else f"({index_right_others[0]},)"
        )
        func_text += f"    table_right = py_data_to_cpp_table({table_var}, {index_str}, right_in_cols, {join_node.n_right_table_cols})\n"
        glbs["right_in_cols"] = MetaType(tuple(right_physical_to_logical_list))
    else:
        eList_r = []
        for i in range(n_keys):
            eList_r.append(f"t2_keys[{i}]")
        for i in range(len(right_other_names)):
            eList_r.append(f"data_right[{i}]")
        func_text += "    info_list_total_r = [{}]\n".format(
            ",".join(f"array_to_info({a})" for a in eList_r)
        )
        func_text += "    table_right = arr_info_list_to_table(info_list_total_r)\n"
    # Add globals that will be used in the function call.
    glbs["vect_same_key"] = np.array(vect_same_key, dtype=np.int64)
    glbs["use_nullable_arr_type"] = np.array(use_nullable_arr_type, dtype=np.int64)
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
    func_text += "    total_rows_np = np.array([0], dtype=np.int64)\n"

    if join_node.point_interval_join_info is not None:
        left_col_types = [left_other_types[k] for k in left_col_nums]
        right_col_types = [right_other_types[k] for k in right_col_nums]
        check_cross_join_coltypes(left_col_types, right_col_types)

        # Add log msg indicating interval join was detected and used
        if bodo.user_logging.get_verbose_level() >= 1:
            join_source = join_node.loc.strformat()
            msg = "Using optimized interval range join implementation:\n%s"
            bodo.user_logging.log_message(
                "Join Optimization",
                msg,
                join_source,
            )
            # TODO: Add more information about the type of join and relevant columns in logging level 2

        if join_node.point_interval_join_info[
            0
        ]:  # i.e. point is on the left side and interval is on the right side
            # join_node.point_interval_join_info[2,3] are the column names, right_var_map gives us the logical location,
            # and then we map it to the physical location using right_logical_physical_map
            start_col_id_interval_side = right_logical_physical_map[
                join_node.right_var_map[join_node.point_interval_join_info[2]]
            ]
            end_col_id_interval_side = right_logical_physical_map[
                join_node.right_var_map[join_node.point_interval_join_info[3]]
            ]
            point_col_id_point_side = left_logical_physical_map[
                join_node.left_var_map[join_node.point_interval_join_info[1]]
            ]
        else:  # i.e. vice-versa
            start_col_id_interval_side = left_logical_physical_map[
                join_node.left_var_map[join_node.point_interval_join_info[2]]
            ]
            end_col_id_interval_side = left_logical_physical_map[
                join_node.left_var_map[join_node.point_interval_join_info[3]]
            ]
            point_col_id_point_side = right_logical_physical_map[
                join_node.right_var_map[join_node.point_interval_join_info[1]]
            ]

        func_text += (
            f"    out_table = interval_join_table("
            "table_left, "
            "table_right, "
            f"{left_parallel}, "
            f"{right_parallel}, "
            f"{join_node.is_left}, "
            f"{join_node.is_right}, "
            # Is point on the left side
            f"{join_node.point_interval_join_info[0]}, "
            # Does the point need to be strictly greater than the interval start
            f"{join_node.point_interval_join_info[4]}, "
            # Does the point need to be strictly less than the interval end
            f"{join_node.point_interval_join_info[5]}, "
            # Column ID of point column on point side in point in interval join.
            f"{point_col_id_point_side}, "
            # Column ID of start column on interval side in point in interval join.
            f"{start_col_id_interval_side}, "
            # Column ID of end column on interval side in point in interval join.
            f"{end_col_id_interval_side}, "
            f"key_in_output.ctypes, "
            f"use_nullable_arr_type.ctypes, "
            f"{join_node.rebalance_output_if_skewed}, "
            f"total_rows_np.ctypes)\n"
        )

    # Joins without equality condition should use nested loop implementation
    # For now, we are having all interval-overlap joins go through cross join
    # as well.
    elif join_node.how == "cross" or not join_node.left_keys:
        left_col_types = [left_other_types[k] for k in left_col_nums]
        right_col_types = [right_other_types[k] for k in right_col_nums]
        check_cross_join_coltypes(left_col_types, right_col_types)

        func_text += (
            f"    out_table = nested_loop_join_table("
            "table_left, "
            "table_right, "
            f"{left_parallel}, "
            f"{right_parallel}, "
            f"{join_node.is_left}, "
            f"{join_node.is_right}, "
            f"key_in_output.ctypes, "
            f"use_nullable_arr_type.ctypes, "
            f"{join_node.rebalance_output_if_skewed}, "
            f"cfunc_cond, "
            f"left_table_cond_columns.ctypes, "
            f"{len(left_col_nums)}, "
            f"right_table_cond_columns.ctypes, "
            f"{len(right_col_nums)}, "
            f"total_rows_np.ctypes)\n"
        )
    else:
        func_text += (
            "    out_table = hash_join_table("
            "table_left, "
            "table_right, "
            f"{left_parallel}, "
            f"{right_parallel}, "
            f"{n_keys}, "
            f"{len(left_data_logical_list)}, "
            f"{len(right_data_logical_list)}, "
            f"vect_same_key.ctypes, "
            f"key_in_output.ctypes, "
            f"use_nullable_arr_type.ctypes, "
            f"{join_node.is_left}, "
            f"{join_node.is_right}, "
            f"{join_node.is_join}, "
            f"{join_node.extra_data_col_num != -1}, "
            f"{join_node.indicator_col_num != -1}, "
            f"{join_node.is_na_equal}, "
            f"{join_node.rebalance_output_if_skewed}, "
            f"cfunc_cond, "
            f"left_table_cond_columns.ctypes, "
            f"{len(left_col_nums)}, "
            f"right_table_cond_columns.ctypes, "
            f"{len(right_col_nums)}, "
            f"total_rows_np.ctypes)\n"
        )
    # Note: Deleting table_left and table_right is done inside C++
    out_types = "(py_table_type, index_col_type)"
    func_text += f"    out_data = cpp_table_to_py_data(out_table, out_col_inds, {out_types}, total_rows_np[0], {join_node.n_out_table_cols})\n"
    if join_node.has_live_out_table_var:
        func_text += "    T = out_data[0]\n"
    else:
        func_text += "    T = None\n"
    if join_node.has_live_out_index_var:
        idx = 1 if join_node.has_live_out_table_var else 0
        func_text += f"    index_var = out_data[{idx}]\n"
    else:
        func_text += "    index_var = None\n"

    glbs["py_table_type"] = out_table_type
    glbs["index_col_type"] = index_col_type
    glbs["out_col_inds"] = MetaType(tuple(out_physical_to_logical_list))

    if bool(join_node.out_used_cols) or index_col_type != types.none:
        # Only delete the C++ table if it is not a nullptr (0 output columns returns nullptr)
        func_text += "    delete_table(out_table)\n"
    if out_table_type != types.none:
        # Create the left map from input key number to output col number
        left_key_num_to_out_col_num = {}
        for i, key in enumerate(join_node.left_keys):
            # If the key is not used then its dead
            if i in left_used_key_nums:
                input_col_num = join_node.left_var_map[key]
                left_key_num_to_out_col_num[i] = join_node.left_to_output_map[
                    input_col_num
                ]
        cast_map = determine_table_cast_map(
            matched_key_types,
            left_key_types,
            left_used_key_nums,
            left_key_num_to_out_col_num,
            False,
        )
        # Create the right map from input key number to output col number
        right_key_num_to_out_col_num = {}
        for i, key in enumerate(join_node.right_keys):
            # If the key is not used then its dead
            if i in right_used_key_nums:
                input_col_num = join_node.right_var_map[key]
                right_key_num_to_out_col_num[i] = join_node.right_to_output_map[
                    input_col_num
                ]
        cast_map.update(
            determine_table_cast_map(
                matched_key_types,
                right_key_types,
                right_used_key_nums,
                # Create an actual map from input key number to output
                right_key_num_to_out_col_num,
                False,
            )
        )
        table_changed = False
        index_changed = False
        if join_node.has_live_out_table_var:
            table_arrs = list(out_table_type.arr_types)
        else:
            table_arrs = None
        for col_num, typ in cast_map.items():
            if col_num < join_node.n_out_table_cols:
                assert join_node.has_live_out_table_var, (
                    "Casting columns for a dead table should not occur"
                )
                table_arrs[col_num] = typ
                table_changed = True
            else:
                index_typ = typ
                index_changed = True
        if table_changed:
            func_text += "    T = bodo.utils.table_utils.table_astype(T, cast_table_type, False, _bodo_nan_to_str=False, used_cols=used_cols)\n"
            # Determine the types that must be loaded.
            pre_cast_table_type = bodo.TableType(tuple(table_arrs))
            # Update the table types
            glbs["py_table_type"] = pre_cast_table_type
            glbs["cast_table_type"] = out_table_type
            glbs["used_cols"] = MetaType(tuple(out_table_used_cols))
        if index_changed:
            glbs["index_col_type"] = index_typ
            glbs["index_cast_type"] = index_col_type
            func_text += (
                "    index_var = bodo.utils.utils.astype(index_var, index_cast_type)\n"
            )
    func_text += "    out_table = T\n"
    func_text += "    out_index = index_var\n"
    return func_text


def determine_table_cast_map(
    matched_key_types: list[types.Type],
    key_types: list[types.Type],
    used_key_nums: set[int] | None,
    output_map: dict[int, int],
    convert_dict_col: bool,
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
        used_key_nums (Set[int]): Set of logical column indices in the table
            that are also live in the output.
        output_map (Dict[int, int]): Dictionary mapping the key number
            to the logical index of the column.
        convert_dict_col (bool): Convert dictionary outputs if column
            types don't match.

    Returns:
        Dict[int, types.Type]: Dictionary mapping the logical column number
        in the output table to the correct output type when loading the table.
    """
    cast_map: dict[int, types.Type] = {}

    # Check the keys for casts.
    n_keys = len(matched_key_types)
    for i in range(n_keys):
        # Determine if the column is live
        if used_key_nums is None or i in used_key_nums:
            # Astype is needed when the key had to be cast for the join
            # (e.g. left=int64 and right=float64 casts the left to float64)
            # and we need the key back to the original type in the output.
            if matched_key_types[i] != key_types[i] and (
                convert_dict_col or key_types[i] != bodo.dict_str_arr_type
            ):
                # This maps the key number to the actual column number
                # TODO [BE-3552]: Ensure the cast are compatible.
                idx = output_map[i]
                cast_map[idx] = matched_key_types[i]

    return cast_map


def _get_interval_join_info(
    join_node: Join,
    left_col_nums: list[int],
    right_col_nums: list[int],
    left_other_types,
    right_other_types,
    left_physical_to_logical_list: list[int],
    right_physical_to_logical_list: list[int],
) -> tuple[bool, str, str, str]:
    """Detect and return relevant info for point in interval join

    Args:
        join_node (Join): Join IR node
        left_col_nums (list(int))): left column numbers in join condition (physical)
        right_col_nums (list(int)): right column numbers in join condition (physical)
        left_other_types (list(types.Type)): data types of left columns in condition
        right_other_types (list(types.Type)): data types of right columns in condition
        left_physical_to_logical_list (list(int)): map physical left column numbers
            (C++ table) to logical (input Python data)
        right_physical_to_logical_list (list(int)): map physical right column numbers
            (C++ table) to logical (input Python data)

    Returns:
        - (bool, string, string, string):
            - bool: True if point is on the left side
            - string: Name of the point col on the point side
            - string: Name of the start col on the interval side
            - string: Name of the end col on the interval side
    """
    from bodo.hiframes.dataframe_impl import _is_col_access, _parse_query_expr
    from bodo.libs.pd_datetime_arr_ext import PandasDatetimeTZDtype

    # check for no equality condition
    require(not join_node.left_keys)
    # Point join has 1 key on one side, 2 keys other. Interval has 2 keys both sides.
    require(
        (len(left_col_nums) == 2 and len(right_col_nums) in (1, 2))
        or (len(right_col_nums) == 2 and len(left_col_nums) in (1, 2))
    )
    # inner join, or point join with outer on the point side
    require(
        join_node.how == "inner"
        or (join_node.how == "left" and len(left_col_nums) == 1)
        or (join_node.how == "right" and len(right_col_nums) == 1)
    )

    # make sure keys are numerical/date/time and the same type
    assert (len(left_other_types) >= left_col_nums[0]) and (
        len(left_other_types) > 0
    ), "_get_interval_join_info: invalid column types"
    key_type = left_other_types[left_col_nums[0]]
    dtype = key_type.dtype
    require(
        isinstance(
            dtype,
            (
                types.Integer,
                types.Float,
                PandasDatetimeTZDtype,
                bodo.TimeType,
                bodo.Decimal128Type,
            ),
        )
        or dtype
        in (bodo.datetime64ns, bodo.datetime_date_type, bodo.datetime_timedelta_type)
    )
    # TODO: We should eventually handle joins between nullable and non-nullable arrays
    require(all(left_other_types[k] == key_type for k in left_col_nums))
    require(all(right_other_types[k] == key_type for k in right_col_nums))

    # Parse expr and check for interval join comparison patterns
    resolver = {"left": 0, "right": 0, "NOT_NA": 0}
    # create fake environment for Expr to enable parsing
    env = pandas.core.computation.scope.ensure_scope(2, {}, {}, (resolver,))

    clean_cols = {}
    # We need to wrap columns like EXPR$0 with ` for pandas to parse
    # Previous step removed ` from cond_expr
    formatted_expr = join_node.gen_cond_expr
    for side, col_names in [
        ("left", join_node.left_col_names),
        ("right", join_node.right_col_names),
    ]:
        for col_name in col_names:
            clean_col = pandas.core.computation.parsing.clean_column_name(col_name)
            clean_cols[(side, clean_col)] = col_name

            col_outer = f"{side}.{col_name}"
            clean_outer = pandas.core.computation.parsing.clean_column_name(col_outer)
            clean_cols[("NOT_NA", clean_outer)] = col_outer

            formatted_expr = formatted_expr.replace(
                f"{side}.{col_name}", f"{side}.`{col_name}`"
            )
            formatted_expr = formatted_expr.replace(
                f"NOT_NA.{side}.`{col_name}`", f"NOT_NA.`{side}.{col_name}`"
            )

    parsed_expr_tree, _, _ = _parse_query_expr(
        formatted_expr, env, [], [], None, join_cleaned_cols=clean_cols
    )
    terms = parsed_expr_tree.terms

    # pattern match Bodo generated NA checks for some types like float and ignore them
    if _all_na_check(terms.lhs):
        terms = terms.rhs

    # condition should be AND of two comparisons
    require(isinstance(terms, pandas.core.computation.ops.BinOp) and terms.op == "&")
    cond1 = terms.lhs
    cond2 = terms.rhs
    require(
        isinstance(cond1, pandas.core.computation.ops.BinOp)
        and isinstance(cond2, pandas.core.computation.ops.BinOp)
    )
    require(_is_col_access(cond1.lhs) and _is_col_access(cond1.rhs))
    require(_is_col_access(cond2.lhs) and _is_col_access(cond2.rhs))

    # normalize to less-than comparisons
    _normalize_expr_cond(cond1)
    _normalize_expr_cond(cond2)

    # check for point interval join
    if len(left_col_nums) == 1:
        point_colname = join_node.left_col_names[
            left_physical_to_logical_list[left_col_nums[0]]
        ]
        (start_col, strict_start_comp), (end_col, strict_end_comp) = _check_point_cases(
            point_colname, cond1, cond2, True
        )
        return (
            True,
            point_colname,
            start_col,
            end_col,
            strict_start_comp,
            strict_end_comp,
        )
    elif len(right_col_nums) == 1:
        point_colname = join_node.right_col_names[
            right_physical_to_logical_list[right_col_nums[0]]
        ]
        (start_col, strict_start_comp), (end_col, strict_end_comp) = _check_point_cases(
            point_colname, cond1, cond2, False
        )
        return (
            False,
            point_colname,
            start_col,
            end_col,
            strict_start_comp,
            strict_end_comp,
        )


def _all_na_check(expr_node) -> bool:
    """Return True if expression node is just NA checks for columns
    (e.g. '((((NOT_NA.left).F)) & (((NOT_NA.left).G))) & (((NOT_NA.right).D))')

    Args:
        expr_node (pd.core.computation.ops.BinOp): input expr node

    Returns:
        bool: True if all NA check
    """
    if isinstance(expr_node, pandas.core.computation.ops.BinOp):
        return _all_na_check(expr_node.lhs) and _all_na_check(expr_node.rhs)

    if hasattr(expr_node, "name"):
        return expr_node.name.startswith("NOT_NA.")
    return str(expr_node).startswith("((NOT_NA.")


def _check_point_cases(
    point_colname: str,
    cond1: pandas.core.computation.ops.BinOp,
    cond2: pandas.core.computation.ops.BinOp,
    is_left: bool,
) -> tuple[tuple[str, bool], tuple[str, bool]]:
    """Check for point interval join in conditions

    Args:
        point_colname: column name of points
        cond1: first condition term in join
        cond2: second condition term in join
        is_left: is left table the points table

    Returns:
        - Name of the interval start column and whether `start (<,<=) point` is a strict inequality
        - Name of the interval end column and whether `point (<,<=) end` is a strict inequality
    """
    point_side = "left" if is_left else "right"
    point_colname = f"{point_side}.{point_colname}"
    # (A < P and P < B)
    start_comp_first: bool = (
        cond1.rhs.name == point_colname and cond2.lhs.name == point_colname
    )
    # (P < B and A < P)
    end_comp_first: bool = (
        cond1.lhs.name == point_colname and cond2.rhs.name == point_colname
    )
    require(start_comp_first or end_comp_first)
    other_side_prefix = "right." if is_left else "left."

    if start_comp_first:
        # In the (A < P and P < B) case, cond1 refers to (A < P) and cond2 refers to (P < B).
        # We want it to be the case that A (cond1.lhs) and B (cond2.rhs) come from the
        # non-point side and that A != B. If this is not the case, it's not a point
        # in interval join.
        require(
            cond1.lhs.name.startswith(other_side_prefix)
            and cond2.rhs.name.startswith(other_side_prefix)
            and cond1.lhs.name != cond2.rhs.name
        )
        return (
            (cond1.lhs.name.removeprefix(other_side_prefix), cond1.op == "<"),
            (cond2.rhs.name.removeprefix(other_side_prefix), cond2.op == "<"),
        )
    else:
        # Similar logic, but for the (P < B and A < P) case.
        require(
            cond1.rhs.name.startswith(other_side_prefix)
            and cond2.lhs.name.startswith(other_side_prefix)
            and cond1.rhs.name != cond2.lhs.name
        )
        return (
            (cond2.lhs.name.removeprefix(other_side_prefix), cond2.op == "<"),
            (cond1.rhs.name.removeprefix(other_side_prefix), cond1.op == "<"),
        )


def _normalize_expr_cond(cond: pandas.core.computation.ops.BinOp) -> None:
    """Normalize join condition comparison to use less-than instead of greater-than

    Args:
        cond: input comparison
    """
    if cond.op == ">":
        cond.op = "<"
        cond.lhs, cond.rhs = cond.operands = cond.rhs, cond.lhs

    if cond.op == ">=":
        cond.op = "<="
        cond.lhs, cond.rhs = cond.operands = cond.rhs, cond.lhs


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
