# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the join and merge"""
from collections import defaultdict
from typing import List, Literal, Union

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    next_label,
    replace_arg_nodes,
    replace_vars_inner,
    visit_vars_inner,
)
from numba.extending import intrinsic

import bodo
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    delete_table,
    delete_table_decref_arrays,
    hash_join_table,
    info_from_table,
    info_to_array,
)
from bodo.libs.int_arr_ext import IntDtype
from bodo.libs.str_arr_ext import cp_str_list_to_array, to_list_if_immutable_arr
from bodo.libs.timsort import getitem_arr_tup, setitem_arr_tup
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    find_common_np_dtype,
    is_dtype_nullable,
    is_nullable_type,
    is_str_arr_type,
    to_nullable_type,
)
from bodo.utils.utils import alloc_arr_tup, debug_prints, is_null_pointer

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
        df_out: str,
        left_df: str,
        right_df: str,
        left_keys: Union[List[str], str],
        right_keys: Union[List[str], str],
        out_data_vars: List[ir.Var],
        left_vars: List[ir.Var],
        right_vars: List[ir.Var],
        how: HOW_OPTIONS,
        suffix_left: str,
        suffix_right: str,
        loc: ir.Loc,
        is_left: bool,
        is_right: bool,
        is_join: bool,
        left_index: bool,
        right_index: bool,
        indicator: bool,
        is_na_equal: bool,
        gen_cond_expr: str,
    ):
        """
        IR node used to represent join operations. These are produced
        by pd.merge, pd.merge_asof, and DataFrame.join. The inputs
        have the following values.

        Keyword arguments:
        df_out -- Name of the output IR variable. Just used for printing.
        left_df -- Name of the left DataFrame's IR variable. Just used for printing.
        right_df -- Name of the right DataFrame's IR variable. Just used for printing.
        left_keys -- Label or list of labels used as the keys for the left DataFrame.
        right_keys -- Label or list of labels used as the keys for the left DataFrame.
        out_data_vars -- List of ir.Var used as the output arrays.
        left_vars -- List of ir.Var used as the left DataFrame's used arrays.
        right_vars -- List of ir.Var used as the right DataFrame's used arrays.
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
        indicator -- Add an indicator column? This indicates if each row
                     contains data from the left DataFrame, right DataFrame
                     or both.
        is_na_equal -- Should NA values be treated as equal when comparing keys?
                       In Pandas this is True, but conforming with SQL behavior
                       this is False.
        gen_cond_expr -- String used to describe the general merge condition. This
                         is used when a more general condition is needed than is
                         provided by equality.
        """
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_keys = left_keys
        self.right_keys = right_keys
        # Create a set for future lookups
        self.left_key_set = set(left_keys)
        self.right_key_set = set(right_keys)
        self.out_data_vars = out_data_vars
        self.left_vars = left_vars
        self.right_vars = right_vars
        self.how = how
        self.suffix_left = suffix_left
        self.suffix_right = suffix_right
        self.loc = loc
        self.is_left = is_left
        self.is_right = is_right
        self.is_join = is_join
        self.left_index = left_index
        self.right_index = right_index
        self.indicator = indicator
        self.is_na_equal = is_na_equal
        self.gen_cond_expr = gen_cond_expr

        if gen_cond_expr:
            # find columns used in general join condition to avoid removing them in rm dead
            # Note: this generates code per key and also is not fully correct. An expression
            # like (left.A)) != (right.B) will look like both A and A) are left key columns
            # based on this check, even though only A) is. Fixing this requires a more detailed
            # refactoring of the parsing.
            self.left_cond_cols = set(
                c for c in left_vars.keys() if f"(left.{c})" in gen_cond_expr
            )
            self.right_cond_cols = set(
                c for c in right_vars.keys() if f"(right.{c})" in gen_cond_expr
            )
        else:
            self.left_cond_cols = set()
            self.right_cond_cols = set()
        # keep the origin of output columns to enable proper dead code elimination
        # columns with common name that are not common keys will get a suffix
        comm_keys = self.left_key_set & self.right_key_set
        comm_data = set(left_vars.keys()) & set(right_vars.keys())
        add_suffix = comm_data - comm_keys
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
        # Track column origins
        self.column_origins = {}
        for c in left_vars.keys():
            rhs = ("left", c)
            if c in add_suffix:
                self.column_origins[str(c) + suffix_left] = rhs
                # If a column is both a data column and a key,
                # add it twice. This should only happen if
                # right_index=True
                if c in self.left_key_set:
                    self.column_origins[c] = rhs
            else:
                self.column_origins[c] = rhs

        for c in right_vars.keys():
            rhs = ("right", c)
            if c in add_suffix:
                self.column_origins[str(c) + suffix_right] = rhs
                # If a column is both a data column and a key,
                # add it twice. This should only happen if
                # left_index=True. See test_merge_left_index_dce
                if c in self.right_key_set:
                    self.column_origins[c] = rhs
            else:
                self.column_origins[c] = rhs

        # Keep the suffix to track possible collisions
        # See test_join::test_merge_suffix_included
        # and test_join::test_merge_suffix_collision
        #
        # Remove $_bodo_index_ as it should never create a suffix.
        # This name is reserved.
        if "$_bodo_index_" in add_suffix:
            add_suffix.remove("$_bodo_index_")
        self.add_suffix = add_suffix

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.out_data_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)

        in_cols = ""
        for (c, v) in self.left_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_left_str = "{}{{{}}}".format(self.left_df, in_cols)

        in_cols = ""
        for (c, v) in self.right_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_right_str = "{}{{{}}}".format(self.right_df, in_cols)
        return "join [{}={}]: {} , {}, {}".format(
            self.left_keys, self.right_keys, df_out_str, df_left_str, df_right_str
        )


def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    """
    Array analysis for the variables in the Join IR node. This states that
    all arrays in the input share a dimension and all arrays in the output
    share a dimension.
    """
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.out_data_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    in_vars = list(join_node.left_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    all_shapes = []
    in_vars = list(join_node.right_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in join_node.out_data_vars.values():
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

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
    for col_var in join_node.left_vars.values():
        left_dist = Distribution(min(left_dist.value, array_dists[col_var.name].value))

    for col_var in join_node.right_vars.values():
        right_dist = Distribution(
            min(right_dist.value, array_dists[col_var.name].value)
        )

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for col_var in join_node.out_data_vars.values():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    # out dist should meet input dist (e.g. REP in causes REP out)
    # output can be stay parallel if any of the inputs is parallel, hence max()
    out_dist1 = Distribution(min(out_dist.value, left_dist.value))
    out_dist2 = Distribution(min(out_dist.value, right_dist.value))
    out_dist = Distribution(max(out_dist1.value, out_dist2.value))
    for col_var in join_node.out_data_vars.values():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        left_dist = out_dist
        right_dist = out_dist

    # assign input distributions
    for col_var in join_node.left_vars.values():
        array_dists[col_var.name] = left_dist

    for col_var in join_node.right_vars.values():
        array_dists[col_var.name] = right_dist

    return


distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis


def join_typeinfer(join_node, typeinferer):
    """
    Type inference for the Join IR node. This enforces typing
    relationships between the inputs and the outputs.
    """

    for out_col_name, out_col_var in join_node.out_data_vars.items():
        # left suffix
        if join_node.indicator and out_col_name == "_merge":
            continue
        if not out_col_name in join_node.column_origins:
            raise BodoError(
                "join(): The variable " + out_col_name + " is absent from the output"
            )
        ePair = join_node.column_origins[out_col_name]
        if ePair[0] == "left":
            col_var = join_node.left_vars[ePair[1]]
        else:
            col_var = join_node.right_vars[ePair[1]]
        typeinferer.constraints.append(
            typeinfer.Propagate(
                dst=out_col_var.name, src=col_var.name, loc=join_node.loc
            )
        )
    return


typeinfer.typeinfer_extensions[Join] = join_typeinfer


def visit_vars_join(join_node, callback, cbdata):
    """
    Visit each variable in the Join IR node.
    """

    if debug_prints():  # pragma: no cover
        print("visiting join vars for:", join_node)
        print("cbdata: ", sorted(cbdata.items()))

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = visit_vars_inner(
            join_node.left_vars[col_name], callback, cbdata
        )
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = visit_vars_inner(
            join_node.right_vars[col_name], callback, cbdata
        )
    # output
    for col_name in list(join_node.out_data_vars.keys()):
        join_node.out_data_vars[col_name] = visit_vars_inner(
            join_node.out_data_vars[col_name], callback, cbdata
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
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    all_cols_dead = True

    for col_name, col_var in join_node.out_data_vars.items():
        if col_var.name in lives:
            all_cols_dead = False
            continue
        # Set the column to delete
        dead_cols.append(col_name)
        # avoid indicator (that is not in column_origins)
        if join_node.indicator and col_name == "_merge":
            # If _merge is removed, switch indicator to False so we don't expect
            # to generate indicator code.
            join_node.indicator = False
            continue
        orig, orig_name = join_node.column_origins[col_name]
        if col_name == "$_bodo_index_" and col_name in join_node.left_vars:
            # $_bodo_index_ may be in left and right so column_origins may
            # be wrong (as the dictionary can only store one orig)
            orig = "left"
        if (
            orig == "left"
            and orig_name not in join_node.left_key_set
            and orig_name not in join_node.left_cond_cols
        ):
            join_node.left_vars.pop(orig_name)
        if col_name == "$_bodo_index_" and col_name in join_node.right_vars:
            # $_bodo_index_ may be in left and right so column_origins may
            # be wrong (as the dictionary can only store one orig)
            orig = "right"
        if (
            orig == "right"
            and orig_name not in join_node.right_key_set
            and orig_name not in join_node.right_cond_cols
        ):
            join_node.right_vars.pop(orig_name)

    for cname in dead_cols:
        join_node.out_data_vars.pop(cname)

    # remove empty join node
    if all_cols_dead:
        return None

    return join_node


ir_utils.remove_dead_extensions[Join] = remove_dead_join


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
    use_set.update({v.name for v in join_node.left_vars.values()})
    use_set.update({v.name for v in join_node.right_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in join_node.out_data_vars.values()})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    """
    Return gen and kill sets for a copy propagation
    data flow analysis. Join doesn't generate any
    copies, it just kills the output columns.
    """
    kill_set = set(v.name for v in join_node.out_data_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Join] = get_copies_join


def apply_copies_join(
    join_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """Apply copy propagate in join node by replacing the inputs
    and the outputs."""

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = replace_vars_inner(
            join_node.left_vars[col_name], var_dict
        )
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = replace_vars_inner(
            join_node.right_vars[col_name], var_dict
        )
    # output
    for col_name in list(join_node.out_data_vars.keys()):
        join_node.out_data_vars[col_name] = replace_vars_inner(
            join_node.out_data_vars[col_name], var_dict
        )

    return


ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join


def build_join_definitions(join_node, definitions=None):
    """
    Construct definitions for the output variables of the
    join node.
    """
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in join_node.out_data_vars.values():
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
    left_parallel, right_parallel = False, False
    if array_dists is not None:
        left_parallel, right_parallel = _get_table_parallel_flags(
            join_node, array_dists
        )

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    n_keys = len(join_node.left_keys)
    # get column variables
    left_key_vars = tuple(join_node.left_vars[c] for c in join_node.left_keys)
    right_key_vars = tuple(join_node.right_vars[c] for c in join_node.right_keys)

    # Optional column refer: When doing a merge on column and index, the key
    # is put also in output, so we need one additional column in that case.
    optional_col_var = ()
    optional_key_tuple = ()
    optional_column = False
    if join_node.left_index and not join_node.right_index and not join_node.is_join:
        optional_key = join_node.right_keys[0]
        # We only need to generate the optional output key if
        # it hasn't been eliminated from the out_vars. We have
        # an optional_column when a column is both a key and
        # a data column. We check this with add_suffix because
        # the data columns may be removed via DCE.
        if (
            optional_key in join_node.add_suffix
            and optional_key in join_node.out_data_vars
        ):
            optional_key_tuple = (optional_key,)
            optional_col_var = (join_node.right_vars[optional_key],)
            optional_column = True

    if join_node.right_index and not join_node.left_index and not join_node.is_join:
        optional_key = join_node.left_keys[0]
        # We only need to generate the optional output key if
        # it hasn't been eliminated from the out_vars. We have
        # an optional_column when a column is both a key and
        # a data column. We check this with add_suffix because
        # the data columns may be removed via DCE.
        if (
            optional_key in join_node.add_suffix
            and optional_key in join_node.out_data_vars
        ):
            optional_key_tuple = (optional_key,)
            optional_col_var = (join_node.left_vars[optional_key],)
            optional_column = True

    # It is a fairly complex construction
    # ---keys can have same name on left and right.
    # ---keys can be two times or one time in output.
    # ---output keys can be computed from just one column (and so additional NaN may occur)
    #  or from two columns
    # ---keys may be from the index or not.
    #
    # The following rules apply:
    # ---A key that is an index or not behave in the same way. If it is an index key
    #  then its name is just the fancy "$_bodo_index_", so please don't use that one.
    # ---Identity of key is determined by their name, whether they are index or not.
    # ---If a key appears on same name on left and right then both columns are used
    #  and so the name will never have additional NaNs

    # Add optional keys to the output first
    merge_out = [join_node.out_data_vars[cname] for cname in optional_key_tuple]
    left_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.left_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.left_key_set
    )
    right_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.right_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.right_key_set
    )
    # get column types
    arg_vars = (
        optional_col_var
        + left_key_vars
        + right_key_vars
        + left_other_col_vars
        + right_other_col_vars
    )
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

    # arg names of non-key columns
    optional_names = tuple("opti_c" + str(i) for i in range(len(optional_col_var)))

    left_other_names = tuple("t1_c" + str(i) for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i) for i in range(len(right_other_col_vars)))
    left_other_types = tuple([typemap[c.name] for c in left_other_col_vars])
    right_other_types = tuple([typemap[c.name] for c in right_other_col_vars])
    left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))
    glbs = {}
    loc = join_node.loc

    func_text = "def f({}{}, {},{}{}{}):\n".format(
        ("{},".format(optional_names[0]) if len(optional_names) == 1 else ""),
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

    # For each key and data column used in the condition function,
    # keep track of it should be live in the output. This is
    # a list of boolean values, one per key/data column.

    # After the lists are populated the final lengths will be
    # len(left_key_in_output) = nkeys + num data_cols used in general cond func
    # len(right_key_in_output) = nkeys - (nshared_keys) + num data_cols used in general cond func
    # where nshared_keys == len(self.left_key_set & self.right_key_set)
    left_key_in_output = []
    right_key_in_output = []

    # Add the keys to the output.
    for cname in join_node.left_keys:
        if cname in join_node.add_suffix:
            cname_work = str(cname) + join_node.suffix_left
        else:
            cname_work = cname
        # If cname_work is not in the out_data_vars the column is
        # dead in the output.
        if cname_work in join_node.out_data_vars:
            merge_out.append(join_node.out_data_vars[cname_work])
            is_live = 1
        else:
            is_live = 0
        left_key_in_output.append(is_live)
    for i, cname in enumerate(join_node.right_keys):
        if not join_node.vect_same_key[i] and not join_node.is_join:

            if cname in join_node.add_suffix:
                cname_work = str(cname) + join_node.suffix_right
            else:
                cname_work = cname
            # If cname_work is not in the out_data_vars the column is
            # dead in the output.
            if cname_work in join_node.out_data_vars:
                merge_out.append(join_node.out_data_vars[cname_work])
                is_live = 1
            else:
                is_live = 0
            right_key_in_output.append(is_live)

    def _get_out_col_name(cname, is_left):
        """Return the name in the output variables
        for an input column.

        Args:
            cname (str): name in the input
            is_left (bool): is the source the left or right table.

        Returns:
            str: Name of the column in join_node.out_data_vars
                 if it is alive.
        """
        if cname in join_node.add_suffix:
            if is_left:
                cname_work = str(cname) + join_node.suffix_left
            else:
                cname_work = str(cname) + join_node.suffix_right
        else:
            cname_work = cname
        return cname_work

    def _get_out_col_var(cname, is_left):
        """Get the output array variable for a given
        input column name. Suffix names are resolved
        via `is_left`

        Args:
            cname (str): Name of the column in the input vars.
            is_left (bool): Is the input var the left or right table.

        Returns:
            ir.Var: Returns the output variable that corresponds
            to this input column name.
        """

        cname_work = _get_out_col_name(cname, is_left)
        return join_node.out_data_vars[cname_work]

    # Append the data columns to merge out.
    for n in sorted(join_node.left_vars.keys(), key=lambda a: str(a)):
        if n not in join_node.left_key_set:
            is_live = 1
            if n in join_node.left_cond_cols:
                cname_work = _get_out_col_name(n, True)
                # If cname_work is not in the out_data_vars the column is
                # dead in the output.
                if cname_work not in join_node.out_data_vars:
                    is_live = 0
                left_key_in_output.append(is_live)
            if is_live:
                merge_out.append(_get_out_col_var(n, True))

    for n in sorted(join_node.right_vars.keys(), key=lambda a: str(a)):
        if n not in join_node.right_key_set:
            is_live = 1
            if n in join_node.right_cond_cols:
                # If cname_work is not in the out_data_vars the column is
                # dead in the output.
                cname_work = _get_out_col_name(n, False)
                if cname_work not in join_node.out_data_vars:
                    is_live = 0
                right_key_in_output.append(is_live)
            if is_live:
                merge_out.append(_get_out_col_var(n, False))

    # Add the indicator variable.
    if join_node.indicator:
        merge_out.append(_get_out_col_var("_merge", False))
    out_names = ["t3_c" + str(i) for i in range(len(merge_out))]

    # Generate a general join condition function if it exists
    # and determine the data columns it needs.
    general_cond_cfunc, left_col_nums, right_col_nums = _gen_general_cond_cfunc(
        join_node,
        typemap,
    )

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
            optional_column,
            left_key_names,
            right_key_names,
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
            [typemap[v.name] for v in merge_out],
            join_node.loc,
            join_node.indicator,
            join_node.is_na_equal,
            general_cond_cfunc,
            left_col_nums,
            right_col_nums,
        )
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

    idx = 0
    if optional_column:
        func_text += f"    {out_names[idx]} = opti_0\n"
        idx += 1
    # Track the offset in right_key_in_output.
    # This is necessary because cond func columns may not be at the
    # front of the table and right will skip keys repeated in the
    # left table.
    right_key_idx = 0

    # Get the set of locations for condition columns.
    left_cond_col_nums_set = set(left_col_nums)
    right_cond_col_nums_set = set(right_col_nums)

    # Compute the key outputs
    for i in range(n_keys):
        if left_key_in_output[i]:
            func_text += f"    {out_names[idx]} = t1_keys_{i}\n"
            idx += 1
    for i in range(n_keys):
        if not join_node.vect_same_key[i] and not join_node.is_join:
            if right_key_in_output[right_key_idx]:
                func_text += f"    {out_names[idx]} = t2_keys_{i}\n"
                idx += 1
            right_key_idx += 1

    # Track the offset in left_key_in_output. All of the left keys
    # must be included, so we only track this for the data columns.
    # This is necessary because cond func columns may not be at the
    # front of the table.
    left_key_idx = n_keys

    # Compute the offsets used to determine the colnums
    # for general merge conditions
    left_key_offset = left_key_idx
    right_key_offset = right_key_idx

    # Compute the data outputs
    for i in range(len(left_other_names)):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + left_key_offset in left_cond_col_nums_set:
            load_arr = left_key_in_output[left_key_idx]
            left_key_idx += 1
        if load_arr:
            func_text += f"    {out_names[idx]} = left_{i}\n"
            idx += 1
    for i in range(len(right_other_names)):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + right_key_offset in right_cond_col_nums_set:
            load_arr = right_key_in_output[right_key_idx]
            right_key_idx += 1
        if load_arr:
            func_text += f"    {out_names[idx]} = right_{i}\n"
            idx += 1
    if join_node.indicator:
        func_text += f"    {out_names[idx]} = indicator_col\n"
        idx += 1
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
    for i in range(len(merge_out)):
        nodes[-len(merge_out) + i].target = merge_out[i]
    return nodes


distributed_pass.distributed_run_extensions[Join] = join_distributed_run


def _gen_general_cond_cfunc(join_node, typemap):
    """Generate cfunc for general join condition and return its address.
    Return 0 (NULL) if there is no general join condition to evaluate.
    The cfunc takes data pointers of table columns and row indices to access as input
    and returns True or False.
    E.g. left_table=[A_data_ptr, B_data_ptr], right_table=[A_data_ptr, C_data_ptr],
    left_ind=3, right_ind=7
    """
    expr = join_node.gen_cond_expr
    if not expr:
        return None, [], []

    label_suffix = next_label()

    # get column name to table column index
    left_col_to_ind = _get_col_to_ind(join_node.left_keys, join_node.left_vars)
    right_col_to_ind = _get_col_to_ind(join_node.right_keys, join_node.right_vars)

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
        left_col_to_ind,
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
        right_col_to_ind,
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
    col_to_ind,
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
    for c, c_ind in col_to_ind.items():
        cname = f"({table_name}.{c})"
        if cname not in expr:
            continue
        getitem_fname = f"getitem_{table_name}_val_{c_ind}"
        val_varname = f"_bodo_{table_name}_val_{c_ind}"
        array_typ = typemap[col_vars[c].name]
        if is_str_arr_type(array_typ):
            # If we have unicode we pass the table variable which is an array info
            func_text += f"  {val_varname}, {val_varname}_size = {getitem_fname}({table_name}_table, {table_name}_ind)\n"
            # Create proper Python string.
            func_text += f"  {val_varname} = bodo.libs.str_arr_ext.decode_utf8({val_varname}, {val_varname}_size)\n"
        else:
            # If we have a numeric type we just pass the data pointers
            func_text += f"  {val_varname} = {getitem_fname}({table_name}_data1, {table_name}_ind)\n"

        table_getitem_funcs[getitem_fname] = bodo.libs.array._gen_row_access_intrinsic(
            array_typ, c_ind
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
            ] = bodo.libs.array._gen_row_na_check_intrinsic(array_typ, c_ind)
            expr = expr.replace(na_cname, na_val_varname)

        # only append the column if it is not a key
        if c_ind >= n_keys:
            col_nums.append(c_ind)
    return expr, func_text, col_nums


def _get_col_to_ind(key_names, col_vars):
    """create a mapping from input dataframe column names to column indices in the C++
    left/right table structure (keys are first, then data columns).
    """
    n_keys = len(key_names)
    col_to_ind = {c: i for (i, c) in enumerate(key_names)}
    i = n_keys
    for c in sorted(col_vars, key=lambda a: str(a)):
        if c in col_to_ind:
            continue
        col_to_ind[c] = i
        i += 1
    return col_to_ind


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
        array_dists[v.name] in par_dists for v in join_node.left_vars.values()
    )
    right_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.right_vars.values()
    )
    if not left_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.left_vars.values()
        )
    if not right_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.right_vars.values()
        )

    if left_parallel or right_parallel:
        assert all(
            array_dists[v.name] in par_dists for v in join_node.out_data_vars.values()
        )

    return left_parallel, right_parallel


def _gen_local_hash_join(
    optional_column,
    left_key_names,
    right_key_names,
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
    out_types,
    loc,
    indicator,
    is_na_equal,
    general_cond_cfunc,
    left_col_nums,
    right_col_nums,
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
    for i in range(len(left_key_names)):
        if left_key_in_output[i]:
            key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
            vect_need_typechange.append(
                needs_typechange(key_type, is_right, vect_same_key[i])
            )

    # Offset for left and right table inside the key in output
    # lists. Every left key is always included so we only count
    # the left table for data columns.
    left_key_in_output_idx = len(left_key_names)
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

    for i in range(len(right_key_names)):
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

    def get_out_type(idx, in_type, in_name, need_nullable, is_same_key):
        """
        Construct a dummy array with a given type for passing to info_to_array.
        If an array is unchanged between input and output we reuse the same array.
        """
        if (
            isinstance(in_type, types.Array)
            and not is_dtype_nullable(in_type.dtype)
            and need_nullable
            and not is_same_key
        ):
            # the only types with another nullable version are int and bool currently
            if isinstance(in_type.dtype, types.Integer):
                int_typ_name = IntDtype(in_type.dtype).name
                assert int_typ_name.endswith("Dtype()")
                int_typ_name = int_typ_name[:-7]
                ins_text = f'    typ_{idx} = bodo.hiframes.pd_series_ext.get_series_data(pd.Series([1], dtype="{int_typ_name}"))\n'
                out_type = f"typ_{idx}"
            else:
                assert (
                    in_type.dtype == types.bool_
                ), "unexpected non-nullable type in join"
                ins_text = (
                    f"    typ_{idx} = bodo.libs.bool_arr_ext.alloc_bool_array(1)\n"
                )
                out_type = f"typ_{idx}"
        elif in_type == bodo.string_array_type:
            # Generate an output for string arrays to handle join DictionaryArray and StringArray
            # with a string output.
            ins_text = (
                f"    typ_{idx} = bodo.libs.str_arr_ext.pre_alloc_string_array(0, 0)\n"
            )
            out_type = f"typ_{idx}"
        else:
            ins_text = ""
            out_type = in_name
        return (ins_text, out_type)

    n_keys = len(left_key_names)
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
    func_text += "    vect_same_key = np.array([{}])\n".format(
        ",".join("1" if x else "0" for x in vect_same_key)
    )
    func_text += "    vect_need_typechange = np.array([{}])\n".format(
        ",".join("1" if x else "0" for x in vect_need_typechange)
    )
    func_text += f"    left_table_cond_columns = np.array({left_col_nums if len(left_col_nums) > 0 else [-1]}, dtype=np.int64)\n"
    func_text += f"    right_table_cond_columns = np.array({right_col_nums if len(right_col_nums) > 0 else [-1]}, dtype=np.int64)\n"
    if general_cond_cfunc:
        func_text += f"    cfunc_cond = add_join_gen_cond_cfunc_sym(general_cond_cfunc, '{general_cond_cfunc.native_name}')\n"
        func_text += (
            f"    cfunc_cond = get_join_cond_addr('{general_cond_cfunc.native_name}')\n"
        )
    else:
        func_text += "    cfunc_cond = 0\n"

    func_text += "    out_table = hash_join_table(table_left, table_right, {}, {}, {}, {}, {}, vect_same_key.ctypes, key_in_output.ctypes, vect_need_typechange.ctypes, {}, {}, {}, {}, {}, {}, cfunc_cond, left_table_cond_columns.ctypes, {}, right_table_cond_columns.ctypes, {})\n".format(
        left_parallel,
        right_parallel,
        n_keys,
        len(left_other_names),
        len(right_other_names),
        is_left,
        is_right,
        is_join,
        optional_column,
        indicator,
        is_na_equal,
        len(left_col_nums),
        len(right_col_nums),
    )
    func_text += "    delete_table(table_left)\n"
    func_text += "    delete_table(table_right)\n"
    idx = 0
    # Load the output optional column
    if optional_column:
        rec_typ = get_out_type(idx, out_types[idx], "opti_c0", False, False)
        func_text += rec_typ[0]
        glbs[f"out_type_{idx}"] = out_types[idx]
        func_text += f"    opti_0 = info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]})\n"
        idx += 1

    # Load the left keys
    for i, t in enumerate(left_key_names):
        if left_key_in_output[i]:
            key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
            rec_typ = get_out_type(
                idx, key_type, f"t1_keys[{i}]", is_right, vect_same_key[i]
            )
            func_text += rec_typ[0]
            glbs[f"out_type_{idx}"] = out_types[idx]
            # use astype only if necessary due to merge bugs
            # see: test_merge_match_key_types"
            if (
                key_type != left_key_types[i]
                and left_key_types[i] != bodo.dict_str_arr_type
            ):
                func_text += f"    t1_keys_{i} = bodo.utils.utils.astype(info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]}), out_type_{idx})\n"
            else:
                func_text += f"    t1_keys_{i} = info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]})\n"
            idx += 1

    # Reset the indices for determining if outputs are live.
    # Again all keys in left_key are always in left_key_in_output,
    # so we just track for the data columns.
    left_key_idx = len(left_key_names)
    right_key_idx = 0

    # Load the left data arrays
    for i, t in enumerate(left_other_names):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + left_key_offset in left_cond_col_nums_set:
            load_arr = left_key_in_output[left_key_idx]
            left_key_idx += 1
        if load_arr:
            rec_typ = get_out_type(idx, left_other_types[i], t, is_right, False)
            func_text += rec_typ[0]
            func_text += "    left_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                i, idx, rec_typ[1]
            )
            idx += 1

    # Load the right keys
    for i, t in enumerate(right_key_names):
        if not vect_same_key[i] and not is_join:
            if right_key_in_output[right_key_idx]:
                key_type = _match_join_key_types(
                    left_key_types[i], right_key_types[i], loc
                )
                rec_typ = get_out_type(idx, key_type, f"t2_keys[{i}]", is_left, False)
                func_text += rec_typ[0]
                # use astype only if necessary due to merge bugs
                # see: test_merge_match_key_types"
                # NOTE: subtracting len(left_other_names) since output right keys are
                # generated before left_other_names
                glbs[f"out_type_{idx}"] = out_types[idx - len(left_other_names)]
                if (
                    key_type != right_key_types[i]
                    and right_key_types[i] != bodo.dict_str_arr_type
                ):
                    func_text += f"    t2_keys_{i} = bodo.utils.utils.astype(info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]}), out_type_{idx})\n"
                else:
                    func_text += f"    t2_keys_{i} = info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]})\n"
                idx += 1
            right_key_idx += 1

    # Load the right data arrays
    for i, t in enumerate(right_other_names):
        load_arr = True
        # Data columns may be used in the non-equality
        # condition function. If so they are treated
        # like a key and can be eliminated from the output
        # but not the input.
        if i + right_key_offset in right_cond_col_nums_set:
            load_arr = right_key_in_output[right_key_idx]
            right_key_idx += 1
        if load_arr:
            rec_typ = get_out_type(idx, right_other_types[i], t, is_left, False)
            func_text += rec_typ[0]
            func_text += "    right_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                i, idx, rec_typ[1]
            )
            idx += 1
    # Load the indicator column
    if indicator:
        func_text += f"    typ_{idx} = pd.Categorical(values=['both'], categories=('left_only', 'right_only', 'both'))\n"
        func_text += f"    indicator_col = info_to_array(info_from_table(out_table, {idx}), typ_{idx})\n"
        idx += 1

    func_text += "    delete_table(out_table)\n"
    return func_text


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
