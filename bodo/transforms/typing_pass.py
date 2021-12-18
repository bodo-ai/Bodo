"""
Bodo type inference pass that performs transformations that enable typing of the IR
according to Bodo requirements (using partial typing).
"""
import copy
import itertools
import operator
import types as pytypes
import warnings
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, types
from numba.core.compiler_machinery import register_pass
from numba.core.extending import register_jitable
from numba.core.ir_utils import (
    GuardException,
    build_definitions,
    compute_cfg_from_blocks,
    dprint_func_ir,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    is_setitem,
    mk_unique_var,
    require,
)
from numba.core.typed_passes import NopythonTypeInference, PartialTypeInference

import bodo
from bodo.hiframes.dataframe_indexing import DataFrameILocType, DataFrameLocType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
from bodo.hiframes.pd_rolling_ext import RollingType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.transform import (
    compile_func_single_block,
    get_call_expr_arg,
    get_const_value_inner,
    set_call_expr_arg,
    update_node_list_definitions,
)
from bodo.utils.typing import (
    CONST_DICT_SENTINEL,
    BodoConstUpdatedError,
    BodoError,
    BodoWarning,
    FilenameType,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_str,
    is_const_func_type,
    is_list_like_index_type,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodo.utils.utils import (
    find_build_tuple,
    get_getsetitem_index_var,
    is_assign,
    is_call,
    is_call_assign,
    is_expr,
)

# global flag indicating that we are in partial type inference, so that error checking
# code can raise regular Exceptions that can potentially be handled here
in_partial_typing = False
# global flag set by error checking code (e.g. df.drop) indicating that a transformation
# in the typing pass is required. Necessary since types.unknown may not be assigned to
# all types by Numba properly, e.g. TestDataFrame::test_df_drop_inplace1.
typing_transform_required = False
# limit on maximum number of total statements generated in loop unrolling to avoid
# very long compilation time
loop_unroll_limit = 10000


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoTypeInference(PartialTypeInference):
    _name = "bodo_type_inference"

    def run_pass(self, state):
        """run Bodo type inference pass"""
        # _raise_errors is a global class attribute, which can be set/unset in recursive
        # calls. It is dangerous since the output type of the function is set only if
        # _raise_errors = True for some reason (see #964 and test_df_apply_func_case2):
        # https://github.com/numba/numba/blob/1ea770564cb3c0c6cb9d8ab92e7faf23cd4c4c19/numba/core/typed_passes.py#L100
        old_raise_errors = self._raise_errors
        try:
            self._raise_errors = False
            return self.run_pass_inner(state)
        finally:
            self._raise_errors = old_raise_errors

    def run_pass_inner(self, state):
        global in_partial_typing, typing_transform_required
        saved_in_partial_typing = in_partial_typing
        saved_typing_transform_required = typing_transform_required
        curr_typing_pass_required = False
        # flag indicating that transformation has run at least once
        ran_transform = False
        # flag for when another transformation pass is needed (to avoid break before
        # next transform)
        needs_transform = False
        while True:
            try:
                # set global partial typing flag, see comment above
                in_partial_typing = True
                typing_transform_required = False
                super(BodoTypeInference, self).run_pass(state)
                curr_typing_pass_required = typing_transform_required
            finally:
                in_partial_typing = saved_in_partial_typing
                typing_transform_required = saved_typing_transform_required

            # done if all types are available and transform not required
            if (
                not state.typing_errors
                and self._all_types_available(state)
                and not curr_typing_pass_required
                and not needs_transform
            ):
                break
            typing_transforms_pass = TypingTransforms(
                state.func_ir,
                state.typingctx,
                state.targetctx,
                state.typemap,
                state.calltypes,
                state.args,
                state.locals,
                state.flags,
                True,
                ran_transform,
            )
            ran_transform = True
            prev_needs_transform = needs_transform
            changed, needs_transform = typing_transforms_pass.run()
            # transform pass has failed if transform was needed but IR is not changed.
            # This avoids infinite loop, see [BE-140]
            if prev_needs_transform and needs_transform and not changed:
                break
            # can't be typed if IR not changed
            if not changed and not needs_transform:
                # error will be raised below if there are still unknown types
                break

        # make sure transformation has run at least once to handle cases that may not
        # throw typing errors like "df.B = v". See test_set_column_setattr
        rerun_typing = False
        if not ran_transform:
            typing_transforms_pass = TypingTransforms(
                state.func_ir,
                state.typingctx,
                state.targetctx,
                state.typemap,
                state.calltypes,
                state.args,
                state.locals,
                state.flags,
                False,
                ran_transform,
            )
            changed, needs_transform = typing_transforms_pass.run()
            # some cases need a second transform pass to raise the proper error
            # see test_df_rename::impl4
            if needs_transform:
                typing_transforms_pass = TypingTransforms(
                    state.func_ir,
                    state.typingctx,
                    state.targetctx,
                    state.typemap,
                    state.calltypes,
                    state.args,
                    state.locals,
                    state.flags,
                    False,
                    True,
                )
                changed, needs_transform = typing_transforms_pass.run()
            # need to rerun type inference if the IR changed
            # see test_set_column_setattr
            rerun_typing = changed or needs_transform

        dprint_func_ir(state.func_ir, "after typing pass")
        self._check_for_errors(state, curr_typing_pass_required or rerun_typing)
        return True

    def _check_for_errors(self, state, curr_typing_pass_required):
        """check for type inference issues and call Numba's type inference to raise
        proper errors if necessary.
        """
        # get return type since partial type inference skips it for some reason
        # similar to: https://github.com/numba/numba/blob/1041fa6ee8430471da99b54b3428a673033e7e44/numba/core/typeinfer.py#L1209
        return_type = None
        ret_types = []
        for blk in state.func_ir.blocks.values():
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                ret_types.append(state.typemap.get(inst.value.name, None))
        if None not in ret_types:
            try:
                return_type = state.typingctx.unify_types(*ret_types)
            except:
                pass
            if return_type is None:
                raise_bodo_error(
                    f"Unable to unify the following function return types: {ret_types}"
                )
            if (
                not isinstance(return_type, types.FunctionType)
                and not return_type.is_precise()
            ):
                return_type = None

        if (
            state.typing_errors
            or curr_typing_pass_required
            or types.unknown in state.typemap.values()
            or state.calltypes is None
            or return_type is None
            or state.func_ir.generator_info
        ):
            # run regular type inference again with _raise_errors=True to raise errors
            NopythonTypeInference().run_pass(state)
        else:
            # last return type check in Numba:
            # https://github.com/numba/numba/blob/0bac18af44d08e913cd512babb9f9b7f6386d30a/numba/core/typed_passes.py#L141
            if isinstance(return_type, types.Function) or isinstance(
                return_type, types.Phantom
            ):
                msg = "Can't return function object ({})"
                raise TypeError(msg.format(return_type))
            state.return_type = return_type

    def _all_types_available(self, state):
        """check to see if all variable types are available in typemap."""
        # Numba's partial type inference may miss typing some variables as "unknown" so
        # we set "unknown" if necessary
        typemap = state.typemap
        for blk in state.func_ir.blocks.values():
            for stmt in blk.body:
                if is_assign(stmt) and stmt.target.name not in typemap:
                    typemap[stmt.target.name] = types.unknown

        return types.unknown not in typemap.values()


class TypingTransforms:
    """
    Transformations that enable typing of the IR according to Bodo requirements.

    Infer possible constant values (e.g. lists) using partial typing info and transform
    them to constants so that functions like groupby() can be typed properly.
    """

    def __init__(
        self,
        func_ir,
        typingctx,
        targetctx,
        typemap,
        calltypes,
        arg_types,
        _locals,
        flags,
        change_required,
        ran_transform,
    ):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        # calltypes may be None (example in forecast code, hard to reproduce in test)
        self.calltypes = {} if calltypes is None else calltypes  # pragma: no cover
        self.arg_types = arg_types
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}
        # labels of rhs of assignments to enable finding nodes that create
        # dataframes such as Arg(df_type), pd.DataFrame(), df[['A','B']]...
        # the use is conservative and doesn't assume complete info
        self.rhs_labels = {}
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        # variables that are transformed away at some point and are potentially dead
        self._transformed_vars = set()
        # variables that are potentially list/set/dict and updated inplace
        self._updated_containers = {}
        # contains variables that need to be constant (e.g. index of df getitem) but
        # couldn't be inferred as constant, so loop unrolling is tried later for them if
        # possible.
        # variable (ir.Var) -> block label it is needed as constant
        self._require_const = {}
        self.locals = _locals
        self.flags = flags
        # a change in the IR in current pass is required to enable typing
        self.change_required = change_required
        # whether transform has run before (e.g. loop unrolling is attempted)
        self.ran_transform = ran_transform
        self.changed = False
        # whether another transformation pass is needed (see _run_setattr)
        self.needs_transform = False

    def run(self):
        # XXX: the block structure shouldn't change in this pass since labels
        # are used in analysis (e.g. df creation points in rhs_labels)
        blocks = self.func_ir.blocks
        topo_order = find_topo_order(blocks)
        self._updated_containers, self._equiv_vars = _find_updated_containers(
            blocks, topo_order
        )
        for label in topo_order:
            block = blocks[label]
            self._working_body = []
            for inst in block.body:
                self._replace_vars(inst)
                out_nodes = [inst]
                self.curr_loc = inst.loc

                # handle potential dataframe set column here
                # df['col'] = arr
                if isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                    out_nodes = self._run_setitem(inst, label)
                elif isinstance(inst, ir.SetAttr):
                    out_nodes = self._run_setattr(inst, label)
                elif isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    self.rhs_labels[inst.value] = label
                    out_nodes = self._run_assign(inst, label)

                self._working_body.extend(out_nodes)
                update_node_list_definitions(out_nodes, self.func_ir)
                for inst in out_nodes:
                    if is_assign(inst):
                        self.rhs_labels[inst.value] = label

            blocks[label].body = self._working_body

        # try loop unrolling if some const values couldn't be resolved
        if self._require_const:
            self._try_loop_unroll_for_const()

        # try unrolling a loop with constant range if everything else failed
        if self.change_required and not self.changed and not self.needs_transform:
            self._try_unroll_const_loop()

        # find transformed variables that are not used anymore so they can be removed
        # removing cases like agg dicts may be necessary since not type stable
        remove_vars = self._transformed_vars.copy()
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                # ignore dead definitions of variables
                if (
                    isinstance(stmt, ir.Assign)
                    and stmt.target.name in self._transformed_vars
                ):
                    continue
                # remove variables that are used somewhere else
                remove_vars -= set(v.name for v in stmt.list_vars())

        # add dummy variable with constant zero in first block after arg nodes to use
        # below
        first_block = self.func_ir.blocks[min(self.func_ir.blocks.keys())]
        zero_var = ir.Var(first_block.scope, mk_unique_var("zero"), first_block.loc)
        const_assign = ir.Assign(
            ir.Const(0, first_block.loc), zero_var, first_block.loc
        )
        self.func_ir._definitions[zero_var.name] = [const_assign.value]
        first_block.body.insert(len(self.arg_types), const_assign)

        # change transformed variable to a trivial case to enable type inference
        for v in remove_vars:
            rhs = get_definition(self.func_ir, v)
            if is_expr(rhs, "build_map"):
                rhs.items = [(zero_var, zero_var)]
            if is_expr(rhs, "build_list"):
                rhs.items = [zero_var]

        return self.changed, self.needs_transform

    def _run_assign(self, assign, label):
        rhs = assign.value

        if isinstance(rhs, ir.Expr) and rhs.op in ("binop", "inplace_binop"):
            return self._run_binop(assign, rhs)

        if is_call(rhs):
            return self._run_call(assign, rhs, label)

        if isinstance(rhs, ir.Expr) and rhs.op in ("getitem", "static_getitem"):
            return self._run_getitem(assign, rhs, label)

        if isinstance(rhs, ir.Expr) and rhs.op == "make_function":
            return self._run_make_function(assign, rhs)

        # remove leftover data types for tuples of make_function values replaced above
        # needed since _replace_arg_with_literal() cannot handle make_function values
        # see test_groupby_agg_const_dict::impl18
        # see test_groupby_agg_func_list
        if isinstance(rhs, ir.Expr) and rhs.op in ("build_tuple", "build_list"):
            tup_typ = self.typemap.get(assign.target.name, None)
            is_func_literal = lambda t: isinstance(
                t, types.MakeFunctionLiteral
            ) or is_expr(t, "make_function")
            # check for BaseTuple since could be types.unknown
            if (
                isinstance(tup_typ, (types.BaseTuple, types.LiteralList))
                and any(is_func_literal(t) for t in tup_typ)
            ) or (
                isinstance(tup_typ, types.List)
                and tup_typ.initial_value is not None
                and any(is_func_literal(t) for t in tup_typ.initial_value)
            ):
                self.typemap.pop(assign.target.name, None)
                # avoid list of func typing errors, see comment in _create_const_var
                rhs.op = "build_tuple"

        return [assign]

    def _run_getitem(self, assign, rhs, label):
        """Handle getitem if necessary.
        df[], df.iloc[], df.loc[] may need constant index values
        """
        target = rhs.value
        target_typ = self.typemap.get(target.name, None)
        nodes = []
        idx = get_getsetitem_index_var(rhs, self.typemap, nodes)
        idx_typ = self.typemap.get(idx.name, None)

        # find constant index for df["A"], df[["A", "B"]] or df.groupby("A")["B"] cases
        # constant index can be string, int or non-bool list
        if (
            isinstance(target_typ, (DataFrameType, DataFrameGroupByType, RollingType))
            and not is_literal_type(idx_typ)
            and (
                idx_typ == bodo.string_type
                or isinstance(idx_typ, types.Integer)
                or (
                    isinstance(idx_typ, types.List) and not idx_typ.dtype == types.bool_
                )
            )
        ):
            # NOTE: avoid using rhs.index for "static_getitem" since it can be wrong
            # see https://github.com/numba/numba/issues/7592
            # try to find index values
            try:
                err_msg = "DataFrame[] requires constant column names"
                val = self._get_const_value(idx, label, rhs.loc, err_msg)
            except (GuardException, BodoConstUpdatedError):
                # couldn't find values, just return to be handled later
                # save for potential loop unrolling
                nodes.append(assign)
                return nodes
            # replace index variable with a new variable holding constant
            new_var = _create_const_var(val, idx.name, idx.scope, idx.loc, nodes)
            if rhs.op == "static_getitem":
                rhs.index_var = new_var
                # update value of static_getitem since it can be wrong
                rhs.index = val
            else:
                rhs.index = new_var
            # old index var is not used anymore
            self._transformed_vars.add(idx.name)
            self.changed = True
            nodes.append(assign)
            return nodes

        # transform df.iloc[:,1:] case here since slice info not available in overload
        if (
            isinstance(target_typ, DataFrameILocType)
            and isinstance(idx_typ, types.BaseTuple)
            and len(idx_typ.types) == 2
            and isinstance(idx_typ.types[1], types.SliceType)
        ):
            # get slice on columns
            tup_list = guard(find_build_tuple, self.func_ir, idx)
            if tup_list is None or len(tup_list) != 2:  # pragma: no cover
                raise BodoError("Invalid df.iloc[slice,slice] case")
            slice_var = tup_list[1]

            # get const value of slice
            col_slice = guard_const(
                get_const_value_inner,
                self.func_ir,
                slice_var,
                self.arg_types,
                self.typemap,
            )
            # try to force constant later (e.g. by unrolling loops) if possible
            if col_slice is None:
                self._require_const[slice_var] = label
                nodes.append(assign)
                return nodes

            # NOTE: dataframe type may have changed in typing pass (e.g. due to df setitem)
            # so we shouldn't use target_typ and should check for the actual df variable
            df_var = self._get_loc_df_var(target)
            df_type = self.typemap.get(df_var.name, None)
            if df_type is None:
                return nodes + [assign]

            # create output df
            columns = df_type.columns[col_slice]
            # get df arrays using const slice
            # data_outs = []
            # Generate the results by reusing the indexing helper functions
            if isinstance(idx_typ.types[0], types.Integer):
                impl = bodo.hiframes.dataframe_indexing._gen_iloc_getitem_row_impl(
                    df_type, columns, "idx"
                )
            elif (
                is_list_like_index_type(idx_typ.types[0])
                and isinstance(idx_typ.types[0].dtype, (types.Integer, types.Boolean))
                or isinstance(idx_typ.types[0], types.SliceType)
            ):
                impl = (
                    bodo.hiframes.dataframe_indexing._gen_iloc_getitem_bool_slice_impl(
                        df_type, columns, idx_typ.types[0], "idx", False
                    )
                )
            else:
                bodo.utils.typing.raise_bodo_error(
                    f"df.iloc[] getitem using {idx_typ} not supported"
                )  # pragma: no cover

            self.changed = True
            # NOTE: not passing "self" since target type may change
            return nodes + compile_func_single_block(
                impl, [target, tup_list[0]], assign.target
            )

        # transform df.loc[:, df.columns != "B"] case here since slice info not
        # available in overload
        if (
            isinstance(target_typ, DataFrameLocType)
            and isinstance(idx_typ, types.BaseTuple)
            and len(idx_typ.types) == 2
            and is_list_like_index_type(idx_typ.types[1])
        ):
            # get column index var
            tup_list = guard(find_build_tuple, self.func_ir, idx)
            if tup_list is None or len(tup_list) != 2:  # pragma: no cover
                raise BodoError("Invalid df.loc[ind,ind] case")
            col_ind_var = tup_list[1]

            # try to find index values
            try:
                err_msg = "DataFrame.loc[] requires constant column names"
                val = self._get_const_value(col_ind_var, label, rhs.loc, err_msg)
            except (GuardException, BodoConstUpdatedError):
                # couldn't find values, just return to be handled later
                nodes.append(assign)
                return nodes

            # NOTE: dataframe type may have changed in typing pass (e.g. due to df setitem)
            # so we shouldn't use target_typ and should check for the actual df variable
            df_var = self._get_loc_df_var(target)
            df_type = self.typemap.get(df_var.name, None)
            if df_type is None:
                return nodes + [assign]

            # avoid transform if selected columns not all in dataframe schema
            # may require schema change, see test_loc_col_select (impl4)
            if (
                len(val) > 0
                and not isinstance(val[0], (bool, np.bool_))
                and not all(c in df_type.columns for c in val)
            ):
                nodes.append(assign)
                return nodes

            impl = bodo.hiframes.dataframe_indexing.gen_df_loc_col_select_impl(
                df_type, val
            )
            self.changed = True
            # NOTE: not passing "self" since target type may change
            return nodes + compile_func_single_block(impl, [target, idx], assign.target)

        # detect if filter pushdown is possible and transform
        # e.g. df = pd.read_parquet(...); df = df[df.A > 3]
        index_def = guard(get_definition, self.func_ir, rhs.index)
        value_def = guard(get_definition, self.func_ir, rhs.value)
        if (
            is_expr(index_def, "binop")
            and is_call(value_def)
            and guard(find_callname, self.func_ir, value_def)
            == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext")
        ):
            working_body = guard(
                self._try_filter_pushdown,
                assign,
                value_def,
                index_def,
                self._working_body,
                self.func_ir,
                {
                    rhs.value.name: value_def
                },  # Basic filter pushdown case only has 1 dataframe to track.
                False,
            )
            # If this function returns a list we have updated the working body.
            # This is done to enable updating a single block that is not yet being processed
            # in the working body.
            if working_body is not None:
                self._working_body = working_body

        nodes.append(assign)
        return nodes

    def _try_filter_pushdown(
        self,
        assign,
        value_def,
        index_def,
        working_body,
        func_ir,
        used_dfs,
        from_bodosql,
    ):
        """detect filter pushdown and add filters to ParquetReader or SQLReader IR nodes if possible.

        working_body is in the inprogress list of statements that should be updated with any filter reordering.
        A new working_body is returned if this is successful. func_ir is FunctionIR object containing the blocks
        with all relevant code.

        used_dfs is a dictionary of intermediate dataframes -> initialization that should be tracked
        to ensure they aren't reused

        Throws GuardException if not possible.
        """
        # avoid empty dataframe
        require(len(value_def.args) > 0)
        data_def = get_definition(func_ir, value_def.args[0])
        assert is_expr(data_def, "build_tuple"), "invalid data tuple in init_dataframe"
        read_node = get_definition(func_ir, data_def.items[0])
        require(
            isinstance(
                read_node,
                (bodo.ir.parquet_ext.ParquetReader, bodo.ir.sql_ext.SqlReader),
            )
        )
        require(all(get_definition(func_ir, v) == read_node for v in data_def.items))
        if isinstance(read_node, bodo.ir.sql_ext.SqlReader):
            # Filter pushdown is only supported for snowflake right now.
            require(read_node.db_type == "snowflake")
        elif isinstance(read_node, bodo.ir.parquet_ext.ParquetReader):
            filename_var = read_node.file_name.name
            # If the filename_var isn't in the typemap we are likely
            # coming from BodoSQL. As a result, we disable this check
            # to ensure filter pushdown works in the common case.
            # TODO: Enable check for BodoSQL
            if filename_var in self.typemap:
                filename_typ = self.typemap[filename_var]
                fname = ""
                if is_overload_constant_str(filename_typ):
                    fname = get_overload_const_str(filename_typ)
                elif isinstance(filename_typ, FilenameType) and isinstance(
                    filename_typ.fname, str
                ):
                    # This path is currently untested.
                    # TODO: Test
                    fname = filename_typ.fname
                if fname:
                    require(
                        not bodo.io.parquet_pio.is_filter_pushdown_disabled_fpath(fname)
                    )

        # make sure all filters have the right form
        lhs_def = get_definition(func_ir, index_def.lhs)
        rhs_def = get_definition(func_ir, index_def.rhs)
        df_var = assign.value.value
        filters = self._get_partition_filters(
            index_def,
            df_var,
            lhs_def,
            rhs_def,
            func_ir,
            # SQL generates different operators than pyarrow
            read_node,
            from_bodosql,
        )
        self._check_non_filter_df_use(set(used_dfs.keys()), assign, func_ir)
        new_working_body = self._reorder_filter_nodes(
            read_node, index_def, used_dfs, filters, working_body, func_ir
        )
        old_filters = read_node.filters
        # If there are existing filters then we need to merge them together because this is
        # an implicit AND. We merge by distributed the AND over ORs
        # (A or B) AND (C or D) -> AC or AD or BC or BD
        # See test_read_partitions_implicit_and_detailed for an example usage.
        if old_filters is not None:
            new_filters = []
            for old_or_cond in old_filters:
                for new_or_cond in filters:
                    new_filters.append(old_or_cond + new_or_cond)
            filters = new_filters

        # set ParquetReader/SQLReader node filters (no exception was raise until this end point
        # so filters are valid)
        read_node.filters = filters
        # remove filtering code since not necessary anymore
        assign.value = assign.value.value
        # Mark the IR as changed
        self.changed = True
        # Return the updates to the working body so we can modify blocks that may not
        # be in the working body yet.
        return new_working_body

    def _check_non_filter_df_use(self, df_names, assign, func_ir):
        """make sure the chain of used dataframe variables are not used after filtering in the
        program. e.g. df2 = df[...]; A = df.A
        Assumes that Numba renames variables if the same df name is used later. e.g.:
            df2 = df[...]
            df = ....  # numba renames df to df.1
        TODO(ehsan): use proper liveness analysis to handle cases with control flow:
            df2 = df[...]
            if flag:
                df = ....
        """
        for block in func_ir.blocks.values():
            for stmt in reversed(block.body):
                # ignore code before the filtering node in the same basic block
                if stmt is assign:
                    break
                require(all(v.name not in df_names for v in stmt.list_vars()))

    def _reorder_filter_nodes(
        self, read_node, index_def, used_dfs, filters, working_body, func_ir
    ):
        """reorder nodes that are used for Parquet/SQL partition filtering to be before the
        Reader node (to be accessible when the Reader is run).

        df_names is a set of variables that need to be tracked to perform the filter pushdown.

        Throws GuardException if not possible.
        """
        # e.g. [[("a", "0", ir.Var("val"))]] -> {"val"}
        filter_vars = set()
        for predicate_list in filters:
            for v in predicate_list:
                # If v[2] is a variable add it to the filter_vars. If its
                # a compile time constant (i.e. NULL) then don't add it.
                if isinstance(v[2], ir.Var):
                    filter_vars.add(v[2].name)
        # data array variables should not be used in filter expressions directly
        non_filter_vars = {v.name for v in read_node.list_vars()}

        # find all variables that are potentially used in filter expressions after the
        # reader node
        # make sure they don't overlap with other nodes (to be conservative)
        i = 0  # will be set to ParquetReader node's reversed index
        # nodes used for filtering output dataframe use filter vars as well but should
        # be excluded since they have dependency to data arrays (e.g. df["A"] == 3)
        filter_nodes = self._get_filter_nodes(index_def, func_ir)
        # get all variables related to filtering nodes in some way, to make sure df is
        # not used in other ways before filtering
        # e.g.
        # df = pd.read_parquet("../tmp/pq_data3")
        # n = len(df)
        # df = df[df["A"] == 2]
        related_vars = set()

        # Get the set of intermediate df names
        df_names = set(used_dfs.keys())
        for node in filter_nodes:
            related_vars.update({v.name for v in node.list_vars()})
        for stmt in reversed(working_body):
            i += 1
            # ignore dataframe filter expression nodes
            if is_assign(stmt) and stmt.value in filter_nodes:
                continue
            # handle a known initialization
            # i.e. df = $1
            if (
                is_assign(stmt)
                and stmt.target.name in df_names
                and (
                    isinstance(stmt.value, ir.Var)
                    or stmt.value is used_dfs[stmt.target.name]
                )
            ):
                continue

            # avoid nodes before the reader
            if stmt is read_node:
                break
            stmt_vars = {v.name for v in stmt.list_vars()}

            # make sure df is not used before filtering
            if not stmt_vars & related_vars:
                # df_names is a non-empty set, so if the intersection
                # is empty then a df_name is in stmt_vars
                require(not (df_names & stmt_vars))
            else:
                related_vars |= stmt_vars - df_names

            if stmt_vars & filter_vars:
                filter_vars |= stmt_vars
            else:
                non_filter_vars |= stmt_vars

        require(not (filter_vars & non_filter_vars))

        # move IR nodes for filter expressions before the reader node
        pq_ind = len(working_body) - i
        new_body = working_body[:pq_ind]
        non_filter_nodes = []
        for i in range(pq_ind, len(working_body)):
            stmt = working_body[i]
            # ignore dataframe filter expression node
            if is_assign(stmt) and stmt.value in filter_nodes:
                non_filter_nodes.append(stmt)
                continue

            stmt_vars = {v.name for v in stmt.list_vars()}
            if stmt_vars & filter_vars:
                new_body.append(stmt)
            else:
                non_filter_nodes.append(stmt)

        # update current basic block with new stmt order
        return new_body + non_filter_nodes

    def _get_filter_nodes(self, index_def, func_ir):
        """find ir.Expr nodes used in filtering output dataframe directly so they can
        be excluded from filter dependency reordering
        """
        # e.g. (df["A"] == 3) | (df["A"] == 4)
        if is_expr(index_def, "binop") and index_def.fn in (
            operator.or_,
            operator.and_,
        ):
            left_nodes = self._get_filter_nodes(
                get_definition(func_ir, index_def.lhs), func_ir
            )
            right_nodes = self._get_filter_nodes(
                get_definition(func_ir, index_def.rhs), func_ir
            )
            return {index_def} | left_nodes | right_nodes
        return {index_def}

    def _get_partition_filters(
        self, index_def, df_var, lhs_def, rhs_def, func_ir, read_node, from_bodosql
    ):
        """get filters for predicate pushdown if possible.
        Returns filters in pyarrow DNF format (e.g. [[("a", "==", 1)][("a", "==", 2)]]):
        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset
        Throws GuardException if not possible.
        """
        require(is_expr(index_def, "binop"))
        # similar to DNF normalization in Sympy:
        # https://github.com/sympy/sympy/blob/da5cd290017814e6100859e5a3f289b3eda4ca6c/sympy/logic/boolalg.py#L1565
        # Or case: call recursively on arguments and concatenate
        # e.g. A or B
        if index_def.fn == operator.or_:
            if is_expr(lhs_def, "binop"):
                l_def = get_definition(func_ir, lhs_def.lhs)
                r_def = get_definition(func_ir, lhs_def.rhs)
                left_or = self._get_partition_filters(
                    lhs_def, df_var, l_def, r_def, func_ir, read_node, from_bodosql
                )
            else:
                require(is_expr(lhs_def, "call"))
                call_list = find_callname(func_ir, lhs_def)
                require(
                    len(call_list) == 2
                    and call_list[0] in ("notna", "isna", "notnull", "isnull")
                    and isinstance(call_list[1], ir.Var)
                )
                colname = self._get_col_name(call_list[1], df_var, func_ir)
                if call_list[0] in ("notna", "notnull"):
                    op = "is not"
                else:
                    op = "is"
                left_or = [[(colname, op, "NULL")]]
            if is_expr(rhs_def, "binop"):
                l_def = get_definition(func_ir, rhs_def.lhs)
                r_def = get_definition(func_ir, rhs_def.rhs)
                right_or = self._get_partition_filters(
                    rhs_def, df_var, l_def, r_def, func_ir, read_node, from_bodosql
                )
            else:
                require(is_expr(rhs_def, "call"))
                call_list = find_callname(func_ir, rhs_def)
                require(
                    len(call_list) == 2
                    and call_list[0] in ("notna", "isna", "notnull", "isnull")
                    and isinstance(call_list[1], ir.Var)
                )
                colname = self._get_col_name(call_list[1], df_var, func_ir)
                if call_list[0] in ("notna", "notnull"):
                    op = "is not"
                else:
                    op = "is"
                right_or = [[(colname, op, "NULL")]]
            return left_or + right_or

        # And case: distribute Or over And to normalize if needed
        if index_def.fn == operator.and_:

            # rhs is Or
            # e.g. "A And (B Or C)" -> "(A And B) Or (A And C)"
            if is_expr(rhs_def, "binop") and rhs_def.fn == operator.or_:
                # lhs And rhs.lhs (A And B)
                new_lhs = ir.Expr.binop(
                    operator.and_, index_def.lhs, rhs_def.lhs, index_def.loc
                )
                new_lhs_rdef = get_definition(func_ir, rhs_def.lhs)
                left_or = self._get_partition_filters(
                    new_lhs,
                    df_var,
                    lhs_def,
                    new_lhs_rdef,
                    func_ir,
                    read_node,
                    from_bodosql,
                )
                # lhs And rhs.rhs (A And C)
                new_rhs = ir.Expr.binop(
                    operator.and_, index_def.lhs, rhs_def.rhs, index_def.loc
                )
                new_rhs_rdef = get_definition(func_ir, rhs_def.rhs)
                right_or = self._get_partition_filters(
                    new_rhs,
                    df_var,
                    lhs_def,
                    new_rhs_rdef,
                    func_ir,
                    read_node,
                    from_bodosql,
                )
                return left_or + right_or

            # lhs is Or
            # e.g. "(B Or C) And A" -> "(B And A) Or (C And A)"
            if is_expr(lhs_def, "binop") and lhs_def.fn == operator.or_:
                # lhs.lhs And rhs (B And A)
                new_lhs = ir.Expr.binop(
                    operator.and_, lhs_def.lhs, index_def.rhs, index_def.loc
                )
                new_lhs_ldef = get_definition(func_ir, lhs_def.lhs)
                left_or = self._get_partition_filters(
                    new_lhs,
                    df_var,
                    new_lhs_ldef,
                    rhs_def,
                    func_ir,
                    read_node,
                    from_bodosql,
                )
                # lhs.rhs And rhs (C And A)
                new_rhs = ir.Expr.binop(
                    operator.and_, lhs_def.rhs, index_def.rhs, index_def.loc
                )
                new_rhs_ldef = get_definition(func_ir, lhs_def.rhs)
                right_or = self._get_partition_filters(
                    new_rhs,
                    df_var,
                    new_rhs_ldef,
                    rhs_def,
                    func_ir,
                    read_node,
                    from_bodosql,
                )
                return left_or + right_or

            # both lhs and rhs are And/literal expressions.
            if is_expr(lhs_def, "binop"):
                l_def = get_definition(func_ir, lhs_def.lhs)
                r_def = get_definition(func_ir, lhs_def.rhs)
                left_or = self._get_partition_filters(
                    lhs_def, df_var, l_def, r_def, func_ir, read_node, from_bodosql
                )
            else:
                require(is_expr(lhs_def, "call"))
                call_list = find_callname(func_ir, lhs_def)
                require(
                    len(call_list) == 2
                    and call_list[0] in ("notna", "isna", "notnull", "isnull")
                    and isinstance(call_list[1], ir.Var)
                )
                colname = self._get_col_name(call_list[1], df_var, func_ir)
                if call_list[0] in ("notna", "notnull"):
                    op = "is not"
                else:
                    op = "is"
                left_or = [[(colname, op, "NULL")]]
            if is_expr(rhs_def, "binop"):
                l_def = get_definition(func_ir, rhs_def.lhs)
                r_def = get_definition(func_ir, rhs_def.rhs)
                right_or = self._get_partition_filters(
                    rhs_def, df_var, l_def, r_def, func_ir, read_node, from_bodosql
                )
            else:
                require(is_expr(rhs_def, "call"))
                call_list = find_callname(func_ir, rhs_def)
                require(
                    len(call_list) == 2
                    and call_list[0] in ("notna", "isna", "notnull", "isnull")
                    and isinstance(call_list[1], ir.Var)
                )
                colname = self._get_col_name(call_list[1], df_var, func_ir)
                if call_list[0] in ("notna", "notnull"):
                    op = "is not"
                else:
                    op = "is"
                right_or = [[(colname, op, "NULL")]]

            # If either expression is an AND, we may still have ORs inside
            # the AND. As a result, distributed ANDs across all ORs.
            # For example
            # ((A | B) & C) & D -> (AC | BC) & D (via the recursion)
            # Now we need to produce (AC | BC) & D -> (ACD | BCD)
            filters = []
            for left_or_cond in left_or:
                for right_or_cond in right_or:
                    filters.append(left_or_cond + right_or_cond)
            return filters

        # literal case
        # TODO(ehsan): support 'in' and 'not in'
        is_sql = isinstance(read_node, bodo.ir.sql_ext.SqlReader)
        op_map = {
            operator.eq: "=" if is_sql else "==",
            operator.ne: "<>" if is_sql else "!=",
            operator.lt: "<",
            operator.le: "<=",
            operator.gt: ">",
            operator.ge: ">=",
        }

        # Operator mapping used to support situations
        # where the column is on the RHS. Since Pyarrow
        # format is ("col", op, scalar), we must invert certain
        # operators.
        right_colname_op_map = {
            operator.eq: "=" if is_sql else "==",
            operator.ne: "<>" if is_sql else "!=",
            operator.lt: ">",
            operator.le: ">=",
            operator.gt: "<",
            operator.ge: "<=",
        }

        require(index_def.fn in op_map)
        left_colname = guard(self._get_col_name, index_def.lhs, df_var, func_ir)
        right_colname = guard(self._get_col_name, index_def.rhs, df_var, func_ir)

        require(
            (left_colname and not right_colname) or (right_colname and not left_colname)
        )
        if right_colname:
            cond = (right_colname, right_colname_op_map[index_def.fn], index_def.lhs)
        else:
            cond = (left_colname, op_map[index_def.fn], index_def.rhs)

        # If this is parquet we need to verify this is a filter we can process.
        # TODO(Nick): Support for BodoSQL, which we can't yet because we don't
        # have typemap information (requires refactoring filter pushdown in BodoSQL).
        if not is_sql and not from_bodosql:
            lhs_arr_typ = read_node.original_out_types[
                read_node.original_df_colnames.index(cond[0])
            ]
            lhs_scalar_typ = bodo.utils.typing.element_type(lhs_arr_typ)
            require(cond[2].name in self.typemap)
            rhs_scalar_typ = self.typemap[cond[2].name]
            # Only apply filter pushdown if its safe to use inside arrow
            require(
                bodo.utils.typing.is_common_scalar_dtype(
                    [lhs_scalar_typ, rhs_scalar_typ]
                )
                or bodo.utils.typing.is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ)
            )

        # Pyarrow format, e.g.: [[("a", "==", 2)]]
        return [[cond]]

    def _get_col_name(self, var, df_var, func_ir):
        """get column name for dataframe column access like df["A"] if possible.
        Throws GuardException if not possible.
        """
        var_def = get_definition(func_ir, var)
        if is_expr(var_def, "getattr") and var_def.value.name == df_var.name:
            return var_def.attr
        if is_expr(var_def, "static_getitem") and var_def.value.name == df_var.name:
            return var_def.index
        # handle case with calls like df["A"].astype(int) > 2
        if is_call(var_def):
            fdef = find_callname(func_ir, var_def)
            # calling pd.to_datetime() on a string column is possible since pyarrow
            # matches the data types before filter comparison (in this case, calls
            # pd.Timestamp on partiton's string value)
            if fdef == ("to_datetime", "pandas"):
                # We don't want to perform filter pushdown if there is a format argument
                # i.e. pd.to_datetime(col, format="%Y-%d-%m")
                # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
                require((len(var_def.args) == 1) and not var_def.kws)
                return self._get_col_name(var_def.args[0], df_var, func_ir)
            require(
                isinstance(fdef, tuple)
                and len(fdef) == 2
                and isinstance(fdef[1], ir.Var)
            )
            return self._get_col_name(fdef[1], df_var, func_ir)

        require(is_expr(var_def, "getitem"))
        require(var_def.value.name == df_var.name)
        return get_const_value_inner(func_ir, var_def.index, arg_types=self.args)

    def _run_setitem(self, inst, label):
        target_typ = self.typemap.get(inst.target.name, None)
        nodes = []
        idx_var = get_getsetitem_index_var(inst, self.typemap, nodes)
        idx_typ = self.typemap.get(idx_var.name, None)

        # df["B"] = A
        if isinstance(target_typ, DataFrameType):
            return self._run_setitem_df(
                inst, target_typ, idx_typ, idx_var, nodes, label
            )

        # transform df.loc[cond, "A"] setitem case here since it may require type change
        if (
            isinstance(target_typ, DataFrameLocType)
            and isinstance(idx_typ, types.BaseTuple)
            and len(idx_typ.types) == 2
        ):
            return self._run_setitem_df_loc(
                inst, target_typ, idx_typ, idx_var, nodes, label
            )

        # transform df.iloc[cond, 1] setitem case here since it may require type change
        if (
            isinstance(target_typ, DataFrameILocType)
            and isinstance(idx_typ, types.BaseTuple)
            and len(idx_typ.types) == 2
        ):
            return self._run_setitem_df_iloc(
                inst, target_typ, idx_typ, idx_var, nodes, label
            )

        return nodes + [inst]

    def _run_setitem_df(self, inst, target_typ, idx_typ, idx_var, nodes, label):
        """transform df setitem nodes, e.g. df["B"] = 3"""
        idx_const = guard_const(
            get_const_value_inner,
            self.func_ir,
            idx_var,
            self.arg_types,
            self.typemap,
        )
        if idx_const is None:
            self._require_const[idx_var] = label
            return nodes + [inst]

        # single column case like df["A"] = 3
        if not isinstance(idx_const, (tuple, list, np.ndarray, pd.Index)):
            return nodes + self._run_df_set_column(inst, idx_const, label)

        nodes += self._gen_df_setitem_full_column(inst, inst.target, idx_const, label)
        self.changed = True
        return nodes

    def _run_setitem_df_loc(self, inst, target_typ, idx_typ, idx_var, nodes, label):
        """transform df.loc setitem nodes, e.g. df.loc[:, "B"] = 3"""

        col_inds, row_ind = self._get_loc_indices(idx_var, label, inst.loc)

        # couldn't find column name values, just return to be handled later
        if col_inds is None:
            return nodes + [inst]

        # NOTE: dataframe type may have changed in typing pass (e.g. due to df setitem)
        # so we shouldn't use target_typ and should check for the actual df variable
        df_var = self._get_loc_df_var(inst.target)
        df_type = self.typemap.get(df_var.name, None)
        if df_type is None:
            return nodes + [inst]

        # get column names if bool list
        if len(col_inds) > 0 and isinstance(col_inds[0], (bool, np.bool_)):
            col_inds = list(pd.Series(df_type.columns, dtype=object)[col_inds])

        # if setting full columns
        if row_ind == slice(None):
            nodes += self._gen_df_setitem_full_column(inst, df_var, col_inds, label)
            self.changed = True
            return nodes

        # avoid transform if selected columns not all in dataframe schema
        # may require schema change, see test_loc_setitem (impl6)
        if not all(c in df_type.columns for c in col_inds):
            nodes.append(inst)
            return nodes

        self.changed = True
        func_text = "def impl(I, idx, value):\n"
        func_text += "  df = I._obj\n"
        for c in col_inds:
            c_idx = df_type.columns.index(c)
            func_text += f"  bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_idx})[idx[0]] = value\n"

        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]
        return nodes + compile_func_single_block(
            impl, [inst.target, idx_var, inst.value], None
        )

    def _run_setitem_df_iloc(self, inst, target_typ, idx_typ, idx_var, nodes, label):
        """transform df.iloc setitem nodes, e.g. df.loc[:, 1] = 3"""

        col_inds, row_ind = self._get_loc_indices(idx_var, label, inst.loc)

        # couldn't find column name values, just return to be handled later
        if col_inds is None:
            return nodes + [inst]

        df_var = self._get_loc_df_var(inst.target)
        df_type = self.typemap.get(df_var.name, None)
        if df_type is None:
            return nodes + [inst]

        # if setting full columns
        if row_ind == slice(None):
            col_names = [df_type.columns[c_ind] for c_ind in col_inds]
            nodes += self._gen_df_setitem_full_column(inst, df_var, col_names, label)
            self.changed = True
            return nodes

        self.changed = True
        func_text = "def impl(I, idx, value):\n"
        func_text += "  df = I._obj\n"
        for c_idx in col_inds:
            func_text += f"  bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_idx})[idx[0]] = value\n"

        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]
        return nodes + compile_func_single_block(
            impl, [inst.target, idx_var, inst.value], None
        )

    def _get_loc_indices(self, idx_var, label, loc):
        """get row/column index values for df.loc/df.iloc if possible"""
        # get column index var
        tup_list = guard(find_build_tuple, self.func_ir, idx_var)
        if tup_list is None or len(tup_list) != 2:  # pragma: no cover
            raise BodoError("Invalid df.loc[ind,ind] case")
        row_ind_var = tup_list[0]
        col_ind_var = tup_list[1]

        # try to find index values
        try:
            err_msg = "df.loc/iloc[] requires constant column names"
            col_inds = self._get_const_value(col_ind_var, label, loc, err_msg)
        except (GuardException, BodoConstUpdatedError):
            col_inds = None

        # normalize single column name to list
        if not isinstance(col_inds, (list, tuple, np.ndarray)):
            col_inds = [col_inds]

        # try to find index values
        # NOTE: not using _get_const_value() since constant isn't fully necessary
        try:
            row_ind = get_const_value_inner(
                self.func_ir,
                row_ind_var,
                self.arg_types,
                self.typemap,
                self._updated_containers,
            )
        except (GuardException, BodoConstUpdatedError):
            row_ind = None

        return col_inds, row_ind

    def _get_loc_df_var(self, target):
        """get dataframe variable from df.loc/iloc nodes.
        just gets the definition of the node (assuming no unusual control flow).
        """
        loc_def = guard(get_definition, self.func_ir, target)
        if not is_expr(loc_def, "getattr"):  # pragma: no cover
            raise BodoError("Invalid df.loc/iloc[] setitem")
        return loc_def.value

    def _get_bodosql_ctx_name_df_typs(self, sql_context_var):  # pragma: no cover
        """
        Extracts the names/types of the dataframes used to intialize the bodosql context.
        This is converted into a tuple of the names/values in
        untyped pass (see _handle_bodosql_BodoSQLContext).
        This function extracts the dataframe types directly from the IR, to avoid any issues
        with incorrect type propogation (specifically, from df setitem).
        """
        sql_ctx_def = guard(get_definition, self.func_ir, sql_context_var)
        df_dict_var = sql_ctx_def.args[0]
        df_dict_def = guard(get_definition, self.func_ir, df_dict_var)
        df_dict_def_items = df_dict_def.items
        # floor divide
        split_idx = (len(df_dict_def_items) // 2) + 1
        # ommit first value, as it is a dummy
        df_name_vars, df_vars = (
            df_dict_def_items[1:split_idx],
            df_dict_def_items[split_idx:],
        )
        df_name_typs = tuple(
            [
                self.typemap.get(df_name_var.name, None).literal_value
                for df_name_var in df_name_vars
            ]
        )
        df_typs = tuple([self.typemap.get(df_var.name, None) for df_var in df_vars])

        return df_name_typs, df_typs

    def _gen_df_setitem_full_column(self, inst, df_var, col_inds, label):
        """Generate code for setitem of df.loc/iloc when setting full columns"""
        nodes = []
        loc = inst.loc
        # value to set could be a scalar or a DataFrame
        val = inst.value
        val_type = self.typemap.get(val.name, None)
        column_values = [val] * len(col_inds)

        # setting multiple columns using a dataframe
        if isinstance(val_type, DataFrameType):
            # get dataframe data arrays to set
            for i in range(len(col_inds)):
                func = eval(
                    "lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, _i)"
                )
                nodes += compile_func_single_block(
                    func, [val], None, extra_globals={"_i": i}
                )
                column_values[i] = nodes[-1].target
        # setting multiple columns using a 2D array
        elif isinstance(val_type, types.Array) and val_type.ndim == 2:
            # get data columns from 2D array
            for i in range(len(col_inds)):
                func = eval("lambda A: np.ascontiguousarray(A[:,_i])")
                nodes += compile_func_single_block(
                    func, [val], None, extra_globals={"_i": i}
                )
                column_values[i] = nodes[-1].target

        for i, c in enumerate(col_inds):
            # setting up definitions and rhs_labels is necessary for
            # _run_df_set_column() to work properly. Needs to be done here since
            # nodes have not gone through the main IR loop in run()
            if i > 0:
                df_expr = nodes[-2].value
                df_var = nodes[-1].target
                self.func_ir._definitions[df_var.name].append(df_expr)
                self.rhs_labels[df_expr] = label
            dummy_inst = ir.SetItem(df_var, df_var, column_values[i], loc)
            nodes += self._run_df_set_column(dummy_inst, c, label)
            # clean up to avoid conflict with later definition update in run()
            if i > 0:
                self.func_ir._definitions[df_var.name].remove(df_expr)
        return nodes

    def _run_setattr(self, inst, label):
        """handle ir.SetAttr node"""
        target_typ = self.typemap.get(inst.target.name, None)

        # another transformation pass is necessary to avoid errors since there is no
        # overload for setattr to catch errors (see test_set_df_column_names::impl5)
        if target_typ == types.unknown:
            self.needs_transform = True

        # DataFrame.attr = val
        if isinstance(target_typ, DataFrameType):
            # df.B = A transform
            # Pandas only allows setting existing columns using setattr
            if inst.attr in target_typ.columns:
                return self._run_df_set_column(inst, inst.attr, label)
            # transform df.columns = new_names
            # creates a new dataframe and replaces the old variable, only possible if
            # df.columns dominates the df creation due to type stability
            if inst.attr == "columns":
                # try to find new column names
                try:
                    err_msg = "Setting dataframe columns requires constant names"
                    columns = self._get_const_value(
                        inst.value, label, inst.loc, err_msg
                    )
                except (GuardException, BodoConstUpdatedError):
                    return [inst]

                # check number of column names
                if len(columns) != len(target_typ.columns):
                    raise BodoError(
                        "DataFrame.columns: number of new column names does not match number of existing columns"
                    )

                # check control flow error
                df_var = inst.target
                err_msg = "DataFrame.columns: setting dataframe column names"
                self._error_on_df_control_flow(df_var, label, err_msg)

                # create output df
                self.changed = True
                data_outs = ", ".join(
                    f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
                    for i in range(len(columns))
                )
                header = "def impl(df):\n"
                impl = bodo.hiframes.dataframe_impl._gen_init_df(
                    header, columns, data_outs
                )
                nodes = compile_func_single_block(impl, [df_var], None, self)
                self.replace_var_dict[df_var.name] = nodes[-1].target
                return nodes

            # transform df.index = new_index
            # creates a new dataframe and replaces the old variable, only possible if
            # df.index dominates the df creation due to type stability
            if inst.attr == "index":

                # check control flow error
                df_var = inst.target
                err_msg = "DataFrame.index: setting dataframe index"
                self._error_on_df_control_flow(df_var, label, err_msg)

                # create output df
                self.changed = True
                data_outs = ", ".join(
                    f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
                    for i in range(len(target_typ.columns))
                )
                header = "def impl(df, new_index):\n"
                # convert to Index type if necessary
                if bodo.hiframes.pd_index_ext.is_index_type(
                    self.typemap.get(inst.value.name, None)
                ):
                    index = "new_index"
                else:
                    index = "bodo.utils.conversion.index_from_array(bodo.utils.conversion.coerce_to_array(new_index, scalar_to_arr_len=len(df)))"
                impl = bodo.hiframes.dataframe_impl._gen_init_df(
                    header, target_typ.columns, data_outs, index
                )
                nodes = compile_func_single_block(
                    impl, [df_var, inst.value], None, self
                )
                self.replace_var_dict[df_var.name] = nodes[-1].target
                return nodes

        # Series.index = arr
        if isinstance(target_typ, SeriesType) and inst.attr == "index":
            # check control flow error
            series_var = inst.target
            err_msg = "Series.index: setting dataframe index"
            self._error_on_df_control_flow(series_var, label, err_msg)

            # create output Series
            self.changed = True
            func_text = "def impl(S, new_index):\n"
            func_text += "  data = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
            func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
            # convert to Index type if necessary
            if bodo.hiframes.pd_index_ext.is_index_type(
                self.typemap.get(inst.value.name, None)
            ):
                index = "new_index"
            else:
                index = "bodo.utils.conversion.index_from_array(bodo.utils.conversion.coerce_to_array(new_index, scalar_to_arr_len=len(S)))"
            func_text += (
                f"  return bodo.hiframes.pd_series_ext.init_series(data, {index}, name)"
            )
            loc_vars = {}
            exec(
                func_text,
                {
                    "bodo": bodo,
                },
                loc_vars,
            )
            impl = loc_vars["impl"]
            nodes = compile_func_single_block(
                impl, [series_var, inst.value], None, self
            )
            self.replace_var_dict[series_var.name] = nodes[-1].target
            return nodes

        return [inst]

    def _run_call(self, assign, rhs, label):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:  # pragma: no cover
            # TODO: test coverage
            return [assign]

        func_name, func_mod = fdef

        if func_mod == "pandas":
            return self._run_call_pd_top_level(assign, rhs, func_name, label)

        # handle df.method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DataFrameType
        ):
            return self._run_call_dataframe(assign, rhs, func_mod, func_name, label)

        # handle Series.method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), SeriesType
        ):
            return self._run_call_series(assign, rhs, func_mod, func_name, label)

        # handle df.groupby().method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DataFrameGroupByType
        ):
            return self._run_call_df_groupby(assign, rhs, func_mod, func_name, label)

        # handle BodoSQLContextType.sql() calls here since the generated code cannot
        # be handled in regular overloads (requires Bodo's untyped pass, typing pass)
        #
        # Note we delay checking BodoSQLContextType until we find a possible match
        # to avoid paying the import overhead for Bodo calls with no BodoSQL.
        if isinstance(func_mod, ir.Var) and func_name in (
            "sql",
            "_test_sql_unoptimized",
            "convert_to_pandas",
        ):  # pragma: no cover

            # Try import BodoSQL and check the type
            try:  # pragma: no cover
                from bodosql.context_ext import BodoSQLContextType
            except:
                # workaround: something that makes isinstance(type, BodoSQLContextType) always false
                BodoSQLContextType = int

            if isinstance(
                self._get_method_obj_type(func_mod, rhs.func), BodoSQLContextType
            ):
                return self._run_call_bodosql_sql(
                    assign, rhs, func_mod, func_name, label
                )

        # handle BodoSQLTablePathType
        if fdef == ("TablePath", "bodosql"):
            # Force table path arguments to be literals if passed to the function.
            return self._run_call_bodosql_table_path(assign, rhs, label)

        # throw proper error when calling a non-JIT function
        if isinstance(
            self.typemap.get(rhs.func.name, None), bodo.utils.typing.FunctionLiteral
        ):
            func_name = "unknown"
            try:
                func_name = self.typemap[rhs.func.name].literal_value.__name__
            except:  # pragma: no cover
                pass
            raise BodoError(
                f"Cannot call non-JIT function '{func_name}' from JIT function (convert to JIT or use objmode).",
                rhs.loc,
            )

        return [assign]

    def _run_call_bodosql_table_path(self, assign, rhs, label):
        nodes = []
        func_args = [(0, "file_path"), (1, "file_type")]
        nodes += self._replace_arg_with_literal(
            "bodosql.TablePath", rhs, func_args, label
        )
        return nodes + [assign]

    def _run_binop(self, assign, rhs):
        arg1_typ = self.typemap.get(rhs.lhs.name, None)
        arg2_typ = self.typemap.get(rhs.rhs.name, None)
        target_typ = self.typemap.get(assign.target.name, None)

        return [assign]

    def _run_call_dataframe(self, assign, rhs, df_var, func_name, label):
        """Handle dataframe calls that need transformation to meet Bodo requirements"""
        lhs = assign.target
        nodes = []

        # find constant values for function arguments that require constants, and
        # replace the argument variable with a new variable with literal type
        # that holds the constants to enable constant access in overloads. This may
        # force some jit function arguments to be literal if required.
        # mapping of df functions to their arguments that require constant values:
        df_call_const_args = {
            "groupby": [(0, "by"), (3, "as_index")],
            "merge": [
                (1, "how"),
                (2, "on"),
                (3, "left_on"),
                (4, "right_on"),
                (5, "left_index"),
                (6, "right_index"),
                (8, "suffixes"),
            ],
            "sort_values": [
                (0, "by"),
                (2, "ascending"),
                (3, "inplace"),
                (5, "na_position"),
            ],
            "join": [
                (1, "on"),
                (2, "how"),
                (3, "lsuffix"),
                (4, "rsuffix"),
            ],
            "rename": [(0, "mapper"), (2, "columns")],
            "drop": [
                (0, "labels"),
                (1, "axis"),
                (3, "columns"),
                (5, "inplace"),
            ],
            "dropna": [
                (0, "axis"),
                (1, "how"),
                (3, "subset"),
            ],
            "astype": [
                (0, "dtype"),
                (1, "copy"),
            ],
            "select_dtypes": [(0, "include"), (1, "exclude")],
            "apply": [(0, "func"), (1, "axis")],
            "to_parquet": [(4, "partition_cols")],
            "insert": [(0, "loc"), (1, "column"), (3, "allow_duplicates")],
            "fillna": [(1, "method")],
        }

        if func_name in df_call_const_args:
            func_args = df_call_const_args[func_name]
            # function arguments are typed as pyobject initially, literalize if possible
            pyobject_to_literal = func_name == "apply"
            nodes += self._replace_arg_with_literal(
                func_name, rhs, func_args, label, pyobject_to_literal
            )

        # transform df.assign() here since (**kwargs) is not supported in overload
        if func_name == "assign":
            return nodes + self._handle_df_assign(assign.target, rhs, df_var, assign)

        # transform df.insert() here since it updates the dataframe inplace
        if func_name == "insert":
            return nodes + self._handle_df_insert(
                assign.target, rhs, df_var, assign, label
            )

        # handle calls that have inplace=True that changes the schema, by replacing the
        # dataframe variable instead of inplace change if possible
        # TODO: handle all necessary df calls
        # map call name to the position of its 'inplace' argument
        df_inplace_call_arg_no = {
            "drop": 5,
            "sort_values": 3,
            "rename": 5,
            "reset_index": 2,
        }
        # call needs handling if not already transformed (avoid infinite loop)
        if func_name in df_inplace_call_arg_no and not self._is_df_call_transformed(
            rhs
        ):
            kws = dict(rhs.kws)
            inplace_arg_no = df_inplace_call_arg_no[func_name]
            inplace_var = get_call_expr_arg(
                func_name, rhs.args, kws, inplace_arg_no, "inplace", ""
            )
            return nodes + self._handle_df_inplace_func(
                assign, lhs, rhs, df_var, inplace_var, label, func_name
            )

        # convert const list to tuple for better optimization
        if func_name == "append":
            self._call_arg_list_to_tuple(rhs, "append", 0, "other", nodes)

        return nodes + [assign]

    def _df_assign_non_lambda_helper(self, lhs, kws_key_val_list, df_var, assign):
        """
        Helper functipn for df.assign. kws_key_val_list is a list of (colname, val), where all values are non lambda/JIT functions.
        Generates returns the assign nodes equivalent to df_var.assign(colname_1 = val_1, colname_2 = val_2 ... etc) by
        generating a single dataframe init.
        """

        kws_val_list = [val for (key, val) in kws_key_val_list]
        kws_key_list = [key for (key, val) in kws_key_val_list]

        df_type = self.typemap.get(df_var.name, None)
        # cannot transform yet if dataframe type is not available yet
        if df_type is None:
            return [assign]
        additional_columns = tuple(kws_key_list)
        previous_columns = set(df_type.columns)
        # columns below are preserved
        preserved_columns = previous_columns - set(additional_columns)
        name_col_total = []
        data_col_total = []

        # preserve original ordering of any columns that were already present
        # in the original dataframe
        for c in df_type.columns:
            if c in preserved_columns:
                name_col_total.append(c)
                data_col_total.append("df['{}'].values".format(c))
            elif c in additional_columns:
                name_col_total.append(c)
                e_col = "bodo.utils.conversion.coerce_to_array(new_arg{}, scalar_to_arr_len=len(df))".format(
                    kws_key_list.index(c)
                )
                data_col_total.append(e_col)

        # The new columns should be added in the order that they apear in kws_key_val_list
        for i, c in enumerate(kws_key_list):
            if c not in df_type.columns:
                name_col_total.append(c)
                e_col = "bodo.utils.conversion.coerce_to_array(new_arg{}, scalar_to_arr_len=len(df))".format(
                    i
                )
                data_col_total.append(e_col)

        data_args = ", ".join(data_col_total)
        header = "def impl(df, {}):\n".format(
            ", ".join("new_arg{}".format(i) for i in range(len(kws_key_val_list)))
        )
        impl = bodo.hiframes.dataframe_impl._gen_init_df(
            header, tuple(name_col_total), data_args
        )

        self.changed = True
        return compile_func_single_block(impl, [df_var] + kws_val_list, lhs)

    def _handle_df_assign(self, lhs, rhs, df_var, assign):
        """replace df.assign() with its implementation to avoid overload errors with
        (**kwargs)
        """
        # In the common case where we have no lambda functions, we can reduce df.assign
        # into a single init dataframe. However, the semantics with lambda/JIT function are a bit janky,

        # let's say we have some df = ({"B": [1,2,3]})
        # if you do df.assign(A = lambda x: x["B"], B = 3, C = lambda x: x["B"]),
        # then A = [1,2,3], C = [3,3,3]
        #
        # worst case you could have somthing like
        #   df.assign(A = ..., B = lambda x: x["A"] + ..., C = lambda x: x["B"] * ...)
        # where each assignment depends on all of the previous assignments.
        # for the sake of pragmatism, for all the arguments preceding the first lambda/JIT function,
        # we handle through the faster dataframe init. For all the arguments following/including
        # the first lambda/JIT function, we handle as a sequence of dataframe assigns.

        df_type = self.typemap.get(df_var.name, None)
        # cannot transform yet if dataframe type is not available yet
        if df_type is None:
            return [assign]
        # cannot transform yet if argument type is not available yet
        for (_, kw_val) in rhs.kws:
            if self.typemap.get(kw_val.name, None) is None:
                return [assign]

        kws_list = list(rhs.kws)

        # create the list of arguments for which we use the optimized codepath,
        # keeping track of the index of the first encountered lambda/JIT function
        first_lambda_idx = 0
        non_lambda_fns = []
        for i, val in enumerate(kws_list):
            (_, kw_val) = val
            if not isinstance(self.typemap.get(kw_val.name), types.Dispatcher):
                non_lambda_fns.append(val)
                first_lambda_idx = i + 1
            else:
                break

        if len(non_lambda_fns) == len(kws_list):
            # If we have no lambda/JIT functions, we're finished
            return self._df_assign_non_lambda_helper(
                lhs, non_lambda_fns, df_var, assign
            )
        else:
            # else, create a temporary df_var to store the output,
            # and update initial_nodes/cur_df_var as appropriate
            copied_df_var = apply_fn_var = ir.Var(
                assign.target.scope, mk_unique_var("copied_df"), rhs.loc
            )
            initial_nodes = self._df_assign_non_lambda_helper(
                copied_df_var, non_lambda_fns, df_var, assign
            )
            cur_df_var = copied_df_var

        # load the setitem global, which will be used for the rest of this function
        setitem_fn_var = ir.Var(
            assign.target.scope, mk_unique_var("setitem_fn_var"), rhs.loc
        )
        setitem_fn_var_assign = ir.Assign(
            ir.Global(
                "set_df_col", bodo.hiframes.dataframe_impl.set_df_col, loc=rhs.loc
            ),
            setitem_fn_var,
            rhs.loc,
        )

        new_assign_list = initial_nodes + [setitem_fn_var_assign]

        # For each of the arguments not already handled in the optimized codepath,
        # Perform the sequence of setitems on the copied dataframe,
        # updating cur_df_var each iteration.
        for (kw_name, kw_val) in kws_list[first_lambda_idx:]:

            if isinstance(self.typemap.get(kw_val.name), types.Dispatcher):
                # handles lambda fns, and passed JIT functions
                # put the setitem value is as the output of an apply on the current dataframe

                # assign the apply functon to a variable
                apply_fn_var = ir.Var(
                    assign.target.scope, mk_unique_var("df_apply_fn"), rhs.loc
                )
                apply_fn_var_assign = ir.Assign(
                    ir.Expr.getattr(cur_df_var, "apply", rhs.loc),
                    apply_fn_var,
                    rhs.loc,
                )

                # create the axis variable
                axis_var = ir.Var(
                    assign.target.scope, mk_unique_var("axis_var"), rhs.loc
                )
                axis_assign = ir.Assign(
                    ir.Const(1, rhs.loc),
                    axis_var,
                    rhs.loc,
                )

                # call the apply function on the make function, with kwds = {"axis": 1}
                apply_call_args = [kw_val]
                apply_fn_call = ir.Expr.call(apply_fn_var, apply_call_args, (), rhs.loc)
                apply_fn_call.kws = (("axis", axis_var),)

                # assign the output of the val to a variable
                apply_output_var = ir.Var(
                    assign.target.scope, mk_unique_var("df_output_var"), rhs.loc
                )
                apply_output_assign = ir.Assign(
                    apply_fn_call, apply_output_var, rhs.loc
                )

                setitem_val = apply_output_var

                new_assign_list.extend(
                    [
                        axis_assign,
                        apply_fn_var_assign,
                        apply_output_assign,
                    ]
                )
            else:
                # handles non lambda/JIT fns
                # In this case, we just do a the setitem value is just the passed in value
                setitem_val = kw_val

            # make the colname variable
            colname_var = ir.Var(
                assign.target.scope, mk_unique_var("col_name"), rhs.loc
            )
            colname_var_assign = ir.Assign(
                ir.Const(kw_name, rhs.loc),
                colname_var,
                rhs.loc,
            )

            # set_df_col(df, cname, arr, inplace)
            inplace_var = ir.Var(
                assign.target.scope, mk_unique_var("col_name"), rhs.loc
            )
            inplace_var_assign = ir.Assign(
                ir.Const(False, rhs.loc),
                inplace_var,
                rhs.loc,
            )

            setitem_args = [
                cur_df_var,
                colname_var,
                setitem_val,
                inplace_var,
            ]
            # assign the value to a new output_df_var
            new_df_var = ir.Var(
                assign.target.scope, mk_unique_var("output_df_var"), rhs.loc
            )

            setitem_call = ir.Expr.call(setitem_fn_var, setitem_args, (), rhs.loc)

            setitem_output_assign = ir.Assign(setitem_call, new_df_var, rhs.loc)
            cur_df_var = new_df_var

            new_assign_list.extend(
                [
                    inplace_var_assign,
                    colname_var_assign,
                    setitem_output_assign,
                ]
            )

        self.needs_transform = True
        self.changed = True
        # create a final assign to the output variable
        # probably can do this without compiling this fn, see BE-1564

        func_text = "def impl(df):\n"
        func_text += "  return df"

        loc_vars = {}
        exec(func_text, dict(), loc_vars)
        impl = loc_vars["impl"]

        # Don't pass the typing ctx, as many of the newly created variables won't be typed yet.
        return new_assign_list + compile_func_single_block(impl, [cur_df_var], lhs)

    def _handle_df_insert(self, lhs, rhs, df_var, assign, label):
        """replace df.insert() here since it changes dataframe type inplace"""

        err_msg = "DataFrame.insert(): setting a new dataframe column inplace"
        self._error_on_df_control_flow(df_var, label, err_msg)

        kws = dict(rhs.kws)
        loc_var = get_call_expr_arg("insert", rhs.args, kws, 0, "loc")
        column_var = get_call_expr_arg("insert", rhs.args, kws, 1, "column")
        value_var = get_call_expr_arg("insert", rhs.args, kws, 2, "value")
        allow_duplicates_var = get_call_expr_arg(
            "insert", rhs.args, kws, 3, "allow_duplicates", ""
        )

        df_type = self.typemap.get(df_var.name, None)
        loc_type = self.typemap.get(loc_var.name, None)
        column_type = self.typemap.get(column_var.name, None)
        allow_duplicates_type = (
            types.BooleanLiteral(False)
            if allow_duplicates_var == ""
            else self.typemap.get(allow_duplicates_var.name, None)
        )
        # cannot transform yet if input types are not available yet
        if (
            df_type is None
            or loc_type is None
            or column_type is None
            or allow_duplicates_type is None
        ):
            return [assign]

        loc, column = self._err_check_df_insert_args(
            df_type, loc_type, column_type, allow_duplicates_type, lhs.loc
        )

        # raise warning if df is an argument and update inplace may be necessary
        df_def = guard(get_definition, self.func_ir, df_var)
        # TODO: consider dataframe alias cases where definition is not directly ir.Arg
        # but dataframe has a parent object
        if isinstance(df_def, ir.Arg):
            warnings.warn(
                BodoWarning(
                    "df.insert(): input dataframe is passed as argument to JIT function, but Bodo does not update it for the caller since the data type changes"
                )
            )

        new_columns = list(df_type.columns)
        new_columns.insert(loc, column)

        out_data = [
            f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
            for i in range(len(df_type.columns))
        ]
        out_data.insert(loc, "new_arr")
        out_data = ", ".join(out_data)

        header = "def impl(df, value):\n"
        header += "  new_arr = bodo.utils.conversion.coerce_to_array(value, scalar_to_arr_len=len(df))\n"
        impl = bodo.hiframes.dataframe_impl._gen_init_df(
            header, tuple(new_columns), out_data
        )
        self.changed = True

        nodes = compile_func_single_block(impl, [df_var, value_var], None, self)
        self.replace_var_dict[df_var.name] = nodes[-1].target
        # output of 'insert' is just None
        nodes.append(ir.Assign(ir.Const(None, lhs.loc), lhs, lhs.loc))
        return nodes

    def _err_check_df_insert_args(
        self, df_type, loc_type, column_type, allow_duplicates_type, var_loc
    ):
        """error check df.insert() arguments and return the necessary constant values"""
        if not is_overload_constant_int(loc_type):
            raise BodoError("df.insert(): 'loc' should be a constant integer", var_loc)

        if not is_literal_type(column_type):
            raise BodoError("df.insert(): 'column' should be a constant", var_loc)

        if not is_overload_constant_bool(allow_duplicates_type):
            raise BodoError(
                "df.insert(): 'allow_duplicates' should be a constant boolean", var_loc
            )

        loc = get_overload_const_int(loc_type)
        column = get_literal_value(column_type)
        allow_duplicates = get_overload_const_bool(allow_duplicates_type)

        if column in df_type.columns and not allow_duplicates:
            raise BodoError(
                f"df.insert(): cannot insert {column}, already exists", var_loc
            )

        return loc, column

    def _run_call_df_groupby(self, assign, rhs, groupby_var, func_name, label):
        """Handle dataframe groupby calls that need transformation to meet Bodo
        requirements
        """
        nodes = []

        # mapping of groupby functions to their arguments that require constant values
        groupby_call_const_args = {
            "agg": [(0, "func")],
            "aggregate": [(0, "func")],
        }

        if func_name in groupby_call_const_args:
            func_args = groupby_call_const_args[func_name]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)

        return nodes + [assign]

    def _run_call_series(self, assign, rhs, series_var, func_name, label):
        """Handle Series calls that need transformation to meet Bodo requirements"""
        nodes = []

        # convert const list to tuple for better optimization
        if func_name == "append":
            self._call_arg_list_to_tuple(rhs, "append", 0, "to_append", nodes)

        # mapping of Series functions to their arguments that require constant values
        series_call_const_args = {
            "map": [(0, "arg")],
            "apply": [(0, "func")],
            "to_frame": [(0, "name")],
            "value_counts": [
                (0, "normalize"),
                (1, "sort"),
            ],
            "astype": [
                (0, "dtype"),
            ],
            "fillna": [(1, "method")],
        }

        if func_name in series_call_const_args:
            # Series.map with dict input doesn't need constant arg
            if func_name == "map":
                var = get_call_expr_arg("map", rhs.args, dict(rhs.kws), 0, "arg")
                if isinstance(self.typemap.get(var.name, None), types.DictType):
                    return nodes + [assign]

            func_args = series_call_const_args[func_name]
            # function arguments are typed as pyobject initially, literalize if possible
            pyobject_to_literal = func_name in ("map", "apply")
            nodes += self._replace_arg_with_literal(
                func_name, rhs, func_args, label, pyobject_to_literal
            )

        return nodes + [assign]

    def _run_call_pd_top_level(self, assign, rhs, func_name, label):
        """transform top-level pandas functions"""
        if func_name == "Series":
            return self._run_call_pd_series(assign, rhs, func_name, label)
        nodes = []

        # mapping of pandas functions to their arguments that require constant values
        top_level_call_const_args = {
            "DataFrame": [(2, "columns")],
            "merge": [
                (2, "how"),
                (3, "on"),
                (4, "left_on"),
                (5, "right_on"),
                (6, "left_index"),
                (7, "right_index"),
                (9, "suffixes"),
            ],
            "merge_asof": [
                (2, "on"),
                (3, "left_on"),
                (4, "right_on"),
                (5, "left_index"),
                (6, "right_index"),
                (10, "suffixes"),
            ],
            # NOTE: this enables const replacement to avoid errors in
            # test_excel1::test_impl2 caused by Numba 0.51 literals
            # TODO: fix underlying issue in Numba
            "read_excel": [
                (3, "names"),
            ],
        }

        if func_name in top_level_call_const_args:
            func_args = top_level_call_const_args[func_name]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)

        # convert const list to tuple for better optimization
        if func_name == "concat":
            self._call_arg_list_to_tuple(rhs, "concat", 0, "objs", nodes)

        return nodes + [assign]

    def _run_call_pd_series(self, assign, rhs, func_name, label):
        nodes = [assign]
        kws = dict(rhs.kws)
        lhs = assign.target
        data_arg = get_call_expr_arg("pd.Series", rhs.args, kws, 0, "data", "")
        idx_arg = get_call_expr_arg("pd.Series", rhs.args, kws, 1, "index", "")

        data_arg_def = guard(get_definition, self.func_ir, data_arg)

        if isinstance(data_arg_def, ir.Expr) and data_arg_def.op == "build_map":
            if idx_arg != "" and self.typemap[idx_arg.name] != types.none:
                raise_bodo_error(
                    "pd.Series(): Cannot specify index argument when initializing with a dictionary"
                )

            msg = "When initializng a series with a dictionary, the keys should be constant strings or constant ints"

            (tuples, new_nodes,) = bodo.utils.transform._convert_const_key_dict(
                rhs.args, self.func_ir, data_arg_def, msg, lhs.scope, lhs.loc
            )
            val_tup_var = tuples[0]
            idx_tup_var = tuples[1]
            # replace data/idx arg with value.idx tuple
            kws["data"] = val_tup_var
            kws["index"] = idx_tup_var

            # Move the rest of the args to kws
            kws_argnames = ["dtype", "name", "copy", "fastpath"]
            for i in range(2, len(rhs.args)):
                kws[kws_argnames[i - 2]] = rhs.args[i]

            rhs.kws = list(kws.items())
            rhs.args = []

            nodes = new_nodes + nodes
            self.changed = True

        return nodes

    def _run_make_function(self, assign, rhs):
        """convert ir.Expr.make_function into a JIT function if possible.
        Replaces MakeFunctionToJitFunction of Numba, and also supports converting
        non-constant freevars into UDF arguments if possible.
        """
        # mostly copied from Numba here:
        # https://github.com/numba/numba/blob/1d50422ab84bef84391f895184e2bd48ba0fab03/numba/core/untyped_passes.py#L562
        kw_default = guard(get_definition, self.func_ir, rhs.defaults)
        ok = False
        if kw_default is None or isinstance(kw_default, ir.Const):
            ok = True
        elif isinstance(kw_default, tuple):
            ok = all(
                [
                    isinstance(guard(get_definition, self.func_ir, x), ir.Const)
                    for x in kw_default
                ]
            )
        elif isinstance(kw_default, ir.Expr):
            if kw_default.op != "build_tuple":
                return [assign]
            ok = all(
                [
                    isinstance(guard(get_definition, self.func_ir, x), ir.Const)
                    for x in kw_default.items
                ]
            )
        if not ok:
            return [assign]

        nodes = []
        try:
            pyfunc = ir_utils.convert_code_obj_to_function(rhs, self.func_ir)
        except BodoError:
            # convert non-constant freevars to UDF arguments if possible and try again
            guard(self._transform_make_function_freevars, assign, rhs, nodes)
            pyfunc = ir_utils.convert_code_obj_to_function(rhs, self.func_ir)

        func = bodo.jit(distributed=False)(pyfunc)
        new_rhs = ir.Global(rhs.code.co_name, func, rhs.loc)
        assign.value = new_rhs
        self.typemap.pop(assign.target.name, None)
        self.changed = True
        nodes.append(assign)
        return nodes

    def _transform_make_function_freevars(self, assign, rhs, nodes):
        """check if ir.Expr.make_function is only used in DataFrame/Series.apply()
        and transform its free variables into arguments.
        Raises GuardException if not possible.
        """
        # requiring rhs.closure to be set to the tuple of free variables in the IR,
        # which seems to be always the case
        require(rhs.closure)
        # cannot handle default arguments yet since we append new args last
        require(rhs.defaults is None)
        # avoid potential complex arg corner cases
        require(rhs.code.co_posonlyargcount == 0 and rhs.code.co_kwonlyargcount == 0)

        # make sure there is only one use which is a DataFrame/Series.apply() call
        apply_assign, args_var = self._ensure_apply_call_use(assign)

        # find and remove free variables that cannot be converted to constants in the
        # IR in convert_code_obj_to_function()
        items = find_build_tuple(self.func_ir, rhs.closure)
        new_args = []
        freevar_names = []
        freevar_inds = []
        for i, freevar in enumerate(items):
            freevar_def = guard(get_definition, self.func_ir, freevar)
            if isinstance(freevar_def, (ir.Const, ir.Global, ir.FreeVar)) or is_expr(
                freevar_def, "make_function"
            ):
                continue
            freevar_inds.append(i)
            new_args.append(freevar)
            freevar_names.append(rhs.code.co_freevars[i])

        code = rhs.code
        # map freevar to its corresponding argument
        freevar_arg_map = {
            f_ind: code.co_argcount + i for i, f_ind in enumerate(freevar_inds)
        }
        new_code = _replace_load_deref_code(
            code.co_code, freevar_arg_map, code.co_argcount
        )

        # we can now change the IR/code since all checks are done (including
        # _replace_load_deref_code)

        # Pop the freevar indices in reverse order to ensure everything
        # stays in the same position
        # i.e. convert [0, 1, 2] to [2, 1, 0]
        for i in reversed(sorted(freevar_inds)):
            items.pop(i)

        new_co_varnames = (
            code.co_varnames[: code.co_argcount]
            + tuple(freevar_names)
            + code.co_varnames[code.co_argcount :]
        )
        new_co_freevars = tuple(set(code.co_freevars) - set(freevar_names))
        rhs.code = pytypes.CodeType(
            code.co_argcount + len(freevar_names),
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
            code.co_nlocals + len(freevar_names),
            code.co_stacksize,
            code.co_flags,
            new_code,
            code.co_consts,
            code.co_names,
            new_co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_lnotab,
            new_co_freevars,
            code.co_cellvars,
        )

        # pass free variables as arguments
        if args_var == "":
            # create a tuple for new arguments to pass as 'args'
            loc = rhs.loc
            args_var = ir.Var(assign.target.scope, mk_unique_var("apply_args"), loc)
            tuple_expr = ir.Expr.build_tuple(new_args, loc)
            nodes.append(ir.Assign(tuple_expr, args_var, loc))
            var_types = [self.typemap.get(v.name, None) for v in new_args]
            if None not in var_types:
                self.typemap[args_var.name] = types.Tuple(var_types)
            self.func_ir._definitions[args_var.name] = [tuple_expr]
            # kws may be a tuple (at least if empty), so create a new list rather than append
            apply_assign.value.kws = list(apply_assign.value.kws) + [
                ("args", args_var),
            ]
        else:
            # guard check for find_build_tuple done in _ensure_apply_call_use
            tup_list = find_build_tuple(self.func_ir, args_var)
            tup_list.extend(new_args)
            self.typemap.pop(args_var.name, None)
            var_types = [self.typemap.get(v.name, None) for v in tup_list]
            if None not in var_types:
                self.typemap[args_var.name] = types.Tuple(var_types)

    def _ensure_apply_call_use(self, assign):
        """make sure output make_function of 'assign' has only one use which is a
        DataFrame/Series.apply() call.
        Return the apply() call assignment.
        """
        func_varname = assign.target.name
        uses = []
        # TODO(ehsan): use a DU-chain to avoid traversing the IR for similar cases?
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if stmt is assign:
                    continue
                if func_varname in {v.name for v in stmt.list_vars()}:
                    uses.append(stmt)

        require(len(uses) == 1 and is_call_assign(uses[0]))
        apply_assign = uses[0]
        fdef = find_callname(self.func_ir, apply_assign.value)
        require(fdef)
        fname, fvar = fdef
        require(fname == "apply")
        require(
            isinstance(fvar, ir.Var)
            and isinstance(
                self.typemap.get(fvar.name, None), (DataFrameType, SeriesType)
            )
        )
        apply_rhs = apply_assign.value
        args_var = get_call_expr_arg(
            "apply", apply_rhs.args, dict(apply_rhs.kws), 4, "args", default=""
        )
        # make sure 'args' tuple can be updated
        if args_var != "":
            find_build_tuple(self.func_ir, args_var)
        return apply_assign, args_var

    def _run_call_bodosql_sql(
        self, assign, rhs, sql_context_var, func_name, label
    ):  # pragma: no cover
        """inline BodoSQLContextType.sql() calls since the generated code cannot
        be handled in regular overloads (requires Bodo's untyped pass, typing pass)

        This code is also used for _test_sql_unoptimized, which is an internal testing
        API that generates Pandas code on the original non-optimized plan. We use the
        testing function to check coverage of operators that would otherwise be
        optimized out. We use this so our test suite can have simple cases but we can
        have confidence in the complex cases when optimizations may not be possible
        (i.e. testing scalar support using literals).
        """
        import bodosql
        from bodosql.context_ext import BodoSQLContextType

        # In order to inline the sql() call, we must insure that the type of the input dataframe(s)
        # are finalized. dataframe type may have changed in typing pass (e.g. due to df setitem)
        # so we shouldn't use the actuall type of the dataframes used to initialize the sql_context_var
        names, df_typs = self._get_bodosql_ctx_name_df_typs(sql_context_var)

        for df_typ in df_typs:
            if df_typ is None:
                return [assign]

        sql_context_type = BodoSQLContextType(names, df_typs)
        # cannot transform yet if type is not available yet
        if sql_context_type is None:
            return [assign]

        # TODO: Add argument error handling (should reuse signature error checking
        # that will be created in df.head PR).

        kws = dict(rhs.kws)
        sql_var = get_call_expr_arg(
            f"BodoSQLContextType.{func_name}", rhs.args, kws, 0, "sql"
        )
        params_var = get_call_expr_arg(
            f"BodoSQLContextType.{func_name}",
            rhs.args,
            kws,
            1,
            "param_dict",
            default=types.none,
        )

        needs_transform = False
        try:
            err_msg = "BodoSQLContextType.sql() requires a constant sql string"
            sql_str = self._get_const_value(sql_var, label, err_msg)
        except (GuardException, BodoConstUpdatedError):
            needs_transform = True

        # TODO: Handle the none case
        if is_overload_none(params_var):
            keys, values = [], []
        else:
            try:
                keys, values = bodo.utils.transform.dict_to_const_keys_var_values_lists(
                    params_var,
                    self.func_ir,
                    self.arg_types,
                    self.typemap,
                    self._updated_containers,
                    self._require_const,
                    label,
                )
            except GuardException:
                needs_transform = True

        # If any variable needs to be a constant, try and
        # transform the code
        if needs_transform:
            self.needs_transform = True
            return [assign]

        keys = tuple(keys)
        value_typs = tuple([self.typemap[value.name] for value in values])

        if func_name == "sql":
            impl = bodosql.context_ext._gen_pd_func_for_query(
                sql_context_type, sql_str, keys, value_typs
            )
        elif func_name == "_test_sql_unoptimized":
            impl = bodosql.context_ext._gen_pd_func_for_unoptimized_query(
                sql_context_type, sql_str, keys, value_typs
            )
        elif func_name == "convert_to_pandas":
            impl = bodosql.context_ext._gen_pd_func_str_for_query(
                sql_context_type, sql_str, keys, value_typs
            )

        self.changed = True
        # BodoSQL generates df.columns setattr, which needs another transform to work
        # (See BodoSQL #189)
        self.needs_transform = True
        block_body = compile_func_single_block(
            impl,
            [sql_context_var] + values,
            assign.target,
            self,
            infer_types=False,
            run_untyped_pass=True,
            flags=self.flags,
            replace_globals=False,
        )
        # Attempt to apply filter pushdown if there is parquet load
        if any(
            [isinstance(stmt, bodo.ir.parquet_ext.ParquetReader) for stmt in block_body]
        ):
            block_body = self._apply_bodosql_filter_pushdown(block_body)
        return block_body

    def _apply_bodosql_filter_pushdown(self, block_body):
        """
        Function that tries to apply filter pushdown to a single "block" produced
        by BodoSQL. The block_body is a list of ir.Stmt.

        Currently we only attempt to apply filter pushdown(s) to files read
        from parquet within the BodoSQL block.

        Returns a new list of statements, updated with the filter pushdown(s).
        """
        # We generate a dummy scope + loc because BodoSQL doesn't contain useful
        # loc information anyways (since its a func_text).
        loc = ir.Loc("", 0)
        scope = ir.Scope(None, loc)
        # Wrap the body in a dummy Block + FunctionIR for Numba APIs
        block = ir.Block(scope, loc)
        block.body = block_body
        blocks = {0: block}
        definitions = build_definitions(blocks)
        func_ir = ir.FunctionIR({0: block}, False, -1, loc, definitions, 0, ())
        working_body = []
        for inst in block.body:
            # Try and determine if we could have a filter pushdown. This only
            # occurs when we have a getitem with a filter.
            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and inst.value.op in ("getitem", "static_getitem")
            ):
                updated_working_body = guard(
                    self._try_bodosql_filter_pushdown, inst, func_ir, working_body
                )
                # If we have an updated working body swap the working body list
                if updated_working_body:
                    working_body = updated_working_body
            working_body.append(inst)
        return working_body

    def _try_bodosql_filter_pushdown(self, inst, func_ir, working_body):
        """
        Tries to convert the given static_getitem expression into
        a filter pushdown. If this is not possible throws a guard exception.

        Pandas code generated from BodoSQL will contain an astype to perform
        filter pushdown and 1 or more df.loc

        For example

        df = pd.read_parquet("filename")
        df = pd.DataFrame(
            {
                "a": df["a"],
                "b": df["b"].astype("category")
                "c": df["c"],
            }
        )
        df1 = df.loc[:, ["b", "c"]]
        df2 = df1[df["b"] == 'a']

        Return a new working body if the pushdown was successful.
        """

        rhs = inst.value
        index_def = get_definition(func_ir, rhs.index)
        if is_expr(index_def, "binop"):
            value_def = get_definition(func_ir, rhs.value)

            # Intermediate DataFrames that need to be checked.
            used_dfs = {rhs.value.name: value_def}

            # If we have any df.loc calls that load all rows, they will appear
            # before the init_dataframe. We can find all rows with a
            # static getitem with a slice of all none for the rows.
            #
            # i.e.
            # df1 = df.loc[:, ["b", "c"]]
            #
            empty_slice = slice(None, None, None)
            # TODO: Refactor df.loc support into a general function and apply to Pandas as well.
            while (
                is_expr(value_def, "static_getitem")
                and isinstance(value_def.index, tuple)
                and len(value_def.index) > 0
                and value_def.index[0] == empty_slice
            ):
                used_name = value_def.value.name
                # Now we confirm we found a df.loc and traverse back to the original dataframe.
                value_def = get_definition(func_ir, value_def.value)
                # Add this to the intermediate DataFrames
                used_dfs[used_name] = value_def
                # If we didn't find a df.loc exit because the code structure is unexpected.
                require(is_expr(value_def, "getattr") and value_def.attr == "loc")
                used_name = value_def.value.name
                # Move the value to the df in df.loc
                value_def = get_definition(func_ir, value_def.value)
                # Add this to the intermediate DataFrames
                used_dfs[used_name] = value_def
            # If we load from parquet with paritions, BodoSQL will generate a pd.DataFrame call that performs
            # an astype with the categorical columns
            # For example:
            #
            # df = pd.read_parquet("filename")
            # df = pd.DataFrame(
            #     {
            #         "a": df["a"],
            #         "b": df["b"].astype("category")
            #         "c": df["c"],
            #     }
            # )
            # In this situation, the DataFrame is ALWAYS immediately clobbered.
            # This means we ignore uses of this variable (to avoid tracking each
            # column) and we rely on the source being a ParquetReader
            # to ensure it is safe to perform filter pushdown.
            #
            # TODO: Enable the normal path without astype in case we just perform predicate
            # pushdown without partition pushdown.
            if is_call(value_def) and guard(find_callname, func_ir, value_def) == (
                "DataFrame",
                "pandas",
            ):
                require(len(value_def.args) > 0)
                # Obtain the dictionary argument to pd.DataFrame
                df_dict = value_def.args[0]
                dict_def = get_definition(func_ir, df_dict)
                require(is_expr(dict_def, "build_tuple"))

                # Find the source DataFrame that should be created from the parquet node.
                # All dataframes are at the end of this build_tuple, so we look for their
                # source. These can either be a static getitem from the df or an astype.
                #
                # Again we don't check the use of these variables because of the assumed
                # code structure.
                source_col = get_definition(func_ir, dict_def.items[-1])
                if is_call(source_col):
                    # We have an astype.
                    source_col = get_definition(func_ir, source_col.func)
                    require(
                        is_expr(source_col, "getattr") and source_col.attr == "astype"
                    )
                    source_col = get_definition(func_ir, source_col.value)
                # The source column is a static getitem to the relevant df
                require(is_expr(source_col, "static_getitem"))
                source_df = get_definition(func_ir, source_col.value)

                # If the filter pushdown was successful we can update the working body
                working_body = self._try_filter_pushdown(
                    inst, source_df, index_def, working_body, func_ir, used_dfs, True
                )

        return working_body

    def _call_arg_list_to_tuple(self, rhs, func_name, arg_no, arg_name, nodes):
        """Convert call argument to tuple if it is a constant list"""
        kws = dict(rhs.kws)
        objs_var = get_call_expr_arg(func_name, rhs.args, kws, arg_no, arg_name, "")
        objs_def = guard(get_definition, self.func_ir, objs_var)
        if (
            is_expr(objs_def, "build_list")
            and objs_var.name not in self._updated_containers
        ):
            loc = objs_var.loc
            tuple_var = ir.Var(objs_var.scope, mk_unique_var("$tuple_var"), loc)
            var_types = [self.typemap.get(v.name, None) for v in objs_def.items]
            if None not in var_types:
                self.typemap[tuple_var.name] = types.Tuple(var_types)
            tuple_call = ir.Expr.build_tuple(objs_def.items, loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            nodes.append(tuple_assign)
            set_call_expr_arg(tuple_var, rhs.args, kws, arg_no, arg_name)
            self.changed = True

    def _is_df_call_transformed(self, rhs):
        """check for _bodo_transformed=True in call arguments to know if df call has
        been transformed already (df variable is replaced for inplace=True)
        """
        kws = dict(rhs.kws)
        return "_bodo_transformed" in kws and guard(
            find_const, self.func_ir, kws["_bodo_transformed"]
        )

    def _handle_df_inplace_func(
        self, assign, lhs, rhs, df_var, inplace_var, label, func_name
    ):
        """handle df.func(inplace=True) using variable replacement
        df.func(inplace=True) -> df2 = df.func(inplace=True)
        replaces df with df2 in the rest of the program. All definitions of df should
        dominate the call site for this approach to work.
        """
        inplace = guard(find_const, self.func_ir, inplace_var)
        if not inplace:
            return [assign]

        # TODO: make sure call post dominates df_var definition or df_var
        # is not used in other code paths
        if self._label_dominates_var_defs(label, df_var):
            # replace old variable with new one
            new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            self.replace_var_dict[df_var.name] = new_df_var
            self.changed = True
            true_var = ir.Var(
                df_var.scope, mk_unique_var("inplace_transform"), df_var.loc
            )
            true_assign = ir.Assign(ir.Const(True, lhs.loc), true_var, lhs.loc)
            rhs.kws.append(("_bodo_transformed", true_var))
            return [true_assign, assign, ir.Assign(lhs, new_df_var, lhs.loc)]
        else:
            raise BodoError(
                (
                    "DataFrame.{}(): non-deterministic inplace change of dataframe schema "
                    "not supported.\nSee "
                    "http://docs.bodo.ai/latest/source/not_supported.html"
                ).format(func_name)
            )

        return [assign]

    def _label_dominates_var_defs(self, label, df_var):
        """See if label dominates all labels of df_var's definitions"""
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        # there could be multiple definitions but all dominated by label
        # TODO: support multiple levels of branching?
        all_defs = self.func_ir._definitions[df_var.name]
        for var in all_defs:
            df_def = guard(get_definition, self.func_ir, var)
            if not (
                df_def in self.rhs_labels
                and label in cfg.post_dominators()[self.rhs_labels[df_def]]
            ):
                return False
        return True

    def _run_df_set_column(self, inst, col_name, label):
        """replace setitem of string index with a call to handle possible
        dataframe case where schema is changed:
        df['new_col'] = arr  ->  df2 = set_df_col(df, 'new_col', arr)
        """
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        self.changed = True
        # setting column possible only when:
        #   1) it dominates the df creation, so we can create a new df variable
        #      to replace the existing variable for the rest of the program,
        #      which avoids inplace update and schema change possiblity
        #      TODO: make sure there is no other reference (refcount==1)
        #   2) setting existing column with same type (inplace)
        # invalid case:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # return df

        # see if setitem dominates creation, # TODO: handle changing labels
        df_var = inst.target
        df_def = guard(get_definition, self.func_ir, df_var)
        dominates = False
        if (
            df_def in self.rhs_labels
            and label in cfg.post_dominators()[self.rhs_labels[df_def]]
        ):
            dominates = True

        # TODO: generalize to more cases
        # for example:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # else:
        #     df['B'] = C
        # return df
        # TODO: check for references to df
        # for example:
        # df = pd.DataFrame({'A': A})
        # df2 = df
        # df['B'] = C
        # return df2

        # create var for string index
        # expressions like list(np.array(target_typ.df_type.columns)[col_inds]) above
        # may create numpy.str_ values which inherit from str. use regular str to avoid
        # Numba errors
        col_name = str(col_name) if isinstance(col_name, str) else col_name
        cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
        self.typemap[cname_var.name] = types.literal(col_name)
        nodes = [ir.Assign(ir.Const(col_name, inst.loc), cname_var, inst.loc)]
        inplace = not dominates or isinstance(df_def, ir.Arg)

        if dominates:
            # rename the dataframe variable to keep schema static
            new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            out_var = new_df_var
            self.replace_var_dict[df_var.name] = new_df_var
        else:
            # cannot replace variable, but can set existing column with the
            # same data type
            # NOTE: data type is checked in _run_call_set_df_column() and
            # set_dataframe_data()
            out_var = df_var

        func = eval(
            "lambda df, cname, arr: bodo.hiframes.dataframe_impl.set_df_col(df, cname, arr, _inplace)"
        )
        args = [df_var, cname_var, inst.value]

        # assign output df type if possible to reduce typing iterations
        if (
            inst.value.name in self.typemap
            and self.typemap[inst.value.name] != types.unknown
        ):
            nodes += compile_func_single_block(
                func, args, out_var, self, extra_globals={"_inplace": inplace}
            )
            self.typemap.pop(out_var.name, None)
            self.typemap[out_var.name] = self.typemap[nodes[-1].value.name]
        else:
            nodes += compile_func_single_block(
                func, args, out_var, extra_globals={"_inplace": inplace}
            )

        return nodes

    def _error_on_df_control_flow(self, df_var, label, err_msg):
        """raise BodoError if 'label' does not dominate definition of 'df_var'"""
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        df_def = guard(get_definition, self.func_ir, df_var)
        dominates = (
            df_def in self.rhs_labels
            and label in cfg.post_dominators()[self.rhs_labels[df_def]]
        )
        if not dominates:
            raise BodoError(
                err_msg + " inside conditionals and loops not supported yet"
            )

    def _replace_vars(self, inst):
        # variable replacement can affect definitions so handling assignment
        # values specifically
        if is_assign(inst):
            lhs = inst.target.name
            self.func_ir._definitions[lhs].remove(inst.value)

        ir_utils.replace_vars_stmt(inst, self.replace_var_dict)

        if is_assign(inst):
            self.func_ir._definitions[lhs].append(inst.value)
            # if lhs changed, TODO: test
            if inst.target.name != lhs:
                self.func_ir._definitions[inst.target.name] = self.func_ir._definitions[
                    lhs
                ]

    def _get_method_obj_type(self, obj_var, func_var):
        """Get obj type for obj.method() calls, e.g. df.drop().
        Sometimes obj variable is not in typemap at this stage, but the bound function
        variable is in typemap, so try both.
        e.g. TestDataFrame::test_df_drop_inplace2
        """
        if obj_var.name in self.typemap:
            return self.typemap[obj_var.name]
        if func_var.name in self.typemap:
            return self.typemap[func_var.name].this

    def _replace_arg_with_literal(
        self, func_name, rhs, func_args, label, pyobject_to_literal=False
    ):
        """replace a function argument that needs to be constant with a literal to
        enable constant access in overload. This may force JIT arguments to be literals
        if needed to satify constant requirements.
        """
        kws = dict(rhs.kws)
        nodes = []
        for (arg_no, arg_name) in func_args:
            var = get_call_expr_arg(func_name, rhs.args, kws, arg_no, arg_name, "")
            # skip if argument not specified or literal already
            if var == "":
                continue
            if self._is_constant_var(var.name):
                if var.name in self._updated_containers:
                    # loop unrolling can potentially make updated lists constants
                    if self.ran_transform:
                        raise BodoError(
                            "{}(): argument '{}' requires a constant value but variable '{}' is updated inplace using '{}'\n{}\n".format(
                                func_name,
                                arg_name,
                                var.name,
                                self._updated_containers[var.name],
                                rhs.loc.strformat(),
                            )
                        )
                    else:
                        # save for potential loop unrolling
                        self._require_const[var] = label
                        self.needs_transform = True
                continue
            # get constant value for variable if possible.
            # Otherwise, just skip, assuming that the issue may be fixed later or
            # overload will raise an error if necessary.
            try:
                err_msg = (
                    f"{func_name}(): argument '{arg_name}' requires a constant value"
                )
                val = self._get_const_value(
                    var, label, rhs.loc, err_msg, pyobject_to_literal
                )
            except (GuardException, BodoConstUpdatedError):
                # save for potential loop unrolling
                continue
            # set values don't have literal types yet
            # convert to list for agg since it is equivalent, but skip otherwise
            # TODO(ehsan): add other functions where set is equivalent to list
            # we can look at is_list_like() use in Pandas
            if isinstance(val, set):
                if func_name in ("agg", "aggregate"):
                    val = list(val)
                    # avoid build_set since it can fail in Numba
                    var_def = guard(get_definition, self.func_ir, var)
                    if is_expr(var_def, "build_set"):
                        var_def.op = "build_list"
                else:
                    continue
            # replace argument variable with a new variable holding constant
            new_var = _create_const_var(val, var.name, var.scope, rhs.loc, nodes)
            set_call_expr_arg(new_var, rhs.args, kws, arg_no, arg_name)
            # var is not used here anymore, add to _transformed_vars so it can
            # potentially be removed since some dictionaries (e.g. in agg) may not be
            # type stable
            self._add_to_transformed_vars(var.name)
            self.changed = True

        rhs.kws = list(kws.items())
        return nodes

    def _get_const_value(
        self, var, label, loc, err_msg=None, pyobject_to_literal=False
    ):
        """get constant value for variable 'var'. If constant not found, saves info to
        run transforms like loop unrolling.
        If err_msg is provided, raise an error if transforms ran but the variable is
        still an updated container.
        """
        try:
            value = get_const_value_inner(
                self.func_ir,
                var,
                self.arg_types,
                self.typemap,
                self._updated_containers,
                pyobject_to_literal=pyobject_to_literal,
            )
        except BodoConstUpdatedError as e:
            # loop unrolling can potentially make updated lists constants
            if self.ran_transform and err_msg:
                raise BodoError(f"{err_msg} but {e}\n{loc.strformat()}\n")
            else:
                # save for potential loop unrolling
                self._require_const[var] = label
                self.needs_transform = True
            raise e
        except GuardException as e2:
            # save for potential loop unrolling
            self._require_const[var] = label
            raise e2
        return value

    def _is_constant_var(self, varname):
        """Return True if 'varname' is a constant variable in the IR"""
        # empty list/set/dict values cannot be typed currently but they are constant
        # TODO(ehsan): handle empty list/set/dict in typing
        var_def = guard(get_definition, self.func_ir, varname)
        if (
            isinstance(var_def, ir.Expr)
            and var_def.op in ("build_list", "build_set", "build_map")
            and not var_def.items
        ):
            return True
        return is_literal_type(self.typemap.get(varname, None))

    def _try_loop_unroll_for_const(self):
        """Try loop unrolling to find constant values in 'self._require_const'
        During unrolling, one loop may need some other loop to unroll to find its
        iteration space. See test_unroll_loop::impl8
        """
        consts = self._require_const.copy()
        for var, label in consts.items():
            changed = guard_const(self._try_loop_unroll_for_const_inner, var, label)
            # perform one unroll in each transform round only since multiple cases
            # may be covered at the same time
            if changed:
                break

        # If unrolling attempt added new constant value requirements, try unrolling to
        # potentially satisfy the new requirements
        if len(consts) != len(self._require_const):
            for var, label in self._require_const.items():
                if var in consts:
                    continue
                changed = guard_const(self._try_loop_unroll_for_const_inner, var, label)
                if changed:
                    break

    def _try_unroll_const_loop(self):
        """Try to unroll a loop with constant iteration range if possible. Otherwise,
        throw GuardException.
        Unrolls at most one loop per call.
        """
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        loops = cfg.loops()
        # unroll loops in program order to find constant dependencies more quickly
        for loop in sorted(loops.values(), key=lambda l: l.header):
            # consider only well-structured loops
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                continue
            if guard_const(self._try_unroll_const_loop_inner, loop):
                return  # only unroll one loop at a time

    def _try_unroll_const_loop_inner(self, loop):
        """unroll loop if possible and return True.
        Otherwise, raises GuardException or BodoConstUpdatedError
        """
        loop_index_var = self._get_loop_index_var(loop)
        iter_vals = self._get_loop_const_iter_vals(loop_index_var)

        # avoid unrolling very large loops (too many iterations and/or body statements)
        unroll_size = len(iter_vals) * sum(
            len(self.func_ir.blocks[l].body) for l in loop.body if l != loop.header
        )
        require(unroll_size < loop_unroll_limit)

        # start the unroll transform
        # no more GuardException since we can't bail out from this point
        self._unroll_loop(loop, loop_index_var, iter_vals)
        self._remove_container_updates(self.func_ir.blocks[list(loop.entries)[0]])
        return True

    def _try_loop_unroll_for_const_inner(self, var, label):
        """Try loop unrolling to make variable 'var' constant in block 'label' if:
        1) 'label' is in a for loop body
        2) iteration range of the loop is constant
        3) 'var' depends on the loop index
        raises GuardException if unrolling is not possible
        Here is an example transformation from:
            for c in df.columns:
                s += df[c].sum()
        to:
            c = 'A'
            s += df[c].sum()
            c = 'B'
            s += df[c].sum()
            ...
        """
        # get loop info and make sure unrolling is possible
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        loop, is_container_update = self._get_enclosing_loop(var, label, cfg)
        loop_index_var = self._get_loop_index_var(loop)
        require(
            (is_container_update and self._updated_in_loop(var, loop))
            or self._vars_dependant(var, loop_index_var)
        )
        iter_vals = self._get_loop_const_iter_vals(
            loop_index_var, True, list(loop.entries)[0]
        )

        # avoid unrolling very large loops (too many iterations and/or body statements)
        unroll_size = len(iter_vals) * sum(
            len(self.func_ir.blocks[l].body) for l in loop.body if l != loop.header
        )
        require(unroll_size < loop_unroll_limit)

        # start the unroll transform
        # no more GuardException since we can't bail out from this point
        if is_container_update:
            # remove patterns like 'if cond: mylist.append()' to enable update removal
            self._transform_container_if_update(loop, cfg)
        self._unroll_loop(loop, loop_index_var, iter_vals)

        if is_container_update:
            self._remove_container_updates(self.func_ir.blocks[list(loop.entries)[0]])

        return True

    def _transform_container_if_update(self, loop, cfg):
        """Remove patterns like 'if cond: mylist.append()' to enable update removal in
        _remove_container_updates() which requires a single loop body without control
        flow. See test_unroll_loop::impl8.
        For example:
            for c in [...]:
                if val in c:
                    l.append(c)
        is converted to:
            for c in [...]:
                if_list_append(l, val in c, c)
        """
        for l in loop.body.copy():
            block = self.func_ir.blocks[l]
            if guard(self._transform_if_update_inner, block, loop):
                guard(self._transform_if_update_branch, l, block, loop, cfg)
                break

    def _transform_if_update_inner(self, block, loop):
        """Pattern match conditional list update inside loop body and remove the control
        flow if possible. For example, the IR for 'if cond: mylist.append()' can be:
                $14contains_op.5.13 = ...
                bool16.14 = global(bool: <class 'bool'>)
                $16pred.15 = call bool16.14($14contains_op.5.13, func=bool16.14,
                    args=(Var($14contains_op.5.13, colname_check2.py:7),), kws=(),
                    vararg=None, target=None)
                branch $16pred.15, 42, 49
            label 42:
                $20list_append.3.17 = getattr(value=$phi18.0.5, attr=append)
                $20list_append.4.18 = call $20list_append.3.17(x.10,
                    func=$20list_append.3.17, args=(Var(x.10, colname_check2.py:7),),
                    kws=(), vararg=None, target=None)
                jump 49
            label 49:
                jump 18
        which is converted to:
                $14contains_op.5.13 = ...
                bool16.14 = global(bool: <class 'bool'>)
                $16pred.15 = call bool16.14($14contains_op.5.13, func=bool16.14,
                    args=(Var($14contains_op.5.13, colname_check2.py:7),), kws=(),
                    vararg=None, target=None)
                if_list_append_call.20 = global(if_list_append: ...)
                $20list_append.4.18 = call if_list_append_call.20($phi18.0.5,
                    $16pred.15, x.10, func=if_list_append_call.20, args=[Var($phi18.0.5,
                    colname_check2.py:7), Var($16pred.15, colname_check2.py:7),
                    Var(x.10, colname_check2.py:7)], kws=(), vararg=None, target=None)
        """
        # TODO(ehsan): support other container update calls in addition to list.append()
        # pattern match conditional update pattern
        require(isinstance(block.terminator, ir.Branch))
        cond_block = self.func_ir.blocks[block.terminator.truebr]
        require(
            isinstance(cond_block.terminator, ir.Jump)
            and cond_block.terminator.target == block.terminator.falsebr
        )
        require(len(cond_block.body) == 3)
        require(is_assign(cond_block.body[0]) and is_assign(cond_block.body[1]))
        getattr_assign = cond_block.body[0]
        call_assign = cond_block.body[1]
        require(
            is_expr(getattr_assign.value, "getattr")
            and isinstance(
                self.typemap.get(getattr_assign.value.value.name, None),
                (types.List, types.ListType),
            )
        )
        require(getattr_assign.value.attr == "append")
        require(
            is_call(call_assign.value)
            and call_assign.value.func.name == getattr_assign.target.name
        )

        # convert to a single block with if_list_append() call
        branch_node = block.body.pop()
        loc = block.loc
        scope = block.scope
        new_call_var = ir.Var(scope, mk_unique_var("if_list_append_call"), loc)
        new_call_var_assign = ir.Assign(
            ir.Global("if_list_append", if_list_append, loc), new_call_var, loc
        )
        call_args = [
            getattr_assign.value.value,
            branch_node.cond,
            call_assign.value.args[0],
        ]
        new_call_assign = ir.Assign(
            ir.Expr.call(new_call_var, call_args, (), loc), call_assign.target, loc
        )
        # NOTE: cannot remove branch_node.falsebr block since it may be target of
        # another block (not necessary also, _unroll_loop simplifies CFG), see [BE-1354]
        self.func_ir.blocks.pop(branch_node.truebr)
        block.body += [
            new_call_var_assign,
            new_call_assign,
            ir.Jump(branch_node.falsebr, loc),
        ]
        loop.body.remove(branch_node.truebr)
        return True

    def _transform_if_update_branch(self, label, block, loop, cfg):
        """Pattern match extra condition for list update and remove the control
        flow if possible. For example, transform
        'if cond1: if_list_append(mylist, cond2, val)' to
        'if_list_append(mylist, cond1 and cond2, val)'.
        This repeats execution of the nodes in block containing 'if_list_append' even
        when cond1 is false, so we have to check these conditions for safety:
        1) nodes are side-effect free
        2) the defined variables are not used in any other loop body block
        3) variables defined later are not used
        TODO(ehsan): generalize to more than two conditions
        """

        # make sure there is a single previous block that branches to this block
        preds = list(cfg.predecessors(label))
        require(len(preds) == 1)
        prev_label = preds[0][0]
        prev_block = self.func_ir.blocks[prev_label]
        require(
            isinstance(prev_block.terminator, ir.Branch)
            and prev_block.terminator.truebr == label
            and prev_block.terminator.falsebr == block.terminator.target
        )

        # make sure nodes are side-effect free and don't use variables defined later
        defined_vars = set()
        for stmt in reversed(block.body[:-3]):
            rhs = stmt.value
            require(is_assign(stmt))
            require(self._has_no_side_effect(rhs))
            defined_vars.add(stmt.target.name)
            # ensure later defined variables are not used in previous statements
            used_vars = {v.name for v in stmt.list_vars() if v.name != stmt.target.name}
            require(not used_vars & defined_vars)

        # make sure defined variables are not used in other loop blocks
        for l in loop.body:
            if l == label:
                continue
            for stmt in self.func_ir.blocks[l].body:
                require(not defined_vars & {v.name for v in stmt.list_vars()})

        # combine branch condition with if_append_call and convert branch to jump
        loc = block.loc
        prev_cond = prev_block.terminator.cond
        prev_block.body[-1] = ir.Jump(label, block.loc)
        if_append_call = block.body[-2].value
        cond_var = if_append_call.args[1]
        new_cond_var = ir.Var(block.scope, mk_unique_var("new_cond"), loc)
        cond_assign = ir.Assign(
            ir.Expr.binop(operator.and_, prev_cond, cond_var, loc), new_cond_var, loc
        )
        if_append_call.args[1] = new_cond_var
        block.body.insert(-2, cond_assign)

    def _remove_container_updates(self, block):
        """remove container updates from 'block' statements if possible.
        Find containers that are initialized within the block and updated using a
        constant value before use. It transforms the code to avoid the update.
        For example, a = []; a.append(2) -> a = [2]
        """
        # containers that are defined but not used yet (except updates handled here)
        defined_containers = set()
        # nodes generated for generating constant values (to redefine containers)
        const_nodes = []
        for stmt in block.body:
            if is_assign(stmt) and isinstance(stmt.value, ir.Expr):
                # find const list/set definition
                # TODO(ehsan): support "build_map"
                if stmt.value.op in ("build_list", "build_set"):
                    defined_containers |= self._equiv_vars[stmt.target.name]
                    continue
                if stmt.value.op == "call":
                    if guard(find_callname, self.func_ir, stmt.value) == (
                        "set",
                        "builtins",
                    ):
                        defined_containers |= self._equiv_vars[stmt.target.name]
                        continue
                    # match container update calls and avoid the update if possible
                    # e.g. a = []; a.append(2) -> a = [2]
                    new_nodes = guard_const(
                        self._try_remove_container_update, stmt, defined_containers
                    )
                    if new_nodes:
                        const_nodes.extend(new_nodes)
                        continue
                    # potential container update call that couldn't be handled in
                    # _try_remove_container_update()
                    call_info = guard_const(
                        self._get_container_call_info, stmt.value, defined_containers
                    )
                    if call_info is not None:
                        _, cont_var = call_info
                        defined_containers -= self._equiv_vars[cont_var.name]
                # getattr nodes like a.append are handled when they are called (above)
                if stmt.value.op == "getattr":
                    continue
                # aliases are already stored in _equiv_vars
                if isinstance(stmt.value, ir.Var):
                    continue
                # potential unhandled container use
                for v in stmt.list_vars():
                    if v.name in defined_containers:
                        defined_containers -= self._equiv_vars[v.name]
        block.body = const_nodes + block.body

    def _get_container_call_info(self, expr, defined_containers):
        """Get call name and container variable for container call 'expr'"""
        fdef = find_callname(self.func_ir, expr)
        require(fdef and len(fdef) == 2)
        require(
            (
                fdef == ("if_list_append", "bodo.transforms.typing_pass")
                and expr.args[0].name in defined_containers
            )
            or (isinstance(fdef[1], ir.Var) and fdef[1].name in defined_containers)
        )
        require(isinstance(fdef[0], str))
        return fdef[0], expr.args[0] if fdef[0] == "if_list_append" else fdef[1]

    def _try_remove_container_update(self, stmt, defined_containers):
        """try to remove container update if possible.
        E.g. a = []; a.append(2) -> a = [2]
        Otherwise, raise GuardException.
        """
        # match container update call, e.g. a.append(2)
        func_name, cont_var = self._get_container_call_info(
            stmt.value, defined_containers
        )

        container_def = get_definition(self.func_ir, cont_var)
        require(
            isinstance(container_def, ir.Expr)
            and container_def.op in ("build_list", "build_set", "call")
        )

        # get constant values of container before update
        # TODO(ehsan): support "build_map"
        if container_def.op in ("build_list", "build_set"):
            container_val = [
                get_const_value_inner(
                    self.func_ir,
                    v,
                    self.arg_types,
                    self.typemap,
                    self._updated_containers,
                )
                for v in container_def.items
            ]
            if container_def.op == "build_set":
                container_val = set(container_val)
        elif container_def.op == "call" and find_callname(
            self.func_ir, container_def
        ) == ("set", "builtins"):
            require(len(container_def.args) == 0)  # TODO: support set() args
            container_val = set()
        else:
            raise GuardException("Invalid container def")

        # update container value by calling the actual update function
        args = stmt.value.args[1:] if func_name == "if_list_append" else stmt.value.args
        arg_vals = [
            get_const_value_inner(
                self.func_ir, v, self.arg_types, self.typemap, self._updated_containers
            )
            for v in args
        ]
        if func_name == "if_list_append":
            out_val = if_list_append(container_val, *arg_vals)
        else:
            out_val = getattr(container_val, func_name)(*arg_vals)

        nodes = []

        # replace container variable in getattr with dummy to avoid use detection later
        # e.g. a.append -> dummy.append
        if func_name != "if_list_append":
            func_var_def = get_definition(self.func_ir, stmt.value.func)
            require(is_expr(func_var_def, "getattr"))
            dummy_val = [1] if container_def.op == "build_list" else {1}
            # no more GuardException from here on since IR is being modified
            func_var_def.value = _create_const_var(
                dummy_val, cont_var.name, cont_var.scope, cont_var.loc, nodes
            )

        # update original container definition, e.g. a = [] -> a = [2]
        if container_def.op == "call" and find_callname(
            self.func_ir, container_def
        ) == ("set", "builtins"):
            # convert set() call into a build_set
            container_def.op = "build_set"
            container_def._kws = {"items": []}
        container_def.items = [
            _create_const_var(v, cont_var.name, cont_var.scope, cont_var.loc, nodes)
            for v in container_val
        ]
        # replace update call with constant output, e.g. b = a.append(2) - > b = None
        self.func_ir._definitions[stmt.target.name].remove(stmt.value)
        stmt.value = _create_const_var(
            out_val, cont_var.name, cont_var.scope, cont_var.loc, nodes
        )
        self.func_ir._definitions[stmt.target.name].append(stmt.value)

        # update defs so next call to _try_remove_container_update can find values of
        # updated variables (build_list items) using _create_const_var
        update_node_list_definitions(nodes, self.func_ir)
        return nodes

    def _unroll_loop(self, loop, loop_index_var, iter_vals):
        """replace loop with its iteration body instances (to enable typing, etc.)"""
        # phis need to be transformed into regular assignments since unrolling changes
        # control flow
        # typemap=None to avoid PreLowerStripPhis's generator manipulation
        numba.core.typed_passes.PreLowerStripPhis().run_pass(
            numba.core.compiler.StateDict({"func_ir": self.func_ir, "typemap": None})
        )

        # get loop label info
        loop_body = {l: self.func_ir.blocks[l] for l in loop.body if l != loop.header}
        with numba.parfors.parfor.dummy_return_in_loop_body(loop_body):
            body_labels = find_topo_order(loop_body)
        first_label = body_labels[0]
        last_label = body_labels[-1]
        loop_entry = list(loop.entries)[0]
        loop_exit = list(loop.exits)[0]
        # previous block's jump node, to be updated after each iter body gen
        prev_jump = self.func_ir.blocks[loop_entry].body[-1]
        scope = loop_index_var.scope

        # generate an instance of the loop body for each iteration
        for c in iter_vals:
            offset = ir_utils.next_label()
            # new unique loop body IR
            new_body = ir_utils.add_offset_to_labels(copy.deepcopy(loop_body), offset)
            new_first_label = first_label + offset
            new_last_label = last_label + offset
            nodes = []
            # create new const value for iteration index and add it to loop body
            _create_const_var(c, loop_index_var.name, scope, loop_index_var.loc, nodes)
            nodes[-1].target = loop_index_var
            new_body[new_first_label].body = nodes + new_body[new_first_label].body
            # adjust previous block's jump
            prev_jump.target = new_first_label
            prev_jump = new_body[new_last_label].body[-1]
            self.func_ir.blocks.update(new_body)

        prev_jump.target = loop_exit

        # clean up original loop IR
        self.func_ir.blocks.pop(loop.header)
        for l in loop_body:
            self.func_ir.blocks.pop(l)

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)

        # call SSA reconstruction to rename variables and prepare for type inference
        numba.core.untyped_passes.ReconstructSSA().run_pass(
            numba.core.compiler.StateDict(
                {"func_ir": self.func_ir, "locals": self.locals}
            )
        )

        self.changed = True

    def _get_enclosing_loop(self, var, label, cfg):
        """find enclosing loop for block 'label' if possible. Also return True if the
        loop updates a container.
        Otherwise, raise GuardException.
        """
        label_doms = cfg.dominators()[label]
        loops = cfg.loops()
        for loop in loops.values():
            # consider only well-structured loops
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                continue
            # cases where a container is updated in a loop and used afterwards
            # use label should dominate the loop
            if (
                var.name in self._updated_containers
                and list(loop.exits)[0] in label_doms
            ):
                return loop, True
            if label in loop.body:
                return loop, False

        raise GuardException("enclosing loop not found")

    def _get_loop_index_var(self, loop):
        """find index variable of 'for' loop. Numba always generates 'pair_first' for
        'for' loop indexes.
        Example header block (from test_unroll_loop):
        label 52:
            s.2 = phi(incoming_values=[Var(s, test_dataframe.py:2688),
                Var(s.1, test_dataframe.py:2690)], incoming_blocks=[0, 54])
            $52for_iter.1 = iternext(value=$phi52.0)
            $52for_iter.2 = pair_first(value=$52for_iter.1)
            $52for_iter.3 = pair_second(value=$52for_iter.1)
            $phi54.1 = $52for_iter.2
            branch $52for_iter.3, 54, 74
        """
        ind_var = None
        for stmt in self.func_ir.blocks[loop.header].body:
            if is_assign(stmt) and is_expr(stmt.value, "pair_first"):
                ind_var = stmt.target
            # use latest copy of index variable which is used in loop body
            if (
                ind_var
                and is_assign(stmt)
                and isinstance(stmt.value, ir.Var)
                and stmt.value.name == ind_var.name
            ):
                ind_var = stmt.target
        require(ind_var is not None)
        return ind_var

    def _get_loop_const_iter_vals(self, ind_var, force_const=False, loop_entry=0):
        """get constant iteration values for loop given its index variable.
        Matches this call sequence generated by Numba
        index_var = pair_first(iternext(getiter(loop_iterations)))
        Raises GuardException if couldn't find constant values
        force_const=True flag means that iteration values are known to be necessary,
        and we need to eventually force them to be constant (using _get_const_value)
        """
        pair_first_expr = get_definition(self.func_ir, ind_var)
        require(is_expr(pair_first_expr, "pair_first"))
        iternext_expr = get_definition(self.func_ir, pair_first_expr.value)
        require(is_expr(iternext_expr, "iternext"))
        getiter_expr = get_definition(self.func_ir, iternext_expr.value)
        require(is_expr(getiter_expr, "getiter"))
        iter_var = getiter_expr.value
        if force_const:
            return self._get_const_value(iter_var, loop_entry, iter_var.loc)
        return get_const_value_inner(
            self.func_ir,
            iter_var,
            self.arg_types,
            self.typemap,
            self._updated_containers,
        )

    def _vars_dependant(self, var1, var2):
        """return True if 'var1' is equivalent to or depends on 'var2'"""
        assert isinstance(var1, ir.Var) and isinstance(var2, ir.Var)
        if var1.name == var2.name or var1.name in self._equiv_vars[var2.name]:
            return True

        var1_def = get_definition(self.func_ir, var1)
        var2_def = get_definition(self.func_ir, var2)

        if var1_def == var2_def:
            return True

        if is_expr(var1_def, "binop"):
            return self._vars_dependant(var1_def.lhs, var2) or self._vars_dependant(
                var1_def.rhs, var2
            )

        # dependant through call, e.g. df["A"+str(i)]
        if is_call(var1_def):
            return any(self._vars_dependant(arg, var2) for arg in var1_def.args)

        return False

    def _updated_in_loop(self, var, loop):
        """return True if 'var' is updated in 'loop', e.g. a.append(3) is in loop body"""
        for l in loop.body:
            for stmt in self.func_ir.blocks[l].body:
                # match updated container call like a.append(3)
                if is_call_assign(stmt):
                    func_def = get_definition(self.func_ir, stmt.value.func)
                    if (
                        is_expr(func_def, "getattr")
                        and func_def.value.name in self._updated_containers
                    ):
                        # a variable that 'var' is dependent on may be updated
                        # e.g. a.append(2); b = [1] + a
                        if self._vars_dependant(var, func_def.value):
                            return True

        return False

    def _has_no_side_effect(self, expr):
        """return True if 'expr' is an IR node without side-effect
        TODO(ehsan): use has_no_side_effect() to be less conservative?
        """
        return (
            isinstance(expr, (ir.Const, ir.Global, ir.FreeVar))
            or (isinstance(expr, ir.Expr) and expr.op not in ("inplace_binop", "call"))
            or (
                is_call(expr)
                and guard(find_callname, self.func_ir, expr, self.typemap)
                == ("bool", "builtins")
            )
        )

    def _add_to_transformed_vars(self, varname):
        """add variable 'varname' to the set of transformed variables to be removed
        later to avoid typing errors.
        If the variable is a constant dict, it looks at the values and removes
        list of functions since they can fail in Numba's typing.
        """
        var_def = guard(get_definition, self.func_ir, varname)
        if is_expr(var_def, "build_map"):
            for v in var_def.items:
                v_def = guard(get_definition, self.func_ir, v[1])
                if is_expr(v_def, "build_list"):
                    if any(
                        is_const_func_type(self.typemap.get(a.name, None))
                        for a in v_def.items
                    ):
                        self._transformed_vars.add(v[1].name)
        self._transformed_vars.add(varname)


def _create_const_var(val, name, scope, loc, nodes):
    """create a new variable that holds constant value 'val'. Generates constant
    creation IR nodes and adds them to 'nodes'.
    """
    # convert pd.Index values (usually coming from "df.columns") to list to enable
    # passing values as constant (list and pd.Index are equivalent for Pandas API calls
    # that take column names).
    if isinstance(val, pd.Index):
        val = list(val)
    new_var = ir.Var(scope, mk_unique_var(name), loc)
    if isinstance(val, tuple):
        const_node = ir.Expr.build_tuple(
            [_create_const_var(v, name, scope, loc, nodes) for v in val], loc
        )
    elif isinstance(val, list):
        # list of functions cannot be typed properly in Numba yet, so we use tuple of
        # functions instead. The only place list of functions can be used is in
        # groupby.agg where list and tuple are equivalent.
        if any(
            is_const_func_type(f) or isinstance(f, numba.core.dispatcher.Dispatcher)
            for f in val
        ):
            const_node = ir.Expr.build_tuple(
                [_create_const_var(v, name, scope, loc, nodes) for v in val], loc
            )
        else:
            const_node = ir.Expr.build_list(
                [_create_const_var(v, name, scope, loc, nodes) for v in val], loc
            )
    # create a tuple with sentinel for dict case since there is no dict literal
    elif isinstance(val, dict):
        # first tuple element is a sentinel specifying that this tuple is a const dict
        const_dict_sentinel_var = ir.Var(
            scope, mk_unique_var("const_dict_sentinel"), loc
        )
        nodes.append(
            ir.Assign(ir.Const(CONST_DICT_SENTINEL, loc), const_dict_sentinel_var, loc)
        )
        items = [
            _create_const_var(v, name, scope, loc, nodes)
            for v in itertools.chain(*val.items())
        ]
        const_node = ir.Expr.build_tuple([const_dict_sentinel_var] + items, loc)
    else:
        const_node = ir.Const(val, loc)
    new_assign = ir.Assign(const_node, new_var, loc)
    nodes.append(new_assign)
    return new_var


def _find_updated_containers(blocks, topo_order):
    """find variables that are potentially list/set/dict containers that are updated
    inplace.
    Just looks for getattr nodes with inplace update methods of list/set/dict like 'pop'
    and setitem nodes.
    Returns a dictionary of variable names and the offending method names.
    """
    updated_containers = {}
    # keep track of potential aliases for variables like lists, which can happen in
    # translation of list comprehension, see test_dataframe_columns_list
    equiv_vars = defaultdict(set)
    for label in topo_order:
        b = blocks[label]
        for stmt in b.body:
            # var to var assignment, creating a potential alias
            if (
                is_assign(stmt)
                and isinstance(stmt.value, ir.Var)
                and stmt.target.name != stmt.value.name
            ):
                lhs = stmt.target.name
                rhs = stmt.value.name
                equiv_vars[lhs].add(rhs)
                equiv_vars[rhs].add(lhs)
                equiv_vars[lhs] |= equiv_vars[rhs]
                equiv_vars[rhs] |= equiv_vars[lhs]
                if rhs in updated_containers:
                    _set_updated_container(
                        lhs, updated_containers[rhs], updated_containers, equiv_vars
                    )
                elif lhs in updated_containers:
                    _set_updated_container(
                        rhs, updated_containers[lhs], updated_containers, equiv_vars
                    )
            elif (
                is_assign(stmt)
                and is_expr(stmt.value, "getattr")
                and stmt.value.attr
                in (
                    # dict
                    "clear",
                    "pop",
                    "popitem",
                    "update",
                    # set
                    "add",
                    "difference_update",
                    "discard",
                    "intersection_update",
                    "remove",
                    "symmetric_difference_update",
                    # list
                    "append",
                    "extend",
                    "insert",
                    "reverse",
                    "sort",
                )
            ):
                _set_updated_container(
                    stmt.value.value.name,
                    stmt.value.attr,
                    updated_containers,
                    equiv_vars,
                )
            elif is_setitem(stmt):
                _set_updated_container(
                    stmt.target.name, "setitem", updated_containers, equiv_vars
                )
            # binop of updated containers creates an updated container
            elif is_assign(stmt) and is_expr(stmt.value, "binop"):
                arg1 = stmt.value.lhs.name
                arg2 = stmt.value.rhs.name
                if arg1 in updated_containers:
                    _set_updated_container(
                        stmt.target.name,
                        updated_containers[arg1],
                        updated_containers,
                        equiv_vars,
                    )
                elif arg2 in updated_containers:
                    _set_updated_container(
                        stmt.target.name,
                        updated_containers[arg2],
                        updated_containers,
                        equiv_vars,
                    )
            # handle simple calls like list(a)
            elif is_call_assign(stmt):
                for v in stmt.value.args:
                    if v.name in updated_containers:
                        _set_updated_container(
                            stmt.target.name,
                            updated_containers[v.name],
                            updated_containers,
                            equiv_vars,
                        )

    # combine all aliases transitively
    old_equiv_vars = copy.deepcopy(equiv_vars)
    for v in old_equiv_vars:
        for w in old_equiv_vars[v]:
            equiv_vars[v] |= equiv_vars[w]
        for w in old_equiv_vars[v]:
            equiv_vars[w] = equiv_vars[v]

    # update updated_containers info based on aliases
    # NOTE: may not capture binop of updated containers in all cases, but
    # get_const_value_inner() will catch the corner cases and avoid invalid results
    for v in list(updated_containers.keys()):
        m = updated_containers[v]
        for w in equiv_vars[v]:
            updated_containers[w] = m

    return updated_containers, equiv_vars


def guard_const(func, *args, **kwargs):
    """Same as guard(), but also checks for BodoConstUpdatedError"""
    try:
        return func(*args, **kwargs)
    except (GuardException, BodoConstUpdatedError):
        return None


@register_jitable
def if_list_append(l, cond, value):
    """helper function to call list.append() if a condition is True"""
    if cond:
        l.append(value)


def _set_updated_container(varname, update_func, updated_containers, equiv_vars):
    """helper to set 'varname' and its aliases as updated containers"""
    updated_containers[varname] = update_func
    # make sure an updated container variable is always equivalent to itself since
    # assumed in _remove_container_updates()
    equiv_vars[varname].add(varname)
    for w in equiv_vars[varname]:
        updated_containers[w] = update_func


def _replace_load_deref_code(code, freevar_arg_map, prev_argcount):
    """replace load of free variables in byte code with load of new arguments and
    adjust local variable indices due to new arguments in co_varnames.
    # https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST
    # https://docs.python.org/3/library/inspect.html
    # https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
    raises GuardException if there is STORE_DEREF in input code (for setting freevars)
    """
    import dis

    from numba.core.bytecode import ARG_LEN, CODE_LEN, NO_ARG_LEN

    # assuming these constants are 1 to simplify the code, very unlikely to change
    assert (
        CODE_LEN == 1 and ARG_LEN == 1 and NO_ARG_LEN == 1
    ), "invalid bytecode version"
    # cannot handle cases that write to free variables
    banned_ops = (dis.opname.index("STORE_DEREF"), dis.opname.index("LOAD_CLOSURE"))
    # local variable access to be adjusted
    local_varname_ops = (
        dis.opname.index("LOAD_FAST"),
        dis.opname.index("STORE_FAST"),
        dis.opname.index("DELETE_FAST"),
    )
    n_new_args = len(freevar_arg_map)

    new_code = np.empty(len(code), np.int8)
    n = len(code)
    i = 0
    while i < n:
        op = code[i]
        arg = code[i + 1]
        require(op not in banned_ops)

        # adjust local variable access since index includes arguments
        if op in local_varname_ops and arg >= prev_argcount:
            arg += n_new_args

        # replace free variable load
        if op == dis.opname.index("LOAD_DEREF") and arg in freevar_arg_map:
            op = dis.opname.index("LOAD_FAST")
            arg = freevar_arg_map[arg]

        new_code[i] = op
        new_code[i + 1] = arg
        i += 2

    return bytes(new_code)
