"""
Bodo type inference pass that performs transformations that enable typing of the IR
according to Bodo requirements (using partial typing).
"""
import copy
import itertools
import re
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, types
from numba.core.compiler_machinery import register_pass
from numba.core.ir_utils import (
    GuardException,
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
    is_const_func_type,
    is_list_like_index_type,
    is_literal_type,
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
            # run regular type inference again with _raise_errors = True to set function
            # return type and raise errors if any
            # TODO: avoid this extra pass when possible in Numba
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
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
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

                new_body.extend(out_nodes)
                update_node_list_definitions(out_nodes, self.func_ir)
                for inst in out_nodes:
                    if is_assign(inst):
                        self.rhs_labels[inst.value] = label

            blocks[label].body = new_body

        # try loop unrolling if some const values couldn't be resolved
        if self._require_const:
            for var, label in self._require_const.items():
                changed = guard(self._try_loop_unroll_for_const, var, label)
                # perform one unroll in each transform round only since multiple cases
                # may be covered at the same time
                if changed:
                    break

        # try unrolling a loop with constant range if everything else failed
        if self.change_required and not self.changed and not self.needs_transform:
            guard(self._try_unroll_const_loop)

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

        # detect type unification errors in control flow
        # TODO(ehsan): update after #2258 is merged
        if is_expr(rhs, "phi"):
            if (
                not rhs.incoming_values
                or rhs.incoming_values[0].name not in self.typemap
            ):  # pragma: no cover
                return [assign]
            first_type = self.typemap[rhs.incoming_values[0].name]
            for v in rhs.incoming_values[1:]:
                if (
                    v.name not in self.typemap
                    or self.typingctx.unify_pairs(first_type, self.typemap[v.name])
                    is None
                ):
                    self.change_required = True

        return [assign]

    def _run_getitem(self, assign, rhs, label):
        target = rhs.value
        target_typ = self.typemap.get(target.name, None)
        nodes = []
        idx = get_getsetitem_index_var(rhs, self.typemap, nodes)
        idx_typ = self.typemap.get(idx.name, None)

        # find constant index for df["A"] or df[["A", "B"]] cases
        if isinstance(target_typ, DataFrameType) and idx_typ in (
            bodo.string_type,
            types.List(bodo.string_type),
        ):
            # static_getitem has the values embedded
            if rhs.op == "static_getitem":
                val = rhs.index
            else:
                # try to find index values
                try:
                    val = get_const_value_inner(
                        self.func_ir, idx, self.arg_types, self.typemap
                    )
                except GuardException:
                    # couldn't find values, just return to be handled later
                    # save for potential loop unrolling
                    self._require_const[idx] = label
                    nodes.append(assign)
                    return nodes
            # replace index variable with a new variable holding constant
            new_var = _create_const_var(val, idx.name, idx.scope, idx.loc, nodes)
            if rhs.op == "static_getitem":
                rhs.index_var = new_var
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
            col_slice = guard(
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

            # create output df
            columns = target_typ.df_type.columns[col_slice]
            # get df arrays using const slice
            # data_outs = []
            # Generate the results by reusing the indexing helper functions
            if isinstance(idx_typ.types[0], types.Integer):
                impl = bodo.hiframes.dataframe_indexing._gen_iloc_getitem_row_impl(
                    target_typ.df_type, columns, "idx"
                )
            elif (
                is_list_like_index_type(idx_typ.types[0])
                and isinstance(idx_typ.types[0].dtype, (types.Integer, types.Boolean))
                or isinstance(idx_typ.types[0], types.SliceType)
            ):
                impl = (
                    bodo.hiframes.dataframe_indexing._gen_iloc_getitem_bool_slice_impl(
                        target_typ.df_type, columns, idx_typ.types[0], "idx", False
                    )
                )
            else:
                bodo.utils.typing.raise_bodo_error(
                    f"df.iloc[] getitem using {idx_typ} not supported"
                )  # pragma: no cover

            self.changed = True
            return nodes + compile_func_single_block(
                impl, [target, tup_list[0]], assign.target, self
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
                val = get_const_value_inner(
                    self.func_ir, col_ind_var, self.arg_types, self.typemap
                )
            except GuardException:
                # couldn't find values, just return to be handled later
                nodes.append(assign)
                return nodes

            # avoid transform if selected columns not all in dataframe schema
            # may require schema change, see test_loc_col_select (impl4)
            if (
                len(val) > 0
                and not isinstance(val[0], (bool, np.bool_))
                and not all(c in target_typ.df_type.columns for c in val)
            ):
                nodes.append(assign)
                return nodes

            impl = bodo.hiframes.dataframe_indexing.gen_df_loc_col_select_impl(
                target_typ.df_type, val
            )
            self.changed = True
            return nodes + compile_func_single_block(
                impl, [target, idx], assign.target, self
            )

        nodes.append(assign)
        return nodes

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
        idx_const = guard(
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

        col_inds, row_ind = self._get_loc_indices(idx_var)

        # couldn't find column name values, just return to be handled later
        if col_inds is None:
            return nodes + [inst]

        # get column names if bool list
        if len(col_inds) > 0 and isinstance(col_inds[0], (bool, np.bool_)):
            col_inds = list(
                pd.Series(target_typ.df_type.columns, dtype=object)[col_inds]
            )

        # if setting full columns
        if row_ind == slice(None):
            df_var = self._get_loc_df_var(inst)
            nodes += self._gen_df_setitem_full_column(inst, df_var, col_inds, label)
            self.changed = True
            return nodes

        # avoid transform if selected columns not all in dataframe schema
        # may require schema change, see test_loc_setitem (impl6)
        if not all(c in target_typ.df_type.columns for c in col_inds):
            nodes.append(inst)
            return nodes

        self.changed = True
        func_text = "def impl(I, idx, value):\n"
        func_text += "  df = I._obj\n"
        for c in col_inds:
            c_idx = target_typ.df_type.columns.index(c)
            func_text += f"  bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_idx})[idx[0]] = value\n"

        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]
        return nodes + compile_func_single_block(
            impl, [inst.target, idx_var, inst.value], None, self
        )

    def _run_setitem_df_iloc(self, inst, target_typ, idx_typ, idx_var, nodes, label):
        """transform df.iloc setitem nodes, e.g. df.loc[:, 1] = 3"""

        col_inds, row_ind = self._get_loc_indices(idx_var)

        # couldn't find column name values, just return to be handled later
        if col_inds is None:
            return nodes + [inst]

        # if setting full columns
        if row_ind == slice(None):
            col_names = [target_typ.df_type.columns[c_ind] for c_ind in col_inds]
            df_var = self._get_loc_df_var(inst)
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
            impl, [inst.target, idx_var, inst.value], None, self
        )

    def _get_loc_indices(self, idx_var):
        """get row/column index values for df.loc/df.iloc if possible"""
        # get column index var
        tup_list = guard(find_build_tuple, self.func_ir, idx_var)
        if tup_list is None or len(tup_list) != 2:  # pragma: no cover
            raise BodoError("Invalid df.loc[ind,ind] case")
        row_ind_var = tup_list[0]
        col_ind_var = tup_list[1]

        # try to find index values
        try:
            col_inds = get_const_value_inner(
                self.func_ir, col_ind_var, self.arg_types, self.typemap
            )
        except GuardException:
            col_inds = None

        # normalize single column name to list
        if not isinstance(col_inds, (list, tuple, np.ndarray)):
            col_inds = [col_inds]

        # try to find index values
        try:
            row_ind = get_const_value_inner(
                self.func_ir, row_ind_var, self.arg_types, self.typemap
            )
        except GuardException:
            row_ind = None

        return col_inds, row_ind

    def _get_loc_df_var(self, inst):
        """get dataframe variable from df.loc/iloc nodes.
        just gets the definition of the node (assuming no unusual control flow).
        """
        loc_def = guard(get_definition, self.func_ir, inst.target)
        if not is_expr(loc_def, "getattr"):  # pragma: no cover
            raise BodoError("Invalid df.loc/iloc[] setitem")
        return loc_def.value

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
                    columns = get_const_value_inner(
                        self.func_ir, inst.value, self.arg_types, self.typemap
                    )
                except GuardException:
                    # couldn't find values, just return to be handled later
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

        return [inst]

    def _run_call(self, assign, rhs, label):

        # import bodosql's BodoSQLContextType if installed
        # avoiding top-level import to prevent circular import issues
        try:  # pragma: no cover
            from bodosql.context_ext import BodoSQLContextType
        except:
            # workaround: something that makes isinstance(type, BodoSQLContextType) always false
            BodoSQLContextType = int

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
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(
                self._get_method_obj_type(func_mod, rhs.func), BodoSQLContextType
            )
            and func_name == "sql"
        ):  # pragma: no cover
            return self._run_call_bodosql_sql(assign, rhs, func_mod, func_name, label)

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

    def _handle_df_assign(self, lhs, rhs, df_var, assign):
        """replace df.assign() with its implementation to avoid overload errors with
        (**kwargs)
        """
        kws = dict(rhs.kws)
        df_type = self.typemap.get(df_var.name, None)
        # cannot transform yet if dataframe type is not available yet
        if df_type is None:
            return [assign]
        additional_columns = tuple(kws.keys())
        previous_columns = set(df_type.columns)
        # columns below are preserved
        preserved_columns = previous_columns - set(additional_columns)
        name_col_total = []
        data_col_total = []
        for c in preserved_columns:
            name_col_total.append(c)
            data_col_total.append("df['{}'].values".format(c))
        # The new columns as constructed by the operation
        for i, c in enumerate(additional_columns):
            name_col_total.append(c)
            e_col = "bodo.utils.conversion.coerce_to_array(new_arg{}, scalar_to_arr_len=len(df))".format(
                i
            )
            data_col_total.append(e_col)
        data_args = ", ".join(data_col_total)
        header = "def impl(df, {}):\n".format(
            ", ".join("new_arg{}".format(i) for i in range(len(kws)))
        )
        impl = bodo.hiframes.dataframe_impl._gen_init_df(
            header, tuple(name_col_total), data_args
        )
        self.changed = True
        return compile_func_single_block(impl, [df_var] + list(kws.values()), lhs, self)

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

    def _run_call_bodosql_sql(
        self, assign, rhs, sql_context_var, func_name, label
    ):  # pragma: no cover
        """inline BodoSQLContextType.sql() calls since the generated code cannot
        be handled in regular overloads (requires Bodo's untyped pass, typing pass)
        """
        import bodosql

        sql_context_type = self.typemap.get(sql_context_var.name, None)
        # cannot transform yet if type is not available yet
        if sql_context_type is None:
            return [assign]

        kws = dict(rhs.kws)
        sql_var = get_call_expr_arg("BodoSQLContextType.sql", rhs.args, kws, 0, "sql")

        # get constant value for variable if possible.
        # Otherwise, just skip, assuming that the issue may be fixed later or
        # overload will raise an error if necessary.
        try:
            sql_str = get_const_value_inner(
                self.func_ir,
                sql_var,
                self.arg_types,
                self.typemap,
                self._updated_containers,
            )
        except GuardException:
            # save for potential loop unrolling
            self._require_const[sql_var] = label
            return [assign]

        impl = bodosql.context_ext._gen_pd_func_for_query(sql_context_type, sql_str)
        self.changed = True
        # BodoSQL generates df.columns setattr, which needs another transform to work
        # (See BodoSQL #189)
        self.needs_transform = True
        return compile_func_single_block(
            impl,
            [sql_context_var],
            assign.target,
            self,
            # BodoSQL Like functionality needs re module
            extra_globals={"re": re},
            infer_types=False,
            run_untyped_pass=True,
            flags=self.flags,
        )

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
            if is_literal_type(self.typemap.get(var.name, None)):
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
                val = get_const_value_inner(
                    self.func_ir,
                    var,
                    self.arg_types,
                    self.typemap,
                    self._updated_containers,
                    pyobject_to_literal=pyobject_to_literal,
                )
            except BodoConstUpdatedError as e:
                # loop unrolling can potentially make updated lists constants
                if self.ran_transform:
                    raise BodoError(
                        "{}(): argument '{}' requires a constant value but {}\n{}\n".format(
                            func_name, arg_name, e, rhs.loc.strformat()
                        )
                    )
                else:
                    # save for potential loop unrolling
                    self._require_const[var] = label
                    self.needs_transform = True
                    continue
            except GuardException:
                # save for potential loop unrolling
                self._require_const[var] = label
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

    def _try_unroll_const_loop(self):
        """Try to unroll a loop with constant iteration range if possible. Otherwise,
        throw GuardException.
        Unrolls at most one loop per call.
        """
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        loops = cfg.loops()
        for loop in loops.values():
            # consider only well-structured loops
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                continue
            loop_index_var = self._get_loop_index_var(loop)
            iter_vals = self._get_loop_const_iter_vals(loop_index_var)
            # start the unroll transform
            # no more GuardException since we can't bail out from this point
            self._unroll_loop(loop, loop_index_var, iter_vals)
            return  # only unroll one loop at a time

        raise GuardException("const loop to unroll not found")

    def _try_loop_unroll_for_const(self, var, label):
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
        loop, is_container_update = self._get_enclosing_loop(var, label)
        loop_index_var = self._get_loop_index_var(loop)
        require(
            (is_container_update and self._updated_in_loop(var, loop))
            or self._vars_dependant(var, loop_index_var)
        )
        iter_vals = self._get_loop_const_iter_vals(loop_index_var)

        # avoid unrolling very large loops (too many iterations and/or body statements)
        unroll_size = len(iter_vals) * sum(
            len(self.func_ir.blocks[l].body) for l in loop.body if l != loop.header
        )
        require(unroll_size < loop_unroll_limit)

        # start the unroll transform
        # no more GuardException since we can't bail out from this point
        self._unroll_loop(loop, loop_index_var, iter_vals)

        if is_container_update:
            self._remove_container_updates(self.func_ir.blocks[list(loop.entries)[0]])

        return True

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
                    new_nodes = guard(
                        self._try_remove_container_update, stmt, defined_containers
                    )
                    if new_nodes:
                        const_nodes.extend(new_nodes)
                        continue
                    # potential container update call that couldn't be handled in
                    # _try_remove_container_update()
                    fdef = guard(find_callname, self.func_ir, stmt.value)
                    if (
                        fdef
                        and len(fdef) == 2
                        and isinstance(fdef[1], ir.Var)
                        and fdef[1].name in defined_containers
                    ):
                        defined_containers -= self._equiv_vars[fdef[1].name]
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

    def _try_remove_container_update(self, stmt, defined_containers):
        """try to remove container update if possible.
        E.g. a = []; a.append(2) -> a = [2]
        Otherwise, raise GuardException.
        """
        # match container update call, e.g. a.append(2)
        fdef = find_callname(self.func_ir, stmt.value)
        require(
            fdef
            and len(fdef) == 2
            and isinstance(fdef[1], ir.Var)
            and fdef[1].name in defined_containers
        )
        require(isinstance(fdef[0], str))
        container_def = get_definition(self.func_ir, fdef[1])
        require(
            isinstance(container_def, ir.Expr)
            and container_def.op in ("build_list", "build_set", "call")
        )

        # get constant values of container before update
        # TODO(ehsan): support "build_map"
        if container_def.op in ("build_list", "build_set"):
            container_val = [
                get_const_value_inner(self.func_ir, v, self.arg_types, self.typemap)
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
        arg_vals = [
            get_const_value_inner(self.func_ir, v, self.arg_types, self.typemap)
            for v in stmt.value.args
        ]
        out_val = getattr(container_val, fdef[0])(*arg_vals)

        nodes = []

        # replace container variable in getattr with dummy to avoid use detection later
        # e.g. a.append -> dummy.append
        func_var_def = get_definition(self.func_ir, stmt.value.func)
        require(is_expr(func_var_def, "getattr"))
        dummy_val = [1] if container_def.op == "build_list" else {1}
        # no more GuardException from here on since IR is being modified
        func_var_def.value = _create_const_var(
            dummy_val, fdef[1].name, fdef[1].scope, fdef[1].loc, nodes
        )

        # update original container definition, e.g. a = [] -> a = [2]
        if container_def.op == "call" and find_callname(
            self.func_ir, container_def
        ) == ("set", "builtins"):
            # convert set() call into a build_set
            container_def.op = "build_set"
            container_def._kws = {"items": []}
        container_def.items = [
            _create_const_var(v, fdef[1].name, fdef[1].scope, fdef[1].loc, nodes)
            for v in container_val
        ]
        # replace update call with constant output, e.g. b = a.append(2) - > b = None
        self.func_ir._definitions[stmt.target.name].remove(stmt.value)
        stmt.value = _create_const_var(
            out_val, fdef[1].name, fdef[1].scope, fdef[1].loc, nodes
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

    def _get_enclosing_loop(self, var, label):
        """find enclosing loop for block 'label' if possible. Also return True if the
        loop updates a container.
        Otherwise, raise GuardException.
        """
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
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

    def _get_loop_const_iter_vals(self, ind_var):
        """get constant iteration values for loop given its index variable.
        Matches this call sequence generated by Numba
        index_var = pair_first(iternext(getiter(loop_iterations)))
        Raises GuardException if couldn't find constant values
        """
        pair_first_expr = get_definition(self.func_ir, ind_var)
        require(is_expr(pair_first_expr, "pair_first"))
        iternext_expr = get_definition(self.func_ir, pair_first_expr.value)
        require(is_expr(iternext_expr, "iternext"))
        getiter_expr = get_definition(self.func_ir, iternext_expr.value)
        require(is_expr(getiter_expr, "getiter"))
        return get_const_value_inner(
            self.func_ir, getiter_expr.value, self.arg_types, self.typemap
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
        if any(is_const_func_type(f) for f in val):
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


def _set_updated_container(varname, update_func, updated_containers, equiv_vars):
    """helper to set 'varname' and its aliases as updated containers"""
    updated_containers[varname] = update_func
    # make sure an updated container variable is always equivalent to itself since
    # assumed in _remove_container_updates()
    equiv_vars[varname].add(varname)
    for w in equiv_vars[varname]:
        updated_containers[w] = update_func
