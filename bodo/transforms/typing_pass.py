"""
Bodo type inference pass that performs transformations that enable typing of the IR
according to Bodo requirements (using partial typing).
"""

import copy
import itertools
import operator
import typing as pt
import warnings
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from numba.core import event, ir, ir_utils, types
from numba.core.compiler_machinery import register_pass
from numba.core.extending import register_jitable
from numba.core.inline_closurecall import inline_closure_call
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
import bodo.ir.filter as bif
import bodo.ir.iceberg_ext
from bodo.hiframes.dataframe_indexing import DataFrameILocType, DataFrameLocType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
from bodo.hiframes.pd_index_ext import DatetimeIndexType
from bodo.hiframes.pd_rolling_ext import RollingType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import PandasTimestampType
from bodo.hiframes.series_dt_impl import SeriesDatetimePropertiesType
from bodo.hiframes.series_str_impl import SeriesStrMethodType
from bodo.io.arrow_reader import ArrowReaderType
from bodo.io.helpers import get_table_iterator
from bodo.ir.filter import (
    build_filter_from_ir,
    supported_arrow_funcs_map,
    supported_funcs_map,
)
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.numba_compat import mini_dce
from bodo.sql_plan_cache import BodoSqlPlanCache
from bodo.utils.transform import (
    ReplaceFunc,
    compile_func_single_block,
    container_update_method_names,
    create_nested_run_pass_event,
    get_call_expr_arg,
    get_const_func_output_type,
    get_const_value_inner,
    get_runtime_join_filter_terms,
    replace_func,
    set_call_expr_arg,
    update_locs,
    update_node_list_definitions,
)
from bodo.utils.typing import (
    CONST_DICT_SENTINEL,
    BodoConstUpdatedError,
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    check_unsupported_args,
    dtype_to_array_type,
    gen_bodosql_case_func,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_str,
    handle_bodosql_case_init_code,
    is_bodosql_context_type,
    is_const_func_type,
    is_immutable,
    is_list_like_index_type,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    is_scalar_type,
    raise_bodo_error,
    unwrap_typeref,
)
from bodo.utils.utils import (
    find_build_tuple,
    get_getsetitem_index_var,
    is_array_typ,
    is_assign,
    is_call,
    is_call_assign,
    is_expr,
)

if pt.TYPE_CHECKING:
    from numba.core.utils import UniqueDict

    from bodo.ir.connector import Connector


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
        # flag indicating that loop unrolling has been tried at least once
        tried_unrolling = False
        # flag for when another transformation pass is needed (to avoid break before
        # next transform)
        needs_transform = False
        # Flag to indicate if some optimizations should be rerun once dead code
        # elimination has completed for possible further optimization.
        rerun_after_dce = True

        # Counter to keep track of the number of iterations that have passed without
        # a change to the IR.
        num_iterations_without_change = 0
        # Constant. Since updating the typemap doesn't count as a change,
        # there are situations where we may need several iterations
        # before everything is fully typed and we actually see a change
        # Therefore, whenever a transformation is required,
        # we attempt to run typing transforms this many times
        # before aborting.
        num_iterations_without_change_before_abort = 2
        while True:
            try:
                # set global partial typing flag, see comment above
                in_partial_typing = True
                typing_transform_required = False
                # Call into Numba's PartialTypeInference but with chrome tracing enabled properly.
                create_nested_run_pass_event(
                    PartialTypeInference.name(), state, super()
                )
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
                tried_unrolling,
            )
            ran_transform = True

            changed, needs_transform, tried_unrolling = typing_transforms_pass.run()
            num_iterations_without_change = (
                0 if changed else num_iterations_without_change + 1
            )

            rerun_after_dce = typing_transforms_pass.rerun_after_dce
            # transform pass has failed if transform was needed but IR is not changed.
            # This avoids infinite loop, see [BE-140]
            # We're adding one additional iteration, in order to handle a possible
            # edge case where a run of typing transforms does not change the IR, but
            # updates types
            if (
                num_iterations_without_change
                > num_iterations_without_change_before_abort
                and not changed
            ):
                break
            # can't be typed if IR not changed
            if not changed and not needs_transform:
                # error will be raised below if there are still unknown types
                break

        # make sure transformation has run at least once to handle cases that may not
        # throw typing errors like "df.B = v". See test_set_column_setattr
        rerun_typing = False
        # Was there a change after the typing step.
        changed_after_typing = False
        # Track if we may need an extra transform to produce
        # an error message.
        skipped_transform_in_typing = not ran_transform
        while rerun_after_dce or not ran_transform:
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
                tried_unrolling,
            )
            ran_transform = True
            (
                local_changed,
                needs_transform,
                tried_unrolling,
            ) = typing_transforms_pass.run()
            changed_after_typing = changed_after_typing or local_changed
            rerun_after_dce = typing_transforms_pass.rerun_after_dce
        # some cases need a second transform pass to raise the proper error
        # see test_df_rename::impl4
        if skipped_transform_in_typing and needs_transform:
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
                tried_unrolling,
            )
            (
                local_changed,
                needs_transform,
                tried_unrolling,
            ) = typing_transforms_pass.run()
            changed_after_typing = changed_after_typing or local_changed
        if skipped_transform_in_typing or changed_after_typing:
            # need to rerun type inference if the IR changed
            # see test_set_column_setattr
            rerun_typing = changed_after_typing or needs_transform

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
            except Exception:
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
            # clear _failed_cache of Bodo JIT calls to make them recompile and capture
            # the errors accurately.
            # If a JIT call fails during partial typing, the error is raised as a
            # BodoException to allow type inference iteration. But we need BodoError to
            # be raised for clear error messages.
            # see test_func_nested_jit_error
            # https://bodo.atlassian.net/browse/BE-2213
            for typ in state.typemap.values():
                if isinstance(typ, types.Dispatcher) and issubclass(
                    typ.dispatcher._compiler.pipeline_class, bodo.compiler.BodoCompiler
                ):
                    typ.dispatcher._compiler._failed_cache.clear()

            # run regular type inference again with _raise_errors=True to raise errors
            create_nested_run_pass_event(
                NopythonTypeInference.name(), state, NopythonTypeInference()
            )
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


unresolved_types = (None, types.unknown, types.undefined)


class TypingTransforms:
    """
    Transformations that enable typing of the IR according to Bodo requirements.

    Infer possible constant values (e.g. lists) using partial typing info and transform
    them to constants so that functions like groupby() can be typed properly.
    """

    def __init__(
        self,
        func_ir: ir.FunctionIR,
        typingctx,
        targetctx,
        typemap: "UniqueDict",
        calltypes,
        arg_types,
        _locals,
        flags,
        change_required,
        ran_transform,
        tried_unrolling,
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
        # DataFrames such as Arg(df_type), pd.DataFrame(), df[['A','B']]...
        # the use is conservative and doesn't assume complete info
        self.rhs_labels = {}
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
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
        # whether transform has run before
        self.ran_transform = ran_transform
        # whether loop unrolling has been tried at least once
        self.tried_unrolling = tried_unrolling
        self.changed = False
        # whether another transformation pass is needed (see _run_setattr)
        self.needs_transform = False
        # Flag indicating if we should try rerunning typing pass again after running DCE
        # to enable further optimizations
        self.rerun_after_dce = False
        # get Scope object for easier access
        assert len(self.func_ir.blocks) > 0, "Invalid empty function IR"
        self.scope = next(iter(self.func_ir.blocks.values())).scope

    def run(self):
        blocks = self.func_ir.blocks
        topo_order = find_topo_order(blocks)
        self._updated_containers, self._equiv_vars = _find_updated_containers(
            blocks, topo_order
        )
        for label in topo_order:
            block = blocks[label]
            self._working_body = []
            for i, inst in enumerate(block.body):
                self._replace_vars(inst)

                out_nodes = [inst]
                self.curr_loc = inst.loc

                # handle potential DataFrame set column here
                # df['col'] = arr
                if isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                    out_nodes = self._run_setitem(inst, label)
                elif isinstance(inst, ir.SetAttr):
                    out_nodes = self._run_setattr(inst, label)
                elif isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    self.rhs_labels[inst.value] = label
                    out_nodes = self._run_assign(inst, label)

                if isinstance(out_nodes, ReplaceFunc):
                    if out_nodes.pre_nodes is not None:
                        self._working_body.extend(out_nodes.pre_nodes)
                        self._update_definitions(out_nodes.pre_nodes)
                    self._handle_inline_func(out_nodes, inst, i, block)
                    # We use block labels in this pass (rhs_labels for finding df definition dominators)
                    # so need to start over when block structure changes.
                    # Returning tried_unrolling=False since control flow changes and need
                    # to try again.
                    return True, self.needs_transform, False

                self._working_body.extend(out_nodes)

                update_node_list_definitions(out_nodes, self.func_ir)
                for inst in out_nodes:
                    if is_assign(inst):
                        self.rhs_labels[inst.value] = label

            blocks[label].body = self._working_body

        # try loop unrolling if some const values couldn't be resolved
        if self._require_const:
            self._try_loop_unroll_for_const()
            self.tried_unrolling = True

        # try unrolling a loop with constant range if everything else failed
        if self.change_required and not self.changed and not self.needs_transform:
            self._try_unroll_const_loop()
            self.tried_unrolling = True

        # Remove any transformed variables that are not used anymore
        # since cases like agg dicts may not be type stable
        mini_dce(self.func_ir)

        return self.changed, self.needs_transform, self.tried_unrolling

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return

    def _handle_inline_func(
        self, replacement_func, orig_inst, orig_inst_idx, cur_block
    ):
        """
        Helper function used to handle ReplaceFunc nodes in run().
        This is largely copied from the code that handles this in series pass.

        Handles inlining the call, appending the remaining instructions in the
        current block to the working body, and then updating the current block body
        """

        # Replace inst.value with a call with target args
        # as expected by inline_closure_call (the only supported format is ir.Call).
        # orig_inst is replaced anyways so it is ok to change it.
        orig_inst.value = ir.Expr.call(
            ir.Var(cur_block.scope, mk_unique_var("dummy"), orig_inst.loc),
            replacement_func.args,
            (),
            orig_inst.loc,
        )

        # Update block body with nodes that are processed already (_working_body)
        cur_block.body = self._working_body + cur_block.body[orig_inst_idx:]

        # use callee_validator mechanism to run untyped pass as a workaround
        def run_untyped_pass(new_ir):
            untyped_pass = bodo.transforms.untyped_pass.UntypedPass(
                new_ir, self.typingctx, replacement_func.arg_types, {}, {}, self.flags
            )
            untyped_pass.run()

        callee_blocks, _ = inline_closure_call(
            self.func_ir,
            replacement_func.glbls,
            cur_block,
            len(self._working_body),
            replacement_func.func,
            callee_validator=run_untyped_pass,
        )
        for c_block in callee_blocks.values():
            c_block.loc = self.curr_loc
            update_locs(c_block.body, self.curr_loc)

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)

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
                idx_typ == bodo.types.string_type
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

        # If we have a constant filter it cannot be a boolean filter.
        if not isinstance(rhs.index, ir.Var):
            return nodes + [assign]

        # detect if filter pushdown is possible and transform
        # e.g. df = pd.read_parquet(...); df = df[df.A > 3]
        self._try_apply_filter_pushdown(assign, label)

        nodes.append(assign)
        return nodes

    def _try_apply_filter_pushdown(self, assign, label):
        """Apply filter pushdown to filter statement if possible

        Args:
            assign (ir.Assign): input filter statement (getitem or table_filter)
            label (int): block label
        """

        rhs = assign.value

        if is_expr(rhs, "getitem"):
            in_table_var = rhs.value
            index_var = rhs.index
        else:
            assert is_call(rhs), "_try_apply_filter_pushdown call expected"
            in_table_var = rhs.args[0]
            index_var = rhs.args[1]

        index_typ = self.typemap.get(index_var.name, None)
        # If we cannot determine the type we will try again later.
        if index_typ in (None, types.unknown, types.undefined):
            self.needs_transform = True
            return

        index_def = guard(get_definition, self.func_ir, index_var)
        # BodoSQL generates wrappers around exprs like Series.values that need removed
        index_def = self._remove_series_wrappers_from_def(index_def)
        value_def = guard(get_definition, self.func_ir, in_table_var)

        # If our filter is a boolean array or series then we can perform filter pushdown.
        if (
            bodo.utils.utils.is_array_typ(index_typ, True)
            and index_typ.dtype == types.boolean
        ):
            pushdown_results = guard(
                self._follow_patterns_to_table_def, in_table_var, self.func_ir
            )
            if pushdown_results is not None:
                value_def, used_dfs, skipped_vars = pushdown_results
                node_res = guard(self._get_filter_read_and_def_nodes, value_def)
                if node_res is not None:
                    table_def_node, read_node = node_res
                    working_body = guard(
                        self._try_filter_pushdown,
                        assign,
                        self._working_body,
                        self.func_ir,
                        table_def_node,
                        read_node,
                        used_dfs,
                        skipped_vars,
                        index_def,
                        label,
                    )
                    # If this function returns a list we have updated the working body.
                    # This is done to enable updating a single block that is not yet being processed
                    # in the working body.
                    if working_body is not None:
                        self._working_body = working_body

    def _follow_patterns_to_table_def(
        self, in_table_var: ir.Var, func_ir: ir.FunctionIR
    ) -> tuple[ir.Inst, dict[str, ir.Inst], set[str]]:
        """
        Takes an ir.Assign that creates a DataFrame/Table used in filter pushdown and
        converts it to the "Expression" that defined the DataFrame/Table.
        A DataFrame that can be used in filter pushdown is required
        to be an ir.Expr that is a call to "init_dataframe". A Table is not
        subject to these constraints so long as the table can be traced back to
        a SqlReader, IcebergReader, or ParquetReader node.

        However, in some code patterns BodoSQL may generate additional code beyond the init_dataframe that
        is otherwise unused. For example, BodoSQL handles unsupported types that can be cast to supported
        types with `__bodosql_replace_columns_dummy`. To handle these patterns this function traverses the IR
        looking for specific layouts in the IR. When these are met we continue to travel up the IR back to the original
        init_dataframe.

        To support this case we create a dictionary, `used_dfs`. These represent the uses of the DataFrame that would typically
        prevent filter pushdown but are accepted as allowed uses of the original DataFrame. In addition, to avoid dangerous
        false positives due to similar looking user code, we check these DataFrames for usage so we do not illicitly perform
        filter pushdown.

        Since some specific patterns can be complicated but include strong guarantees, we have a third dictionary
        called skipped_vars. These are variables that can be explicitly skipped because we know some strong invariant.

        Args:
            in_table_var (ir.Assign): Input table/dataframe variable
            func_ir (ir.FunctionIR): Function IR

        Returns:
            Tuple[ir.Inst, Dict[str, ir.Inst], Set[str]]: Tuple of values collected. These are:
                - ir instruction that should match creating the DataFrame
                - ir Variables that need to be tracked and the original definition
                - ir Variables to skip. These require a VERY specific pattern.
        """
        # TODO(Nick): Refactor the code in this function with clearer variable names and explicit IR examples.
        value_def = get_definition(func_ir, in_table_var)
        # Intermediate DataFrames that need to be checked.
        # Maps dataframe names to their definitions
        used_dfs = {in_table_var.name: value_def}
        skipped_vars = set()

        # If we have any df.loc calls that load all rows, they will appear
        # before the init_dataframe. We can find all rows with a
        # static getitem with a slice of all none for the rows.
        #
        # i.e.
        # df1 = df.loc[:, ["b", "c"]]

        empty_slice = slice(None, None, None)

        # TODO: Currently, this code will only do work for BodoSQL, as the df.loc's will have already
        # been replaced by typing pass if the code originates from Bodo.
        # It is an open issue to refactor this df.loc support into a general support so that we can see
        # The benefits for Bodo:
        # https://bodo.atlassian.net/browse/BE-1522
        while (
            is_expr(value_def, "static_getitem")
            and isinstance(value_def.index, tuple)
            and len(value_def.index) > 0
            and value_def.index[0] == empty_slice
        ):  # pragma: no cover
            # pragma: no cover is due to the fact that bodo will never enter this branch as is
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

        # Iceberg read with _bodo_merge_into=True generates a tuple. As a result
        # we will need to traverse this tuple to the original init_dataframe if it
        # exists and track intermediate values generated (which need to be handled).
        # SQL streaming Snowflake read also creates static_getitem but we just need to
        # match the specific codegen pattern generated.
        if is_expr(value_def, "static_getitem"):
            # at this stage the IR looks like:
            # $actual_val = call init_dataframe ...
            # $tuple = build_tuple(items=[$actual_val, ...])
            # $alias = $tuple
            # $iterator = exhaust_iter($alias)
            # $orig_val = static_getitem($iterator, index=0)
            # Note we could still have __bodosql_replace_columns_dummy after
            # init_dataframe (TODO(Nick): Remove)
            tuple_index = value_def.index
            exhaust_iter_def = get_definition(func_ir, value_def.value)
            require(is_expr(exhaust_iter_def, "exhaust_iter"))
            # An exhaust iter can be used only once since we are tracking the usage
            # of the variable output we skip this variable to avoid issues with the
            # other members of the tuple.
            used_name = value_def.value.name
            skipped_vars.add(used_name)
            # Find the build tuple
            tuple_def = get_definition(func_ir, exhaust_iter_def.value)
            if is_expr(tuple_def, "build_tuple"):
                # Add the build tuple to tracking
                used_name = exhaust_iter_def.value.name
                used_dfs[used_name] = tuple_def
                value_def = get_definition(func_ir, tuple_def.items[tuple_index])
                # Add the original DF to tracking
                used_name = tuple_def.items[tuple_index].name
                used_dfs[used_name] = value_def
            else:
                require(
                    find_callname(self.func_ir, tuple_def)
                    == ("read_arrow_next", "bodo.io.arrow_reader")
                )

        if is_call(value_def) and guard(find_callname, func_ir, value_def) == (
            "__bodosql_replace_columns_dummy",
            "bodo.hiframes.dataframe_impl",
        ):  # pragma: no cover
            # This is the new BodoSQL path.
            # args of __bodosql_replace_columns_dummy are (df, col_names_to_replace, cols_to_replace_with)
            # the dataframe argument is always initialized by a call to init_dataframe
            # This is init dataframe call is what we want to back to _try_filter_pushdown.
            # NOTE: We skip the old value_def because this will break filter pushdown. The reason is
            # that the existing implementation will use the Series values before the filter to enable type
            # casting. However this function call never uses the data, so we do not track it. This function
            # is required to be generated ONLY by BodoSQL, which is why we are okay with this assumption.
            used_name = value_def.args[0].name
            skipped_vars.add(used_name)
            value_def = get_definition(func_ir, value_def.args[0])

        return value_def, used_dfs, skipped_vars

    def _get_filter_read_and_def_nodes(self, value_def) -> tuple[ir.Expr, "Connector"]:
        """Find table definition node and read node for filter pushdown. value_def is
        the definition of the DataFrame/Table being filtered and could be
        init_dataframe, reader, or static_getitem node.
        Performs several checks to make sure a valid filter pushdown is happening.

        Args:
            value_def (ir.Expr): definition of the DataFrame/Table that is filtered

        Returns:
            tuple(ir.Expr, ir.Expr): table definition node and reader node
        """
        call_name = guard(find_callname, self.func_ir, value_def)
        if call_name == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            # avoid empty dataframe
            require(len(value_def.args) > 0)
            data_def = get_definition(self.func_ir, value_def.args[0])
            assert is_expr(data_def, "build_tuple"), (
                "invalid data tuple in init_dataframe"
            )

            # table_def_node is the IR node that creates the input table, which is
            # read_arrow_next() in the streaming read case but otherwise same as read node.
            table_def_node = read_node = get_definition(self.func_ir, data_def.items[0])
            # TODO: Change to Walrus Operator once Cython 3 is supported
            arrow_iter_name = guard(get_table_iterator, read_node, self.func_ir)
            if arrow_iter_name:
                read_node = get_definition(self.func_ir, arrow_iter_name)
            require(
                all(
                    get_definition(self.func_ir, v) == table_def_node
                    for v in data_def.items
                )
            )
        else:
            table_def_node = read_node = value_def
            arrow_iter_name = guard(get_table_iterator, read_node, self.func_ir)
            if arrow_iter_name:
                read_node = get_definition(self.func_ir, arrow_iter_name)

        require(
            isinstance(
                read_node,
                (
                    bodo.ir.parquet_ext.ParquetReader,
                    bodo.ir.iceberg_ext.IcebergReader,
                    bodo.ir.sql_ext.SqlReader,
                ),
            )
        )
        if isinstance(read_node, bodo.ir.sql_ext.SqlReader):
            # Filter pushdown is only supported for snowflake and iceberg
            # right now.
            require(read_node.db_type == "snowflake")
        # If the reader is streaming, and thus implied from BodoSQL
        # and the filters are already present in the reader, then we
        # can assume BodoSQL has already done the filter pushdown.
        # The same is true for a limit from BodoSQL.
        # Bodo doesn't need to attempt further
        if read_node.is_streaming:
            is_sql_node = isinstance(read_node, bodo.ir.sql_ext.SqlReader)
            require(
                read_node.filters is None
                and (not is_sql_node or read_node.limit is None)
            )

        return table_def_node, read_node

    def _try_filter_pushdown(
        self,
        assign,
        working_body,
        func_ir,
        table_def_node,
        read_node: "Connector",
        used_dfs: dict[str, ir.Inst],
        skipped_vars: set[str],
        index_def,
        label: int,
    ):
        """detect filter pushdown and add filters to ParquetReader or SQLReader IR nodes if possible.

        working_body is in the in progress list of statements that should be updated with any filter reordering.
        A new working_body is returned if this is successful. func_ir is FunctionIR object containing the blocks
        with all relevant code.

        used_dfs is a dictionary of intermediate DataFrames -> initialization that should be tracked
        to ensure they aren't reused

        skipped_vars is a set of IR variables that can be skipped in reordering

        Throws GuardException if not possible.
        """

        # make sure all filters have the right form
        # If we don't have a binary operation, then we just pass
        # the single def as the index_def and set the lhs_def
        # and rhs_def to none.
        if self._is_logical_op_filter_pushdown(index_def, func_ir):
            lhs_def = get_definition(func_ir, get_binop_arg(index_def, 0))
            lhs_def = self._remove_series_wrappers_from_def(lhs_def)
            rhs_def = get_definition(func_ir, get_binop_arg(index_def, 1))
            rhs_def = self._remove_series_wrappers_from_def(rhs_def)
        else:
            lhs_def = None
            rhs_def = None

        # assign could be getitem or table_filter()
        df_var = assign.value.args[0] if is_call_assign(assign) else assign.value.value
        new_ir_assigns = []
        filters = self._get_partition_filters(
            index_def,
            df_var,
            lhs_def,
            rhs_def,
            func_ir,
            # SQL generates different operators than pyarrow
            read_node,
            # Some filters may require generating additional IR variables.
            # This will be updated in place.
            new_ir_assigns,
        )
        # Append any new assigns to the working body
        working_body = working_body + new_ir_assigns
        self._check_non_filter_df_use_after_filter(
            set(used_dfs.keys()), assign, func_ir
        )
        new_working_body, is_ir_reordered = self._reorder_filter_nodes(
            table_def_node,
            read_node,
            index_def,
            used_dfs,
            skipped_vars,
            filters,
            working_body,
            func_ir,
            label,
        )

        old_filters = read_node.filters

        # Combine old and new filters to avoid duplicates somehow
        # Also checks if filter is contained in the old filters
        # likely indicating we reached a static state
        if old_filters is None:
            filters_changed = True
        else:
            combined_filters = bif.Op("AND", old_filters, filters)
            filters = bif.SimplifyFilterVisitor().visit(combined_filters)
            filters_changed = filters != old_filters

        # Update the logs with the successful filter pushdown.
        # (no exception was raise until this end point so filters are valid)
        if bodo.user_logging.get_verbose_level() >= 1:
            msg = "Filter pushdown successfully performed. Moving filter step:\n%s\ninto IO step:\n%s\n"
            filter_source = index_def.loc.strformat()
            read_source = read_node.loc.strformat()
            bodo.user_logging.log_message(
                "Filter Pushdown",
                msg,
                filter_source,
                read_source,
            )
        # Set node filters (no exception was raise until this end point
        # so filters are valid)
        read_node.filters = filters
        # Merge into only filters by file not within rows, so we cannot remove the filter.
        keep_filter = (
            isinstance(read_node, bodo.ir.iceberg_ext.IcebergReader)
            and read_node.is_merge_into
        )
        if not keep_filter:
            # remove filtering code since not necessary anymore
            # assign could be getitem or table_filter()
            assign.value = (
                assign.value.args[0] if is_call_assign(assign) else assign.value.value
            )
            self.rerun_after_dce = True

        # Mark the IR as changed if modified the IR or filters at all.
        # This is important because if we don't delete the filter and an error
        # in the code requires a transformation we will wrongfully believe the IR
        # has changed. See test_merge_into_filter_compilation_errors.py::test_requires_transform.
        self.changed = self.changed or (
            is_ir_reordered or filters_changed or (not keep_filter)
        )
        # Return the updates to the working body so we can modify blocks that may not
        # be in the working body yet.
        return new_working_body

    def _check_non_filter_df_use_after_filter(self, df_names, assign, func_ir):
        """make sure the chain of used dataframe variables are not used AFTER filtering in the
        program. e.g. df2 = df[...]; A = df.A
        Assumes that Numba renames variables if the same df name is used later. e.g.:
            df2 = df[...]
            df = ....  # numba renames df to df.1

        This DOES NOT detect any extra uses of the DataFrame before the filter. Those will be
        captured by _reorder_filter_nodes.

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
                require(self._is_not_filter_df_use(df_names, stmt))

    def _is_not_filter_df_use(self, df_names, stmt):
        """Helper function for _check_non_filter_df_use_after_filter, that checks a particular
        statement doesn't use any of the DataFrames in df_names, for purposes of
        performing filter pushdown.

        Args:
            df_names (set(String)): Set of DataFrames that can't be used
            stmt (IR.Stmt): statement to check for usage

        Returns:
            True/False
        """

        """In BodoSQL, when doing a MERGE INTO operation with iceberg,
        we generate code that looks something like:

            dest_df = pd.read_sql(*args*, _bodo_with_orig_file_metadata=True)
            _____
            (Some code that does a bunch of joins with the dest_df,
            and creates the delta table)
            _____
            writeback_df = do_delta_merge_with_target(dest_df, delta_df)

        The issue is that, we don't want do_delta_merge_with_target to count
        as a use of dest_df, since we want it to be fully filtered at the
        input to do_delta_merge_with_target.

        Since we don't use this function for any other purpose,
        we can completely ignore the use of the dest_df in the function,
        for determining what filters can be applied.
        (no columns should be pruned, as we'll need to write back every column)"""
        if isinstance(stmt, ir.Assign):
            call_name = guard(find_callname, self.func_ir, stmt.value, self.typemap)
            if call_name in (
                (
                    "do_delta_merge_with_target",
                    "bodosql.libs.iceberg_merge_into",
                ),
                ("pushdown_safe_init_df", "bodo.hiframes.pd_dataframe_ext"),
            ):
                # Only need to check if the delta table is in df names.
                # To be totally correct, we could check for uses of the delta table argument,
                # but we don't need to since it will never be used after
                # do_delta_merge_with_target
                return True
        return all(v.name not in df_names for v in stmt.list_vars())

    def _find_target_node_location_for_filtering(
        self,
        block_body: list[ir.Stmt],
        target_node: ir.Stmt,
        filter_nodes: set[ir.Expr],
        filter_vars: set[ir.Var],
        non_filter_vars: set[ir.Var],
        related_vars: set[ir.Var],
        skipped_vars: set[ir.Var],
        df_names: set[str],
        used_dfs: dict[str, ir.Expr],
    ) -> int:
        """For a given block_body, find the location within
        the IR at which the target_node is located for the
        purposes of reordering filters. Throughout this iteration
        several restrictions on variable usage are enforced, effectively
        requiring there can't be any overlap between filter_vars and non_filter_vars.
        The enforce these, the sets filter_vars, non_filter_vars, related_vars, and
        skipped_vars, as well as the dictionary used_dfs will be updated throughout
        the execution of this function.

        Args:
            block_body (List[ir.Stmt]): The body of an IR block to search.
            target_node (ir.Stmt): The IR statements we are trying to find
            filter_nodes (Set[ir.Expr]): The nodes that are part of the filter and should
                be skipped.
            filter_vars[in, out] (Set[ir.Var]): Variables that are used in computing
                the filter.
            non_filter_vars[in, out] (Set[ir.Var]): Variables that are used in computation
                unrelated to the filter. To be conservative these cannot overlap with filter variables.
            related_vars[in, out] (Set[ir.Var]): Set of variables that are related to the use of the filter.
                These are variables that are intermediate steps for computing the filter that need to be
                checked for illicit use.
            skipped_vars[in, out] (Set[ir.Var]): Set of variables that are explicitly skipped because
                of a prior pattern matching step. Any directly assignments to these variables are also
                updated.
            df_names (Set[str]): The variable names for known DataFrames that
                can be skipped.
            used_dfs[in, out] (Dict[str, ir.Expr]): A mapping between each DataFrame
                variable involved with pushdown and the actual IR expression.


        Raises:
            GuardException: Is there any overlap between filter and
                non-filter variables (which means we cannot safely do pushdown)
                or have types not finalized yet in a way that interferes with
                pushdown.

        Returns:
            int: The line in block body at which the target_node can be found.
        """
        i = 0  # will be set to ParquetReader node's reversed index
        for stmt in reversed(block_body):
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
                if isinstance(stmt.value, ir.Var):
                    # If we have an IR variable update df_names and used_dfs
                    # to match the target. This is necessary because there could
                    # be intermediate assignments between the lhs given in used_dfs
                    df_names.add(stmt.value.name)
                    used_dfs[stmt.value.name] = used_dfs[stmt.target.name]
                continue
            # Ignore variables whose creation we are directly skipping.
            if is_assign(stmt):
                if stmt.target.name in skipped_vars:
                    continue
                # For direct assignments (a = b), update the skipped vars
                elif isinstance(stmt.value, ir.Var) and stmt.value.name in skipped_vars:
                    skipped_vars.add(stmt.target.name)
                    continue

            # avoid nodes before the reader
            if stmt is target_node:
                break
            stmt_vars: set[str] = {v.name for v in stmt.list_vars()}

            # make sure df is not used before filtering
            if not (stmt_vars & related_vars):
                # df_names is a non-empty set, so if the intersection
                # is non empty then a df_name is in stmt_vars
                require(not (df_names & stmt_vars))
            else:
                related_vars.update(stmt_vars - df_names)

            # If the target of an assignment is a filter_var, then
            # the inputs are involved with computing filters so they must
            # be filter_var's too
            if is_assign(stmt) and stmt.target.name in filter_vars:
                filter_vars.update(stmt_vars)
                continue

            # For BoundFunction's, we need to check if the original bounded value
            # is a filter var, so add it to stmt_vars
            if is_call_assign(stmt):
                # This is tested in test_snowflake_filter_pushdown_edgecase,
                # which doesn't run on PR CI, so the pragma is needed
                if stmt.value.func.name not in self.typemap:  # pragma: no cover
                    self.needs_transform = True
                    raise GuardException
                elif isinstance(
                    self.typemap[stmt.value.func.name], types.BoundFunction
                ):
                    def_loc = get_definition(self.func_ir, stmt.value.func)
                    stmt_vars.update(v.name for v in def_loc.list_vars())

            # Otherwise, if the stmt uses a filter var and the filter_var is
            # not immutable, assume that all vars are involved and must be filter_var's
            # If filter_var is immutable, then don't need to assume that
            used_filter_vars: set[str] = stmt_vars & filter_vars

            # This is a defensive check, and isn't expected to be hit
            # so we need the pragma to let coverage pass
            for fvar in used_filter_vars:
                if fvar not in self.typemap:  # pragma: no cover
                    self.needs_transform = True
                    raise GuardException

            if used_filter_vars and any(
                not is_immutable(self.typemap[fvar]) for fvar in used_filter_vars
            ):
                filter_vars.update(stmt_vars)
            else:
                non_filter_vars.update(stmt_vars - filter_vars)

        require(not (filter_vars & non_filter_vars))
        return len(block_body) - i

    @staticmethod
    def _move_filter_nodes(
        block_body: list[ir.Stmt],
        reader_ind: int,
        filter_nodes: set[ir.Expr],
        filter_vars: set[ir.Var],
    ) -> tuple[list[ir.Stmt], list[ir.Stmt]]:
        """
        Given a block that should be reordered based on filter nodes, splits the block into
        two lists, one containing all the nodes before the reader and any reordered filters,
        and one containing the reader and any nodes that remain after it.

        Args:
            block_body (List[ir.Stmt]): The body of an IR block to search.
            reader_ind (int): The location in block_body of the reader node.
                All filter variables must be moved before this.
            filter_nodes (Set[ir.Expr]): The nodes that are part of the filter and should
                not be reordered.
            filter_vars (Set[ir.Var]): Variables that are used in computing
                the filter and should be reordered.

        Returns:
            Tuple[List[ir.Stmt], List[ir.Stmt]]: Returns two lists:
                - A list containing all statements before the reader
                  and the reordered filter nodes.
                - A list containing the reader statement and all statements
                  that remain after it.
        """
        # move IR nodes for filter expressions before the reader node
        new_body = block_body[:reader_ind]
        non_filter_nodes = []
        for i in range(reader_ind, len(block_body)):
            stmt = block_body[i]
            # ignore dataframe filter expression node
            if is_assign(stmt) and stmt.value in filter_nodes:
                non_filter_nodes.append(stmt)
                continue

            # Should only be true if:
            # - Target of stmt is a filter_var
            # - A mutable filter var was used in stmt
            # In both cases, we all all used vars as filter_vars
            # In the case that an immutable filter var is used in the stmt
            # we don't need to move the stmt. The definition will be moved up
            # but that should be safe
            if len(stmt.list_vars()) > 0 and all(
                v.name in filter_vars for v in stmt.list_vars()
            ):
                # NOTE: length check is for Jump nodes which don't have any variables
                new_body.append(stmt)
            else:
                non_filter_nodes.append(stmt)

        return new_body, non_filter_nodes

    def _reorder_filter_nodes(
        self,
        table_def_node,
        read_node,
        index_def,
        used_dfs,
        skipped_vars,
        filters,
        working_body: list[ir.Stmt],
        func_ir: ir.FunctionIR,
        label: int,
    ):
        """reorder nodes that are used for Parquet/SQL partition filtering to be before the
        Reader node (to be accessible when the Reader is run).

        df_names is a set of variables that need to be tracked to perform the filter pushdown.
        table_def_node is read_arrow_next() in streaming case, otherwise it's read_node.

        label is the current basic block number being processed.

        Categorizes all statements after the read node as either filter variable nodes
        or not. For example, given df["B"] == a, any node that is used for
        computing 'a' is filter variable node. The node df["B"] == a itself is not in
        this set.
        Moves the filter variable nodes before
        the read node to enable filter pushdown.

        Throws GuardException if not possible.
        """
        # e.g. [[("a", "0", ir.Var("val"))]] -> {"val"}
        filter_vars: set[str] = {
            v.name for v in bodo.ir.connector.get_filter_vars(filters)
        }

        # data array/table variables should not be used in filter expressions directly
        non_filter_vars = {v.name for v in table_def_node.list_vars()}

        # find all variables that are potentially used in filter expressions after the
        # reader node
        # make sure they don't overlap with other nodes (to be conservative)
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

        pq_ind = self._find_target_node_location_for_filtering(
            working_body,
            table_def_node,
            filter_nodes,
            filter_vars,
            non_filter_vars,
            related_vars,
            skipped_vars,
            df_names,
            used_dfs,
        )
        new_body, non_filter_nodes = self._move_filter_nodes(
            working_body, pq_ind, filter_nodes, filter_vars
        )

        # Check if the code has changed. There is only 1 change that
        # can occur, a node is moved from later into the IR earlier
        # via new body. This changes will be found in location
        #  new_body[pq_ind: len(new_body)]. As a result we can just
        # check this code for changes
        start_idx, end_idx = pq_ind, len(new_body)
        changed = working_body[start_idx:end_idx] != new_body[start_idx:end_idx]

        # update current basic block with new stmt order
        if read_node == table_def_node:
            new_working_body = new_body + non_filter_nodes
        else:  # pragma: no cover
            # The filter pushdown structure requires the read is located at the front
            # of the block to avoid accidentally moving any unrelated nodes via
            # new_body. This is enforced in the current codegen where read_arrow_next
            # should always be the first statement of the block.
            require(pq_ind == 0)
            # Overview of this path:
            # https://bodo.atlassian.net/wiki/spaces/B/pages/1412366337/Dictionary-Encoding+Related+BodoSQL+Streaming+Filter+Pushdown+Changes
            # move filter nodes before the actual read IR node, not read_arrow_next()
            new_working_body = non_filter_nodes
            # new_body must consist of only the filter nodes (there are no nodes before
            # the read and the only moved nodes must be part of the filter). Assuming
            # the codegen pattern is supported we will move these nodes to the IR block
            # with the actual Reader call.
            filter_nodes = new_body
            read_found = False
            # read node is before the while loop so it's in a predecessor block
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            new_target_block = []
            new_target_label = -1
            preds = self._get_reorder_filter_predecessor_labels(
                label,
                cfg,
                func_ir,
                read_node,
                filter_nodes,
                filter_vars,
                non_filter_vars,
                related_vars,
                skipped_vars,
                df_names,
                used_dfs,
            )
            for pred in preds:
                body = func_ir.blocks[pred].body
                # NOTE: Numba as of 0.59 may generate an indirect self-loop with an
                # extra empty block
                if (pred == label) or (
                    len(body) == 1
                    and isinstance(body[0], ir.Jump)
                    and body[0].target == label
                ):
                    # Ignore self-loops.
                    continue
                # Simplify the filter pushdown requirements. There can
                # only be one path we select.
                require(new_target_label == -1)
                pq_ind = self._find_target_node_location_for_filtering(
                    body,
                    read_node,
                    filter_nodes,
                    filter_vars,
                    non_filter_vars,
                    related_vars,
                    skipped_vars,
                    df_names,
                    used_dfs,
                )
                # We only support a single predecessor, so we
                # must find the node.
                require(body[pq_ind] == read_node)
                new_body, non_filter_nodes = self._move_filter_nodes(
                    body, pq_ind, filter_nodes, filter_vars
                )
                start_idx, end_idx = pq_ind, len(new_body)
                # Update changed
                changed = (
                    changed or body[start_idx:end_idx] != new_body[start_idx:end_idx]
                )
                # Note: new_body already contains all nodes before the reader.
                new_target_block = new_body + filter_nodes + non_filter_nodes
                new_target_label = pred
                read_found = True
            assert read_found, (
                "_reorder_filter_nodes: read node not found in streaming I/O"
            )
            require(new_target_label != -1)
            func_ir.blocks[new_target_label].body = new_target_block

        return new_working_body, changed

    def _get_reorder_filter_predecessor_labels(
        self,
        label,
        cfg,
        func_ir,
        read_node,
        filter_nodes,
        filter_vars,
        non_filter_vars,
        related_vars,
        skipped_vars,
        df_names,
        used_dfs,
    ):
        """Get predecessor basic block labels for reordering filter nodes in streaming
        read case.
        Handles the Python 3.14 case where there is an extra basic block inserted
        before the loop that contains initialization code.
        """
        preds = [l for l, _ in cfg.predecessors(label)]

        # Handle Python 3.14 case with extra basic block before loop basic block
        # that handles variable initialization.
        # See test_batched_read_agg for example.
        if len(preds) == 1:
            # Match Python 3.14 pattern of basic header block dominating the loop body
            # E.g. 0->1, 1->2,3, 2->1 (1 is header block, 2 is loop block)
            dom_label = preds[0]
            label_successors = [l for l, _ in cfg.successors(label)]
            dom_preds = [l for l, _ in cfg.predecessors(dom_label)]

            if (
                len(label_successors) == 1
                and label_successors[0] == dom_label
                and len(dom_preds) == 2
                and self._find_target_node_location_for_filtering(
                    func_ir.blocks[dom_label].body,
                    read_node,
                    filter_nodes,
                    filter_vars,
                    non_filter_vars,
                    related_vars,
                    skipped_vars,
                    df_names,
                    used_dfs,
                )
                == 0
            ):
                # NOTE: _find_target_node_location_for_filtering() has side effects of
                # updating data structures with the passed block body
                return dom_preds

        return preds

    def _get_filter_nodes(self, index_def, func_ir):
        """find ir.Expr nodes used in filtering output dataframe directly so they can
        be excluded from filter dependency reordering
        """
        # e.g. (df["A"] == 3) | (df["A"] == 4)
        if self._is_logical_not_filter_pushdown(index_def, func_ir):
            nodes = self._get_filter_nodes(
                self._remove_series_wrappers_from_def(
                    get_definition(func_ir, get_unary_arg(index_def))
                ),
                func_ir,
            )
            return {index_def} | nodes
        if self._is_or_filter_pushdown(
            index_def, func_ir
        ) or self._is_and_filter_pushdown(index_def, func_ir):
            left_nodes = self._get_filter_nodes(
                self._remove_series_wrappers_from_def(
                    get_definition(func_ir, get_binop_arg(index_def, 0))
                ),
                func_ir,
            )
            right_nodes = self._get_filter_nodes(
                self._remove_series_wrappers_from_def(
                    get_definition(func_ir, get_binop_arg(index_def, 1))
                ),
                func_ir,
            )
            return {index_def} | left_nodes | right_nodes
        return {index_def}

    def _negate_column_filter(self, old_filter: bif.Op) -> bif.Filter:
        """Negate an individual filter by wrapping it in the NOT operator.
        If we

        Args:
            old_filter (Tuple[str, str, ir.Var]): The filter to be negated

        Returns:
            bif.Filter: The new filter with the operator negated.
        """
        require(isinstance(old_filter, bif.Op))
        if old_filter.op.lower() == "not":
            # NOT(NOT X) = X, so we just remove the previous NOT.
            return old_filter.args[0]

        return bif.Op("NOT", old_filter)

    def _check_comparison_typing(
        self,
        col_arr_type: types.ArrayCompatible,
        scalar_var,
        read_node: "Connector",
    ) -> None:
        # If this is parquet we need to verify this is a filter we can process.
        if isinstance(read_node, bodo.ir.parquet_ext.ParquetReader):
            lhs_scalar_typ = bodo.utils.typing.element_type(col_arr_type)
            require(scalar_var.name in self.typemap)
            rhs_scalar_typ = self.typemap[scalar_var.name]
            # Only apply filter pushdown if its safe to use inside arrow
            require(
                bodo.utils.typing.is_common_scalar_dtype(
                    [lhs_scalar_typ, rhs_scalar_typ]
                )
                or bodo.utils.typing.is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ)
            )

    def _get_column_filter(
        self,
        col_def: ir.Expr,
        func_ir: ir.FunctionIR,
        df_var: ir.Var,
        df_col_names,
        df_col_types,
        is_sql_op: bool,
        read_node: "Connector",
        new_ir_assigns: list[ir.Stmt],
    ) -> bif.Filter:
        """
        Function used by _get_partition_filters to extract filters
        related to columns with boolean data. Returns a single filter
        comparing the boolean column to True.

        Raises Guard exception if this is not possible.

        Args:
            col_def (ir.Expr): The IR expression for the column. There may be a transformation
                function on the column.
            func_ir (ir.FunctionIR): The function IR used for traversing definitions and calls.
            df_var (ir.Var): The DataFrame variable to check for columns
            df_col_names (N Tuple[str, ...]): A tuple of column names for the DataFrame.
            is_sql_op (bool): Should the equality operator have SQL or arrow syntax.
            new_ir_assigns (List[ir.Stmt]): List of statements to add the IR when new IR variables
                must be created. This is update with the True variable.

        Returns:
             bif.Filter: The filter for a column with a boolean output type.
        """

        colname, col_typ = self._get_col_name(
            col_def,
            df_var,
            df_col_names,
            df_col_types,
            func_ir,
            read_node,
        )

        # This is a defensive check, and isn't expected to be hit
        # so we need the pragma to let coverage pass
        if df_var.name not in self.typemap:  # pragma: no cover
            self.needs_transform = True
            raise GuardException
        # Verify that the column has a boolean type.
        require(
            bodo.utils.utils.is_array_typ(col_typ, False)
            and col_typ.dtype == types.boolean
        )

        # Generate a == TRUE for the column. This allows using the partition filters in
        # Arrow.
        # Generate the True variable.
        # Create a new IR Expr for the constant pattern.
        expr_value = ir.Const(True, col_def.loc)
        # Generate a variable from the Expr
        new_name = mk_unique_var("true_var")
        new_var = ir.Var(self.scope, new_name, col_def.loc)
        new_assign = ir.Assign(target=new_var, value=expr_value, loc=col_def.loc)
        # Append the assign so we update the IR.
        new_ir_assigns.append(new_assign)
        # Update the definitions. This is safe since the name is unique.
        func_ir._definitions[new_name] = [expr_value]
        # TODO: Just have a IS_TRUE function instead to avoid scalar creation?
        return bif.Op("=" if is_sql_op else "==", colname, bif.Scalar(new_var))

    def _get_call_filter(
        self,
        call_def: ir.Expr,
        func_ir: ir.FunctionIR,
        df_var,
        df_col_names,
        df_col_types,
        is_sql_op: bool,
        read_node,
        new_ir_assigns,
    ) -> bif.Filter:
        """
        Function used by _get_partition_filters to extract filters
        related to series or array method calls.

        Currently this supports null related filters and isin.
        """
        require(is_expr(call_def, "call"))
        call_list = find_callname(func_ir, call_def, self.typemap)
        # checking call_list[1] == "pandas" to handle pd.isna/pd.notna cases generated
        # by BodoSQL
        require(
            len(call_list) == 2
            and (
                isinstance(call_list[1], ir.Var)
                or call_list[1] in ("pandas", "bodo.pandas")
                or call_list[1] == "bodosql.kernels"
            )
        )
        if call_list[0] in ("notna", "isna", "notnull", "isnull"):
            return self._get_null_filter(
                call_def,
                call_list,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                read_node,
            )
        elif call_list[0] == "isin":
            return self._get_isin_filter(
                call_list,
                call_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                read_node,
            )
        elif call_list == (
            "is_in",
            "bodosql.kernels",
        ):  # pragma: no cover
            return self._get_bodosql_array_kernel_is_in_filter(
                call_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                read_node,
            )
        elif call_list[0] in ("startswith", "endswith"):  # pragma: no cover
            return self._get_starts_ends_with_filter(
                call_list,
                call_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                read_node,
            )
        elif call_list == (
            "like_kernel",
            "bodosql.kernels",
        ):  # pragma: no cover
            return self._get_like_filter(
                call_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                is_sql_op,
                read_node,
                new_ir_assigns,
            )
        elif call_list == (
            "regexp_like",
            "bodosql.kernels",
        ):  # pragma: no cover
            return self._get_regexp_like_filter(
                call_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                read_node,
            )

        else:
            # Trigger a GuardException because we have hit an unknown function.
            # This should be caught by a surrounding function.
            raise GuardException

    def _get_null_filter(
        self,
        call_def,
        call_list,
        func_ir: ir.FunctionIR,
        df_var,
        df_col_names,
        df_col_types,
        read_node: "Connector",
    ):
        """
        Function used by _get_partition_filters to extract null related
        filters from series method calls.
        """
        # support both Series.isna() and pd.isna() forms
        arr_var = call_list[1] if isinstance(call_list[1], ir.Var) else call_def.args[0]
        col_name, _ = self._get_col_name(
            arr_var, df_var, df_col_names, df_col_types, func_ir, read_node
        )
        if call_list[0] in ("notna", "notnull"):
            op = "IS_NOT_NULL"
        else:
            op = "IS_NULL"
        return bif.Op(op, col_name)

    def _get_isin_filter(
        self,
        call_list,
        call_def,
        func_ir: ir.FunctionIR,
        df_var,
        df_col_names,
        df_col_types,
        read_node: "Connector",
    ) -> bif.Filter:
        """
        Function used by _get_partition_filters to extract isin related
        filters from series method calls.
        """
        # We must check for a list/set because this isn't previously checked
        # if we have binops (i.e. filter = (df.A < 3) & df.B.isin([1, 2]))
        list_set_arg = call_def.args[0]
        list_set_typ = self.typemap.get(list_set_arg.name, None)
        require(
            isinstance(list_set_typ, (types.List, types.Set))
            and list_set_typ.dtype != bodo.types.datetime64ns
            and not isinstance(
                list_set_typ.dtype, bodo.hiframes.pd_timestamp_ext.PandasTimestampType
            )
        )
        colname, _ = self._get_col_name(
            call_list[1], df_var, df_col_names, df_col_types, func_ir, read_node
        )
        # TODO: Add do _check_comparison_typing check. Requires list / set handling
        return bif.Op("IN", colname, bif.Scalar(list_set_arg))

    def _get_bodosql_array_kernel_is_in_filter(
        self,
        call_def,
        func_ir: ir.FunctionIR,
        df_var,
        df_col_names,
        df_col_types,
        read_node: "Connector",
    ):  # pragma: no cover
        """
        Function used by _get_partition_filters to extract isin related
        filters from the bodosql is_in kernel.
        """

        # This is normally already be checked in _is_isin_filter_pushdown_func,
        # However, we need to have this check here in case this expression is
        # the a sub expression in a binop (i.e. filter = (df.A < 3) & is_in(df.B, pd.array([1, 2])))
        arg1_arr_type = self.typemap.get(call_def.args[1].name, None)

        # We require that arg1 is a replicated array to perform filter pushdown.
        # In the bodoSQL codegen, this value should be lowered
        # as a global, and all globals are required to be replicated.
        is_arg1_global = isinstance(
            guard(get_definition, self.func_ir, call_def.args[1].name),
            numba.core.ir.Global,
        )

        # TODO: check if this requirement needs to be enforced
        require(
            is_arg1_global
            and arg1_arr_type.dtype != bodo.types.datetime64ns
            and not isinstance(
                arg1_arr_type.dtype, bodo.hiframes.pd_timestamp_ext.PandasTimestampType
            )
        )

        colname, _ = self._get_col_name(
            call_def.args[0], df_var, df_col_names, df_col_types, func_ir, read_node
        )
        # TODO: Add do _check_comparison_typing check. Requires list / set handling
        return bif.Op("IN", colname, bif.Scalar(call_def.args[1]))

    def _get_starts_ends_with_filter(
        self,
        call_list,
        call_def,
        func_ir: ir.FunctionIR,
        df_var,
        df_col_names,
        df_col_types,
        read_node: "Connector",
    ):  # pragma: no cover
        colname, coltype = self._get_col_name(
            call_list[1], df_var, df_col_names, df_col_types, func_ir, read_node
        )
        self._check_comparison_typing(coltype, call_def.args[0], read_node)
        return bif.Op(call_list[0], colname, bif.Scalar(call_def.args[0]))

    def _get_regexp_like_filter(
        self,
        call_def: ir.Expr,
        func_ir: ir.FunctionIR,
        df_var: ir.Var,
        df_col_names,
        df_col_types,
        read_node: "Connector",
    ) -> bif.Filter:  # pragma: no cover
        args = call_def.args
        # Get the column names
        col_name, col_type = self._get_col_name(
            args[0], df_var, df_col_names, df_col_types, func_ir, read_node
        )

        # Get the other args. We can only do filter pushdown if all of them are literals
        pattern_arg = args[1]
        regex_param_arg = args[2]

        pattern_type = self.typemap.get(pattern_arg.name, None)
        regex_param_type = self.typemap.get(regex_param_arg.name, None)

        invalid_types = (None, types.undefined, types.unknown)
        if (pattern_type in invalid_types) or (regex_param_type in invalid_types):
            self.needs_transform = True
            raise GuardException

        require(
            is_overload_constant_str(pattern_type)
            and is_overload_constant_str(regex_param_type)
        )

        pattern_type = types.unliteral(pattern_type)
        regex_param_type = types.unliteral(regex_param_type)

        require(
            pattern_type in (types.unicode_type, types.none)
            and regex_param_type in (types.unicode_type, types.none)
        )

        self._check_comparison_typing(col_type, pattern_arg, read_node)
        return bif.Op("REGEXP_LIKE", col_name, *(bif.Scalar(x) for x in args[1:]))

    def _get_like_filter(
        self,
        call_def: ir.Expr,
        func_ir: ir.FunctionIR,
        df_var: ir.Var,
        df_col_names,
        df_col_types,
        is_sql_op: bool,
        read_node,
        new_ir_assigns: list[ir.Var],
    ) -> bif.Filter:  # pragma: no cover
        """Generate a filter for like. If the values in like are proper constants
        then this generates the correct operations

        Args:
            call_def (ir.Expr): An IR expression representing the call in the IR. This contains
                access to the call details, such as the arguments.
            func_ir (ir.FunctionIR): The current IR for the function. This is needed to update
                definitions and extract information like column names.
            df_var (ir.Var): The IR variable for the DataFrame being accessed by the kernel. This is
                used to determine the original column name for filter pushdown.
            df_col_names (Tuple[str, ...]): n-tuple of DataFrame column names
            is_sql_op (bool): Is the operation targeting sql or parquet/iceberg. This
                influences the generated op string.
            new_ir_assigns (List[ir.Stmt]): A list of ir.Stmt values that are created when generating
                the filters. If this filter succeeds we will append to this list.

        Returns:
            Tuple[str, str, ir.Var]: A tuple for this particular filter. It has the form
                (column_name, op (e.g. <), Variable). Since like/ilike require a transformation
                to get the pattern into this standard form we generate a new Variable add append
                it to the IR.

        Raises GuardException: If the inputs cannot be converted to a valid filter.
        """
        args = call_def.args
        # Get the column names
        colname, _ = self._get_col_name(
            args[0], df_var, df_col_names, df_col_types, func_ir, read_node
        )
        # Get the other args. We can only do filter pushdown if all of them are literals
        pattern_arg = args[1]
        pattern_type = self.typemap.get(pattern_arg.name, None)
        if pattern_type in (None, types.undefined, types.unknown):
            self.needs_transform = True
            raise GuardException
        escape_arg = args[2]
        escape_type = self.typemap.get(escape_arg.name, None)
        if escape_type in (None, types.undefined, types.unknown):
            self.needs_transform = True
            raise GuardException
        case_insensitive_arg = args[3]
        case_insensitive_type = self.typemap.get(case_insensitive_arg.name, None)
        if case_insensitive_type in (None, types.undefined, types.unknown):
            self.needs_transform = True
            raise GuardException
        require(is_overload_constant_bool(case_insensitive_type))
        # Fetch case sensitive information. If is_case_insensitive == True
        # then we will create new operations that can be replaced by
        # arrow and snowflake.
        is_case_insensitive = get_overload_const_bool(case_insensitive_type)
        # If either pattern or escape is not literal it must be the regex case
        # or always False.
        match_nothing = False
        requires_regex = False
        if not (
            is_overload_constant_str(pattern_type)
            and is_overload_constant_str(escape_type)
        ):
            # We don't support filter pushdown with optional types
            # or array types.
            pattern_type = types.unliteral(pattern_type)
            escape_type = types.unliteral(escape_type)
            require(
                pattern_type in (types.unicode_type, types.none)
                and escape_type in (types.unicode_type, types.none)
            )
            if pattern_type == types.none or escape_type == types.none:
                match_nothing = True
                # Generate a dummy pattern to get an ir.Var.
                final_pattern = ""
            else:
                requires_regex = True
        else:
            pattern_const = get_overload_const_str(pattern_type)
            escape_const = get_overload_const_str(escape_type)
            # Convert the pattern from SQL to Python for pushdown.
            (
                final_pattern,
                requires_regex,
                must_match_start,
                must_match_end,
                match_anything,
            ) = bodo.ir.filter.convert_sql_pattern_to_python_compile_time(
                pattern_const, escape_const, is_case_insensitive
            )

        filter_args = []

        # We cannot do filter pushdown if the expression requires us to keep like/use a regex.
        if requires_regex:
            # Regex can only be handled with a SQL interface
            require(is_sql_op)
            if is_case_insensitive:
                op = "ilike"
            else:
                op = "like"

            filter_args = [pattern_arg, escape_arg]
        else:
            if match_nothing:
                return bif.Op("ALWAYS_NULL")
            elif match_anything:
                return bif.Op("ALWAYS_TRUE")

            elif must_match_start and must_match_end:
                # This is equality
                if is_case_insensitive:
                    op = "case_insensitive_equality"
                else:
                    op = "=" if is_sql_op else "=="
            elif must_match_start:
                if is_case_insensitive:
                    op = "case_insensitive_startswith"
                else:
                    op = "startswith"
            elif must_match_end:
                if is_case_insensitive:
                    op = "case_insensitive_endswith"
                else:
                    op = "endswith"
            else:
                if is_case_insensitive:
                    op = "case_insensitive_contains"
                else:
                    op = "contains"

            # Create a new IR Expr for the constant pattern.
            expr_value = ir.Const(final_pattern, call_def.loc)

            # Generate a variable from the Expr
            new_name = mk_unique_var("like_python_var")
            new_var = ir.Var(self.scope, new_name, call_def.loc)
            new_assign = ir.Assign(target=new_var, value=expr_value, loc=call_def.loc)
            # Append the assign so we update the IR.
            new_ir_assigns.append(new_assign)
            # Update the definitions. This is safe since the name is unique.
            func_ir._definitions[new_name] = [expr_value]

            # Add new argument to filter args
            filter_args = [new_var]

        return bif.Op(op.upper(), colname, *(bif.Scalar(arg) for arg in filter_args))

    def _get_partition_filters(
        self,
        index_def,
        df_var,
        lhs_def,
        rhs_def,
        func_ir: ir.FunctionIR,
        read_node: "Connector",
        new_ir_assigns,
    ) -> bif.Filter:
        """
        Get filters for predicate pushdown if possible.
        Returns filters as Bodo IR Filter objects:
        Throws GuardException if not possible.
        """
        is_sql = isinstance(read_node, bodo.ir.sql_ext.SqlReader)
        # NOTE: I/O nodes update out_table_col_names in DCE but typing pass is before
        # optimizations
        # Only ParquetReader has original_df_colnames so needs special casing
        df_col_names = (
            read_node.original_df_colnames
            if isinstance(read_node, bodo.ir.parquet_ext.ParquetReader)
            else read_node.out_table_col_names
        )

        df_col_types = (
            read_node.original_table_col_types
            if isinstance(read_node, bodo.ir.parquet_ext.ParquetReader)
            else read_node.out_table_col_types
        )

        # Call recursively on arguments and concatenate
        def get_child_filter(child_def) -> bif.Filter:
            """
            Function that abstracts away the recursive steps of getting the filters
            from the child exprs of index_def.
            """
            if self._is_logical_not_filter_pushdown(child_def, func_ir):
                arg_var = get_unary_arg(child_def)
                arg_def = get_definition(func_ir, arg_var)
                child_or = self._negate_column_filter(get_child_filter(arg_def))

            elif self._is_logical_op_filter_pushdown(child_def, func_ir):
                l_def = get_definition(func_ir, get_binop_arg(child_def, 0))
                l_def = self._remove_series_wrappers_from_def(l_def)
                r_def = get_definition(func_ir, get_binop_arg(child_def, 1))
                r_def = self._remove_series_wrappers_from_def(r_def)
                child_or = self._get_partition_filters(
                    child_def, df_var, l_def, r_def, func_ir, read_node, new_ir_assigns
                )
            elif self._is_call_op_filter_pushdown(child_def, func_ir):
                child_or = self._get_call_filter(
                    child_def,
                    func_ir,
                    df_var,
                    df_col_names,
                    df_col_types,
                    is_sql,
                    read_node,
                    new_ir_assigns,
                )
            else:
                child_or = self._get_column_filter(
                    child_def,
                    func_ir,
                    df_var,
                    df_col_names,
                    df_col_types,
                    is_sql,
                    read_node,
                    new_ir_assigns,
                )
            return child_or

        if self._is_logical_not_filter_pushdown(index_def, func_ir):
            arg_var = get_unary_arg(index_def)
            arg_def = get_definition(func_ir, arg_var)
            filters = get_child_filter(arg_def)
            return self._negate_column_filter(filters)

        if self._is_logical_op_filter_pushdown(index_def, func_ir):
            if self._is_or_filter_pushdown(index_def, func_ir):
                left_or = get_child_filter(lhs_def)
                right_or = get_child_filter(rhs_def)
                return bif.Op("OR", left_or, right_or)

            if self._is_and_filter_pushdown(index_def, func_ir):
                # both lhs and rhs are And/literal expressions.
                left_and = get_child_filter(lhs_def)
                right_and = get_child_filter(rhs_def)
                return bif.Op("AND", left_and, right_and)

        if self._is_cmp_op_filter_pushdown(index_def, func_ir):
            lhs = get_binop_arg(index_def, 0)
            rhs = get_binop_arg(index_def, 1)
            left_colname = guard(
                self._get_col_name,
                lhs,
                df_var,
                df_col_names,
                df_col_types,
                func_ir,
                read_node,
            )
            right_colname = guard(
                self._get_col_name,
                rhs,
                df_var,
                df_col_names,
                df_col_types,
                func_ir,
                read_node,
            )

            require(
                (left_colname and not right_colname)
                or (right_colname and not left_colname)
            )
            col: tuple[bif.Filter, types.ArrayCompatible] = (
                left_colname if left_colname else right_colname
            )  # type: ignore
            colname, coltype = col
            scalar = rhs if left_colname else lhs
            # This is a defensive check, and isn't expected to be hit
            # so we need the pragma to let coverage pass
            if scalar.name not in self.typemap:  # pragma: no cover
                self.needs_transform = True
                raise GuardException
            op = get_cmp_operator(
                index_def,
                self.typemap[scalar.name],
                is_sql,
                right_colname is not None,
                self.func_ir,
            )
            self._check_comparison_typing(coltype, scalar, read_node)
            return bif.Op(op.upper(), colname, bif.Scalar(scalar))

        elif self._is_call_op_filter_pushdown(index_def, func_ir):
            cond = self._get_call_filter(
                index_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                is_sql,
                read_node,
                new_ir_assigns,
            )
        else:
            # Filter pushdown is just on a boolean column.
            cond = self._get_column_filter(
                index_def,
                func_ir,
                df_var,
                df_col_names,
                df_col_types,
                is_sql,
                read_node,
                new_ir_assigns,
            )

        return cond

    def _get_col_name(
        self,
        var,
        df_var,
        df_col_names,
        df_col_types,
        func_ir: ir.FunctionIR,
        read_node: "Connector",
    ) -> tuple[bif.Filter, types.ArrayCompatible]:
        """
        Get column name for dataframe column access like df["A"] if possible.
        Throws GuardException if not possible.
        """

        def are_supported_kws(kws: list[tuple[str, ir.Var]]) -> bool:
            """Verify that keyword args passed into a function are supported,
            which are only the special arguments for filter pushdown.

            Args:
                kws (List[Tuple[str, ir.Var]]): The kws passed into the IR.Call

            Returns:
                bool: Are the kws either empty or just dict_encoding_state and func_id.
            """
            if len(kws) == 0:
                return True
            else:
                kws_dict = dict(kws)
                return (
                    len(kws_dict) == 2
                    and "dict_encoding_state" in kws_dict
                    and "func_id" in kws_dict
                )

        def get_col_type(col_val: str) -> types.ArrayCompatible:
            """Return column type for column representation in filter predicate

            Args:
                col_val (str): column name

            Returns:
                types.Type: array type of predicate column
            """
            return df_col_types[df_col_names.index(col_val)]

        var_def = get_definition(func_ir, var)
        var_def = self._remove_series_wrappers_from_def(var_def)

        # In form of df.A
        if is_expr(var_def, "getattr") and var_def.value.name == df_var.name:
            return bif.Ref(var_def.attr), get_col_type(var_def.attr)

        # In form of df["A"]
        if is_expr(var_def, "static_getitem") and var_def.value.name == df_var.name:
            return bif.Ref(var_def.index), get_col_type(var_def.index)

        if (
            is_expr(var_def, "call")
            and find_callname(self.func_ir, var_def)
            == ("get_table_data", "bodo.hiframes.table")
            and var_def.args[0].name == df_var.name
        ):
            col_val = df_col_names[find_const(func_ir, var_def.args[1])]
            return bif.Ref(col_val), get_col_type(col_val)

        # handle case with calls like df["A"].astype(int) > 2
        if is_call(var_def):
            fdef = find_callname(func_ir, var_def)
            require(fdef is not None)
            # calling pd.to_datetime() on a string column is possible since pyarrow
            # matches the data types before filter comparison (in this case, calls
            # pd.Timestamp on partition's string value)
            # BodoSQL generates pd.Series(arr) calls for expressions
            if fdef in (
                ("to_datetime", "pandas"),
                ("Series", "pandas"),
                ("to_datetime", "bodo.pandas"),
                ("Series", "bodo.pandas"),
            ):
                # We don't want to perform filter pushdown if there is a format argument
                # i.e. pd.to_datetime(col, format="%Y-%d-%m")
                # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
                require((len(var_def.args) == 1) and not var_def.kws)
                return self._get_col_name(
                    var_def.args[0],
                    df_var,
                    df_col_names,
                    df_col_types,
                    func_ir,
                    read_node,
                )
            # BodoSQL generates get_dataframe_data() calls for projections
            if (
                fdef == ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext")
                and var_def.args[0].name == df_var.name
            ):
                col_ind = get_const_value_inner(
                    func_ir, var_def.args[1], arg_types=self.arg_types
                )
                col_val = df_col_names[col_ind]
                return bif.Ref(col_val), get_col_type(col_val)

            is_bodosql_array_kernel = fdef[1] == "bodosql.kernels"

            # coalesce can be called on the filter column, which will be pushed down
            # e.g. where coalesce(L_COMMITDATE, current_date()) >= '1998-10-30'
            if is_bodosql_array_kernel and fdef[0] in (
                "coalesce",
                "concat_ws",
                "least",
                "greatest",
            ):  # pragma: no cover
                # coalesce takes a tuple input
                # We only push down 2-arg scalar case, i.e. coalesce((column, scalar))
                if fdef[0] == "concat_ws":  # pragma: no cover
                    require(
                        (
                            isinstance(read_node, bodo.ir.sql_ext.SqlReader)
                            and read_node.db_type == "snowflake"
                        )
                        or isinstance(read_node, bodo.ir.iceberg_ext.IcebergReader)
                    )
                else:
                    require((len(var_def.args) == 1) and are_supported_kws(var_def.kws))

                args = find_build_tuple(self.func_ir, var_def.args[0])
                require(len(args) == 2)

                # make sure arg[1] is scalar
                arg_type = self.typemap.get(args[1].name, None)
                if arg_type in (None, types.undefined, types.unknown):
                    self.needs_transform = True
                    raise GuardException

                require(
                    is_scalar_type(arg_type)
                    and arg_type != types.none
                    and not isinstance(arg_type, types.Optional)
                )

                col_name, col_type = self._get_col_name(
                    args[0], df_var, df_col_names, df_col_types, func_ir, read_node
                )
                return bif.Op(fdef[0].upper(), col_name, bif.Scalar(args[1])), col_type

            if is_bodosql_array_kernel and fdef[0] in (
                "dayofweek",
                "weekofyear",
                "week",
                "yearofweek",
            ):  # pragma: no cover
                bodosql_kernel_name = fdef[0]
                require(bodosql_kernel_name in supported_funcs_map)

                if bodosql_kernel_name not in supported_arrow_funcs_map:
                    require(
                        (
                            isinstance(read_node, bodo.ir.sql_ext.SqlReader)
                            and read_node.db_type == "snowflake"
                        )
                        or isinstance(read_node, bodo.ir.iceberg_ext.IcebergReader)
                    )

                args = var_def.args

                col_name, col_type = self._get_col_name(
                    args[0], df_var, df_col_names, df_col_types, func_ir, read_node
                )

                return bif.Op(bodosql_kernel_name.upper(), col_name), col_type

            # All other BodoSQL functions
            if is_bodosql_array_kernel and are_supported_kws(
                var_def.kws
            ):  # pragma: no cover
                bodosql_kernel_name = fdef[0]
                require(bodosql_kernel_name in supported_funcs_map)

                if bodosql_kernel_name not in supported_arrow_funcs_map:
                    require(
                        (
                            isinstance(read_node, bodo.ir.sql_ext.SqlReader)
                            and read_node.db_type == "snowflake"
                        )
                        or isinstance(read_node, bodo.ir.iceberg_ext.IcebergReader)
                    )

                args = var_def.args

                arg_type = self.typemap.get(args[0].name, None)
                require(is_array_typ(arg_type))

                for i, arg in enumerate(args):
                    arg_type = self.typemap.get(arg.name, None)

                    if arg_type in (None, types.undefined, types.unknown):
                        self.needs_transform = True
                        raise GuardException

                    if i == 0:
                        # We make the assumption that for every array kernel we
                        # have the array as the 0th argument.
                        require(is_array_typ(arg_type))
                    else:
                        require(is_scalar_type(arg_type))

                col_name, col_type = self._get_col_name(
                    args[0], df_var, df_col_names, df_col_types, func_ir, read_node
                )
                return (
                    bif.Op(
                        fdef[0].upper(), col_name, *(bif.Scalar(x) for x in args[1:])
                    ),
                    col_type,
                )

            if fdef[0] in ("str.lower", "str.upper"):
                # make sure fdef[1] is a Series
                arg_type = self.typemap.get(fdef[1].name, None)
                if arg_type in (None, types.undefined, types.unknown):
                    self.needs_transform = True
                    raise GuardException
                require(isinstance(arg_type, SeriesType))
                col_name, col_type = self._get_col_name(
                    fdef[1], df_var, df_col_names, df_col_types, func_ir, read_node
                )

                return bif.Op(fdef[0].split(".")[1].upper(), col_name), col_type

            # We only support astype at this point
            require(
                isinstance(fdef, tuple)
                and len(fdef) == 2
                and isinstance(fdef[1], ir.Var)
                and fdef[0] == "astype"
            )
            return self._get_col_name(
                fdef[1], df_var, df_col_names, df_col_types, func_ir, read_node
            )

        require(is_expr(var_def, "getitem"))
        require(var_def.value.name == df_var.name)
        col_val = get_const_value_inner(
            func_ir, var_def.index, arg_types=self.arg_types
        )
        return bif.Ref(col_val), get_col_type(col_val)

    def _remove_series_wrappers_from_def(self, var_def: ir.Expr) -> ir.Expr:
        """Returns the definition node of the Series variable in
        pd.Series()/Series.values/Series.str/bodo.pd_hiframes.series_ext.get_series_data nodes.
        This effectively removes the Series wrappers that BodoSQL currently generates to
        convert to/from arrays.

        Args:
            var_def (ir.Expr): expression node that may be a Series wrapper

        Returns:
            ir.Expr: expression node without Series wrapper
        """

        # Get Series value from Series.str/Series.values
        if is_expr(var_def, "getattr") and var_def.attr in ("str", "values"):
            var_def = guard(get_definition, self.func_ir, var_def.value)
            return self._remove_series_wrappers_from_def(var_def)

        # remove pd.Series() calls
        if (
            is_call(var_def)
            and guard(find_callname, self.func_ir, var_def)
            in (("Series", "pandas"), ("Series", "bodo.pandas"))
            and (len(var_def.args) == 1)
            and not var_def.kws
        ):
            var_def = guard(get_definition, self.func_ir, var_def.args[0])
            return self._remove_series_wrappers_from_def(var_def)

        # remove bodo.hiframes.pd_series_ext.get_series_data calls
        if (
            is_call(var_def)
            and guard(find_callname, self.func_ir, var_def)
            == ("get_series_data", "bodo.hiframes.pd_series_ext")
            and (len(var_def.args) == 1)
            and not var_def.kws
        ):
            var_def = guard(get_definition, self.func_ir, var_def.args[0])
            return self._remove_series_wrappers_from_def(var_def)

        return var_def

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

        col_inds, row_ind = self._get_loc_indices(idx_var, label, inst.loc, target_typ)

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
        col_inds, row_ind = self._get_loc_indices(idx_var, label, inst.loc, target_typ)

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

    def _get_loc_indices(self, idx_var, label, loc, target_typ):
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
            # if is slice, form list form column names / numbers as appropriate
            if isinstance(col_inds, slice):
                all_cols = list(target_typ.df_type.columns)
                # if iloc, use numbers
                if isinstance(target_typ, DataFrameILocType):
                    all_cols = list(range(len(all_cols)))
                col_inds = all_cols[col_inds]
            else:
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

    def _get_bodosql_ctx_name_df_typs(self, folded_args):  # pragma: no cover
        """
        Extracts the names/types of the dataframes used to intialize the bodosql context.
        This is converted into a tuple of the names/values in
        untyped pass (see _handle_bodosql_BodoSQLContext).
        This function extracts the dataframe types directly from the IR, to avoid any issues
        with incorrect type propogation (specifically, from df setitem).

        folded_args are the arguments passed to the BodoSQLContext constructor, placing
        any KWS into their standard location. This ensures the DataFrames are always at
        the same location if passed by keyword instead.
        """
        df_dict_var = folded_args[0]
        df_dict_def = guard(get_definition, self.func_ir, df_dict_var)
        df_dict_def_items = df_dict_def.items
        # floor divide
        split_idx = (len(df_dict_def_items) // 2) + 1
        # ommit first value, as it is a dummy
        df_name_vars, df_vars = (
            df_dict_def_items[1:split_idx],
            df_dict_def_items[split_idx:],
        )
        df_name_typs = []
        for df_name_var in df_name_vars:
            name_val = self.typemap.get(df_name_var.name, None)
            if name_val is not None:
                name_val = name_val.literal_value
            df_name_typs.append(name_val)
        df_name_typs = tuple(df_name_typs)
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
                    "lambda _b_df: bodo.hiframes.pd_dataframe_ext.get_dataframe_data(_b_df, _i)"
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
        if isinstance(target_typ, DataFrameType) and not target_typ.has_runtime_cols:
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

            # df.B = A transform
            # Pandas only allows setting existing columns using setattr
            if inst.attr in target_typ.columns:
                return self._run_df_set_column(inst, inst.attr, label)

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

        if fdef in (("read_sql_table", "pandas"), ("read_sql_table", "bodo.pandas")):
            return self._run_call_read_sql_table(assign, rhs, func_name, label)

        if func_mod in ("pandas", "bodo.pandas"):
            return self._run_call_pd_top_level(assign, rhs, func_name, label)

        # handle pd.Timestamp.method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), PandasTimestampType
        ):
            return self._run_call_pd_timestamp(assign, rhs, func_mod, func_name, label)

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

        # handle Series.str.method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), SeriesStrMethodType
        ):
            return self._run_call_str_method(assign, rhs, func_mod, func_name, label)

        # handle BodoSQLContextType.sql() calls here since the generated code cannot
        # be handled in regular overloads (requires Bodo's untyped pass, typing pass)
        #
        # Note we delay checking BodoSQLContextType until we find a possible match
        # to avoid paying the import overhead for Bodo calls with no BodoSQL.
        if isinstance(func_mod, ir.Var) and func_name in (
            "sql",
            "convert_to_pandas",
        ):  # pragma: no cover
            # Try import BodoSQL and check the type

            if is_bodosql_context_type(self._get_method_obj_type(func_mod, rhs.func)):
                return self._run_call_bodosql_sql(
                    assign, rhs, func_mod, func_name, label
                )

        # handle BodoSQLTablePathType
        if fdef == ("TablePath", "bodosql"):
            # Force table path arguments to be literals if passed to the function.
            return self._run_call_bodosql_table_path(assign, rhs, label)

        # handle PandasDatetimeArray
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DatetimeArrayType
        ):
            return self._run_call_pd_datetime_array(assign, rhs, func_name, label)

        # handle SeriesDatetimePropertiesType
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), SeriesDatetimePropertiesType
        ):
            return self._run_call_pd_datetime_properties(assign, rhs, func_name, label)

        # handle DatetimeIndex
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DatetimeIndexType
        ):
            return self._run_call_pd_datetime_index(assign, rhs, func_name, label)

        # throw proper error when calling a non-JIT function
        if isinstance(
            self.typemap.get(rhs.func.name, None), bodo.utils.typing.FunctionLiteral
        ):
            func_name = "unknown"
            try:
                func_name = self.typemap[rhs.func.name].literal_value.__name__
            except Exception:  # pragma: no cover
                pass
            raise BodoError(
                f"Cannot call non-JIT function '{func_name}' from JIT function (convert to JIT or use objmode).",
                rhs.loc,
            )

        if fdef == ("table_filter", "bodo.hiframes.table"):
            self._try_apply_filter_pushdown(assign, label)

        # Infer CASE output type if not provided by BodoSQL
        if (
            fdef == ("bodosql_case_placeholder", "bodo.utils.typing")
            and unwrap_typeref(self.typemap[rhs.args[-1].name]) == types.unknown
        ):
            return self._run_call_bodosql_case_placeholder(assign, rhs)

        return [assign]

    def _run_call_bodosql_case_placeholder(self, assign, rhs):
        """Infers output type of CASE kernel when not provided by BodoSQL and updates
        the IR.
        """
        import bodosql  # noqa: F401

        if any(self.typemap[v.name] in unresolved_types for v in rhs.args[:-1]):
            self.needs_transform = True
            return [assign]

        init_code = self.typemap[rhs.args[2].name].instance_type.meta
        body_code = self.typemap[rhs.args[3].name].instance_type.meta
        arr_variable_name = get_overload_const_str(self.typemap[rhs.args[4].name])
        indexing_variable_name = get_overload_const_str(self.typemap[rhs.args[5].name])

        # Replace output array setitem with scalar to infer scalar type from output
        out_setitem = f"{arr_variable_name}[{indexing_variable_name}]"
        body_code = body_code.replace(out_setitem, arr_variable_name)

        named_params = dict(rhs.kws)
        named_param_args = ", ".join(named_params.keys())

        var_names, _ = handle_bodosql_case_init_code(init_code)

        # skip_allocation=True since we want output to be scalar to infer its type
        f, _ = gen_bodosql_case_func(
            init_code,
            body_code,
            named_param_args,
            var_names,
            arr_variable_name,
            indexing_variable_name,
            None,
            self.func_ir.func_id.func.__globals__,
            skip_allocation=True,
        )

        in_arg_types = [self.typemap[v.name] for v in rhs.args[:2]]
        out_dtype = get_const_func_output_type(
            bodo.jit(distributed=False)(f),
            in_arg_types,
            {k: self.typemap[v.name] for k, v in named_params.items()},
            self.typingctx,
            numba.core.registry.cpu_target.target_context,
        )
        output_array_type = dtype_to_array_type(out_dtype, True)

        # Replace last argument with inferred array type
        new_type_var = ir.Var(
            assign.target.scope, mk_unique_var("output_array_type"), rhs.loc
        )
        new_type_var_assign = ir.Assign(
            ir.Global("output_array_type", output_array_type, rhs.loc),
            new_type_var,
            rhs.loc,
        )
        rhs.args[-1] = new_type_var
        self.typemap.pop(assign.target.name)
        self.typemap[assign.target.name] = output_array_type
        self.changed = True
        return [new_type_var_assign, assign]

    def _run_call_pd_datetime_array(self, assign, rhs, func_name, label):
        """Handle calls to pandas.DatetimeArray methods that need transformation"""
        nodes = []
        if func_name == "tz_convert":
            func_args = [(0, "tz")]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)
        return nodes + [assign]

    def _run_call_pd_datetime_index(self, assign, rhs, func_name, label):
        nodes = []
        if func_name == "tz_convert":
            func_args = [(0, "tz")]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)
        return nodes + [assign]

    def _run_call_pd_datetime_properties(self, assign, rhs, func_name, label):
        nodes = []
        if func_name == "tz_convert":
            func_args = [(0, "tz")]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)
        return nodes + [assign]

    def _run_call_bodosql_table_path(self, assign, rhs, label):
        nodes = []
        func_args = [
            (0, "file_path"),
            (1, "file_type"),
            (2, "conn_str"),
            (3, "reorder_io"),
            (4, "db_schema"),
            (5, "bodo_read_as_dict"),
        ]
        nodes += self._replace_arg_with_literal(
            "bodosql.TablePath", rhs, func_args, label
        )
        return nodes + [assign]

    def _run_binop(self, assign, rhs):
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
            "pivot": [(0, "index"), (1, "columns"), (2, "values")],
            "pivot_table": [
                (0, "values"),
                (1, "index"),
                (2, "columns"),
                (3, "aggfunc"),
            ],
            "explode": [
                (0, "column"),
            ],
            "melt": [
                (0, "id_vars"),
                (1, "value_vars"),
                (2, "var_name"),
                (3, "value_name"),
            ],
            "memory_usage": [(0, "index")],
        }
        if func_name in df_call_const_args:
            func_args = df_call_const_args[func_name]
            # function arguments are typed as pyobject initially, literalize if possible
            pyobject_to_literal = func_name == "apply"
            nodes += self._replace_arg_with_literal(
                func_name, rhs, func_args, label, pyobject_to_literal
            )
        if func_name == "astype":
            return nodes + self._handle_df_astype(assign.target, rhs, df_var, assign)

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
                data_col_total.append(f"df['{c}'].values")
            elif c in additional_columns:
                name_col_total.append(c)
                e_col = f"bodo.utils.conversion.coerce_to_array(new_arg{kws_key_list.index(c)}, scalar_to_arr_len=len(df))"
                data_col_total.append(e_col)

        # The new columns should be added in the order that they apear in kws_key_val_list
        for i, c in enumerate(kws_key_list):
            if c not in df_type.columns:
                name_col_total.append(c)
                e_col = f"bodo.utils.conversion.coerce_to_array(new_arg{i}, scalar_to_arr_len=len(df))"
                data_col_total.append(e_col)

        data_args = ", ".join(data_col_total)
        header = "def impl(df, {}):\n".format(
            ", ".join(f"new_arg{i}" for i in range(len(kws_key_val_list)))
        )
        impl = bodo.hiframes.dataframe_impl._gen_init_df(
            header,
            tuple(name_col_total),
            data_args,
        )

        self.changed = True
        return compile_func_single_block(impl, [df_var] + kws_val_list, lhs)

    def _handle_df_astype(self, lhs, rhs, df_var, assign):
        """Handle special optimizations in Bodo to enable converting types that
        are not possible in generic Pandas because they map to "object" type.

        This function checks if the rhs is generating the dtype by looking at the
        Bodo type of an object and if so appends/updates an additional argument
        that is a typeref to that object's type.
        """
        nodes = []
        kws_dict = dict(rhs.kws)
        dtypes_arg = get_call_expr_arg(
            "DataFrame.astype", rhs.args, kws_dict, 0, "dtype"
        )
        dtypes_source = guard(get_definition, self.func_ir, dtypes_arg)
        # Currently this is only implemented for DataFrame.dtypes.
        # TODO: Handle additional sources (i.e. S.dtype or arr.dtype).
        if is_expr(dtypes_source, "getattr") and dtypes_source.attr == "dtypes":
            source_obj = self.typemap.get(dtypes_source.value.name, None)
            if isinstance(source_obj, DataFrameType):
                # Check if there is already a schema. typeref_arg will be None if it doesn't exist.
                typeref_arg = get_call_expr_arg(
                    "DataFrame.astype",
                    rhs.args,
                    kws_dict,
                    5,
                    "_bodo_object_typeref",
                    use_default=True,
                )
                # Add a new argument if either the schema doesn't exist or has changed.
                if typeref_arg is None or self.typemap.get(
                    typeref_arg.name, None
                ) != types.TypeRef(source_obj):
                    # Create a new variable for the typeref
                    var_name = mk_unique_var("bodo_object_typeref")
                    typeref_var = ir.Var(assign.target.scope, var_name, rhs.loc)
                    new_var_assign = ir.Assign(
                        ir.Const(source_obj, assign.loc), typeref_var, assign.loc
                    )
                    nodes.append(new_var_assign)
                    self.typemap[var_name] = types.TypeRef(source_obj)
                    # Create a copy of the rhs.
                    new_rhs = ir.Expr.call(
                        rhs.func,
                        rhs.args,
                        # Note: We use a dict for set_call_expr_arg only then convert
                        # back to a list of tuples.
                        kws_dict,
                        rhs.loc,
                        vararg=rhs.vararg,
                        target=rhs.target,
                    )
                    # Update the kws
                    set_call_expr_arg(
                        typeref_var,
                        new_rhs.args,
                        new_rhs.kws,
                        5,
                        "_bodo_object_typeref",
                        add_if_missing=True,
                    )
                    # Convert kws back to list of tuples for consistency
                    new_rhs.kws = list(new_rhs.kws.items())
                    # Create a new assign to avoid mutating the IR
                    assign = ir.Assign(new_rhs, assign.target, assign.loc)
                    self.changed = True

        return nodes + [assign]

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
        for _, kw_val in rhs.kws:
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
        for kw_name, kw_val in kws_list[first_lambda_idx:]:
            if isinstance(self.typemap.get(kw_val.name), types.Dispatcher):
                # handles lambda fns, and passed JIT functions
                # put the setitem value is as the output of an apply on the current dataframe

                # assign the apply function to a variable
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
        exec(func_text, {}, loc_vars)
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

    def _run_call_str_method(self, assign, rhs, str_var, func_name, label):
        """Handle Series.str calls that need transformation to meet Bodo
        requirements
        """
        nodes = []

        # mapping of groupby functions to their arguments that require constant values
        str_call_const_args = {
            "extract": [(0, "pat"), (1, "flags")],
            "extractall": [(0, "pat"), (1, "flags")],
        }

        if func_name in str_call_const_args:
            func_args = str_call_const_args[func_name]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)

        return nodes + [assign]

    def _run_call_series(self, assign, rhs, series_var, func_name, label):
        """Handle Series calls that need transformation to meet Bodo requirements"""
        nodes = []

        # mapping of Series functions to their arguments that require constant values
        series_call_const_args = {
            "map": [(0, "arg"), (1, "na_action")],
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
            "rank": [(1, "method"), (3, "na_option"), (5, "pct")],
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

    def _run_call_pd_timestamp(self, assign, rhs, ts_var, func_name, label):
        """Handle pd.Timestamp calls that need transformation to meet Bodo requirements"""
        nodes = []

        # mapping of Series functions to their arguments that require constant values
        pd_timestamp_call_const_args = {
            "tz_convert": [(0, "tz")],
            "tz_localize": [(0, "tz")],
        }

        if func_name in pd_timestamp_call_const_args:
            func_args = pd_timestamp_call_const_args[func_name]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args, label)

        return nodes + [assign]

    def _run_call_read_sql_table(
        self, assign: ir.Assign, rhs: ir.Expr, func_name, label
    ):
        """transform pd.read_sql_table into a SQL node"""
        import bodo.io.iceberg
        import bodo.io.iceberg.read_compilation
        from bodo.ir.iceberg_ext import IcebergConnectionType

        func_str = "pandas.read_sql_table"
        lhs = assign.target
        kws = dict(rhs.kws)
        # Note schema here refers to a database schema, so we
        # will use the variable name database_schema.
        supported_args = ["table_name", "con", "schema"]
        arg_values = []

        def handle_con(con_arg):
            """
            Extracts the con_str from the con arg
            """
            err_msg = "pandas.read_sql_table(): 'con', if provided, must be a constant string or an IcebergConnectionType"
            con_type = self.typemap[con_arg.name]

            if isinstance(con_type, IcebergConnectionType):
                con_str = con_type.conn_str
            else:
                con_str = self._get_const_value(
                    con_arg, label, rhs.loc, err_msg=err_msg
                )

            # TODO: BSE-3331: This shouldn't have to change the con_arg, con_arg should be able to stay as an ir.Var
            if not isinstance(con_type, IcebergConnectionType):
                con_arg = ir.Const(con_str, con_arg.loc)

            if not isinstance(con_str, str):
                raise BodoError(err_msg)

            return con_arg, con_str

        for i, arg in enumerate(supported_args):
            err_msg = f"pandas.read_sql_table(): '{arg}', if provided, must be a constant string."
            temp = get_call_expr_arg(func_str, rhs.args, kws, i, arg)

            if arg == "con":
                temp, con_str = handle_con(temp)
            else:
                temp = self._get_const_value(temp, label, rhs.loc, err_msg=err_msg)
                if not isinstance(temp, str):
                    raise BodoError(err_msg)

            arg_values.append(temp)
        table_name, con, database_schema = arg_values

        # Generate Output DataFrame Type
        arg_defaults = {
            "index_col": None,
            "coerce_float": True,
            "parse_dates": None,
            "columns": None,
            "chunksize": None,
        }
        unsupported_args = {}
        for i, (arg, default) in enumerate(arg_defaults.items()):
            # we support args 0-2, so we start at len(supported_args)=3
            temp = get_call_expr_arg(
                func_str,
                rhs.args,
                kws,
                i + len(supported_args),
                arg,
                default=default,
                use_default=True,
            )
            if temp != default:
                temp = self._get_const_value(
                    temp,
                    label,
                    rhs.loc,
                    f"pandas.read_sql_table(): '{arg}', if provided must be a constant and the default value {default}.",
                )
            unsupported_args[arg] = temp
        check_unsupported_args(
            "read_sql_table", unsupported_args, arg_defaults, package_name="pandas"
        )
        err_msg = "pandas.read_sql_table(): '_bodo_merge_into', if provided, must be a constant boolean."
        # Support the bodo specific `_bodo_merge_into` argument. This argument should only be present if we are
        # performing merge into and should have the effect of only filtering on files during filter pushdown.
        _bodo_merge_into_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_merge_into",
            default=False,
            use_default=True,
        )
        _bodo_merge_into = (
            self._get_const_value(_bodo_merge_into_var, label, rhs.loc, err_msg=err_msg)
            if _bodo_merge_into_var
            else False
        )

        # _bodo_read_as_table allows specifying that the output should be a Table rather than
        # be wrapped in a DataFrame
        _bodo_read_as_table = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_read_as_table",
            default=None,
            use_default=True,
        )

        # _bodo_read_as_dict allows users to specify columns as dictionary-encoded
        # string arrays manually. This is in addition to whatever columns bodo
        # determines should be read in with dictionary encoding.
        _bodo_read_as_dict_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_read_as_dict",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_read_as_dict', if provided, must be a constant list of column names."
        _bodo_read_as_dict = (
            self._get_const_value(
                _bodo_read_as_dict_var, label, rhs.loc, err_msg=err_msg
            )
            if _bodo_read_as_dict_var
            else []
        )
        if _bodo_read_as_dict == None:
            _bodo_read_as_dict = []
        if not isinstance(_bodo_read_as_dict, list):
            raise BodoError(err_msg)

        # _bodo_detect_dict_cols allows users to disable dict-encoding detection
        # This is useful when getting a list of files from the table takes too long.
        # This is possible when:
        # - The table is too large and has a lot of files
        # - The table is located on a high-latency filesystem
        # - There are some other entities (firewalls) slowing filesystem ops
        # Note that columns passed in via _bodo_read_as_dict will still be dictionary-encoded
        _bodo_detect_dict_cols_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_detect_dict_cols",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_detect_dict_cols', if provided, must be a constant boolean."
        detect_dict_cols = (
            self._get_const_value(
                _bodo_detect_dict_cols_var, label, rhs.loc, err_msg=err_msg
            )
            if _bodo_detect_dict_cols_var
            else True
        )
        if not isinstance(detect_dict_cols, bool):
            raise BodoError(err_msg)

        # Whether Bodo should do the dictionary encoding (for the columns selected for dict-encoding)
        # after reading the columns as string from Arrow. This is useful for testing purposes. In
        # practice, we don't expect users to specify this.
        _bodo_dict_encode_in_bodo_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_dict_encode_in_bodo",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_dict_encode_in_bodo', if provided, must be a constant boolean."
        dict_encode_in_bodo = None
        if _bodo_dict_encode_in_bodo_var:
            dict_encode_in_bodo = self._get_const_value(
                _bodo_dict_encode_in_bodo_var, label, rhs.loc, err_msg=err_msg
            )
            if not isinstance(dict_encode_in_bodo, bool):
                raise BodoError(err_msg)
        else:
            # If the user did not specify where the dict-encoding should happen,
            # we will make the decision ourselves. At this point, we will only
            # do the dict-encoding ourselves if the table is a Snowflake managed
            # Iceberg table. If it isn't, we will let Arrow do it.
            dict_encode_in_bodo = (
                bodo.io.iceberg.read_compilation.is_snowflake_managed_iceberg_wh(
                    con_str
                )
            )

        # _bodo_chunksize enables streaming Iceberg reads with specified batch-size
        _bodo_chunksize_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_chunksize",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_chunksize', if provided, must be a constant integer."
        chunksize = (
            self._get_const_value(_bodo_chunksize_var, label, rhs.loc, err_msg=err_msg)
            if _bodo_chunksize_var
            else None
        )
        if chunksize and not isinstance(chunksize, int):
            raise BodoError(err_msg)

        _bodo_columns_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_columns",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_columns_var', if provided, must be a constant."
        columns_obj: list[str] | None = (
            self._get_const_value(_bodo_columns_var, label, rhs.loc, err_msg=err_msg)
            if _bodo_columns_var
            else None
        )
        if chunksize is None and columns_obj is not None:  # pragma: no cover
            raise BodoError(
                "pandas.read_sql_table(): '_bodo_columns' can only be used with '_bodo_chunksize'"
            )

        if _bodo_merge_into and chunksize is not None:
            raise BodoError(
                "pandas.read_sql_table(): Batched reads does not support MERGE INTO"
            )

        _bodo_filter_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_filter",
            default=None,
            use_default=True,
        )
        if _bodo_filter_var is None:
            filter_obj = None
        else:
            top_def = get_definition(self.func_ir, _bodo_filter_var)
            if isinstance(top_def, ir.Const) and is_overload_none(top_def):
                filter_obj = None
            else:
                filter_obj = build_filter_from_ir(
                    _bodo_filter_var, self.func_ir, self.typemap
                )

        _bodo_limit_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,  # Support this argument by keyword only
            "_bodo_limit",
            default=None,
            use_default=True,
        )
        limit_obj = (
            self._get_const_value(_bodo_limit_var, label, rhs.loc)
            if _bodo_limit_var
            else None
        )

        # Operator ID assigned by the planner for query profile purposes.
        # Only applicable in the streaming case.
        _bodo_sql_op_id_var = get_call_expr_arg(
            func_str,
            rhs.args,
            kws,
            -1,
            "_bodo_sql_op_id",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_bodo_sql_op_id', if provided, must be a constant integer."
        _bodo_sql_op_id_const = (
            self._get_const_value(_bodo_sql_op_id_var, label, rhs.loc, err_msg=err_msg)
            if _bodo_sql_op_id_var
            else -1
        )

        table_id: str = (
            f"{database_schema}.{table_name}" if database_schema else table_name
        )

        _snapshot_id_arg = get_call_expr_arg(
            "read_sql",
            rhs.args,
            kws,
            -1,
            "_snapshot_id",
            default=None,
            use_default=True,
        )
        _snapshot_timestamp_ms_arg = get_call_expr_arg(
            "read_sql",
            rhs.args,
            kws,
            -1,
            "_snapshot_timestamp_ms",
            default=None,
            use_default=True,
        )
        err_msg = "pandas.read_sql_table(): '_snapshot_id', if provided, must be a constant integer."
        snapshot_id = (
            self._get_const_value(_snapshot_id_arg, label, rhs.loc, err_msg=err_msg)
            if _snapshot_id_arg is not None
            else -1
        )
        err_msg = "pandas.read_sql_table(): '_snapshot_timestamp_ms', if provided, must be a constant integer."
        snapshot_timestamp_ms = (
            self._get_const_value(
                _snapshot_timestamp_ms_arg, label, rhs.loc, err_msg=err_msg
            )
            if _snapshot_timestamp_ms_arg is not None
            else -1
        )

        snapshot_id = bodo.io.iceberg.read_compilation.resolve_snapshot_id(
            con_str, table_id, snapshot_id, snapshot_timestamp_ms
        )

        (orig_col_names, orig_arr_types, pyarrow_table_schema, col_names, arr_types) = (
            bodo.io.iceberg.read_compilation.get_orig_and_runtime_schema(
                con_str,
                table_id,
                selected_cols=columns_obj,
                read_as_dict_cols=_bodo_read_as_dict,
                detect_dict_cols=detect_dict_cols,
                is_merge_into_cow=_bodo_merge_into,
                snapshot_id=snapshot_id,
            )
        )

        df_type = DataFrameType(
            tuple(arr_types),
            bodo.types.RangeIndexType(None),
            tuple(col_names),
            is_table_format=True,
        )

        # Merge INTO default output types
        file_list_type = types.pyobject_of_list_type
        snapshot_id_type = types.int64

        if chunksize is None:
            # Normal, non-streaming case
            data_arrs = [
                ir.Var(lhs.scope, mk_unique_var("sql_table"), lhs.loc),
                # Note index_col will always be dead since we don't support
                # index_col yet.
                ir.Var(lhs.scope, mk_unique_var("index_col"), lhs.loc),
                ir.Var(lhs.scope, mk_unique_var("file_list"), lhs.loc),
                ir.Var(lhs.scope, mk_unique_var("snapshot_id"), lhs.loc),
            ]

            self.typemap[data_arrs[0].name] = df_type.table_type  # type: ignore
            self.typemap[data_arrs[1].name] = types.none
            # If we have a merge into we will return a list of original iceberg files
            # + the snapshot id.
            if _bodo_merge_into:
                self.typemap[data_arrs[2].name] = file_list_type
                self.typemap[data_arrs[3].name] = snapshot_id_type
            else:
                self.typemap[data_arrs[2].name] = types.none
                self.typemap[data_arrs[3].name] = types.none
        else:
            # Streaming case
            data_arrs = [
                ir.Var(lhs.scope, mk_unique_var("arrow_iterator"), lhs.loc),
            ]
            self.typemap[data_arrs[0].name] = ArrowReaderType(col_names, df_type.data)
        # Retrieve the tuple of runtime join filters in the form
        # ((state_1, indices_1), (state_2, indices_2)...) where each
        # state is a join state object and each indices is a tuple of
        # column indices.
        _bodo_runtime_join_filters_arg = get_call_expr_arg(
            "read_sql",
            rhs.args,
            kws,
            -1,
            "_bodo_runtime_join_filters",
            default=None,
            use_default=True,
        )
        rtjf_terms = get_runtime_join_filter_terms(
            self.func_ir, _bodo_runtime_join_filters_arg
        )
        if rtjf_terms is not None and len(rtjf_terms):
            assert chunksize is not None, (
                "Cannot provide rtjf_terms in a non-streaming read"
            )

        nodes = [
            bodo.ir.iceberg_ext.IcebergReader(
                table_id,
                con,
                lhs.name,
                list(df_type.columns),
                list(df_type.data),
                data_arrs,
                lhs.loc,
                [],  # unsupported_columns
                [],  # unsupported_arrow_types
                None,  # index_column_name
                types.none,  # index_column_type
                pyarrow_table_schema,  # pyarrow_table_schema
                _bodo_merge_into,  # is_merge_into
                file_list_type,  # file_list_type
                snapshot_id_type,  # snapshot_id_type
                snapshot_id,  # snapshot_id
                chunksize=chunksize,
                used_cols=columns_obj,
                initial_filter=filter_obj,
                initial_limit=limit_obj,
                orig_col_names=orig_col_names,
                orig_col_types=orig_arr_types,
                sql_op_id=_bodo_sql_op_id_const,
                dict_encode_in_bodo=dict_encode_in_bodo,
                rtjf_terms=rtjf_terms,
            )
        ]

        if chunksize is not None:
            nodes += [ir.Assign(data_arrs[0], lhs, lhs.loc)]
        else:
            data_args = ["table_val", "idx_arr_val", "file_list_val", "snapshot_id_val"]
            # Create the index + dataframe
            index_arg = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({data_args[0]}), 1, None)"

            if _bodo_read_as_table:
                df_value = data_args[0]
            else:
                df_value = f"bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args[0]},), {index_arg}, __col_name_meta_value_read_sql_table)"

            func_text = f"def _init_df({data_args[0]}, {data_args[1]}, {data_args[2]}, {data_args[3]}):\n"
            if _bodo_merge_into:
                # If merge_into we return a tuple of values
                func_text += f"  return ({df_value}, {data_args[2]}, {data_args[3]})\n"
            else:
                # Otherwise just return the DataFrame
                func_text += f"  return {df_value}\n"
            loc_vars = {}
            exec(
                func_text,
                {
                    "__col_name_meta_value_read_sql_table": ColNamesMetaType(
                        df_type.columns
                    )
                },
                loc_vars,
            )
            _init_df = loc_vars["_init_df"]

            nodes += compile_func_single_block(
                _init_df,
                data_arrs,
                lhs,
                self,
                extra_globals={
                    "__col_name_meta_value_read_sql_table": ColNamesMetaType(
                        df_type.columns
                    )
                },
            )

        # Mark the IR as changed
        self.changed = True
        return nodes

    def _run_call_pd_top_level(self, assign, rhs, func_name, label):
        """transform top-level pandas functions"""
        if func_name == "Series":
            return self._run_call_pd_series(assign, rhs, func_name, label)
        nodes = []

        # mapping of pandas functions to their arguments that require constant values
        top_level_call_const_args = {
            "concat": [(1, "axis"), (3, "ignore_index")],
            "DataFrame": [(2, "columns")],
            "melt": [
                (1, "id_vars"),
                (2, "value_vars"),
                (3, "var_name"),
                (4, "value_name"),
            ],
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
            "pivot": [(1, "index"), (2, "columns"), (3, "values")],
            "pivot_table": [
                (1, "values"),
                (2, "index"),
                (3, "columns"),
                (4, "aggfunc"),
            ],
            "Timestamp": [(2, "tz")],
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
            if data_arg.name in self._updated_containers:
                # TODO[BSE-4021]: Move to _convert_const_key_dict
                non_const_msg = "pd.Series(): When initializing series with a dictionary, it is required that the dict has constant keys"
                raise_bodo_error(non_const_msg)
            if idx_arg != "":
                # This is a defensive check, and isn't expected to be hit
                # so we need the pragma to let coverage pass
                if idx_arg.name not in self.typemap:  # pragma: no cover
                    self.needs_transform = True
                    return nodes
                elif self.typemap[idx_arg.name] != types.none:
                    raise_bodo_error(
                        "pd.Series(): Cannot specify index argument when initializing with a dictionary"
                    )
            msg = "When initializng a series with a dictionary, the keys should be constant strings or constant ints"
            (
                tuples,
                new_nodes,
            ) = bodo.utils.transform._convert_const_key_dict(
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
                isinstance(guard(get_definition, self.func_ir, x), ir.Const)
                for x in kw_default
            )
        elif isinstance(kw_default, ir.Expr):
            if kw_default.op != "build_tuple":
                return [assign]
            ok = all(
                isinstance(guard(get_definition, self.func_ir, x), ir.Const)
                for x in kw_default.items
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
            code,
            freevar_arg_map,
        )

        # we can now change the IR/code since all checks are done (including
        # _replace_load_deref_code)

        # Pop the freevar indices in reverse order to ensure everything
        # stays in the same position
        # i.e. convert [0, 1, 2] to [2, 1, 0]
        for i in sorted(freevar_inds, reverse=True):
            items.pop(i)

        new_co_varnames = (
            code.co_varnames[: code.co_argcount]
            + tuple(freevar_names)
            + code.co_varnames[code.co_argcount :]
        )
        new_co_freevars = tuple(set(code.co_freevars) - set(freevar_names))
        rhs.code = code.replace(
            co_argcount=code.co_argcount + len(freevar_names),
            co_nlocals=code.co_nlocals + len(freevar_names),
            co_code=new_code,
            co_varnames=new_co_varnames,
            co_freevars=new_co_freevars,
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
        """
        import bodosql
        import bodosql.compiler  # isort:skip # noqa
        from bodosql.context_ext import BodoSQLContextType

        # In order to inline the sql() call, we must ensure that the type of the input dataframe(s)
        # are finalized. dataframe type may have changed in typing pass (e.g. due to df setitem)
        # so we shouldn't use the actual type of the dataframes used to initialize the sql_context_var
        def determine_bodosql_context_type(sql_context_var):
            """
            Determine the output type of a BodoSQL context after
            potentially many transformations. Returns None if the
            type cannot be determined.
            """
            sql_ctx_def = guard(get_definition, self.func_ir, sql_context_var)
            if isinstance(sql_ctx_def, ir.Arg):
                # Variable type always available in typemap since it is an argument.
                return self.typemap[sql_context_var.name]
            fdef = guard(find_callname, self.func_ir, sql_ctx_def)
            if fdef is None or len(fdef) < 2:
                # len(fdef) < 2 should always be False, but check
                # ensure we don't have an indexError in another pass.
                return None
            if fdef[0] == "BodoSQLContext":
                # Fold arguments in case someone passes catalog via kwargs
                _, folded_args = bodo.utils.typing.fold_typing_args(
                    fdef[0],
                    sql_ctx_def.args,
                    sql_ctx_def.kws,
                    ("dataframes", "catalog", "default_tz"),
                    {"catalog": None, "default_tz": None},
                )
                names, df_typs = self._get_bodosql_ctx_name_df_typs(folded_args)
                for df_typ in df_typs:
                    if df_typ is None:
                        # Return None if a transformation failed
                        # because a dataframe type is unknown.
                        return None
                # Cannot estimate length from compiled code at this time.
                estimate_row_counts = tuple([None] * len(df_typs))
                catalog_var = folded_args[1]
                if isinstance(catalog_var, ir.Var):
                    if catalog_var.name not in self.typemap:
                        return None
                    catalog_typ = self.typemap[catalog_var.name]
                else:
                    catalog_typ = types.none
                default_tz_var = folded_args[2]
                if isinstance(default_tz_var, ir.Var):
                    if default_tz_var.name not in self.typemap:
                        return None
                    default_tz_typ = self.typemap[default_tz_var.name]
                else:
                    default_tz_typ = types.none
                return BodoSQLContextType(
                    names,
                    df_typs,
                    estimate_row_counts,
                    catalog_typ,
                    default_tz_typ,
                )
            elif fdef[0] == "add_or_replace_view":
                context_type = determine_bodosql_context_type(fdef[1])
                name_typ = self.typemap.get(sql_ctx_def.args[0].name, None)
                if name_typ is None or not is_literal_type(name_typ):
                    # If we can't determine a literal type for name
                    # we must fail.
                    return None
                name_typ = name_typ.literal_value
                df_typ = self.typemap.get(sql_ctx_def.args[1].name, None)
                # Map to a new BodoSQLContextType
                new_names = []
                new_df_typs = []
                new_estimated_row_counts = []
                for old_name_typ, old_df_typ, old_row_count in zip(
                    context_type.names,
                    context_type.dataframes,
                    context_type.estimated_row_counts,
                ):
                    if old_name_typ != name_typ:
                        new_names.append(old_name_typ)
                        new_df_typs.append(old_df_typ)
                        new_estimated_row_counts.append(old_row_count)
                new_names.append(name_typ)
                new_df_typs.append(df_typ)
                # Cannot estimate length from compiled code at this time.
                new_estimated_row_counts.append(None)
                return BodoSQLContextType(
                    tuple(new_names),
                    tuple(new_df_typs),
                    tuple(new_estimated_row_counts),
                    context_type.catalog_type,
                    context_type.default_tz,
                )
            elif fdef[0] == "remove_view":
                context_type = determine_bodosql_context_type(fdef[1])
                name_typ = self.typemap.get(sql_ctx_def.args[0].name, None)
                if name_typ is None or not is_literal_type(name_typ):
                    # If we can't determine a literal type for name
                    # we must fail.
                    return None
                name_typ = name_typ.literal_value
                # Map to a new BodoSQLContextType
                new_names = []
                new_df_typs = []
                new_estimated_row_counts = []
                for old_name_typ, old_df_typ, old_row_count in zip(
                    context_type.names,
                    context_type.dataframes,
                    context_type.estimated_row_counts,
                ):
                    if old_name_typ != name_typ:
                        new_names.append(old_name_typ)
                        new_df_typs.append(old_df_typ)
                        new_estimated_row_counts.append(old_row_count)
                return BodoSQLContextType(
                    tuple(new_names),
                    tuple(new_df_typs),
                    tuple(new_estimated_row_counts),
                    context_type.catalog_type,
                    context_type.default_tz,
                )
            elif fdef[0] in ("add_or_replace_catalog", "remove_catalog"):
                context_type = determine_bodosql_context_type(fdef[1])
                if fdef[0] == "add_or_replace_catalog":
                    catalog_typ = self.typemap.get(sql_ctx_def.args[0].name, None)
                    if catalog_typ is None:
                        # If we can't determine a type for the catalog
                        # we must fail.
                        return None
                else:
                    catalog_typ = types.none
                return BodoSQLContextType(
                    context_type.names,
                    context_type.dataframes,
                    context_type.estimated_row_counts,
                    catalog_typ,
                    context_type.default_tz,
                )
            return None

        sql_context_type = determine_bodosql_context_type(sql_context_var)

        if sql_context_type is None:
            # cannot transform yet if type is not available yet
            return [assign]

        # TODO: Add argument error handling (should reuse signature error checking
        # that will be created in df.head PR).
        kws = dict(rhs.kws)
        sql_var = get_call_expr_arg(
            f"BodoSQLContextType.{func_name}", rhs.args, kws, 0, "sql"
        )
        params_dict_var = get_call_expr_arg(
            f"BodoSQLContextType.{func_name}",
            rhs.args,
            kws,
            1,
            "params_dict",
            default=types.none,
        )
        dynamic_params_var = get_call_expr_arg(
            f"BodoSQLContextType.{func_name}",
            rhs.args,
            kws,
            2,
            "dynamic_params_list",
            default=types.none,
        )
        # Make sure JIT options are not used inside JIT functions
        for k in kws.keys():
            if k not in ("sql", "params_dict", "dynamic_params_list"):
                raise BodoError(
                    f"Argument '{k}' is not supported for BodoSQLContextType.{func_name}() inside JIT functions."
                )

        needs_transform = False
        try:
            err_msg = "BodoSQLContextType.sql() requires a constant sql string"
            sql_str = self._get_const_value(sql_var, label, err_msg)
        except (GuardException, BodoConstUpdatedError):
            needs_transform = True

        if is_overload_none(params_dict_var):
            named_param_keys, named_param_values = [], []
        else:
            arg_name = params_dict_var.name
            if (arg_name in self.typemap) and is_overload_none(self.typemap[arg_name]):
                named_param_keys, named_param_values = [], []
            else:
                try:
                    (
                        named_param_keys,
                        named_param_values,
                    ) = bodo.utils.transform.dict_to_const_keys_var_values_lists(
                        params_dict_var,
                        self.func_ir,
                        self.arg_types,
                        self.typemap,
                        self._updated_containers,
                        self._require_const,
                        label,
                    )
                except GuardException:
                    needs_transform = True

        pre_nodes = []
        if is_overload_none(dynamic_params_var):
            dynamic_param_values = []
        else:
            arg_name = dynamic_params_var.name
            if (arg_name in self.typemap) and is_overload_none(self.typemap[arg_name]):
                dynamic_param_values = []
            elif arg_name in self.typemap and isinstance(
                self.typemap[arg_name], types.BaseTuple
            ):
                try:
                    # Try to fetch the original variables if we have a constant tuple to avoid generating
                    # extra IR.
                    dynamic_param_values = (
                        bodo.utils.transform.tuples_to_vars_value_list(
                            dynamic_params_var, self.func_ir
                        )
                    )
                except GuardException:
                    # If we failed to fetch a constant tuple we can generate valid code by introducing a new variable
                    # for each tuple getitem.
                    dynamic_param_values = []
                    tuple_type = self.typemap[arg_name]
                    for i in range(len(tuple_type.types)):
                        # Create the index variable.
                        index_var = ir.Var(
                            dynamic_params_var.scope,
                            mk_unique_var("$index_var"),
                            dynamic_params_var.loc,
                        )
                        self.typemap[index_var.name] = types.intp
                        assign = ir.Assign(
                            ir.Const(i, dynamic_params_var.loc),
                            index_var,
                            dynamic_params_var.loc,
                        )
                        pre_nodes.append(assign)
                        # Get the variable for the tuple element.
                        var = ir.Var(
                            dynamic_params_var.scope,
                            mk_unique_var("$tuple_var"),
                            dynamic_params_var.loc,
                        )
                        self.typemap[var.name] = tuple_type.types[i]
                        getitem = ir.Expr.getitem(
                            dynamic_params_var, index_var, var.loc
                        )
                        assign = ir.Assign(getitem, var, var.loc)
                        pre_nodes.append(assign)
                        dynamic_param_values.append(var)
            else:
                try:
                    dynamic_param_values = bodo.utils.transform.list_to_vars_value_list(
                        dynamic_params_var, self.func_ir
                    )
                except GuardException:
                    needs_transform = True

        # If any variable needs to be a constant, try and
        # transform the code
        if needs_transform:
            self.needs_transform = True
            return [assign]

        named_param_keys = tuple(named_param_keys)

        for value in named_param_values:
            if value.name not in self.typemap:
                self.needs_transform = True
                return [assign]

        named_param_value_types = tuple(
            [self.typemap[value.name] for value in named_param_values]
        )

        for value in dynamic_param_values:
            if value.name not in self.typemap:
                self.needs_transform = True
                return [assign]

        dynamic_param_value_types = tuple(
            [self.typemap[value.name] for value in dynamic_param_values]
        )

        # Generate a chrome tracing event inside the Numba infrastructure for accurately
        # measuring the time spent in BodoSQL in compilation.
        ev_details = {"name": "BodoSQL Planning: [...]"}
        with event.trigger_event("numba:run_pass", data=ev_details):
            if func_name == "sql":
                (
                    impl,
                    additional_globals_to_lower,
                    sql_plan,
                ) = bodosql.context_ext._gen_sql_plan_pd_func_and_glbls_for_query(
                    sql_context_type,
                    sql_str,
                    dynamic_param_value_types,
                    named_param_keys,
                    named_param_value_types,
                )
                # Save the plan if a cache location is set up.
                BodoSqlPlanCache.cache_bodosql_plan(sql_plan, sql_str)
            elif func_name == "convert_to_pandas":
                (
                    impl,
                    additional_globals_to_lower,
                ) = bodosql.context_ext._gen_pd_func_str_for_query(
                    sql_context_type,
                    sql_str,
                    dynamic_param_value_types,
                    named_param_keys,
                    named_param_value_types,
                )

        self.changed = True
        # BodoSQL generates df.columns setattr, which needs another transform to work
        # (See BodoSQL #189)
        self.needs_transform = True

        # Update the function globals with SQL globals since needed for case handling.
        # See https://github.com/bodo-ai/Bodo/blob/53369bb1817c30e751975b1694ec3f65a648294b/bodo/transforms/dataframe_pass.py#L939
        if (
            self.func_ir.func_id.func.__globals__.keys()
            & additional_globals_to_lower.keys()
        ):  # pragma: no cover
            warnings.warn(
                "SQL globals overlap with JIT globals which may cause errors. This could be because of multiple sql() calls in the same JIT function."
            )

        self.func_ir.func_id.func.__globals__.update(additional_globals_to_lower)

        return replace_func(
            self,
            impl,
            [sql_context_var] + dynamic_param_values + named_param_values,
            extra_globals=additional_globals_to_lower,
            pre_nodes=pre_nodes,
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
            new_df_var = ir.Var(
                df_var.scope, mk_unique_var(df_var.name) + "_2", df_var.loc
            )
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
                f"DataFrame.{func_name}(): non-deterministic inplace change of dataframe schema "
                "not supported.\nSee "
                "https://docs.bodo.ai/latest/bodo_parallelism/not_supported/"
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
            # Adding extra suffix to avoid name conflicts with Numba's renaming
            # e.g. df.1, see test_df_set_col_rename_bug
            new_df_var = ir.Var(
                df_var.scope, mk_unique_var(df_var.name) + "_1", df_var.loc
            )
            out_var = new_df_var
            self.replace_var_dict[df_var.name] = new_df_var
        else:
            # cannot replace variable, but can set existing column with the
            # same data type
            # NOTE: data type is checked in _run_call_set_df_column() and
            # set_dataframe_data()
            out_var = df_var

        # NOTE: avoiding "df" as input argument name to avoid conflicts with user code.
        # see test_df_set_col_rename_bug
        # TODO: rename variables in generated functions
        func = eval(
            "lambda _b_df, cname, arr: bodo.hiframes.dataframe_impl.set_df_col(_b_df, cname, arr, _inplace)"
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
            assert nodes[-1].value.name in self.typemap, (
                f"Internal error in _run_df_set_column: {nodes[-1].value.name} not present in type map"
            )
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
        else:
            # impl6 in test_set_df_column_names in bodo/tests/test_dataframe_part2.py
            # Fails if this case throws an error. I'm not certain if this is function
            # is intended to return None, if the type cannot be found,
            # or if this is a bug. For now I'm just going to
            # keep it as is and return "None"
            return None

    def _replace_arg_with_literal(
        self, func_name, rhs, func_args, label, pyobject_to_literal=False
    ):
        """replace a function argument that needs to be constant with a literal to
        enable constant access in overload. This may force JIT arguments to be literals
        if needed to satify constant requirements.
        """

        kws = dict(rhs.kws)
        nodes = []
        for arg_no, arg_name in func_args:
            var = get_call_expr_arg(func_name, rhs.args, kws, arg_no, arg_name, "")
            # skip if argument not specified or literal already
            if var == "":
                continue
            if self._is_constant_var(var.name):
                if var.name in self._updated_containers:
                    # loop unrolling can potentially make updated lists constants
                    if self.ran_transform:
                        raise BodoError(
                            f"{func_name}(): argument '{arg_name}' requires a constant value but variable '{var.name}' is updated inplace using '{self._updated_containers[var.name]}'\n{rhs.loc.strformat()}\n"
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
            if self.ran_transform and err_msg and self.tried_unrolling:
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
            # TODO[BSE-4021]: Do we need to check self._updated_containers?
            # We need to be clear about the "constant" guarantee here.
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
        state = numba.core.compiler.StateDict(
            {"func_ir": self.func_ir, "typemap": None}
        )
        strip_phis_pass = numba.core.typed_passes.PreLowerStripPhis()
        create_nested_run_pass_event(strip_phis_pass.name(), state, strip_phis_pass)

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
            # NOTE: Using ir_utils.next_label() can cause conflicts with block numbers
            # across copied iteration bodies. For example, for block numbers (1, 2, 3),
            # and initial next label 1, the first and second iteration bodies will
            # conflict with (2, 3, 4) and (3, 4, 5) labels overlapping.
            offset = max(self.func_ir.blocks.keys()) + 1
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
        state = numba.core.compiler.StateDict(
            {"func_ir": self.func_ir, "locals": self.locals}
        )
        ssa_pass = numba.core.untyped_passes.ReconstructSSA()
        create_nested_run_pass_event(ssa_pass.name(), state, ssa_pass)
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

    def _is_call_op_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """Performs an equality check on the index_def expr with the possible
        call expressions that represent filters.

        Args:
            index_def (ir.Expr): The expression that may be a valid call filter.
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid logical operator?
        """
        if is_expr(index_def, "call"):
            call_list = find_callname(func_ir, index_def, self.typemap)
            if len(call_list) == 2 and (
                isinstance(call_list[1], ir.Var)
                # checking call_list[1] == "pandas" to handle pd.isna/pd.notna cases generated
                # by BodoSQL
                or call_list[1] in ("pandas", "bodo.pandas")
                or call_list[1] == "bodosql.kernels"
            ):
                return call_list[0] in (
                    "notna",
                    "isna",
                    "notnull",
                    "isnull",
                    "isin",
                    "startswith",
                    "endswith",
                ) or call_list in (
                    ("is_in", "bodosql.kernels"),
                    ("like_kernel", "bodosql.kernels"),
                    ("regexp_like", "bodosql.kernels"),
                )
        return False

    def _is_logical_not_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """Performs an equality check on the index_def expr with the
        NOT operators, operator.invert (Pandas uses ~) and the
        boolnot bodosql_array_kernel.

        Args:
            index_def (ir.Expr): The expression that may be a valid not expression.
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid not operator?
        """
        if is_expr(index_def, "unary"):
            return index_def.fn == operator.invert
        elif is_call(index_def):
            call_name = guard(find_callname, func_ir, index_def, self.typemap)
            return call_name == ("boolnot", "bodosql.kernels")
        return False

    def _is_logical_op_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """Performs an equality check on the index_def expr with the possible
        logical operators (AND, OR, or a comparison operator). This also supports
        the equivalent BodoSQL array kernels to ensure that
        we can support filter pushdown for both the Pythonic version of these
        comparison operators and their SQL array kernels.

        Args:
            index_def (ir.Expr): The expression that may be a valid logical operation
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid logical operator?
        """
        return (
            self._is_cmp_op_filter_pushdown(index_def, func_ir)
            or self._is_and_filter_pushdown(index_def, func_ir)
            or self._is_or_filter_pushdown(index_def, func_ir)
        )

    def _is_cmp_op_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """
        Performs an equality check on the index_def expr with the valid
        comparison operators (e.g. !=) or their equivalent BodoSQL array kernels
        (e.g. bodosql.kernels.not_equal). This is to ensure that
        we can support filter pushdown for both the Pythonic version of these
        comparison operators and their SQL array kernels.

        Args:
            index_def (ir.Expr): The expression that may be a valid comparison operation.
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid comparison operation?
        """
        if is_expr(index_def, "binop"):
            return index_def.fn in (
                operator.eq,
                operator.ne,
                operator.lt,
                operator.gt,
                operator.le,
                operator.ge,
            )
        elif is_call(index_def):
            call_name = guard(find_callname, func_ir, index_def, self.typemap)
            if len(call_name) == 2 and call_name[1] == "bodosql.kernels":
                return call_name[0] in (
                    "equal",
                    "not_equal",
                    "less_than",
                    "greater_than",
                    "less_than_or_equal",
                    "greater_than_or_equal",
                )
        return False

    def _is_and_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """
        Performs an equality check on the index_def expr with & / AND,
        depending on whether the operator is a binop or a function call respectively.
        This is to ensure that we can support filter pushdown for AND as well instead of just &.

        Args:
            index_def (ir.Expr): The expression that may be a valid AND operation.
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid AND operation?
        """
        if is_expr(index_def, "binop"):
            return index_def.fn == operator.and_
        elif is_call(index_def):
            call_name = guard(find_callname, func_ir, index_def, self.typemap)
            return call_name == ("booland", "bodosql.kernels")
        else:
            return False

    def _is_or_filter_pushdown(
        self, index_def: ir.Expr, func_ir: ir.FunctionIR
    ) -> bool:
        """
        Performs an equality check on the index_def expr with | / OR,
        depending on whether the operator is a binop or a function call respectively.
        This is to ensure that we can support filter pushdown for OR as well instead of just |.

        Args:
            index_def (ir.Expr): The expression that may be a valid OR operation.
            func_ir (ir.FunctionIR): Function IR used for finding the call expression.

        Returns:
            bool: Is this expression a valid OR operation?
        """
        if is_expr(index_def, "binop"):
            return index_def.fn == operator.or_
        elif is_call(index_def):
            call_name = guard(find_callname, func_ir, index_def, self.typemap)
            return call_name == ("boolor", "bodosql.kernels")
        else:
            return False

    def _is_na_filter_pushdown_func(self, index_def, index_call_name):
        """
        Does an expression match a supported is/not na call that can
        be used in filterpushdown.
        """
        if not (
            is_call(index_def)
            and index_call_name[0] in ("isna", "isnull", "notna", "notnull")
        ):
            return False

        # handle both Series.isna() and pd.isna() forms
        varname = (
            index_call_name[1].name
            if isinstance(index_call_name[1], ir.Var)
            else index_def.args[0].name
        )
        method_obj_type = self.typemap.get(varname, None)

        # rerun type inference if we don't have the method's object type yet
        # read_sql_table (and other I/O calls in the future) is handled in typing pass
        # so the Series type may not be available yet
        if method_obj_type in (None, types.unknown, types.undefined):
            self.needs_transform = True
            return False

        # BodoSQL generates pd.isna(arr)
        return is_array_typ(method_obj_type, True)

    def _is_isin_filter_pushdown_func(self, index_def, index_call_name):
        """
        Does an expression match a supported isin call that can be
        used in filter pushdown.

        Note: we only allow series isin with lists/sets. We don't support Series/Array
        because we don't want to worry about distributed data and tuples aren't
        supported in the isin API.
        """

        if not is_call(index_def):
            # Immediately return false if we don't have a call
            return False
        elif index_call_name[0] == "isin":
            method_obj_type = self.typemap.get(index_call_name[1].name, None)

            # rerun type inference if we don't have the method's object type yet
            # read_sql_table (and other I/O calls in the future) is handled in typing pass
            # so the Series type may not be available yet
            if method_obj_type in (None, types.unknown, types.undefined):
                self.needs_transform = True
                return False

            if not isinstance(method_obj_type, SeriesType):
                return False

            list_set_typ = self.typemap.get(index_def.args[0].name, None)
            # We don't support casting pd_timestamp_type/datetime64 values in arrow, so we avoid
            # filter pushdown in that situation.
            return (
                isinstance(list_set_typ, (types.List, types.Set))
                and list_set_typ.dtype != bodo.types.datetime64ns
                and not isinstance(
                    list_set_typ.dtype,
                    bodo.hiframes.pd_timestamp_ext.PandasTimestampType,
                )
            )
        elif index_call_name == ("is_in", "bodosql.kernels"):
            # In the case that we're hadling the bodsql is_in array kernel, we expect arg1 to be
            # an array. We need to rerun type inference if we don't have the type yet
            arg1_arr_type = self.typemap.get(index_def.args[1].name, None)
            if arg1_arr_type in (None, types.unknown, types.undefined):
                self.needs_transform = True
                return False

            # We require that arg1 is a replicated array to perform filter pushdown.
            # In the bodoSQL codegen, this value should be lowered
            # as a global, and all globals are required to be replicated.
            is_arg1_global = isinstance(
                guard(get_definition, self.func_ir, index_def.args[1].name),
                numba.core.ir.Global,
            )

            # TODO: verify if we have the same issue with datetime64ns/timestamps that the
            # series isin implementation does.
            return (
                is_arg1_global
                and arg1_arr_type.dtype != bodo.types.datetime64ns
                and not isinstance(
                    arg1_arr_type.dtype,
                    bodo.hiframes.pd_timestamp_ext.PandasTimestampType,
                )
            )

        else:
            # In all other cases, return False
            return False

    def _starts_ends_with_filter_pushdown_func(self, index_def, index_call_name):
        """
        Does an expression match a supported startswith/endswith call that can be
        used in filter pushdown.
        """
        if not (
            is_call(index_def) and index_call_name[0] in ("startswith", "endswith")
        ):
            return False

        method_obj_type = self.typemap.get(index_call_name[1].name, None)

        # rerun type inference if we don't have the method's object type yet
        # read_sql_table (and other I/O calls in the future) is handled in typing pass
        # so the Series type may not be available yet
        if method_obj_type in (None, types.unknown, types.undefined):
            self.needs_transform = True
            return False

        if not isinstance(method_obj_type, SeriesStrMethodType):
            return False

        return True

    def _is_like_filter_pushdown_func(self, index_def: ir.Stmt, index_call_name):
        """Does an expression match a like call that may be possible to support
        in filter pushdown?

        Args:
            index_def (ir.Stmt): The index expression to check.
            index_call_name (Tuple[str, str | ir.Var]): A 2-tuple identifying the function call.
        """
        if not (
            is_call(index_def) and index_call_name == ("like_kernel", "bodosql.kernels")
        ):
            return False

        # Filter pushdown is currently only possible if both the pattern and escape are constants
        # and we are doing case sensitive matching.
        args = index_def.args
        # Pattern and escape are args 1 and 2
        for arg_no in (1, 2):
            const_arg = args[arg_no].name
            arg_type = self.typemap.get(const_arg, None)
            if arg_type in (None, types.unknown, types.undefined):
                self.needs_transform = True
                return False

            if types.unliteral(arg_type) not in (types.unicode_type, types.none):
                # We don't support filter pushdown with optional types
                # or arrays.
                return False

        # case insensitive is argument 3
        case_insensitive = args[3].name
        case_insensitive_type = self.typemap.get(case_insensitive, None)
        if case_insensitive_type in (None, types.unknown, types.undefined):
            self.needs_transform = True
            return False

        # We may need to recompile to get the constant version to avoid errors.
        if not is_overload_constant_bool(case_insensitive_type):
            return False

        # We can do filter pushdown for both values of case_insensitive.
        return True


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
                if lhs not in equiv_vars:
                    equiv_vars[rhs].add(lhs)
                    equiv_vars[lhs] = equiv_vars[rhs]
                elif rhs not in equiv_vars:
                    equiv_vars[lhs].add(rhs)
                    equiv_vars[rhs] = equiv_vars[lhs]
                else:
                    equiv_vars[lhs].add(rhs)
                    equiv_vars[lhs].add(lhs)
                    equiv_vars[lhs] |= equiv_vars[rhs]
                    equiv_vars[rhs] = equiv_vars[lhs]
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
                and stmt.value.attr in container_update_method_names
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


def get_unary_arg(child_def: ir.Expr) -> ir.Var:
    """The child accessors of an expr differ depending on whether
    the expr is a unary or a function call.

    Therefore this is a wrapper method to support both types of expr
    with a single interface.

    Args:
        child_def (ir.Expr): The expression that maps to a unary function.
            It is either op = "call" or op = "unary"

    Returns:
        ir.Var: The IR variable for the argument.
    """
    if is_expr(child_def, "unary"):
        return child_def.value
    require(is_expr(child_def, "call") and len(child_def.args) == 1)
    return child_def.args[0]


def get_binop_arg(child_def, arg_no):
    """
    The child accessors of an expr differ depending on whether
    the expr is a binop or a function call.

    Therefore this is a wrapper method to support both types of expr
    with a single interface, using an index (arg_no) of either 0 or 1, which
    maps to child_def.lhs and child_def.rhs respectively.
    """
    require(arg_no == 0 or arg_no == 1)

    if is_expr(child_def, "binop"):
        if arg_no == 0:
            return child_def.lhs
        else:
            return child_def.rhs

    require(is_expr(child_def, "call") and len(child_def.args) == 2)
    return child_def.args[arg_no]


def get_cmp_operator(
    index_def: ir.Expr,
    scalar_type: types.Type,
    is_sql_op: bool,
    reverse_op: bool,
    func_ir: ir.FunctionIR,
) -> str:
    """Derive the operator string used for filter pushdown with binary
    comparison operators. These operators can either be a binop or a call to
    a BodoSQL array kernel.

    Args:
        index_def (ir.Expr): The expression in the IR responsible for the comparison.
            This is either a binop or a call expression.
        scalar_type (types.Type): The type of the scalar type. This is used for validation
            and a special None case with call expressions.
        is_sql_op (bool): Should the operator be output using standard SQL syntax or pyarrow
            sytnax.
        reverse_op (bool): Is the column arg1 and not arg0. This is important for "reversing" certain
            operators as opposed to the function. For example (scalar < column) should output
            ">" despite the binop being operator.lt.
        func_ir (ir.FunctionIR): The function IR object. This is used to get the callname.

    Returns:
        str: The filter pushdown operator string.

    Raise GuardException: The function is not formatted in way that can handle filter pushdown.
    """
    # Map the BodoSQL array kernels to equivalent operators.
    fn_name_map = {
        "equal": operator.eq,
        "not_equal": operator.ne,
        "less_than": operator.lt,
        "greater_than": operator.gt,
        "less_than_or_equal": operator.le,
        "greater_than_or_equal": operator.ge,
    }
    # Map the operator to its filter pushdown operator string.
    if reverse_op:
        # Operator mapping used to support situations
        # where the column is on the RHS. Since Pyarrow
        # format is ("col", op, scalar), we must invert certain
        # operators.
        op_map = {
            operator.eq: "=" if is_sql_op else "==",
            operator.ne: "<>" if is_sql_op else "!=",
            operator.lt: ">",
            operator.le: ">=",
            operator.gt: "<",
            operator.ge: "<=",
        }
    else:
        op_map = {
            operator.eq: "=" if is_sql_op else "==",
            operator.ne: "<>" if is_sql_op else "!=",
            operator.lt: "<",
            operator.le: "<=",
            operator.gt: ">",
            operator.ge: ">=",
        }
    # The other argument must be a scalar, not an array
    require(not bodo.utils.utils.is_array_typ(scalar_type, True))
    if is_call(index_def):
        # We can't do filter pushdown with optional types yet.
        require(scalar_type != types.optional)
        if scalar_type == types.none:
            # SQL comparison functions always return NULL if an input is NULL.
            return "ALWAYS_NULL"
        callname = guard(find_callname, func_ir, index_def)
        require(callname is not None)
        op = fn_name_map[callname[0]]
    else:
        # Python operators shouldn't compare with None or optional values.
        require(scalar_type not in (types.optional, types.none))
        op = index_def.fn
    return op_map[op]


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


def _bc_stream_to_bytecode(bc_stream, original_code):
    """convert a stream of unpacked bytecode to a bytearray, reverses disassembly"""
    import dis

    from numba.core.bytecode import ARG_LEN, CODE_LEN

    out = bytearray(original_code.co_code)
    # TODO: Update this to match Python 3.13 dis code:
    # https://github.com/python/cpython/blob/6280bb547840b609feedb78887c6491af75548e8/Lib/dis.py#L862
    for (
        offset,
        op,
        arg,
    ) in bc_stream:
        out[offset] = op
        if op in dis.hasarg:
            for i in range(ARG_LEN):
                out[offset + CODE_LEN + i] = arg & 0xFF
                arg >>= 8

        else:
            assert arg is None
    return out


def _replace_load_deref_code(code, freevar_arg_map):
    """replace load of free variables in byte code with load of new arguments and
    adjust local variable indices due to new arguments in co_varnames.
    # https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST
    # https://docs.python.org/3/library/inspect.html
    # https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
    raises GuardException if there is STORE_DEREF in input code (for setting freevars)
    """
    import dis

    prev_argcount = code.co_argcount
    prev_n_locals = code.co_nlocals

    def _patch_opargs(code, freevar_arg_map, prev_argcount, prev_n_locals):
        """
        Patch the code object to replace the free variable load with an argument load.
        Returns a stream of the updated bytecode.
        """

        # cannot handle cases that write to free variables
        banned_ops = (dis.opmap["STORE_DEREF"], dis.opmap["LOAD_CLOSURE"])
        # local variable access to be adjusted
        local_varname_ops = (
            dis.opmap["LOAD_FAST"],
            dis.opmap["STORE_FAST"],
            dis.opmap["DELETE_FAST"],
        )
        n_new_args = len(freevar_arg_map)
        for inst in dis.get_instructions(code):
            (
                offset,
                op,
                arg,
            ) = inst.offset, inst.opcode, inst.arg
            require(op not in banned_ops)

            # adjust local variable access since index includes arguments
            if op in local_varname_ops and arg >= prev_argcount:
                arg += n_new_args

            # Python 3.11 copies free vars into local variables in the beginning of the
            # function. We need to update LOAD_DEREF indices accordingly. See:
            # https://docs.python.org/3.11/library/dis.html#opcode-COPY_FREE_VARS
            # https://github.com/python/cpython/blob/cce6ba91b3a0111110d7e1db828bd6311d58a0a7/Python/ceval.c#L3206
            if "COPY_FREE_VARS" in dis.opmap and op == dis.opmap["COPY_FREE_VARS"]:
                freevar_arg_map = {
                    k + prev_n_locals: v for k, v in freevar_arg_map.items()
                }

            # replace free variable load
            if op == dis.opmap["LOAD_DEREF"] and arg in freevar_arg_map:
                op = dis.opmap["LOAD_FAST"]
                arg = freevar_arg_map[arg]
            yield offset, op, arg

    return bytes(
        _bc_stream_to_bytecode(
            _patch_opargs(code, freevar_arg_map, prev_argcount, prev_n_locals),
            code,
        )
    )


def _get_state_defining_call(func_ir, state, fn):
    """Find the expression defining `state` (e.g. join/table builder state)
    and return it only if it is a call to `fn`."""
    defn = get_definition(func_ir, state)
    init = find_callname(func_ir, defn)
    if init != fn:
        raise GuardException("initialization is not the expected call")
    return defn
