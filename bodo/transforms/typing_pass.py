"""
Bodo type inference pass that performs transformations that enable typing of the IR
according to Bodo requirements (using partial typing).
"""
import operator
import itertools
import numba
from numba.core import types, ir, ir_utils
from numba.core.compiler_machinery import register_pass
from numba.core.typed_passes import NopythonTypeInference, PartialTypeInference
from numba.core.ir_utils import (
    find_topo_order,
    build_definitions,
    mk_unique_var,
    guard,
    find_callname,
    get_definition,
    find_const,
    compile_to_numba_ir,
    compute_cfg_from_blocks,
    replace_arg_nodes,
    dprint_func_ir,
    require,
    GuardException,
)
import bodo
from bodo.utils.typing import (
    ConstList,
    ConstSet,
    BodoError,
    get_registry_consts,
    add_consts_to_registry,
    is_literal_type,
    CONST_DICT_SENTINEL,
)
from bodo.utils.utils import (
    is_assign,
    is_call,
    is_expr,
    get_getsetitem_index_var,
    find_build_tuple,
)
from bodo.utils.transform import (
    update_node_list_definitions,
    compile_func_single_block,
    get_call_expr_arg,
    get_const_value_inner,
    set_call_expr_arg,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.dataframe_indexing import DataFrameILocType
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType

# global flag indicating that we are in partial type inference, so that error checking
# code can raise regular Exceptions that can potentially be handled here
in_partial_typing = False
# global flag set by error checking code (e.g. df.drop) indicating that a transformation
# in the typing pass is required. Necessary since types.unknown may not be assigned to
# all types by Numba properly, e.g. TestDataFrame::test_df_drop_inplace1.
typing_transform_required = False


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoTypeInference(PartialTypeInference):
    _name = "bodo_type_inference"

    def run_pass(self, state):
        """run Bodo type inference pass
        """
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
                types.unknown not in state.typemap.values()
                and not curr_typing_pass_required
            ):
                break
            typing_transforms_pass = TypingTransforms(
                state.func_ir,
                state.typingctx,
                state.typemap,
                state.calltypes,
                state.args,
            )
            changed = typing_transforms_pass.run()
            # can't be typed if IR not changed
            if not changed:
                # error will be raised below if there are still unknown types
                break

        dprint_func_ir(state.func_ir, "after typing pass")
        # run regular type inference again with _raise_errors = True to set function
        # return type and raise errors if any
        # TODO: avoid this extra pass when possible in Numba
        try:
            self._raise_errors = True
            NopythonTypeInference.run_pass(self, state)
        finally:
            self._raise_errors = False
        return True


class TypingTransforms:
    """
    Transformations that enable typing of the IR according to Bodo requirements.

    Infer possible constant values (e.g. lists) using partial typing info and transform
    them to constants so that functions like groupby() can be typed properly.
    """

    def __init__(self, func_ir, typingctx, typemap, calltypes, arg_types):
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
        self.changed = False

    def run(self):
        # XXX: the block structure shouldn't change in this pass since labels
        # are used in analysis (e.g. df creation points in rhs_labels)
        blocks = self.func_ir.blocks
        topo_order = find_topo_order(blocks)
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
            if rhs.op == "build_map":
                rhs.items = [(zero_var, zero_var)]
            if rhs.op == "build_list":
                rhs.items = [zero_var]

        return self.changed

    def _run_assign(self, assign, label):
        rhs = assign.value

        if isinstance(rhs, ir.Expr) and rhs.op in ("binop", "inplace_binop"):
            return self._run_binop(assign, rhs)

        if is_call(rhs):
            return self._run_call(assign, rhs, label)

        if isinstance(rhs, ir.Expr) and rhs.op in ("getitem", "static_getitem"):
            return self._run_getitem(assign, rhs)

        # replace df.columns with constant StringIndex
        if (
            is_expr(rhs, "getattr")
            and rhs.value.name in self.typemap
            and isinstance(self.typemap[rhs.value.name], DataFrameType)
            and rhs.attr == "columns"
        ):
            vals = self.typemap[rhs.value.name].columns
            return self._gen_const_string_index(assign.target, rhs, vals)

        # replace ConstSet with Set since we don't support 'set' methods yet
        # e.g. s.add()
        if (
            is_expr(rhs, "getattr")
            and rhs.value.name in self.typemap
            and isinstance(self.typemap[rhs.value.name], ConstSet)
        ):  # pragma: no cover
            # TODO: test coverage
            val_typ = self.typemap[rhs.value.name]
            self.typemap.pop(rhs.value.name)
            self.typemap[rhs.value.name] = types.Set(val_typ.dtype)

        return [assign]

    def _run_getitem(self, assign, rhs):
        target = rhs.value
        target_typ = self.typemap.get(target.name, None)
        nodes = []
        idx = get_getsetitem_index_var(rhs, self.typemap, nodes)
        idx_typ = self.typemap.get(idx.name, None)

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
            col_slice = guard(_get_const_slice, self.func_ir, slice_var)
            if col_slice is None:
                raise BodoError("slice2 in df.iloc[slice1,slice2] should be constant")

            # create output df
            columns = target_typ.df_type.columns
            # get df arrays using const slice
            data_outs = []
            for i in range(len(columns))[col_slice]:
                arr = "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})[idx]".format(
                    i
                )
                data_outs.append(arr)
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx]"
            header = "def impl(I, idx):\n"
            header += "  df = I._obj\n"
            impl = bodo.hiframes.dataframe_impl._gen_init_df(
                header, columns[col_slice], ", ".join(data_outs), index
            )
            return nodes + compile_func_single_block(
                impl, [target, tup_list[0]], assign.target, self
            )

        nodes.append(assign)
        return nodes

    def _run_setitem(self, inst, label):
        target_typ = self.typemap.get(inst.target.name, None)
        nodes = []
        # idx = get_getsetitem_index_var(inst, self.typemap, nodes)
        # idx_typ = self.typemap.get(idx.name, None)

        # df["B"] = A
        if isinstance(target_typ, DataFrameType):
            # cfg needed for set df column
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            self.changed = True
            return nodes + self._run_df_set_column(inst, label, cfg)

        return nodes + [inst]

    def _run_call(self, assign, rhs, label):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:  # pragma: no cover
            # TODO: test coverage
            return [assign]

        func_name, func_mod = fdef

        # handle df.method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DataFrameType
        ):
            return self._run_call_dataframe(assign, rhs, func_mod, func_name, label)

        # handle df.groupby().method() calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self._get_method_obj_type(func_mod, rhs.func), DataFrameGroupByType
        ):
            return self._run_call_df_groupby(assign, rhs, func_mod, func_name)

        # set() with constant list arg
        if (
            fdef == ("set", "builtins")
            and len(rhs.args) == 1
            and self._get_const_seq(rhs.args[0]) is not None
        ):
            vals = self._get_const_seq(rhs.args[0])
            return self._gen_consts_call(assign.target, vals, False)

        # list() with constant list/set arg
        if (
            fdef == ("list", "builtins")
            and len(rhs.args) == 1
            and self._get_const_seq(rhs.args[0]) is not None
        ):
            vals = self._get_const_seq(rhs.args[0])
            return self._gen_consts_call(assign.target, vals)

        return [assign]

    def _run_binop(self, assign, rhs):
        arg1_typ = self.typemap.get(rhs.lhs.name, None)
        arg2_typ = self.typemap.get(rhs.rhs.name, None)
        target_typ = self.typemap.get(assign.target.name, None)

        # add of constant lists, e.g. ["A"] + ["B"]
        if (
            isinstance(arg1_typ, ConstList)
            and isinstance(arg2_typ, ConstList)
            and rhs.fn == operator.add
        ):
            vals = get_registry_consts(arg1_typ.const_no) + get_registry_consts(
                arg2_typ.const_no
            )
            return self._gen_consts_call(assign.target, vals)

        # sub of constant sets, e.g. {"A", "B"} - {"B"}
        if (
            isinstance(arg1_typ, ConstSet)
            and isinstance(arg2_typ, ConstSet)
            and rhs.fn == operator.sub
        ):
            # sort output of set diff op to be consistent across processors since const
            # values can be keys to groupby/sort/join which require consistent order
            vals = sorted(
                set(get_registry_consts(arg1_typ.const_no))
                - set(get_registry_consts(arg2_typ.const_no))
            )
            return self._gen_consts_call(assign.target, vals, False)

        # replace ConstSet with Set if there a set op we don't support here
        # e.g. s1 | s2
        # target type may be unknown
        if isinstance(arg1_typ, ConstSet) and isinstance(
            target_typ, ConstSet
        ):  # pragma: no cover
            # TODO: test coverage
            arg_def = guard(get_definition, self.func_ir, rhs.lhs)
            assert is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "add_consts_to_type",
                "bodo.utils.typing",
            )
            rhs.lhs = arg_def.args[0]
            self.typemap.pop(assign.target.name)
            self.typemap[assign.target.name] = types.Set(target_typ.dtype)

        if isinstance(arg2_typ, ConstSet) and isinstance(
            target_typ, ConstSet
        ):  # pragma: no cover
            # TODO: test coverage
            arg_def = guard(get_definition, self.func_ir, rhs.rhs)
            assert is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "add_consts_to_type",
                "bodo.utils.typing",
            )
            rhs.rhs = arg_def.args[0]
            self.typemap.pop(assign.target.name)
            self.typemap[assign.target.name] = types.Set(target_typ.dtype)

        return [assign]

    def _run_call_dataframe(self, assign, rhs, df_var, func_name, label):
        """Handle dataframe calls that need transformation to meet Bodo requirements
        """
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
            "rename": [(2, "columns")],
        }

        if func_name in df_call_const_args:
            func_args = df_call_const_args[func_name]
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args)

        # transform df.assign() here since (**kwargs) is not supported in overload
        if func_name == "assign":
            return nodes + self._handle_df_assign(assign.target, rhs, df_var)

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

    def _handle_df_assign(self, lhs, rhs, df_var):
        """replace df.assign() with its implementation to avoid overload errors with
        (**kwargs)
        """
        kws = dict(rhs.kws)
        df_type = self.typemap[df_var.name]
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
        return compile_func_single_block(impl, [df_var] + list(kws.values()), lhs, self)

    def _run_call_df_groupby(self, assign, rhs, groupby_var, func_name):
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
            nodes += self._replace_arg_with_literal(func_name, rhs, func_args)

        return nodes + [assign]

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
        """See if label dominates all labels of df_var's definitions
        """
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

    def _run_df_set_column(self, inst, label, cfg):
        """replace setitem of string index with a call to handle possible
        dataframe case where schema is changed:
        df['new_col'] = arr  ->  df2 = set_df_col(df, 'new_col', arr)
        dataframe_pass will replace set_df_col() with regular setitem if target
        is not dataframe
        """
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
        # TODO: add this check back in
        # if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
        #     raise ValueError("setting dataframe columns inside conditionals and"
        #                      " loops not supported yet")
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
        cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
        nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]
        inplace = not dominates

        func = lambda df, cname, arr: bodo.hiframes.dataframe_impl.set_df_col(
            df, cname, arr, _inplace
        )  # pragma: no cover
        f_block = compile_to_numba_ir(
            func, {"bodo": bodo, "_inplace": inplace}
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df_var, cname_var, inst.value])
        nodes += f_block.body[:-2]

        if dominates:
            # rename the dataframe variable to keep schema static
            new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            nodes[-1].target = new_df_var
            self.replace_var_dict[df_var.name] = new_df_var
        else:
            # cannot replace variable, but can set existing column with the
            # same data type
            # TODO: check data type and throw clear error
            nodes[-1].target = df_var

        return nodes

    def _gen_consts_call(self, target, vals, is_list=True):
        const_obj, const_no = add_consts_to_registry(vals)
        val_reps = ", ".join(["'{}'".format(c) for c in vals])
        in_brac = "[" if is_list else "set(["
        out_brac = "]" if is_list else "])"
        func_text = "def _build_f():\n"
        func_text += "  return bodo.utils.typing.add_consts_to_type({}{}{}, {})\n".format(
            in_brac, val_reps, out_brac, const_no
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        _build_f = loc_vars["_build_f"]
        nodes = compile_func_single_block(_build_f, (), target, self)
        self.typemap.pop(target.name)
        self.typemap[target.name] = self.typemap[nodes[-1].value.name]
        self.changed = True

        # HACK keep const values object around as long as the function is being compiled
        # by adding it as an attribute to some compilation object
        setattr(self.func_ir, "const_obj{}".format(const_no), const_obj)
        return nodes

    def _gen_const_string_index(self, target, rhs, vals):
        const_obj, const_no = add_consts_to_registry(vals)
        val_reps = ", ".join(["'{}'".format(c) for c in vals])
        const_call = "bodo.utils.typing.add_consts_to_type([{}], {})".format(
            val_reps, const_no
        )
        str_arr = "bodo.libs.str_arr_ext.str_arr_from_sequence({})".format(const_call)
        func_text = "def _gen_str_ind():\n"
        func_text += "  return bodo.hiframes.pd_index_ext.init_string_index({})\n".format(
            str_arr
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        _gen_str_ind = loc_vars["_gen_str_ind"]
        nodes = compile_func_single_block(_gen_str_ind, (), target, self)

        # HACK keep const values object around as long as the function is being compiled
        # by adding it as an attribute to some compilation object
        setattr(self.func_ir, "const_obj{}".format(const_no), const_obj)
        return nodes

    def _get_const_seq(self, var):
        """returns constant values if 'var' represents a constant list/set. Otherwise,
        returns None
        """
        # constant list/set type
        if var.name in self.typemap and isinstance(
            self.typemap[var.name], (ConstList, ConstSet)
        ):
            return get_registry_consts(self.typemap[var.name].const_no)

        # constant StringIndex case coming from df.columns
        # pattern match init_string_index(str_arr_from_sequence(["A", "B"]))
        var_def = guard(get_definition, self.func_ir, var)
        if is_call(var_def) and guard(find_callname, self.func_ir, var_def) == (
            "init_string_index",
            "bodo.hiframes.pd_index_ext",
        ):
            arg_def = guard(get_definition, self.func_ir, var_def.args[0])
            if is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "str_arr_from_sequence",
                "bodo.libs.str_arr_ext",
            ):
                if arg_def.args[0].name in self.typemap and isinstance(
                    self.typemap[arg_def.args[0].name], ConstList
                ):
                    return get_registry_consts(
                        self.typemap[arg_def.args[0].name].const_no
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

    def _replace_arg_with_literal(self, func_name, rhs, func_args):
        """replace a function argument that needs to be constant with a literal to
        enable constant access in overload. This may force JIT arguments to be literals
        if needed to satify constant requirements.
        """
        kws = dict(rhs.kws)
        nodes = []
        for (arg_no, arg_name) in func_args:
            var = get_call_expr_arg(func_name, rhs.args, kws, arg_no, arg_name, "")
            # skip if argument not specified or literal already
            if var == "" or is_literal_type(self.typemap.get(var.name, None)):
                continue
            # get constant value for variable if possible.
            # Otherwise, just skip, assuming that the issue may be fixed later or
            # overload will raise an error if necessary.
            try:
                val = get_const_value_inner(
                    self.func_ir, var, self.arg_types, self.typemap
                )
            except GuardException:
                continue
            # replace argument variable with a new variable holding constant
            new_var = _create_const_var(val, var.name, var.scope, rhs.loc, nodes)
            set_call_expr_arg(new_var, rhs.args, kws, arg_no, arg_name)
            # var is not used here anymore, add to _transformed_vars so it can
            # potentially be removed since some dictionaries (e.g. in agg) may not be
            # type stable
            self._transformed_vars.add(var.name)
            self.changed = True

        rhs.kws = list(kws.items())
        return nodes


def _get_const_slice(func_ir, var):
    """get constant slice value for a slice variable if possible
    """
    # var definition should be a slice() call
    var_def = get_definition(func_ir, var)
    require(find_callname(func_ir, var_def) == ("slice", "builtins"))
    require(len(var_def.args) in (2, 3))
    # get start/stop/step values from call
    start = find_const(func_ir, var_def.args[0])
    stop = find_const(func_ir, var_def.args[1])
    step = find_const(func_ir, var_def.args[2]) if len(var_def.args) == 3 else None
    return slice(start, stop, step)


def _create_const_var(val, name, scope, loc, nodes):
    """create a new variable that holds constant value 'val'. Generates constant
    creation IR nodes and adds them to 'nodes'.
    """
    new_var = ir.Var(scope, mk_unique_var(name), loc)
    # NOTE: create a tuple for both list/tuple, assuming that all functions accept both
    # equally (e.g. passing a tuple instead of a list is not an error).
    if isinstance(val, (tuple, list)):
        const_node = ir.Expr.build_tuple(
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
