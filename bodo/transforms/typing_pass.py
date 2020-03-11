"""
Bodo type inference pass that performs transformations that enable typing of the IR
according to Bodo requirements (using partial typing).
"""
import operator
from numba import types, ir, ir_utils
from numba.compiler_machinery import register_pass
from numba.typed_passes import NopythonTypeInference, PartialTypeInference
from numba.ir_utils import (
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
)
import bodo
from bodo.libs.str_ext import string_type
from bodo.utils.typing import ConstList, ConstSet, BodoError
from bodo.utils.utils import is_call, is_assign, is_expr, get_getsetitem_index_var
from bodo.utils.transform import update_node_list_definitions, compile_func_single_block
from bodo.hiframes.pd_dataframe_ext import DataFrameType


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
        global in_partial_typing, typing_transform_required
        while True:
            try:
                # set global partial typing flag, see comment above
                in_partial_typing = True
                typing_transform_required = False
                super(BodoTypeInference, self).run_pass(state)
            finally:
                in_partial_typing = False

            # done if all types are available and transform not required
            if (
                types.unknown not in state.typemap.values()
                and not typing_transform_required
            ):
                break
            typing_transforms_pass = TypingTransforms(
                state.func_ir, state.typingctx, state.typemap, state.calltypes,
            )
            changed = typing_transforms_pass.run()
            # can't be typed if IR not changed
            if not changed:
                # error will be raised below if there are still unknown types
                break

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

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}
        # labels of rhs of assignments to enable finding nodes that create
        # dataframes such as Arg(df_type), pd.DataFrame(), df[['A','B']]...
        # the use is conservative and doesn't assume complete info
        self.rhs_labels = {}
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
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

        return self.changed

    def _run_assign(self, assign, label):
        rhs = assign.value

        if isinstance(rhs, ir.Expr) and rhs.op in ("binop", "inplace_binop"):
            return self._run_binop(assign, rhs)

        if is_call(rhs):
            return self._run_call(assign, rhs, label)

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
        arg1_typ = self.typemap[rhs.lhs.name]
        arg2_typ = self.typemap[rhs.rhs.name]
        target_typ = self.typemap[assign.target.name]

        # add of constant lists, e.g. ["A"] + ["B"]
        if (
            isinstance(arg1_typ, ConstList)
            and isinstance(arg2_typ, ConstList)
            and rhs.fn == operator.add
        ):
            vals = arg1_typ.consts + arg2_typ.consts
            return self._gen_consts_call(assign.target, vals)

        # sub of constant sets, e.g. {"A", "B"} - {"B"}
        if (
            isinstance(arg1_typ, ConstSet)
            and isinstance(arg2_typ, ConstSet)
            and rhs.fn == operator.sub
        ):
            # sort output of set diff op to be consistent across processors since const
            # values can be keys to groupby/sort/join which require consistent order
            vals = sorted(set(arg1_typ.consts) - set(arg2_typ.consts))
            return self._gen_consts_call(assign.target, vals, False)

        # replace ConstSet with Set if there a set op we don't support here
        # e.g. s1 | s2
        if isinstance(arg1_typ, ConstSet):  # pragma: no cover
            # TODO: test coverage
            arg_def = guard(get_definition, self.func_ir, rhs.lhs)
            assert is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "add_consts_to_type",
                "bodo.utils.typing",
            )
            rhs.lhs = arg_def.args[0]
            self.typemap.pop(assign.target.name)
            self.typemap[assign.target.name] = types.Set(target_typ.dtype)

        if isinstance(arg2_typ, ConstSet):  # pragma: no cover
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

        # df.drop("B", axis=1, inplace=True) changes dataframe type so inplace has to be
        # transformed away using variable replacement
        if func_name == "drop":
            kws = dict(rhs.kws)
            inplace_var = self._get_arg("drop", rhs.args, kws, 5, "inplace", "")
            replace_func = lambda: bodo.hiframes.dataframe_impl.drop_inplace  # pragma: no cover
            return self._handle_df_inplace_func(
                assign, lhs, rhs, df_var, inplace_var, replace_func, label
            )

        if func_name == "sort_values":
            # handle potential df.sort_values(inplace=True) here since it needs
            # variable replacement
            kws = dict(rhs.kws)
            inplace_var = self._get_arg("sort_values", rhs.args, kws, 3, "inplace", "")
            replace_func = lambda: bodo.hiframes.dataframe_impl.sort_values_inplace  # pragma: no cover
            return self._handle_df_inplace_func(
                assign, lhs, rhs, df_var, inplace_var, replace_func, label
            )
        return [assign]

    def _handle_df_inplace_func(
        self, assign, lhs, rhs, df_var, inplace_var, replace_func, label
    ):
        """handle possible df.func(inplace=True)
        lhs = A.func(inplace=True) -> A1, lhs = func_inplace(...)
        replace A with A1
        """
        inplace = guard(find_const, self.func_ir, inplace_var)
        if not inplace:
            return [assign]

        if self._label_dominates_var_defs(label, df_var):
            # TODO: make sure call post dominates df_var definition or df_var
            # is not used in other code paths
            # replace func variable with replace_func
            f_block = compile_to_numba_ir(
                replace_func, {"bodo": bodo}
            ).blocks.popitem()[1]
            nodes = f_block.body[:-2]
            new_func_var = nodes[-1].target
            rhs.func = new_func_var
            rhs.args.insert(0, df_var)
            # new tuple return
            ret_tup = ir.Var(lhs.scope, mk_unique_var("tuple_ret"), lhs.loc)
            assign.target = ret_tup
            nodes.append(assign)
            new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            zero_var = ir.Var(df_var.scope, mk_unique_var("zero"), df_var.loc)
            one_var = ir.Var(df_var.scope, mk_unique_var("one"), df_var.loc)
            nodes.append(ir.Assign(ir.Const(0, lhs.loc), zero_var, lhs.loc))
            nodes.append(ir.Assign(ir.Const(1, lhs.loc), one_var, lhs.loc))
            getitem0 = ir.Expr.static_getitem(ret_tup, 0, zero_var, lhs.loc)
            nodes.append(ir.Assign(getitem0, new_df_var, lhs.loc))
            getitem1 = ir.Expr.static_getitem(ret_tup, 1, one_var, lhs.loc)
            nodes.append(ir.Assign(getitem1, lhs, lhs.loc))
            # replace old variable with new one
            self.replace_var_dict[df_var.name] = new_df_var
            self.changed = True
            return nodes

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
        val_reps = ", ".join(["'{}'".format(c) for c in vals])
        in_brac = "[" if is_list else "set(["
        out_brac = "]" if is_list else "])"
        func_text = "def _build_f():\n"
        func_text += "  return bodo.utils.typing.add_consts_to_type({0}{1}{2}, {1})\n".format(
            in_brac, val_reps, out_brac
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        _build_f = loc_vars["_build_f"]
        nodes = compile_func_single_block(_build_f, (), target, self)
        self.typemap.pop(target.name)
        self.typemap[target.name] = self.typemap[nodes[-1].value.name]
        self.changed = True
        return nodes

    def _gen_const_string_index(self, target, rhs, vals):
        val_reps = ", ".join(["'{}'".format(c) for c in vals])
        const_call = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(val_reps)
        str_arr = "bodo.libs.str_arr_ext.str_arr_from_sequence({})".format(const_call)
        func_text = "def _gen_str_ind():\n"
        func_text += "  return bodo.hiframes.pd_index_ext.init_string_index({})\n".format(
            str_arr
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        _gen_str_ind = loc_vars["_gen_str_ind"]
        return compile_func_single_block(_gen_str_ind, (), target, self)

    def _get_const_seq(self, var):
        """returns constant values if 'var' represents a constant list/set. Otherwise,
        returns None
        """
        # constant list/set type
        if var.name in self.typemap and isinstance(
            self.typemap[var.name], (ConstList, ConstSet)
        ):
            return self.typemap[var.name].consts

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
                    return self.typemap[arg_def.args[0].name].consts

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):  # pragma: no cover
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "{} requires '{}' argument".format(f_name, arg_name)
            raise BodoError(err_msg)
        return arg

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
