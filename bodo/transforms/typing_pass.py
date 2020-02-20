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
)
import bodo
from bodo.libs.str_ext import string_type
from bodo.utils.typing import ConstList, ConstSet
from bodo.utils.utils import is_call, is_assign, is_expr
from bodo.utils.transform import (
    gen_add_consts_to_type,
    update_node_list_definitions,
    compile_func_single_block,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType


# global flag indicating that we are in partial type inference, so that error checking
# code can raise regular Exceptions that can potentially be handled here
in_partial_typing = False


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoTypeInference(PartialTypeInference):
    _name = "bodo_type_inference"

    def run_pass(self, state):
        global in_partial_typing
        while True:
            try:
                in_partial_typing = True
                super(BodoTypeInference, self).run_pass(state)
            finally:
                in_partial_typing = False

            # done if all types are available
            if types.unknown not in state.typemap.values():
                break
            infer_consts_pass = InferConstsPass(
                state.func_ir, state.typingctx, state.typemap, state.calltypes,
            )
            changed = infer_consts_pass.run()
            # can't be typed if IR not changed
            if not changed:
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


class InferConstsPass:
    """
    Infer possible constant values (e.g. lists) using partial typing info and transform
    them to constants so that functions like groupby() can be typed properly.
    """

    # keeps the set of temporary variables generated in this pass to know when a
    # constant transformation is already done for an operation.
    # TODO: use proper way to know if lhs is going into add_const_to_type or not
    tmp_vars = set()

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        self.changed = False

    def run(self):
        blocks = self.func_ir.blocks
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            for inst in block.body:
                out_nodes = [inst]
                self.curr_loc = inst.loc

                if isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst)

                new_body.extend(out_nodes)
                update_node_list_definitions(out_nodes, self.func_ir)

            blocks[label].body = new_body

        return self.changed

    def _run_assign(self, assign):
        rhs = assign.value

        if isinstance(rhs, ir.Expr) and rhs.op in ("binop", "inplace_binop"):
            return self._run_binop(assign, rhs)

        if is_call(rhs):
            return self._run_call(assign, rhs)

        # replace df.columns with constant StringIndex
        if (
            is_expr(rhs, "getattr")
            and isinstance(self.typemap[rhs.value.name], DataFrameType)
            and rhs.attr == "columns"
            and assign.target.name not in self.tmp_vars
        ):
            vals = self.typemap[rhs.value.name].columns
            return self._gen_const_string_index(assign.target, rhs, vals)

        # replace ConstSet with Set since we don't support 'set' methods yet
        # e.g. s.add()
        if is_expr(rhs, "getattr") and isinstance(
            self.typemap[rhs.value.name], ConstSet
        ):
            val_typ = self.typemap[rhs.value.name]
            self.typemap.pop(rhs.value.name)
            self.typemap[rhs.value.name] = types.Set(val_typ.dtype)

        return [assign]

    def _run_call(self, assign, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            return [assign]

        # set() with constant list arg
        if (
            fdef == ("set", "builtins")
            and len(rhs.args) == 1
            and self._get_const_seq(rhs.args[0]) is not None
            and assign.target.name not in self.tmp_vars
        ):
            vals = self._get_const_seq(rhs.args[0])
            return self._gen_consts_call(assign.target, vals, False)

        # list() with constant list/set arg
        if (
            fdef == ("list", "builtins")
            and len(rhs.args) == 1
            and self._get_const_seq(rhs.args[0]) is not None
            and assign.target.name not in self.tmp_vars
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
            and assign.target.name not in self.tmp_vars
        ):
            vals = arg1_typ.consts + arg2_typ.consts
            return self._gen_consts_call(assign.target, vals)

        # sub of constant sets, e.g. {"A", "B"} - {"B"}
        if (
            isinstance(arg1_typ, ConstSet)
            and isinstance(arg2_typ, ConstSet)
            and rhs.fn == operator.sub
            and assign.target.name not in self.tmp_vars
        ):
            vals = set(arg1_typ.consts) - set(arg2_typ.consts)
            print(vals)
            return self._gen_consts_call(assign.target, vals, False)

        # replace ConstSet with Set if there a set op we don't support here
        # e.g. s1 | s2
        if isinstance(arg1_typ, ConstSet):
            arg_def = guard(get_definition, self.func_ir, rhs.lhs)
            assert is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "add_consts_to_type",
                "bodo.utils.typing",
            )
            rhs.lhs = arg_def.args[0]
            self.typemap.pop(assign.target.name)
            self.typemap[assign.target.name] = types.Set(target_typ.dtype)

        if isinstance(arg2_typ, ConstSet):
            arg_def = guard(get_definition, self.func_ir, rhs.rhs)
            assert is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "add_consts_to_type",
                "bodo.utils.typing",
            )
            rhs.rhs = arg_def.args[0]
            self.typemap.pop(assign.target.name)
            self.typemap[assign.target.name] = types.Set(target_typ.dtype)

        return [assign]

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
        if isinstance(self.typemap[var.name], (ConstList, ConstSet)):
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
                if isinstance(self.typemap[arg_def.args[0].name], ConstList):
                    return self.typemap[arg_def.args[0].name].consts
