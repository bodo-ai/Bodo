from numba import types, ir, ir_utils
from numba.compiler_machinery import register_pass
from numba.typed_passes import NopythonTypeInference, PartialTypeInference
from numba.ir_utils import find_topo_order, build_definitions, mk_unique_var
import bodo
from bodo.libs.str_ext import string_type
from bodo.utils.typing import BodoNotConstError, ConstList
from bodo.utils.utils import is_call_assign, is_assign, is_expr
from bodo.utils.transform import gen_add_consts_to_type, update_node_list_definitions


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoTypeInference(PartialTypeInference):
    _name = "bodo_type_inference"

    def run_pass(self, state):
        while True:
            super(BodoTypeInference, self).run_pass(state)
            fix_pass = FixConstsPass(
                state.func_ir, state.typingctx, state.typemap, state.calltypes,
            )
            fix_pass.run()
            self._raise_errors = True
            NopythonTypeInference.run_pass(self, state)
            self._raise_errors = False
            break
        return True


class FixConstsPass:
    """
    Transformations before typing to enable type inference.
    This pass transforms the IR to remove operations that cannot be handled in Numba's
    type inference due to complexity such as pd.read_csv().
    """

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc

    def run(self):
        blocks = self.func_ir.blocks
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            for inst in block.body:
                out_nodes = [inst]

                if isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst)

                new_body.extend(out_nodes)
                update_node_list_definitions(out_nodes, self.func_ir)

            blocks[label].body = new_body

        return

    def _run_assign(self, assign):
        rhs = assign.value
        if isinstance(rhs, ir.Expr) and rhs.op in ("binop", "inplace_binop"):
            return self._run_binop(assign, rhs)
        return [assign]

    def _run_binop(self, assign, rhs):
        arg1_typ = self.typemap[rhs.lhs.name]
        arg2_typ = self.typemap[rhs.rhs.name]
        if isinstance(arg1_typ, ConstList) and isinstance(arg2_typ, ConstList):
            vals = arg1_typ.consts + arg2_typ.consts
            target = assign.target
            tmp_target = ir.Var(target.scope, mk_unique_var(target.name), rhs.loc)
            self.typemap[tmp_target.name] = types.List(string_type)
            tmp_assign = ir.Assign(rhs, tmp_target, rhs.loc)
            nodes = [tmp_assign]
            nodes += gen_add_consts_to_type(vals, tmp_target, assign.target, self)
            self.typemap.pop(assign.target.name)
            # self.typemap[assign.target.name] = self.typemap[nodes[-1].value.name]
            return nodes
        return [assign]
