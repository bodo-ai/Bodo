from difflib import unified_diff

import pytest
from numba.core import ir

import bodo

# TODO[BSE-5071]: Re-enable native typer when its coverage improved
pytestmark = [pytest.mark.compiler, pytest.mark.skip]


def normalize_ir(fir: ir.FunctionIR):
    """Normalize IR to remove inconsistencies for string comparison"""

    # All call arguments should be a list of Vars, not a tuple
    for block in fir.blocks.values():
        for expr in block.find_exprs("call"):
            expr.args = list(expr.args)


def compare_ir_str(func):
    py_ir = bodo.numba_compat.run_frontend(func)
    normalize_ir(py_ir)
    py_str = py_ir.dump_to_string()
    native_str = bodo.transforms.type_inference.native_typer.ir_native_to_string(py_ir)
    if py_str != native_str:
        diff = "\n".join(
            unified_diff(py_str.splitlines(), native_str.splitlines(), lineterm="")
        )
        raise AssertionError(f"IR strings are different:\n{diff}")


def test_rhs():
    GLOBAL = 1

    def impl(a):
        # Constant
        b = 1  # noqa: F841
        # Variable & arg
        c = a  # noqa: F841
        # Global
        d = GLOBAL  # noqa: F841
        # TODO: Freevar

    compare_ir_str(impl)


def test_simple_exprs():
    """Test simple expressions"""

    def impl(a):
        # BinOpExpr
        c = a + a
        # UnaryOpExpr
        d = not c
        # InPlaceBinOpExpr
        c += d
        # CallExpr
        b = bool(a)
        # BuildTupleExpr
        e = (a, b)  # noqa: F841
        # BuildListExpr
        f = [a, b]  # noqa: F841
        # TODO: PairFirst and PairSecond

    compare_ir_str(impl)


def test_container_exprs():
    """Test expressions related to containers"""

    # L is a list
    def impl(l, a):
        # GetIterExpr
        it = iter(l)
        # IterNextExpr
        out = next(it)  # noqa: F841
        # TODO: ExhaustIterExpr
        # GetAttrExpr
        b = l.append  # noqa: F841
        # GetItemExpr
        c = l[a]  # noqa: F841
        # TODO: TypedGetItemExpr
        # StaticGetItemExpr
        d = l[0]  # noqa: F841
        e = l[:2]  # noqa: F841
        # TODO: CastExpr?

    compare_ir_str(impl)


def test_control_flow():
    """Test control flow for branching, termination, and phi nodes"""

    def impl(a):
        # branch and raise
        if a:
            b = 2
        else:
            raise ValueError(f"Failed {a}")

        # jump from looping and phi
        while b:
            a *= 2
            b -= 1

        # static raise
        if a > 100:
            raise ValueError()

    compare_ir_str(impl)


def test_other_stmts():
    """Test other statements missed in previous tests"""

    def impl(l, a, obj):
        # SetItem
        l[a] = 1
        # StaticSetItem
        l[0] = 2
        # SetAttr
        obj.content = "hello"
        # Del
        del a
        # Print
        print(True)
        # EnterWith
        with obj:
            print("inside")
        # TODO: PopBlock and Parfor

    compare_ir_str(impl)
