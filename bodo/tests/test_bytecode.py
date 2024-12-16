"""Tests bytecode changes that are unique to Bodo."""

import numba
from numba.core import ir

from bodo.tests.utils import DeadcodeTestPipeline, check_func


def test_very_large_tuple(memory_leak_check):
    """
    Tests that when there is a very large tuple it only
    generates a single build_tuple in the IR.
    """
    func_text = "def impl(t):\n"
    func_text += "  return (\n"
    for i in range(200):
        func_text += f"    t[{i}],\n"
    func_text += "  )\n"

    local_vars = {}
    exec(func_text, {}, local_vars)
    impl = local_vars["impl"]
    t = tuple(range(100)) + tuple([str(i) for i in range(100)])
    check_func(impl, (t,))
    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    bodo_func(t)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    num_build_tuples = 0
    for block in fir.blocks.values():
        for stmt in block.body:
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "build_tuple"
            ):
                num_build_tuples += 1
    assert (
        num_build_tuples == 1
    ), "After DCE the IR should only contain a single build tuple"
