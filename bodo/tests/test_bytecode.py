"""Tests bytecode changes that are unique to Bodo."""

import numba  # noqa TID253
from numba.core import ir  # noqa TID253

from bodo.tests.utils import check_func, pytest_mark_one_rank


def test_very_large_tuple(memory_leak_check):
    """
    Tests that when there is a very large tuple it only
    generates a single build_tuple in the IR.
    """
    from bodo.tests.utils_jit import DeadcodeTestPipeline

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
    assert num_build_tuples == 1, (
        "After DCE the IR should only contain a single build tuple"
    )


def get_bc_stream(code):
    import dis

    for inst in dis.get_instructions(code):
        yield inst.offset, inst.opcode, inst.arg


@pytest_mark_one_rank
def test_bc_stream_to_bytecode():
    """
    Tests that _bc_stream_bytecode correctly reverses disassembled bytecode.
    """
    from bodo.transforms.typing_pass import _bc_stream_to_bytecode

    test_code = _bc_stream_to_bytecode.__code__
    assert (
        _bc_stream_to_bytecode(get_bc_stream(test_code), test_code) == test_code.co_code
    )


@pytest_mark_one_rank
def test_bc_stream_to_bytecode_all_typing_pass():
    """
    Tests that _bc_stream_bytecode correctly reverses disassembled bytecode for all typing pass functions.
    """
    import bodo.decorators  # isort:skip noqa
    import bodo.transforms.typing_pass
    from bodo.transforms.typing_pass import _bc_stream_to_bytecode

    for obj in vars(bodo.transforms.typing_pass).values():
        if callable(obj) and hasattr(obj, "__code__"):
            test_code = obj.__code__
            assert (
                _bc_stream_to_bytecode(get_bc_stream(test_code), test_code)
                == test_code.co_code
            )
