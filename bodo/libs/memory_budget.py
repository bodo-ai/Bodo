"""Interface to C++ memory_budget utilities"""

from enum import Enum

import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, lower_builtin

import bodo
from bodo.ext import memory_budget_cpp

ll.add_symbol("init_operator_comptroller", memory_budget_cpp.init_operator_comptroller)
ll.add_symbol(
    "init_operator_comptroller_with_budget",
    memory_budget_cpp.init_operator_comptroller_with_budget,
)
ll.add_symbol("register_operator", memory_budget_cpp.register_operator)
ll.add_symbol(
    "compute_satisfiable_budgets", memory_budget_cpp.compute_satisfiable_budgets
)

## Only used for unit testing purposes
ll.add_symbol("reduce_operator_budget", memory_budget_cpp.reduce_operator_budget)
ll.add_symbol("increase_operator_budget", memory_budget_cpp.increase_operator_budget)


class OperatorType(Enum):
    """All supported streaming operator types. The order here must match the order in _memory_budget.h::OperatorType"""

    UNKNOWN = 0
    SNOWFLAKE_WRITE = 1
    JOIN = 2
    GROUPBY = 3
    UNION = 4
    WINDOW = 5
    ACCUMULATE_TABLE = 6
    SORT = 7
    GROUPING_SETS = 8


@intrinsic
def init_operator_comptroller(typingctx):
    """Wrapper for init_operator_comptroller in _memory_budget.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_operator_comptroller"
        )
        builder.call(fn_typ, ())
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none()
    return sig, codegen


@intrinsic
def init_operator_comptroller_with_budget(typingctx, budget):
    """Wrapper for init_operator_comptroller_with_budget in _memory_budget.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_operator_comptroller_with_budget"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none(budget)
    return sig, codegen


def register_operator(
    operator_id, operator_type, min_pipeline_id, max_pipeline_id, estimate
):
    pass


@lower_builtin(
    register_operator,
    types.int64,
    types.EnumMember(OperatorType, types.int64),
    types.int64,
    types.int64,
    types.int64,
)
def lower_register_operator(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(64),
        ],
    )
    fn_typ = cgutils.get_or_insert_function(
        builder.module, fnty, name="register_operator"
    )
    builder.call(fn_typ, args)
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)


@intrinsic
def compute_satisfiable_budgets(typingctx):
    """Wrapper for compute_satisfiable_budgets in _memory_budget.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="compute_satisfiable_budgets"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none()
    return sig, codegen


@intrinsic
def reduce_operator_budget(typingctx, operator_id, new_estimate):
    """
    Wrapper for reduce_operator_budget in _memory_budget.cpp
    Only used for unit testing purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="reduce_operator_budget"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none(operator_id, new_estimate)
    return sig, codegen


@intrinsic
def increase_operator_budget(typingctx, operator_id):
    """
    Wrapper for increase_operator_budget in _memory_budget.cpp
    Only used for unit testing purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(64)])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="increase_operator_budget"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none(operator_id)
    return sig, codegen
