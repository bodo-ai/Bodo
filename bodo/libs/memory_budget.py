# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ memory_budget utilities"""

from enum import Enum

import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
from bodo.ext import memory_budget_cpp

ll.add_symbol("init_operator_comptroller", memory_budget_cpp.init_operator_comptroller)
ll.add_symbol("register_operator", memory_budget_cpp.register_operator)
ll.add_symbol(
    "compute_satisfiable_budgets", memory_budget_cpp.compute_satisfiable_budgets
)
ll.add_symbol(
    "delete_operator_comptroller", memory_budget_cpp.delete_operator_comptroller
)
ll.add_symbol("increment_pipeline_id", memory_budget_cpp.increment_pipeline_id)


class OperatorType(Enum):
    """All supported streaming operator types. The order here must match the order in _memory_budget.h::OperatorType"""

    UNKNOWN = 0
    SNOWFLAKE_WRITE = 1
    SNOWFLAKE_READ = 2
    JOIN = 3
    GROUPBY = 4
    UNION = 5
    ACCUMULATE_TABLE = 6
    ENCODE_DICT = 7


@intrinsic
def init_operator_comptroller(typingctx):
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
def register_operator(
    typingctx, operator_id, operator_type, min_pipeline_id, max_pipeline_id, estimate
):
    def codegen(context, builder, sig, args):
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
        return

    sig = types.none(
        operator_id, operator_type, min_pipeline_id, max_pipeline_id, estimate
    )
    return sig, codegen


@intrinsic
def increment_pipeline_id(typingctx):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="increment_pipeline_id"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none()
    return sig, codegen


@intrinsic
def compute_satisfiable_budgets(typingctx):
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
def delete_operator_comptroller(typingctx):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_operator_comptroller"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none()
    return sig, codegen
