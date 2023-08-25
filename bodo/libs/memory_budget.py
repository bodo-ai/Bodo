# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ memory_budget utilities"""

import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
from bodo.ext import memory_budget_cpp

ll.add_symbol("init_operator_comptroller", memory_budget_cpp.init_operator_comptroller)
ll.add_symbol("register_operator", memory_budget_cpp.register_operator)
ll.add_symbol(
    "delete_operator_comptroller", memory_budget_cpp.delete_operator_comptroller
)
ll.add_symbol("increment_pipeline_id", memory_budget_cpp.increment_pipeline_id)


@intrinsic
def init_operator_comptroller(typingctx):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_operator_comptroller"
        )
        ret = builder.call(fn_typ, ())
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.void()
    return sig, codegen


@intrinsic
def register_operator(
    typingctx, operator_id, min_pipeline_id, max_pipeline_id, estimate
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [lir.IntType(64), lir.IntType(64), lir.IntType(64), lir.IntType(64)],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="register_operator"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none(operator_id, min_pipeline_id, max_pipeline_id, estimate)
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
