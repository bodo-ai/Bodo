# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ memory_budget utilities"""


import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
from bodo.ext import query_profile_collector_cpp

ll.add_symbol(
    "init_query_profile_collector_py_entry",
    query_profile_collector_cpp.init_query_profile_collector_py_entry,
)
ll.add_symbol(
    "start_pipeline_query_profile_collector_py_entry",
    query_profile_collector_cpp.start_pipeline_query_profile_collector_py_entry,
)
ll.add_symbol(
    "end_pipeline_query_profile_collector_py_entry",
    query_profile_collector_cpp.end_pipeline_query_profile_collector_py_entry,
)
ll.add_symbol(
    "submit_operator_stage_row_counts_query_profile_collector_py_entry",
    query_profile_collector_cpp.submit_operator_stage_row_counts_query_profile_collector_py_entry,
)
ll.add_symbol(
    "finalize_query_profile_collector_py_entry",
    query_profile_collector_cpp.finalize_query_profile_collector_py_entry,
)


@intrinsic(prefer_literal=True)
def init(typingctx):
    """Wrapper for init_py_entry in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_query_profile_collector_py_entry"
        )
        builder.call(fn_typ, ())
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none()
    return sig, codegen


@intrinsic(prefer_literal=True)
def start_pipeline(typingctx, pipeline_id):
    """Wrapper for start_pipeline in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="start_pipeline_query_profile_collector_py_entry"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none(pipeline_id)
    return sig, codegen


@intrinsic(prefer_literal=True)
def end_pipeline(typingctx, pipeline_id, num_iterations):
    """Wrapper for end_pipeline in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="end_pipeline_query_profile_collector_py_entry"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none(pipeline_id, num_iterations)
    return sig, codegen


@intrinsic(prefer_literal=True)
def submit_operator_stage_row_counts(typingctx):
    """Wrapper for submit_operator_stage_row_counts in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="submit_operator_stage_row_counts_query_profile_collector_py_entry",
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none()
    return sig, codegen


@intrinsic(prefer_literal=True)
def finalize(typingctx):
    """Wrapper for finalize in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="finalize_query_profile_collector_py_entry"
        )
        builder.call(fn_typ, ())
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none()
    return sig, codegen