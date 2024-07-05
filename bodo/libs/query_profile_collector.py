# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ memory_budget utilities"""

import llvmlite.binding as ll
import numba
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
    "submit_operator_stage_time_query_profile_collector_py_entry",
    query_profile_collector_cpp.submit_operator_stage_time_query_profile_collector_py_entry,
)
ll.add_symbol(
    "get_operator_duration_query_profile_collector_py_entry",
    query_profile_collector_cpp.get_operator_duration_query_profile_collector_py_entry,
)
ll.add_symbol(
    "finalize_query_profile_collector_py_entry",
    query_profile_collector_cpp.finalize_query_profile_collector_py_entry,
)
ll.add_symbol(
    "get_output_row_counts_for_op_stage_py_entry",
    query_profile_collector_cpp.get_output_row_counts_for_op_stage_py_entry,
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
def submit_operator_stage_row_counts(
    typingctx, operator_id, pipeline_id, output_row_count
):
    """Wrapper for submit_operator_stage_row_counts in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
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

    sig = types.none(operator_id, pipeline_id, output_row_count)
    return sig, codegen


@intrinsic(prefer_literal=True)
def submit_operator_stage_time(typingctx, operator_id, stage_id, time):
    """Wrapper for submit_operator_stage_time_query_profile_collector_py_entry in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.DoubleType(),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="submit_operator_stage_time_query_profile_collector_py_entry",
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none(operator_id, stage_id, time)
    return sig, codegen


@intrinsic(prefer_literal=True)
def get_operator_duration(typingctx, operator_id):
    """Wrapper for get_operator_duration_query_profile_collector_py_entry in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(64)])
        fn_typ = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="get_operator_duration_query_profile_collector_py_entry",
        )
        ret = builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.float64(operator_id)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _finalize(typingctx, verbose_level):
    """Wrapper for finalize in _query_profile_collector.cpp"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(64)])
        fn_typ = cgutils.get_or_insert_function(
            builder.module, fnty, name="finalize_query_profile_collector_py_entry"
        )
        builder.call(fn_typ, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.get_dummy_value()

    sig = types.none(verbose_level)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True, no_unliteral=True)
def finalize():
    """Wrapper for finalize in _query_profile_collector.cpp"""

    def impl():  # pragma: no cover
        verbose_level = bodo.user_logging.get_verbose_level()
        _finalize(verbose_level)

    return impl


## Only used for unit testing purposes

get_output_row_counts_for_op_stage_f = types.ExternalFunction(
    "get_output_row_counts_for_op_stage_py_entry",
    types.int64(types.int64, types.int64),
)


@numba.njit
def get_output_row_counts_for_op_stage(op_id, stage_id):  # pragma: no cover
    return get_output_row_counts_for_op_stage_f(op_id, stage_id)
