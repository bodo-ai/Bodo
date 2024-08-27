# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ memory_budget utilities"""

import llvmlite.binding as ll
import numba
from numba.core import types

import bodo
from bodo.ext import query_profile_collector_cpp
from bodo.transforms.typeinfer import ExternalFunctionErrorChecked

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


init = ExternalFunctionErrorChecked(
    "init_query_profile_collector_py_entry", types.none()
)


start_pipeline = ExternalFunctionErrorChecked(
    "start_pipeline_query_profile_collector_py_entry", types.none(types.int64)
)


end_pipeline = ExternalFunctionErrorChecked(
    "end_pipeline_query_profile_collector_py_entry",
    types.none(types.int64, types.int64),
)

submit_operator_stage_row_counts = ExternalFunctionErrorChecked(
    "submit_operator_stage_row_counts_query_profile_collector_py_entry",
    types.none(types.int64, types.int64, types.int64),
)


submit_operator_stage_time = ExternalFunctionErrorChecked(
    "submit_operator_stage_time_query_profile_collector_py_entry",
    types.none(types.int64, types.int64, types.float64),
)


get_operator_duration = ExternalFunctionErrorChecked(
    "get_operator_duration_query_profile_collector_py_entry", types.float64(types.int64)
)


_finalize = ExternalFunctionErrorChecked(
    "finalize_query_profile_collector_py_entry", types.none(types.int64)
)


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
