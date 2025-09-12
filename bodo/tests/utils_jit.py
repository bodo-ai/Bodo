"""
Utilities for testing JIT. This file cannot be imported at the top level.
"""

import numba
import numpy as np
import pandas as pd

# Import compiler
import bodo.decorators  # isort:skip # noqa
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import NopythonRewrites
from numba.core.untyped_passes import PreserveIR


@numba.njit
def get_rank():
    return bodo.libs.distributed_api.get_rank()


@numba.njit(cache=True)
def get_start_end(n):
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(n, n_pes, rank)
    return start, end


@numba.njit
def reduce_sum(val):
    sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
    return bodo.libs.distributed_api.dist_reduce(val, np.int32(sum_op))


class DeadcodeTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_join_deadcode_cleanup and test_csv_remove_col0_used_for_len
    with an additional PreserveIR pass then bodo_pipeline
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, NopythonRewrites)
        pipeline.finalize()
        return [pipeline]


class SeriesOptTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_series_apply_df_output with an additional PreserveIR pass
    after SeriesPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIRTypeMap, bodo.compiler.BodoSeriesPass)
        pipeline.finalize()
        return [pipeline]


class ParforTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_parfor_optimizations with an additional PreserveIR pass
    after ParforPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.ParforPreLoweringPass)
        pipeline.finalize()
        return [pipeline]


class ColumnDelTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_column_del_pass with an additional PreserveIRTypeMap pass
    after BodoTableColumnDelPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIRTypeMap, bodo.compiler.BodoTableColumnDelPass)
        pipeline.finalize()
        return [pipeline]


@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIRTypeMap(PreserveIR):
    """
    Extension to PreserveIR that also saves the typemap.
    """

    _name = "preserve_ir_typemap"

    def __init__(self):
        PreserveIR.__init__(self)

    def run_pass(self, state):
        PreserveIR.run_pass(self, state)
        state.metadata["preserved_typemap"] = state.typemap.copy()
        state.metadata["preserved_calltypes"] = state.calltypes.copy()
        return False


class TypeInferenceTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in bodosql tests with an additional PreserveIR pass
    after BodoTypeInference. This is used to monitor the code being generated.
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.BodoTypeInference)
        pipeline.finalize()
        return [pipeline]


class DistTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline with an additional PreserveIR pass
    after DistributedPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.BodoDistributedPass)
        pipeline.finalize()
        return [pipeline]


class SeqTestPipeline(bodo.compiler.BodoCompiler):
    """
    Bodo sequential pipeline with an additional PreserveIR pass
    after LowerBodoIRExtSeq
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=False, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.LowerBodoIRExtSeq)
        pipeline.finalize()
        return [pipeline]


@register_pass(analysis_only=False, mutates_CFG=True)
class ArrayAnalysisPass(FunctionPass):
    _name = "array_analysis_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        array_analysis = numba.parfors.array_analysis.ArrayAnalysis(
            state.typingctx,
            state.func_ir,
            state.typemap,
            state.calltypes,
        )
        array_analysis.run(state.func_ir.blocks)
        state.func_ir._definitions = numba.core.ir_utils.build_definitions(
            state.func_ir.blocks
        )
        state.metadata["preserved_array_analysis"] = array_analysis
        return False


class AnalysisTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_dataframe_array_analysis()
    additional ArrayAnalysis pass that preserves analysis object
    """

    # Avoid copy propagation so we don't delete variables used to
    # check array analysis.
    avoid_copy_propagation = True

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(ArrayAnalysisPass, bodo.compiler.BodoSeriesPass)
        pipeline.finalize()
        return [pipeline]


@bodo.jit(distributed=False)
def _nullable_float_arr_maker(L, to_null, to_nan):
    n = len(L)
    data_arr = np.empty(n, np.float64)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    A = bodo.libs.float_arr_ext.init_float_array(data_arr, nulls)
    for i in range(len(L)):
        if i in to_null:
            bodo.libs.array_kernels.setna(A, i)
        elif i in to_nan:
            A[i] = np.nan
        else:
            A[i] = L[i]
    return pd.Series(A)


# simple UDF dependency for test_udf_other_module
@bodo.jit
def udf_dep(n):
    return np.arange(n).sum()
