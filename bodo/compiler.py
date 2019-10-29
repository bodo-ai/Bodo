# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Defines Bodo's compiler pipeline.
"""
import os
import warnings
import bodo
import bodo.transforms
import bodo.transforms.untyped_pass
import bodo.transforms.series_pass
from bodo.transforms.untyped_pass import UntypedPass
from bodo.transforms.series_pass import SeriesPass
from bodo.transforms.dataframe_pass import DataFramePass
import numba
import numba.compiler
from numba.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.untyped_passes import (
    ExtractByteCode,
    TranslateByteCode,
    FixupArgs,
    IRProcessing,
    DeadBranchPrune,
    RewriteSemanticConstants,
    InlineClosureLikes,
    GenericRewrites,
    WithLifting,
    InlineInlinables,
)

from numba.typed_passes import (
    NopythonTypeInference,
    AnnotateTypes,
    NopythonRewrites,
    PreParforPass,
    ParforPass,
    DumpParforDiagnostics,
    IRLegalization,
    NoPythonBackend,
    InlineOverloads,
)

from numba import ir_utils, ir, postproc
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import guard, get_definition
from numba.inline_closurecall import inline_closure_call, InlineClosureCallPass
from bodo import config
import bodo.libs
import bodo.libs.array_kernels  # side effect: install Numba functions
import bodo.libs.int_arr_ext  # side effect
import bodo.utils
import bodo.utils.typing
import bodo.io

if config._has_h5py:
    from bodo.io import h5

# workaround for Numba #3876 issue with large labels in mortgage benchmark
from llvmlite import binding

binding.set_option("tmp", "-non-global-value-max-name-size=2048")


from numba.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


class BodoCompiler(numba.compiler.CompilerBase):
    """Bodo compiler pipeline
    """

    def define_pipelines(self):
        return self._create_bodo_pipeline(True)

    def _create_bodo_pipeline(self, distributed):
        pm = PassManager("bodo")

        if self.state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")

        pm.add_pass(WithLifting, "Handle with contexts")

        # pre typing
        if not self.state.flags.no_rewrites:
            pm.add_pass(GenericRewrites, "nopython rewrites")
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
        pm.add_pass(InlineClosureLikes, "inline calls to locally defined closures")

        # inline functions that have been determined as inlinable and rerun
        # branch pruning this needs to be run after closures are inlined as
        # the IR repr of a closure masks call sites if an inlinable is called
        # inside a closure
        pm.add_pass(InlineInlinables, "inline inlinable functions")
        if not self.state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(InlinePass, "inline funcs")
        pm.add_pass(BodoUntypedPass, "untyped pass")

        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")

        # Series pass should be before pre_parfor since
        # S.call to np.call transformation is invalid for
        # Series (e.g. S.var is not the same as np.var(S))
        pm.add_pass(BodoDataFramePass, "typed dataframe pass")
        pm.add_pass(BodoSeriesPass, "typed series pass")

        if self.state.flags.auto_parallel.enabled:
            pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not self.state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        if self.state.flags.auto_parallel.enabled:
            pm.add_pass(ParforPass, "convert to parfors")

        if distributed:
            pm.add_pass(BodoDistributedPass, "convert to distributed")
        else:
            pm.add_pass(LowerParforSeq, "lower parfor sequentially")

        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")
        pm.add_pass(BodoDumpDiagnosticsPass, "dump distributed diagnostics")
        pm.finalize()
        return [pm]


# TODO: use Numba's new inline feature
@register_pass(mutates_CFG=True, analysis_only=False)
class InlinePass(FunctionPass):

    _name = "inline_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Inline function calls (to enable distributed pass analysis)
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        inline_calls(state.func_ir, state.locals)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoUntypedPass(FunctionPass):

    _name = "bodo_untyped_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Fix IR before typing to handle untypeable cases
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        untyped_pass = UntypedPass(
            state.func_ir, state.typingctx, state.args, state.locals, state.metadata
        )
        untyped_pass.run()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoDistributedPass(FunctionPass):

    _name = "bodo_distributed_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        parallelize for distributed-memory
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        from bodo.transforms.distributed_pass import DistributedPass

        dist_pass = DistributedPass(
            state.func_ir,
            state.typingctx,
            state.targetctx,
            state.type_annotation.typemap,
            state.type_annotation.calltypes,
            state.metadata,
        )
        dist_pass.run()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoSeriesPass(FunctionPass):

    _name = "bodo_series_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert Series after typing
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        series_pass = SeriesPass(
            state.func_ir,
            state.typingctx,
            state.type_annotation.typemap,
            state.type_annotation.calltypes,
        )
        series_pass.run()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoDataFramePass(FunctionPass):

    _name = "bodo_dataframe_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert DataFrames after typing
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        df_pass = DataFramePass(
            state.func_ir,
            state.typingctx,
            state.type_annotation.typemap,
            state.type_annotation.calltypes,
        )
        df_pass.run()
        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class BodoDumpDiagnosticsPass(FunctionPass):

    _name = "bodo_dump_diagnostics_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Print distributed diagnostics information if environment variable is
        set.
        """
        diag_level = 0
        env_name = "BODO_DISTRIBUTED_DIAGNOSTICS"
        try:
            diag_level = int(os.environ[env_name])
        except:
            pass

        if diag_level > 0:
            state.metadata["distributed_diagnostics"].dump(diag_level)
        return True


class BodoCompilerSeq(BodoCompiler):
    """Bodo pipeline without the distributed pass (used in rolling kernels)
    """

    def define_pipelines(self):
        return self._create_bodo_pipeline(False)


@register_pass(mutates_CFG=False, analysis_only=True)
class LowerParforSeq(FunctionPass):

    _name = "bodo_lower_parfor_seq_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        numba.parfor.lower_parfor_sequential(
            state.typingctx, state.func_ir, state.typemap, state.calltypes
        )
        return True


def inline_calls(func_ir, _locals):
    """Inlines all decorated functions. Use Numba's #3743 when merged.
    """
    work_list = list(func_ir.blocks.items())
    while work_list:
        _label, block = work_list.pop()
        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Assign):
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    func_def = guard(get_definition, func_ir, expr.func)
                    if isinstance(func_def, (ir.Global, ir.FreeVar)) and isinstance(
                        func_def.value, CPUDispatcher
                    ):
                        py_func = func_def.value.py_func
                        inline_out = inline_closure_call(
                            func_ir,
                            py_func.__globals__,
                            block,
                            i,
                            py_func,
                            work_list=work_list,
                        )

                        # TODO remove if when inline_closure_call() output fix
                        # is merged in Numba
                        if isinstance(inline_out, tuple):
                            var_dict = inline_out[1]
                            # TODO: update '##distributed' and '##threaded'
                            # in _locals
                            _locals.update(
                                (var_dict[k].name, v)
                                for k, v in func_def.value.locals.items()
                                if k in var_dict
                            )
                        # for block in new_blocks:
                        #     work_list.append(block)
                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break

    # sometimes type inference fails after inlining since blocks are inserted
    # at the end and there are agg constraints (categorical_split case)
    # CFG simplification fixes this case
    func_ir.blocks = ir_utils.simplify_CFG(func_ir.blocks)
