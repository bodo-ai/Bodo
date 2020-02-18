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
from bodo.transforms.typing_pass import BodoTypeInference
import numba
from numba.compiler import DefaultPassBuilder
from numba.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.untyped_passes import WithLifting

from numba.typed_passes import (
    NopythonTypeInference,
    PreParforPass,
    ParforPass,
    DumpParforDiagnostics,
)

from numba import ir_utils, ir, postproc
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import guard, get_definition
from numba.inline_closurecall import inline_closure_call, InlineClosureCallPass
from bodo import config
import bodo.libs
import bodo.libs.array_kernels  # side effect: install Numba functions
import bodo.libs.int_arr_ext  # side effect
import bodo.libs.re_ext  # side effect: initialize Numba extensions
import bodo.hiframes.datetime_timedelta_ext  # side effect: initialize Numba extensions
import bodo.hiframes.datetime_datetime_ext  # side effect: initialize Numba extensions
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
        name = "bodo" if distributed else "bodo_seq"
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state, name)

        # inline other jit functions right after IR is available
        # NOTE: calling after WithLifting since With blocks should be handled before
        # simplify_CFG() is called (block number is used in EnterWith nodes)
        pm.add_pass_after(InlinePass, WithLifting)
        # run untyped pass right before type inference
        add_pass_before(pm, BodoUntypedPass, NopythonTypeInference)
        replace_pass(pm, BodoTypeInference, NopythonTypeInference)

        # Series pass should be before pre_parfor since
        # S.call to np.call transformation is invalid for
        # Series (e.g. S.var is not the same as np.var(S))
        add_pass_before(pm, BodoDataFramePass, PreParforPass)
        pm.add_pass_after(BodoSeriesPass, BodoDataFramePass)

        if distributed:
            pm.add_pass_after(BodoDistributedPass, ParforPass)
        else:
            pm.add_pass_after(LowerParforSeq, ParforPass)

        pm.add_pass_after(BodoDumpDiagnosticsPass, DumpParforDiagnostics)
        pm.finalize()
        return [pm]


# TODO: remove this helper function when available in Numba
def add_pass_before(pm, pass_cls, location):
    """
    Add a pass `pass_cls` to the PassManager's compilation pipeline right before
    the pass `location`.
    """
    # same as add_pass_after, except first argument to "insert"
    assert pm.passes
    pm._validate_pass(pass_cls)
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise ValueError("Could not find pass %s" % location)
    pm.passes.insert(idx, (pass_cls, str(pass_cls)))
    # if a pass has been added, it's not finalized
    pm._finalized = False


def replace_pass(pm, pass_cls, location):
    """
    Replace pass `location` in PassManager's compilation pipeline with the pass
    `pass_cls`.
    """
    assert pm.passes
    pm._validate_pass(pass_cls)
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise ValueError("Could not find pass %s" % location)
    pm.passes[idx] = (pass_cls, str(pass_cls))
    # if a pass has been added, it's not finalized
    pm._finalized = False


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
            state.func_ir,
            state.typingctx,
            state.args,
            state.locals,
            state.metadata,
            state.flags,
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
            state.flags,
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
                        _, var_dict = inline_closure_call(
                            func_ir,
                            py_func.__globals__,
                            block,
                            i,
                            py_func,
                            work_list=work_list,
                        )

                        _locals.update(
                            (var_dict[k].name, v)
                            for k, v in func_def.value.locals.items()
                            if k in var_dict
                        )
                        # TODO: support options like "distributed" if applied to the
                        # inlined function

                        # for block in new_blocks:
                        #     work_list.append(block)
                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break

    # sometimes type inference fails after inlining since blocks are inserted
    # at the end and there are agg constraints (categorical_split case)
    # CFG simplification fixes this case
    func_ir.blocks = ir_utils.simplify_CFG(func_ir.blocks)
