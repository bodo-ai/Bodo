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
from numba.core.compiler import DefaultPassBuilder
from numba.core.compiler_machinery import (
    FunctionPass, AnalysisPass, register_pass, PassManager
)
from numba.core.untyped_passes import WithLifting

from numba.core.typed_passes import (
    NopythonTypeInference,
    PreParforPass,
    ParforPass,
    DumpParforDiagnostics,
)

from numba.core import ir_utils, ir, postproc
from numba.core.registry import CPUDispatcher
from numba.core.ir_utils import guard, get_definition
from numba.core.inline_closurecall import inline_closure_call, InlineClosureCallPass
from bodo import config
import bodo.libs
import bodo.libs.array_kernels  # side effect: install Numba functions
import bodo.libs.int_arr_ext  # side effect
import bodo.libs.re_ext  # side effect: initialize Numba extensions
import bodo.hiframes.datetime_timedelta_ext  # side effect: initialize Numba extensions
import bodo.hiframes.datetime_datetime_ext  # side effect: initialize Numba extensions
import bodo.hiframes.dataframe_indexing  # side effect: initialize Numba extensions
import bodo.utils
import bodo.utils.typing
import bodo.io
# required for _compile_for_args defined below (can be removed once the
# function is fixed in numba)
from numba.core import types, errors
from numba.core.utils import reraise
from numba.core.typing.typeof import Purpose, typeof

if config._has_h5py:
    from bodo.io import h5

# workaround for Numba #3876 issue with large labels in mortgage benchmark
from llvmlite import binding

binding.set_option("tmp", "-non-global-value-max-name-size=2048")


from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


# global flag for whether all Bodo functions should be inlined
inline_all_calls = False


class BodoCompiler(numba.core.compiler.CompilerBase):
    """Bodo compiler pipeline which adds the following passes to Numba's pipeline:
    InlinePass, BodoUntypedPass, BodoTypeInference, BodoDataFramePass, BodoSeriesPass,
    LowerParforSeq, BodoDumpDiagnosticsPass.
    See class docstrings for more info.
    """

    def define_pipelines(self):
        return self._create_bodo_pipeline(True)

    def _create_bodo_pipeline(self, distributed):
        name = "bodo" if distributed else "bodo_seq"
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state, name)

        # inline other jit functions right after IR is available
        # NOTE: calling after WithLifting since With blocks should be handled before
        # simplify_CFG() is called (block number is used in EnterWith nodes)
        if inline_all_calls:  # pragma: no cover
            pm.add_pass_after(InlinePass, WithLifting)
        # run untyped pass right before type inference
        add_pass_before(pm, BodoUntypedPass, NopythonTypeInference)
        # replace Numba's type inference pass with Bodo's version, which incorporates
        # constant inference using partial type inference
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
    """inline other jit functions, mainly to enable automatic parallelism
    """

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
    """
    Transformations before typing to enable type inference.
    This pass transforms the IR to remove operations that cannot be handled in Numba's
    type inference due to complexity such as pd.read_csv().
    """

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
    """
    This pass analyzes the IR to decide parallelism of arrays and parfors for
    distributed transformation, then parallelizes the IR for distributed execution and
    inserts MPI calls.
    Specialized IR nodes are also transformed to regular IR here since all analysis and
    transformations are done.
    """

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
    """
    This pass converts Series operations to array operations as much as possible to
    provide implementation and enable optimization.
    """

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
    """
    This pass converts data frame operations to Series and Array operations as much as
    possible to provide implementation and enable optimization. Creates specialized
    IR nodes for complex operations like Join.
    """

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
class BodoDumpDiagnosticsPass(AnalysisPass):
    """Print Bodo's distributed diagnostics info if needed
    """

    _name = "bodo_dump_diagnostics_pass"

    def __init__(self):
        AnalysisPass.__init__(self)

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
    """Lower parfors to regular loops to avoid threading of Numba
    """

    _name = "bodo_lower_parfor_seq_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        numba.parfors.parfor.lower_parfor_sequential(
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


# This replaces Numba's numba.core.dispatcher._DispatcherBase._compile_for_args
# method to delete args before returning the dispatcher object. Otherwise
# the code is the same
def _compile_for_args(self, *args, **kws):  # pragma: no cover
    """
    For internal use.  Compile a specialized version of the function
    for the given *args* and *kws*, and return the resulting callable.
    """
    assert not kws

    def error_rewrite(e, issue_type):
        """
        Rewrite and raise Exception `e` with help supplied based on the
        specified issue_type.
        """
        if numba.core.config.SHOW_HELP:
            help_msg = errors.error_extras[issue_type]
            e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
        if numba.core.config.FULL_TRACEBACKS:
            raise e
        else:
            reraise(type(e), e, None)

    argtypes = []
    for a in args:
        if isinstance(a, numba.core.dispatcher.OmittedArg):
            argtypes.append(types.Omitted(a.value))
        else:
            argtypes.append(self.typeof_pyval(a))
    try:
        return self.compile(tuple(argtypes))
    except errors.ForceLiteralArg as e:
        # Received request for compiler re-entry with the list of arguments
        # indicated by e.requested_args.
        # First, check if any of these args are already Literal-ized
        already_lit_pos = [i for i in e.requested_args
                           if isinstance(args[i], types.Literal)]
        if already_lit_pos:
            # Abort compilation if any argument is already a Literal.
            # Letting this continue will cause infinite compilation loop.
            m = ("Repeated literal typing request.\n"
                 "{}.\n"
                 "This is likely caused by an error in typing. "
                 "Please see nested and suppressed exceptions.")
            info = ', '.join('Arg #{} is {}'.format(i, args[i])
                             for i in  sorted(already_lit_pos))
            raise errors.CompilerError(m.format(info))
        # Convert requested arguments into a Literal.
        args = [(types.literal
                 if i in e.requested_args
                 else lambda x: x)(args[i])
                for i, v in enumerate(args)]
        # Re-enter compilation with the Literal-ized arguments
        return self._compile_for_args(*args)

    except errors.TypingError as e:
        # Intercept typing error that may be due to an argument
        # that failed inferencing as a Numba type
        failed_args = []
        for i, arg in enumerate(args):
            val = arg.value if isinstance(arg, numba.core.dispatcher.OmittedArg) else arg
            try:
                tp = typeof(val, Purpose.argument)
            except ValueError as typeof_exc:
                failed_args.append((i, str(typeof_exc)))
            else:
                if tp is None:
                    failed_args.append(
                        (i,
                         "cannot determine Numba type of value %r" % (val,)))
        if failed_args:
            # Patch error message to ease debugging
            msg = str(e).rstrip() + (
                "\n\nThis error may have been caused by the following argument(s):\n%s\n"
                % "\n".join("- argument %d: %s" % (i, err)
                            for i, err in failed_args))
            e.patch_message(msg)

        error_rewrite(e, 'typing')
    except errors.UnsupportedError as e:
        # Something unsupported is present in the user code, add help info
        error_rewrite(e, 'unsupported_error')
    except (errors.NotDefinedError, errors.RedefinedError,
            errors.VerificationError) as e:
        # These errors are probably from an issue with either the code supplied
        # being syntactically or otherwise invalid
        error_rewrite(e, 'interpreter')
    except errors.ConstantInferenceError as e:
        # this is from trying to infer something as constant when it isn't
        # or isn't supported as a constant
        error_rewrite(e, 'constant_inference')
    except Exception as e:
        if numba.core.config.SHOW_HELP:
            if hasattr(e, 'patch_message'):
                help_msg = errors.error_extras['reportable']
                e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
        # ignore the FULL_TRACEBACKS config, this needs reporting!
        raise e
    finally:
        # avoid issue of reference leak of arguments to jitted function:
        # https://github.com/numba/numba/issues/5419
        del args


# workaround for Numba #5419 issue (https://github.com/numba/numba/issues/5419)
# first we check that the hash of the Numba function that we are replacing
# matches the one of the function that we copied from Numba
import inspect
import hashlib

lines = inspect.getsource(numba.core.dispatcher._DispatcherBase._compile_for_args)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "1e12bb18f3ed09e608ba6c56a7fcd4cf2fe342b71af9d2e9767aee817d92f4b8"
):  # pragma: no cover
    warnings.warn(
        bodo.utils.typing.BodoWarning(
            "numba.core.dispatcher._DispatcherBase._compile_for_args has changed"
        )
    )
# now replace the function with our own
numba.core.dispatcher._DispatcherBase._compile_for_args = _compile_for_args
