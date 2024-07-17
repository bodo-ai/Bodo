# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Defines decorators of Bodo. Currently just @jit.
"""
import hashlib
import inspect
import warnings

import numba
from numba.core import cpu
from numba.core.options import _mapping
from numba.core.targetconfig import Option, TargetConfig

import bodo
from bodo import master_mode

# Add Bodo's options to Numba's allowed options/flags
numba.core.cpu.CPUTargetOptions.all_args_distributed_block = _mapping(
    "all_args_distributed_block"
)
numba.core.cpu.CPUTargetOptions.all_args_distributed_varlength = _mapping(
    "all_args_distributed_varlength"
)
numba.core.cpu.CPUTargetOptions.all_returns_distributed = _mapping(
    "all_returns_distributed"
)
numba.core.cpu.CPUTargetOptions.returns_maybe_distributed = _mapping(
    "returns_maybe_distributed"
)
numba.core.cpu.CPUTargetOptions.args_maybe_distributed = _mapping(
    "args_maybe_distributed"
)
numba.core.cpu.CPUTargetOptions.distributed = _mapping("distributed")
numba.core.cpu.CPUTargetOptions.distributed_block = _mapping("distributed_block")
numba.core.cpu.CPUTargetOptions.replicated = _mapping("replicated")
numba.core.cpu.CPUTargetOptions.threaded = _mapping("threaded")
numba.core.cpu.CPUTargetOptions.pivots = _mapping("pivots")
numba.core.cpu.CPUTargetOptions.h5_types = _mapping("h5_types")


class Flags(TargetConfig):
    enable_looplift = Option(
        type=bool,
        default=False,
        doc="Enable loop-lifting",
    )
    enable_pyobject = Option(
        type=bool,
        default=False,
        doc="Enable pyobject mode (in general)",
    )
    enable_pyobject_looplift = Option(
        type=bool,
        default=False,
        doc="Enable pyobject mode inside lifted loops",
    )
    enable_ssa = Option(
        type=bool,
        default=True,
        doc="Enable SSA",
    )
    force_pyobject = Option(
        type=bool,
        default=False,
        doc="Force pyobject mode inside the whole function",
    )
    release_gil = Option(
        type=bool,
        default=False,
        doc="Release GIL inside the native function",
    )
    no_compile = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    debuginfo = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    boundscheck = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    forceinline = Option(
        type=bool,
        default=False,
        doc="Force inlining of the function. Overrides _dbg_optnone.",
    )
    no_cpython_wrapper = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    no_cfunc_wrapper = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    auto_parallel = Option(
        type=cpu.ParallelOptions,
        default=cpu.ParallelOptions(False),
        doc="""Enable automatic parallel optimization, can be fine-tuned by
taking a dictionary of sub-options instead of a boolean, see parfor.py for
detail""",
    )
    nrt = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    no_rewrites = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    error_model = Option(
        type=str,
        default="python",
        doc="TODO",
    )
    fastmath = Option(
        type=cpu.FastMathOptions,
        default=cpu.FastMathOptions(False),
        doc="TODO",
    )
    noalias = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    inline = Option(
        type=cpu.InlineOptions,
        default=cpu.InlineOptions("never"),
        doc="TODO",
    )

    dbg_extend_lifetimes = Option(
        type=bool,
        default=False,
        doc=(
            "Extend variable lifetime for debugging. "
            "This automatically turns on with debug=True."
        ),
    )

    dbg_optnone = Option(
        type=bool,
        default=False,
        doc=(
            "Disable optimization for debug. "
            "Equivalent to adding optnone attribute in the LLVM Function."
        ),
    )

    dbg_directives_only = Option(
        type=bool,
        default=False,
        doc=("Make debug emissions directives-only. " "Used when generating lineinfo."),
    )

    # Bodo change: add Bodo-specific options
    all_args_distributed_block = Option(
        type=bool,
        default=False,
        doc="All args have 1D distribution",
    )

    all_args_distributed_varlength = Option(
        type=bool,
        default=False,
        doc="All args have 1D_Var distribution",
    )

    all_returns_distributed = Option(
        type=bool,
        default=False,
        doc="All returns are distributed",
    )

    returns_maybe_distributed = Option(
        type=bool,
        default=True,
        doc="Returns may be distributed",
    )
    args_maybe_distributed = Option(
        type=bool,
        default=True,
        doc="Arguments may be distributed",
    )

    distributed = Option(
        type=set,
        default=set(),
        doc="distributed arguments or returns",
    )

    distributed_block = Option(
        type=set,
        default=set(),
        doc="distributed 1D arguments or returns",
    )

    replicated = Option(
        type=set,
        default=set(),
        doc="replicated arguments or returns",
    )

    threaded = Option(
        type=set,
        default=set(),
        doc="Threaded arguments or returns",
    )

    pivots = Option(
        type=dict,
        default=dict(),
        doc="pivot values",
    )

    h5_types = Option(
        type=dict,
        default=dict(),
        doc="HDF5 read data types",
    )


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.nrt = True

# Check if Flags source code has changed
if bodo.numba_compat._check_numba_change:
    lines = inspect.getsource(numba.core.compiler.Flags)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "c55571413d2aa723a2c5eb18f1dccf3acfd3b900ab25f751302773a8e5bf48d3"
    ):  # pragma: no cover
        warnings.warn("numba.core.compiler.Flags has changed")

numba.core.compiler.Flags = Flags
numba.core.compiler.DEFAULT_FLAGS = DEFAULT_FLAGS


# adapted from parallel_diagnostics()
def distributed_diagnostics(self, signature=None, level=1):
    """
    Print distributed diagnostic information for the given signature. If no
    signature is present it is printed for all known signatures. level is
    used to adjust the verbosity, level=1 (default) is minimal verbosity,
    and 2, 3, and 4 provide increasing levels of verbosity.
    """
    if signature is None and len(self.signatures) == 0:
        raise bodo.utils.typing.BodoError(
            "Distributed diagnostics not available for a function that is"
            " not compiled yet"
        )

    if bodo.get_rank() != 0:  # only print on 1 process
        return

    def dump(sig):
        ol = self.overloads[sig]
        pfdiag = ol.metadata.get("distributed_diagnostics", None)
        if pfdiag is None:
            msg = "No distributed diagnostic available"
            raise bodo.utils.typing.BodoError(msg)
        pfdiag.dump(level, self.get_metadata(sig))

    if signature is not None:
        dump(signature)
    else:
        [dump(sig) for sig in self.signatures]


numba.core.dispatcher.Dispatcher.distributed_diagnostics = distributed_diagnostics


def master_mode_wrapper(numba_jit_wrapper):  # pragma: no cover
    def _wrapper(pyfunc):
        dispatcher = numba_jit_wrapper(pyfunc)
        return master_mode.MasterModeDispatcher(dispatcher)

    return _wrapper


# shows whether jit compilation is on inside a function or not. The overloaded version
# returns True while regular interpreted version returns False.
# example:
# @bodo.jit
# def f():
#     print(bodo.is_jit_execution())  # prints True
# def g():
#     print(bodo.is_jit_execution())  # prints False
def is_jit_execution():  # pragma: no cover
    return False


@numba.extending.overload(is_jit_execution)
def is_jit_execution_overload():
    return lambda: True  # pragma: no cover


def jit(signature_or_function=None, pipeline_class=None, **options):
    _init_extensions()

    # set nopython by default
    if "nopython" not in options:
        options["nopython"] = True

    # options['parallel'] = True
    options["parallel"] = {
        "comprehension": True,
        "setitem": False,  # FIXME: support parallel setitem
        # setting the new inplace_binop option to False until it is tested and handled
        # TODO: evaluate and enable
        "inplace_binop": False,
        "reduction": True,
        "numpy": True,
        # parallelizing stencils is not supported yet
        "stencil": False,
        "fusion": True,
    }

    pipeline_class = (
        bodo.compiler.BodoCompiler if pipeline_class is None else pipeline_class
    )
    if "distributed" in options and isinstance(options["distributed"], bool):
        dist = options.pop("distributed")
        pipeline_class = pipeline_class if dist else bodo.compiler.BodoCompilerSeq

    if "replicated" in options and isinstance(options["replicated"], bool):
        rep = options.pop("replicated")
        pipeline_class = bodo.compiler.BodoCompilerSeq if rep else pipeline_class

    numba_jit = numba.jit(
        signature_or_function, pipeline_class=pipeline_class, **options
    )
    if (
        master_mode.master_mode_on and bodo.get_rank() == master_mode.MASTER_RANK
    ):  # pragma: no cover
        # when options are passed, this function is called with
        # signature_or_function==None, so numba.jit doesn't return a Dispatcher
        # object. it returns a decorator ("_jit.<locals>.wrapper") to apply
        # to the Python function, and we need to wrap that around our own
        # decorator
        if isinstance(numba_jit, numba.dispatcher._DispatcherBase):
            return master_mode.MasterModeDispatcher(numba_jit)
        else:
            return master_mode_wrapper(numba_jit)
    else:
        return numba_jit


def _init_extensions():
    """initialize Numba extensions for supported packages that are imported.
    This reduces Bodo import time since we don't have to try to import unused packages.
    This is done in as soon as possible since values types in typeof() are needed for
    starting the compilation.
    """
    import sys

    need_refresh = False

    if "sklearn" in sys.modules and "bodo.libs.sklearn_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.libs.sklearn_ext  # noqa

        need_refresh = True

    if "matplotlib" in sys.modules and "bodo.libs.matplotlib_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.libs.matplotlib_ext  # noqa

        need_refresh = True

    if "xgboost" in sys.modules and "bodo.libs.xgb_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.libs.xgb_ext  # noqa

        need_refresh = True

    if "h5py" in sys.modules and "bodo.io.h5_api" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.io.h5_api  # noqa

        if bodo.utils.utils.has_supported_h5py():
            from bodo.io import h5  # noqa

        need_refresh = True

    if "pyspark" in sys.modules and "bodo.libs.pyspark_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import pyspark.sql.functions  # noqa

        import bodo.libs.pyspark_ext  # noqa

        bodo.utils.transform.no_side_effect_call_tuples.update(
            {
                ("col", pyspark.sql.functions),
                (pyspark.sql.functions.col,),
                ("sum", pyspark.sql.functions),
                (pyspark.sql.functions.sum,),
            }
        )

        need_refresh = True

    if "scipy" in sys.modules and "bodo.libs.fft_kernels" not in sys.modules:
        import bodo.libs.fft_kernels  # noqa

        need_refresh = True

    if need_refresh:
        numba.core.registry.cpu_target.target_context.refresh()
