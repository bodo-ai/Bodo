"""
Defines decorators of Bodo. Currently just @jit.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import types as pytypes
import warnings

# NOTE: 'numba_compat' has to be imported first in bodo package to make sure all Numba
# patches are applied before Bodo's Numba use (e.g. 'overload' is replaced properly)
import bodo.numba_compat  # isort:skip

import numba
from numba.core import cgutils, cpu, serialize, types
from numba.core.options import _mapping
from numba.core.targetconfig import Option, TargetConfig
from numba.core.typing.templates import signature
from numba.extending import lower_builtin, models, register_model

import bodo
import bodo.compiler  # noqa # side effect: initialize JIT compiler
from bodo.pandas_compat import bodo_pandas_udf_execution_engine

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
numba.core.cpu.CPUTargetOptions.spawn = _mapping("spawn")
numba.core.cpu.CPUTargetOptions.propagate_env = _mapping("propagate_env")
numba.core.cpu.CPUTargetOptions.distributed_diagnostics = _mapping(
    "distributed_diagnostics"
)


class Flags(TargetConfig):
    __slots__ = ()

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
        doc=("Make debug emissions directives-only. Used when generating lineinfo."),
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
        default={},
        doc="pivot values",
    )

    h5_types = Option(
        type=dict,
        default={},
        doc="HDF5 read data types",
    )

    spawn = Option(
        type=bool,
        default=False,
        doc="Spawn MPI processes",
    )

    propagate_env = Option(
        type=list,
        default=[],
        doc="Environment variables to propagate to spawned MPI processes",
    )

    distributed_diagnostics = Option(
        type=bool,
        default=False,
        doc="Print distributed diagnostics information",
    )


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.nrt = True

# Check if Flags source code has changed
if bodo.numba_compat._check_numba_change:
    lines = inspect.getsource(numba.core.compiler.Flags)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "834e3920054f7758de2170c87ea884e59c35fd57f5777d559168a95e4ba2ec56"
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


bodo.is_jit_execution = is_jit_execution
bodo.jitclass = bodo.numba_compat.jitclass


def jit(signature_or_function=None, pipeline_class=None, **options):
    # Use spawn mode if specified in decorator or enabled globally (decorator takes
    # precedence)
    disable_jit = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"
    dist_mode = options.get("distributed", True) is not False

    py_func = None
    if isinstance(signature_or_function, pytypes.FunctionType):
        py_func = signature_or_function

    if options.get("spawn", bodo.spawn_mode) and not disable_jit and dist_mode:
        from bodo.spawn.spawner import SpawnDispatcher
        from bodo.spawn.worker_state import is_worker

        if is_worker():
            # If we are already in the worker, just use regular to
            # compile/execute directly
            return _jit(
                signature_or_function=signature_or_function,
                pipeline_class=pipeline_class,
                **options,
            )

        def return_wrapped_fn(py_func):
            submit_jit_args = {**options}
            submit_jit_args["pipeline_class"] = pipeline_class
            return SpawnDispatcher(py_func, submit_jit_args)

        if py_func is not None:
            return return_wrapped_fn(py_func)

        bodo_jit = return_wrapped_fn
    elif "propagate_env" in options:
        raise bodo.utils.typing.BodoError(
            "spawn=False while propagate_env is set. No worker to propagate env vars."
        )
    else:
        bodo_jit = _jit(signature_or_function, pipeline_class, **options)

    # Return jit decorator that can be used in Pandas UDF function. See definition of
    # bodo_pandas_udf_execution_engine for more details.
    if py_func is None:
        bodo_jit.__pandas_udf__ = bodo_pandas_udf_execution_engine

    return bodo_jit


jit.__pandas_udf__ = bodo_pandas_udf_execution_engine
bodo.jit = jit


def _jit(signature_or_function=None, pipeline_class=None, **options):
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
    return numba_jit


def _cfunc(signature_or_function=None, pipeline_class=None, **options):
    """Internal wrapper for Cfunc for UDFs in DataFrame Library."""
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
        bodo.compiler.BodoCompilerSeq if pipeline_class is None else pipeline_class
    )

    numba_cfunc = numba.cfunc(
        signature_or_function, pipeline_class=pipeline_class, **options
    )

    return numba_cfunc


def _init_extensions():
    """initialize Numba extensions for supported packages that are imported.
    This reduces Bodo import time since we don't have to try to import unused packages.
    This is done in as soon as possible since values types in typeof() are needed for
    starting the compilation.
    """
    import sys

    need_refresh = False

    if "sklearn" in sys.modules and "bodo.ml_support.sklearn_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.ml_support.sklearn_ext  # noqa
        import bodo.ml_support.sklearn_cluster_ext  # noqa
        import bodo.ml_support.sklearn_ensemble_ext  # noqa
        import bodo.ml_support.sklearn_feature_extraction_ext  # noqa
        import bodo.ml_support.sklearn_linear_model_ext  # noqa
        import bodo.ml_support.sklearn_metrics_ext  # noqa
        import bodo.ml_support.sklearn_model_selection_ext  # noqa
        import bodo.ml_support.sklearn_naive_bayes_ext  # noqa
        import bodo.ml_support.sklearn_preprocessing_ext  # noqa
        import bodo.ml_support.sklearn_svm_ext  # noqa
        import bodo.ml_support.sklearn_utils_ext  # noqa

        need_refresh = True

    if "matplotlib" in sys.modules and "bodo.libs.matplotlib_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.libs.matplotlib_ext  # noqa

        need_refresh = True

    if "xgboost" in sys.modules and "bodo.ml_support.xgb_ext" not in sys.modules:
        # side effect: initialize Numba extensions
        import bodo.ml_support.xgb_ext  # noqa

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

    if need_refresh:
        numba.core.registry.cpu_target.target_context.refresh()


class WrapPythonDispatcher:
    """Dispatcher for JIT wrapped Python functions."""

    def __init__(self, py_func, return_type):
        self.py_func = py_func
        self.return_type = return_type
        # Copy function name attributes similar to regular Dispatchers to allow
        # handling in find_callname()
        self.__name__ = py_func.__name__
        self.__qualname__ = py_func.__qualname__
        # Default argument values match py_func
        self.__defaults__ = py_func.__defaults__
        self.__code__ = py_func.__code__
        # Required for compiler frontend used in _get_df_apply_used_cols(), see:
        # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/bytecode.py#L32
        self.__numba__ = "py_func"

    def __call__(self, *args, **kwargs):
        return self.py_func(*args, **kwargs)

    @property
    def _numba_type_(self):
        return WrapPythonDispatcherType(self)


def _check_return_type(return_type):
    """Check and convert wrap_python return type to Numba type."""

    from numba.core import sigutils, types

    from bodo.utils.typing import BodoError

    if isinstance(return_type, str):
        return_type = sigutils._parse_signature_string(return_type)

    if isinstance(return_type, types.abstract._TypeMetaclass):
        raise BodoError(
            f"wrap_python requires full data types, not just data type "
            f"classes. For example, 'bodo.types.DataFrameType((bodo.types.float64[::1],), "
            f"bodo.types.RangeIndexType(), ('A',))' is a valid data type but 'bodo.types.DataFrameType' is not.\n"
            f"Return type is type class {return_type}."
        )
    if not isinstance(return_type, types.Type):
        raise BodoError(
            f"A data type is required for wrap_python return type annotation, not {return_type}."
        )

    # list/set reflection is irrelevant in wrap_python
    if isinstance(return_type, (types.List, types.Set)):
        return_type = return_type.copy(reflected=False)

    return return_type


def wrap_python(return_type: str | types.Type):
    """Creates a JIT wrapper around a regular Python function to allow its use inside
    JIT functions (including UDFs).
    The data type of the function output must be specified.

    Args:
        return_type (types.Type|str): data type of function output
    """
    return_type = _check_return_type(return_type)

    def wrapper(func):
        return WrapPythonDispatcher(func, return_type)

    return wrapper


bodo.wrap_python = wrap_python


class WrapPythonDispatcherType(numba.types.Callable, numba.types.Opaque):
    """Data type for JIT wrapper dispatcher."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self._overload_cache = {}
        self._sigs = []
        super().__init__(name=f"WrapPythonDispatcherType({dispatcher})")

    def get_call_type(self, context, args, kws):
        """Get call signature for JIT wrapper dispatcher call and install its lowering
        implementation.
        """
        pysig = numba.core.utils.pysignature(self.dispatcher.py_func)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)

        def impl(context, builder, sig, args):
            pyapi = context.get_python_api(builder)
            env_manager = context.get_env_manager(builder)
            c = numba.core.pythonapi._BoxContext(context, builder, pyapi, env_manager)

            func_obj = _load_wrap_python_function(pyapi, self.dispatcher.py_func)

            arg_objs = []
            for arg_type, arg in zip(sig.args, args):
                c.context.nrt.incref(c.builder, arg_type, arg)
                arg_obj = pyapi.from_native_value(arg_type, arg, env_manager)
                arg_objs.append(arg_obj)

            # Output handling similar to:
            # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/lowering.py#L963

            out_obj = c.pyapi.call_function_objargs(func_obj, arg_objs)

            # Check for user function exceptions
            with builder.if_then(c.pyapi.c_api_error()):
                context.call_conv.return_exc(builder)

            # Check output type to match the expected return type
            type_checker_func_obj = c.pyapi.unserialize(
                c.pyapi.serialize_object(bodo.utils.typing._check_objmode_type)
            )
            type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(sig.return_type))
            fixed_out_obj = c.pyapi.call_function_objargs(
                type_checker_func_obj, [out_obj, type_obj]
            )

            # Error during type check
            with builder.if_then(c.pyapi.c_api_error()):
                context.call_conv.return_exc(builder)

            pyapi.decref(out_obj)
            pyapi.decref(type_checker_func_obj)
            pyapi.decref(type_obj)

            out = pyapi.to_native_value(sig.return_type, fixed_out_obj)

            # Release objs
            pyapi.decref(fixed_out_obj)
            for arg_obj in arg_objs:
                pyapi.decref(arg_obj)

            # Error during unboxing
            with builder.if_then(out.is_error):
                context.call_conv.return_exc(builder)

            # cleanup output
            if callable(out.cleanup):
                out.cleanup()

            return out.value

        self._overload_cache[folded_args] = impl
        lower_builtin(impl, *folded_args)(impl)

        out_sig = signature(self.dispatcher.return_type, *folded_args).replace(
            pysig=pysig
        )
        self._sigs.append(out_sig)
        return out_sig

    def get_call_signatures(self):
        return self._sigs, True

    def get_impl_key(self, sig):
        return self._overload_cache[sig.args]

    @property
    def key(self):
        return self.dispatcher.py_func, self.dispatcher.return_type


register_model(WrapPythonDispatcherType)(models.OpaqueModel)


def _load_wrap_python_function(pyapi, py_func):
    """Load the JIT wrapper function object to call. Handles serialization and
    unserialization (if serializable) to support caching.
    Also, caches the unseralized
    function in a global variable to avoid repeated serialization/unserialization.
    """
    from llvmlite import ir as lir

    # Adapted from:
    # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/pythonapi.py#L1663
    builder = pyapi.builder
    tyctx = pyapi.context
    m = builder.module

    # Add a global variable to cache the unserialized function
    gv = lir.GlobalVariable(
        m,
        pyapi.pyobj,
        name=m.get_unique_name("cached_wrap_python_py_func"),
    )
    gv.initializer = gv.type.pointee(None)
    gv.linkage = "internal"

    cached = builder.load(gv)
    with builder.if_then(cgutils.is_null(builder, cached)):
        if serialize.is_serialiable(py_func):
            callee = pyapi.unserialize(pyapi.serialize_object(py_func))
        else:
            callee = tyctx.add_dynamic_addr(
                builder,
                id(py_func),
                info="wrap_python_function",
            )
        # Incref the function and cache it
        pyapi.incref(callee)
        builder.store(callee, gv)

    callee = builder.load(gv)
    return callee


_already_checked_rts = False


def _check_numba_rtsys():
    """Check if Numba RTS is already initialized, which indicates that Numba JIT
    compilation has already happened before Bodo JIT is imported.
    This can cause issues in memory management, leading to crashes (see BSE-5112).
    """
    global _already_checked_rts
    if _already_checked_rts:
        return
    _already_checked_rts = True

    from numba.core.runtime import rtsys

    if rtsys._init:  # pragma: no cover
        # Avoid spawner errors in finalization
        bodo.spawn.spawner.spawner = None
        raise RuntimeError(
            "Bodo JIT must be imported before any Numba JIT compilation if Bodo is used later in the program. "
            "Please add 'import bodo.decorators' before Numba JIT uses."
        )


_check_numba_rtsys()
