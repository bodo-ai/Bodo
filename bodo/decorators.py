# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Defines decorators of Bodo. Currently just @jit.
"""
import numba
import bodo


# Add Bodo's options to Numba's allowed options/flags
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_args_distributed"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_args_distributed_varlength"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_returns_distributed"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["distributed"] = set
numba.compiler.Flags.OPTIONS["all_args_distributed"] = False
numba.compiler.Flags.OPTIONS["all_args_distributed_varlength"] = False
numba.compiler.Flags.OPTIONS["all_returns_distributed"] = False
numba.compiler.Flags.OPTIONS["distributed"] = set()


# Add Bodo's options to 'set_flags' function of numba.targets.options.TargetOptions
# and replace it since it checks for allowed flags
def set_flags(self, flags):
    """
    Provide default flags setting logic.
    Subclass can override.
    """
    kws = self.values.copy()

    if kws.pop('nopython', False) == False:
        flags.set("enable_pyobject")

    if kws.pop("forceobj", False):
        flags.set("force_pyobject")

    if kws.pop('looplift', True):
        flags.set("enable_looplift")

    if kws.pop('boundcheck', False):
        flags.set("boundcheck")

    if kws.pop('_nrt', True):
        flags.set("nrt")

    if kws.pop('debug', numba.config.DEBUGINFO_DEFAULT):
        flags.set("debuginfo")
        flags.set("boundcheck")

    if kws.pop('nogil', False):
        flags.set("release_gil")

    if kws.pop('no_rewrites', False):
        flags.set('no_rewrites')

    if kws.pop('no_cpython_wrapper', False):
        flags.set('no_cpython_wrapper')

    if 'parallel' in kws:
        flags.set('auto_parallel', kws.pop('parallel'))

    if 'fastmath' in kws:
        flags.set('fastmath', kws.pop('fastmath'))

    if 'error_model' in kws:
        flags.set('error_model', kws.pop('error_model'))

    if 'inline' in kws:
        flags.set('inline', kws.pop('inline'))

    flags.set("enable_pyobject_looplift")

    if kws.pop('all_args_distributed', False):
        flags.set("all_args_distributed")

    if kws.pop('all_args_distributed_varlength', False):
        flags.set("all_args_distributed_varlength")

    if kws.pop('all_returns_distributed', False):
        flags.set("all_returns_distributed")

    if 'distributed' in kws:
        flags.set('distributed', kws.pop('distributed'))

    if kws:
        # Unread options?
        raise NameError("Unrecognized options: %s" % kws.keys())


numba.targets.options.TargetOptions.set_flags = set_flags


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
            raise ValueError(msg)
        pfdiag.dump(level)

    if signature is not None:
        dump(signature)
    else:
        [dump(sig) for sig in self.signatures]


numba.dispatcher.Dispatcher.distributed_diagnostics = distributed_diagnostics


def jit(signature_or_function=None, **options):
    # set nopython by default
    if "nopython" not in options:
        options["nopython"] = True

    _locals = options.pop("locals", {})
    assert isinstance(_locals, dict)

    # put pivots in locals TODO: generalize numba.jit options
    pivots = options.pop("pivots", {})
    assert isinstance(pivots, dict)
    for var, vals in pivots.items():
        _locals[var + ":pivot"] = vals

    h5_types = options.pop("h5_types", {})
    assert isinstance(h5_types, dict)
    for var, vals in h5_types.items():
        _locals[var + ":h5_types"] = vals

    distributed_varlength = set(options.pop("distributed_varlength", set()))
    assert isinstance(distributed_varlength, (set, list))
    _locals["##distributed_varlength"] = distributed_varlength

    threaded = set(options.pop("threaded", set()))
    assert isinstance(threaded, (set, list))
    _locals["##threaded"] = threaded

    options["locals"] = _locals

    # options['parallel'] = True
    options["parallel"] = {
        "comprehension": True,
        "setitem": False,  # FIXME: support parallel setitem
        "reduction": True,
        "numpy": True,
        "stencil": True,
        "fusion": True,
    }

    return numba.jit(
        signature_or_function, pipeline_class=bodo.compiler.BodoCompiler, **options
    )
