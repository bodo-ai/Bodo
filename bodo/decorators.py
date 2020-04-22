# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Defines decorators of Bodo. Currently just @jit.
"""
import numba
import bodo
from bodo import master_mode


# Add Bodo's options to Numba's allowed options/flags
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_args_distributed_block"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_args_distributed_varlength"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["all_returns_distributed"] = bool
numba.targets.cpu.CPUTargetOptions.OPTIONS["distributed"] = set
numba.targets.cpu.CPUTargetOptions.OPTIONS["distributed_block"] = set
numba.targets.cpu.CPUTargetOptions.OPTIONS["threaded"] = set
numba.targets.cpu.CPUTargetOptions.OPTIONS["pivots"] = dict
numba.targets.cpu.CPUTargetOptions.OPTIONS["h5_types"] = dict
numba.compiler.Flags.OPTIONS["all_args_distributed_block"] = False
numba.compiler.Flags.OPTIONS["all_args_distributed_varlength"] = False
numba.compiler.Flags.OPTIONS["all_returns_distributed"] = False
numba.compiler.Flags.OPTIONS["distributed"] = set()
numba.compiler.Flags.OPTIONS["distributed_block"] = set()
numba.compiler.Flags.OPTIONS["threaded"] = set()
numba.compiler.Flags.OPTIONS["pivots"] = dict()
numba.compiler.Flags.OPTIONS["h5_types"] = dict()


def bodo_set_flags(self, flags):
    """Add Bodo's options to 'set_flags' function of numba.targets.options.TargetOptions
    Handles Bodo flags, then calls Numba for handling regular Numba flags
    """
    # remove Bodo options from 'values', call 'numba_set_flags', restore Bodo options
    orig_values = self.values.copy()
    kws = self.values

    if kws.pop("all_args_distributed_block", False):
        flags.set("all_args_distributed_block")

    if kws.pop("all_args_distributed_varlength", False):
        flags.set("all_args_distributed_varlength")

    if kws.pop("all_returns_distributed", False):
        flags.set("all_returns_distributed")

    if "distributed" in kws:
        flags.set("distributed", kws.pop("distributed"))

    if "distributed_block" in kws:
        flags.set("distributed_block", kws.pop("distributed_block"))

    if "threaded" in kws:
        flags.set("threaded", kws.pop("threaded"))

    if "pivots" in kws:
        flags.set("pivots", kws.pop("pivots"))

    if "h5_types" in kws:
        flags.set("h5_types", kws.pop("h5_types"))

    self.numba_set_flags(flags)
    self.values = orig_values


numba.targets.options.TargetOptions.numba_set_flags = numba.targets.options.TargetOptions.set_flags
numba.targets.options.TargetOptions.set_flags = bodo_set_flags


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


def master_mode_wrapper(numba_jit_wrapper):  # pragma: no cover
    def _wrapper(pyfunc):
        dispatcher = numba_jit_wrapper(pyfunc)
        return master_mode.MasterModeDispatcher(dispatcher)
    return _wrapper


def jit(signature_or_function=None, **options):
    # set nopython by default
    if "nopython" not in options:
        options["nopython"] = True

    # options['parallel'] = True
    options["parallel"] = {
        "comprehension": True,
        "setitem": False,  # FIXME: support parallel setitem
        "reduction": True,
        "numpy": True,
        "stencil": True,
        "fusion": True,
    }

    numba_jit = numba.jit(
        signature_or_function, pipeline_class=bodo.compiler.BodoCompiler, **options
    )
    if master_mode.master_mode_on and bodo.get_rank() == master_mode.MASTER_RANK:  # pragma: no cover
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
