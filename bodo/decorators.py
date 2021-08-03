# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Defines decorators of Bodo. Currently just @jit.
"""
import numba

import bodo
from bodo import master_mode

# Add Bodo's options to Numba's allowed options/flags
numba.core.cpu.CPUTargetOptions.OPTIONS["all_args_distributed_block"] = bool
numba.core.cpu.CPUTargetOptions.OPTIONS["all_args_distributed_varlength"] = bool
numba.core.cpu.CPUTargetOptions.OPTIONS["all_returns_distributed"] = bool
numba.core.cpu.CPUTargetOptions.OPTIONS["returns_maybe_distributed"] = bool
numba.core.cpu.CPUTargetOptions.OPTIONS["args_maybe_distributed"] = bool
numba.core.cpu.CPUTargetOptions.OPTIONS["distributed"] = set
numba.core.cpu.CPUTargetOptions.OPTIONS["distributed_block"] = set
numba.core.cpu.CPUTargetOptions.OPTIONS["threaded"] = set
numba.core.cpu.CPUTargetOptions.OPTIONS["pivots"] = dict
numba.core.cpu.CPUTargetOptions.OPTIONS["h5_types"] = dict
numba.core.compiler.Flags.OPTIONS["all_args_distributed_block"] = False
numba.core.compiler.Flags.OPTIONS["all_args_distributed_varlength"] = False
numba.core.compiler.Flags.OPTIONS["all_returns_distributed"] = False
numba.core.compiler.Flags.OPTIONS["returns_maybe_distributed"] = True
numba.core.compiler.Flags.OPTIONS["args_maybe_distributed"] = True
numba.core.compiler.Flags.OPTIONS["distributed"] = set()
numba.core.compiler.Flags.OPTIONS["distributed_block"] = set()
numba.core.compiler.Flags.OPTIONS["threaded"] = set()
numba.core.compiler.Flags.OPTIONS["pivots"] = dict()
numba.core.compiler.Flags.OPTIONS["h5_types"] = dict()


def bodo_set_flags(self, flags):
    """Add Bodo's options to 'set_flags' function of numba.core.options.TargetOptions
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

    if not kws.pop("returns_maybe_distributed", True):
        flags.unset("returns_maybe_distributed")

    if not kws.pop("args_maybe_distributed", True):
        flags.unset("args_maybe_distributed")

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


numba.core.options.TargetOptions.numba_set_flags = (
    numba.core.options.TargetOptions.set_flags
)
numba.core.options.TargetOptions.set_flags = bodo_set_flags


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

    # turn off automatic distribution detection for args/returns if some distribution
    # is manually specified by the user
    if "distributed" in options or "distributed_block" in options:
        if "args_maybe_distributed" not in options:
            options["args_maybe_distributed"] = False
        if "returns_maybe_distributed" not in options:
            options["returns_maybe_distributed"] = False

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
