# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Defines decorators of Bodo. Currently just @jit.
"""
import numba
import bodo


# adapted from parallel_diagnostics()
def distributed_diagnostics(self, signature=None, level=1):
    """
    Print distributed diagnostic information for the given signature. If no
    signature is present it is printed for all known signatures. level is
    used to adjust the verbosity, level=1 (default) is minimal verbosity,
    and 2, 3, and 4 provide increasing levels of verbosity.
    """

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

    distributed = set(options.pop("distributed", set()))
    assert isinstance(distributed, (set, list))
    _locals["##distributed"] = distributed

    distributed_varlength = set(options.pop("distributed_varlength", set()))
    assert isinstance(distributed_varlength, (set, list))
    _locals["##distributed_varlength"] = distributed_varlength

    all_args_distributed = options.pop("all_args_distributed", False)
    assert isinstance(all_args_distributed, bool)
    _locals["##all_args_distributed"] = all_args_distributed

    all_args_distributed_varlength = options.pop(
        "all_args_distributed_varlength", False
    )
    assert isinstance(all_args_distributed_varlength, bool)
    _locals["##all_args_distributed_varlength"] = all_args_distributed_varlength

    all_returns_distributed = options.pop("all_returns_distributed", False)
    assert isinstance(all_returns_distributed, bool)
    _locals["##all_returns_distributed"] = all_returns_distributed

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
