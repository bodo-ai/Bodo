import sys

from numba.extending import overload, overload_method

import bodo
from bodo.utils.py_objs import install_py_obj_class

this_module = sys.modules[__name__]
# We do the overloads in a Python file (instead of tracing.pyx) because they
# don't work when done from Cython-compiled code (exact cause is yet unknown)


BodoTracingEventType, _ = install_py_obj_class(
    types_name="bodo_tracing_event_type",
    python_type=bodo.utils.tracing.Event,
    module=this_module,
    class_name="BodoTracingEventType",
    model_name="BodoTracingEventModel",
)


@overload(bodo.utils.tracing.Event, no_unliteral=True)
def tracing_Event_overload(name, is_parallel=True, sync=True):
    def _tracing_Event_impl(name, is_parallel=True, sync=True):  # pragma: no cover
        with bodo.no_warning_objmode(e="bodo_tracing_event_type"):
            e = bodo.utils.tracing.Event(name, is_parallel=is_parallel, sync=sync)
        return e

    return _tracing_Event_impl


@overload_method(BodoTracingEventType, "finalize", no_unliteral=True)
def overload_event_finalize(e, aggregate=True):
    def _event_finalize_overload_impl(e, aggregate=True):  # pragma: no cover
        with bodo.no_warning_objmode:
            e.finalize(aggregate=aggregate)

    return _event_finalize_overload_impl


@overload_method(BodoTracingEventType, "add_attribute", no_unliteral=True)
def overload_event_add_attribute(e, name, value):
    def _event_add_attribute_overload_impl(e, name, value):  # pragma: no cover
        with bodo.no_warning_objmode:
            e.add_attribute(name, value)

    return _event_add_attribute_overload_impl


@overload(bodo.utils.tracing.reset, no_unliteral=True)
def tracing_reset_overload(trace_fname=None):
    def _tracing_reset_overload_impl(trace_fname=None):  # pragma: no cover
        with bodo.no_warning_objmode:
            bodo.utils.tracing.reset(trace_fname=trace_fname)

    return _tracing_reset_overload_impl


@overload(bodo.utils.tracing.start, no_unliteral=True)
def tracing_start_overload(trace_fname=None):
    def _tracing_start_overload_impl(trace_fname=None):  # pragma: no cover
        with bodo.no_warning_objmode:
            bodo.utils.tracing.start(trace_fname=trace_fname)

    return _tracing_start_overload_impl


@overload(bodo.utils.tracing.stop, no_unliteral=True)
def tracing_stop_overload():
    def _tracing_stop_overload_impl():  # pragma: no cover
        with bodo.no_warning_objmode:
            bodo.utils.tracing.stop()

    return _tracing_stop_overload_impl


@overload(bodo.utils.tracing.is_tracing, no_unliteral=True)
def tracing_is_tracing_overload():
    def _tracing_is_tracing_overload_impl():  # pragma: no cover
        with bodo.no_warning_objmode(b="types.boolean"):
            b = bodo.utils.tracing.is_tracing()
        return b

    return _tracing_is_tracing_overload_impl


@overload(bodo.utils.tracing.dump, no_unliteral=True)
def tracing_dump_overload(fname=None, clear_traces=True):
    def _tracing_dump_overload_impl(fname=None, clear_traces=True):  # pragma: no cover
        with bodo.no_warning_objmode:
            bodo.utils.tracing.dump(fname=fname, clear_traces=clear_traces)

    return _tracing_dump_overload_impl
