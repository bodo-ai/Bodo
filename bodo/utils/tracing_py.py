import numba
from numba.core import types
from numba.extending import (
    NativeValue,
    box,
    models,
    overload,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo

# We do the overloads in a Python file (instead of tracing.pyx) because they
# don't work when done from Cython-compiled code (exact cause is yet unknown)


class BodoTracingEventType(types.Opaque):
    def __init__(self):
        super(BodoTracingEventType, self).__init__(name="BodoTracingEventType")


bodo_tracing_event_type = BodoTracingEventType()
types.bodo_tracing_event_type = bodo_tracing_event_type

register_model(BodoTracingEventType)(models.OpaqueModel)


@typeof_impl.register(bodo.utils.tracing.Event)
def typeof_event(val, c):
    return bodo_tracing_event_type


@box(BodoTracingEventType)
def box_bodo_tracing_event(typ, val, c):
    # NOTE: we can't just let Python steal a reference since boxing can happen
    # at any point and even in a loop, which can make refcount invalid.
    # see implementation of str.contains and test_contains_regex
    # TODO: investigate refcount semantics of boxing in Numba when variable is returned
    # from function versus not returned
    c.pyapi.incref(val)
    return val


@unbox(BodoTracingEventType)
def unbox_bodo_tracing_event(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(bodo.utils.tracing.Event, no_unliteral=True)
def tracing_Event_overload(name, is_parallel=True, sync=True):
    def _tracing_Event_impl(name, is_parallel=True, sync=True):  # pragma: no cover
        with numba.objmode(e="bodo_tracing_event_type"):
            e = bodo.utils.tracing.Event(name, is_parallel=is_parallel, sync=sync)
        return e

    return _tracing_Event_impl


@overload_method(BodoTracingEventType, "finalize", no_unliteral=True)
def overload_event_finalize(e, sync=True, aggregate=True):
    def _event_finalize_overload_impl(e, sync=True, aggregate=True):  # pragma: no cover
        with numba.objmode:
            e.finalize(sync=sync, aggregate=aggregate)

    return _event_finalize_overload_impl


@overload_method(BodoTracingEventType, "add_attribute", no_unliteral=True)
def overload_event_add_attribute(e, name, value):
    def _event_add_attribute_overload_impl(e, name, value):  # pragma: no cover
        with numba.objmode:
            e.add_attribute(name, value)

    return _event_add_attribute_overload_impl


@overload(bodo.utils.tracing.reset, no_unliteral=True)
def tracing_reset_overload(trace_fname=None):
    def _tracing_reset_overload_impl(trace_fname=None):  # pragma: no cover
        with numba.objmode:
            bodo.utils.tracing.reset(trace_fname=trace_fname)

    return _tracing_reset_overload_impl


@overload(bodo.utils.tracing.start, no_unliteral=True)
def tracing_start_overload(trace_fname=None):
    def _tracing_start_overload_impl(trace_fname=None):  # pragma: no cover
        with numba.objmode:
            bodo.utils.tracing.start(trace_fname=trace_fname)

    return _tracing_start_overload_impl
