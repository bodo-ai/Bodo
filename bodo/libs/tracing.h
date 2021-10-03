#ifndef _TRACING_H_INCLUDED
#define _TRACING_H_INCLUDED

#include <Python.h>

namespace tracing {

/**
  * An Event instance corresponds to an instance of bodo.utils.tracing.Event
  * and this just a C++ wrapper. Refer to bodo/utils/tracing.pyx for more
  * information.
  */
class Event {

public:

    Event(const std::string& name, bool is_parallel=true, bool sync=true) {
        // TODO pass aggregate and sync options for finalize here in case
        // we want to use these options on Event destruction when going out of
        // scope
        // TODO error checking
        // import bodo.utils.tracing
        PyObject* tracing_mod = PyImport_ImportModule("bodo.utils.tracing");
        PyObject* is_tracing_func = PyObject_GetAttrString(tracing_mod, "is_tracing");
        PyObject* is_tracing_obj = PyObject_CallFunction(is_tracing_func, NULL);
        tracing = (is_tracing_obj == Py_True);
        Py_DECREF(is_tracing_obj);
        Py_DECREF(is_tracing_func);
        if (tracing) {
            // event_py = tracing.Event(name, is_parallel=is_parallel, sync=sync)
            PyObject *event_ctor = PyObject_GetAttrString(tracing_mod, "Event");
            event_py = PyObject_CallFunction(event_ctor, "si", name.c_str(), int(is_parallel), int(sync));
            Py_DECREF(event_ctor);
        }
        Py_DECREF(tracing_mod);
    }

    ~Event() {
        if (event_py != nullptr) {
            // If PyErr_Occurred we don't call finalize(), because it calls
            // into Python, triggering a new (different) error.
            // By not calling finalize we are effectively canceling this event
            // Set same value for sync and aggregate.
            if (!finalized && !PyErr_Occurred()) finalize();
            Py_DECREF(event_py);
        }
    }

    bool is_tracing() { return tracing; }

    void add_attribute(const std::string& name, size_t value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "sn", name.c_str(), value);
    }

    void add_attribute(const std::string& name, int64_t value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "sL", name.c_str(), value);
    }

    void add_attribute(const std::string& name, int value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "si", name.c_str(), value);
    }

    void add_attribute(const std::string& name, double value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "sd", name.c_str(), value);
    }

    void add_attribute(const std::string& name, float value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "sf", name.c_str(), value);
    }

    void finalize(bool aggregate=true, bool sync=true) {
        // call event_py.finalize(aggregate=aggregate, sync=sync)
        if (event_py)
            PyObject_CallMethod(event_py, "finalize", "ii", int(sync), int(aggregate));
        finalized = true;
    }

    void add_attribute(const std::string& name, const std::string& value) {
        if (event_py)
            PyObject_CallMethod(event_py, "add_attribute", "ss", name.c_str(), value.c_str());
    }

private:
    bool tracing = false;
    PyObject* event_py = nullptr;  // instance of bodo.utils.tracing.Event
    bool finalized = false;
};

}

#endif // _TRACING_H_INCLUDED
