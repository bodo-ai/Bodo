#include "_bodo_ext.h"
#include "_bodo_common.h"

extern "C" {

/**
 * @brief Get the Cython-generated plan_optimizer module, which requires special
 * initialization.
 *
 * @return PyObject* plan_optimizer module object or nullptr on failure.
 */
PyObject* get_plan_optimizer_module() {
    // Cython uses multi-phase initialization which needs
    // PyModule_FromDefAndSpec(). See:
    // https://docs.python.org/3/c-api/module.html#c.PyModuleDef
    PyModuleDef* moddef = nullptr;  // (PyModuleDef*)PyInit_plan_optimizer();

    PyObject* machinery = PyImport_ImportModule("importlib.machinery");
    if (!machinery) {
        PyErr_Print();
        return nullptr;
    }

    PyObject* module_spec_cls = PyObject_GetAttrString(machinery, "ModuleSpec");
    Py_DECREF(machinery);
    if (!module_spec_cls) {
        PyErr_Print();
        return nullptr;
    }

    PyObject* args = Py_BuildValue("sO", "plan_optimizer", Py_None);
    if (!args) {
        PyErr_Print();
        Py_DECREF(module_spec_cls);
        return nullptr;
    }

    PyObject* spec = PyObject_CallObject(module_spec_cls, args);
    Py_DECREF(module_spec_cls);
    Py_DECREF(args);
    if (!spec) {
        PyErr_Print();
        return nullptr;
    }

    PyObject* mod = PyModule_FromDefAndSpec(moddef, spec);
    Py_DECREF(spec);
    if (!mod) {
        PyErr_Print();
        return nullptr;
    }

    if (PyModule_ExecDef(mod, moddef) < 0) {
        PyErr_Print();
        Py_DECREF(mod);
        return nullptr;
    }
    return mod;
}

PyMODINIT_FUNC PyInit_ext(void) {
    PyObject* m;
    MOD_DEF(m, "ext", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromPyInit(m, hdist);
    SetAttrStringFromPyInit(m, hstr_ext);
    SetAttrStringFromPyInit(m, decimal_ext);
    SetAttrStringFromPyInit(m, hdatetime_ext);
    SetAttrStringFromPyInit(m, hio);
    SetAttrStringFromPyInit(m, array_ext);
    SetAttrStringFromPyInit(m, s3_reader);
    SetAttrStringFromPyInit(m, hdfs_reader);
#ifndef NO_HDF5
    SetAttrStringFromPyInit(m, _hdf5);
#endif
    SetAttrStringFromPyInit(m, arrow_cpp);
    SetAttrStringFromPyInit(m, csv_cpp);
    SetAttrStringFromPyInit(m, json_cpp);
    SetAttrStringFromPyInit(m, memory_budget_cpp);
    SetAttrStringFromPyInit(m, stream_join_cpp);
    SetAttrStringFromPyInit(m, stream_groupby_cpp);
    SetAttrStringFromPyInit(m, stream_dict_encoding_cpp);
    SetAttrStringFromPyInit(m, stream_sort_cpp);
    SetAttrStringFromPyInit(m, table_builder_cpp);
    SetAttrStringFromPyInit(m, query_profile_collector_cpp);
#ifdef BUILD_WITH_V8
    SetAttrStringFromPyInit(m, javascript_udf_cpp);
#endif

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    // // Setup the Cython-generated plan_optimizer module
    // PyObject* plan_opt_mod = get_plan_optimizer_module();
    // if (!plan_opt_mod) {
    //     PyErr_Print();
    //     return nullptr;
    // }
    // if (PyObject_SetAttrString(m, "plan_optimizer", plan_opt_mod) < 0) {
    //     PyErr_Print();
    //     Py_DECREF(plan_opt_mod);
    //     return nullptr;
    // }
    // Py_DECREF(plan_opt_mod);

    return m;
}

} /* extern "C" */
