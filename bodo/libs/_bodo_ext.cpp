#include "_bodo_ext.h"
#include "_bodo_common.h"

// Declarations for functions defined in theta_utils.cpp
extern "C" {
void bodo_theta_utils_delete_sketches(uintptr_t ptr);
PyObject *bodo_theta_utils_compact_serialize(uintptr_t ptr);
PyObject *bodo_theta_utils_merge_and_write_puffin(
    PyObject *serialized_list, const char *puffin_file_loc,
    const char *bucket_region, int64_t snapshot_id, int64_t sequence_number,
    PyObject *iceberg_schema_py, PyObject *pyarrow_fs_py,
    const char *existing_puffin_loc);
}

extern "C" {

/**
 * @brief Python method: delete an UpdateSketchCollection.
 * Argument: integer (uintptr_t pointer to UpdateSketchCollection).
 */
PyObject *delete_sketches_py_entrypt(PyObject *self, PyObject *args) {
    uintptr_t ptr = 0;
    if (!PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }
    bodo_theta_utils_delete_sketches(ptr);
    Py_RETURN_NONE;
}

/**
 * @brief Python method: compact an UpdateSketchCollection and serialize
 * the result to a Python bytes object.
 * Argument: integer (uintptr_t pointer to UpdateSketchCollection).
 * Returns: bytes object, or None if pointer is 0.
 */
PyObject *compact_sketches_py_entrypt(PyObject *self, PyObject *args) {
    uintptr_t ptr = 0;
    if (!PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }
    return bodo_theta_utils_compact_serialize(ptr);
}

/**
 * @brief Python method: merge pre-serialized CompactSketchCollections and
 * write the puffin file.
 *
 * Arguments:
 *   serialized_list   - list of bytes objects (one per rank)
 *   puffin_file_loc   - output path for the puffin file
 *   bucket_region     - S3 bucket region (empty for local)
 *   snapshot_id       - Iceberg snapshot ID (int64)
 *   sequence_number   - Iceberg sequence number (int64)
 *   iceberg_schema    - PyArrow schema
 *   arrow_fs          - PyArrow FileSystem
 *   existing_puffin_loc - path to existing puffin for append (str, default "")
 *
 * Returns: a StatisticsFile PyObject on success, NULL on error.
 */
PyObject *merge_and_write_puffin_py_entrypt(PyObject *self, PyObject *args) {
    PyObject *serialized_list = nullptr;
    const char *puffin_file_loc = nullptr;
    const char *bucket_region = nullptr;
    int64_t snapshot_id = 0;
    int64_t sequence_number = 0;
    PyObject *iceberg_schema_py = nullptr;
    PyObject *pyarrow_fs_py = nullptr;
    const char *existing_puffin_loc = "";

    if (!PyArg_ParseTuple(args, "OssllOOz", &serialized_list, &puffin_file_loc,
                          &bucket_region, &snapshot_id, &sequence_number,
                          &iceberg_schema_py, &pyarrow_fs_py,
                          &existing_puffin_loc)) {
        return nullptr;
    }

    if (!PyList_Check(serialized_list)) {
        PyErr_SetString(PyExc_TypeError,
                        "serialized_list must be a list of bytes objects");
        return nullptr;
    }

    return bodo_theta_utils_merge_and_write_puffin(
        serialized_list, puffin_file_loc, bucket_region, snapshot_id,
        sequence_number, iceberg_schema_py, pyarrow_fs_py, existing_puffin_loc);
}

} /* extern "C" */

/**
 * @brief Get the Cython-generated plan_optimizer module, which requires special
 * initialization.
 *
 * @return PyObject* plan_optimizer module object or nullptr on failure.
 */
PyObject *get_plan_optimizer_module() {
    // Cython uses multi-phase initialization which needs
    // PyModule_FromDefAndSpec(). See:
    // https://docs.python.org/3/c-api/module.html#c.PyModuleDef
    PyModuleDef *moddef = (PyModuleDef *)PyInit_plan_optimizer();

    PyObject *machinery = PyImport_ImportModule("importlib.machinery");
    if (!machinery) {
        PyErr_Print();
        return nullptr;
    }

    PyObject *module_spec_cls = PyObject_GetAttrString(machinery, "ModuleSpec");
    Py_DECREF(machinery);
    if (!module_spec_cls) {
        PyErr_Print();
        return nullptr;
    }

    PyObject *args = Py_BuildValue("sO", "plan_optimizer", Py_None);
    if (!args) {
        PyErr_Print();
        Py_DECREF(module_spec_cls);
        return nullptr;
    }

    PyObject *spec = PyObject_CallObject(module_spec_cls, args);
    Py_DECREF(module_spec_cls);
    Py_DECREF(args);
    if (!spec) {
        PyErr_Print();
        return nullptr;
    }

    PyObject *mod = PyModule_FromDefAndSpec(moddef, spec);
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

static PyMethodDef ext_methods[] = {
    {"delete_sketches_py_entrypt", delete_sketches_py_entrypt, METH_VARARGS,
     "Delete an UpdateSketchCollection"},
    {"compact_sketches_py_entrypt", compact_sketches_py_entrypt, METH_VARARGS,
     "Compact and serialize theta sketches to bytes"},
    {"merge_and_write_puffin_py_entrypt", merge_and_write_puffin_py_entrypt,
     METH_VARARGS, "Merge serialized sketches and write puffin file"},
    {nullptr, nullptr, 0, nullptr},
};

extern "C" {

PyMODINIT_FUNC PyInit_ext(void) {
    PyObject *m;
    MOD_DEF(m, "ext", "No docs", ext_methods);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromPyInit(m, hdist);
    SetAttrStringFromPyInit(m, hstr_ext);
    SetAttrStringFromPyInit(m, decimal_ext);
    SetAttrStringFromPyInit(m, quantile_alg);
    SetAttrStringFromPyInit(m, lateral_cpp);
    SetAttrStringFromPyInit(m, theta_sketches);
    SetAttrStringFromPyInit(m, puffin_file);
    SetAttrStringFromPyInit(m, lead_lag);
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
    SetAttrStringFromPyInit(m, stream_window_cpp);
    SetAttrStringFromPyInit(m, stream_dict_encoding_cpp);
    SetAttrStringFromPyInit(m, stream_sort_cpp);
    SetAttrStringFromPyInit(m, listagg);
    SetAttrStringFromPyInit(m, table_builder_cpp);
    SetAttrStringFromPyInit(m, query_profile_collector_cpp);
    SetAttrStringFromPyInit(m, uuid_cpp);
#ifdef BUILD_WITH_V8
    SetAttrStringFromPyInit(m, javascript_udf_cpp);
#endif

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    // Setup the Cython-generated plan_optimizer module
    PyObject *plan_opt_mod = get_plan_optimizer_module();
    if (!plan_opt_mod) {
        PyErr_Print();
        return nullptr;
    }
    if (PyObject_SetAttrString(m, "plan_optimizer", plan_opt_mod) < 0) {
        PyErr_Print();
        Py_DECREF(plan_opt_mod);
        return nullptr;
    }
    Py_DECREF(plan_opt_mod);

    return m;
}

} /* extern "C" */
