#include <Python.h>
#include <listobject.h>

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

static PyMethodDef theta_sketches_entry_methods[] = {
    {"delete_sketches_py_entrypt", delete_sketches_py_entrypt, METH_VARARGS,
     "Delete an UpdateSketchCollection"},
    {"compact_sketches_py_entrypt", compact_sketches_py_entrypt, METH_VARARGS,
     "Compact and serialize theta sketches to bytes"},
    {"merge_and_write_puffin_py_entrypt", merge_and_write_puffin_py_entrypt,
     METH_VARARGS, "Merge serialized sketches and write puffin file"},
    {nullptr, nullptr, 0, nullptr},
};

PyMODINIT_FUNC PyInit_theta_sketches_entry(void) {
    PyObject *m;
    MOD_DEF(m, "theta_sketches_entry", "No docs", theta_sketches_entry_methods);
    if (m == nullptr) {
        return nullptr;
    }
    bodo_common_init();
    return m;
}
