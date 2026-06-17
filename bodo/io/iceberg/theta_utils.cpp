/**
 * @brief Plain C functions for the Iceberg theta sketch lifecycle, callable
 * from Python via ctypes. Bridges between standard Python types and C++ theta
 * sketch / puffin file infrastructure.
 */

#include <Python.h>
#include <cstdint>

#include "../libs/_puffin.h"
#include "../libs/_theta_sketches.h"

#define EXPORT __attribute__((visibility("default")))

extern "C" {

EXPORT void bodo_theta_utils_delete_sketches(uintptr_t ptr) {
    if (ptr == 0)
        return;
    try {
        delete reinterpret_cast<UpdateSketchCollection *>(ptr);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Write puffin file by delegating to write_puffin_file_py_entrypt.
 * The ptr is an UpdateSketchCollection* passed as uintptr_t.
 */
EXPORT PyObject *bodo_theta_utils_write_puffin(
    uintptr_t ptr, const char *puffin_file_loc, const char *bucket_region,
    int64_t snapshot_id, int64_t sequence_number, PyObject *iceberg_schema_py,
    PyObject *pyarrow_fs_py, const char *existing_puffin_loc) {
    if (ptr == 0) {
        Py_RETURN_NONE;
    }
    try {
        // Ensure we hold the GIL before calling into Python
        PyGILState_STATE gstate = PyGILState_Ensure();
        auto *sketches = reinterpret_cast<UpdateSketchCollection *>(ptr);
        PyObject *result = write_puffin_file_py_entrypt(
            puffin_file_loc, bucket_region, snapshot_id, sequence_number,
            sketches, iceberg_schema_py, pyarrow_fs_py, existing_puffin_loc);
        PyGILState_Release(gstate);
        return result;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

}  // extern "C"
