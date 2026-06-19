/**
 * @brief Functions for the Iceberg theta sketch lifecycle in the DataFrame
 * library path.
 *
 * For the DataFrame library (df_lib) path, we cannot call
 * write_puffin_file_py_entrypt directly because it internally calls
 * merge_parallel_sketches() which does MPI gather — but in the df_lib path
 * only the spawner (rank 0) calls it after workers have finished.
 *
 * Instead, the flow is:
 * 1. Each rank compacts its UpdateSketchCollection and serializes it to bytes
 *    via bodo_theta_utils_compact_serialize().
 * 2. The serialized bytes are gathered to rank 0 via Python/MPI.
 * 3. Rank 0 calls bodo_theta_utils_merge_and_write_puffin() which deserializes,
 *    merges (non-MPI), and writes the puffin file.
 */

#include <Python.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../libs/_puffin.h"
#include "../libs/_theta_sketches.h"

#define EXPORT __attribute__((visibility("default")))

extern "C" {

/**
 * @brief Delete an UpdateSketchCollection.
 */
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
 * @brief Compact an UpdateSketchCollection and serialize the result to a
 * Python bytes object.  This is a local (non-MPI) operation.
 *
 * Wire format (all little-endian):
 *   uint32  n_sketches
 *   For each sketch i in [0, n_sketches):
 *     uint32  len_i          (0 means absent/null sketch)
 *     byte[]  serialized_i   (len_i bytes)
 *
 * @return A Python bytes object, or NULL on error.
 */
EXPORT PyObject *bodo_theta_utils_compact_serialize(uintptr_t ptr) {
    if (ptr == 0) {
        Py_RETURN_NONE;
    }
    try {
        auto *sketches = reinterpret_cast<UpdateSketchCollection *>(ptr);
        std::shared_ptr<CompactSketchCollection> compact =
            sketches->compact_sketches();
        std::vector<std::optional<std::string>> serialized =
            compact->serialize_sketches();

        // Compute total size
        uint32_t n_sketches = static_cast<uint32_t>(serialized.size());
        size_t total = sizeof(uint32_t);  // n_sketches header
        for (auto &s : serialized) {
            total += sizeof(uint32_t);  // length field
            if (s.has_value()) {
                total += s.value().size();
            }
        }

        // Build Python bytes
        PyObject *bytes_obj = PyBytes_FromStringAndSize(nullptr, total);
        if (!bytes_obj)
            return nullptr;
        char *buf = PyBytes_AS_STRING(bytes_obj);
        size_t offset = 0;

        memcpy(buf + offset, &n_sketches, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        for (auto &s : serialized) {
            uint32_t len =
                s.has_value() ? static_cast<uint32_t>(s.value().size()) : 0;
            memcpy(buf + offset, &len, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            if (len > 0) {
                memcpy(buf + offset, s.value().data(), len);
                offset += len;
            }
        }
        return bytes_obj;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Merge pre-serialized CompactSketchCollections and write the puffin
 * file.  This is a rank-0-only, non-MPI operation.
 *
 * @param serialized_list  A Python list of bytes objects, one per rank.
 * @param puffin_file_loc  Output path for the puffin file.
 * @param bucket_region    S3 bucket region (empty for local).
 * @param snapshot_id      Iceberg snapshot ID.
 * @param sequence_number  Iceberg sequence number.
 * @param iceberg_schema_py  PyArrow schema as a PyObject*.
 * @param pyarrow_fs_py    PyArrow FileSystem as a PyObject*.
 * @param existing_puffin_loc  Path to existing puffin file for append (empty
 *                             string for new writes).
 * @return A StatisticsFile PyObject* on success, NULL on error.
 */
EXPORT PyObject *bodo_theta_utils_merge_and_write_puffin(
    PyObject *serialized_list, const char *puffin_file_loc,
    const char *bucket_region, int64_t snapshot_id, int64_t sequence_number,
    PyObject *iceberg_schema_py, PyObject *pyarrow_fs_py,
    const char *existing_puffin_loc) {
    try {
        Py_ssize_t n_ranks = PyList_GET_SIZE(serialized_list);
        std::vector<std::shared_ptr<CompactSketchCollection>> collections;
        collections.reserve(n_ranks);

        for (Py_ssize_t r = 0; r < n_ranks; r++) {
            PyObject *item = PyList_GET_ITEM(serialized_list, r);
            if (item == Py_None || !PyBytes_Check(item)) {
                // Empty/absent collection for this rank
                std::vector<std::optional<datasketches::compact_theta_sketch>>
                    empty;
                collections.push_back(std::make_shared<CompactSketchCollection>(
                    std::move(empty)));
                continue;
            }
            const char *data = PyBytes_AS_STRING(item);
            size_t buf_size = static_cast<size_t>(PyBytes_GET_SIZE(item));
            size_t offset = 0;

            // Validate we can read n_sketches header
            if (offset + sizeof(uint32_t) > buf_size) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Theta sketch serialized data too short for "
                    "header in bodo_theta_utils_merge_and_write_puffin");
                return nullptr;
            }
            uint32_t n_sketches = 0;
            memcpy(&n_sketches, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);

            std::vector<std::optional<std::string>> serialized_sketches;
            serialized_sketches.reserve(n_sketches);
            for (uint32_t i = 0; i < n_sketches; i++) {
                // Validate we can read length field
                if (offset + sizeof(uint32_t) > buf_size) {
                    PyErr_SetString(
                        PyExc_ValueError,
                        "Theta sketch serialized data too short for sketch "
                        "length in bodo_theta_utils_merge_and_write_puffin");
                    return nullptr;
                }
                uint32_t len = 0;
                memcpy(&len, data + offset, sizeof(uint32_t));
                offset += sizeof(uint32_t);
                if (len > 0) {
                    // Validate we have enough bytes for the sketch data
                    if (offset + len > buf_size) {
                        PyErr_SetString(
                            PyExc_ValueError,
                            "Theta sketch serialized data too short for sketch "
                            "data in bodo_theta_utils_merge_and_write_puffin");
                        return nullptr;
                    }
                    serialized_sketches.emplace_back(
                        std::string(data + offset, len));
                    offset += len;
                } else {
                    serialized_sketches.emplace_back(std::nullopt);
                }
            }
            collections.push_back(CompactSketchCollection::deserialize_sketches(
                serialized_sketches));
        }

        // Merge all collections (non-MPI)
        std::shared_ptr<CompactSketchCollection> merged =
            CompactSketchCollection::merge_sketches(std::move(collections));

        // Delegate to the existing write function (non-MPI version)
        // Ensure GIL is held for Python C API calls inside the write function
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject *result = write_puffin_from_compact_sketches_py_entrypt(
            puffin_file_loc, bucket_region, snapshot_id, sequence_number,
            merged, iceberg_schema_py, pyarrow_fs_py, existing_puffin_loc);
        PyGILState_Release(gstate);
        return result;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

} /* extern "C" */
