#pragma once

#include <Python.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <chrono>
#include <optional>
#include <string>
#include <tuple>
#include "_array_build_buffer.h"

/**
 * @brief Get the number of ranks on this node and the
 * current rank's position.
 *
 * We do this by creating sub-communicators based
 * on shared-memory. This is a collective operation
 * and therefore all ranks must call it.
 */
std::tuple<int, int> dist_get_ranks_on_node();

/**
 * @brief Get human readable string version for a given number of bytes.
 *
 * Shamelessly copied from DuckDB
 * (https://github.com/duckdb/duckdb/blob/6545a55cfe4a09af251826d4dd72980c424d9215/src/common/string_util.cpp#L157).
 * Modified it to use GiB, MiB, etc. instead of GB, MB, etc. since that's our
 * usual convention for calculations.
 *
 * @param bytes
 * @return std::string
 */
std::string BytesToHumanReadableString(const size_t bytes);

/// @brief Helper Function to Get the Current Time if `get` argument is true
std::optional<std::chrono::steady_clock::time_point> start_now(bool get);

/// @brief Get the amount of physically installed memory in bytes
uint64_t get_physically_installed_memory();

#ifndef __FILE_NAME__
// Windows does define __FILE_NAME__
#define __FILE_NAME__ __FILE__
#endif

// Helper macro to check for conditions. Unlike the builtin 'assert', this is
// executed even in release builds, so it should be used judiciously. It
// includes the line number and file name (basename) in the error message.
// NOTE: The calling file must include <fmt/format.h>.
#define ASSERT_WITH_ERR_MSG(x, ERR_MSG_PREFIX)                            \
    do {                                                                  \
        if (!(x)) {                                                       \
            std::string err_msg =                                         \
                fmt::format("{}Failed assertion '{}' on line {} in {}.",  \
                            ERR_MSG_PREFIX, #x, __LINE__, __FILE_NAME__); \
            throw std::runtime_error(err_msg);                            \
        }                                                                 \
    } while (0)

#define ASSERT(x) ASSERT_WITH_ERR_MSG(x, "")

/// @brief Class to manage PyObject pointers with automatic reference counting.
class PyObjectPtr : public std::unique_ptr<PyObject, void (*)(PyObject*)> {
   public:
    static void decref_check_none(PyObject* obj) {
        // Python 3.12 allows Py_XDECREF to be used on nullptr and all decrefs
        // have no effect on immortals (None becomes an immortal in 3.12).
        // Once python 3.12 is our min version we can just use Py_XDECREF
        // and remove this function.
        if (obj != nullptr && obj != Py_None) {
            Py_DECREF(obj);
        }
    }

    PyObjectPtr(PyObject* obj)
        : std::unique_ptr<PyObject, void (*)(PyObject*)>(
              obj, &(this->decref_check_none)) {}
    operator PyObject*() const { return get(); }
};

// Serialize an Arrow Table to a byte buffer (IPC Stream format)
std::shared_ptr<arrow::Buffer> SerializeTableToIPC(
    const std::shared_ptr<arrow::Table>& table);

// Deserialize an IPC buffer back to an Arrow Table
std::shared_ptr<arrow::Table> DeserializeIPC(
    std::shared_ptr<arrow::Buffer> buffer);
