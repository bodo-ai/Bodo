#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <tuple>

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
