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
