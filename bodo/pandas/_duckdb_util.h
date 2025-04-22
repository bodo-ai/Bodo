#pragma once

#include <variant>
#include <cstdint>
#include "duckdb/common/types/value.hpp"

/**
 * @brief Convert duckdb value to C++ variant.
 *
 * @param expr - the duckdb value to convert
 * @return the C++ variant converted value
 */
std::variant<int8_t, int16_t, int32_t, int64_t, float, double> extractValue(const duckdb::Value& value);

