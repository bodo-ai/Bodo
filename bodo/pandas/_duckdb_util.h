#pragma once

#include <cstdint>
#include <variant>
#include "duckdb/common/types/value.hpp"
#include <arrow/api.h>

/**
 * @brief Convert duckdb value to C++ variant.
 *
 * @param expr - the duckdb value to convert
 * @return the C++ variant converted value
 */
std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double, arrow::TimestampScalar>
extractValue(const duckdb::Value& value);
