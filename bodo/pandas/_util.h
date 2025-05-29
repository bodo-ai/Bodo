#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <variant>
#include "duckdb/common/types/value.hpp"

/**
 * @brief Convert duckdb value to C++ variant.
 *
 * @param expr - the duckdb value to convert
 * @return the C++ variant converted value
 */
std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double, arrow::TimestampScalar>
extractValue(const duckdb::Value &value);

/**
 * @brief Return a string representation of the column names in the Arrow schema
 * for printing purposes (e.g. plan prints).
 *
 * @param arrow_schema input Arrow schema
 * @return std::string string representation of the column names
 */
std::string schemaColumnNamesToString(
    const std::shared_ptr<arrow::Schema> arrow_schema);

/**
 * @brief Dynamic cast of base pointer to derived pointer.
 *
 * @param base_ptr - the base pointer to cast from
 * @return a non-NULL pointer of the derived type if the cast is possible else
 *         NULL
 */
template <typename Derived, typename Base>
duckdb::unique_ptr<Derived> dynamic_cast_unique_ptr(
    duckdb::unique_ptr<Base> &&base_ptr) noexcept {
    // Perform dynamic_cast on the raw pointer
    if (Derived *derived_raw = dynamic_cast<Derived *>(base_ptr.get())) {
        // Release ownership from the base_ptr and transfer it to a new
        // unique_ptr
        base_ptr.release();  // Release the ownership of the raw pointer
        return duckdb::unique_ptr<Derived>(derived_raw);
    }
    // If the cast fails, return a nullptr unique_ptr
    return nullptr;
}

/**
 * @brief Initialize mapping of input column orders so that keys are in the
 * beginning of build/probe tables to match streaming join APIs. See
 * https://github.com/bodo-ai/Bodo/blob/905664de2c37741d804615cdbb3fb437621ff0bd/bodo/libs/streaming/join.py#L189
 * @param col_inds input mapping to fill
 * @param keys key column indices
 * @param ncols number of columns in the table
 */
void initInputColumnMapping(std::vector<int64_t> &col_inds,
                            std::vector<uint64_t> &keys, uint64_t ncols);
