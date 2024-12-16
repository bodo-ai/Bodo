#pragma once
#include <arrow/api.h>

// Copied from Arrow:
// https://github.com/apache/arrow/blob/a61f4af724cd06c3a9b4abd20491345997e532c0/cpp/src/parquet/arrow/schema.cc#L243
// Must match ICEBERG_FIELD_ID_MD_KEY in bodo/io/iceberg.py
// and bodo_iceberg_connector/schema_helper.py
static constexpr char ICEBERG_FIELD_ID_MD_KEY[] = "PARQUET:field_id";

/**
 * @brief Get the Iceberg Field ID from an Arrow field.
 *
 * @param field Arrow field to get the Iceberg field ID from.
 * @return int Iceberg Field ID extracted from the Arrow field.
 */
int get_iceberg_field_id(const std::shared_ptr<arrow::Field>& field);
