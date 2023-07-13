// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Function to convert Bodo arrays to Arrow arrays

#pragma once

#include "_bodo_common.h"
#include "_datetime_utils.h"

#if _MSC_VER >= 1900
#undef timezone
#endif

std::shared_ptr<arrow::DataType> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const std::shared_ptr<array_info> array,
    std::shared_ptr<arrow::Array> *out, bool convert_timedelta_to_int64,
    const std::string &tz, arrow::TimeUnit::type &time_unit,
    bool downcast_time_ns_to_us);

/**
 * @brief convert Bodo table to Arrow table
 *
 * @param table input Bodo table
 * @return std::shared_ptr<arrow::Table> Arrow table
 */
std::shared_ptr<arrow::Table> bodo_table_to_arrow(
    std::shared_ptr<table_info> table);

/**
 * @brief Convert Arrow array to Bodo array_info with zero-copy.
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted. Currently, only string, dict-encoded string,
 * and numeric arrays are supported.
 *
 * @param arrow_arr Input Arrow array
 * @param array_id Identifier for an array to say two arrays are equivalent.
 * Currently only used for string arrays that function as dictionaries.
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_array_to_bodo(
    std::shared_ptr<arrow::Array> arrow_arr, int64_t array_id = -1);
