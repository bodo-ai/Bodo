// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Function to convert Bodo arrays to Arrow arrays

#pragma once

#include "_bodo_common.h"
#include "_datetime_utils.h"

#if _MSC_VER >= 1900
#undef timezone
#endif

std::shared_ptr<arrow::DataType> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    std::shared_ptr<arrow::Array> *out, const std::string &tz,
    arrow::TimeUnit::type &time_unit, bool copy);

/**
 * @brief convert Bodo table to Arrow table
 *
 * @param table input Bodo table
 * @return std::shared_ptr<arrow::Table> Arrow table
 */
std::shared_ptr<arrow::Table> bodo_table_to_arrow(table_info *table);
