// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Function to convert Bodo arrays to Arrow arrays

#pragma once

#include "_bodo_common.h"
#include "_datetime_utils.h"

#if _MSC_VER >= 1900
#undef timezone
#endif

void bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    const std::string &col_name,
    std::vector<std::shared_ptr<arrow::Field>> &schema_vector,
    std::shared_ptr<arrow::ChunkedArray> *out, const std::string &tz,
    arrow::TimeUnit::type &time_unit, bool copy);
