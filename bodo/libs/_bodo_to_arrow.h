// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Function to convert Bodo arrays to Arrow arrays

#pragma once

#include "_bodo_common.h"
#include "_datetime_utils.h"

#if _MSC_VER >= 1900
#undef timezone
#endif

/**
 * @brief Arrow buffer that holds a reference to a Bodo meminfo and
 * decrefs/deallocates if necessary. Alternative to Arrow's PoolBuffer:
 * https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L838
 */
class BodoBuffer : public arrow::MutableBuffer {
   public:
    BodoBuffer(uint8_t *data, const int64_t size, NRT_MemInfo *meminfo_)
        : MutableBuffer(data, size), meminfo(meminfo_) {
        incref_meminfo(meminfo);
    }

    ~BodoBuffer() override {
        // Adapted from:
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L844
        uint8_t *ptr = mutable_data();
        // TODO(ehsan): add global_state.is_finalizing() check to match Arrow
        if (ptr) {
            decref_meminfo(meminfo);
        }
    }

   private:
    NRT_MemInfo *meminfo;
};

std::shared_ptr<arrow::DataType> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    std::shared_ptr<arrow::Array> *out, bool convert_timedelta_to_int64,
    const std::string &tz, arrow::TimeUnit::type &time_unit, bool copy,
    bool downcast_time_ns_to_us);
/**
 * @brief convert Bodo table to Arrow table
 *
 * @param table input Bodo table
 * @return std::shared_ptr<arrow::Table> Arrow table
 */
std::shared_ptr<arrow::Table> bodo_table_to_arrow(table_info *table);

/**
 * @brief convert Arrow array to Bodo array_info with zero-copy.
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted. Currently, only string arrays are supported.
 *
 * @param arrow_arr input Arrow array (string array currently)
 * @return array_info* output Bodo array
 */
array_info *arrow_array_to_bodo(std::shared_ptr<arrow::Array> arrow_arr);
