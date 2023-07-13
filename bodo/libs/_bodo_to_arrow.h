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
 * decrefs/deallocates if necessary to Bodo's BufferPool.
 * Alternative to Arrow's PoolBuffer:
 * https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L838
 */
class BodoBuffer : public arrow::ResizableBuffer {
   public:
    /**
     * @brief Construct a new Bodo Buffer
     *
     * @param data Pointer to data buffer
     * @param size Size of the buffer
     * @param meminfo_ MemInfo object which manages the underlying data buffer
     * @param incref Whether to incref (default: true)
     * @param pool Pool that was used for allocating the data buffer. The same
     * pool should be used for resizing the buffer in the future.
     * @param mm Memory manager associated with the pool.
     */
    BodoBuffer(uint8_t *data, const int64_t size, NRT_MemInfo *meminfo_,
               bool incref = true,
               bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
               std::shared_ptr<::arrow::MemoryManager> mm =
                   bodo::default_buffer_memory_manager())
        : ResizableBuffer(data, size, std::move(mm)),
          meminfo(meminfo_),
          pool_(pool) {
        if (incref) {
            incref_meminfo(meminfo);
        }
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

    NRT_MemInfo *getMeminfo() { return meminfo; }

    /**
     * @brief Ensure that buffer can accommodate specified total number of bytes
     * without re-allocation.
     *
     * Copied from Arrow's PoolBuffer since it is not exposed to use as base
     * class:
     * https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/memory_pool.cc
     * Bodo change: alignment is not passed to alloc calls which defaults to
     * 64-byte. Meminfo attributes are also updated.
     *
     * @param capacity minimum total capacity required in bytes
     * @return arrow::Status
     */
    arrow::Status Reserve(const int64_t capacity) override {
        if (capacity < 0) {
            return arrow::Status::Invalid("Negative buffer capacity: ",
                                          capacity);
        }
        if (!data_ || capacity > capacity_) {
            int64_t new_capacity =
                arrow::bit_util::RoundUpToMultipleOf64(capacity);
            if (data_) {
                // NOTE: meminfo->data needs to be passed to buffer pool manager
                // since it stores swips
                RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity,
                                                (uint8_t **)&(meminfo->data)));
            } else {
                RETURN_NOT_OK(pool_->Allocate(new_capacity,
                                              (uint8_t **)&(meminfo->data)));
            }
            data_ = (uint8_t *)meminfo->data;
            capacity_ = new_capacity;
            // Updating size is necessary since used by the buffer pool manager
            // for freeing.
            meminfo->size = new_capacity;
        }
        return arrow::Status::OK();
    }

    /**
     * @brief Make sure there is enough capacity and resize the buffer.
     * If shrink_to_fit=true and new_size is smaller than existing size, updates
     * capacity to the nearest multiple of 64 bytes.
     *
     * Copied from Arrow's PoolBuffer since it is not exposed to use as base
     * class:
     * https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/memory_pool.cc
     * Bodo change: alignment is not passed to alloc calls which defaults to
     * 64-byte. Meminfo attributes are also updated.
     *
     * @param new_size New size of the buffer
     * @param shrink_to_fit if new size is smaller than the existing size,
     * reallocate to fit.
     * @return arrow::Status
     */
    arrow::Status Resize(const int64_t new_size,
                         bool shrink_to_fit = true) override {
        if (ARROW_PREDICT_FALSE(new_size < 0)) {
            return arrow::Status::Invalid("Negative buffer resize: ", new_size);
        }
        if (data_ && shrink_to_fit && new_size <= size_) {
            // Buffer is non-null and is not growing, so shrink to the requested
            // size without excess space.
            int64_t new_capacity =
                arrow::bit_util::RoundUpToMultipleOf64(new_size);
            if (capacity_ != new_capacity) {
                // Buffer hasn't got yet the requested size.
                RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity,
                                                (uint8_t **)&(meminfo->data)));
                data_ = (uint8_t *)meminfo->data;
                capacity_ = new_capacity;
                // Updating size is necessary since used by the buffer pool
                // manager for freeing.
                meminfo->size = new_capacity;
            }
        } else {
            RETURN_NOT_OK(Reserve(new_size));
        }
        size_ = new_size;

        return arrow::Status::OK();
    }

   private:
    NRT_MemInfo *meminfo;
    arrow::MemoryPool *pool_;
};

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
