#pragma once

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_table_builder.h"

// Pre-allocate 32 bytes per string for now.
// Keep in sync with value in
// test_stream_join.py::test_long_strings_chunked_table_builder
#define CHUNKED_TABLE_DEFAULT_STRING_PREALLOCATION 32LL

/**
 * @brief Array Builder for Chunked Table Builder.
 * Tracks size and capacity of the buffers.
 * For fixed size data types, the required capacity is
 * reserved up front.
 * For strings, we allocate certain amount up front,
 * and then resize as needed/allowed.
 * See design doc here: https://bodo.atlassian.net/l/cp/mzidHW9G.
 *
 */
struct ChunkedTableArrayBuilder {
    // Internal array with data values
    std::shared_ptr<array_info> data_array;

    // Shared dictionary builder
    std::shared_ptr<DictionaryBuilder> dict_builder = nullptr;
    // Dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ChunkedTableArrayBuilder> dict_indices;

    // Current number of elements in the buffers.
    size_t size = 0;
    // Maximum number of rows this array is allowed to have.
    const size_t capacity;

    /* Only applicable to variable size data types like strings */

    // Number of times this array's buffers have been resized.
    size_t resize_count = 0;
    // Maximum number of times that this array's buffers can be
    // resized.
    const size_t max_resize_count = 2;

    // XXX In the future, we can generally keep track of the amount
    // of memory that has been allocated, etc. and use that to determine
    // when to stop building a chunk.

    /**
     * @brief Construct a new Chunked Table Array Builder object
     *
     * @param _data_array Underlying array_info into whose buffers we will
     * insert the data.
     * @param _dict_builder If this is a dictionary encoded string array,
     * a DictBuilder must be provided that will be used as the dictionary.
     * The dictionary of the data_array (_data_array->child_arrays[0]) must
     * be the dictionary in dict_builder (_dict_builder->dict_buff->data_array).
     * @param chunk_size Maximum number of rows this chunk is allowed to have.
     * @param max_resize_count Maximum number of times the buffers of this array
     * can be resized (grow by x2). This is only applicable to variable size
     * data types like strings. Even in those cases, only the variable sized
     * buffers (e.g. data1 for strings) will be resized.
     */
    ChunkedTableArrayBuilder(
        std::shared_ptr<array_info> _data_array,
        std::shared_ptr<DictionaryBuilder> _dict_builder = nullptr,
        size_t chunk_size = 8192, size_t max_resize_count = 2);

    /**
     * @brief Get the total number of bytes that all buffers
     * of this array have allocated (so far).
     *
     * @return size_t
     */
    size_t GetTotalBytes();

    /**
     * @brief Check if this array can resize (grow by x2).
     *
     * @return true If it has variable-size data (e.g. string) and resize_count
     * < max_resize_count.
     * @return false Otherwise.
     */
    bool CanResize();

    /**
     * @brief Check if the buffers of this array can accommodate row_ind'th row
     * in in_arr without needing to resize (in the variable size data type
     * case).
     * A row can only be appended if size < capacity. For fixed size data
     * types, this check is sufficient since we always allocate sufficient space
     * upfront.
     * For variable size data types like strings, this will check
     * if there's enough space in the buffers (without resizing).
     *
     * @param in_arr Array to insert row from.
     * @param row_ind Index of the row to insert.
     * @return true
     * @return false
     */
    bool CanAppendRowWithoutResizing(const std::shared_ptr<array_info>& in_arr,
                                     const int64_t row_ind);

    /**
     * @brief Check if the buffers of this array can accommodate row_ind'th row
     * in in_arr.
     * A row can only be appended if size < capacity. For fixed size data types,
     * this check is sufficient since we always allocate sufficient space
     * upfront. For variable size data types like strings, this will check
     * if there's enough space (including after allowed number of resizes) in
     * the buffers.
     *
     * @param in_arr Array to insert row from.
     * @param row_ind Index of the row to insert.
     */
    bool CanAppendRow(const std::shared_ptr<array_info>& in_arr,
                      const int64_t row_ind);

    /**
     * @brief Append row_ind'th row from in_arr into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * @param in_arr Array to insert row from.
     * @param row_ind Index of the row to insert.
     */
    void UnsafeAppendRow(const std::shared_ptr<array_info>& in_arr,
                         const int64_t row_ind);

    /**
     * @brief Append row_ind'th row from in_arr into this array safely.
     * This will handle resizing if required.
     * If this array cannot accommodate this row (including after resizing),
     * an error will be thrown.
     * XXX Explore using arrow::Status instead?
     *
     * @param in_arr Array to insert row from.
     * @param row_ind Index of the row in in_arr to insert.
     */
    void AppendRow(const std::shared_ptr<array_info>& in_arr,
                   const int64_t row_ind);

    /**
     * @brief Finalize this array. Once done, no more inserts are allowed (not
     * enforced). If shrink_to_fit == true, all buffers will be resized to
     * appropriate size based on the number of rows inserted. The minimum size
     * of any buffer in bytes is still the smallest block size in the Buffer
     * Pool.
     *
     * @param shrink_to_fit Whether to resize (i.e. shrink) buffers for
     * compactness.
     *
     */
    void Finalize(bool shrink_to_fit = true);
};

/**
 * @brief Chunked Table Builder for use cases like outputs
 * of streaming operators, etc.
 * Columnar table chunks (essentially PAX) are maintained such that
 * each chunk is of size at most 'active_chunk_capacity'
 * rows. The chunks are stored in a std::deque to provide
 * queue like behavior, while allowing iteration over the
 * chunks without removing them.
 * See design doc here: https://bodo.atlassian.net/l/cp/mzidHW9G.
 *
 */
struct ChunkedTableBuilder {
    // Queue of finalized chunks. We use a deque instead of
    // a regular queue since it gives us ability to both
    // iterate over elements as well as pop/push.
    std::deque<std::shared_ptr<table_info>> chunks;

    /* Active chunk state */

    // Active chunk
    std::shared_ptr<table_info> active_chunk;
    // Number of rows inserted into the active chunk
    size_t active_chunk_size = 0;
    // Max number of rows that can be inserted into the chunk
    const size_t active_chunk_capacity;
    // Keep a handle on the arrays in the table so we can
    // append and finalize them correctly.
    std::vector<ChunkedTableArrayBuilder> active_chunk_array_builders;
    // Maximum number of times that array build buffers are allowed to
    // grow by factor of 2x. Only applicable for variable size
    // data types like strings.
    const size_t max_resize_count_for_variable_size_dtypes;

    // Dummy output chunk that will be returned when there are
    // no more rows left in the buffer.
    std::shared_ptr<table_info> dummy_output_chunk;

    // XXX In the future, we can keep track of the amount
    // of memory that has been allocated, etc. and use that to determine
    // when to stop building the active chunk.

    /* Aggregate statistics */

    // Total rows inserted into the table across its lifetime.
    size_t total_size = 0;
    // Total rows that are in "unpopped" chunks (including the active chunk)
    size_t total_remaining = 0;

    // XXX In the future, we could keep track of the
    // allocated memory as well. This might be useful
    // for the build table buffers of inactive partitions
    // since we can use that information to decide
    // whether to re-partition or not.

    /**
     * @brief Construct a new Chunked Table with the given schema.
     *
     * @param arr_c_types Data types of the columns.
     * @param arr_array_types Array types of the columns.
     * @param dict_builders Dictionary builders to use for DICT arrays.
     * @param chunk_size Max number of rows allowed in each chunk.
     * @param max_resize_count_for_variable_size_dtypes How many times are
     * we allowed to resize (grow by 2x) buffers for variable size
     * data types like strings. 0 means resizing is not allowed. 2 means
     * that the final size could be 4x of the original.
     */
    ChunkedTableBuilder(
        const std::vector<int8_t>& arr_c_types,
        const std::vector<int8_t>& arr_array_types,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        size_t chunk_size, size_t max_resize_count_for_variable_size_dtypes);

    /**
     * @brief Finalize the active chunk and create a new active chunk.
     * This will call Finalize on all the arrays in this chunk, and then
     * insert the resulting table_info into this->chunks.
     * If the active chunk is empty, this will be a NOP.
     * In the future, it will also unpin the finalized chunk.
     *
     * @param shrink_to_fit Whether the array buffers should be
     * shrunk to fit the required capacity.
     *
     */
    void FinalizeActiveChunk(bool shrink_to_fit = true);

    /**
     * @brief Append a new row into this chunked table.
     * If any of the arrays are full, we will finalize
     * the active chunk and start a new chunk.
     *
     * @param in_table Table to insert row from.
     * @param row_ind Index of the row to insert.
     */
    void AppendRow(const std::shared_ptr<table_info>& in_table,
                   int64_t row_ind);

    /**
     * @brief Similar to AppendRow, except we can insert the
     * specified row indices.
     *
     * @param in_table The table to insert the rows from.
     * @param rowInds Vector of row indices to insert.
     */
    void AppendRows(std::shared_ptr<table_info> in_table,
                    const std::span<const int64_t> row_inds);

    /**
     * @brief Similar to AppendRow, but specifically for
     * Join Output Buffer use case. In join computation, we collect
     * the row indices from the build and probe tables which need
     * to be inserted together. Furthermore, due to column pruning,
     * we need to only append from required columns.
     *
     * @param build_table Build/Left table to insert rows from.
     * @param probe_table Probe/Right table to insert rows from.
     * @param build_idxs Indices from the build table.
     * @param probe_idxs Corresponding indices from the probe table.
     * @param build_kept_cols Indices of the columns from the build table.
     * @param probe_kept_cols Indices of the columns from the probe table.
     */
    void AppendJoinOutput(std::shared_ptr<table_info> build_table,
                          std::shared_ptr<table_info> probe_table,
                          const std::span<const int64_t> build_idxs,
                          const std::span<const int64_t> probe_idxs,
                          const std::vector<uint64_t>& build_kept_cols,
                          const std::vector<uint64_t>& probe_kept_cols);

    /**
     * @brief Finalize this chunked table. This will finalize
     * the active chunk. If the active chunk is empty, it will
     * be discarded and its memory will be freed.
     * No more rows should be appended once it is finalized
     * (not enforced at this point).
     *
     * @param shrink_to_fit If we finalize the active chunk, should
     * we shrink the buffers to fit the required size.
     */
    void Finalize(bool shrink_to_fit = true);

    /**
     * @brief Get the first available chunk. This will pop
     * an element from this->chunks.
     *
     * @param force_return If this->chunks is
     * empty, it will finalize and return the active chunk
     * if force_return=True (useful in the "is_last" case),
     * else it will return an empty table.
     * Note that it might return an empty table even in the
     * force_return case if the active chunk is empty.
     * @return std::tuple<std::shared_ptr<table_info>, int64_t> Tuple
     * of the chunk and the size of the chunk (in case all columns are dead).
     */
    std::tuple<std::shared_ptr<table_info>, int64_t> PopChunk(
        bool force_return = false);
};
