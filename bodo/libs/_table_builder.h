#pragma once
#include "_chunked_table_builder.h"
#include "_dict_builder.h"

/**
 * @brief Wrapper around table_info to turn it into build buffer.
 * It allows appending rows while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct TableBuildBuffer {
    // internal data table with values
    std::shared_ptr<table_info> data_table;
    // buffer wrappers around arrays of data table
    std::vector<ArrayBuildBuffer> array_buffers;
    /// @brief Whether the table is currently pinned.
    bool pinned_ = true;

    // Only used for temporary objects. In particular,
    // in HashJoinState constructor, we cannot initialize
    // the shuffle buffers in the initialization list since
    // we need to build the dict_builders first. So we need
    // to provide this default constructor so that it
    // is initialized to an empty buffer by default and then
    // we can create and replace it with the actual TableBuildBuffer
    // later in the constructor.
    TableBuildBuffer() = default;

    /**
     * @brief Constructor for a TableBuildBuffer.
     *
     * @param schema Schema of the table
     * @param dict_builders DictBuilders for the columns.
     * Element corresponding to a column must be provided in the
     * DICT array case and should be nullptr otherwise.
     * @param pool IBufferPool to use for allocating the underlying data
     * buffers.
     * @param mm MemoryManager for the 'pool'.
     */
    TableBuildBuffer(
        const std::shared_ptr<bodo::Schema>& schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    size_t EstimatedSize() const;

    void UnifyTablesAndAppend(
        const std::shared_ptr<table_info>& in_table,
        std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders);

    /**
     * @brief Append a batch of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table Input table with the new rows
     * @param append_rows Bit vector indicating which rows to append
     * @param append_rows_sum Total number of rows to append. This is just the
     * number of 'true' values in append_rows.
     */
    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum);
    /**
     * @brief Append a batch of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new rows
     * @param append_rows bit vector indicating which rows to append
     */
    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table,
                           const std::vector<bool>& append_rows);

    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append key columns of a row of input table, assuming there is
     * already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new row
     * @param row_ind index of new row in input table
     * @param n_keys number of key columns
     */
    void AppendRowKeys(const std::shared_ptr<table_info>& in_table,
                       int64_t row_ind, uint64_t n_keys);

    /**
     * @brief increment size of data columns by one to allow appending a new
     * data row
     *
     * @param n_keys number of key columns
     */
    void IncrementSizeDataColumns(uint64_t n_keys);

    /**
     * @brief increment size of all columns by addln_size.
     * NOTE: The table should have enough capacity already
     * reserved before this call.
     */
    void IncrementSize(size_t new_size);

    /**
     * @brief Reserve enough space to potentially append rows from
     * in_table (based on reserve_rows bitmap).
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to
     * reserve
     * @param reserve_rows bit vector indicating which rows to reserve
     * @param reserve_rows_sum Total number of rows. This is just the number of
     * 'true' values in reserve_rows.
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table,
                      const std::vector<bool>& reserve_rows,
                      uint64_t reserve_rows_sum);

    /**
     * @brief Reserve enough space to potentially append all columns of
     * input table to buffer (the rows specified using the 'reserve_rows'
     * bitmap).
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to
     * reserve
     * @param reserve_rows bit vector indicating which rows to reserve
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table,
                      const std::vector<bool>& reserve_rows);

    void ReserveTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Reserve enough space to append row_idx'th row on in_table to the
     * buffer. This includes reserving space for variable-sized elements like
     * strings.
     *
     * @param in_table Input table used for finding new buffer sizes to
     * reserve.
     * @param row_idx Index of the row to reserve space for.
     */
    void ReserveTableRow(const std::shared_ptr<table_info>& in_table,
                         size_t row_idx);

    /**
     * @brief Reserve enough space to hold size + new_data_len elements
     */
    void ReserveTableSize(const size_t new_data_len);

    /**
     * @brief Reserve enough space to be able to append all the finalized chunks
     * of a ChunkedTableBuilder.
     * NOTE: This requires reserving space for variable-sized elements like
     * strings and nested arrays.
     *
     * @param chunked_tb ChunkedTableBuilder whose chunks we want to append
     * to this TableBuildBuffer.
     */
    void ReserveTable(const ChunkedTableBuilder& chunked_tb);

    /**
     * @brief Reserve enough space to be able to append all the chunks in the
     * input vector.
     * NOTE: This requires reserving space for variable-sized elements like
     * strings and nested arrays.
     *
     * @param chunks tables to append
     * @param input_is_unpinned Are the chunks unpinned and need to be pinned in
     * case we need to get size information.
     */
    void ReserveTable(const std::vector<std::shared_ptr<table_info>>& chunks,
                      const bool input_is_unpinned = false);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionaries are reset to
     * use the original dictionary builders provided during
     * creation.
     */
    void Reset();

    /**
     * @brief Pin this table buffer. This is idempotent.
     * Currently, this simply calls 'pin' on the underlying
     * table_info.
     *
     */
    void pin();

    /**
     * @brief Unpin this table buffer. This is idempotent.
     * Currently, this simply calls 'unpin' on the underlying
     * table_info.
     *
     */
    void unpin();
};

struct ChunkedTableBuilderState {
    const std::shared_ptr<bodo::Schema> table_schema;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::unique_ptr<ChunkedTableBuilder> builder;

    ChunkedTableBuilderState(const std::shared_ptr<bodo::Schema> table_schema_,
                             size_t chunk_size)
        : table_schema(std::move(table_schema_)) {
        // Create dictionary builders for all columns
        for (const std::unique_ptr<bodo::DataType>& t :
             table_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(t->copy(), false));
        }
        builder = std::make_unique<ChunkedTableBuilder>(ChunkedTableBuilder(
            table_schema, dict_builders, chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES));
    }
};
