#include "_bodo_common.h"
#include "_table_builder.h"

#define SORT_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

#define SORT_OPERATOR_MAX_CHUNK_NUMBER 100

/**
 * @brief A wrapper around a sorted table_info that stores the min and max rows.
 * It is assumed that range is always pinned, and that table is usually
 * unpinned. This allows for fast sorting of chunks without unpinning.
 *
 */
struct TableAndRange {
    // The data table - assumed to be sorted. Can be pinned or unpinned.
    std::shared_ptr<table_info> table;
    // table with only 2 rows for the min and max of table - always pinned
    std::shared_ptr<table_info> range;
    // Offset of the first row in the range table. This is to represent tables
    // where we only want the suffix of the table, and the first row of range
    // (the min) is actully the `offset`th row of the table.
    int64_t offset;

    TableAndRange(std::shared_ptr<table_info> table, int64_t n_key_t,
                  int64_t offset = 0);

    /**
     * Update the offset into the table and adjust the range accordingly
     */
    void UpdateOffset(int64_t n_key_t, int64_t offset);

    /**
     * @brief For debugging purpose
     */
    friend std::ostream& operator<<(std::ostream& os, const TableAndRange& obj);
};

struct HeapComparator {
    int64_t n_key_t;
    // TODO(aneesh) can these be vectors instead?
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    // TODO(aneesh) we should either templetize KeyComparisonAsPython or move to
    // something like converting rows to bitstrings for faster comparision.
    bool operator()(const TableAndRange& a, const TableAndRange& b) {
        return !KeyComparisonAsPython(n_key_t, vect_ascending.data(),
                                      a.range->columns, 0, 0, b.range->columns,
                                      0, 0, na_position.data());
    }
};

struct SortedChunkedTableBuilder {
    const int64_t n_key_t;
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    size_t chunk_size;
    const size_t num_chunks;

    HeapComparator comp;

    std::vector<TableAndRange> table_heap;

    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::unique_ptr<ChunkedTableBuilder> sorted_table_builder;

    bodo::IBufferPool* pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    SortedChunkedTableBuilder(
        int64_t n_key_t, const std::vector<int64_t>& vect_ascending,
        const std::vector<int64_t>& na_position,
        const std::vector<int64_t>& dead_keys, size_t num_chunks_,
        size_t chunk_size = 4096,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager())
        : n_key_t(n_key_t),
          vect_ascending(vect_ascending),
          na_position(na_position),
          dead_keys(dead_keys),
          chunk_size(chunk_size),
          num_chunks(num_chunks_),
          comp({n_key_t, vect_ascending, na_position, dead_keys}),
          pool(pool),
          mm(mm) {}

    /**
     * @brief Update chunk_size with ChunkSize
     *
     * @param ChunkSize New chunk size
     */
    void UpdateChunkSize(size_t ChunkSize) { chunk_size = ChunkSize; }

    /**
     * @brief Initialize sorted_table_builder
     *
     * @param schema Table schema
     */
    void InitCTB(std::shared_ptr<bodo::Schema> schema);

    /**
     * Append an table to the builder, sorting it if required.
     *
     * @param chunk Table to append. Must be pinned.
     * @param sorted should be true if the input chunk is already sorted and can
     * be appended directly.
     */
    void AppendChunk(std::shared_ptr<table_info> chunk, bool sorted = false);

    /**
     * Sort all chunks into a list of sorted chunks. E.g. if we were do
     * the following with a chunksize of 3:
     *   AppendChunk([0, 5, 3])
     *   AppendChunk([4, 2, 1])
     *   Finalize()
     *     = [[0, 1, 2], [3, 4, 5]]
     * Note that in the example above the inputs appear to be vectors, but
     * inputs/output are all actually tables. The output chunks will be wrapped
     * in a TableAndRange and every table will be unpinned.
     */
    std::vector<TableAndRange> Finalize();

    /**
     * Merge a list of sorted lists to produce a single sorted list. E.g.
     *   chunk_list_a = [[1, 3, 5], [7, 9, 11]]
     *   chunk_list_b = [[2, 4, 6], [8, 10, 12]]
     *   MergeChunks([chunk_list_a, chunk_list_b])
     *     = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
     * All tables in the output will be unpinned.
     */
    std::vector<TableAndRange> MergeChunks(
        std::vector<std::vector<TableAndRange>>&& sorted_chunks);
};

struct StreamSortMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    // consume metrics
    time_t local_sort_chunk_time;
    time_t local_sort_time;

    // produce metrics
    time_t sampling_time;
    time_t partition_chunks_time;
    time_t communication_phase;
    time_t global_append_chunk_time;
    time_t global_sort_time;

    int64_t output_row_count;
};

enum class StreamSortPhase {
    INIT,
    BUILD,
    GLOBAL_SORT,
    PRODUCE_OUTPUT,
    INVALID
};

struct StreamSortState {
    int64_t op_id;
    int64_t n_key_t;
    std::vector<int64_t> vect_ascending;
    std::vector<int64_t> na_position;
    std::vector<int64_t> dead_keys;
    bool parallel = true;

    uint64_t mem_budget_bytes;
    size_t num_chunks;
    std::pair<int64_t, int64_t> row_info;
    bodo::IBufferPool* op_pool;
    // Memory manager instance for op_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;

    // builder will create a sorted list from the chunks passed to consume batch
    SortedChunkedTableBuilder builder;
    // output from the sorted builder
    std::vector<TableAndRange> local_chunks;

    // Index of chunk to yield
    size_t output_idx = 0;
    // List of chunks ready to yield
    std::vector<std::shared_ptr<table_info>> output_chunks;

    // We need to track the phase of the algorithm to know which objects we need
    // to initialize
    StreamSortPhase phase = StreamSortPhase::INIT;

    // Type of input table
    std::shared_ptr<bodo::Schema> schema;
    // Empty table created with the same schema as the input
    std::shared_ptr<table_info> dummy_output_chunk;

    /**
     * @brief Helper function to get operator budget from op_id
     *
     * @param op_id operator ID
     * @return Operator budget. If unset (value of -1) will get all available
     * budgets
     */
    uint64_t GetBudget(int64_t op_id) const;

    // Metrics for sorting
    StreamSortMetrics metrics;

    StreamSortState(int64_t op_id, int64_t n_key_t,
                    std::vector<int64_t>&& vect_ascending_,
                    std::vector<int64_t>&& na_position_,
                    std::shared_ptr<bodo::Schema> schema, bool parallel = true,
                    size_t chunk_size = 4096);
    /**
     * @brief Consume an unsorted table and use it for global sorting
     *
     * @param table unsorted input table
     * @param parallel whether or not multiple ranks are participating in the
     * sort
     * @param is_last if this table is guaranteed to be the last input this
     * state will see.
     * @return boolean indicating if the consume phase is complete
     */
    void ConsumeBatch(std::shared_ptr<table_info> table, bool parallel,
                      bool is_last);

    /**
     * Get bounds for parallel sorting based on the chunks consumed so far.
     * These bounds determine which elements are assigned to which ranks.
     */
    std::shared_ptr<table_info> GetParallelSortBounds();

    /**
     * Sort all chunks passed to ConsumeBatch across all ranks
     */
    void GlobalSort();

    /**
     * Get the next sorted output chunk. Returns a pair of table and a boolean
     * indicating if the last chunk has been reached
     */
    std::pair<std::shared_ptr<table_info>, bool> GetOutput();

    // Helper methods

    // Partition all sorted input chunks a list of chunks per rank
    // All of the returned data needs to be communicated.
    std::vector<std::vector<std::shared_ptr<table_info>>> PartitionChunksByRank(
        int n_pes, std::shared_ptr<table_info> bounds);

    // Report all metrics
    void ReportMetrics();
};
