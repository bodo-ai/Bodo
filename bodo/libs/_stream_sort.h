#include "_bodo_common.h"
#include "_stream_shuffle.h"
#include "_table_builder.h"

#define SORT_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

#define SORT_OPERATOR_MAX_CHUNK_NUMBER 100

#define SORT_SMALL_LIMIT_OFFSET_CAP 16384

/**
 * @brief A wrapper around recording limit/offset to be applied
 * during MergeChunks.
 * Always decrement offset because it comes before limit.
 *
 */
struct SortLimits {
    size_t limit;
    size_t offset;
    SortLimits(size_t limit_, size_t offset_)
        : limit(limit_), offset(offset_) {}
    size_t sum() const { return limit + offset; }
    // First decrement
    void operator-=(const size_t n) {
        if (n <= offset) {
            offset -= n;
        } else {
            limit -= n - offset;
            offset = 0;
        }
    }
};

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

/**
 * Similar to ChunkedTableBuilder but provides a stream of TableAndRange objects
 * instead of plain tables.
 *
 * Note that ChunkedTableAndRangeBuilder assumes that it's input is a stream of
 * sorted rows!
 */
struct ChunkedTableAndRangeBuilder : AbstractChunkedTableBuilder {
   public:
    // TODO(aneesh) override AppendBatch so that there's a debug assert that
    // input rows are sorted - unsorted rows will make the range unreliable

    /// Number of columns to consider as keys
    int64_t n_key;

    ChunkedTableAndRangeBuilder(
        int n_key, const std::shared_ptr<bodo::Schema>& schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        size_t chunk_size,
        size_t max_resize_count_for_variable_size_dtypes =
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager())
        : AbstractChunkedTableBuilder(schema, dict_builders, chunk_size,
                                      max_resize_count_for_variable_size_dtypes,
                                      pool, mm),
          n_key(n_key) {}

    // Queue of finalized chunks. We use a deque instead of
    // a regular queue since it gives us ability to both
    // iterate over elements as well as pop/push. Finalized chunks are unpinned.
    // If we want to access finalized chunks, we need to pin and unpin them
    // manually.
    std::deque<TableAndRange> chunks;

   private:
    size_t NumReadyChunks() final { return chunks.size(); }

    std::shared_ptr<table_info> PopFront() final {
        TableAndRange chunk = chunks.front();
        // Pin the table before returning it
        chunk.table->pin();
        chunks.pop_front();
        return chunk.table;
    }

    void PushActiveChunk() final {
        // Get the range before we unpin the table
        TableAndRange chunk{std::move(active_chunk), n_key};
        chunk.table->unpin();

        chunks.emplace_back(std::move(chunk));
    }

    void ResetInternal() final { chunks.clear(); }
};

// The row idx for the minimum row in a TableAndRange
#define RANGE_MIN 0
// The row idx for the maximum row in a TableAndRange
#define RANGE_MAX 1

/**
 * Builder that accepts a stream of tables and sorts all rows
 */
struct SortedChunkedTableBuilder {
    std::shared_ptr<bodo::Schema> schema;

    const int64_t n_key_t;
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    bool parallel = true;

    // Number of rows per chunk
    size_t chunk_size;
    // Maximum number of chunks that will be pinned simultaneously during
    // sort.
    const size_t num_chunks;

    // Total amount of memory budget. If 0, an unlimited budget will be assumed
    uint64_t mem_budget_bytes;

    // Whether we apply limit/offset during Finalize / MergeChunks
    std::optional<SortLimits> sortlimits = std::nullopt;

    /**
     * Comparator for creating a heap of TableAndRange objects sorted by the
     * minimum values in the range. Does not pin the underlying table.
     */
    struct HeapComparator {
        SortedChunkedTableBuilder& builder;

        bool operator()(const TableAndRange& a, const TableAndRange& b) const {
            // Returns true if a.range[MIN] >= b.range[MIN]
            return !builder.Compare(a.range, RANGE_MIN, b.range, RANGE_MIN);
        }
    } comp;

    // Holds unpinned references to the result of sorting chunks submitted to
    // AppendChunk. This will be consumed when Finalize is called.
    std::deque<TableAndRange> input_chunks;

    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders;

    bodo::IBufferPool* pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    SortedChunkedTableBuilder(
        std::shared_ptr<bodo::Schema> schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        int64_t n_key_t, const std::vector<int64_t>& vect_ascending,
        const std::vector<int64_t>& na_position,
        const std::vector<int64_t>& dead_keys, size_t num_chunks_,
        size_t chunk_size = 4096, uint64_t mem_budget_bytes = 0,
        int64_t limit = -1, int64_t offset = -1, bool parallel_ = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager())
        : schema(schema),
          n_key_t(n_key_t),
          vect_ascending(vect_ascending),
          na_position(na_position),
          dead_keys(dead_keys),
          parallel(parallel_),
          chunk_size(chunk_size),
          num_chunks(num_chunks_),
          mem_budget_bytes(mem_budget_bytes),
          comp(*this),
          dict_builders(dict_builders_),
          pool(pool),
          mm(mm) {
        // Either both limit and offset are -1, or they are both >= 0
        if (limit >= 0)
            sortlimits = std::make_optional(SortLimits(limit, offset));
    }

    // Return true if table1[row1] < table2[row2]
    bool Compare(std::shared_ptr<table_info> table1, size_t row1,
                 std::shared_ptr<table_info> table2, size_t row2) const;

    /**
     * @brief Update chunk_size with ChunkSize
     *
     * @param ChunkSize New chunk size
     */
    void UpdateChunkSize(size_t ChunkSize) { chunk_size = ChunkSize; }

    /**
     * Append a table to the builder, sorting it before appeending.
     *
     * @param chunk Table to append. Must be pinned.
     */
    void AppendChunk(std::shared_ptr<table_info> chunk);

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
     *
     * @param sortlimit If it's std::nullopt then neither limit nor offset it
     * set Othereise sortlimit.limit is how many rows to include as outputs at
     * maximum sortlimit.offset is how many rows to skip
     */
    std::deque<TableAndRange> Finalize();

    /**
     * Merge a list of sorted lists to produce a single sorted list. E.g.
     *   chunk_list_a = [[1, 3, 5], [7, 9, 11]]
     *   chunk_list_b = [[2, 4, 6], [8, 10, 12]]
     *   MergeChunks([chunk_list_a, chunk_list_b])
     *     = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
     * All tables in the output will be unpinned.
     * @param sortlimit If it's std::nullopt then neither limit nor offset it
     * set Othereise sortlimit.limit is how many rows to include as outputs at
     * maximum sortlimit.offset is how many rows to skip
     * @param is_last Whether this is final iteration of Finalize
     * For all but the last call to MergeChunks, we only keep all rows
     * from [0, limit + offset). For the last call to MergeChunks, we
     * only keep rows from [offset, limit + offset).
     */
    // TODO(aneesh) make this a private or static method
    template <bool is_last>
    std::deque<TableAndRange> MergeChunks(
        std::vector<std::deque<TableAndRange>>&& sorted_chunks);
};

struct StreamSortMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    time_t local_sort_chunk_time;
    time_t global_sort_time;
    time_t global_append_chunk_time;
    time_t communication_phase;
    time_t partition_chunks_time;
    time_t sampling_time;

    int64_t output_row_count = 0;

    // TODO(aneesh) report these metrics and refactor to avoid using this object
    // as is, instead having more finer grained metrics
    IncrementalShuffleMetrics ishuffle_metrics;
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
    // Pair of <number of tables, sum of table sizes> this can be used to get
    // the average row size after aggregating this pair across all ranks.
    std::pair<int64_t, int64_t> row_info;
    bodo::IBufferPool* op_pool;
    // Memory manager instance for op_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;

    // Index of chunk to yield
    size_t output_idx = 0;
    // List of chunks ready to yield
    std::vector<std::shared_ptr<table_info>> output_chunks;

    // Type of input table
    std::shared_ptr<bodo::Schema> schema;
    // Empty table created with the same schema as the input
    std::shared_ptr<table_info> dummy_output_chunk;

    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

    int64_t chunk_size;
    // builder will create a sorted list from the chunks passed to consume batch
    ChunkedTableBuilder builder;

    // exposed for test only - output of GetParallelSortBounds
    std::shared_ptr<table_info> bounds_;

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
    virtual void ConsumeBatch(std::shared_ptr<table_info> table, bool is_last);

    /**
     * @brief call after ConsumeBatch is called with is_last set to true to
     * finalize the build phase
     */
    void FinalizeBuild();

    /**
     * Get the next sorted output chunk. Returns a pair of table and a boolean
     * indicating if the last chunk has been reached
     */
    std::pair<std::shared_ptr<table_info>, bool> GetOutput();

    // Helper methods

    /**
     * @brief Helper function to get operator budget from op_id
     *
     * @param op_id operator ID
     * @return Operator budget. If unset (value of -1) will get all available
     * budgets
     */
    uint64_t GetBudget() const;

    /**
     * Sort all chunks across all ranks
     */
    virtual void GlobalSort(
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    /**
     * Get bounds for parallel sorting based on the chunks consumed so far.
     * These bounds determine which elements are assigned to which ranks.
     */
    std::shared_ptr<table_info> GetParallelSortBounds(
        std::deque<std::shared_ptr<table_info>>& local_chunks);

    // Partition all sorted input chunks a list of chunks per rank
    // All of the returned data needs to be communicated.
    std::vector<std::deque<std::shared_ptr<table_info>>> PartitionChunksByRank(
        SortedChunkedTableBuilder& global_builder, int n_pes,
        std::shared_ptr<table_info> bounds,
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Chunk of code for non-parallel part of GlobalSort
    // Made a separate function to reduce duplicated code
    void GlobalSort_NonParallel(
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Chunks of code for partitioning + async communication part of GlobalSort
    // Made a separate function to reduce duplicated code
    SortedChunkedTableBuilder GlobalSort_Partition(
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Helper function that gets override in child class to pass in limit/offset
    virtual SortedChunkedTableBuilder GetGlobalBuilder() {
        return SortedChunkedTableBuilder(
            schema, dict_builders, n_key_t, vect_ascending, na_position,
            dead_keys, num_chunks, chunk_size, mem_budget_bytes, -1, -1,
            parallel, op_pool, op_mm);
    }

    // Report all metrics
    void ReportMetrics();
    virtual ~StreamSortState() = default;
};

struct StreamSortLimitOffsetState : StreamSortState {
    bool limit_small_flag = false;
    SortLimits sortlimit;
    SortedChunkedTableBuilder top_k;

    StreamSortLimitOffsetState(int64_t op_id, int64_t n_key_t,
                               std::vector<int64_t>&& vect_ascending_,
                               std::vector<int64_t>&& na_position_,
                               std::shared_ptr<bodo::Schema> schema,
                               bool parallel = true, int64_t limit = -1,
                               int64_t offset = -1, size_t chunk_size = 4096,
                               bool enable_small_limit_optimization = false);

    // Override base class to pass in limit / offset
    SortedChunkedTableBuilder GetGlobalBuilder() override {
        return SortedChunkedTableBuilder(
            schema, dict_builders, n_key_t, vect_ascending, na_position,
            dead_keys, num_chunks, chunk_size, mem_budget_bytes,
            sortlimit.limit, sortlimit.offset, parallel, op_pool, op_mm);
    }

    // Override base class that adds limit / offset related logic
    void GlobalSort(
        std::deque<std::shared_ptr<table_info>>&& local_chunks) override;

    // Override base class to maintain a heap of at most limit + offset elements
    void ConsumeBatch(std::shared_ptr<table_info> table, bool is_last) override;

    // Compute local limit based on global limit after partitioning
    void ComputeLocalLimit(SortedChunkedTableBuilder& global_builder);

    // Separate logic for when limit + offset is small. Send all data to rank 0
    // and sort everything on rank 0
    // TODO: scatter data from rank 0 to all rank
    void SmallLimitOptim();

    ~StreamSortLimitOffsetState() override = default;
};
