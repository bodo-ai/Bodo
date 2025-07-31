#include <optional>
#include <random>

#include "../_bodo_common.h"
#include "../_chunked_table_builder.h"
#include "../_query_profile_collector.h"
#include "../_table_builder.h"
#include "_shuffle.h"

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
    // Offset of the first row in the range table. This is to represent tables
    // where we only want the suffix of the table, and the first row of range
    // (the min) is actually the `offset`th row of the table.
    int64_t offset;
    // Table with only 2 rows for the min and max of table - always pinned.
    // We use a TBB so that we don't need to reallocate when modifying the
    // offset (which can be often during a K-way merge for instance).
    TableBuildBuffer range;

    TableAndRange(
        std::shared_ptr<table_info> table, int64_t n_keys,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        int64_t offset = 0);

    /**
     * Update the offset into the table and adjust the range accordingly
     */
    void UpdateOffset(int64_t n_keys, int64_t offset);

    /**
     * @brief For debugging purpose
     */
    friend std::ostream& operator<<(std::ostream& os, const TableAndRange& obj);
};

/**
 * Similar to ChunkedTableBuilder but provides a stream of TableAndRange objects
 * instead of plain tables.
 *
 * NOTE: ChunkedTableAndRangeBuilder assumes that it's input is a stream of
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
        TableAndRange chunk{std::move(active_chunk), n_key,
                            this->dict_builders};
        chunk.table->unpin();

        chunks.emplace_back(std::move(chunk));
    }

    void ResetInternal() final { chunks.clear(); }
};

/**
 * @brief Similar to ChunkedTableBuilder, except that the chunks are
 * individually sorted. Every time 'chunk_size' many rows have been accumulated,
 * we sort them and append the sorted result as a chunk. Note that while the
 * chunks are guaranteed to be sorted, there is no guarantee of a global order.
 *
 */
struct SortedChunkedTableBuilder : AbstractChunkedTableBuilder {
    const int64_t n_keys = -1;
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;

    // Queue of finalized chunks. We use a deque instead of
    // a regular queue since it gives us ability to both
    // iterate over elements as well as pop/push. Finalized chunks are unpinned.
    // If we want to access finalized chunks, we need to pin and unpin them
    // manually.
    std::deque<TableAndRange> chunks;

    // Total time spent sorting
    MetricBase::TimerValue sort_time = 0;
    MetricBase::TimerValue sort_copy_time = 0;

    // TODO Add an option to re-use the "active chunk" since in this case, we
    // are going to copy it anyway. Similarly, there's no reason to
    // "shrink_to_fit" since the buffer can be reused.

    // TODO Add limit/offset information to allow truncating the "active_chunk"
    // during finalization.

    SortedChunkedTableBuilder(
        std::shared_ptr<bodo::Schema> schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        int64_t n_keys_, const std::vector<int64_t>& vect_ascending_,
        const std::vector<int64_t>& na_position_, size_t chunk_size,
        size_t max_resize_count_for_variable_size_dtypes =
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager())
        : AbstractChunkedTableBuilder(schema, dict_builders_, chunk_size,
                                      max_resize_count_for_variable_size_dtypes,
                                      pool, mm),
          n_keys(n_keys_),
          vect_ascending(vect_ascending_),
          na_position(na_position_) {}

   private:
    /// @brief Number of "finalized" chunks.
    size_t NumReadyChunks() final { return this->chunks.size(); }

    /// @brief Return the next available chunk.
    std::shared_ptr<table_info> PopFront() final {
        TableAndRange chunk = this->chunks.front();
        // Pin the table before returning it
        chunk.table->pin();
        this->chunks.pop_front();
        return chunk.table;
    }

    /// @brief Once the 'active_chunk' has accumulated enough rows, this
    /// function is called to append the chunk to the deque of chunks. Before
    /// adding it to the deque of chunks, we sort the active_chunk.
    void PushActiveChunk() final;

    /// @brief Reset internal state. This releases the finalized chunks.
    void ResetInternal() final { this->chunks.clear(); }
};

// The row idx for the minimum row in a TableAndRange
#define RANGE_MIN 0
// The row idx for the maximum row in a TableAndRange
#define RANGE_MAX 1

/**
 * @brief Metrics for the Finalize step of ExternalKWayMergeSorter which
 * performs a chunk-merge based sort.
 *
 */
struct ExternalKWayMergeSorterFinalizeMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t merge_chunks_processing_chunk_size = -1;
    stat_t merge_chunks_K = -1;
    time_t kway_merge_sort_total_time = 0;
    time_t merge_input_builder_total_sort_time = 0;
    time_t merge_input_builder_total_sort_copy_time = 0;
    time_t merge_input_builder_finalize_time = 0;
    stat_t merge_n_input_chunks = 0;
    stat_t merge_approx_input_chunks_total_bytes = 0;
    stat_t merge_approx_max_input_chunk_size_bytes = 0;
    stat_t performed_inmem_concat_sort = 0;
    time_t finalize_inmem_concat_time = 0;
    time_t finalize_inmem_sort_time = 0;
    time_t finalize_inmem_output_append_time = 0;
    stat_t n_chunk_merges = 0;
    stat_t n_merge_levels = 0;
    time_t merge_chunks_total_time = 0;
    time_t merge_chunks_make_heap_time = 0;
    time_t merge_chunks_output_append_time = 0;
    time_t merge_chunks_pop_heap_time = 0;
    time_t merge_chunks_push_heap_time = 0;

    // Add metrics to the 'metrics' vector.
    void ExportMetrics(std::vector<MetricBase>& metrics);
};

/**
 * Builder that accepts a stream of tables and sorts all rows
 */
struct ExternalKWayMergeSorter {
   public:
    const std::shared_ptr<bodo::Schema> schema;
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders;
    const int64_t n_keys = -1;
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    // Total amount of memory budget.
    const uint64_t mem_budget_bytes;

    // Size of the final output chunks.
    const size_t output_chunk_size = STREAMING_BATCH_SIZE;

    // Whether to enable the in-memory concat sort merge optimization when all
    // data fits in memory. Only for unit testing purposes.
    const bool enable_inmem_concat_sort = true;

    const bool debug_mode = false;

    /**
     * Comparator for creating a heap of TableAndRange objects sorted by the
     * minimum values in the range. Does not pin the underlying table.
     */
    struct HeapComparator {
        ExternalKWayMergeSorter& builder;

        bool operator()(const TableAndRange& a, const TableAndRange& b) const {
            // Returns true if a.range[MIN] >= b.range[MIN]
            return !builder.Compare(a.range.data_table, RANGE_MIN,
                                    b.range.data_table, RANGE_MIN);
        }
    } comp;

    // Holds unpinned references to the result of sorting chunks (in batches)
    // submitted to AppendChunk. This will be consumed when Finalize is called.
    std::unique_ptr<SortedChunkedTableBuilder> sorted_input_chunks_builder =
        nullptr;

    // Memory pool and manager to use for allocations.
    bodo::IBufferPool* pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    // Metrics from the finalize step (K-way merge sort)
    ExternalKWayMergeSorterFinalizeMetrics metrics;

    /**
     * @brief Construct a new External K Way Merge Sorter.
     *
     * @param schema Schema of the table.
     * @param dict_builders_ Dictionary builders for the columns.
     * @param n_keys Number of sort key columns.
     * @param vect_ascending Whether to sort ascending or descending.
     * @param na_position Whether to put NAs at the end of front.
     * @param dead_keys Key columns that can be skipped from the output.
     * @param mem_budget_bytes_ Memory available for computation. This is used
     * to calculate the optimal K and chunk-size for the K-way merge (unless
     * they are explicitly provided).
     * @param bytes_per_row_ Average number of bytes per row. This is also used
     * to calculate the optimal K and chunk-size.
     * @param limit Limit to apply. -1 if no limit exists.
     * @param offset Offset to apply. -1 if no offset exists.
     * @param output_chunk_size_ Chunk size of the output chunks.
     * @param K_ K for the K-way merge. -1 if it should be calculated based on
     * memory budget and row size.
     * @param processing_chunk_size_ Chunk size to use during the K-way merge.
     * -1 if it should be calculated based on the memory budget and row size.
     * @param enable_inmem_concat_sort_ Whether to enable the optimization that
     * simply concatenates and sorts all chunks if they fit in memory (and skips
     * the K-way merge entirely).
     * @param pool Memory pool to use for all allocations.
     * @param mm Corresponding memory manager.
     */
    ExternalKWayMergeSorter(
        std::shared_ptr<bodo::Schema> schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        int64_t n_keys, const std::vector<int64_t>& vect_ascending,
        const std::vector<int64_t>& na_position,
        const std::vector<int64_t>& dead_keys, uint64_t mem_budget_bytes_ = 0,
        int64_t bytes_per_row_ = -1, int64_t limit = -1, int64_t offset = -1,
        size_t output_chunk_size_ = STREAMING_BATCH_SIZE, int64_t K_ = -1,
        int64_t processing_chunk_size_ = -1,
        bool enable_inmem_concat_sort_ = true, bool debug_mode_ = false,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    // Return true if table1[row1] < table2[row2]
    bool Compare(std::shared_ptr<table_info> table1, size_t row1,
                 std::shared_ptr<table_info> table2, size_t row2) const;

    /**
     * Append a table to the builder.
     *
     * @param chunk Table to append. Must be pinned.
     */
    void AppendChunk(std::shared_ptr<table_info> chunk);

    /**
     * @brief Update the limit/offset.
     *
     * @param new_limit
     * @param new_offset
     */
    void UpdateLimitOffset(int64_t new_limit, int64_t new_offset);

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
     * @param reset_input_builder Whether the input builder should be reset.
     * This is used for the small-limit case (by setting it to false) where we
     * use this as the top-k heap and want to reuse the builder.
     */
    std::deque<TableAndRange> Finalize(bool reset_input_builder = true);

   private:
    /**
     * Merge a list of sorted lists to produce a single sorted list. E.g.
     *   chunk_list_a = [[1, 3, 5], [7, 9, 11]]
     *   chunk_list_b = [[2, 4, 6], [8, 10, 12]]
     *   MergeChunks([chunk_list_a, chunk_list_b])
     *     = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
     * All tables in the output will be unpinned.
     * @param sorted_chunks List of list of chunks to sort. Every top level list
     * is expected to have globally sorted chunks.
     * @tparam is_last Whether this is final iteration of Finalize
     * For all but the last call to MergeChunks, we only keep all rows
     * from [0, limit + offset). For the last call to MergeChunks, we
     * only keep rows from [offset, limit + offset).
     * @tparam has_limit_offset Whether there's a limit/offset to apply.
     */
    // TODO(aneesh) make this a private or static method
    template <bool is_last, bool has_limit_offset>
    std::deque<TableAndRange> MergeChunks(
        std::vector<std::deque<TableAndRange>>&& sorted_chunks) /*const*/;

    template <typename IndexT>
        requires(std::is_same_v<IndexT, int32_t> ||
                 std::is_same_v<IndexT, int64_t>)
    std::deque<TableAndRange> SelectRowsAndProduceOutputInMem(
        std::shared_ptr<table_info> table, bodo::vector<IndexT>&& out_idxs);

    // Number of rows per chunk
    size_t processing_chunk_size;
    // Maximum number of chunks that will be pinned simultaneously during
    // sort.
    size_t K;
    // Whether we apply limit/offset during Finalize / MergeChunks
    std::optional<SortLimits> sortlimits = std::nullopt;
};

/**
 * @brief Metrics for the sort computation. This consists of metrics for both
 * the regular as well as the limit-offset cases. The user is responsible for a
 * reporting a subset of these based on relevancy.
 *
 */
struct StreamSortMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    //// Stage 1 (Consume and Finalize)

    /// Local consume batch metrics
    stat_t input_chunks_size_bytes_total = 0;
    stat_t n_input_chunks = 0;
    time_t input_append_time = 0;
    // Only in the small limit case:
    time_t topk_heap_append_chunk_time = 0;
    time_t topk_heap_update_time = 0;

    /// FinalizeBuild metrics
    time_t global_dict_unification_time = 0;
    time_t total_finalize_time = 0;

    /// Partitioning Metrics
    time_t kway_merge_sorter_append_time = 0;
    time_t get_bounds_total_time = 0;
    time_t get_bounds_dict_unify_time = 0;
    time_t get_bounds_gather_samples_time = 0;
    time_t get_bounds_compute_bounds_time = 0;
    time_t partition_chunks_total_time = 0;
    time_t partition_chunks_pin_time = 0;
    time_t parition_chunks_allreduce_time = 0;
    time_t partition_chunks_append_time = 0;
    time_t partition_chunks_sort_time = 0;
    time_t partition_chunks_sort_copy_time = 0;
    time_t partition_chunks_compute_dest_rank_time = 0;
    stat_t shuffle_chunk_size = 0;
    time_t shuffle_total_time = 0;
    time_t shuffle_issend_time = 0;
    time_t shuffle_send_done_check_time = 0;
    stat_t shuffle_n_send_done_checks = 0;
    time_t shuffle_irecv_time = 0;
    stat_t shuffle_n_irecvs = 0;
    time_t shuffle_recv_done_check_time = 0;
    stat_t shuffle_n_recv_done_checks = 0;
    time_t shuffle_barrier_test_time = 0;
    stat_t shuffle_n_barrier_tests = 0;
    stat_t n_shuffle_send = 0;
    stat_t n_shuffle_recv = 0;
    stat_t shuffle_total_sent_nrows = 0;
    stat_t shuffle_total_recv_nrows = 0;
    stat_t shuffle_total_approx_sent_size_bytes = 0;
    stat_t shuffle_approx_sent_size_bytes_dicts = 0;
    stat_t shuffle_total_recv_size_bytes = 0;
    stat_t n_rows_after_shuffle = 0;
    stat_t max_concurrent_sends = 0;
    stat_t max_concurrent_recvs = 0;
    // We only get approx_recv_size_bytes_dicts and dict_unification_time
    // from this.
    // TODO(aneesh) Refactor to avoid using this object as is, instead having
    // more finer grained metrics.
    IncrementalShuffleMetrics ishuffle_metrics;

    /// SortedCTB Finalize metrics
    ExternalKWayMergeSorterFinalizeMetrics
        external_kway_merge_sort_finalize_metrics;

    /// LimitOffset Finalize metrics
    // Only in the small limit case:
    time_t small_limit_local_concat_time = 0;
    time_t small_limit_gather_time = 0;
    time_t small_limit_rank0_sort_time = 0;
    time_t small_limit_rank0_output_append_time = 0;
    // Only in the non-small-limit case:
    time_t compute_local_limit_time = 0;

    //// Stage 2 (Produce output)

    int64_t output_row_count = 0;
};

#define DEFAULT_SAMPLE_SIZE 2048

/**
 * @brief Metrics for the sampling state.
 *
 */
struct ReservoirSamplingMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    time_t sampling_process_input_time = 0;
    stat_t n_sampling_buffer_rebuilds = 0;
    stat_t n_samples_taken = 0;

    // Add sampling metrics to the 'metrics' vector.
    void ExportMetrics(std::vector<MetricBase>& metrics);
};

/**
 * Infrastructure for randomly sampling rows from a stream of chunks. Implements
 * algorithm L as presented here:
 * https://en.wikipedia.org/wiki/Reservoir_sampling
 */
class ReservoirSamplingState {
   private:
    int64_t sample_size = DEFAULT_SAMPLE_SIZE;

    // Indices of columns to be selected for sampling
    std::vector<int64_t> column_indices;
    // Schema of sampled table
    std::shared_ptr<bodo::Schema> schema;
    // Dictionary builders to use for the samples
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

    int64_t row_to_sample = -1;
    // Counter for how many input rows have been consumed so far
    int64_t total_rows_seen = 0;
    // Random variable to determine the next row to sample from - see the
    // algorithm in the URL above.
    double W;

    // State for random number generation
    std::mt19937 e;
    std::uniform_real_distribution<double> dis;

    // Builder of sampled rows
    TableBuildBuffer samples;
    // A vector of all indicies that are marked as true in the selection vector
    std::vector<int64_t> selected_rows;

   public:
    // Metrics
    ReservoirSamplingMetrics metrics;

    ReservoirSamplingState(
        int64_t n_keys, int64_t sample_size_,
        std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        std::shared_ptr<bodo::Schema> schema)
        : sample_size(sample_size_), dis(0, 1) {
        char* sample_size_str = std::getenv("BODO_STREAM_SORT_SAMPLE_SIZE");
        if (sample_size_str != nullptr) {
            sample_size = std::stoi(sample_size_str);
        }

        std::vector<std::unique_ptr<bodo::DataType>> key_types(n_keys);
        for (int64_t i = 0; i < n_keys; i++) {
            key_types[i] = schema->column_types[i]->copy();
            dict_builders.push_back(dict_builders_[i]);
            column_indices.push_back(i);
        }

        std::iota(column_indices.begin(), column_indices.end(), 0);
        schema = std::make_shared<bodo::Schema>(std::move(key_types));
        samples = TableBuildBuffer(schema, dict_builders);

        // Initialize W with a random value
        W = random();
    };

    /**
     * @brief Take in a chunk as input and randomly sample rows from it. The
     * rows to sample from are picked by incrementing row_to_sample by a random
     * amount determined by W. These rows will be appended to an internal buffer
     * and can be obtained by calling Finalize when all input has been
     * processed. Note that it is assumed that input_chunk has already been
     * unified with the dict_builders used to construct this state.
     *
     * @param input table to sample from.
     */
    void processInput(const std::shared_ptr<table_info>& input_chunk);

    /**
     * @brief Returns the set of local samples on this rank by combing the state
     * in tables, sampled_tables, and replacements.
     * For every row of tables, we look up if that row has a replacement in
     * replacements and if so, select a row from the correponding table in
     * sampled_tables instead. Note that the order of rows in the output table
     * can be arbitrary.
     */
    std::shared_ptr<table_info> Finalize();

   private:
    /**
     * @brief Get a uniform random number between 0 and 1
     */
    double random();
};

struct StreamSortState {
    const int64_t op_id = -1;
    const int64_t n_keys = -1;
    const std::vector<int64_t> vect_ascending;
    const std::vector<int64_t> na_position;
    const std::vector<int64_t> dead_keys;
    const bool parallel = true;
    const size_t output_chunk_size = STREAMING_BATCH_SIZE;
    bool build_finalized = false;
    bool debug_mode = false;

    // These are only for unit testing purposes. -1 means selecting the optimal
    // values based on the budget, number of ranks, etc.
    const int64_t kway_merge_chunksize = -1;
    const int64_t kway_merge_k = -1;
    const bool enable_inmem_concat_sort = true;
    const int64_t shuffle_chunksize = -1;
    // This will either be overridden or set to MPI_Comm_size during
    // initialization.
    const size_t shuffle_max_concurrent_msgs;

    // Initialized during `FinalizeBuild` once all rows have been seen.
    int64_t bytes_per_row = -1;

    uint64_t mem_budget_bytes;

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

    // Dictionary builders
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

    // Empty table created with the same schema as the input
    std::shared_ptr<table_info> dummy_output_chunk;

    // Input chunks
    ChunkedTableBuilder builder;

    ReservoirSamplingState reservoir_sampling_state;

    // NOTE: Exposed for test only - output of GetParallelSortBounds
    std::shared_ptr<table_info> bounds_;

    // Metrics for sorting
    StreamSortMetrics metrics;
    bool reported_build_metrics = false;

    StreamSortState(
        int64_t op_id, int64_t n_keys, std::vector<int64_t>&& vect_ascending_,
        std::vector<int64_t>&& na_position_,
        std::shared_ptr<bodo::Schema> schema, bool parallel = true,
        size_t sample_size = DEFAULT_SAMPLE_SIZE,
        size_t output_chunk_size_ = STREAMING_BATCH_SIZE,
        int64_t kway_merge_chunk_size_ = -1, int64_t kway_merge_k_ = -1,
        std::optional<bool> enable_inmem_concat_sort_ = std::nullopt,
        int64_t shuffle_chunksize_ = -1,
        int64_t shuffle_max_concurrent_sends_ = -1);
    /**
     * @brief Consume an unsorted table and use it for global sorting
     *
     * @param table unsorted input table
     * @return boolean indicating if the consume phase is complete
     */
    virtual void ConsumeBatch(std::shared_ptr<table_info> table);

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

    /**
     * Get all sorted output chunks. Note that all table_infos should be assumed
     * to be unpinned.
     */
    std::vector<std::shared_ptr<table_info>> GetAllOutputUnpinned();

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
        std::shared_ptr<table_info>&& local_samples);

    // Partition all sorted input chunks a list of chunks per rank
    // All of the returned data needs to be communicated.
    std::vector<std::deque<std::shared_ptr<table_info>>> PartitionChunksByRank(
        const ExternalKWayMergeSorter& global_builder, int n_pes,
        std::shared_ptr<table_info> bounds,
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Chunk of code for non-parallel part of GlobalSort
    // Made a separate function to reduce duplicated code
    void GlobalSort_NonParallel(
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Chunks of code for partitioning + async communication part of GlobalSort
    // Made a separate function to reduce duplicated code
    ExternalKWayMergeSorter GlobalSort_Partition(
        std::deque<std::shared_ptr<table_info>>&& local_chunks);

    // Helper function that gets override in child class to pass in limit/offset
    virtual ExternalKWayMergeSorter GetKWayMergeSorter() const {
        return ExternalKWayMergeSorter(
            this->schema, this->dict_builders, this->n_keys,
            this->vect_ascending, this->na_position, this->dead_keys,
            this->mem_budget_bytes, this->bytes_per_row, -1, -1,
            this->output_chunk_size, this->kway_merge_k,
            this->kway_merge_chunksize, this->enable_inmem_concat_sort,
            this->debug_mode, op_pool, op_mm);
    }

    /// Report metrics for the build stage. The function is idempotent and
    /// metrics will only be reported once.
    /// @param metrics_out Vector to append metrics to and report.
    virtual void ReportBuildMetrics(std::vector<MetricBase>& metrics_out);

    virtual ~StreamSortState() = default;
};

struct StreamSortLimitOffsetState : StreamSortState {
    bool limit_small_flag = false;
    SortLimits sortlimit;
    ExternalKWayMergeSorter top_k;

    StreamSortLimitOffsetState(
        int64_t op_id, int64_t n_keys, std::vector<int64_t>&& vect_ascending_,
        std::vector<int64_t>&& na_position_,
        std::shared_ptr<bodo::Schema> schema, bool parallel = true,
        int64_t limit = -1, int64_t offset = -1,
        size_t output_chunk_size_ = STREAMING_BATCH_SIZE,
        int64_t kway_merge_chunk_size_ = -1, int64_t kway_merge_k_ = -1,
        std::optional<bool> enable_inmem_concat_sort_ = std::nullopt,
        bool enable_small_limit_optimization = true);

    // Override base class to pass in limit / offset
    ExternalKWayMergeSorter GetKWayMergeSorter() const override {
        return ExternalKWayMergeSorter(
            this->schema, this->dict_builders, this->n_keys,
            this->vect_ascending, this->na_position, this->dead_keys,
            this->mem_budget_bytes, this->bytes_per_row, this->sortlimit.limit,
            this->sortlimit.offset, this->output_chunk_size, this->kway_merge_k,
            this->kway_merge_chunksize, this->enable_inmem_concat_sort,
            this->debug_mode, op_pool, op_mm);
    }

    // Override base class that adds limit / offset related logic
    virtual void GlobalSort(
        std::deque<std::shared_ptr<table_info>>&& local_chunks) override final;

    // Override base class to maintain a heap of at most limit + offset elements
    virtual void ConsumeBatch(std::shared_ptr<table_info> table) override final;

    // Compute local limit based on number of local rows.
    SortLimits ComputeLocalLimit(size_t local_nrows) /*const*/;

    // Separate logic for when limit + offset is small. Send all data to rank 0
    // and sort everything on rank 0
    // TODO: scatter data from rank 0 to all rank
    void SmallLimitOptim();

    /// Report metrics for the build stage. The function is idempotent and
    /// metrics will only be reported once.
    /// @param metrics_out Vector to append metrics to and report.
    virtual void ReportBuildMetrics(
        std::vector<MetricBase>& metrics_out) override final;

    ~StreamSortLimitOffsetState() override = default;
};
