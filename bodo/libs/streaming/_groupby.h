#pragma once

#include "../_bodo_common.h"
#include "../_chunked_table_builder.h"
#include "../_dict_builder.h"
#include "../_operator_pool.h"
#include "../_query_profile_collector.h"
#include "../_table_builder.h"
#include "../groupby/_groupby_col_set.h"
#include "../vendored/hyperloglog.hpp"

#include "_shuffle.h"

// Default threshold for Groupby operator's OperatorBufferPool
#define GROUPBY_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD 0.5

// Use all available memory by default
#define GROUPBY_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

#define GROUPBY_DEFAULT_MAX_PARTITION_DEPTH 15

// Chunk size of build_table_buffer_chunked in GroupbyPartition
#define INACTIVE_PARTITION_TABLE_CHUNK_SIZE 16 * 1024

class GroupbyPartition;
class GroupbyIncrementalShuffleState;
class GroupbyState;

// Track the type of operation due to overlap between window functions and
// aggregates.
enum class AggregationType {
    AGGREGATE = 0,
    MRNF = 1,
    WINDOW = 2,
};

/**
 * @brief Get the string representation of the AggregationType
 * for metrics purposes.
 *
 * @param type The AggregationType to convert to string.
 * @return std::string String representation of the AggregationType.
 */
std::string get_aggregation_type_string(AggregationType type);

template <bool is_local>
struct HashGroupbyTable {
    /**
     * provides row hashes for groupby hash table (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or input table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the input table
     *    at index (-iRow - 1).
     *
     * When is_local = true, build table is the build_table in the
     * groupby_partition.
     * When is_local = false, build table is the shuffle_table in the
     * groupby_state.
     *
     * @param iRow row number
     * @return hash of row iRow
     */
    uint32_t operator()(const int64_t iRow) const;
    /// This will be a nullptr when is_local=false
    GroupbyPartition* groupby_partition;
    /// This will be a nullptr when is_local=true
    GroupbyIncrementalShuffleState* groupby_shuffle_state;
};

template <bool is_local>
struct KeyEqualGroupbyTable {
    /**
     * provides row comparison for groupby hash table
     * (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or input table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the input table
     *    at index (-iRow - 1).
     *
     * When is_local = true, build table is the build_table in the
     * groupby_partition.
     * When is_local = false, build table is the shuffle_table in the
     * groupby_state.
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true if equal else false
     */
    bool operator()(const int64_t iRowA, const int64_t iRowB) const;
    /// This will be a nullptr when is_local=false
    GroupbyPartition* groupby_partition;
    /// This will be a nullptr when is_local=true
    GroupbyIncrementalShuffleState* groupby_shuffle_state;
    const uint64_t n_keys;
};

/**
 * @brief Struct for storing the Groupby metrics.
 *
 */
struct GroupbyMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    /**
     * @brief Struct for metrics collected while executing
     * get_grouping_infos_for_update_table.
     *
     */
    struct GetGroupInfoMetrics {
        time_t hashing_time = 0;
        stat_t hashing_nrows = 0;
        time_t grouping_time = 0;
        stat_t grouping_nrows = 0;
        time_t hll_time = 0;
        stat_t hll_nrows = 0;
    };

    /**
     * @brief Struct for metrics collected while executing get_update_table,
     * compute_local_mrnf, or compute_local_window.
     *
     */
    struct AggUpdateMetrics {
        GetGroupInfoMetrics grouping_metrics;
        time_t colset_update_time = 0;
        stat_t colset_update_nrows = 0;
    };

    ///// Required Metrics
    stat_t build_input_row_count = 0;
    stat_t output_row_count = 0;

    ///// Optional Metrics

    /// Partition stats
    stat_t n_partitions = 1;
    stat_t n_repartitions_in_append = 0;
    stat_t n_repartitions_in_finalize = 0;

    /// Time spent repartitioning
    time_t repartitioning_time = 0;  // Overall
    time_t repartitioning_part_hashing_time = 0;
    stat_t repartitioning_part_hashing_nrows = 0;
    // Active case
    time_t repartitioning_active_part1_append_time = 0;
    stat_t repartitioning_active_part1_append_nrows = 0;
    time_t repartitioning_active_part2_append_time = 0;
    stat_t repartitioning_active_part2_append_nrows = 0;
    // Inactive case
    // Most of the cost of PopChunk is expected to be from disk IO.
    time_t repartitioning_inactive_pop_chunk_time = 0;
    stat_t repartitioning_inactive_pop_chunk_n_chunks = 0;
    // nrows for this is repartitioning_part_hashing_nrows
    time_t repartitioning_inactive_append_time = 0;

    /// Agg
    // - Time in get_update_table
    time_t pre_agg_total_time = 0;
    AggUpdateMetrics pre_agg_metrics;
    // NOTE: Input nrows is the same as overall local input rows.
    stat_t pre_agg_output_nrows = 0;
    // - Time spent groupby hashing (both local input and shuffle output)
    time_t input_groupby_hashing_time = 0;

    // - UpdateGroupsAndCombine (including from ActivatePartition)
    time_t rebuild_ht_hashing_time = 0;
    stat_t rebuild_ht_hashing_nrows = 0;
    time_t rebuild_ht_insert_time = 0;
    stat_t rebuild_ht_insert_nrows = 0;
    // This is the time spent in UpdateGroupsAndCombine doing the ReserveTable
    // and the update_groups_helper loop.
    time_t update_logical_ht_time = 0;
    stat_t update_logical_ht_nrows = 0;
    time_t combine_input_time = 0;
    stat_t combine_input_nrows = 0;

    /// Acc (both AppendBuildBatch and ActivatePartition)
    time_t appends_active_time = 0;
    stat_t appends_active_nrows = 0;

    /// Common for both agg and acc cases
    // - Time spent partition hashing (both local input and shuffle output)
    time_t input_part_hashing_time = 0;
    stat_t input_hashing_nrows = 0;  // For both part and groupby hashes
    time_t input_partition_check_time = 0;
    stat_t input_partition_check_nrows = 0;
    time_t appends_inactive_time = 0;
    stat_t appends_inactive_nrows = 0;

    // Time spent updating the histogram buckets
    time_t update_histogram_buckets_time = 0;

    // Time set updating append_row_to_build_table
    time_t append_row_to_build_table_append_time = 0;
    time_t append_row_to_build_table_flip_time = 0;

    /// FinalizeBuild
    time_t finalize_time = 0;  // Overall

    // Only relevant for the Agg case
    time_t finalize_activate_groupby_hashing_time = 0;

    // Only relevant for the acc, MRNF, and window cases
    time_t finalize_get_update_table_time = 0;  // Overall
    time_t finalize_compute_mrnf_time = 0;      // Overall
    time_t finalize_window_compute_time = 0;    // Overall
    // - get_update_table / compute_local_mrnf
    AggUpdateMetrics finalize_update_metrics;

    // Relevant for all cases
    // - Eval time across all partitions
    time_t finalize_eval_time = 0;  // Not relevant in MRNF case
    stat_t finalize_eval_nrows = 0;
    // - ActivatePartition
    time_t finalize_activate_partition_time = 0;  // Overall time
    // -- Either through the iterator or PopChunk
    time_t finalize_activate_pin_chunk_time = 0;
    stat_t finalize_activate_pin_chunk_n_chunks = 0;

    /// NOTE: We don't track any metrics for the ProduceOutput stage.
    /// Essentially all the time (already tracked by codegen) can be attributed
    /// to disk IO during the PopChunk calls.
};

template <bool is_local>
using grpby_hash_table_t =
    bodo::unord_map_container<int64_t, int64_t, HashGroupbyTable<is_local>,
                              KeyEqualGroupbyTable<is_local>>;

/**
 * @brief Holds the state of a single partition
 * during Groupby execution. This includes the
 * logical hashtable (which comprises of the build_table_buffer,
 * an unordered map which maps rows to groups and the hashes for
 * the groups), references to the ColSets, etc.
 * When the partition is not active, it stores the input
 * in a ChunkedTableBuilder.
 * 'top_bitmask' and 'num_top_bits' define the partition
 * itself, i.e. a record is in this partition if the top
 * 'num_top_bits' bits of its hash are 'top_bitmask'.
 *
 */
class GroupbyPartition {
   public:
    using hash_table_t = grpby_hash_table_t</*is_local*/ true>;

    explicit GroupbyPartition(
        size_t num_top_bits_, uint32_t top_bitmask_,
        const std::shared_ptr<bodo::Schema> build_table_schema_,
        const std::shared_ptr<bodo::Schema> separate_out_cols_schema_,
        const uint64_t n_keys_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            build_table_dict_builders_,
        const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
        const std::vector<int32_t>& f_in_offsets_,
        const std::vector<int32_t>& f_in_cols_,
        const std::vector<int32_t>& f_running_value_offsets_, bool is_active_,
        bool accumulate_before_update_, bool req_extended_group_info_,
        GroupbyMetrics& metrics_, bodo::OperatorBufferPool* op_pool_,
        const std::shared_ptr<::arrow::MemoryManager> op_mm_,
        bodo::OperatorScratchPool* op_scratch_pool_,
        const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm_);

    // The schema of the build table.
    std::shared_ptr<bodo::Schema> build_table_schema;
    // Dictionary builders (shared by all partitions)
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Current number of groups
    int64_t next_group = 0;
    // Map row number to group number
    std::unique_ptr<hash_table_t> build_hash_table;
    // Hashes of data in build_table_buffer
    // XXX Might be better to use unique_ptrs to guarantee
    // that memory is released during FinalizeBuild.
    bodo::vector<uint32_t> build_table_groupby_hashes;
    // Current running values (output of "combine" step)
    // Note that in the accumulation case, the "running values" is instead the
    // entire input table.
    std::unique_ptr<TableBuildBuffer> build_table_buffer;
    /// @brief This is the number of rows that have been "safely"
    /// appended to the build_table_buffer. This is only relevant in the AGG
    /// case and only when the partition is active. This is used in
    /// SplitPartition to only split the first build_safely_appended_groups rows
    /// of the build_table_buffer. This is required since there might
    /// be a memory error during "UpdateGroupsAndCombine" which might happen
    /// _after_ we've appended to the build_table_buffer. In those
    /// cases, the append will be retried, so we need to ensure that
    /// previous incomplete append is invalidated.
    size_t build_safely_appended_groups = 0;

    // Chunked append-only buffer, only used if a partition is inactive and
    // not yet finalized
    std::unique_ptr<ChunkedTableBuilder> build_table_buffer_chunked;

    // The types of the columns in the separate_out_cols table.
    std::shared_ptr<bodo::Schema> separate_out_cols_schema;
    // Separate output columns with one column for each colset
    // that requires them. Only used in the AGG case.
    std::unique_ptr<TableBuildBuffer> separate_out_cols;

    // temporary batch data
    std::shared_ptr<table_info> in_table = nullptr;
    std::shared_ptr<uint32_t[]> in_table_hashes = nullptr;

    // ColSets and related information, owned by the GroupbyState and shared
    // by all partitions
    const std::vector<std::shared_ptr<BasicColSet>>& col_sets;
    const std::vector<int32_t>& f_in_offsets;
    const std::vector<int32_t>& f_in_cols;
    const std::vector<int32_t>& f_running_value_offsets;

    // Reference to the metrics for this operator. Shared with the global state
    // and all other partitions.
    GroupbyMetrics& metrics;

    /// @brief Get number of bits in the 'top_bitmask'.
    size_t get_num_top_bits() const { return this->num_top_bits; }

    /// @brief Get the 'top_bitmask'.
    uint32_t get_top_bitmask() const { return this->top_bitmask; }

    /// @brief Check if a row is part of this partition based on its
    /// partition hash.
    /// @param hash Partition hash for the row
    /// @return True if row is part of partition, False otherwise.
    inline bool is_in_partition(const uint32_t& hash) const noexcept;

    /// @brief Is the partition active?
    inline bool is_active_partition() const { return this->is_active; }

    /**
     * @brief Update the groups and combine them with values from
     * the input batch.
     *
     * NOTE: Only used in the aggregating code path (i.e.
     * accumulate_before_update = false)
     *
     * @tparam is_active Whether the partition is active. If the partition is
     * not active, we simply append the input into build_table_buffer_chunked to
     * be processed later when the partition has been activated.
     * @param in_table Input batch to update and combine with the current set of
     * running values.
     * @param batch_hashes_groupby Groupby Hashes for input batch.
     */
    template <bool is_active>
    void UpdateGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby);

    /**
     * @brief Same as above, except we will only process the rows
     * whose corresponding value in 'append_rows' is true.
     *
     * @tparam is_active Whether the partition is active. If the partition is
     * not active, we simply append the input into build_table_buffer_chunked to
     * be processed later when the partition has been activated.
     * @param in_table Input batch to update and combine with the current set of
     * running values.
     * @param batch_hashes_groupby Groupby Hashes for input batch.
     * @param append_rows Bitmask specifying the rows to use for the update and
     * combine steps.
     */
    template <bool is_active>
    void UpdateGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
        const std::vector<bool>& append_rows);

    /**
     * @brief Append an input batch to the 'build_table_buffer'
     * (or 'build_table_buffer_chunked' if the partition is not active).
     * No additional computation is done in this case.
     *
     * NOTE: Only used in the accumulating code path (i.e.
     * accumulate_before_update = true)
     *
     * @tparam is_active Whether the partition is active. If it is not,
     * we append into 'build_table_buffer_chunked' instead of
     * 'build_table_buffer'
     * @param in_table Input batch to append.
     */
    template <bool is_active>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Same as above, except we only append the rows specified
     * by the append_rows bitmask.
     * Note that we reserve space for the entire table before appending,
     * and not just for the required rows. This is based on the small
     * batch assumption where over-allocating is faster without increasing
     * memory usage disproportionately.
     *
     * @tparam is_active Whether the partition is active. If it is not,
     * we append into 'build_table_buffer_chunked' instead of
     * 'build_table_buffer'
     * @param in_table Input batch to append rows from.
     * @param append_rows Bitmask specifying the rows to append.
     */
    template <bool is_active>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table,
                          const std::vector<bool>& append_rows);

    /**
     * @brief Split the partition into 2^num_levels partitions.
     * This will produce a new set of partitions, each with their
     * new build_table_buffer and build_table_join_hashes.
     * If the partition is active, this create one active and 2^num_levels - 1
     * inactive partitions. If the partition is inactive, this
     * creates 2^num_levels inactive partitions.
     *
     * @tparam is_active Is this an active partition.
     * @param num_levels Number of levels to split the partition. Only '1' is
     * supported at this point.
     * @return std::vector<std::shared_ptr<GroupbyPartition>>
     */
    template <bool is_active>
    std::vector<std::shared_ptr<GroupbyPartition>> SplitPartition(
        size_t num_levels = 1);

    /**
     * @brief Finalize this partition and return its output.
     * This will also clear the build state (i.e. all memory
     * associated with the logical hash table).
     * This works for both the aggregating and accumulating
     * cases.
     *
     * @return std::shared_ptr<table_info> Output of this partition.
     */
    std::shared_ptr<table_info> Finalize();

    /**
     * @brief Finalize the partition in the MRNF case.
     * The computation is similar to that in 'get_update_table', except
     * it's slightly modified for the MRNF case.
     * In particular, unlike the regular case where we return an output table
     * that is then appended into the output buffer, this will append the output
     * directly into the 'output_buffer'. We can do this since we're just
     * appending certain rows (filter) from the 'build_table_buffer' and no
     * other transformation is required.
     *
     * This will also clear the build state (i.e. all memory associated with the
     * logical hash table).
     *
     * @param cols_to_keep Bitmask specifying the columns to
     * retain in the output.
     * @param output_buffer The output buffer to append the generated output to.
     */
    void FinalizeMrnf(const std::vector<bool>& cols_to_keep, size_t n_sort_keys,
                      ChunkedTableBuilder& output_buffer);

    /**
     * @brief Finalize the partition in the Window case.
     * The computation is similar to that in 'get_update_table', except
     * it's slightly modified for the Window case.
     * In particular, unlike the regular case where we return an output table
     * that is then appended into the output buffer, this will append the output
     * directly into the 'output_buffer'. We can do this since we're just
     * appending certain rows and the window function columns.
     *
     * This will also clear the build state (i.e. all memory associated with the
     * logical hash table).
     *
     * @param partition_by_cols_to_keep Bitmask specifying the partition by
     * columns to retain in the output.
     * @param order_by_cols_to_keep Bitmask specifying the order by columns to
     * retain in the output.
     * @param output_buffer The output buffer to append the generated output to.
     */
    void FinalizeWindow(
        const std::vector<bool>& cols_to_keep_bitmask, size_t n_sort_keys,
        ChunkedTableBuilder& output_buffer,
        std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builder,
        std::vector<int32_t> f_in_offsets, std::vector<int32_t> f_in_cols);

   private:
    const size_t num_top_bits = 0;
    const uint32_t top_bitmask = 0ULL;
    const uint64_t n_keys;
    const bool accumulate_before_update;
    const bool req_extended_group_info;
    /// @brief Whether the partition is active / has been activated yet.
    /// The 0th partition is active to begin with, and other partitions
    /// are activated at the end of the build step.
    /// When false, this means that the build data is still in the
    /// 'build_table_buffer_chunked'. When true, this means that the data has
    /// been moved to the 'build_table_buffer'.
    bool is_active = false;

    /// @brief OperatorBufferPool of the Groupby operator that this is a
    /// partition in. The pool is owned by the parent GroupbyState, so
    /// we keep a raw pointer for simplicity.
    bodo::OperatorBufferPool* const op_pool;
    /// @brief Memory manager instance for op_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;
    /// @brief Pointer to the OperatorScratchPool corresponding to the
    /// op_pool.
    bodo::OperatorScratchPool* const op_scratch_pool;
    /// @brief Memory manager instance for op_scratch_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm;

    /**
     * @brief Add rows from build_table_buffer into the hash table. This
     * computes hashes for all rows in the build_table_buffer that don't already
     * have hashes and then adds rows starting from 'next_group', up to the
     * size of the build_table_buffer to the hash table.
     * This function is idempotent.
     * This is only used in the AGG case and is only called from
     * UpdateGroupsAndCombine.
     *
     */
    inline void RebuildHashTableFromBuildBuffer();

    /**
     * @brief Activate this partition as part of the FinalizeBuild process.
     * In the ACC case, if the partition is inactive, this simply moves the data
     * from build_table_buffer_chunked to build_table_buffer. We then mark the
     * partition as active.
     * In the AGG case, if the partition is inactive, this will call
     * UpdateGroupsAndCombine on the chunks in build_table_buffer_chunked. The
     * chunks in build_table_buffer_chunked are only freed and the partition is
     * only marked as active if UpdateGroupsAndCombine was successful on all
     * chunks. This ensures that in case of a OperatorPoolThresholdError, we can
     * simply re-partition and retry.
     *
     */
    void ActivatePartition();

    /**
     * @brief Clear the "build" state. This releases
     * all memory associated with the logical hash table.
     * This is meant to be called in FinalizeBuild.
     *
     */
    void ClearBuildState();
};

class GroupbyIncrementalShuffleMetrics {
   public:
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    // Number of times we reset the hash table because it grew too large.
    stat_t n_ht_reset = 0;
    stat_t peak_ht_size_bytes = 0;
    // Number of times we reset the hashes buffer because it grew too large.
    stat_t n_hashes_reset = 0;
    stat_t peak_hashes_size_bytes = 0;

    // Stats related to dynamically determining if we want to insert into
    // the hash table before shuffling.
    stat_t n_possible_shuffle_reductions = 0;
    stat_t n_shuffle_reductions = 0;
    stat_t n_pre_reduction_hashes_reset = 0;
    stat_t n_pre_reduction_buffer_reset = 0;
    // Time spent appending to the shuffle buffer in the aggregate path.
    time_t shuffle_agg_buffer_append_time = 0;

    /// Local reduction stats

    // Time spent in additional hashing in the nunique-only case
    time_t nunique_hll_hashing_time = 0;
    // Time spent getting the HLL estimate to determine whether or not to do a
    // local reduction
    time_t hll_time = 0;
    // Num of times we ended up doing a local reduction
    stat_t n_local_reductions = 0;
    // Number of input/output rows from local reduction.
    // Can be used to measure how effective the local reduction was.
    stat_t local_reduction_input_nrows = 0;
    stat_t local_reduction_output_nrows = 0;
    time_t local_reduction_time = 0;  // Overall

    /// UpdateGroupsAndCombine

    // This is the time spent in the ReserveTable and the update_groups_helper
    // loop.
    time_t shuffle_update_logical_ht_time = 0;
    time_t shuffle_combine_input_time = 0;
    stat_t shuffle_update_logical_ht_and_combine_nrows = 0;

    // Time spent running HLL estimations + insertions
    time_t shuffle_hll_time = 0;

    // Local reduction time breakdown for the MRNF case:
    GroupbyMetrics::AggUpdateMetrics local_reduction_mrnf_metrics;

    /**
     * @brief Helper function for exporting metrics during reporting steps in
     * GroupBy.
     *
     * @param metrics Vector of metrics to append to.
     * @param accumulate_before_update Whether we are in the
     * accumulate-before-update for generating the metrics.
     */
    void add_to_metrics(std::vector<MetricBase>& metrics,
                        const bool accumulate_before_update);
};

/**
 * @brief Extend the general shuffle state for streaming groupby.
 * In particular, for the incremental aggregation case, we need
 * to maintain the next-group-id, hash-table and hashes corresponding to the
 * shuffle-table.
 * In the nunique-only accumulate-input case, we need a sightly modified shuffle
 * step where we may call drop-duplicates on the shuffle table before the
 * shuffle.
 *
 */
class GroupbyIncrementalShuffleState : public IncrementalShuffleState {
   public:
    using shuffle_hash_table_t = grpby_hash_table_t</*is_local*/ false>;

    // Current number of groups
    int64_t next_group = 0;
    // Map row number to group number
    std::unique_ptr<shuffle_hash_table_t> hash_table;
    // Hashes of data in table_buffer
    bodo::vector<uint32_t> groupby_hashes;
    // Temporary batch data (used by HashGroupbyTable and KeyEqualGroupbyTable)
    std::shared_ptr<table_info> in_table = nullptr;
    std::shared_ptr<uint32_t[]> in_table_hashes = nullptr;

    /**
     * @brief Constructor. Same as the base class constructor.
     *
     * @param shuffle_table_schema_ Schema of the shuffle table.
     * @param dict_builders_ Dictionary builders for the top level columns.
     * @param col_sets_ Column sets for performing reductions.
     * @param mrnf_n_sort_cols Number of columns to sort based off of in the
     * MRNF case
     * @param n_keys_ Number of key columns (to shuffle based off of).
     * @param curr_iter_ Reference to the iteration counter from parent
     * operator. e.g. In Groupby, this is 'build_iter'. For HashJoin, this could
     * be either 'build_iter' or 'probe_iter' based on whether it's the
     * build_shuffle_state or probe_shuffle_state, respectively.
     * @param sync_freq_ Reference to the synchronization frequency variable of
     * the parent state. This will be modified by this state adaptively (if
     * enabled).
     * @param nunique_only_ Whether this is the nunique-only accumulate input
     * case.
     * @param agg_type_ Is this MRNF, WINDOW, or regular Aggregation?
     * @param f_running_value_offsets_ Running value offsets for the Groupby.
     * This gives access for updating the hash table.
     */
    GroupbyIncrementalShuffleState(
        const std::shared_ptr<bodo::Schema> shuffle_table_schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
        const uint64_t mrnf_n_sort_cols_, const uint64_t n_keys_,
        const uint64_t& curr_iter_, int64_t& sync_freq_, int64_t op_id_,
        const bool nunique_only_, const AggregationType agg_type_,
        const std::vector<int32_t>& f_running_value_offsets_,
        const bool accumulate_before_update_);

    virtual ~GroupbyIncrementalShuffleState() = default;

    /**
     * @brief Clears the "shuffle" state. This releases
     * the memory of the logical shuffle hash table, i.e.
     * the shuffle buffer, shuffle hash-table and shuffle
     * hashes.
     * This is meant to be called in GroupbyState::FinalizeBuild.
     */
    void Finalize() override;

    /**
     * @brief Export shuffle metrics into the provided vector.
     *
     * @param[in, out] metrics Vector to append the metrics to.
     */
    void ExportMetrics(std::vector<MetricBase>& metrics) override {
        IncrementalShuffleState::ExportMetrics(metrics);
        this->metrics.add_to_metrics(metrics, this->accumulate_before_update);
    }

    /**
     * @brief Reset metrics. This is useful for the Union case where we want to
     * do this after every pipeline.
     *
     */
    void ResetMetrics() override {
        IncrementalShuffleState::ResetMetrics();
        this->metrics = GroupbyIncrementalShuffleMetrics();
    }

    /**
     * @brief Determine if our shuffle buffers exceed the shuffle size after
     * doing any reduction on the data in the accumulate path. To avoid wasted
     * compute we determine if we are doing a reduction based on a running HLL
     * value.
     * @param is_last Have we reached the last iteration and therefore must
     * shuffle if there is any data.
     * @return If our buffers exceed the shuffle size threshold after
     * processing.
     */
    bool ShouldShuffleAfterProcessing(bool is_last) override;

    friend class GroupbyState;

   protected:
    /// Metrics
    GroupbyIncrementalShuffleMetrics metrics;
    /// @brief Shuffle data buffer for data that we may reduce before shuffling,
    /// but has not yet been inserted into the hash table.
    std::unique_ptr<TableBuildBuffer> pre_reduction_table_buffer;
    bodo::vector<uint32_t> pre_reduction_hashes;

    /// @brief Threshold for what percentage of rows must be predicated to be
    /// unique before we skip doing a local reduction on shuffle. This is only
    /// considered for the incremental aggregation code path.
    double agg_reduction_threshold;

    /**
     * @brief Helper for ShuffleIfRequired. This is the same implementation as
     * the base class, except for the nunique-only and mrnf only cases. In the
     * nunique-only case, we may drop-duplicates from the shuffle-table before
     * the shuffle. In the MRNF-only case we may perform a local reduction
     * before the shuffle.
     *
     * @return std::tuple<
     * std::shared_ptr<table_info>,
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
     * std::shared_ptr<uint32_t[]>,
     * std::unique_ptr<uint8_t[]>>
     */
    std::tuple<
        std::shared_ptr<table_info>,
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
        std::shared_ptr<uint32_t[]>, std::unique_ptr<uint8_t[]>>
    GetShuffleTableAndHashes() override;

    /**
     * @brief Reset the shuffle state after a shuffle.
     * This clears the shuffle hash table, resets the shuffle buffer,
     * clears the shuffle hashes vector and resets shuffle_next_group
     * back to 0.
     * Note that this doesn't usually release memory.
     * If the hash table gets larger than MAX_SHUFFLE_HASHTABLE_SIZE,
     * we reset it to reduce peak memory. We do the same with the hashes
     * vector based on MAX_SHUFFLE_TABLE_SIZE.
     *
     */
    void ResetAfterShuffle() override;

   private:
    /// Column sets from the parent GroupbyState, used in some local reductions
    /// before shuffling
    const std::vector<std::shared_ptr<BasicColSet>> col_sets;
    /// Number of columns to sort based off of in the MRNF case
    const uint64_t mrnf_n_sort_cols;

    // If we keep adding more of these we should convert to an enum
    /// @brief Whether this is the nunique-only case.
    const bool nunique_only = false;
    /// Whether this is the mrnf-only case.
    const AggregationType agg_type = AggregationType::AGGREGATE;

    hll::HyperLogLog running_reduction_hll;

    /// @brief State shared with the Groupby State for updating the hash table.
    const std::vector<int32_t>& f_running_value_offsets;
    const bool accumulate_before_update;

    /**
     * @brief Update the groups in the hash table buffer and combine with values
     * from the input batch.
     *
     * NOTE: Only used in the aggregating code path (i.e.
     * accumulate_before_update = false)
     *
     * @param in_table Input batch to update and combine using.
     * @param batch_hashes_groupby Groupby hashes for the input batch records.
     */
    void UpdateGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby);
};

/**
 * @brief Metrics for Groupby Output stage. In particular, this contains metrics
 * for any work-redistribution that might've been done.
 *
 */
struct GroupbyOutputStateMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    // Global
    stat_t n_ranks_done_at_timer_start = -1;
    stat_t n_ranks_done_before_work_redistribution = -1;
    stat_t num_shuffles = 0;

    // Local
    time_t redistribute_work_total_time = 0;
    time_t determine_redistribution_time = 0;
    time_t determine_batched_send_counts_time = 0;
    stat_t num_recv_rows = 0;
    stat_t num_sent_rows = 0;
    time_t shuffle_data_prep_time = 0;
    time_t shuffle_dict_unification_time = 0;
    time_t shuffle_time = 0;
    time_t shuffle_output_append_time = 0;
};

/**
 * @brief Wrapper around the CTB that holds the output of a Groupby/MRNF/Window
 * operator. This also enables work-stealing during the output production stage,
 * if enabled.
 *
 */
class GroupbyOutputState {
   public:
    // Output buffer and associated dict-builders.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    ChunkedTableBuilder buffer;
    // Iteration counter for the output production stage, must be updated by the
    // caller.
    uint64_t iter = 0;

    GroupbyOutputState(
        const std::shared_ptr<bodo::Schema>& schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        size_t chunk_size,
        size_t max_resize_count_for_variable_size_dtypes =
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        bool enable_work_stealing_ = false);

    ~GroupbyOutputState();

    /**
     * @brief Finalize the output appends during the normal Groupby Build stage.
     * If work-stealing is enabled, this will only finalize the active chunk
     * since we may need to append more batches later after redistributing work.
     *
     */
    void Finalize();

    /**
     * @brief Return an output batch and whether it was the final one. This will
     * also perform work-stealing if it's required and enabled.
     *
     * @param produce_output Whether we should return an actual batch or just a
     * dummy batch.
     * @return std::tuple<std::shared_ptr<table_info>, bool>
     */
    std::tuple<std::shared_ptr<table_info>, bool> PopBatch(
        const bool produce_output);

    /**
     * @brief Export output redistribution metrics into the provided vector.
     *
     * @param metrics
     */
    void ExportMetrics(std::vector<MetricBase>& metrics);

   private:
    /// Work-stealing state

    // Whether work-stealing is enabled.
    const bool enable_work_stealing = false;
    // Debug mode for work-stealing.
    bool debug_work_stealing = false;
    // Whether we've started the timer for work-stealing. This is started when a
    // threshold fraction of ranks are done.
    bool work_steal_timer_started = false;
    // Stores the timestamp when the work-stealing timer was started. While this
    // is stored on all ranks, we only use the value on rank-0 for any decisions
    // to avoid issues related to clock synchronization.
    time_pt work_steal_start_time;
    // Whether work stealing has been done.
    bool work_stealing_done = false;
    // Sync frequency for work-stealing checks.
    uint64_t work_stealing_sync_iter = DEFAULT_SYNC_ITERS;
    // The number of ranks that must be done after which we will start the
    // work-stealing timer.
    uint64_t work_stealing_num_ranks_done_threshold = 1;
    // The time since the work-stealing timer was started after which we will
    // start redistributing the data.
    // XXX This is a constant for now, but we should eventually consider the
    // amount of work left and the amount of skew since it's possible that just
    // one rank has 2 batches left, in which case work-stealing is not very
    // useful. We might also want to consider how much slower it is compared to
    // the "finished" ranks. e.g. If other ranks finished in 1s on average, then
    // we should probably start work stealing very soon.
    uint64_t work_stealing_timer_threshold_us = 30 * 1000 * 1000;
    // Maximum number of batches that any rank can send/receive in a single
    // shuffle during redistribution. This is to protect against OOM errors.
    int64_t work_stealing_max_batched_send_recv_per_rank = -1;

    // Parallel work stealing decision state

    // Whether work stealing code decided to perform work redistribution
    bool performed_work_redistribution = false;

    // Work stealing "command" output from rank 0
    bool should_steal_work = false;

    // A separate communicator for work stealing decision messages to avoid
    // message conflicts with other operators
    MPI_Comm mpi_comm;

    // Work stealing "command" broadcast communication state
    MPI_Request steal_work_bcast_request = MPI_REQUEST_NULL;
    bool steal_work_bcast_started = false;
    bool steal_work_bcast_done = false;

    // Work stealing "done" message state
    MPI_Request done_request = MPI_REQUEST_NULL;
    bool done_sent = false;

    // Work stealing management state on rank 0
    bool recvs_posted = false;
    bool command_sent = false;
    std::vector<MPI_Request> done_recv_requests;
    // Dummy buffer for posting done receives
    std::unique_ptr<bool[]> done_recv_buff;
    std::vector<bool> done_received;
    int num_ranks_done = 0;

    ///

    // MPI number of processes and this process' rank.
    int n_pes, myrank;

    // Metrics
    GroupbyOutputStateMetrics metrics;

    /**
     * @brief If work-stealing is enabled, this synchronizes output production
     * state across all ranks and initiates work-redistribution if required.
     *
     */
    void StealWorkIfNeeded();

    /**
     * @brief Helper function that manages work stealing state on rank 0 and
     broadcasts work stealing "command" to all ranks (do work stealing or not).
     *
     */
    void manage_work_stealing_rank_0();

    /**
     * @brief Helper to redistribute the remaining batches between ranks if
     * we've determined that work-stealing is required.
     *
     */
    void RedistributeWork();

    /**
     * @brief Static helper to determine how many batches should be moved from
     * every rank to every other rank. Currently, this tries to move data evenly
     * from all ranks with excess data to all ranks with remaining capacity.
     *
     * @param num_batches_ranks A vector of size 'n_pes' which contains the
     * number of output batches remaining on each rank.
     * @return std::vector<std::vector<size_t>> A matrix of shape n_pes x n_pes
     * specifying the number of batches to move.
     */
    static std::vector<std::vector<size_t>> determine_redistribution(
        const std::vector<uint64_t>& num_batches_ranks);

    /**
     * @brief Static helper to determine how many batches to send from every
     * rank to every other rank during a batched shuffle/redistribution while
     * ensuring that no rank sends or receives any more than
     * max_batches_send_recv_per_rank batches.
     *
     * @param[in, out] batches_to_send_overall Number of remaining batches we
     * need to send from every rank to every other rank. This is updated
     * in-place. The update is essentially an element-wise subtraction of the
     * output matrix.
     * @param max_batches_send_recv_per_rank The maximum number of batches any
     * rank can send/receive.
     * @return std::vector<std::vector<size_t>> A matrix of shape n_pes x n_pes
     * specifying the number of batches to move in this shuffle iteration.
     */
    static std::vector<std::vector<size_t>> determine_batched_send_counts(
        std::vector<std::vector<size_t>>& batches_to_send_overall,
        const size_t max_batches_send_recv_per_rank);

    /**
     * @brief Is another shuffle required. This just checks if there are any
     * non-0 values in 'batches_to_send', i.e. are there any batches that still
     * need to be moved.
     * TODO Replace this with a simpler check, potentially in
     * 'determine_batched_send_counts'.
     *
     * @param batches_to_send Number of batches remaining to be sent from every
     * rank to every other rank.
     * @return true Another round of shuffle is required.
     * @return false All redistribution is done.
     */
    static bool needs_another_shuffle(
        const std::vector<std::vector<size_t>>& batches_to_send);

    /**
     * @brief Helper for performing a round of shuffle/redistribution.
     *
     * @param to_send_this_iter n_pes x n_pes matrix specifying how many batches
     * need to be moved from one rank to every other rank in this round.
     * @param redistribution_tbb TBB to use accumulating the shuffle batches
     * that need to be sent. We pass this in to reduce the number allocations by
     * re-using it between all shuffle rounds. This is cleared after use.
     * @return std::tuple<std::shared_ptr<table_info>, size_t> Received table
     * and number of rows sent to other ranks.
     */
    std::tuple<std::shared_ptr<table_info>, size_t> redistribute_batches_helper(
        const std::vector<std::vector<size_t>>& to_send_this_iter,
        std::shared_ptr<TableBuildBuffer>& redistribution_tbb);
};

class GroupbyState {
   private:
    // NOTE: These need to be declared first so that they are
    // removed at the very end during destruction.

    /// @brief OperatorBufferPool for this operator.
    const std::unique_ptr<bodo::OperatorBufferPool> op_pool;
    /// @brief Memory manager for op_pool. This is used during buffer
    /// allocations.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;

    /// @brief OperatorScratchPool corresponding to the op_pool.
    const std::unique_ptr<bodo::OperatorScratchPool> op_scratch_pool;
    /// @brief Memory manager for op_scratch_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm;

   public:
    // Current stage ID. 0 is for Initialization.
    // For regular Groupby, build is stage 1 and produce_output is stage 2.
    // For Union, there is one build stage per table, i.e. from 1 to
    // num_union_tables. produce_output is stage 'num_union_tables + 1'.
    uint32_t curr_stage_id = 0;
    // Partitioning information.
    std::vector<std::shared_ptr<GroupbyPartition>> partitions;
    // Partition state: Tuples of the form (num_top_bits, top_bitmask).
    // We maintain this for testing purposes since this information is
    // otherwise lost during FinalizeBuild which frees the partitions.
    std::vector<std::pair<size_t, uint32_t>> partition_state;

    // TODO Decide this dynamically using a heuristic based
    // on total available memory, total disk space, etc.
    const size_t max_partition_depth;

    const uint64_t n_keys;
    bool parallel;
    const int64_t output_batch_size;

    // drop_na flag in DataFrame library
    const bool pandas_drop_na;

    std::vector<std::shared_ptr<BasicColSet>> col_sets;

    // Shuffle state
    std::unique_ptr<GroupbyIncrementalShuffleState> shuffle_state;

    // indices of input columns for each function
    // f_in_offsets contains the offsets into f_in_cols.
    // f_in_cols is a list of physical column indices.
    // For example:
    //
    // f_in_offsets = (0, 1, 6)
    // f_in_cols = (0, 7, 1, 3, 4, 0)
    // The first function uses the columns in f_in_cols[0:1]. IE physical index
    // 0 in the input table. The second function uses the column f_in_cols[1:6].
    // IE physical index 7, 1, 3, 4, 0 in the input table.
    const std::vector<int32_t> f_in_offsets;
    const std::vector<int32_t> f_in_cols;

    // Indices of update and combine columns for each function.
    // Applicable to both the ACC and AGG cases. In the AGG case,
    // these are the offsets in the build_table_buffer directly.
    // In the ACC case, these are the offsets into the update
    // table returned by 'get_update_table</*is_acc_case*/ true>'.
    // This is not used in the MRNF case since MRNF goes through
    // a special Finalize path.
    std::vector<int32_t> f_running_value_offsets;

    // Min-Row Number Filter (MRNF) specific attributes.
    AggregationType agg_type = AggregationType::AGGREGATE;
    const std::vector<bool> sort_asc;
    const std::vector<bool> sort_na;
    const std::vector<bool> cols_to_keep_bitmask;

    // The number of iterations between syncs (adjusted by
    // shuffle_state)
    int64_t sync_iter;

    // Current iteration of build steps
    uint64_t build_iter = 0;

    // Accumulating all values before update is needed
    // when one of the groupby functions is
    // median/nunique/...
    // similar to shuffle_before_update in non-streaming groupby
    // see:
    // https://bodo.atlassian.net/wiki/spaces/B/pages/1346568245/Vectorized+Groupby+Design#Getting-All-Group-Data-before-Computation
    bool accumulate_before_update = false;
    bool req_extended_group_info = false;

    // True if all ColSet functions are nunique, which enables optimization of
    // dropping duplicate shuffle table rows before shuffle
    bool nunique_only = false;

    // This is used in groupby_agg_build_consume_batch and
    // groupby_acc_build_consume_batch. We're caching this allocation for
    // performance reasons.
    std::vector<bool> append_row_to_build_table;

    // Output state
    // This will be lazily initialized during the end of the build step to
    // simplify specifying the output column types.
    // TODO(njriasan): Move to initialization information.
    std::shared_ptr<GroupbyOutputState> output_state = nullptr;
    // By default, enable work-stealing for window and disable it for regular
    // groupby (+ MRNF). This can be overriden by explicitly setting
    // BODO_STREAM_WINDOW_DISABLE_OUTPUT_WORK_STEALING and
    // BODO_STREAM_GROUPBY_ENABLE_OUTPUT_WORK_STEALING.
    bool enable_output_work_stealing_groupby = false;
    bool enable_output_work_stealing_window = true;

    // Simple concatenation of key_dict_builders and
    // non key dict builders. The key_dict_builders will be shared between the
    // build_shuffle_buffer, build_table_buffers of all partitions and the
    // output buffer. Key dict builders are always at the beginning of the
    // vector, and non-key dict builders follow. For all columns, if the array
    // type is not dict encoded, the value is nullptr These will be shared
    // between build_table_buffers of all partitions.
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the final steps.
    bool build_input_finalized = false;

    /// @brief Whether we should print debug information
    /// about partitioning such as when a partition is split.
    bool debug_partitioning = false;

    /// @brief Whether partitioning is currently enabled.
    bool partitioning_enabled = true;

    GroupbyMetrics metrics;
    const int64_t op_id;

    // Group By histogram information that can be used by the
    // accumulate path. This is general to any shuffle operation
    // that "blocks" on having a total table calculation, so it
    // should eventually be moved to a helper class.
    std::vector<int64_t> histogram_buckets;
    uint64_t num_histogram_bits = 0;
    bool compute_histogram = false;

    // The IBarrier request used for is_last synchronization
    MPI_Request is_last_request = MPI_REQUEST_NULL;
    bool is_last_barrier_started = false;
    bool global_is_last = false;
    MPI_Comm shuffle_comm;

    GroupbyState(
        const std::unique_ptr<bodo::Schema>& in_schema_,
        std::vector<int32_t> ftypes_, std::vector<int32_t> window_ftypes_,
        std::vector<int32_t> f_in_offsets_, std::vector<int32_t> f_in_cols_,
        uint64_t n_keys_, std::vector<bool> sort_asc_vec_,
        std::vector<bool> sort_na_pos_, std::vector<bool> cols_to_keep_bitmask_,
        std::shared_ptr<table_info> window_args, int64_t output_batch_size_,
        bool parallel_, int64_t sync_iter_, int64_t op_id_,
        int64_t op_pool_size_bytes_, bool allow_any_work_stealing = true,
        std::optional<std::vector<std::shared_ptr<DictionaryBuilder>>>
            key_dict_builders_ = std::nullopt,
        bool use_sql_rules = true, bool pandas_drop_na_ = false);

    ~GroupbyState() { MPI_Comm_free(&this->shuffle_comm); }

    /**
     * @brief Unify dictionaries of input table with build table
     * (build_table_buffer of all partitions and build_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @param only_keys only unify key columns
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with build table dictionaries.
     */
    std::shared_ptr<table_info> UnifyBuildTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table, bool only_keys = false);

    /**
     * @brief Get dictionary hashes of dict-encoded string key columns (nullptr
     * for other key columns).
     * NOTE: output vector does not have values for data columns (length is
     * this->n_keys).
     *
     * @return
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
     */
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
    GetDictionaryHashesForKeys();

    /**
     * @brief Update the groups and combine with values from the input batch.
     * It will figure out the correct partition based on the partitioning
     * hashes.
     *
     * It is slightly optimized for the case where there's a single partition.
     *
     * NOTE: Only used in the aggregating code path (i.e.
     * accumulate_before_update = false)
     *
     * In case of a threshold enforcement error, this process is retried after
     * splitting the 0th partition (which is only one that the enforcement error
     * could've come from). This is done until we successfully append the batch
     * or we go over max partition depth while splitting the 0th partition.
     *
     * @param in_table Input batch to update and combine using.
     * @param partitioning_hashes Partitioning hashes for the records.
     * @param batch_hashes_groupby Groupby hashes for the input batch records.
     */
    void UpdateGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby);

    /**
     * @brief Same as above, except we will only append rows for which
     * append_rows bit-vector is true.
     *
     * @param in_table Input batch to update and combine using.
     * @param partitioning_hashes Partitioning hashes for the records.
     * @param batch_hashes_groupby Groupby hashes for the input batch records.
     * @param append_rows Bitmask specifying the rows to use.
     */
    void UpdateGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
        const std::vector<bool>& append_rows);

    /**
     * @brief Update the groups in the shuffle buffer and combine with values
     * from the input batch. Depending on the function this may not actually
     * insert into the hash table while it waits to do a HLL estimate.
     *
     * NOTE: Only used in the aggregating code path (i.e.
     * accumulate_before_update = false)
     *
     * @param in_table Input batch to update and combine using.
     * @param batch_hashes_groupby Groupby hashes for the input batch records.
     * @param append_rows Bitmask specifying the rows to use.
     */
    void UpdateShuffleGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
        const std::vector<bool>& append_rows);

    /**
     * @brief Append the input batch to the build_tables of the partitions. The
     * partition that a row belongs to is determined using the partitioning
     * hashes.
     * It is slightly optimized for the single partition case.
     *
     * In case of a threshold enforcement error, this process is retried after
     * splitting the 0th partition (which is only one that the enforcement error
     * could've come from). This is done until we successfully append the batch
     * or we go over max partition depth while splitting the 0th partition.
     *
     * NOTE: Only used in the accumulating code path (i.e.
     * accumulate_before_update = true)
     *
     * @param in_table Input batch to append.
     * @param partitioning_hashes Partitioning hashes for the input batch
     * records.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /**
     * @brief Same as above, except we only append the rows specified by
     * the append_rows bitmask.
     *
     * @param in_table Input batch to append rows from.
     * @param partitioning_hashes Partitioning hashes for the input batch
     * @param append_rows Bitmask specifying the rows to append.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

    /**
     * @brief Initialize the output buffer in the MRNF case using
     * the schema of the dummy_build_table. The dummy_build_table
     * is the 'build_table_buffer' from any of the GroupbyPartitions in this
     * GroupbyState.
     * We will use 'cols_to_keep' to
     * determine the columns to retain.
     *
     * NOTE: The function is idempotent and only initializes once. All
     * calls after the first one are ignored.
     *
     * @param dummy_build_table Underlying table_info of 'build_table_buffer'
     * from any of the partitions.
     */
    void InitOutputBufferMrnf(
        const std::shared_ptr<table_info>& dummy_build_table);

    /**
     * @brief Initialize the output buffer in the Window case using
     * the schema of the dummy_build_table. The dummy_build_table
     * is the 'build_table_buffer' from any of the GroupbyPartitions in this
     * GroupbyState. This function will then generate the logic to add the
     * window columns to the output buffer.
     * We will use 'cols_to_keep' to
     * determine the columns to retain.
     *
     * NOTE: The function is idempotent and only initializes once. All
     * calls after the first one are ignored.
     *
     * @param dummy_build_table Underlying table_info of 'build_table_buffer'
     * from any of the partitions.
     */
    void InitOutputBufferWindow(
        const std::shared_ptr<table_info>& dummy_build_table);

    /**
     * @brief Initialize the output buffer using schema information
     * from the dummy table.
     *
     * @param dummy_table Dummy table to extract schema information for the
     * output buffer from.
     */
    void InitOutputBuffer(const std::shared_ptr<table_info>& dummy_table);

    /**
     * @brief Unify dictionaries of output columns with output buffer.
     * NOTE: key columns are not unified since they already have the same
     * dictionary as input
     *
     * @param out_table output table to unify
     * @return std::shared_ptr<table_info> output table with unified data
     * columns
     */
    std::shared_ptr<table_info> UnifyOutputDictionaryArrays(
        const std::shared_ptr<table_info>& out_table);

    /**
     * @brief Get global is_last flag given local is_last using ibarrier
     *
     * @param local_is_last local is_last flag (all input is consumed). This
     * assumes that local_is_last will always be true after the first iteration
     * it is set to true.
     */
    bool GetGlobalIsLast(bool local_is_last);

    /**
     * @brief Report the current set of build stage metrics and reset them in
     * preparation for the next stage. The multiple stages are only relevant in
     * the UNION case, but this serves as a nice generalization.
     *
     */
    void ReportAndResetBuildMetrics(bool is_final);

    /**
     * @brief Report the metrics for the output production stage. This primarily
     * consists of metrics related to any work redistribution that might've been
     * performed during execution.
     *
     */
    void ReportOutputMetrics();

    /**
     * @brief Finalize the build step. This will finalize all the partitions,
     * append their outputs to the output buffer, clear the build state and set
     * build_input_finalized to prevent future repetitions of the build step.
     *
     */
    void FinalizeBuild();

    /**
     * @brief Disable partitioning by disabling threshold
     * enforcement in the OperatorBufferPool.
     *
     * If threshold enforcement is disabled, the
     * OperatorPoolThresholdExceededError error will never
     * be thrown and we will never trigger a partition
     * split.
     *
     */
    void DisablePartitioning();

    /**
     * @brief Enable (or re-enable) partitioning by enabling
     * threshold enforcement in the OperatorBufferPool.
     *
     * Note that in the case where we are re-enabling
     * partitioning, if the memory usage has overall increased
     * and gone beyond the threshold since disabling it, this
     * could raise the OperatorPoolThresholdExceededError error.
     *
     */
    void EnablePartitioning();

    /// @brief Get the number of bytes allocated through this Groupby operator's
    /// OperatorBufferPool that are currently pinned.
    uint64_t op_pool_bytes_pinned() const;

    /// @brief Get the number of bytes that are currently allocated through this
    /// Groupby operator's OperatorBufferPool.
    uint64_t op_pool_bytes_allocated() const;

    /**
     * @brief Get a string representation of the partitioning state.
     * This is used for Query Profile.
     *
     * @return std::string
     */
    std::string GetPartitionStateString() const;

   private:
    /**
     * Helper function that gets the running column types for a given function.
     * This is used to initialize the build state. Currently, this creates a
     * dummy colset, and calls getRunningValueColumnTypes on it. This is pretty
     * ugly, but it works for now. window_ftype is used for populating any
     * window related colsets, which currently only support a single window
     * function.
     */
    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        std::vector<std::shared_ptr<array_info>> local_input_cols,
        std::vector<std::unique_ptr<bodo::DataType>>&& in_dtypes, int ftype,
        int window_ftype, std::shared_ptr<table_info> window_args);

    /**
     * Helper function that gets the output column types for a given function.
     * This is used to initialize the build state. Implemented in a similar
     * fashion to getRunningValueColumnTypes. window_ftype is used for
     * populating any window related colsets, which currently only support a
     * single window function.
     */
    std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumns(
        std::vector<std::shared_ptr<array_info>> local_input_cols, int ftype,
        int window_ftype);

    /*@brief Split the partition at index 'idx' into two partitions.
     * This must only be called in the event of a threshold enforcement error.
     *
     *@param idx Index of the partition(in this->partitions) to split.
     */
    void SplitPartition(size_t idx);

    /// @brief Helpers for AppendBuildBatch with most of the core logic
    /// for appending a build batch to the Groupby state. These are the
    /// functions that are retried in AppendBuildBatch in case the 0th partition
    /// needs to be split to fit within the memory threshold assigned to this
    /// partition.
    void AppendBuildBatchHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

    void AppendBuildBatchHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /// @brief Helpers for UpdateGroupsAndCombine with most of the core logic
    /// for appending a build batch to the Groupby state. These are the
    /// functions that are retried in UpdateGroupsAndCombine in case the 0th
    /// partition needs to be split to fit within the memory threshold assigned
    /// to this partition.
    void UpdateGroupsAndCombineHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby);

    void UpdateGroupsAndCombineHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
        const std::vector<bool>& append_rows);

    /**
     * @brief Clear the "build" state. This releases
     * all memory except for the output buffer.
     * The output buffer is the only thing we will need
     * during the output production stage, so all other
     * memory can be released.
     * This is meant to be called in FinalizeBuild.
     *
     */
    void ClearBuildState();

    /**
     * @brief Clear the state of all the ColSets
     * by calling '.clear()' on them. This is
     * used in FinalizeBuild when we encounter a
     * threshold enforcement error from the op-pool.
     *
     */
    void ClearColSetsStates();

    /// Snapshot of the combined metrics from the key columns.
    /// We will "subtract" these from the key dict-builder metrics
    /// to get the metrics for every subsequent stage. This is only required for
    /// the UNION case where there may be multiple pipelines. UNION only has key
    /// columns, so we only need to keep a snapshot of those metrics.
    DictBuilderMetrics key_dict_builder_metrics_prev_stage_snapshot;

    /**
     * @brief Determine if based on the histogram information
     * the partition described by the given number of bits
     * and bit mask will always exceed the provided threshold
     * in the largest partition. This threshold should always
     * be a value between 0 and 1.
     *
     * @param num_bits The number of bits for the partition.
     * @param bitmask The bitmask for the partition.
     * @param threshold The threshold for disabling partitioning.
     * @return true Partitioning should be disabled.
     * @return false Partitioning should be enabled.
     */
    bool MaxPartitionExceedsThreshold(size_t num_bits, uint32_t bitmask,
                                      double threshold);
};

struct GroupingSetsMetrics {
    using stat_t = MetricBase::StatValue;
    stat_t output_row_count = 0;
};

/**
 * @brief State for a group by operator that is used as a wrapper
 * around several GroupbyState objects.
 */
class GroupingSetsState {
   public:
    GroupingSetsState(
        std::unique_ptr<bodo::Schema> keys_schema_,
        std::vector<std::unique_ptr<GroupbyState>> groupby_states_,
        std::vector<std::vector<int64_t>> input_columns_remaps_,
        std::vector<std::vector<int64_t>> output_columns_remaps_,
        std::vector<std::vector<int64_t>> missing_output_columns_remaps_,
        std::vector<int64_t> grouping_output_idxs_,
        std::vector<std::vector<int64_t>> grouping_values_,
        std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders_,
        int64_t op_id_)
        : op_id(op_id_),
          keys_schema(std::move(keys_schema_)),
          groupby_states(std::move(groupby_states_)),
          input_columns_remaps(std::move(input_columns_remaps_)),
          output_columns_remaps(std::move(output_columns_remaps_)),
          missing_output_columns_remaps(
              std::move(missing_output_columns_remaps_)),
          grouping_output_idxs(std::move(grouping_output_idxs_)),
          grouping_values(std::move(grouping_values_)),
          key_dict_builders(std::move(key_dict_builders_)) {
        this->current_output_idx = groupby_states.size() - 1;
    }

    /**
     * @brief Consume a batch of the input data and dispatch to each group by
     * state. Returns a pair of booleans. The first boolean indicates if the
     * input was the final input across all states. The second boolean indicates
     * if any state wants us to pause the input.
     *
     * @param input_table The input table to consume.
     * @param is_last Is this our local last entry.
     * @return std::pair<bool, bool> The pair of outputs for determining if we
     * are globally done receiving inputs (with false negatives) and if we
     * should pause the input.
     */
    std::pair<bool, bool> ConsumeBuildBatch(
        std::shared_ptr<table_info> input_table, bool is_last);

    /**
     * @brief Produce an output batch of data from one of the group by states.
     * This includes the logic to remap the columns to the final output schema,
     * inserting nulls where necessary.
     *
     * @param produce_output Should output be produced.
     * @return std::pair<std::shared_ptr<table_info>, bool> The output table and
     * if this is the final output.
     */
    std::pair<std::shared_ptr<table_info>, bool> ProduceOutputBatch(
        bool produce_output);

    const int64_t op_id;

    GroupingSetsMetrics metrics;

   private:
    // The schema of keys so we can output null values of the correct type.
    const std::unique_ptr<bodo::Schema> keys_schema;
    // States to dispatch to for each group by operation.
    const std::vector<std::unique_ptr<GroupbyState>> groupby_states;
    // Mapping on the build side to go from the general input to the subset
    // used by the groupby_states.
    const std::vector<std::vector<int64_t>> input_columns_remaps;
    // Mapping from the output type from each group by state to its locations
    // in the final output type. This is largely the same as the
    // input_columns_remaps but the data columns are replaced with the output
    // columns from the functions.
    const std::vector<std::vector<int64_t>> output_columns_remaps;
    // Mapping of which columns are missing from the output type.
    const std::vector<std::vector<int64_t>> missing_output_columns_remaps;
    // Locations where "GROUPING" values need to be inserted.
    const std::vector<int64_t> grouping_output_idxs;
    // Values for each "evaluated" grouping function.
    const std::vector<std::vector<int64_t>> grouping_values;

    bool finalized_output = false;
    // Index for tracking which group by state we are currently producing
    // output for.
    int current_output_idx;
    // Dictionary builders that are shared between all group by states.
    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;
};

/**
 * @brief Logic to consume a build table batch. This is called
 * directly by groupby_build_consume_batch_py_entry to avoid
 * complex exception handling with grouping sets.
 *
 * @param groupby_state groupby state pointer
 * @param in_table build table batch
 * @param is_last is last batch (in this pipeline) locally
 * @param is_final_pipeline Is this the final pipeline. Only relevant for the
 * Union-Distinct case where this is called in multiple pipelines. For regular
 * groupby, this should always be true. We only call FinalizeBuild in the last
 * pipeline.
 * @param[out] request_input whether to request input rows from preceding
 * operators.
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool groupby_build_consume_batch(GroupbyState* groupby_state,
                                 std::shared_ptr<table_info> input_table,
                                 bool is_last, const bool is_final_pipeline,
                                 bool* request_input);

/**
 * @brief Function to produce an output table called directly from
 * Python. This handles all the functionality separately for exception
 * handling with grouping sets.
 *
 * @param groupby_state groupby state pointer
 * @param[out] out_is_last is last batch
 * @param produce_output whether to produce output
 * @return table_info* output table batch
 */
std::shared_ptr<table_info> groupby_produce_output_batch_wrapper(
    GroupbyState* groupby_state, bool* out_is_last, bool produce_output);
