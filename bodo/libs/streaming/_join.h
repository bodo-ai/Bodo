#pragma once
#include <unordered_set>

#include "../_bodo_common.h"
#include "../_chunked_table_builder.h"
#include "../_dict_builder.h"
#include "../_join.h"
#include "../_operator_pool.h"
#include "../_pinnable.h"
#include "../_query_profile_collector.h"
#include "../_table_builder.h"
#include "../vendored/simd-block-fixed-fpp.h"
#include "_shuffle.h"

using BloomFilter = SimdBlockFilterFixed<::hashing::SimpleMixSplit>;

// Default threshold for Join operator's OperatorBufferPool
#define JOIN_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD 0.5

// Use all available memory by default
#define JOIN_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

#define JOIN_MAX_PARTITION_DEPTH 15

// The default maximum number of unique join keys allowed before
// the join build stops trying to build the hash set of unique keys.
#define DEFAULT_UNIQUE_VALUES_LIMIT 20

// Chunk size of build_table_buffer_chunked and probe_table_buffer_chunked in
// JoinPartition
#define INACTIVE_PARTITION_TABLE_CHUNK_SIZE 16 * 1024

// Stage IDs for build and probe for the QueryProfileCollector
#define QUERY_PROFILE_BUILD_STAGE_ID 1
#define QUERY_PROFILE_PROBE_STAGE_ID 2

class JoinPartition;
struct HashHashJoinTable {
    /**
     * provides row hashes for join hash table (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRow row number
     * @return hash of row iRow
     */
    uint32_t operator()(const int64_t iRow) const;
    JoinPartition* join_partition;

    HashHashJoinTable(JoinPartition* join_partition)
        : join_partition(join_partition) {}
};

struct KeyEqualHashJoinTable {
    /**
     * provides row comparison for join hash table (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true if equal else false
     */
    bool operator()(const int64_t iRowA, const int64_t iRowB) const;
    JoinPartition* join_partition;
    const uint64_t n_keys;

    KeyEqualHashJoinTable(JoinPartition* join_partition, const uint64_t n_keys)
        : join_partition(join_partition), n_keys(n_keys) {}
};

/**
 * @brief Struct for storing the Join metrics.
 *
 */
struct JoinMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;
    // Required Metrics for all Joins
    stat_t build_input_row_count = 0;
    stat_t probe_input_row_count = 0;
    stat_t probe_output_row_count = 0;

    // Time spent dealing with min/max updates
    time_t build_min_max_update_time = 0;
    // Time spent dealing with the min/max parallel finalization
    time_t build_min_max_finalize_time = 0;
};

/**
 * @brief Struct for storing HashJoin metrics.
 *
 */
struct HashJoinMetrics : public JoinMetrics {
    ///// Build

    // Global: Did we do a Broadcast hash join (0: false, 1: true)
    stat_t is_build_bcast_join = 0;
    // Is the build side unique on the current rank (0: false, 1: true)
    stat_t is_build_unique = 0;
    //  If so, time spent in allgather, etc.
    time_t build_bcast_time = 0;
    // Global: Did we use a bloom filter (0: false, 1: true)
    stat_t bloom_filter_enabled = 0;

    /// Time spent in appends to TBBs and CTBs of the partitions.
    time_t build_appends_active_time = 0;
    stat_t build_appends_active_nrows = 0;
    time_t build_appends_inactive_time = 0;
    stat_t build_appends_inactive_nrows = 0;

    // Time spent finding the partition for each row in the
    // multi-partition case.
    time_t build_input_partition_check_time = 0;
    stat_t build_input_partition_check_nrows = 0;

    /// Time spent in BuildHashTable
    time_t build_ht_hashing_time = 0;  // Hashing
    stat_t build_ht_hashing_nrows = 0;
    time_t build_ht_insert_time = 0;  // Inserting into hashmap
    stat_t build_ht_insert_nrows = 0;

    /// Partition stats
    stat_t n_partitions = 1;    // Number of partitions
    stat_t build_nrows = 0;     // Number of rows across all partitions
    stat_t build_n_groups = 0;  // Number of "groups" across all partitions

    /// Number of partition splits
    stat_t n_repartitions_in_appends = 0;
    stat_t n_repartitions_in_finalize = 0;

    /// Time spent repartitioning
    time_t repartitioning_time = 0;  // Overall
    time_t repartitioning_part_hashing_time = 0;
    stat_t repartitioning_part_hashing_nrows = 0;
    time_t repartitioning_active_part1_append_time = 0;
    stat_t repartitioning_active_part1_append_nrows = 0;
    time_t repartitioning_active_part2_append_time = 0;
    stat_t repartitioning_active_part2_append_nrows = 0;
    // nrows for this is the same as repartitioning_part_hashing_nrows:
    time_t repartitioning_inactive_append_time = 0;
    // Most of the cost of PopChunk is expected to be from disk IO.
    time_t repartitioning_inactive_pop_chunk_time = 0;
    stat_t repartitioning_inactive_n_pop_chunks = 0;

    // Time spent finalizing
    time_t build_finalize_time = 0;           // Overall
    time_t build_finalize_groups_time = 0;    // Time spent in FinalizeGroups
    time_t build_finalize_activate_time = 0;  // Time spent in ActivatePartition
    // NOTE: We don't measure the time spent in BuildHashTable since it's
    // already measured separately ('build_ht_hashing_time' and
    // 'build_ht_insert_time')

    // Time spent computing partition hashes in the main loop. 'nrows' for this
    // is build_filter_na_output_nrows + total number of build
    // shuffle output rows.
    time_t build_input_part_hashing_time = 0;
    // Time spent adding to the bloom filter. 'nrows' for this is
    // build_filter_na_output_nrows
    time_t bloom_filter_add_time = 0;
    // Time spent in bloom filter union reduction
    time_t bloom_filter_union_reduction_time = 0;

    // (Estimated) Size of the largest partition
    stat_t max_partition_size_bytes = 0;
    // Total size (estimated) of all partitions
    stat_t total_partitions_size_bytes = 0;
    // Final op budget (local)
    stat_t final_op_pool_size_bytes = 0;

    // Time spent in filter_na_values
    time_t build_filter_na_time = 0;
    stat_t build_filter_na_output_nrows = 0;

    // Time spent dealing with unique values updates
    time_t build_unique_values_update_time = 0;

    // Time spent dealing with the unique values parallel finalization
    time_t build_unique_values_finalize_time = 0;

    // Partitioning state
    blob_t final_partitioning_state;

    // Snapshot of the combined metrics from the key columns.
    // We will "subtract" these from the final key dict-builder metrics
    // to get the metrics for the probe stage.
    DictBuilderMetrics key_dict_builder_metrics_build_stage_snapshot;

    ///// Probe

    // Track the total probe table rows processed on this rank.
    stat_t num_processed_probe_table_rows = 0;

    // Time spent in appends (AppendInactiveProbeBatch)
    time_t append_probe_inactive_partitions_time = 0;
    stat_t append_probe_inactive_partitions_nrows = 0;
    // Time spent finding the partitions in the multi partition case
    // (AppendProbeBatchToInactivePartition). 'nrows' for this is the same as
    // append_probe_inactive_partitions_nrows.
    time_t probe_inactive_partition_check_time = 0;
    // Time spent probing the hash table / finding group-id for the rows.
    // 'nrows' for this is num_processed_probe_table_rows.
    time_t ht_probe_time = 0;

    // Time spent in produce_probe_output, except the time spent appending
    // output to the output buffer. In the non-equi condition case, this
    // includes the time to evaluate the cond_func. 'nrows' for this is
    // effectively num_processed_probe_table_rows.
    time_t produce_probe_out_idxs_time = 0;
    // Time spent in generate_build_table_outer_rows_for_partition. 'nrows' for
    // this is 'build_nrows'.
    time_t build_outer_output_idx_time = 0;

    // Time spent computing join hashes. 'nrows' for this is
    // probe_filter_na_output_nrows + total size of the shuffle outputs.
    time_t probe_join_hashing_time = 0;
    // Time spent computing partition hashes. 'nrows' for this is
    // probe_filter_na_output_nrows + (total size of the shuffle outputs if >1
    // partitions else 0).
    time_t probe_part_hashing_time = 0;

    /// Time spent in finalizing the inactive partitions.
    time_t finalize_probe_inactive_partitions_time = 0;  // Overall
    // Time spent "pinning" the partitions for finalization
    time_t finalize_probe_inactive_partitions_pin_partition_time = 0;
    // Time spent in reading back inactive probe chunks during
    // FinalizeProbeForInactivePartition
    time_t probe_inactive_pop_chunk_time = 0;
    stat_t probe_inactive_pop_chunk_n_chunks = 0;

    // Time spent in filter_na_values. The input count is the total number of
    // local input rows.
    time_t probe_filter_na_time = 0;
    stat_t probe_filter_na_output_nrows = 0;

    // Number of rows pruned out by the bloom filter in the probe outer case.
    // 'nrows' for this is probe_filter_na_output_nrows.
    stat_t probe_outer_bloom_filter_misses = 0;

    ///// JoinFilter (only relevant in the probe-inner case)
    stat_t num_runtime_filter_applied_rows = 0;
    // Time spent hashing for bloom filter
    time_t join_filter_bloom_filter_hashing_time = 0;
    stat_t join_filter_bloom_filter_hashing_nrows = 0;
    // Time spent probing bloom filter
    time_t join_filter_bloom_filter_probe_time = 0;
    stat_t join_filter_bloom_filter_probe_nrows = 0;
};

/**
 * @brief Holds the state of a single partition during
 * a join execution. This includes the build table buffer,
 * the hashtable (unord_map_container), bitmap of the matches
 * in build records, etc.
 * 'top_bitmask' and 'num_top_bits' define the partition
 * itself, i.e. a record is in this partition if the top
 * 'num_top_bits' bits of its hash are 'top_bitmask'.
 *
 */
class JoinPartition {
   public:
    using hash_table_t =
        bodo::unord_map_container<int64_t, size_t, HashHashJoinTable,
                                  KeyEqualHashJoinTable>;
    explicit JoinPartition(
        size_t num_top_bits_, uint32_t top_bitmask_,
        const std::shared_ptr<bodo::Schema> build_table_schema_,
        const std::shared_ptr<bodo::Schema> probe_table_schema_,
        const uint64_t n_keys_, bool build_table_outer_,
        bool probe_table_outer_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            build_table_dict_builders_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            probe_table_dict_builders_,
        bool is_active_, HashJoinMetrics& metrics_,
        bodo::OperatorBufferPool* op_pool_,
        const std::shared_ptr<::arrow::MemoryManager> op_mm_,
        bodo::OperatorScratchPool* op_scratch_pool_,
        const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm_,
        bool is_na_equal_ = false);

    // The types of the columns in the build table and probe tables.
    const std::shared_ptr<bodo::Schema> build_table_schema;
    const std::shared_ptr<bodo::Schema> probe_table_schema;
    // Dictionary builders for build and probe tables
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;
    std::vector<std::shared_ptr<DictionaryBuilder>> probe_table_dict_builders;

    // Contiguous append-only buffer, used if a partition is active or after
    // the Finalize step for inactive partitions.
    std::unique_ptr<TableBuildBuffer> build_table_buffer;
    // These allocations will go through the OperatorBufferPool
    // so that we can enforce limits and trigger re-partitioning.
    bodo::pinnable<bodo::vector<uint32_t>> build_table_join_hashes;
    // Pin guard which is populated and stored in pin(). This is what
    // will be used for all access to this vector.
    std::optional<bodo::pin_guard<decltype(build_table_join_hashes)>>
        build_table_join_hashes_guard;

    // Chunked append-only buffer, only used if a partition is inactive and
    // not yet finalized.
    std::unique_ptr<ChunkedTableBuilder> build_table_buffer_chunked;

    // Join hash table (key row number -> matching row numbers).
    // These allocations will go through the OperatorBufferPool
    // so that we can enforce limits and trigger re-partitioning.
    std::unique_ptr<bodo::pinnable<hash_table_t>> build_hash_table;
    // Pin guard which is populated and stored in pin(). This is what
    // will be used for all access to this hash table.
    std::optional<bodo::pin_guard<bodo::pinnable<hash_table_t>>>
        build_hash_table_guard;

    // Temporary array to track the group sizes of build table
    // rows. It will be released in FinalizeGroups once groups and
    // groups_offsets are populated.
    std::unique_ptr<bodo::pinnable<bodo::vector<size_t>>> num_rows_in_group;
    std::optional<bodo::pin_guard<bodo::pinnable<bodo::vector<size_t>>>>
        num_rows_in_group_guard;
    // Temporary array to track the group id of every row in the build table.
    // This will be used in 'FinalizeGroups' to populate 'groups' and
    // 'groups_offsets'. Its memory will be released once this is done.
    // This array allows us to avoid a hash-map lookup. This does require more
    // memory, but the performance benefits of a simple vector lookup are
    // sufficient to warrant it.
    std::unique_ptr<bodo::pinnable<bodo::vector<size_t>>>
        build_row_to_group_map;
    std::optional<bodo::pin_guard<bodo::pinnable<bodo::vector<size_t>>>>
        build_row_to_group_map_guard;

    // 'groups' is single contiguous buffer of row ids arranged by groups.
    // 'group_offsets' store the offsets for the individual groups within the
    // 'groups' buffer (similar to how we store strings in array_info). We will
    // resize these to the exact required sizes and populate them using
    // 'num_rows_in_group' and 'build_row_to_group_map' during 'FinalizeGroups'.
    bodo::pinnable<bodo::vector<size_t>> groups;
    bodo::pinnable<bodo::vector<size_t>> groups_offsets;
    std::optional<bodo::pin_guard<decltype(groups)>> groups_guard;
    std::optional<bodo::pin_guard<decltype(groups_offsets)>>
        groups_offsets_guard;

    // Probe state (for outer joins). Note we don't use
    // vector<bool> because we may need to do an allreduce
    // on the data directly and that can't be accessed for bool.
    // These allocations will go through the OperatorBufferPool
    // so that we can enforce limits and trigger re-partitioning.
    bodo::pinnable<bodo::vector<uint8_t>>
        build_table_matched;  // state for building output table
    // Pin guard which is populated and stored in pin(). This is what
    // will be used for all access to this vector.
    std::optional<bodo::pin_guard<decltype(build_table_matched)>>
        build_table_matched_guard;

    /// @brief This is the number of rows that have been "safely"
    /// appended to the build_table_buffer. This is only relevant
    /// when the partition is active. This is used in SplitPartition
    /// to only split the first build_safely_appended_nrows rows
    /// of the build_table_buffer. This is required since there might
    /// be a memory error during "AppendBatch" which might happen
    /// _after_ we've appended to the build_table_buffer. In those
    /// cases, the append will be retried, so we need to ensure that
    /// previous incomplete append is invalidated.
    size_t build_safely_appended_nrows = 0;

    // Probe state (only used when this partition is inactive).
    // We don't need partitioning hashes since we should never
    // need to repartition.

    // Buffer to hold probe input for inactive partitions.
    // This is initialized only for all partitions except the 0th
    // partition in the first iteration of the probe loop
    // through the InitInactiveProbeInputBuffer API.
    std::unique_ptr<ChunkedTableBuilder> probe_table_buffer_chunked = nullptr;
    // TODO Convert this into a chunked buffer
    bodo::pinnable<bodo::vector<uint32_t>> probe_table_buffer_join_hashes;

    // Temporary state during probe step. These will be
    // reset between iterations.
    std::shared_ptr<table_info> probe_table;

    // Offset into the probe table hashes. This is used when processing
    // the probe shuffle buffer. We need to use this since we only have hashes
    // for the current probe shuffle buffer batch, but we need to use the global
    // row when adding the output to the output buffer.
    int64_t probe_table_hashes_offset = 0;
    // Dummy probe table. Useful for the build_table_outer case.
    std::shared_ptr<table_info> dummy_probe_table;
    // Join hashes corresponding to data in probe_table.
    // We're using a raw pointer here so we can populate this field using the
    // vector (using .data()) or the uint32_t[] shared_ptr directly (using
    // .get())
    uint32_t* probe_table_hashes;

    // Matches Pandas behavior by treating NA values as equal.
    const bool is_na_equal;

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
     * @brief Split the partition into 2^num_levels partitions.
     * This will produce a new set of partitions, each with their
     * new build_table_buffer and build_table_join_hashes.
     * The caller must explicitly rebuild the hash table on
     * the partition.
     * If the partition is active, this create one active and 2^num_levels - 1
     * inactive partitions. If the partition is inactive, this
     * creates 2^num_levels inactive partitions.
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     * @tparam is_active Is this an active partition.
     * @param num_levels Number of levels to split the partition. Only '1' is
     * supported at this point.
     * @return std::vector<std::shared_ptr<JoinPartition>>
     */
    template <bool is_active>
    std::vector<std::shared_ptr<JoinPartition>> SplitPartition(
        size_t num_levels = 1);

    /**
     * @brief Add rows from build_table_buffer into the hash table. This
     * computes hashes for all rows in the build_table_buffer that don't already
     * have hashes and then adds rows starting from curr_build_size, up to the
     * size of the build_table_buffer to the hash table.
     * It will also populate 'num_rows_in_groups' and 'build_row_to_group_map'
     * vectors. This function is idempotent.
     * NOTE: The function assumes that the partition is pinned when it's called.
     *
     */
    inline void BuildHashTable();

    /**
     * @brief This will populate 'groups' and 'groups_offsets' using
     * 'num_rows_in_groups' and 'build_row_to_group_map'. Once this
     * is done, it will free 'num_rows_in_groups' and
     * 'build_row_to_group_map'.
     * This function is idempotent, but not incremental, i.e.
     * it must be called after we have seen all the data for this
     * partition and have populated the hash table.
     * This is meant to be called during FinalizeBuild and after
     * the partition has been activated and all rows have been added
     * to the hash table.
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     */
    void FinalizeGroups();

    /**
     * @brief Add all rows from in_table to this partition.
     * For inactive partitions, this function simply appends
     * the rows to the chunked build buffer and the hashes
     * to the hashes vector.
     * For active partitions, we first reserve space in the
     * build_table_buffer and then append the rows into it.
     * We also populate the hash table using BuildHashTable.
     * To make this function "retry-able", we also
     * build the hash table as the first step. In most cases,
     * this will be a NOP. However, in the case that the function
     * is being retried due to a threshold enforcement error,
     * this will act as a way to get the partition to the correct
     * state before appending new entries.
     * To make it transactional, we update build_safely_appended_nrows
     * only after the rows have been added to the build_table_buffer,
     * the hashes vector _and_ the hash table.
     *
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     * @tparam is_active Is this the active partition.
     * @param in_table Table to insert.
     */
    template <bool is_active>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Same as the other function, except this time we
     * also supply a bitmap for the rows to append. Note that
     * we reserve space for the entire table before appending,
     * and not just for the required rows.
     *
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     * @tparam is_active Is this the active partition.
     * @param in_table Table to insert.
     * @param append_rows Vector of booleans indicating whether to append the
     * row
     */
    template <bool is_active>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table,
                          const std::vector<bool>& append_rows);

    /**
     * @brief Is the build table unique for the current partition, meaning
     * that every hash entry points to exactly 1 row. This is only
     * meaningful once all data has been added from the build side.
     * @return True if the data for the current partition must be unique
     * based on the hash keys.
     */
    bool IsUniqueBuildPartition();

    /**
     * @brief Finalize the build step for this partition.
     * This will activate the partition (if not already active), i.e.
     * transfer data from the chunked build table to the contiguous
     * build table, build the hash table, finalize the groups
     * and initialize the 'build_table_matched' bitmap in the
     * build_table_outer case.
     * This function and its steps are idempotent.
     *
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     */
    void FinalizeBuild();

    /**
     * @brief Initialize probe_table_buffer_chunked.
     * This is a NOP if it is already initialized.
     *
     */
    void InitProbeInputBuffer();

    /**
     * @brief Append a batch of data into the probe table buffer.
     * Note that this is only used for inactive partitions
     * to buffer the inputs before we start processing them.
     *
     * @param in_table Table from which we're adding the row.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash for the record.
     * @param append_row Whether to append the row.
     * @param in_table_start_offset Starting offset into the 'in_table'. e.g. if
     * 'append_rows' is [t, f, t, f, t] and 'in_table_start_offset' is 4, then
     * we will append rows 4, 6 & 8 from the input table. Note that this offset
     * only applies to the table and not to the join hashes.
     * @param table_nrows Number of rows to process from 'in_table' (starting
     * from 'in_table_start_offset'). If it's set to -1, we process all rows
     * starting from 'in_table_start_offset'.
     */
    void AppendInactiveProbeBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::vector<bool>& append_rows,
        const int64_t in_table_start_offset = 0, int64_t table_nrows = -1);

    /**
     * @brief Process the records in the probe table buffer
     * and produce the outputs for this partition. The outputs
     * will be appended to the provided 'output_buffer'.
     * Note that this is only used for inactive (non index 0)
     * partitions.
     *
     * @tparam build_table_outer
     * @tparam probe_table_outer
     * @tparam non_equi_condition
     * @param cond_func Condition function for the non-equi condition case,
     * nullptr otherwise.
     * @param build_kept_cols Which columns to generate in the output on the
     * build side.
     * @param probe_kept_cols Which columns to generate in the output on the
     * probe side.
     * @param[in, out] output_buffer The output buffer of the join state that
     * we should append the output to.
     *
     * NOTE: The function assumes that the partition is pinned
     * when it's called.
     *
     */
    template <bool build_table_outer, bool probe_table_outer,
              bool non_equi_condition>
    void FinalizeProbeForInactivePartition(
        cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols,
        const std::shared_ptr<ChunkedTableBuilder>& output_buffer);

    /**
     * @brief Activate this partition as part of the FinalizeBuild process.
     * If the partition is inactive, this simply moves the data from
     * build_table_buffer_chunked to build_table_buffer. We then mark the
     * partition as active and update build_safely_appended_nrows.
     * NOTE: This does not populate the hash map.
     * NOTE: The function assumes that the partition is pinned when it's called.
     */
    void ActivatePartition();

    /**
     * @brief Pin this partition into memory.
     * This essentially pins the build table buffer
     * and creates pin guards (and therefore pinning)
     * for the hash table, the join hashes, the groups
     * and group offsets vectors and the other temporary
     * vectors like num_rows_in_group and build_row_to_group_map.
     *
     */
    void pin();

    /**
     * @brief Unpin this partition. This essentially unpins
     * the build table buffer and releases the pin guards for
     * the hash table, join hashes, etc.
     */
    void unpin();

   private:
    const size_t num_top_bits = 0;
    const uint32_t top_bitmask = 0ULL;
    const bool build_table_outer = false;
    const bool probe_table_outer = false;
    const uint64_t n_keys;
    // OperatorBufferPool of the HashJoin operator that this is a partition in.
    // The pool is owned by the parent HashJoinState, so we keep a raw pointer
    // for simplicity.
    bodo::OperatorBufferPool* const op_pool;
    // Memory manager instance for op_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;
    /// @brief OperatorScratchPool corresponding to the op_pool.
    bodo::OperatorScratchPool* const op_scratch_pool;
    /// @brief Memory manager instance for op_scratch_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm;
    /// @brief Whether the partition is active / has been activated yet.
    /// The 0th partition is active to begin with, and other partitions
    /// are activated at the end of the build step.
    /// When false, this means that the build data is still in the
    /// 'build_table_buffer_chunked'. When true, this means that the data has
    /// been moved to the 'build_table_buffer', but the hash table
    /// may or may not have been populated yet. To populate the hash table
    /// use BuildHashTable, which is an idempotent function and hence safe to
    /// call it multiple times.
    bool is_active = false;
    // Tracks the current size of the build hash table, i.e.
    // the number of rows from the build_table_buffer
    // that have been added to the hash table.
    // This is only meaningful once the partition has been
    // activated (i.e. is_active = true).
    size_t curr_build_size = 0;
    /// Whether the partition is currently pinned.
    bool pinned_ = false;
    // Whether the 'groups' and 'groups_offsets' for this partition are already
    // populated and finalized. This is used in FinalizeGroups to make it
    // idempotent.
    bool finalized_groups = false;

    // Reference to the metrics (shared with parent HashJoinState and all other
    // partitions).
    HashJoinMetrics& metrics;
};

/**
 * @brief Helper for UnifyBuildTableDictionaryArrays and
 * UnifyProbeTableDictionaryArrays. Unifies dictionaries of input table with
 * dictionaries in dict_builders by appending its new dictionary values to
 * buffer's dictionaries and transposing input's indices.
 *
 * @param in_table input table
 * @param dict_builders Dictionary builders to unify with. The dict builders
 * will be appended with the new values from dictionaries in input_table.
 * @param n_keys number of key columns
 * @param only_transpose_existing_on_key_cols For key columns, whether or not to
 * only transpose the values that already exist in the dictionary-builder (and
 * set the rest as nulls) instead of full unification which would append any
 * values not already in the dictionary-builder to the dictionary-builder. This
 * is used in the Probe-Inner case in Join since we statically know that the
 * dictionary should not grow because if a value doesn't already exist, it will
 * get filtered out by the Join anyway.
 * @return std::shared_ptr<table_info> input table with dictionaries unified
 * with build table dictionaries.
 */
std::shared_ptr<table_info> unify_dictionary_arrays_helper(
    const std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    uint64_t n_keys, bool only_transpose_existing_on_key_cols = false);

class JoinState {
   public:
    // The types of the columns in the build table and probe tables.
    const std::shared_ptr<bodo::Schema> build_table_schema;
    const std::shared_ptr<bodo::Schema> probe_table_schema;
    // Join properties
    const uint64_t n_keys;
    cond_expr_fn_t cond_func;
    const bool build_table_outer;
    const bool probe_table_outer;
    const bool force_broadcast;
    // Matches Pandas behavior by treating NA values as equal.
    const bool is_na_equal;
    // Note: This isn't constant because we may change it
    // via broadcast decisions.
    bool build_parallel;
    const bool probe_parallel;
    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the outer
    // join output.
    bool build_input_finalized = false;
    bool probe_input_finalized = false;
    // Is the build table empty on all ranks. Should only be
    // computed in FinalizeBuild() for each supported join type.
    bool global_build_empty = false;
    const int64_t output_batch_size;
    // The number of iterations between syncs
    int64_t sync_iter = DEFAULT_SYNC_ITERS;
    // Current iteration of the build and probe steps
    uint64_t build_iter = 0;
    uint64_t probe_iter = 0;

    // Dictionary builders for the key columns. This is
    // always of length n_keys and is nullptr for non DICT keys.
    // These will be shared between the build_table_buffers and
    // probe_table_buffers of all partitions and the build_shuffle_buffer
    // and probe_shuffle_buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;

    // Simple concatenation of key_dict_builders and
    // non key dict builders
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Simple concatenation of key_dict_builders and
    // non key dict builders
    std::vector<std::shared_ptr<DictionaryBuilder>> probe_table_dict_builders;

    // Output buffer
    // This will be lazily initialized during the probe step
    // since we don't know the required columns (after column
    // pruning) until then.
    std::shared_ptr<ChunkedTableBuilder> output_buffer = nullptr;

    // Dummy probe table. Useful for the build_table_outer case.
    std::shared_ptr<table_info> dummy_probe_table;

    // Vector that stores an array for each possible build column that
    // can generate a minimum or maximum value. Each array
    // has 2 values: the minimum key, and the maximum key.
    std::vector<std::optional<std::shared_ptr<array_info>>> min_max_values;

    // Vector that stores the bitmask of each dictionary encoded build column
    // indicating which indices were hit 1+ times on the build side. For
    // non dictionary encoded columns, the vectors are empty.
    std::vector<std::vector<bool>> build_dict_hit_bitmap;

    // Metrics for join execution.
    JoinMetrics metrics;

    // The IBarrier request used for is_last synchronization
    MPI_Request is_last_request = MPI_REQUEST_NULL;
    bool is_last_barrier_started = false;
    bool global_is_last = false;
    MPI_Comm shuffle_comm;

    // Non-blocking join probe reduce for outer build bitmap communication state
    MPI_Request probe_reduce_request = MPI_REQUEST_NULL;
    bool global_probe_reduce_done = false;
    bool probe_reduce_started = false;

    JoinState(const std::shared_ptr<bodo::Schema> build_table_schema_,
              const std::shared_ptr<bodo::Schema> probe_table_schema_,
              uint64_t n_keys_, bool build_table_outer_, bool force_broadcast_,
              bool probe_table_outer_, cond_expr_fn_t cond_func_,
              bool build_parallel_, bool probe_parallel_,
              int64_t output_batch_size_, int64_t sync_iter_, int64_t op_id_,
              bool is_na_equal_ = false);

    virtual ~JoinState() {}

    virtual void FinalizeBuild() {
        if (!this->build_input_finalized) {
            this->build_input_finalized = true;
            // Report metrics to the QueryProfileCollector
            std::vector<MetricBase> metrics;
            this->ReportBuildStageMetrics(metrics);
            QueryProfileCollector::Default().RegisterOperatorStageMetrics(
                QueryProfileCollector::MakeOperatorStageID(
                    this->op_id, QUERY_PROFILE_BUILD_STAGE_ID),
                std::move(metrics));
        }
    }

    virtual void FinalizeProbe() {
        this->output_buffer->Finalize(/*shrink_to_fit*/ true);
        if (!this->probe_input_finalized) {
            this->probe_input_finalized = true;
            // Report metrics to the QueryProfileCollector
            std::vector<MetricBase> metrics;
            this->ReportProbeStageMetrics(metrics);
            QueryProfileCollector::Default().RegisterOperatorStageMetrics(
                QueryProfileCollector::MakeOperatorStageID(
                    this->op_id, QUERY_PROFILE_PROBE_STAGE_ID),
                std::move(metrics));
        }
    }

    virtual void InitOutputBuffer(const std::vector<uint64_t>& build_kept_cols,
                                  const std::vector<uint64_t>& probe_kept_cols);

    /**
     * @brief Unify dictionaries of input table with build table
     * (build_table_buffer of all partitions and build_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with build table dictionaries.
     */
    std::shared_ptr<table_info> UnifyBuildTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Unify dictionaries of input table with probe table
     * (probe_table_buffer of all partitions and probe_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @param only_transpose_existing_on_key_cols See description of the same
     * parameter in the 'unify_dictionary_arrays_helper' function for details.
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with probe table dictionaries.
     */
    std::shared_ptr<table_info> UnifyProbeTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table,
        bool only_transpose_existing_on_key_cols = false);

    // Operator ID.
    const int64_t op_id;

    /**
     * Returns whether a key column of a hash join can have a runtime
     * join filter based on its min/max values.
     */
    static bool IsValidRuntimeJoinFilterMinMaxColumn(
        std::unique_ptr<bodo::DataType>& dtype);

    /**
     * Processes a batch of data to include it in the accumulated min/max values
     * for a specific key column.
     *
     * @param[in] arr The batch of data that is being processed to update
     * the min/max values of a column.
     * @param[in] col_idx: which column to insert.
     */
    void UpdateKeysMinMax(const std::shared_ptr<array_info>& arr,
                          size_t col_idx);

    /**
     * Helper for UpdateKeysMinMax on string arrays.
     *
     * @param[in] in_arr: the string array.
     * @param[in] col_idx: the column index having its data inserted.
     */
    void UpdateKeysMinMaxString(const std::shared_ptr<array_info>& in_arr,
                                size_t col_idx);

    /**
     * Helper for UpdateKeysMinMax on dictionary encoded arrays.
     *
     * @param[in] in_arr: the dictionary encoded array.
     * @param[in] col_idx: the column index having its data inserted.
     */
    void UpdateKeysMinMaxDict(const std::shared_ptr<array_info>& in_arr,
                              size_t col_idx);

    /**
     * Does the parallel finalization of the min/max values for each key column
     * to ensure that ever rank has the global min/max of the key columns before
     * any runtime join filters.
     */
    void FinalizeKeysMinMax();

   protected:
    /**
     * @brief Helper function to report metrics from the build stage to the
     * QueryProfileCollector. This called in FinalizeBuild. At this point, this
     * is a NOP. Children classes are expected to override this.
     *
     */
    virtual void ReportBuildStageMetrics(std::vector<MetricBase>& metrics_out);

    /**
     * @brief Helper function to report metrics from the probe stage to the
     * QueryProfileCollector. This is called in FinalizeBuild. Currently, this
     * just reports some metrics from the output buffer. Children classes are
     * expected to override and extend this.
     *
     */
    virtual void ReportProbeStageMetrics(std::vector<MetricBase>& metrics_out);
};

class HashJoinState : public JoinState {
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
    // Partitioning information.
    std::vector<std::shared_ptr<JoinPartition>> partitions;

    // TODO Decide this dynamically using a heuristic based
    // on total available memory, total disk space, etc.
    const size_t max_partition_depth;

    // Shuffle states
    IncrementalShuffleState build_shuffle_state;
    IncrementalShuffleState probe_shuffle_state;

    // Global bloom-filter. This is built during the build step
    // and used during the probe step.
    std::unique_ptr<BloomFilter> global_bloom_filter;

    // Keep a table of NA keys for bypassing the hash table
    // if we have an outer join and any keys can contain NAs.
    ChunkedTableBuilder build_na_key_buffer;
    // How many NA values have we seen. This is used for consistent
    // partitioning if the build table is replicated and the probe table
    // distributed. This is unused if the build table is distributed or
    // the final output is replicated.
    size_t build_na_counter = 0;
    // Same as build_na_counter but for probe NAs
    size_t probe_na_counter = 0;

    /// @brief Whether we should print debug information
    /// about partitioning such as when a partition is split.
    bool debug_partitioning = false;

    // Vector that stores an array for each equi-join key. Each set
    // stores the unique values encountered so far (as type int64_t). If
    // it ever goes above a certain threshold, replace with std::nullopt
    // to indicate that a low-cardinality filter is not possible
    std::vector<std::optional<std::unordered_set<int64_t>>> unique_values;

    // The magic number used to determine how many unique values are allowed in
    // the build side of a join before a low-NDV runtime join filter is aborted.
    size_t unique_values_limit = DEFAULT_UNIQUE_VALUES_LIMIT;

    HashJoinState(const std::shared_ptr<bodo::Schema> build_table_schema_,
                  const std::shared_ptr<bodo::Schema> probe_table_schema_,
                  uint64_t n_keys_, bool build_table_outer_,
                  bool probe_table_outer_, bool force_broadcast_,
                  cond_expr_fn_t cond_func_, bool build_parallel_,
                  bool probe_parallel_, int64_t output_batch_size_,
                  int64_t sync_iter_, int64_t op_id_,
                  // If -1, we'll use 100% of the total buffer
                  // pool size. Else we'll use the provided size.
                  int64_t op_pool_size_bytes = -1,
                  size_t max_partition_depth_ = JOIN_MAX_PARTITION_DEPTH,
                  bool is_na_equal_ = false);

    ~HashJoinState() { MPI_Comm_free(&this->shuffle_comm); }

    /**
     * @brief Create a global bloom filter for this Hash Join
     * operation. This will return a nullptr in case bloom
     * filters are not supported on this architecture.
     *
     * @return std::unique_ptr<BloomFilter>
     */
    std::unique_ptr<BloomFilter> create_bloom_filter() {
        if (bloom_filter_supported()) {
            // Estimate the number of rows to specify based on
            // the target size in bytes in the env or 1MiB
            // if not provided.
            int64_t target_bytes = 1 * 1024 * 1024;
            char* env_target_bytes =
                std::getenv("BODO_STREAM_JOIN_BLOOM_FILTER_TARGET_BYTES");
            if (env_target_bytes) {
                target_bytes = std::stoi(env_target_bytes);
            }
            int64_t num_entries = num_elements_for_bytes(target_bytes);
            return std::make_unique<BloomFilter>(num_entries);
        } else {
            return nullptr;
        }
    }

    /**
     * @brief Clear the existing partition(s) and replace with a single
     * partition with the correct type information. This creates equivalent
     * partition state as when HashJoinState is initialized except there
     * may be some additional dictionary builder information.
     *
     */
    void ResetPartitions();

    /**
     * @brief Append a batch of rows. It will figure out the correct
     * partition based on the partitioning hash. If the record
     * is in the "active" (i.e. index 0) partition, it will be
     * added to the hash table of that active partition
     * as well. If record belongs to an inactive partition, it
     * will be simply added to the build buffer of the partition.
     * It is slightly optimized for the single partition case.
     *
     * In case of a threshold enforcement error, this process is retried after
     * splitting the 0th partition (which is only one that the enforcement error
     * could've come from). This is done until we successfully append the batch
     * or we go over max partition depth while splitting the 0th partition.
     *
     * @param in_table Table to add the rows from.
     * @param partitioning_hashes Partitioning hashes for the records.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /**
     * @brief Same as the function above, except we will only
     * append rows for which append_rows bit vector is true.
     *
     * @param in_table Table to add the rows from.
     * @param partitioning_hashes Partitioning hashes for the records.
     * @param append_rows Vector of booleans indicating whether to append the
     * row
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

    void InitOutputBuffer(
        const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols) override;

    /**
     * @brief Get a string representation of the partitioning state.
     * This is used for Query Profile.
     *
     * @return std::string
     */
    std::string GetPartitionStateString() const;

    /**
     * @brief Finalize build step for all partitions.
     * This will process the partitions one by one (only one is pinned in memory
     * at one time), build hash tables, split partitions as necessary, etc.
     * We call JoinPartition::FinalizeBuild on each of the partitions.
     * In case we encounter a threshold enforcement error during FinalizeBuild
     * of a partition, that partition is split until it can be finalized
     * without going over the pinned memory threshold. This process will be
     * repeated until we are successful or until we reach the max partition
     * depth.
     *
     */
    void FinalizeBuild() override;

    /**
     * @brief Apply a runtime filter to a table based on information about the
     * key columns collected during the build stage. At this time, we apply two
     * types of filters:
     * 1. Bloom filter: This is applied when none of the entries in
     * join_key_idxs are -1.
     * 2. DictionaryBuilder: If a key is a dictionary-encoded column, then we
     * can use the DictionaryBuilder for that column to filter out rows where
     * the entry in the corresponding column doesn't exist in the dictionary.
     *
     * NOTE: This function is effectively a const. The only reason it cannot be
     * marked as const is because the transpose cache of the DictionaryBuilder
     * is modified. No actual changes are made to the JoinState in this
     * function.
     * @param in_table Table to filter. We assume this is a small batch (e.g. 4K
     * rows).
     * @param row_bitmask Bitmask representing which rows to keep after
     * filtering. Expects array_type=NULLABLE_INTO_BOOL, dtype=BOOL.
     * @param join_key_idxs A vector of length this->n_keys. The element at the
     * i'th index is the index of the column in the input table that corresponds
     * to the i'th join key. -1 indicates that a column corresponding to the
     * join key doesn't exist. e.g. If we have 3 keys and join_key_idxs is [2,
     * 0, -1], then in_table->columns[2] corresponds to the first join key,
     * in_table->columns[0] corresponds to the second join key and there's no
     * column corresponding to the third join key. The bloom filter is applied
     * when none of the entries are -1, i.e. we have all key columns. When this
     * is the case, this must also be the last instance of RuntimeJoinFilter.
     * @param process_col_bitmask A vector length this->n_keys that specifies
     * whether or not to apply a column level filter for the i'th join key. This
     * is required because ideally we only want to apply the column level filter
     * only once.
     * @return bool whether any filters were applied.
     */
    bool RuntimeFilter(std::shared_ptr<table_info> in_table,
                       std::shared_ptr<array_info> row_bitmask,
                       const std::vector<int64_t>& join_key_idxs,
                       const std::vector<bool>& process_col_bitmask);

    /**
     * @brief Finalize any statistic information
     * needed when the probe step is finished.
     *
     */
    void FinalizeProbe() override;

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
     * @brief Initialize probe_table_buffer_chunked of
     * partitions 1 onwards (i.e. not the 0th).
     * This is meant to be called in the first probe
     * iteration, however, it is idempotent and can
     * be called again safely.
     *
     */
    void InitProbeInputBuffers();

    /**
     * @brief Append probe batch to the probe table buffer of the
     * appropriate inactive partition. This assumes that the row
     * is _not_ in the active (index 0) partition.
     *
     * @param in_table Table to add the record from.
     * @param row_ind Index of the row to append.
     * @param join_hash Join hash for the record.
     * @param partitioning_hash Partitioning hash for the record.
     * @param append_rows Whether to append the row
     * @param in_table_start_offset Starting offset into the 'in_table'. e.g. if
     * 'append_rows' is [t, f, t, f, t] and 'in_table_start_offset' is 4, then
     * we will append rows 4, 6 & 8 from the input table. Note that this offset
     * only applies to the table and not to the join hashes or the partitioning
     * hashes.
     * @param table_nrows Number of rows to process from 'in_table' (starting
     * from 'in_table_start_offset'). If it's set to -1, we process all rows
     * starting from 'in_table_start_offset'.
     */
    void AppendProbeBatchToInactivePartition(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows,
        const int64_t in_table_start_offset = 0, int64_t table_nrows = -1);

    /**
     * @brief Finalize Probe step for all the inactive partitions.
     * This will insert all the outputs directly into the output_buffer.
     *
     * @tparam build_table_outer
     * @tparam probe_table_outer
     * @tparam non_equi_condition
     * @param build_kept_cols Which columns to generate in the output on the
     * build side.
     * @param probe_kept_cols Which columns to generate in the output on the
     * probe side.
     */
    template <bool build_table_outer, bool probe_table_outer,
              bool non_equi_condition>
    void FinalizeProbeForInactivePartitions(
        const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols);

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
     * Returns whether a key column of a hash join can have a runtime
     * join filter based on its unique values.
     */
    static bool IsValidRuntimeJoinFilterUniqueValuesColumn(
        std::unique_ptr<bodo::DataType>& dtype);

    /**
     * Processes a batch of data to include it in the accumulated unique
     * values for a specific key column.
     *
     * @param[in] arr The batch of data that is being processed to update
     * the min/max values of a column.
     * @param[in] col_idx: which column to insert.
     */
    void UpdateKeysUniqueValues(const std::shared_ptr<array_info>& arr,
                                size_t col_idx);

    /**
     * Does the parallel finalization of the unique values for each key column
     * to ensure that ever rank has the global unique of the key columns before
     * any runtime join filters.
     */
    void FinalizeKeysUniqueValues();

    /// @brief Get the current budget of the op-pool for this Join operator.
    uint64_t op_pool_budget_bytes() const;

    /// @brief Get the number of bytes allocated through this Join operator's
    /// OperatorBufferPool that are currently pinned.
    uint64_t op_pool_bytes_pinned() const;

    /// @brief Get the number of bytes that are currently allocated through this
    /// Join operator's OperatorBufferPool.
    uint64_t op_pool_bytes_allocated() const;

    // Metrics for the query profile
    HashJoinMetrics metrics;

   private:
    /**
     * @brief Split the partition at index 'idx' into two partitions.
     * This must only be called in the event of a threshold enforcement error.
     *
     * @param idx Index of the partition (in this->partitions) to split.
     */
    void SplitPartition(size_t idx);

    /// @brief Helpers for AppendBuildBatch with most of the core logic
    /// for appending a build side batch to the Join state. These are the
    /// functions that are retried in AppendBuildBatch in case the 0th partition
    /// needs to be split to fit within the memory threshold assigned to this
    /// partition.
    void AppendBuildBatchHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    void AppendBuildBatchHelper(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

   protected:
    /// Override the metric reporting helpers to report the detailed metrics
    /// collected during the build and probe stages.
    void ReportBuildStageMetrics(std::vector<MetricBase>& metrics_out) override;
    void ReportProbeStageMetrics(std::vector<MetricBase>& metrics_out) override;
};

/**
 * @brief Struct for storing the NestedLoopJoin metrics.
 *
 */
struct NestedLoopJoinMetrics : public JoinMetrics {
    ///// Build

    // Block size used for cross join.
    stat_t block_size_bytes = 0;
    // Number of rows in each chunk of the build table.
    stat_t chunk_size_nrows = 0;
    // Time spent finalizing
    time_t build_finalize_time = 0;
    // Global: Whether we did a build allgather (0: false, 1: true)
    stat_t is_build_bcast_join = 0;
    // If so, time spent in bcast
    time_t build_bcast_time = 0;

    ///// Probe

    // Time spent making dicts global
    time_t probe_global_dict_unification_time = 0;
    // Total (approximate) bytes that were broadcasted by this rank to all other
    // ranks. In the parallel case, this is essentially the total size of all
    // the input batches. In the non-parallel case, we don't need to broadcast,
    // so this will be 0.
    stat_t probe_bcast_size_bytes = 0;
    // Time spent in bcast
    time_t probe_bcast_table_time = 0;
    // Time spent in ProcessProbeChunk
    time_t probe_compute_matches_time = 0;
    // Time spent adding unmatched rows from build in last iter
    time_t probe_add_unmatched_build_rows_time = 0;
};

class NestedLoopJoinState : public JoinState {
   public:
    // Build state
    std::unique_ptr<ChunkedTableBuilder>
        build_table_buffer;  // Append only buffer.
    bodo::pinnable<bodo::vector<uint8_t>>
        build_table_matched;  // state for building output
                              // table (for outer joins)

    NestedLoopJoinState(const std::shared_ptr<bodo::Schema> build_table_schema_,
                        const std::shared_ptr<bodo::Schema> probe_table_schema_,
                        bool build_table_outer_, bool probe_table_outer_,
                        std::vector<int64_t> interval_build_col_idxs,
                        bool force_broadcast_, cond_expr_fn_t cond_func_,
                        bool build_parallel_, bool probe_parallel_,
                        int64_t output_batch_size_, int64_t sync_iter_,
                        int64_t op_id_)
        : JoinState(
              build_table_schema_, probe_table_schema_,
              /*n_keys_*/ 0,  // NestedLoopJoin is only used when n_keys is 0
              build_table_outer_, probe_table_outer_, force_broadcast_,
              cond_func_, build_parallel_, probe_parallel_, output_batch_size_,
              sync_iter_, op_id_) {
        // TODO: Integrate dict_builders for nested loop join.
        this->sync_iter =
            this->sync_iter == -1 ? DEFAULT_SYNC_ITERS : this->sync_iter;

        // Use the default block size unless the env var is set.
        // The env var is primarily used for testing purposes for verifying
        // correct behavior when there are multiple chunks in the build
        // table buffer. Setting the env var to a small value allows us to
        // do this even with a small input.
        int64_t block_size_bytes = DEFAULT_BLOCK_SIZE_BYTES;
        char* block_size = std::getenv("BODO_CROSS_JOIN_BLOCK_SIZE");
        if (block_size) {
            block_size_bytes = std::stoi(block_size);
        }
        if (block_size_bytes <= 0) {
            throw std::runtime_error(
                "NestedLoopJoinState: block_size_bytes <= 0");
        }
        size_t chunk_size =
            std::ceil(((float)(block_size_bytes)) /
                      ((float)get_row_bytes(
                          this->build_table_schema)));  // each chunk has
                                                        // a single block
        this->build_table_buffer = std::make_unique<ChunkedTableBuilder>(
            this->build_table_schema, this->build_table_dict_builders,
            chunk_size, DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);

        // Update metrics
        this->metrics.block_size_bytes = block_size_bytes;
        this->metrics.chunk_size_nrows = chunk_size;

        // Set the min/max candidate columns.
        for (size_t i = 0; i < this->build_table_schema->column_types.size();
             i++) {
            this->min_max_values.emplace_back(std::nullopt);
        }
        for (int64_t build_col_idx : interval_build_col_idxs) {
            std::unique_ptr<bodo::DataType>& dtype =
                this->build_table_schema->column_types[build_col_idx];
            if (this->IsValidRuntimeJoinFilterMinMaxColumn(dtype)) {
                if (dtype->array_type == bodo_array_type::NUMPY ||
                    dtype->array_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    this->min_max_values[build_col_idx] =
                        alloc_nullable_array_all_nulls(
                            2, this->build_table_schema
                                   ->column_types[build_col_idx]
                                   ->c_type);
                } else {
                    assert(dtype->array_type == bodo_array_type::STRING ||
                           dtype->array_type == bodo_array_type::DICT);
                    std::shared_ptr<array_info> str_arr =
                        alloc_string_array(Bodo_CTypes::STRING, 2, 0);
                    str_arr->set_null_bit<bodo_array_type::STRING>(0, 0);
                    str_arr->set_null_bit<bodo_array_type::STRING>(1, 0);
                    this->min_max_values[build_col_idx] = str_arr;
                }
            }
        }
        this->build_dict_hit_bitmap.resize(
            this->build_table_schema->column_types.size());
    }

    // Metrics for the query profile.
    NestedLoopJoinMetrics metrics;

    /**
     * @brief Finalize build step for nested loop join.
     * This may lead to a broadcast join if the build table is small
     * enough.
     *
     */
    void FinalizeBuild() override;

    /**
     * @brief Process a probe chunk and add the output
     * to the output buffer.
     *
     * NOTE: Must be called on all ranks when doing a outer
     * join on the probe side (i.e. either full-outer or probe-outer)
     * and the build side is distributed.
     *
     * @param probe_table The probe table chunk to process.
     * @param build_kept_cols The build table columns to keep in the output.
     * @param probe_kept_cols The probe table columns to keep in the output.
     */
    void ProcessProbeChunk(std::shared_ptr<table_info> probe_table,
                           const std::vector<uint64_t>& build_kept_cols,
                           const std::vector<uint64_t>& probe_kept_cols);

   protected:
    /// Override the metric reporting helpers to report the detailed metrics
    /// collected during the build and probe stages.
    void ReportBuildStageMetrics(std::vector<MetricBase>& metrics_out) override;
    void ReportProbeStageMetrics(std::vector<MetricBase>& metrics_out) override;
};

/**
 * @brief Python wrapper to consume build table batch in nested loop join
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch locally
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last);

/**
 * @brief consume probe table batch in streaming nested loop join
 * Design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1373896721/Vectorized+Nested+Loop+Join+Design
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param is_last is last batch locally
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool nested_loop_join_probe_consume_batch(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool is_last);

bool stream_join_sync_is_last(bool local_is_last, JoinState* join_state);

/**
 * @brief consume build table batch in streaming join (insert into hash
 * table)
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 * @return updated is_last
 */
bool join_build_consume_batch(HashJoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              bool use_bloom_filter, bool local_is_last);

/**
 * @brief consume probe table batch in streaming join.
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the
 * build side.
 * @param probe_kept_cols Which columns to generate in the output on the
 * probe side.
 * @param is_last is last batch
 * @param parallel parallel flag
 * @return updated global is_last with the possibility of false positives
 * due to iterations between syncs
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool use_bloom_filter>
bool join_probe_consume_batch(HashJoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              const std::vector<uint64_t> build_kept_cols,
                              const std::vector<uint64_t> probe_kept_cols,
                              bool local_is_last);
