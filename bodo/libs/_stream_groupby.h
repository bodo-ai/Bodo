#pragma once

#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_chunked_table_builder.h"
#include "_operator_pool.h"
#include "_stream_shuffle.h"
#include "_table_builder.h"

#include "_groupby.h"
#include "_groupby_col_set.h"
#include "_groupby_ftypes.h"
#include "_groupby_groups.h"

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
        bodo::OperatorBufferPool* op_pool_,
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

    /// @brief Get number of bits in the 'top_bitmask'.
    size_t get_num_top_bits() const { return this->num_top_bits; }

    /// @brief Get the 'top_bitmask'.
    uint32_t get_top_bitmask() const { return this->top_bitmask; }

    /// @brief Check if a row is part of this partition based on its
    /// partition hash.
    /// @param hash Partition hash for the row
    /// @return True if row is part of partition, False otherwise.
    inline bool is_in_partition(const uint32_t& hash) const;

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
     * NOTE: Only active partitions are supported at this point.
     * Support for inactive partitions will be added in the future.
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
     * NOTE: Only active partitions are supported at this point.
     * Support for inactive partitions will be added in the future.
     *
     * @param mrnf_part_cols_to_keep Bitmask specifying the partition columns to
     * retain in the output.
     * @param mrnf_sort_cols_to_keep Bitmask specifying the order-by columns to
     * retain in the output.
     * @param output_buffer The output buffer to append the generated output to.
     */
    void FinalizeMrnf(
        const std::vector<bool>& mrnf_part_cols_to_keep,
        const std::vector<bool>& mrnf_sort_cols_to_keep,
        const std::shared_ptr<ChunkedTableBuilder>& output_buffer);

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
     * @param mrnf_only_ Whether this is the MRNF-only case.
     */
    GroupbyIncrementalShuffleState(
        const std::shared_ptr<bodo::Schema> shuffle_table_schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
        const uint64_t mrnf_n_sort_cols_, const uint64_t n_keys_,
        const uint64_t& curr_iter_, int64_t& sync_freq_, int64_t op_id_,
        const bool nunique_only_, const bool mrnf_only_);

    virtual ~GroupbyIncrementalShuffleState() = default;

    /**
     * @brief Clears the "shuffle" state. This releases
     * the memory of the logical shuffle hash table, i.e.
     * the shuffle buffer, shuffle hash-table and shuffle
     * hashes.
     * This is meant to be called in GroupbyState::FinalizeBuild.
     */
    void Finalize() override;

    friend class GroupbyState;

   protected:
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
    const bool mrnf_only = false;
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

    std::vector<std::shared_ptr<BasicColSet>> col_sets;

    // Shuffle state
    std::unique_ptr<GroupbyIncrementalShuffleState> shuffle_state;

    // indices of input columns for each function
    // f_in_offsets contains the offsets into f_in_cols.
    // f_in_cols is a list of physical column indices.
    // For example:
    //
    // f_in_offsets = (0, 1, 5)
    // f_in_cols = (0, 7, 1, 3, 4, 0)
    // The first function uses the columns in f_in_cols[0:1]. IE physical index
    // 0 in the input table. The second function uses the column f_in_cols[1:5].
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
    bool mrnf_only = false;
    const std::vector<bool> mrnf_sort_asc;
    const std::vector<bool> mrnf_sort_na;
    const std::vector<bool> mrnf_part_cols_to_keep;
    const std::vector<bool> mrnf_sort_cols_to_keep;

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

    // Output buffer
    // This will be lazily initialized during the end of
    // the build step to simplify specifying the output column types.
    // TODO(njriasan): Move to initialization information.
    std::shared_ptr<ChunkedTableBuilder> output_buffer = nullptr;

    // Dictionary builders for the key columns. This is
    // always of length n_keys and is nullptr for non DICT keys.
    // These will be shared between the build_shuffle_buffer,
    // build_table_buffers of all partitions and the output buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;

    // Simple concatenation of key_dict_builders and
    // non key dict builders.
    // Key dict builders are always at the beginning of the vector, and non-key
    // dict builders follow. For all columns, if the array type is not dict
    // encoded, the value is nullptr
    // These will be shared between build_table_buffers of all partitions.
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Dictionary builders for output columns
    std::vector<std::shared_ptr<DictionaryBuilder>> out_dict_builders;

    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the final steps.
    bool build_input_finalized = false;

    /// @brief Whether we should print debug information
    /// about partitioning such as when a partition is split.
    bool debug_partitioning = false;

    /// @brief Whether partitioning is currently enabled.
    bool partitioning_enabled = true;

    tracing::ResumableEvent groupby_event;

    GroupbyState(const std::unique_ptr<bodo::Schema>& in_schema_,
                 std::vector<int32_t> ftypes_,
                 std::vector<int32_t> f_in_offsets_,
                 std::vector<int32_t> f_in_cols_, uint64_t n_keys_,
                 std::vector<bool> mrnf_sort_asc_vec_,
                 std::vector<bool> mrnf_sort_na_pos_,
                 std::vector<bool> mrnf_part_cols_to_keep_,
                 std::vector<bool> mrnf_sort_cols_to_keep_,
                 int64_t output_batch_size_, bool parallel_, int64_t sync_iter_,
                 int64_t op_id_, int64_t op_pool_size_bytes_);

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
     * from the input batch.
     *
     * NOTE: Only used in the aggregating code path (i.e.
     * accumulate_before_update = false)
     *
     * @param in_table Input batch to update and combine using.
     * @param partitioning_hashes Partitioning hashes for the records.
     * @param batch_hashes_groupby Groupby hashes for the input batch records.
     * @param not_append_rows Flipped bitmask specifying the rows to use. We use
     * a flipped bitmask for practical reasons based on how this function is
     * used. This allows us to avoid doing a .flip() on an existing bitmask.
     */
    void UpdateShuffleGroupsAndCombine(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
        const std::vector<bool>& not_append_rows);

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
     * We will use 'mrnf_part_cols_to_keep' and 'mrnf_sort_cols_to_keep' to
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

   private:
    /**
     * Helper function that gets the running column types for a given function.
     * This is used to initialize the build state. Currently, this creates a
     * dummy colset, and calls getRunningValueColumnTypes on it. This is pretty
     * ugly, but it works for now.
     */
    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        std::vector<std::shared_ptr<array_info>> local_input_cols,
        std::vector<std::unique_ptr<bodo::DataType>>&& in_dtypes, int ftype);

    /**
     * Helper function that gets the output column types for a given function.
     * This is used to initialize the build state. Implemented in a similar
     * fashion to getRunningValueColumnTypes.
     */
    std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumns(
        std::vector<std::shared_ptr<array_info>> local_input_cols, int ftype);

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
};
