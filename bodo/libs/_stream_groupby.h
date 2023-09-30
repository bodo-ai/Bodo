#pragma once
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_chunked_table_builder.h"
#include "_operator_pool.h"
#include "_table_builder.h"

#include "_groupby.h"
#include "_groupby_col_set.h"
#include "_groupby_ftypes.h"
#include "_groupby_groups.h"

// Default threshold for Groupby operator's OperatorBufferPool
#define GROUPBY_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD 0.5

// Use all available memory by default
#define GROUPBY_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

class GroupbyPartition;
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
    GroupbyState* groupby_state;
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
    GroupbyState* groupby_state;
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
        const std::vector<int8_t>& build_arr_c_types_,
        const std::vector<int8_t>& build_arr_array_types_,
        const uint64_t n_keys_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            build_table_dict_builders_,
        const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
        const std::vector<int32_t>& f_in_offsets_,
        const std::vector<int32_t>& f_in_cols_,
        const std::vector<int32_t>& f_running_value_offsets_,
        const uint64_t batch_size_, bool is_active_,
        bool accumulate_before_update_, bool req_extended_group_info_,
        bool parallel_, bodo::OperatorBufferPool* op_pool_,
        const std::shared_ptr<::arrow::MemoryManager> op_mm_);

    // The types of the columns in the build table.
    const std::vector<int8_t> build_arr_c_types;
    const std::vector<int8_t> build_arr_array_types;
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

    // Chunked append-only buffer, only used if a partition is inactive and
    // not yet finalized
    ChunkedTableBuilder build_table_buffer_chunked;

    // XXX TODO Make both build_table_buffer and build_table_buffer_chunked
    // unique_ptrs. In ctor, only initialize one of them based on is_active.
    // If is_active=false, initialize build_table_buffer in "Activate".

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
    const bool parallel;

    /// @brief OperatorBufferPool of the Groupby operator that this is a
    /// partition in. The pool is owned by the parent GroupbyState, so
    /// we keep a raw pointer for simplicity.
    bodo::OperatorBufferPool* const op_pool;
    /// @brief Memory manager instance for op_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;

    /**
     * @brief Clear the "build" state. This releases
     * all memory associated with the logical hash table.
     * This is meant to be called in FinalizeBuild.
     *
     */
    void ClearBuildState();
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

   public:
    using shuffle_hash_table_t = grpby_hash_table_t</*is_local*/ false>;

    // Partitioning information.
    std::vector<std::shared_ptr<GroupbyPartition>> partitions;

    // TODO Decide this dynamically using a heuristic based
    // on total available memory, total disk space, etc.
    const size_t max_partition_depth = 6;

    const uint64_t n_keys;
    bool parallel;
    const int64_t output_batch_size;

    std::vector<std::shared_ptr<BasicColSet>> col_sets;

    // temporary batch data (for the shuffle case)
    std::shared_ptr<table_info> in_table = nullptr;
    std::shared_ptr<uint32_t[]> in_table_hashes = nullptr;

    // Shuffle build state
    int64_t shuffle_next_group = 0;
    std::unique_ptr<shuffle_hash_table_t> shuffle_hash_table;
    bodo::vector<uint32_t> shuffle_table_groupby_hashes;
    std::unique_ptr<TableBuildBuffer> shuffle_table_buffer;

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

    // indices of update and combine columns for each function
    std::vector<int32_t> f_running_value_offsets;

    // The number of iterations between syncs
    int64_t sync_iter;
    // Counter of number of syncs since previous sync freq update.
    // Set to -1 if not updating adaptively (user has specified sync freq).
    int64_t adaptive_sync_counter = -1;
    // The iteration number of last shuffle (used for adaptive sync estimation)
    uint64_t prev_shuffle_iter = 0;

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

    tracing::ResumableEvent groupby_event;

    GroupbyState(std::vector<int8_t> in_arr_c_types,
                 std::vector<int8_t> in_arr_array_types,
                 std::vector<int32_t> ftypes,
                 std::vector<int32_t> f_in_offsets_,
                 std::vector<int32_t> f_in_cols_, uint64_t n_keys_,
                 int64_t output_batch_size_, bool parallel_, int64_t sync_iter_,
                 // If -1, we'll use 100% of the total buffer
                 // pool size. Else we'll use the provided size.
                 int64_t op_pool_size_bytes = -1,
                 size_t max_partition_depth_ = 5);

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
     * @brief Reset the shuffle state. This is meant to be
     * called after a shuffle operation.
     * This clears the shuffle hash table, resets the shuffle buffer,
     * clears the shuffle hashes vector and resets shuffle_next_group
     * back to 0.
     * Note that this doesn't release memory.
     *
     */
    void ResetShuffleState();

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

   private:
    /**
     * Helper function that gets the running column types for a given function.
     * This is used to initialize the build state. Currently, this creates a
     * dummy colset, and calls getRunningValueColumnTypes on it. This is pretty
     * ugly, but it works for now.
     */
    std::tuple<std::vector<bodo_array_type::arr_type_enum>,
               std::vector<Bodo_CTypes::CTypeEnum>>
    getRunningValueColumnTypes(
        std::vector<std::shared_ptr<array_info>> local_input_cols,
        std::vector<bodo_array_type::arr_type_enum>& in_arr_types,
        std::vector<Bodo_CTypes::CTypeEnum>& in_dtypes, int ftype);

    /**
     * @brief Clear the "shuffle" state. This releases
     * the memory of the logical shuffle hash table, i.e.
     * the shuffle buffer, shuffle hash-table and shuffle
     * hashes.
     *
     * This is meant to be called in FinalizeBuild.
     *
     */
    void ClearShuffleState();

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
};
