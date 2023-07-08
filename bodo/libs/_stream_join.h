#pragma once
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_chunked_table_builder.h"
#include "_join.h"
#include "simd-block-fixed-fpp.h"

using BloomFilter = SimdBlockFilterFixed<::hashing::SimpleMixSplit>;

class JoinPartition;
struct HashHashJoinTable {
    /**
     * provides row hashes for join hash table (bodo::unordered_multimap)
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
     * provides row comparison for join hash table (bodo::unordered_multimap)
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
 * @brief Holds the state of a single partition during
 * a join execution. This includes the build table buffer,
 * the hashtable (unordered_multimap), bitmap of the matches
 * in build records, etc.
 * 'top_bitmask' and 'num_top_bits' define the partition
 * itself, i.e. a record is in this partition if the top
 * 'num_top_bits' bits of its hash are 'top_bitmask'.
 *
 */
class JoinPartition {
   public:
    explicit JoinPartition(
        size_t num_top_bits_, uint32_t top_bitmask_,
        const std::vector<int8_t>& build_arr_c_types,
        const std::vector<int8_t>& build_arr_array_types,
        const std::vector<int8_t>& probe_arr_c_types,
        const std::vector<int8_t>& probe_arr_array_types,
        const uint64_t n_keys_, bool build_table_outer_,
        bool probe_table_outer_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            build_table_dict_builders,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            probe_table_dict_builders)
        : build_table_buffer(build_arr_c_types, build_arr_array_types,
                             build_table_dict_builders),
          build_table({}, HashHashJoinTable(this),
                      KeyEqualHashJoinTable(this, n_keys_)),
          probe_table_buffer(probe_arr_c_types, probe_arr_array_types,
                             probe_table_dict_builders),
          num_top_bits(num_top_bits_),
          top_bitmask(top_bitmask_),
          build_table_outer(build_table_outer_),
          probe_table_outer(probe_table_outer_),
          n_keys(n_keys_) {}

    // Build state
    TableBuildBuffer build_table_buffer;  // Append only buffer.
    bodo::vector<uint32_t> build_table_join_hashes;

    bodo::unord_map_container<int64_t, size_t, HashHashJoinTable,
                              KeyEqualHashJoinTable>
        build_table;  // join hash table (key row number -> matching row
                      // numbers)
    // Use std::vector to avoid allocation overhead of bodo::vector.
    // TODO: use bodo::vector when we support spilling
    std::vector<std::vector<size_t>> groups;

    // Probe state (for outer joins). Note we don't use
    // vector<bool> because we may need to do an allreduce
    // on the data directly and that can't be accessed for bool.
    bodo::vector<uint8_t>
        build_table_matched;  // state for building output table

    // Probe state (only used when this partition is inactive).
    // We don't need partitioning hashes since we should never
    // need to repartition.
    // XXX These will be converted to use chunked arrays.
    TableBuildBuffer probe_table_buffer;
    bodo::vector<uint32_t> probe_table_buffer_join_hashes;

    // Temporary state during probe step. These will be
    // reset between iterations.
    std::shared_ptr<table_info> probe_table;
    // Join hashes corresponding to data in probe_table.
    // We're using a raw pointer here so we can populate this field using the
    // vector (using .data()) or the uint32_t[] shared_ptr directly (using
    // .get())
    uint32_t* probe_table_hashes;

    /// @brief Get number of bits in the 'top_bitmask'.
    size_t get_num_top_bits() const { return this->num_top_bits; }

    /// @brief Get the 'top_bitmask'.
    uint32_t get_top_bitmask() const { return this->top_bitmask; }

    /// @brief Check if a row is part of this partition based on its
    /// partition hash.
    /// @param hash Partition hash for the row
    /// @return True if row is part of partition, False otherwise.
    inline bool is_in_partition(const uint32_t& hash);

    /// @brief Is the partition near full? This is used
    /// to determine whether this partition should be
    /// split into multiple partitions.
    inline bool is_near_full() const {
        // TODO Replace with proper implementation based
        // on buffer sizes, memory budget and Allocator statistics.
        return false;
    }

    /**
     * @brief Split the partition into 2^num_levels partitions.
     * This will produce a new set of partitions, each with their
     * new build_table_buffer and build_table_join_hashes.
     * The caller must explicitly rebuild the build_table on
     * the partition.
     *
     * @param num_levels Number of levels to split the partition. Only '1' is
     * supported at this point.
     * @return std::vector<std::shared_ptr<JoinPartition>>
     */
    std::vector<std::shared_ptr<JoinPartition>> SplitPartition(
        size_t num_levels = 1);

    /**
     * @brief Reserve space in build_table_buffer and build_table_join_hashes to
     * add all rows from in_table.
     *
     * @param in_table Table to reserve based on.
     */
    void ReserveBuildTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Reserve space in probe_table_buffer and
     * probe_table_join_hashes to add all rows from in_table.
     *
     * @param in_table Table to reserve based on.
     */
    void ReserveProbeTable(const std::shared_ptr<table_info>& in_table);

    /// @brief Add rows from build_table_buffer into the
    /// hash table. This adds rows starting from curr_build_size.
    /// This is useful for rebuilding the hash table after
    /// repartitioning and for building hash tables of inactive
    /// partitions at the end of the build step (after we've
    /// seen all the data).
    void BuildHashTable();

    /**
     * @brief Add all rows from in_table to this partition.
     * This includes populating the hash table.
     *
     * @tparam is_active Is this the active partition.
     * @param in_table Table to insert.
     * @param join_hashes Join hashes for the table records.
     * @param partitioning_hashes Partitioning hashes for the table records.
     */
    template <bool is_active = false>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table,
                          const std::shared_ptr<uint32_t[]>& join_hashes);

    /**
     * @brief Inserts the last row of build buffer
     * (build_table_buffer[curr_build_size]) into build hash map
     *
     */
    inline void InsertLastRowIntoMap();

    /**
     * @brief Add all rows from in_table to this partition.
     * This includes populating the hash table.
     *
     * @tparam is_active Is this the active partition.
     * @param in_table Table to insert.
     * @param join_hashes Join hashes for the table records.
     * @param partitioning_hashes Partitioning hashes for the table records.
     * @param append_rows Vector of booleans indicating whether to append the
     * row
     */
    template <bool is_active = false>
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table,
                          const std::shared_ptr<uint32_t[]>& join_hashes,
                          const std::vector<bool>& append_rows);

    /**
     * @brief Finalize the build step for this partition.
     * At this time, this just initializes the build_table_matched
     * bitmap in the build_table_outer case.
     *
     */
    void FinalizeBuild();

    /**
     * @brief Append a batch of data into the probe table buffer.
     * Note that this is only used for inactive partitions
     * to buffer the inputs before we start processing them.
     *
     * @param in_table Table from which we're adding the row.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash for the record.
     * @param append_row Whether to append the row.
     */
    void AppendInactiveProbeBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::vector<bool>& append_rows);

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
     * @param build_needs_reduction Do the build misses need a reduction?
     * @param[in, out] output_buffer The output buffer of the join state that
     * we should append the output to.
     *
     */
    template <bool build_table_outer, bool probe_table_outer,
              bool non_equi_condition>
    void FinalizeProbeForInactivePartition(
        cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols,
        const bool build_needs_reduction,
        const std::shared_ptr<ChunkedTableBuilder>& output_buffer);

   private:
    const size_t num_top_bits = 0;
    const uint32_t top_bitmask = 0ULL;
    const bool build_table_outer = false;
    const bool probe_table_outer = false;
    const uint64_t n_keys;
    // Tracks the current size of the build table, i.e.
    // the number of rows from the build_table_buffer
    // that have been added to the hash table.
    int64_t curr_build_size = 0;
};

class JoinState {
   public:
    // The types of the columns in the build table and probe tables.
    const std::vector<int8_t> build_arr_c_types;
    const std::vector<int8_t> build_arr_array_types;
    const std::vector<int8_t> probe_arr_c_types;
    const std::vector<int8_t> probe_arr_array_types;
    // Join properties
    const uint64_t n_keys;
    cond_expr_fn_t cond_func;
    const bool build_table_outer;
    const bool probe_table_outer;
    // Note: This isn't constant because we may change it
    // via broadcast decisions.
    bool build_parallel;
    const bool probe_parallel;
    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the outer
    // join output.
    bool build_input_finalized = false;
    bool probe_input_finalized = false;
    const int64_t output_batch_size;

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

    JoinState(const std::vector<int8_t>& build_arr_c_types_,
              const std::vector<int8_t>& build_arr_array_types_,
              const std::vector<int8_t>& probe_arr_c_types_,
              const std::vector<int8_t>& probe_arr_array_types_,
              uint64_t n_keys_, bool build_table_outer_,
              bool probe_table_outer_, cond_expr_fn_t cond_func_,
              bool build_parallel_, bool probe_parallel_,
              int64_t output_batch_size_);

    virtual ~JoinState() {}

    virtual void FinalizeBuild() { this->build_input_finalized = true; }

    void FinalizeProbe() {
        this->output_buffer->Finalize(/*shrink_to_fit*/ true);
        this->probe_input_finalized = true;
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
     * @param only_keys only unify key columns
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with build table dictionaries.
     */
    std::shared_ptr<table_info> UnifyBuildTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table, bool only_keys = false);

    /**
     * @brief Unify dictionaries of input table with probe table
     * (probe_table_buffer of all partitions and probe_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @param only_keys only unify key columns
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with probe table dictionaries.
     */
    std::shared_ptr<table_info> UnifyProbeTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table, bool only_keys = false);
};

class HashJoinState : public JoinState {
   public:
    // Partitioning information.
    std::vector<std::shared_ptr<JoinPartition>> partitions;

    const size_t max_partition_depth = 5;

    // Shuffle state
    TableBuildBuffer build_shuffle_buffer;
    TableBuildBuffer probe_shuffle_buffer;

    // Global bloom-filter. This is built during the build step
    // and used during the probe step.
    std::unique_ptr<BloomFilter> global_bloom_filter;

    // Keep a table of NA keys for bypassing the hash table
    // if we have an outer join and any keys can contain NAs.
    TableBuildBuffer build_na_key_buffer;
    // How many NA values have we seen. This is used for consistent
    // partitioning if the build table is replicated and the probe table
    // distributed. This is unused if the build table is distributed or
    // the final output is replicated.
    size_t build_na_counter = 0;

    // Current iteration of the build and probe steps
    uint64_t build_iter;
    uint64_t probe_iter;

    HashJoinState(const std::vector<int8_t>& build_arr_c_types,
                  const std::vector<int8_t>& build_arr_array_types,
                  const std::vector<int8_t>& probe_arr_c_types,
                  const std::vector<int8_t>& probe_arr_array_types,
                  uint64_t n_keys_, bool build_table_outer_,
                  bool probe_table_outer_, cond_expr_fn_t cond_func_,
                  bool build_parallel_, bool probe_parallel_,
                  int64_t output_batch_size_, size_t max_partition_depth_ = 5);

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
            // the target size in bytes in the env or 1MB
            // if not provided.
            int64_t target_bytes = 1000000;
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
     * @brief Split the partition at index 'idx' into two partitions.
     *
     * @param idx Index of the partition (in this->partitions) to split.
     */
    void SplitPartition(size_t idx);

    /**
     * @brief Clear the existing partition(s) and replace with a single
     * partition with the correct type information. This creates equivalent
     * partition state as when HashJoinState is initialized except there
     * may be some additional dictionary builder information.
     *
     */
    void ResetPartitions();

    /**
     * @brief Reserve space in build_table_buffer, build_table_join_hashes, etc.
     * of all partitions to add all rows from 'in_table'.
     * XXX This will likely change to only allocate required memory (or do away
     * with upfront reserve altogether -- at least for inactive partitions)
     *
     * @param in_table Reference table to reserve memory based on.
     */
    void ReserveBuildTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append a build row. It will figure out the correct
     * partition based on the partitioning hash. If the record
     * is in the "active" (i.e. index 0) partition, it will be
     * added to the hash table of that active partition
     * as well. If record belongs to an inactive partition, it
     * will be simply added to the build buffer of the partition.
     * It is slightly optimized for the single partition case.
     *
     * @param in_table Table to add the rows from.
     * @param join_hashes Join hashes for the records.
     * @param partitioning_hashes Partitioning hashes for the records.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /**
     * @brief Append a build row. It will figure out the correct
     * partition based on the partitioning hash. If the record
     * is in the "active" (i.e. index 0) partition, it will be
     * added to the hash table of that active partition
     * as well. If record belongs to an inactive partition, it
     * will be simply added to the build buffer of the partition.
     * It is slightly optimized for the single partition case.
     *
     * @param in_table Table to add the rows from.
     * @param join_hashes Join hashes for the records.
     * @param partitioning_hashes Partitioning hashes for the records.
     * @param append_rows Vector of booleans indicating whether to append the
     * row
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

    void InitOutputBuffer(
        const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols) override;

    /**
     * @brief Finalize build step for all partitions.
     * This will process the partitions one by one (only one is pinned in memory
     * at one time), build hash tables, split partitions as necessary, etc.
     *
     */
    void FinalizeBuild() override;

    /**
     * @brief Reserve enough space to accommodate in_table
     * in probe buffers of each of the inactive partitions.
     *
     * @param in_table Reference table to reserve space based on.
     */
    void ReserveProbeTableForInactivePartitions(
        const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append probe batch to the probe table buffer of the
     * appropriate inactive partition. This assumes that the row
     * is _not_ in the active (index 0) partition.
     *
     * @param in_table Table to add the record from.
     * @param row_ind Index of the row to append.
     * @param join_hash Join hash for the record.
     * @param partitioning_hash Partitioning hash for the record.
     * @param append_row Whether to append the row
     */
    void AppendProbeBatchToInactivePartition(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes,
        const std::vector<bool>& append_rows);

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

    tracing::ResumableEvent join_event;
};

class NestedLoopJoinState : public JoinState {
   public:
    // Build state
    TableBuildBuffer build_table_buffer;        // Append only buffer.
    bodo::vector<uint8_t> build_table_matched;  // state for building output
                                                // table (for outer joins)

    NestedLoopJoinState(const std::vector<int8_t>& build_arr_c_types,
                        const std::vector<int8_t>& build_arr_array_types,
                        const std::vector<int8_t>& probe_arr_c_types,
                        const std::vector<int8_t>& probe_arr_array_types,
                        bool build_table_outer_, bool probe_table_outer_,
                        cond_expr_fn_t cond_func_, bool build_parallel_,
                        bool probe_parallel_, int64_t output_batch_size_)
        : JoinState(build_arr_c_types, build_arr_array_types, probe_arr_c_types,
                    probe_arr_array_types, 0, build_table_outer_,
                    probe_table_outer_, cond_func_, build_parallel_,
                    probe_parallel_,
                    output_batch_size_),  // NestedLoopJoin is only used when
                                          // n_keys is 0
          build_table_buffer(build_arr_c_types, build_arr_array_types,
                             build_table_dict_builders),
          join_event("NestedLoopJoin") {
        // TODO: Integrate dict_builders for nested loop join.
    }

    tracing::ResumableEvent join_event;

    /**
     * @brief Finalize build step for nested loop join.
     * This may lead to a broadcast join if the build table is small
     * enough.
     *
     */
    void FinalizeBuild() override;
};

/**
 * @brief Python wrapper to consume build table batch in nested loop join
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last);

/**
 * @brief consume probe table batch in streaming nested loop join
 * Design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1373896721/Vectorized+Nested+Loop+Join+Design
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param is_last is last batch
 */
void nested_loop_join_probe_consume_batch(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool is_last);

/**
 * @brief Get the dtypes and arr types from an existing table
 *
 * @param table Reference table
 * @return std::tuple<std::vector<int8_t>, std::vector<int8_t>>
 */
static std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_table(const std::shared_ptr<table_info>& table) {
    size_t n_cols = table->columns.size();
    std::vector<int8_t> arr_c_types(n_cols);
    std::vector<int8_t> arr_array_types(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        arr_c_types[i] = table->columns[i]->dtype;
        arr_array_types[i] = table->columns[i]->arr_type;
    }
    return std::make_tuple(arr_c_types, arr_array_types);
}
