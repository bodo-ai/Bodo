#include "_stream_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"

/* --------------------------- Helper Functions --------------------------- */

std::tuple<std::vector<int8_t>, std::vector<int8_t>>
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

/* ------------------------------------------------------------------------ */

/* --------------------------- HashHashJoinTable -------------------------- */

uint32_t HashHashJoinTable::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        return (*(
            this->join_partition->build_table_join_hashes_guard.value()))[iRow];
    } else {
        return this->join_partition->probe_table_hashes[-iRow - 1];
    }
}

/* ------------------------------------------------------------------------ */

/* ------------------------ KeyEqualHashJoinTable ------------------------- */

bool KeyEqualHashJoinTable::operator()(const int64_t iRowA,
                                       const int64_t iRowB) const {
    const std::shared_ptr<table_info>& build_table =
        this->join_partition->build_table_buffer.data_table;
    const std::shared_ptr<table_info>& probe_table =
        this->join_partition->probe_table;

    bool is_build_A = iRowA >= 0;
    bool is_build_B = iRowB >= 0;

    size_t jRowA = is_build_A ? iRowA : -iRowA - 1;
    size_t jRowB = is_build_B ? iRowB : -iRowB - 1;

    const std::shared_ptr<table_info>& table_A =
        is_build_A ? build_table : probe_table;
    const std::shared_ptr<table_info>& table_B =
        is_build_B ? build_table : probe_table;

    // All NA keys have already been pruned.
    bool test =
        TestEqualJoin(table_A, table_B, jRowA, jRowB, this->n_keys, false);
    return test;
}

/* ------------------------------------------------------------------------ */

/* ---------------------------- JoinPartition ----------------------------- */

JoinPartition::JoinPartition(
    size_t num_top_bits_, uint32_t top_bitmask_,
    const std::vector<int8_t>& build_arr_c_types_,
    const std::vector<int8_t>& build_arr_array_types_,
    const std::vector<int8_t>& probe_arr_c_types_,
    const std::vector<int8_t>& probe_arr_array_types_, const uint64_t n_keys_,
    bool build_table_outer_, bool probe_table_outer_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>&
        build_table_dict_builders_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>&
        probe_table_dict_builders_,
    const uint64_t batch_size_, bool is_active_,
    bodo::OperatorBufferPool* op_pool_,
    const std::shared_ptr<::arrow::MemoryManager> op_mm_)
    : build_arr_c_types(build_arr_c_types_),
      build_arr_array_types(build_arr_array_types_),
      probe_arr_c_types(probe_arr_c_types_),
      probe_arr_array_types(probe_arr_array_types_),
      build_table_dict_builders(build_table_dict_builders_),
      probe_table_dict_builders(probe_table_dict_builders_),
      build_table_buffer(build_arr_c_types, build_arr_array_types,
                         build_table_dict_builders, op_pool_, op_mm_),
      build_table_join_hashes(op_pool_),
      build_table_buffer_chunked(
          build_arr_c_types, build_arr_array_types, build_table_dict_builders,
          batch_size_, DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES),
      build_hash_table(std::make_unique<bodo::pinnable<hash_table_t>>(
          0, HashHashJoinTable(this), KeyEqualHashJoinTable(this, n_keys_),
          op_pool_)),
      num_rows_in_group(
          std::make_unique<bodo::pinnable<bodo::vector<size_t>>>(op_pool_)),
      build_row_to_group_map(
          std::make_unique<bodo::pinnable<bodo::vector<size_t>>>(op_pool_)),
      groups(op_pool_),
      groups_offsets(op_pool_),
      build_table_matched(op_pool_),
      probe_table_buffer_chunked(
          probe_arr_c_types, probe_arr_array_types, probe_table_dict_builders,
          batch_size_, DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES),
      batch_size(batch_size_),
      dummy_probe_table(alloc_table(probe_arr_c_types, probe_arr_array_types)),
      num_top_bits(num_top_bits_),
      top_bitmask(top_bitmask_),
      build_table_outer(build_table_outer_),
      probe_table_outer(probe_table_outer_),
      n_keys(n_keys_),
      op_pool(op_pool_),
      op_mm(op_mm_),
      is_active(is_active_) {
    // Pin everything by default during initialization.
    // From here on out, the HashJoinState will manage pinning
    // and unpinning of the partitions.
    this->pin();

    // Reserve some space in num_rows_in_group and build_row_to_group_map
    // for active partitions.
    // For partitions that will be activated later, we can reserve much more
    // accurately at that time.
    if (is_active_) {
        // Allocate the smallest size-class to start off.
        // TODO Tune this and/or use hints from optimizer/compiler about
        // expected build table size and number of groups.
        const size_t init_reserve_size =
            bodo::BufferPool::Default()->GetSmallestSizeClassSize() /
            sizeof(size_t);
        this->num_rows_in_group_guard.value()->reserve(init_reserve_size);
        this->build_row_to_group_map_guard.value()->reserve(init_reserve_size);
    }
}

inline bool JoinPartition::is_in_partition(const uint32_t& hash) const {
    if (this->num_top_bits == 0) {
        // Shifting uint32_t by 32 bits is undefined behavior.
        // Ref:
        // https://stackoverflow.com/questions/18799344/shifting-a-32-bit-integer-by-32-bits
        return true;
    } else {
        constexpr size_t uint32_bits = sizeof(uint32_t) * CHAR_BIT;
        return (hash >> (uint32_bits - this->num_top_bits)) ==
               this->top_bitmask;
    }
}

template <bool is_active>
std::vector<std::shared_ptr<JoinPartition>> JoinPartition::SplitPartition(
    size_t num_levels) {
    assert(this->pinned_);
    if (num_levels != 1) {
        throw std::runtime_error(
            "We currently only support splitting a partition into 2 at a "
            "time.");
    }
    constexpr size_t uint32_bits = sizeof(uint32_t) * CHAR_BIT;
    if (this->num_top_bits >= (uint32_bits - 1)) {
        throw std::runtime_error(
            "Cannot split the partition further. Out of hash bits.");
    }

    // Release the hash-table memory:
    this->build_hash_table_guard.reset();
    this->build_hash_table.reset();

    // Get dictionary hashes from the dict-builders of build table.
    // Dictionaries of key columns are shared between build and probe tables,
    // so using either is fine.
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = std::make_shared<
            bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>();
    dict_hashes->reserve(this->n_keys);
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (this->build_table_dict_builders[i] == nullptr) {
            dict_hashes->push_back(nullptr);
        } else {
            dict_hashes->emplace_back(
                this->build_table_dict_builders[i]->GetDictionaryHashes());
        }
    }

    // Create the two new partitions. These will differ on the next bit.
    // Partitions are pinned when created, so we can use the guards
    // on bodo::pinnable attributes safely for the rest of the function.
    std::shared_ptr<JoinPartition> new_part1 = std::make_shared<JoinPartition>(
        this->num_top_bits + 1, (this->top_bitmask << 1),
        this->build_arr_c_types, this->build_arr_array_types,
        this->probe_arr_c_types, this->probe_arr_array_types, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        this->batch_size, is_active, this->op_pool, this->op_mm);
    std::shared_ptr<JoinPartition> new_part2 = std::make_shared<JoinPartition>(
        this->num_top_bits + 1, (this->top_bitmask << 1) + 1,
        this->build_arr_c_types, this->build_arr_array_types,
        this->probe_arr_c_types, this->probe_arr_array_types, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        this->batch_size, false, this->op_pool, this->op_mm);

    std::vector<bool> append_partition1;
    if (is_active) {
        // In the active case, partition this->build_table_buffer directly

        // Compute partitioning hashes
        std::shared_ptr<uint32_t[]> build_table_partitioning_hashes =
            hash_keys_table(this->build_table_buffer.data_table, this->n_keys,
                            SEED_HASH_PARTITION, false, false, dict_hashes);

        // Put the build data in the new partitions.
        append_partition1.resize(this->build_table_buffer.data_table->nrows(),
                                 false);

        // We will only append the entries until
        // build_safely_appended_nrows. If there are more entries
        // in the build_table_buffer, this means we triggered this partition
        // split in the middle of an append. That means that those entries
        // weren't added safely and will be retried. Therefore, we will skip
        // those entries here (and leave their default to false).
        for (size_t i_row = 0; i_row < this->build_safely_appended_nrows;
             i_row++) {
            append_partition1[i_row] = new_part1->is_in_partition(
                build_table_partitioning_hashes[i_row]);
        }

        // Calculate number of rows going to 1st new partition
        uint64_t append_partition1_sum = std::accumulate(
            append_partition1.begin(), append_partition1.end(), (uint64_t)0);

        // Reserve space in hashes vector. This doesn't inhibit
        // exponential growth since we're only doing it at the start.
        // Future appends will still allow for regular exponential growth.
        new_part1->build_table_join_hashes_guard.value()->reserve(
            append_partition1_sum);

        // Copy the hash values to the new active partition. We might
        // not have hashes for every row, so copy over whatever we can.
        // Subsequent BuildHashTable steps will compute the rest.
        // We drop the hashes that would go to the new inactive partition
        // for now and will re-compute them later when needed.
        size_t n_hashes_to_copy_over =
            std::min(this->build_safely_appended_nrows,
                     this->build_table_join_hashes_guard.value()->size());
        for (size_t i_row = 0; i_row < n_hashes_to_copy_over; i_row++) {
            if (append_partition1[i_row]) {
                new_part1->build_table_join_hashes_guard.value()->push_back(
                    (*this->build_table_join_hashes_guard.value())[i_row]);
            }
        }

        // Reserve space for append (append_partition1 already accounts for
        // build_safely_appended_nrows)
        new_part1->build_table_buffer.ReserveTable(
            this->build_table_buffer.data_table, append_partition1,
            append_partition1_sum);
        new_part1->build_table_buffer.UnsafeAppendBatch(
            this->build_table_buffer.data_table, append_partition1,
            append_partition1_sum);

        // Update safely appended row count for the new active partition:
        new_part1->build_safely_appended_nrows = append_partition1_sum;

        append_partition1.flip();
        std::vector<bool>& append_partition2 = append_partition1;
        // The rows between this->build_safely_appended_nrows
        // and this->build_table_buffer.data_table->nrows() shouldn't
        // be copied over to either partition:
        for (size_t i = this->build_safely_appended_nrows;
             i < append_partition2.size(); i++) {
            append_partition2[i] = false;
        }

        new_part2->build_table_buffer_chunked.AppendBatch(
            this->build_table_buffer.data_table, append_partition2);

        // We do not rebuild the hash table here (for new_part1 which is the new
        // active partition). That needs to be handled by the caller.

    } else {
        // In the inactive case, partition build_table_buffer chunk by chunk
        this->build_table_buffer_chunked.Finalize();

        while (!this->build_table_buffer_chunked.chunks.empty()) {
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked.PopChunk();

            // Compute partitioning hashes
            std::shared_ptr<uint32_t[]> build_table_partitioning_hashes_chunk =
                hash_keys_table(build_table_chunk, this->n_keys,
                                SEED_HASH_PARTITION, false, false, dict_hashes);

            // Put the build data in the sub partitions.
            // XXX Might be faster to pre-calculate the required
            // sizes, build a bitmap, pre-allocated space and then append
            // into the new partitions.
            append_partition1.resize(build_table_nrows_chunk, false);
            for (int64_t i_row = 0; i_row < build_table_nrows_chunk; i_row++) {
                append_partition1[i_row] = new_part1->is_in_partition(
                    build_table_partitioning_hashes_chunk[i_row]);
            }

            new_part1->build_table_buffer_chunked.AppendBatch(
                build_table_chunk, append_partition1);

            append_partition1.flip();
            std::vector<bool>& append_partition2 = append_partition1;

            new_part2->build_table_buffer_chunked.AppendBatch(
                build_table_chunk, append_partition2);
        }
    }

    // Splitting happens at build time, so the probe buffers should
    // be empty.

    return {new_part1, new_part2};
}

inline void JoinPartition::BuildHashTable() {
    assert(this->pinned_);
    // First compute the join hashes:
    auto& build_table_join_hashes_ =
        this->build_table_join_hashes_guard.value();
    size_t build_table_nrows = this->build_table_buffer.data_table->nrows();
    size_t join_hashes_cur_len = build_table_join_hashes_->size();

    // TODO: Do this processing in batches of 4K rows (for handling inactive
    // partition case where we will do this for the entire table)!!

    size_t n_unhashed_rows = build_table_nrows - join_hashes_cur_len;
    if (n_unhashed_rows > 0) {
        // Compute hashes for the batch:
        std::unique_ptr<uint32_t[]> join_hashes = hash_keys_table(
            this->build_table_buffer.data_table, this->n_keys, SEED_HASH_JOIN,
            /*is_parallel*/ false,
            /*global_dict_needed*/ false, /*dict_hashes*/ nullptr,
            /*start_row_offset*/ join_hashes_cur_len);
        // Append the hashes:
        build_table_join_hashes_->insert(build_table_join_hashes_->end(),
                                         join_hashes.get(),
                                         join_hashes.get() + n_unhashed_rows);
    }

    // Create reference variables for easier usage.
    auto& num_rows_in_group_ = this->num_rows_in_group_guard.value();
    auto& build_row_to_group_map_ = this->build_row_to_group_map_guard.value();
    auto& build_hash_table_ = this->build_hash_table_guard.value();

    // Add all the rows in the build_table_buffer that haven't
    // already been added to the hash table.
    while (this->curr_build_size <
           static_cast<int64_t>(this->build_table_buffer.data_table->nrows())) {
        size_t& group_id = (*build_hash_table_)[this->curr_build_size];
        // group_id==0 means key doesn't exist in map
        if (group_id == 0) {
            // Update the value of group_id stored in the hash map
            // as well since its passed by reference.
            group_id = num_rows_in_group_->size() + 1;
            // Initialize group size to 0.
            num_rows_in_group_->emplace_back(0);
        }
        // Increment count for the group
        (*num_rows_in_group_)[group_id - 1]++;
        // Store the group id for this row. This will allow us to avoid an
        // expensive hashmap lookup during 'FinalizeGroups'.
        build_row_to_group_map_->emplace_back(group_id);
        this->curr_build_size++;
    }
}

void JoinPartition::FinalizeGroups() {
    assert(this->pinned_);
    // Check if groups are already finalized. If so,
    // we can return immediately.
    if (this->finalized_groups) {
        return;
    }

    auto& num_rows_in_group_ = this->num_rows_in_group_guard.value();
    auto& groups_offsets_ = this->groups_offsets_guard.value();
    // Number of groups is the same as the size of num_rows_in_group.
    size_t num_groups = num_rows_in_group_->size();
    // Resize offsets vector based on the number of groups
    groups_offsets_->resize(num_groups + 1);
    // First offset should always be 0
    (*groups_offsets_)[0] = 0;
    // Do a cumulative sum and fill the rest of groups_offsets:
    if (num_groups > 0) {
        std::partial_sum(num_rows_in_group_->cbegin(),
                         num_rows_in_group_->cend(),
                         groups_offsets_->begin() + 1);
    }

    // Release the pin guard
    this->num_rows_in_group_guard.reset();
    // Free num_rows_in_group memory
    this->num_rows_in_group.reset();

    // Resize groups based on how many total elements are in all groups
    // (same as build table size essentially):
    auto& groups_ = this->groups_guard.value();
    groups_->resize((*groups_offsets_)[groups_offsets_->size() - 1]);

    /// Fill the groups vector. The build_row_to_group_map vector is already
    /// populated from the first iteration.

    // Store counters for each group, so we can put the row-id in the correct
    // location.
    bodo::vector<size_t> group_fill_counter(num_groups, 0);
    auto& build_row_to_group_map_ = this->build_row_to_group_map_guard.value();
    size_t build_table_rows = this->build_table_buffer.data_table->nrows();
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        // Get this row's group_id from build_row_to_group_map
        const size_t group_id = (*build_row_to_group_map_)[i_build];
        // Find next available index for this group:
        size_t groups_idx =
            (*groups_offsets_)[group_id - 1] + group_fill_counter[group_id - 1];
        // Insert and update group_fill_counter:
        (*groups_)[groups_idx] = i_build;
        group_fill_counter[group_id - 1]++;
    }
    // Reset the pin guard
    this->build_row_to_group_map_guard.reset();
    // Free build_row_to_group_map memory
    this->build_row_to_group_map.reset();

    // Mark the groups as finalized.
    this->finalized_groups = true;

    // group_fill_counter will go out of scope and its memory will be freed
    // automatically.
}

template <bool is_active>
void JoinPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table) {
    assert(this->pinned_);
    if (is_active) {
        /// Start "transaction":

        // Idempotent. This will be a NOP unless we're re-trying this step
        // after a partition split.
        this->BuildHashTable();
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer.ReserveTable(in_table);
        // Now append the rows. This will always succeed since we've
        // reserved space upfront.
        this->build_table_buffer.UnsafeAppendBatch(in_table);
        // Compute the hashes and add rows to the hash table now.
        this->BuildHashTable();

        /// Commit "transaction". Only update this after all the rows have
        /// been appended to build_table_buffer, the hash table _and_
        /// build_table_join_hashes.
        this->build_safely_appended_nrows = this->curr_build_size;
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked.AppendBatch(in_table);
    }
}

template <bool is_active>
void JoinPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    assert(this->pinned_);
    if (is_active) {
        /// Start "transaction":

        // Idempotent. This will be a NOP unless we're re-trying this step
        // after a partition split.
        this->BuildHashTable();
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer.ReserveTable(in_table);
        // Now append the rows. This will always succeed since we've
        // reserved space upfront.
        this->build_table_buffer.UnsafeAppendBatch(in_table, append_rows);
        // Compute the hashes and add rows to the hash table now.
        this->BuildHashTable();

        /// Commit "transaction". Only update this after all the rows have
        /// been appended to build_table_buffer, the hash table _and_
        /// build_table_join_hashes.
        this->build_safely_appended_nrows = this->curr_build_size;
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked.AppendBatch(in_table, append_rows);
    }
}

void JoinPartition::FinalizeBuild() {
    assert(this->pinned_);
    // TODO For inactive partitions, we know the required size
    // of build_row_to_group_map (number of rows), so maybe we can reserve that
    // altogether during either ActivatePartition or BuildHashTable.
    // Size of num_rows_in_group is not known, but we could reserve
    // build table size in that case as well as an upper bound.

    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    this->ActivatePartition();
    // Make sure all rows from build_table_buffer have been inserted
    // into the hash table. This is idempotent.
    this->BuildHashTable();
    // Finalize the groups. This step is idempotent.
    this->FinalizeGroups();
    if (this->build_table_outer) {
        // This step is idempotent by definition.
        this->build_table_matched_guard.value()->resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer.data_table->nrows()),
            0);
    }
}

void JoinPartition::AppendInactiveProbeBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::vector<bool>& append_rows) {
    auto probe_table_buffer_join_hashes_(
        bodo::pin(this->probe_table_buffer_join_hashes));
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            probe_table_buffer_join_hashes_->push_back(join_hashes[i_row]);
        }
    }
    this->probe_table_buffer_chunked.AppendBatch(in_table, append_rows);
}

/**
 * @brief Helper function for join_probe_consume_batch and
 * FinalizeProbeForInactivePartition to update 'build_idxs'
 * and 'probe_idxs'. It also updates the 'build_table_matched'
 * bitmap of the partition in the `build_table_outer` case.
 *
 * NOTE: Assumes that the row is in the partition.
 * NOTE: Inlined since it's called inside loops.
 *
 * @tparam build_table_outer
 * @tparam probe_table_outer
 * @tparam non_equi_condition
 * @param cond_func Condition function to use. `nullptr` for the
 * all-equality conditions case.
 * @param[in, out] partition Partition that this row belongs to.
 *  NOTE: This function assumes that the partition is already pinned.
 * @param i_row Row index in partition->probe_table to produce the output
 * for,
 * @param[in, out] build_idxs Build table indices for the output. This will
 * be updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will
 * be updated in place.
 *
 * The rest of the parameters are output of get_gen_cond_data_ptrs on the
 * build and probe table and are only relevant for the condition function
 * case:
 * @param build_table_info_ptrs
 * @param probe_table_info_ptrs
 * @param build_col_ptrs
 * @param probe_col_ptrs
 * @param build_null_bitmaps
 * @param probe_null_bitmaps
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
inline void handle_probe_input_for_partition(
    cond_expr_fn_t cond_func, JoinPartition* partition, size_t i_row,
    bodo::vector<int64_t>& build_idxs, bodo::vector<int64_t>& probe_idxs,
    std::vector<array_info*>& build_table_info_ptrs,
    std::vector<array_info*>& probe_table_info_ptrs,
    std::vector<void*>& build_col_ptrs, std::vector<void*>& probe_col_ptrs,
    std::vector<void*>& build_null_bitmaps,
    std::vector<void*>& probe_null_bitmaps) {
    auto iter = partition->build_hash_table_guard.value()->find(-i_row - 1);
    if (iter == partition->build_hash_table_guard.value()->end()) {
        if (probe_table_outer) {
            // Add unmatched rows from probe table to output table
            build_idxs.push_back(-1);
            probe_idxs.push_back(i_row);
        }
        return;
    }
    // TODO Pass pinned groups_offsets vector instead of pinning for each
    // row.
    auto& partition_groups_offsets_ = partition->groups_offsets_guard.value();
    const size_t group_start_idx =
        (*partition_groups_offsets_)[iter->second - 1];
    const size_t group_end_idx =
        (*partition_groups_offsets_)[iter->second - 1 + 1];
    // Initialize to true for pure hash join so the final branch
    // is non-equality condition only.
    bool has_match = !non_equi_condition;
    // TODO Pass pinned groups vector instead of pinning every time.
    auto& partition_groups_ = partition->groups_guard.value();
    // TODO Pass pinned build_table_matched instead of pinning every time.
    auto& partition_build_table_matched_ =
        partition->build_table_matched_guard.value();
    for (size_t idx = group_start_idx; idx < group_end_idx; idx++) {
        const size_t j_build = (*partition_groups_)[idx];
        if (non_equi_condition) {
            // Check for matches with the non-equality portion.
            bool match =
                cond_func(build_table_info_ptrs.data(),
                          probe_table_info_ptrs.data(), build_col_ptrs.data(),
                          probe_col_ptrs.data(), build_null_bitmaps.data(),
                          probe_null_bitmaps.data(), j_build, i_row);
            if (!match) {
                continue;
            }
            has_match = true;
        }
        if (build_table_outer) {
            SetBitTo(partition_build_table_matched_->data(), j_build, 1);
        }
        build_idxs.push_back(j_build);
        probe_idxs.push_back(i_row);
    }
    // non-equality condition only branch
    if (!has_match && probe_table_outer) {
        // Add unmatched rows from probe table to output table
        build_idxs.push_back(-1);
        probe_idxs.push_back(i_row);
    }
}

/**
 * @brief Generate output build and probe table indices
 * for the `build_table_outer` case. Essentially finds
 * all the build records that didn't match, and adds them to the output
 * (with NULL on the probe side).
 * @tparam requires_reduction Whether the build matches require a reduction
 * because the probe table is distributed but the build table is replicated.
 *
 * @param partition Join partition to produce the output for.
 *  NOTE: This function assumes that the partition is pinned.
 * @param[in, out] build_idxs Build table indices for the output. This will
 * be updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will
 * be updated in place.
 */
template <bool requires_reduction>
void generate_build_table_outer_rows_for_partition(
    JoinPartition* partition, bodo::vector<int64_t>& build_idxs,
    bodo::vector<int64_t>& probe_idxs) {
    auto& build_table_matched_ = partition->build_table_matched_guard.value();
    if (requires_reduction) {
        MPI_Allreduce_bool_or(*build_table_matched_);
    }
    int n_pes, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Add unmatched rows from build table to output table
    for (size_t i_row = 0;
         i_row < partition->build_table_buffer.data_table->nrows(); i_row++) {
        if ((!requires_reduction || ((i_row % n_pes) == my_rank))) {
            bool has_match = GetBit(build_table_matched_->data(), i_row);
            if (!has_match) {
                build_idxs.push_back(i_row);
                probe_idxs.push_back(-1);
            }
        }
    }
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
void JoinPartition::FinalizeProbeForInactivePartition(
    cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols,
    const bool build_needs_reduction,
    const std::shared_ptr<ChunkedTableBuilder>& output_buffer) {
    assert(this->pinned_);
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;
    // Raw array pointers from arrays for passing to non-equijoin condition
    std::vector<array_info*> build_table_info_ptrs;
    std::vector<array_info*> probe_table_info_ptrs;
    // Vectors for data and null bitmaps for fast null checking from the
    // cfunc
    std::vector<void*> build_col_ptrs;
    std::vector<void*> probe_col_ptrs;
    std::vector<void*> build_null_bitmaps;
    std::vector<void*> probe_null_bitmaps;

    auto probe_table_buffer_join_hashes_(
        bodo::pin(this->probe_table_buffer_join_hashes));

    // Loop over chunked probe table's buffers/hashes and process one at a
    // time At each loop iteration we mutate `this->probe_table`, so all
    // `i_row` and `probe_idxs` indices refer only to the local chunk.
    // Therefore, adding unmatched rows from the probe table to the output
    // must be done at each iteration, while that probe table is in memory.
    // Adding unmatched rows from the build table should be done at the end.
    this->probe_table_hashes = probe_table_buffer_join_hashes_->data();
    this->probe_table_buffer_chunked.Finalize();
    while (!this->probe_table_buffer_chunked.chunks.empty()) {
        auto [probe_table_chunk, probe_table_nrows] =
            this->probe_table_buffer_chunked.PopChunk();
        this->probe_table = probe_table_chunk;

        if (non_equi_condition) {
            get_gen_cond_data_ptrs(this->build_table_buffer.data_table,
                                   &build_table_info_ptrs, &build_col_ptrs,
                                   &build_null_bitmaps);
            get_gen_cond_data_ptrs(this->probe_table, &probe_table_info_ptrs,
                                   &probe_col_ptrs, &probe_null_bitmaps);
        }

        for (size_t i_row = 0; i_row < this->probe_table->nrows(); i_row++) {
            handle_probe_input_for_partition<
                build_table_outer, probe_table_outer, non_equi_condition>(
                cond_func, this, i_row, build_idxs, probe_idxs,
                build_table_info_ptrs, probe_table_info_ptrs, build_col_ptrs,
                probe_col_ptrs, build_null_bitmaps, probe_null_bitmaps);
        }

        output_buffer->AppendJoinOutput(
            this->build_table_buffer.data_table, this->probe_table, build_idxs,
            probe_idxs, build_kept_cols, probe_kept_cols);
        build_idxs.clear();
        probe_idxs.clear();

        this->probe_table_hashes += probe_table_nrows;
    }

    // Add unmatched rows from build table to output table
    if (build_table_outer) {
        if (build_needs_reduction) {
            generate_build_table_outer_rows_for_partition<true>(
                this, build_idxs, probe_idxs);
        } else {
            generate_build_table_outer_rows_for_partition<false>(
                this, build_idxs, probe_idxs);
        }

        output_buffer->AppendJoinOutput(
            this->build_table_buffer.data_table, this->dummy_probe_table,
            build_idxs, probe_idxs, build_kept_cols, probe_kept_cols);
        build_idxs.clear();
        probe_idxs.clear();
    }

    this->probe_table.reset();
    this->probe_table_hashes = nullptr;
}

void JoinPartition::ActivatePartition() {
    assert(this->pinned_);
    if (!this->is_active) {
        /// Concatenate all build chunks into contiguous build buffer

        // Finalize the chunked table builder:
        this->build_table_buffer_chunked.Finalize();

        // Do a single ReserveTable call to allocate all required space in a
        // single call:
        this->build_table_buffer.ReserveTable(this->build_table_buffer_chunked);

        // This will work without error because we've already allocated
        // all the required space:
        while (!this->build_table_buffer_chunked.chunks.empty()) {
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked.PopChunk();
            this->build_table_buffer.UnsafeAppendBatch(build_table_chunk);
        }

        // Mark this partition as activated once we've moved the data
        // from the chunked buffer to a contiguous buffer:
        this->is_active = true;
        this->build_safely_appended_nrows =
            this->build_table_buffer.data_table->nrows();
    }
}

void JoinPartition::pin() {
    if (!this->pinned_) {
        this->build_table_buffer.pin();
        this->build_table_join_hashes_guard.emplace(
            this->build_table_join_hashes);
        // 'build_hash_table', 'num_rows_in_group' and
        // 'build_row_to_group_map' are unique_ptrs and may be reset during
        // execution, so we shouldn't try to pin them unless they're
        // pointing to something.
        if (this->build_hash_table.get() != nullptr) {
            this->build_hash_table_guard.emplace(*this->build_hash_table);
        }
        if (this->num_rows_in_group.get() != nullptr) {
            this->num_rows_in_group_guard.emplace(*this->num_rows_in_group);
        }
        if (this->build_row_to_group_map.get() != nullptr) {
            this->build_row_to_group_map_guard.emplace(
                *this->build_row_to_group_map);
        }
        this->groups_guard.emplace(this->groups);
        this->groups_offsets_guard.emplace(this->groups_offsets);
        this->build_table_matched_guard.emplace(this->build_table_matched);
    }
    this->pinned_ = true;
}

void JoinPartition::unpin() {
    if (this->pinned_) {
        this->build_table_buffer.unpin();
        this->build_table_join_hashes_guard.reset();
        this->build_hash_table_guard.reset();
        this->num_rows_in_group_guard.reset();
        this->build_row_to_group_map_guard.reset();
        this->groups_guard.reset();
        this->groups_offsets_guard.reset();
        this->build_table_matched_guard.reset();
    }
    this->pinned_ = false;
}

/* ------------------------------------------------------------------------ */

/* --------------------------- JoinState ---------------------------------- */

JoinState::JoinState(const std::vector<int8_t>& build_arr_c_types_,
                     const std::vector<int8_t>& build_arr_array_types_,
                     const std::vector<int8_t>& probe_arr_c_types_,
                     const std::vector<int8_t>& probe_arr_array_types_,
                     uint64_t n_keys_, bool build_table_outer_,
                     bool probe_table_outer_, cond_expr_fn_t cond_func_,
                     bool build_parallel_, bool probe_parallel_,
                     int64_t output_batch_size_, uint64_t sync_iter)
    : build_arr_c_types(build_arr_c_types_),
      build_arr_array_types(build_arr_array_types_),
      probe_arr_c_types(probe_arr_c_types_),
      probe_arr_array_types(probe_arr_array_types_),
      n_keys(n_keys_),
      cond_func(cond_func_),
      build_table_outer(build_table_outer_),
      probe_table_outer(probe_table_outer_),
      build_parallel(build_parallel_),
      probe_parallel(probe_parallel_),
      output_batch_size(output_batch_size_),
      sync_iter(sync_iter),
      dummy_probe_table(alloc_table(probe_arr_c_types, probe_arr_array_types)) {
    this->key_dict_builders.resize(this->n_keys);

    // Create dictionary builders for key columns:
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (build_arr_array_types[i] == bodo_array_type::DICT) {
            if (probe_arr_array_types[i] != bodo_array_type::DICT) {
                throw std::runtime_error(
                    "HashJoinState: Key column array types don't match "
                    "between build and probe tables!");
            }
            std::shared_ptr<array_info> dict = alloc_array(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            this->key_dict_builders[i] =
                std::make_shared<DictionaryBuilder>(dict, true);
            // Also set this as the dictionary of the dummy probe table
            // for consistency, else there will be issues during output
            // generation.
            this->dummy_probe_table->columns[i]->child_arrays[0] = dict;
        } else {
            this->key_dict_builders[i] = nullptr;
        }
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        build_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in build table:
    for (size_t i = this->n_keys; i < this->build_arr_array_types.size(); i++) {
        if (this->build_arr_array_types[i] == bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            build_table_non_key_dict_builders.emplace_back(
                std::make_shared<DictionaryBuilder>(dict, false));
        } else {
            build_table_non_key_dict_builders.emplace_back(nullptr);
        }
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        probe_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in probe table:
    for (size_t i = this->n_keys; i < this->probe_arr_array_types.size(); i++) {
        if (this->probe_arr_array_types[i] == bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            probe_table_non_key_dict_builders.emplace_back(
                std::make_shared<DictionaryBuilder>(dict, false));
            // Also set this as the dictionary of the dummy probe table
            // for consistency, else there will be issues during output
            // generation.
            this->dummy_probe_table->columns[i]->child_arrays[0] = dict;
        } else {
            probe_table_non_key_dict_builders.emplace_back(nullptr);
        }
    }

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(), this->key_dict_builders.begin(),
        this->key_dict_builders.end());
    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(),
        build_table_non_key_dict_builders.begin(),
        build_table_non_key_dict_builders.end());

    this->probe_table_dict_builders.insert(
        this->probe_table_dict_builders.end(), this->key_dict_builders.begin(),
        this->key_dict_builders.end());
    this->probe_table_dict_builders.insert(
        this->probe_table_dict_builders.end(),
        probe_table_non_key_dict_builders.begin(),
        probe_table_non_key_dict_builders.end());
}

void JoinState::InitOutputBuffer(const std::vector<uint64_t>& build_kept_cols,
                                 const std::vector<uint64_t>& probe_kept_cols) {
    if (this->output_buffer != nullptr) {
        // Already initialized. We only initialize on the first
        // iteration.
        return;
    }
    std::vector<int8_t> arr_c_types, arr_array_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    arr_c_types.reserve(build_kept_cols.size() + probe_kept_cols.size());
    arr_array_types.reserve(build_kept_cols.size() + probe_kept_cols.size());
    dict_builders.reserve(build_kept_cols.size() + probe_kept_cols.size());
    for (uint64_t i_col : build_kept_cols) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)this->build_arr_array_types[i_col];
        Bodo_CTypes::CTypeEnum dtype =
            (Bodo_CTypes::CTypeEnum)this->build_arr_c_types[i_col];
        // In the probe outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        // TODO: Move to a helper function.
        if (this->probe_table_outer && ((arr_type == bodo_array_type::NUMPY) &&
                                        (is_integer(dtype) || is_float(dtype) ||
                                         dtype == Bodo_CTypes::_BOOL))) {
            arr_type = bodo_array_type::NULLABLE_INT_BOOL;
        }
        arr_c_types.push_back(dtype);
        arr_array_types.push_back(arr_type);
        dict_builders.push_back(this->build_table_dict_builders[i_col]);
    }
    for (uint64_t i_col : probe_kept_cols) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)this->probe_arr_array_types[i_col];
        Bodo_CTypes::CTypeEnum dtype =
            (Bodo_CTypes::CTypeEnum)this->probe_arr_c_types[i_col];
        // In the build outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        if (this->build_table_outer && ((arr_type == bodo_array_type::NUMPY) &&
                                        (is_integer(dtype) || is_float(dtype) ||
                                         dtype == Bodo_CTypes::_BOOL))) {
            arr_type = bodo_array_type::NULLABLE_INT_BOOL;
        }
        arr_c_types.push_back(dtype);
        arr_array_types.push_back(arr_type);
        dict_builders.push_back(this->probe_table_dict_builders[i_col]);
    }
    this->output_buffer = std::make_shared<ChunkedTableBuilder>(
        arr_c_types, arr_array_types, dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}

std::shared_ptr<table_info> unify_dictionary_arrays_helper(
    const std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    uint64_t n_keys, bool only_keys) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type != bodo_array_type::DICT ||
            (only_keys && (i >= n_keys))) {
            out_arr = in_arr;
        } else {
            out_arr = dict_builders[i]->UnifyDictionaryArray(in_arr);
        }
        out_arrs.emplace_back(out_arr);
    }
    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> JoinState::UnifyBuildTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    return unify_dictionary_arrays_helper(
        in_table, this->build_table_dict_builders, this->n_keys, only_keys);
}

std::shared_ptr<table_info> JoinState::UnifyProbeTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    return unify_dictionary_arrays_helper(
        in_table, this->probe_table_dict_builders, this->n_keys, only_keys);
}

/* ------------------------------------------------------------------------ */

/* ---------------------------- HashJoinState ----------------------------- */

HashJoinState::HashJoinState(const std::vector<int8_t>& build_arr_c_types,
                             const std::vector<int8_t>& build_arr_array_types,
                             const std::vector<int8_t>& probe_arr_c_types,
                             const std::vector<int8_t>& probe_arr_array_types,
                             uint64_t n_keys_, bool build_table_outer_,
                             bool probe_table_outer_, cond_expr_fn_t cond_func_,
                             bool build_parallel_, bool probe_parallel_,
                             int64_t output_batch_size_, uint64_t sync_iter,
                             int64_t op_pool_size_bytes,
                             size_t max_partition_depth_)
    : JoinState(build_arr_c_types, build_arr_array_types, probe_arr_c_types,
                probe_arr_array_types, n_keys_, build_table_outer_,
                probe_table_outer_, cond_func_, build_parallel_,
                probe_parallel_, output_batch_size_, sync_iter),
      // Create the operator buffer pool
      op_pool(std::make_unique<bodo::OperatorBufferPool>(
          ((op_pool_size_bytes == -1)
               ? static_cast<uint64_t>(
                     bodo::BufferPool::Default()->get_memory_size_bytes() *
                     JOIN_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL)
               : op_pool_size_bytes),
          bodo::BufferPool::Default(),
          JOIN_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD)),
      op_mm(bodo::buffer_memory_manager(op_pool.get())),
      max_partition_depth(max_partition_depth_),
      build_shuffle_buffer(std::make_unique<TableBuildBuffer>(
          build_arr_c_types, build_arr_array_types,
          this->build_table_dict_builders)),
      probe_shuffle_buffer(std::make_unique<TableBuildBuffer>(
          probe_arr_c_types, probe_arr_array_types,
          this->probe_table_dict_builders)),
      // Create a build buffer for NA values to skip the hash table.
      build_na_key_buffer(build_arr_c_types, build_arr_array_types,
                          this->build_table_dict_builders, output_batch_size_,
                          DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES),
      build_event("HashJoinBuild"),
      probe_event("HashJoinProbe") {
    // For now, we will allow re-partitioning only in the case where the
    // build side is distributed. If we allow re-partitioning in the
    // replicated build side case, we must assume that the partitioning
    // state is identical on all ranks. This might not always be true.
    // Therefore, we will turn of threshold enforcement in the replicated
    // build case altogether.
    // XXX Revisit this in the future if needed.
    if (!this->build_parallel) {
        this->DisablePartitioning();
    }

    // Disable partitioning if
    char* disable_partitioning_env_ =
        std::getenv("BODO_STREAM_HASH_JOIN_DISABLE_PARTITIONING");
    if (disable_partitioning_env_ &&
        (std::strcmp(disable_partitioning_env_, "1") == 0)) {
        this->DisablePartitioning();
    }

    if (char* debug_partitioning_env_ =
            std::getenv("BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING")) {
        this->debug_partitioning = !std::strcmp(debug_partitioning_env_, "1");
    }

    // Create the initial partition
    this->partitions.emplace_back(std::make_shared<JoinPartition>(
        0, 0, build_arr_c_types, build_arr_array_types, probe_arr_c_types,
        probe_arr_array_types, n_keys_, build_table_outer_, probe_table_outer_,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        /*batch_size*/ this->output_batch_size, /*is_active*/ true,
        this->op_pool.get(), this->op_mm));

    this->global_bloom_filter = create_bloom_filter();
}

void HashJoinState::SplitPartition(size_t idx) {
    if (this->partitions[idx]->get_num_top_bits() >=
        this->max_partition_depth) {
        // TODO Eventually, this should lead to falling back
        // to nested loop join for this partition.
        // (https://bodo.atlassian.net/browse/BSE-535).
        throw std::runtime_error(
            "Cannot split partition beyond max partition depth of: " +
            std::to_string(max_partition_depth));
    }

    // Threshold enforcement should already be disabled in the
    // replicated build case, so SplitPartition should never get called.
    // We're adding this check to protect against edge cases.
    if (!this->build_parallel) {
        throw std::runtime_error(
            "HashJoinState::SplitPartition: We cannot split a partition "
            "when the build table is replicated!");
    }

    if (this->debug_partitioning) {
        std::cerr << "[DEBUG] Splitting partition " << idx << "." << std::endl;
    }

    // Temporarily disable threshold enforcement during partition
    // split.
    this->op_pool->DisableThresholdEnforcement();

    // Call SplitPartition on the idx'th partition:
    std::vector<std::shared_ptr<JoinPartition>> new_partitions;
    if (this->partitions[idx]->is_active_partition()) {
        new_partitions = this->partitions[idx]->SplitPartition<true>();
    } else {
        new_partitions = this->partitions[idx]->SplitPartition<false>();
    }
    // Remove the current partition (this should release its memory)
    this->partitions.erase(this->partitions.begin() + idx);
    // Insert the new partitions in its place
    this->partitions.insert(this->partitions.begin() + idx,
                            new_partitions.begin(), new_partitions.end());

    // Re-enable threshold enforcement now that we have split the
    // partition successfully.
    this->op_pool->EnableThresholdEnforcement();

    // TODO Check if the new active partition needs to be split up further.
    // XXX Might not be required if we split proactively and there isn't
    // a single hot key (in which case we need to fall back to nested loop
    // join for this partition).
}

void HashJoinState::ResetPartitions() {
    this->partitions.clear();
    this->partitions.emplace_back(std::make_shared<JoinPartition>(
        0, 0, this->build_arr_c_types, this->build_arr_array_types,
        this->probe_arr_c_types, this->probe_arr_array_types, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        /*batch_size*/ this->output_batch_size, /*is_active*/ true,
        this->op_pool.get(), this->op_mm));
}

void HashJoinState::AppendBuildBatchHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch<true>(in_table);
        return;
    }
    std::vector<std::vector<bool>> append_rows_by_partition(
        this->partitions.size());
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] = std::vector<bool>(in_table->nrows());
    }

    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        bool found_partition = false;

        // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize
        // partition search by storing a tree representation of the
        // partition space.
        for (size_t i_part = 0;
             (i_part < this->partitions.size() && !found_partition); i_part++) {
            if (this->partitions[i_part]->is_in_partition(
                    partitioning_hashes[i_row])) {
                found_partition = true;
                append_rows_by_partition[i_part][i_row] = true;
            }
        }
        if (!found_partition) {
            throw std::runtime_error(
                "HashJoinState::AppendBuildBatch: Couldn't find "
                "any matching partition for row!");
        }
    }
    this->partitions[0]->AppendBuildBatch<true>(in_table,
                                                append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendBuildBatch<false>(
            in_table, append_rows_by_partition[i_part]);
    }
}

void HashJoinState::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    while (true) {
        try {
            this->AppendBuildBatchHelper(in_table, partitioning_hashes);
            break;
        } catch (
            bodo::OperatorBufferPool::OperatorPoolThresholdExceededError&) {
            // Split the 0th partition into 2 in case of an
            // OperatorPoolThresholdExceededError. This will always succeed
            // since it creates one active and another inactive partition.
            // The new active partition can only be as large as the original
            // partition, and since threshold enforcement is disabled, it
            // should fit in memory.
            if (this->debug_partitioning) {
                std::cerr << "[DEBUG] HashJoinState::AppendBuildBatch[2]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }
            this->SplitPartition(0);
        }
    }
}

void HashJoinState::AppendBuildBatchHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch<true>(in_table, append_rows);
        return;
    }

    std::vector<std::vector<bool>> append_rows_by_partition;
    append_rows_by_partition.resize(this->partitions.size());
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] = std::vector<bool>(in_table->nrows());
    }

    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            bool found_partition = false;

            // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize
            // partition search by storing a tree representation of the
            // partition space.
            for (size_t i_part = 0;
                 (i_part < this->partitions.size() && !found_partition);
                 i_part++) {
                if (this->partitions[i_part]->is_in_partition(
                        partitioning_hashes[i_row])) {
                    found_partition = true;
                    append_rows_by_partition[i_part][i_row] = true;
                }
            }
            if (!found_partition) {
                throw std::runtime_error(
                    "HashJoinState::AppendBuildBatch: Couldn't find "
                    "any matching partition for row!");
            }
        }
    }
    this->partitions[0]->AppendBuildBatch<true>(in_table,
                                                append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendBuildBatch<false>(
            in_table, append_rows_by_partition[i_part]);
    }
}

void HashJoinState::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows) {
    while (true) {
        try {
            this->AppendBuildBatchHelper(in_table, partitioning_hashes,
                                         append_rows);
            break;
        } catch (
            bodo::OperatorBufferPool::OperatorPoolThresholdExceededError&) {
            // Split the 0th partition into 2 in case of an
            // OperatorPoolThresholdExceededError. This will always succeed
            // since it creates one active and another inactive partition.
            // The new active partition can only be as large as the original
            // partition, and since threshold enforcement is disabled, it
            // should fit in memory.
            if (this->debug_partitioning) {
                std::cerr << "[DEBUG] HashJoinState::AppendBuildBatch[3]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }
            this->SplitPartition(0);
        }
    }
}

void HashJoinState::InitOutputBuffer(
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    if (this->output_buffer != nullptr) {
        // Already initialized. We only initialize on the first
        // iteration.
        return;
    }
    JoinState::InitOutputBuffer(build_kept_cols, probe_kept_cols);
    // Append any NA values from the NA side.
    if (this->build_table_outer) {
        std::vector<int64_t> build_idxs;
        // build_idxs can only become as large as the max chunk size
        build_idxs.reserve(this->build_na_key_buffer.active_chunk_capacity);
        // Start probe_idxs off with -1s. We will resize it in the loop.
        std::vector<int64_t> probe_idxs(
            this->build_na_key_buffer.active_chunk_capacity, -1);
        while (!this->build_na_key_buffer.chunks.empty()) {
            // The buffer should already be finalized by now, so
            // we can just go over the chunks.
            auto [build_na_table_chunk, n_rows] =
                this->build_na_key_buffer.PopChunk();
            // Create the idxs.
            for (int64_t i = 0; i < n_rows; i++) {
                build_idxs.push_back(i);
            }
            // Just resize to the required size and fill
            // any extra required space with -1s. In most cases,
            // this should be a NOP.
            probe_idxs.resize(n_rows, -1);
            this->output_buffer->AppendJoinOutput(
                build_na_table_chunk, this->dummy_probe_table, build_idxs,
                probe_idxs, build_kept_cols, probe_kept_cols);
            // We don't need to clear probe_idxs since in the next iteration
            // we will just resize it (and contents don't need to be
            // changed).
            build_idxs.clear();
            // 'build_na_table_chunk' will go out of scope and be freed
            // automatically here.
        }
    }
}

void HashJoinState::FinalizeBuild() {
    // Free build shuffle buffer
    this->build_shuffle_buffer.reset();

    // Finalize the NA buffer now that we've seen all the input.
    this->build_na_key_buffer.Finalize();

    // Finalize all the partitions and split them as needed:
    size_t total_rows = 0;
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        // TODO Add logic to check if partition is too big
        // (build_table_buffer size + approximate hash table size) and needs
        // to be repartitioned upfront.

        while (true) {
            try {
                // The partition should already be pinned
                // by default, so this should usually be a NOP,
                // but we're adding it to make this idempotent.
                this->partitions[i_part]->pin();
                // Finalize the partition (transfer data from
                // Chunked buffer to contiguous buffer, build
                // hash table, build groups, create matched_outer bitmap,
                // etc.)
                this->partitions[i_part]->FinalizeBuild();
                total_rows += this->partitions[i_part]
                                  ->build_table_buffer.data_table->nrows();
                // Unpin the partition once we're done.
                this->partitions[i_part]->unpin();
                break;
            } catch (
                bodo::OperatorBufferPool::OperatorPoolThresholdExceededError&) {
                // Split the partition into 2. This will always succeed. If
                // the partition is inactive, we create 2 new inactive
                // partitions that take up no memory from the
                // OperatorBufferPool. If the partition is active, it
                // creates one active and another inactive partition. The
                // new active partition can only be as large as the original
                // partition, and since threshold enforcement is disabled,
                // it should fit in memory just fine.
                if (this->debug_partitioning) {
                    std::cerr
                        << "[DEBUG] HashJoinState::FinalizeBuild: "
                           "Encountered OperatorPoolThresholdExceededError "
                           "while finalizing partition "
                        << i_part << "." << std::endl;
                }
                this->SplitPartition(i_part);
            }
        }
    }

    if (this->debug_partitioning) {
        std::cerr << "[DEBUG] HashJoinState::FinalizeBuild: Total number of "
                     "partitions: "
                  << this->partitions.size() << "." << std::endl;
    }

    // Add the tracing information.
    this->build_event.add_attribute("num_partitions", this->partitions.size());
    this->build_event.add_attribute("g_use_bloom_filters",
                                    this->global_bloom_filter != nullptr);
    this->build_event.add_attribute("build_table_nrows", total_rows);
    // Note: These fields can change because of broadcast decisions.
    this->build_event.add_attribute("g_build_parallel", this->build_parallel);
    this->build_event.add_attribute("g_probe_parallel", this->probe_parallel);
    JoinState::FinalizeBuild();
}

void HashJoinState::FinalizeProbe() {
    // Free the probe shuffle buffer's memory:
    this->probe_shuffle_buffer.reset();

    // Add the tracing information.
    this->probe_event.add_attribute("num_bloom_filter_misses",
                                    this->num_bloom_filter_misses);
    this->probe_event.add_attribute("num_processed_probe_table_rows",
                                    this->num_processed_probe_table_rows);
    this->probe_event.add_attribute("num_input_probe_table_rows",
                                    this->num_input_probe_table_rows);
    this->probe_event.add_attribute("num_output_rows",
                                    this->output_buffer->total_size);
    this->probe_event.add_attribute("max_output_buffer_rows",
                                    this->output_buffer->max_reached_size);
    JoinState::FinalizeProbe();
}

void HashJoinState::AppendProbeBatchToInactivePartition(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows) {
    std::vector<std::vector<bool>> append_rows_by_partition;
    append_rows_by_partition.resize(this->partitions.size());
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] = std::vector<bool>(in_table->nrows());
    }

    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            bool found_partition = false;

            // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize
            // partition search by storing a tree representation of the
            // partition space.
            for (size_t i_part = 1;
                 (i_part < this->partitions.size() && !found_partition);
                 i_part++) {
                if (this->partitions[i_part]->is_in_partition(
                        partitioning_hashes[i_row])) {
                    found_partition = true;
                    append_rows_by_partition[i_part][i_row] = true;
                }
            }
            if (!found_partition) {
                throw std::runtime_error(
                    "HashJoinState::AppendProbeBatchToInactivePartition: "
                    "Couldn't find any matching partition for row!");
            }
        }
    }
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendInactiveProbeBatch(
            in_table, join_hashes, append_rows_by_partition[i_part]);
    }
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
void HashJoinState::FinalizeProbeForInactivePartitions(
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    // We need a reduction of build misses if the probe table is distributed
    // and the build table is not.
    bool build_needs_reduction = this->probe_parallel && !this->build_parallel;
    for (size_t i = 1; i < this->partitions.size(); i++) {
        // Pin the partition
        this->partitions[i]->pin();
        this->partitions[i]
            ->FinalizeProbeForInactivePartition<
                build_table_outer, probe_table_outer, non_equi_condition>(
                this->cond_func, build_kept_cols, probe_kept_cols,
                build_needs_reduction, this->output_buffer);
        // Free the partition
        this->partitions[i].reset();
    }
}

std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
HashJoinState::GetDictionaryHashesForKeys() {
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = std::make_shared<
            bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>(
            this->n_keys);
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (this->key_dict_builders[i] == nullptr) {
            (*dict_hashes)[i] = nullptr;
        } else {
            (*dict_hashes)[i] =
                this->key_dict_builders[i]->GetDictionaryHashes();
        }
    }
    return dict_hashes;
}

uint64_t HashJoinState::op_pool_bytes_pinned() const {
    return this->op_pool->bytes_pinned();
}

uint64_t HashJoinState::op_pool_bytes_allocated() const {
    return this->op_pool->bytes_allocated();
}

template <bool build_table_outer>
std::shared_ptr<table_info> filter_na_values(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    uint64_t n_keys) {
    bodo::vector<bool> not_na(in_table->nrows(), true);
    bool can_have_na = false;
    for (uint64_t i = 0; i < n_keys; i++) {
        // Determine which columns can contain NA/contain NA
        const std::shared_ptr<array_info>& col = in_table->columns[i];
        if (col->can_contain_na()) {
            can_have_na = true;
            bodo::vector<bool> col_not_na = col->get_notna_vector();
            // Do an elementwise logical and to update not_na
            std::transform(not_na.begin(), not_na.end(), col_not_na.begin(),
                           not_na.begin(), std::logical_and<>());
        }
    }
    if (!can_have_na) {
        // No NA values, just return.
        return in_table;
    }
    // Retrieve table takes a list of columns. Convert the boolean array.
    bodo::vector<int64_t> idx_list;
    // For appending NAs in outer join.
    std::vector<bool> append_nas(in_table->nrows(), false);
    // If we have a replicated build table without a replicated probe table.
    // then we only add a fraction of NA rows to the output. Otherwise we
    // add all rows.
    const bool add_all =
        join_state->build_parallel || !join_state->probe_parallel;
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for (size_t i = 0; i < in_table->nrows(); i++) {
        if (not_na[i]) {
            idx_list.emplace_back(i);
        } else if (build_table_outer) {
            // If build table is replicated but output is not
            // replicated evenly divide NA values across all ranks.
            append_nas[i] =
                add_all || ((join_state->build_na_counter % n_pes) == myrank);
            join_state->build_na_counter++;
        }
    }
    if (idx_list.size() == in_table->nrows()) {
        // No NA values, skip the copy.
        return in_table;
    } else {
        if (build_table_outer) {
            // If have an outer join we must push the NA values directly to
            // the output, not just filter them.
            join_state->build_na_key_buffer.AppendBatch(in_table, append_nas);
        }
        return RetrieveTable(std::move(in_table), std::move(idx_list));
    }
}

void HashJoinState::DisablePartitioning() {
    this->op_pool->DisableThresholdEnforcement();
}

/* ------------------------------------------------------------------------ */

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
                              bool use_bloom_filter, bool is_last) {
    if (join_state->build_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "join_build_consume_batch: Received non-empty in_table "
                "after "
                "the build was already finalized!");
        }
        // Nothing left to do for build
        // When build is finalized global is_last has been seen so no need
        // for additional synchronization
        return true;
    }
    auto buildEvent(join_state->build_event.iteration());
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Make is_last global
    is_last =
        join_stream_sync_is_last(is_last, join_state->build_iter, join_state);

    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices NOTE: key columns in build_table_buffer (of
    // all partitions), probe_table_buffers (of all partitions),
    // build_shuffle_buffer and probe_shuffle_buffer use the same dictionary
    // object for consistency. Non-key DICT columns of build_table_buffer
    // and build_shuffle_buffer also share their dictionaries and will also
    // be unified.
    in_table = join_state->UnifyBuildTableDictionaryArrays(in_table);

    // Dictionary hashes for the key columns which will be used for
    // the partitioning hashes:
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = join_state->GetDictionaryHashesForKeys();

    // Prune any rows with NA keys. If this is an build_table_outer = False,
    // then we can prune these rows from the table entirely. If
    // build_table_outer = True then we can skip adding these rows to the
    // hash table (as they can't match), but must write them to the Join
    // output.
    // TODO: Have outer join skip the build table/avoid shuffling.
    if (join_state->build_table_outer) {
        in_table = filter_na_values<true>(join_state, std::move(in_table),
                                          join_state->n_keys);
    } else {
        in_table = filter_na_values<false>(join_state, std::move(in_table),
                                           join_state->n_keys);
    }

    // Get hashes of the new batch (different hashes for partitioning and
    // hash table to reduce conflict)
    // NOTE: Partition hashes need to be consistent across ranks so need to use
    // dictionary hashes. Since we are using dictionary hashes, we don't need
    // dictionaries to be global. In fact, hash_keys_table will ignore the
    // dictionaries entirely when dict_hashes are provided.
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        join_state->build_parallel, false, dict_hashes);

    // Add to the bloom filter.
    if (use_bloom_filter) {
        join_state->global_bloom_filter->AddAll(batch_hashes_partition, 0,
                                                in_table->nrows());
    }

    std::vector<bool> append_row_to_build_table(in_table->nrows(), false);
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        append_row_to_build_table[i_row] =
            (!join_state->build_parallel ||
             hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank);
    }

    // Insert batch into the correct partition.
    // TODO[BSE-441]: tune initial buffer buffer size and expansion strategy
    // using heuristics (e.g. SQL planner statistics)
    join_state->AppendBuildBatch(in_table, batch_hashes_partition,
                                 append_row_to_build_table);

    append_row_to_build_table.flip();
    std::vector<bool>& append_row_to_shuffle_table = append_row_to_build_table;
    join_state->build_shuffle_buffer->ReserveTable(in_table);
    join_state->build_shuffle_buffer->UnsafeAppendBatch(
        in_table, append_row_to_shuffle_table);

    batch_hashes_partition.reset();
    in_table.reset();

    // If the build table is small enough, broadcast it to all ranks
    // so the probe table can be joined locally.
    // NOTE: broadcasting build table is incorrect if the probe table is
    // replicated.
    // TODO: Simplify this logic into helper functions
    // and/or move to FinalizeBuild?
    if (is_last && join_state->build_parallel && join_state->probe_parallel) {
        // Only consider a broadcast join if we have a single partition
        bool single_partition = join_state->partitions.size() == 1;
        MPI_Allreduce(MPI_IN_PLACE, &single_partition, 1, MPI_C_BOOL, MPI_LAND,
                      MPI_COMM_WORLD);
        if (single_partition) {
            int64_t global_table_size = table_global_memory_size(
                join_state->partitions[0]->build_table_buffer.data_table);
            global_table_size += table_global_memory_size(
                join_state->build_shuffle_buffer->data_table);
            if (global_table_size < get_bcast_join_threshold()) {
                // Mark the build side as replicated.
                join_state->build_parallel = false;
                // Now that we'll have a single partition, disable
                // partitioning altogether. This essentially
                // disables threshold enforcement during any
                // AppendBuildTable or FinalizeBuild calls.
                join_state->DisablePartitioning();

                // We have decided to do a broadcast join. To do this we
                // will execute the following steps:
                //
                // 1. Combine the shuffle buffer into the existing
                // partition. This is necessary so we can shuffle a single
                // table.
                //
                // 2. Broadcast the table across all ranks with allgatherv.
                //
                // 3. Clear the existing JoinPartition state. This is
                // necessary because the allgatherv includes rows that we
                // have already processed and we need to avoid processing
                // them twice.
                //
                // 4. Insert the entire table into the new partition.

                // Step 1: Combine the shuffle buffer into the existing
                // partition

                // Append all the shuffle data to the partition. This allows
                // us to just shuffle 1 table.
                // Dictionary hashes for the key columns which will be used
                // for the partitioning hashes:
                dict_hashes = join_state->GetDictionaryHashesForKeys();

                batch_hashes_partition = hash_keys_table(
                    join_state->build_shuffle_buffer->data_table,
                    join_state->n_keys, SEED_HASH_PARTITION, false, false,
                    dict_hashes);
                join_state->AppendBuildBatch(
                    join_state->build_shuffle_buffer->data_table,
                    batch_hashes_partition);

                // Free the hashes
                batch_hashes_partition.reset();
                // Reset the build shuffle buffer. This will also
                // reset the dictionaries to point to the shared
                // dictionaries and reset the dictionary related flags. This
                // is crucial for correctness.
                join_state->build_shuffle_buffer->Reset();

                // Step 2: Broadcast the table.

                bool all_gather = true;
                // Gather the partition data.
                std::shared_ptr<table_info> gathered_table = gather_table(
                    join_state->partitions[0]->build_table_buffer.data_table,
                    -1, all_gather, true);

                gathered_table =
                    join_state->UnifyBuildTableDictionaryArrays(gathered_table);

                // Step 3: Clear the existing JoinPartition state
                join_state->ResetPartitions();

                // Step 4: Insert the broadcast table.

                // Dictionary hashes for the key columns which will be used
                // for the partitioning hashes:
                dict_hashes = join_state->GetDictionaryHashesForKeys();

                // Get hashes of the new batch (different hashes for
                // partitioning and hash table to reduce conflict) NOTE:
                // Partition hashes need to be consistent across ranks so
                // need to use dictionary hashes. Since we are using
                // dictionary hashes, we don't need dictionaries to be
                // global. In fact, hash_keys_table will ignore the
                // dictionaries entirely when dict_hashes are provided.
                batch_hashes_partition = hash_keys_table(
                    gathered_table, join_state->n_keys, SEED_HASH_PARTITION,
                    false, false, dict_hashes);

                if (use_bloom_filter) {
                    join_state->global_bloom_filter->AddAll(
                        batch_hashes_partition, 0, gathered_table->nrows());
                }
                join_state->AppendBuildBatch(gathered_table,
                                             batch_hashes_partition);
                batch_hashes_partition.reset();
                gathered_table.reset();
            }
        }
    }

    if (shuffle_this_iter(join_state->build_parallel, is_last,
                          join_state->build_shuffle_buffer->data_table,
                          join_state->build_iter, join_state->sync_iter)) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->build_shuffle_buffer->data_table;
        // NOTE: shuffle hashes need to be consistent with partition hashes
        // above. Since we're using dict_hashes, global dictionaries are not
        // required.
        std::shared_ptr<uint32_t[]> shuffle_hashes =
            hash_keys_table(shuffle_table, join_state->n_keys,
                            SEED_HASH_PARTITION, join_state->build_parallel,
                            /*global_dict_needed*/ false, dict_hashes);
        // make dictionaries global for shuffle
        for (size_t i = 0; i < shuffle_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = shuffle_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr,
                                                  join_state->build_parallel);
            }
        }
        mpi_comm_info comm_info_table(shuffle_table->columns);
        comm_info_table.set_counts(shuffle_hashes, join_state->build_parallel);
        std::shared_ptr<table_info> new_data =
            shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                                 comm_info_table, join_state->build_parallel);
        shuffle_hashes.reset();

        // Reset the build shuffle buffer. This will also
        // reset the dictionaries to point to the shared dictionaries
        // and reset the dictionary related flags.
        // This is crucial for correctness.
        join_state->build_shuffle_buffer->Reset();

        // unify dictionaries to allow consistent hashing and fast key
        // comparison using indices
        new_data = join_state->UnifyBuildTableDictionaryArrays(new_data);
        dict_hashes = join_state->GetDictionaryHashesForKeys();
        // NOTE: Partition hashes need to be consistent across ranks, so
        // need to use dictionary hashes. Since we are using dictionary
        // hashes, we don't need dictionaries to be global. In fact,
        // hash_keys_table will ignore the dictionaries entirely when
        // dict_hashes are provided.
        std::shared_ptr<uint32_t[]> batch_hashes_partition =
            hash_keys_table(new_data, join_state->n_keys, SEED_HASH_PARTITION,
                            join_state->build_parallel,
                            /*global_dict_needed*/ false, dict_hashes);

        // Add new batch of data to partitions (bulk insert)
        join_state->AppendBuildBatch(new_data, batch_hashes_partition);
        batch_hashes_partition.reset();
    }

    // Finalize build on all partitions if it's the last input batch.
    if (is_last) {
        if (use_bloom_filter && join_state->build_parallel) {
            // Make the bloom filter global.
            join_state->global_bloom_filter->union_reduction();
        }
        join_state->FinalizeBuild();
    }
    join_state->build_iter++;
    return is_last;
}

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
                              bool is_last) {
    if (join_state->probe_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "join_probe_consume_batch: Received non-empty in_table "
                "after "
                "the probe was already finalized!");
        }
        // No processing left.
        // When probe is finalized global is_last has been seen so no need
        // for additional synchronization
        return true;
    }
    auto probeEvent(join_state->probe_event.iteration());
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Make is_last global
    is_last =
        join_stream_sync_is_last(is_last, join_state->probe_iter, join_state);

    // Update the number of received input rows
    join_state->num_input_probe_table_rows += in_table->nrows();

    // Update active partition state (temporarily) for hashing and
    // comparison functions.
    std::shared_ptr<JoinPartition>& active_partition =
        join_state->partitions[0];

    // Pin the first partition in memory. This only pins the data structures
    // in the first iteration and is a NOP in all future iterations.
    active_partition->pin();

    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices.
    // NOTE: key columns in build_table_buffer (of all partitions),
    // probe_table_buffers (of all partitions), build_shuffle_buffer and
    // probe_shuffle_buffer use the same dictionary object for consistency.
    // Non-key DICT columns of probe_table_buffer and probe_shuffle_buffer also
    // share their dictionaries and will also be unified.
    in_table = join_state->UnifyProbeTableDictionaryArrays(in_table);

    active_partition->probe_table = in_table;

    // Determine if a shuffle could be required.
    const bool shuffle_possible =
        join_state->build_parallel && join_state->probe_parallel;

    // Compute join hashes
    std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
        in_table, join_state->n_keys, SEED_HASH_JOIN, shuffle_possible, false);
    active_partition->probe_table_hashes = batch_hashes_join.get();

    // Compute partitioning hashes:
    // NOTE: partition hashes need to be consistent across ranks so need to
    // use dictionary hashes
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = join_state->GetDictionaryHashesForKeys();
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        shuffle_possible, true, dict_hashes);

    join_state->probe_shuffle_buffer->ReserveTable(in_table);

    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    // Fetch the raw array pointers from the arrays for passing
    // to the non-equijoin condition
    std::vector<array_info*> build_table_info_ptrs, probe_table_info_ptrs;
    // Vectors for data
    std::vector<void*> build_col_ptrs, probe_col_ptrs;
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
    if (non_equi_condition) {
        std::tie(build_table_info_ptrs, build_col_ptrs, build_null_bitmaps) =
            get_gen_cond_data_ptrs(
                active_partition->build_table_buffer.data_table);
        std::tie(probe_table_info_ptrs, probe_col_ptrs, probe_null_bitmaps) =
            get_gen_cond_data_ptrs(active_partition->probe_table);
    }

    // probe hash table
    std::vector<bool> append_to_probe_shuffle_buffer(in_table->nrows(), false);
    std::vector<bool> append_to_probe_inactive_partition(in_table->nrows(),
                                                         false);
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        // If just build_parallel = False then we have a broadcast join on
        // the build side. So process all rows.
        //
        // If just probe_parallel = False and build_parallel = True then we
        // still need to check batch_hashes_partition to know which rows to
        // process.
        bool process_on_rank =
            !join_state->build_parallel ||
            hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank;
        // Check the bloom filter if we would need to shuffle data
        // or if we would process the row.
        bool check_bloom_filter = shuffle_possible || process_on_rank;
        // Check bloom filter
        if (use_bloom_filter && check_bloom_filter) {
            // We use batch_hashes_partition to use consistent hashing
            // across ranks for dict-encoded string arrays
            if (!join_state->global_bloom_filter->Find(
                    batch_hashes_partition[i_row])) {
                join_state->num_bloom_filter_misses++;
                if (probe_table_outer) {
                    // Add unmatched rows from probe table to output table
                    build_idxs.push_back(-1);
                    probe_idxs.push_back(i_row);
                }
                continue;
            }
        }
        if (process_on_rank) {
            join_state->num_processed_probe_table_rows++;
            // TODO Add a fast path without this check for the single
            // partition case.
            if (active_partition->is_in_partition(
                    batch_hashes_partition[i_row])) {
                handle_probe_input_for_partition<
                    build_table_outer, probe_table_outer, non_equi_condition>(
                    join_state->cond_func, active_partition.get(), i_row,
                    build_idxs, probe_idxs, build_table_info_ptrs,
                    probe_table_info_ptrs, build_col_ptrs, probe_col_ptrs,
                    build_null_bitmaps, probe_null_bitmaps);
            } else {
                append_to_probe_inactive_partition[i_row] = true;
            }
        } else if (shuffle_possible) {
            append_to_probe_shuffle_buffer[i_row] = true;
        }
    }

    join_state->AppendProbeBatchToInactivePartition(
        in_table, batch_hashes_join, batch_hashes_partition,
        append_to_probe_inactive_partition);
    join_state->probe_shuffle_buffer->UnsafeAppendBatch(
        in_table, append_to_probe_shuffle_buffer);
    append_to_probe_inactive_partition.clear();
    append_to_probe_shuffle_buffer.clear();

    // Reset active partition state
    active_partition->probe_table = nullptr;
    active_partition->probe_table_hashes = nullptr;

    // Free hash memory
    batch_hashes_partition.reset();
    batch_hashes_join.reset();

    // Insert output rows into the output buffer:
    join_state->output_buffer->AppendJoinOutput(
        active_partition->build_table_buffer.data_table, std::move(in_table),
        build_idxs, probe_idxs, build_kept_cols, probe_kept_cols);
    build_idxs.clear();
    probe_idxs.clear();

    if (shuffle_this_iter(shuffle_possible, is_last,
                          join_state->probe_shuffle_buffer->data_table,
                          join_state->probe_iter, join_state->sync_iter)) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->probe_shuffle_buffer->data_table;
        // NOTE: shuffle hashes need to be consistent with partition hashes
        // above
        std::shared_ptr<uint32_t[]> shuffle_hashes = hash_keys_table(
            shuffle_table, join_state->n_keys, SEED_HASH_PARTITION,
            shuffle_possible, false, dict_hashes);
        // make dictionaries global for shuffle
        for (size_t i = 0; i < shuffle_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = shuffle_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr,
                                                  /*is_parallel*/ true);
            }
        }

        mpi_comm_info comm_info_table(shuffle_table->columns);
        comm_info_table.set_counts(shuffle_hashes, /*is_parallel*/ true);
        std::shared_ptr<table_info> new_data =
            shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                                 comm_info_table, /*is_parallel*/ true);
        shuffle_hashes.reset();

        // Reset the probe shuffle buffer. This will also
        // reset the dictionaries to point to the shared dictionaries
        // and reset the dictionary related flags.
        // This is crucial for correctness.
        join_state->probe_shuffle_buffer->Reset();

        // Unify dictionaries to allow consistent hashing and fast key
        // comparison using indices.
        new_data = join_state->UnifyProbeTableDictionaryArrays(
            new_data, /*only_keys*/ false);

        // NOTE: partition hashes need to be consistent across ranks so need
        // to use dictionary hashes
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
            new_data_dict_hashes = join_state->GetDictionaryHashesForKeys();
        // NOTE: Partition hashes need to be consistent across ranks, so
        // need to use dictionary hashes. Since we are using dictionary
        // hashes, we don't need dictionaries to be global. In fact,
        // hash_keys_table will ignore the dictionaries entirely when
        // dict_hashes are provided.
        std::shared_ptr<uint32_t[]> batch_hashes_partition = hash_keys_table(
            new_data, join_state->n_keys, SEED_HASH_PARTITION, shuffle_possible,
            /*global_dict_needed*/ false, new_data_dict_hashes);

        // probe hash table with new data
        std::shared_ptr<uint32_t[]> batch_hashes_join =
            hash_keys_table(new_data, join_state->n_keys, SEED_HASH_JOIN,
                            shuffle_possible, false);
        active_partition->probe_table = new_data;
        active_partition->probe_table_hashes = batch_hashes_join.get();

        // Fetch the raw array pointers from the arrays for passing
        // to the non-equijoin condition
        std::vector<array_info*> build_table_info_ptrs, probe_table_info_ptrs;
        // Vectors for data
        std::vector<void*> build_col_ptrs, probe_col_ptrs;
        // Vectors for null bitmaps for fast null checking from the cfunc
        std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
        if (non_equi_condition) {
            std::tie(build_table_info_ptrs, build_col_ptrs,
                     build_null_bitmaps) =
                get_gen_cond_data_ptrs(
                    active_partition->build_table_buffer.data_table);
            std::tie(probe_table_info_ptrs, probe_col_ptrs,
                     probe_null_bitmaps) =
                get_gen_cond_data_ptrs(active_partition->probe_table);
        }
        join_state->num_processed_probe_table_rows += new_data->nrows();

        append_to_probe_inactive_partition.resize(new_data->nrows(), false);
        for (size_t i_row = 0; i_row < new_data->nrows(); i_row++) {
            // TODO Add a fast path without this check for the single
            // partition case and another one which uses AppendBatch
            // for the single partition non bloom filter case.
            if (active_partition->is_in_partition(
                    batch_hashes_partition[i_row])) {
                handle_probe_input_for_partition<
                    build_table_outer, probe_table_outer, non_equi_condition>(
                    join_state->cond_func, active_partition.get(), i_row,
                    build_idxs, probe_idxs, build_table_info_ptrs,
                    probe_table_info_ptrs, build_col_ptrs, probe_col_ptrs,
                    build_null_bitmaps, probe_null_bitmaps);
            } else {
                append_to_probe_inactive_partition[i_row] = true;
            }
        }

        join_state->AppendProbeBatchToInactivePartition(
            new_data, batch_hashes_join, batch_hashes_partition,
            append_to_probe_inactive_partition);

        // Reset active partition state
        active_partition->probe_table_hashes = nullptr;
        active_partition->probe_table = nullptr;

        batch_hashes_join.reset();
        batch_hashes_partition.reset();

        join_state->output_buffer->AppendJoinOutput(
            active_partition->build_table_buffer.data_table,
            std::move(new_data), build_idxs, probe_idxs, build_kept_cols,
            probe_kept_cols);
        build_idxs.clear();
        probe_idxs.clear();
    }

    if (is_last && build_table_outer) {
        // We need a reduction of build misses if the probe table is
        // distributed and the build table is not.
        bool build_needs_reduction =
            join_state->probe_parallel && !join_state->build_parallel;
        // Add unmatched rows from build table to output table
        if (build_needs_reduction) {
            generate_build_table_outer_rows_for_partition<true>(
                active_partition.get(), build_idxs, probe_idxs);
        } else {
            generate_build_table_outer_rows_for_partition<false>(
                active_partition.get(), build_idxs, probe_idxs);
        }

        // Use the dummy probe table since all indices are -1
        join_state->output_buffer->AppendJoinOutput(
            active_partition->build_table_buffer.data_table,
            join_state->dummy_probe_table, build_idxs, probe_idxs,
            build_kept_cols, probe_kept_cols);
        build_idxs.clear();
        probe_idxs.clear();
    }

    join_state->probe_iter++;

    if (is_last) {
        // Free the 0th partition:
        join_state->partitions[0].reset();
        active_partition.reset();
        // Finalize and produce output from the inactive partitions.
        // This will pin the partitions (one at a time), generate all the
        // output from it and then free it.
        join_state->FinalizeProbeForInactivePartitions<
            build_table_outer, probe_table_outer, non_equi_condition>(
            build_kept_cols, probe_kept_cols);
        // Finalize the probe step:
        join_state->FinalizeProbe();
    }
    return is_last;
}

/**
 * @brief Initialize a new streaming join state for specified array types
 * and number of keys (called from Python)
 *
 * @param arr_c_types array types of build table columns (Bodo_CTypes ints)
 * @param n_arrs number of build table columns
 * @param n_keys number of join keys
 * @param build_table_outer whether to produce left outer join
 * @param probe_table_outer whether to produce right outer join
 * @param cond_func pointer to function that evaluates non-equality
 * condition. If there is no non-equality condition, this should be NULL.
 * @param build_parallel whether the build table is distributed
 * @param probe_parallel whether the probe table is distributed
 * @param output_batch_size Batch size for reading output.
 * @param op_pool_size_bytes Size of the operator buffer pool for this join
 * operator. This is only applicable for the hash join case at this time.
 * If it's set to -1, we will use a fixed portion of the total available
 * memory (based on JOIN_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL).
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(
    int8_t* build_arr_c_types, int8_t* build_arr_array_types, int n_build_arrs,
    int8_t* probe_arr_c_types, int8_t* probe_arr_array_types, int n_probe_arrs,
    uint64_t n_keys, bool build_table_outer, bool probe_table_outer,
    cond_expr_fn_t cond_func, bool build_parallel, bool probe_parallel,
    int64_t output_batch_size, uint64_t sync_iter, int64_t op_pool_size_bytes) {
    // nested loop join is required if there are no equality keys
    if (n_keys == 0) {
        return new NestedLoopJoinState(
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs),
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            std::vector<int8_t>(probe_arr_c_types,
                                probe_arr_c_types + n_probe_arrs),
            std::vector<int8_t>(probe_arr_array_types,
                                probe_arr_array_types + n_probe_arrs),
            build_table_outer, probe_table_outer, cond_func, build_parallel,
            probe_parallel, output_batch_size, sync_iter);
    }

    return new HashJoinState(
        std::vector<int8_t>(build_arr_c_types,
                            build_arr_c_types + n_build_arrs),
        std::vector<int8_t>(build_arr_array_types,
                            build_arr_array_types + n_build_arrs),
        std::vector<int8_t>(probe_arr_c_types,
                            probe_arr_c_types + n_probe_arrs),
        std::vector<int8_t>(probe_arr_array_types,
                            probe_arr_array_types + n_probe_arrs),
        n_keys, build_table_outer, probe_table_outer, cond_func, build_parallel,
        probe_parallel, output_batch_size, sync_iter, op_pool_size_bytes);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param join_state_ join state pointer
 * @param in_table build table batch
 * @param is_last is last batch locally
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool join_build_consume_batch_py_entry(JoinState* join_state_,
                                       table_info* in_table, bool is_last) {
    // nested loop join is required if there are no equality keys
    if (join_state_->n_keys == 0) {
        return nested_loop_join_build_consume_batch_py_entry(
            (NestedLoopJoinState*)join_state_, in_table, is_last);
    }

    HashJoinState* join_state = (HashJoinState*)join_state_;

    try {
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
        return join_build_consume_batch(join_state,
                                        std::unique_ptr<table_info>(in_table),
                                        has_bloom_filter, is_last);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return false;
}

/**
 * @brief Python wrapper to consume probe table batch and produce output
 * table batch
 *
 * @param join_state_ join state pointer
 * @param in_table probe table batch
 * @param kept_build_col_nums indices of kept columns in build table
 * @param num_kept_build_cols Length of kept_build_col_nums
 * @param kept_probe_col_nums indices of kept columns in probe table
 * @param num_kept_probe_cols Length of kept_probe_col_nums
 * @param[out] total_rows Store the number of rows in the output batch in
 * case all columns are dead.
 * @param is_last is last batch
 * @param produce_output whether to produce output rows
 * @param[out] request_input whether to request input rows from preceding
 * operators
 * @return table_info* output table batch
 */
table_info* join_probe_consume_batch_py_entry(
    JoinState* join_state_, table_info* in_table, uint64_t* kept_build_col_nums,
    int64_t num_kept_build_cols, uint64_t* kept_probe_col_nums,
    int64_t num_kept_probe_cols, int64_t* total_rows, bool is_last,
    bool* out_is_last, bool produce_output, bool* request_input) {
    // Request input rows from preceding operators by default
    *request_input = true;

    // Step 1: Initialize output buffer
    try {
        std::vector<uint64_t> build_kept_cols(
            kept_build_col_nums, kept_build_col_nums + num_kept_build_cols);
        std::vector<uint64_t> probe_kept_cols(
            kept_probe_col_nums, kept_probe_col_nums + num_kept_probe_cols);
        join_state_->InitOutputBuffer(build_kept_cols, probe_kept_cols);

        std::unique_ptr<table_info> input_table =
            std::unique_ptr<table_info>(in_table);

        // nested loop join is required if there are no equality keys
        if (join_state_->n_keys == 0) {
            is_last = nested_loop_join_probe_consume_batch(
                (NestedLoopJoinState*)join_state_, std::move(input_table),
                std::move(build_kept_cols), std::move(probe_kept_cols),
                is_last);
        } else {
            HashJoinState* join_state = (HashJoinState*)join_state_;

#ifndef CONSUME_PROBE_BATCH
#define CONSUME_PROBE_BATCH(build_table_outer, probe_table_outer,           \
                            has_non_equi_cond, use_bloom_filter,            \
                            build_table_outer_exp, probe_table_outer_exp,   \
                            has_non_equi_cond_exp, use_bloom_filter_exp)    \
    if (build_table_outer == build_table_outer_exp &&                       \
        probe_table_outer == probe_table_outer_exp &&                       \
        has_non_equi_cond == has_non_equi_cond_exp &&                       \
        use_bloom_filter == use_bloom_filter_exp) {                         \
        is_last = join_probe_consume_batch<                                 \
            build_table_outer_exp, probe_table_outer_exp,                   \
            has_non_equi_cond_exp, use_bloom_filter_exp>(                   \
            join_state, std::move(input_table), std::move(build_kept_cols), \
            std::move(probe_kept_cols), is_last);                           \
    }
#endif

            bool contain_non_equi_cond = join_state->cond_func != NULL;

            bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

            CONSUME_PROBE_BATCH(
                join_state->build_table_outer, join_state->probe_table_outer,
                contain_non_equi_cond, has_bloom_filter, true, true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                false, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, true,
                                false, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                false, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                contain_non_equi_cond, has_bloom_filter, false,
                                false, false, false)
#undef CONSUME_PROBE_BATCH
        }

        // If after emitting the next batch we'll have more than a full
        // batch left then we don't need to request input. This is to avoid
        // allocating more memory than necessary and increasing cache
        // coherence
        if (join_state_->output_buffer->total_remaining >
            (2 * join_state_->output_buffer->active_chunk_capacity)) {
            *request_input = false;
        }

        table_info* out_table;
        if (!produce_output) {
            *total_rows = 0;
            out_table =
                new table_info(*join_state_->output_buffer->dummy_output_chunk);
        } else {
            auto [out_table_shared, chunk_size] =
                join_state_->output_buffer->PopChunk(
                    /*force_return*/ is_last);
            *total_rows = chunk_size;
            out_table = new table_info(*out_table_shared);
        }
        // This is the last output if we've already seen all input (i.e.
        // is_last) and there's no more output remaining in the output_buffer:
        *out_is_last =
            is_last && join_state_->output_buffer->total_remaining == 0;
        return out_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief delete join state (called from Python after probe loop is
 * finished)
 *
 * @param join_state join state pointer to delete
 */
void delete_join_state(JoinState* join_state_) {
    // nested loop join is required if there are no equality keys
    if (join_state_->n_keys == 0) {
        NestedLoopJoinState* join_state = (NestedLoopJoinState*)join_state_;
        delete join_state;
    } else {
        HashJoinState* join_state = (HashJoinState*)join_state_;
        delete join_state;
    }
}

uint64_t get_op_pool_bytes_pinned(JoinState* join_state) {
    return ((HashJoinState*)join_state)->op_pool_bytes_pinned();
}

uint64_t get_op_pool_bytes_allocated(JoinState* join_state) {
    return ((HashJoinState*)join_state)->op_pool_bytes_allocated();
}

uint32_t get_num_partitions(JoinState* join_state) {
    return ((HashJoinState*)join_state)->partitions.size();
}

uint32_t get_partition_num_top_bits_by_idx(JoinState* join_state, int64_t idx) {
    try {
        std::vector<std::shared_ptr<JoinPartition>>& partitions =
            ((HashJoinState*)join_state)->partitions;
        if (idx >= static_cast<int64_t>(partitions.size())) {
            throw std::runtime_error(
                "get_partition_num_top_bits_by_idx: partition index " +
                std::to_string(idx) + " out of bound: " + std::to_string(idx) +
                " >= " + std::to_string(partitions.size()));
        }
        return partitions[idx]->get_num_top_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

uint32_t get_partition_top_bitmask_by_idx(JoinState* join_state, int64_t idx) {
    try {
        std::vector<std::shared_ptr<JoinPartition>>& partitions =
            ((HashJoinState*)join_state)->partitions;
        if (idx >= partitions.size()) {
            throw std::runtime_error(
                "get_partition_top_bitmask_by_idx: partition index " +
                std::to_string(idx) + " out of bound: " + std::to_string(idx) +
                " >= " + std::to_string(partitions.size()));
        }
        return partitions[idx]->get_top_bitmask();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

PyMODINIT_FUNC PyInit_stream_join_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_join_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, join_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, join_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, join_probe_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_join_state);
    SetAttrStringFromVoidPtr(m, nested_loop_join_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, generate_array_id);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_pinned);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_allocated);
    SetAttrStringFromVoidPtr(m, get_num_partitions);
    SetAttrStringFromVoidPtr(m, get_partition_num_top_bits_by_idx);
    SetAttrStringFromVoidPtr(m, get_partition_top_bitmask_by_idx);

    return m;
}
