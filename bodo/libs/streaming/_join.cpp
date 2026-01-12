#include "_join.h"
#include <arrow/util/bit_util.h>
#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <arrow/array.h>
#include <arrow/compute/api_aggregate.h>
#include "../_array_hash.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_bodo_to_arrow.h"
#include "../_datetime_utils.h"
#include "../_dict_builder.h"
#include "../_distributed.h"
#include "../_memory_budget.h"
#include "../_nested_loop_join_impl.h"
#include "../_query_profile_collector.h"
#include "../_shuffle.h"
#include "../_table_builder_utils.h"
#include "../_utils.h"
#include "_shuffle.h"

// When estimating the required size of the OperatorBufferPool, we add some
// headroom to be conservative. These macros define the bounds of this headroom.
// The headroom is at most 16MiB or 5% of the size of the largest partition.
#define OP_POOL_EST_MAX_HEADROOM (size_t)(16UL * 1024 * 1024)
#define OP_POOL_EST_HEADROOM_FRACTION 0.05

/* --------------------------- HashHashJoinTable -------------------------- */

uint32_t HashHashJoinTable::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        return (*(
            this->join_partition->build_table_join_hashes_guard.value()))[iRow];
    } else {
        return this->join_partition->probe_table_hashes
            [-(iRow + this->join_partition->probe_table_hashes_offset) - 1];
    }
}

/* ------------------------------------------------------------------------ */

/* ------------------------ KeyEqualHashJoinTable ------------------------- */

bool KeyEqualHashJoinTable::operator()(const int64_t iRowA,
                                       const int64_t iRowB) const {
    const std::shared_ptr<table_info>& build_table =
        this->join_partition->build_table_buffer->data_table;
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

    // For joins in SQL, all NA keys have already been pruned.
    bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB, this->n_keys,
                              this->join_partition->is_na_equal);
    return test;
}

/* ------------------------------------------------------------------------ */

/* ---------------------------- JoinPartition ----------------------------- */
#pragma region  // JoinPartition

JoinPartition::JoinPartition(
    size_t num_top_bits_, uint32_t top_bitmask_,
    const std::shared_ptr<bodo::Schema> build_table_schema_,
    const std::shared_ptr<bodo::Schema> probe_table_schema_,
    const uint64_t n_keys_, bool build_table_outer_, bool probe_table_outer_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>&
        build_table_dict_builders_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>&
        probe_table_dict_builders_,
    bool is_active_, HashJoinMetrics& metrics_,
    bodo::OperatorBufferPool* op_pool_,
    const std::shared_ptr<::arrow::MemoryManager> op_mm_,
    bodo::OperatorScratchPool* op_scratch_pool_,
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm_,
    bool is_na_equal_, bool is_mark_join_)
    : build_table_schema(build_table_schema_),
      probe_table_schema(probe_table_schema_),
      build_table_dict_builders(build_table_dict_builders_),
      probe_table_dict_builders(probe_table_dict_builders_),
      build_table_buffer(std::make_unique<TableBuildBuffer>(
          build_table_schema_, build_table_dict_builders, op_pool_, op_mm_)),
      build_table_join_hashes(op_pool_),
      build_hash_table(std::make_unique<bodo::pinnable<hash_table_t>>(
          0, HashHashJoinTable(this), KeyEqualHashJoinTable(this, n_keys_),
          op_scratch_pool_)),
      num_rows_in_group(std::make_unique<bodo::pinnable<bodo::vector<size_t>>>(
          op_scratch_pool_)),
      build_row_to_group_map(
          std::make_unique<bodo::pinnable<bodo::vector<size_t>>>(
              op_scratch_pool_)),
      groups(op_scratch_pool_),
      groups_offsets(op_scratch_pool_),
      build_table_matched(op_scratch_pool_),
      dummy_probe_table(alloc_table(probe_table_schema_)),
      is_na_equal(is_na_equal_),
      is_mark_join(is_mark_join_),
      num_top_bits(num_top_bits_),
      top_bitmask(top_bitmask_),
      build_table_outer(build_table_outer_),
      probe_table_outer(probe_table_outer_),
      n_keys(n_keys_),
      op_pool(op_pool_),
      op_mm(op_mm_),
      op_scratch_pool(op_scratch_pool_),
      op_scratch_mm(op_scratch_mm_),
      is_active(is_active_),
      metrics(metrics_) {
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
    } else {
        this->build_table_buffer_chunked =
            std::make_unique<ChunkedTableBuilder>(
                this->build_table_schema, this->build_table_dict_builders,
                INACTIVE_PARTITION_TABLE_CHUNK_SIZE,
                DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
    }
}

inline bool JoinPartition::is_in_partition(
    const uint32_t& hash) const noexcept {
    constexpr size_t uint32_bits = sizeof(uint32_t) * CHAR_BIT;
    // Shifting uint32_t by 32 bits is undefined behavior.
    // Ref:
    // https://stackoverflow.com/questions/18799344/shifting-a-32-bit-integer-by-32-bits
    return (this->num_top_bits == 0)
               ? true
               : ((hash >> (uint32_bits - this->num_top_bits)) ==
                  this->top_bitmask);
}

template <bool is_active>
std::vector<std::shared_ptr<JoinPartition>> JoinPartition::SplitPartition(
    size_t num_levels) {
    assert(this->pinned_);
    if (num_levels != 1) {
        throw std::runtime_error(
            "JoinPartition::SplitPartition: We currently only support "
            "splitting a partition into 2 at a time.");
    }
    constexpr size_t uint32_bits = sizeof(uint32_t) * CHAR_BIT;
    if (this->num_top_bits >= (uint32_bits - 1)) {
        throw std::runtime_error(
            "Cannot split the partition further. Out of hash bits.");
    }

    // Release the hash-table memory:
    this->build_hash_table_guard.reset();
    this->build_hash_table.reset();
    // Release group info memory:
    this->num_rows_in_group_guard.reset();
    this->num_rows_in_group.reset();
    this->build_row_to_group_map_guard.reset();
    this->build_row_to_group_map.reset();
    this->groups_guard.value()->resize(0);
    this->groups_guard.value()->shrink_to_fit();
    this->groups_guard.reset();
    this->groups_offsets_guard.value()->resize(0);
    this->groups_offsets_guard.value()->shrink_to_fit();
    this->groups_offsets_guard.reset();
    // Release the bitmap memory
    this->build_table_matched_guard.value()->resize(0);
    this->build_table_matched_guard.value()->shrink_to_fit();
    this->build_table_matched_guard.reset();

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
        this->build_table_schema, this->probe_table_schema, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        is_active, this->metrics, this->op_pool, this->op_mm,
        this->op_scratch_pool, this->op_scratch_mm, this->is_na_equal,
        this->is_mark_join);
    std::shared_ptr<JoinPartition> new_part2 = std::make_shared<JoinPartition>(
        this->num_top_bits + 1, (this->top_bitmask << 1) + 1,
        this->build_table_schema, this->probe_table_schema, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders, false,
        this->metrics, this->op_pool, this->op_mm, this->op_scratch_pool,
        this->op_scratch_mm, this->is_na_equal, this->is_mark_join);

    std::vector<bool> append_partition1;
    if (is_active) {
        // In the active case, partition this->build_table_buffer directly

        // Compute partitioning hashes
        time_pt start = start_timer();
        std::shared_ptr<uint32_t[]> build_table_partitioning_hashes =
            hash_keys_table(this->build_table_buffer->data_table, this->n_keys,
                            SEED_HASH_PARTITION, false, false, dict_hashes);
        this->metrics.repartitioning_part_hashing_time += end_timer(start);
        this->metrics.repartitioning_part_hashing_nrows +=
            this->build_table_buffer->data_table->nrows();

        // Put the build data in the new partitions.
        append_partition1.resize(this->build_table_buffer->data_table->nrows(),
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
        start = start_timer();
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
        new_part1->build_table_buffer->ReserveTable(
            this->build_table_buffer->data_table, append_partition1,
            append_partition1_sum);
        new_part1->build_table_buffer->UnsafeAppendBatch(
            this->build_table_buffer->data_table, append_partition1,
            append_partition1_sum);
        this->metrics.repartitioning_active_part1_append_time +=
            end_timer(start);
        this->metrics.repartitioning_active_part1_append_nrows +=
            append_partition1_sum;

        // Update safely appended row count for the new active partition:
        new_part1->build_safely_appended_nrows = append_partition1_sum;

        start = start_timer();
        append_partition1.flip();
        std::vector<bool>& append_partition2 = append_partition1;
        // The rows between this->build_safely_appended_nrows
        // and this->build_table_buffer->data_table->nrows() shouldn't
        // be copied over to either partition:
        for (size_t i = this->build_safely_appended_nrows;
             i < append_partition2.size(); i++) {
            append_partition2[i] = false;
        }

        new_part2->build_table_buffer_chunked->AppendBatch(
            this->build_table_buffer->data_table, append_partition2);
        this->metrics.repartitioning_active_part2_append_time +=
            end_timer(start);
        this->metrics.repartitioning_active_part2_append_nrows +=
            (this->build_safely_appended_nrows - append_partition1_sum);

        // We do not rebuild the hash table here (for new_part1 which is the new
        // active partition). That needs to be handled by the caller.

    } else {
        // In the inactive case, partition build_table_buffer chunk by chunk
        this->build_table_buffer_chunked->Finalize();

        // Free build_table_buffer in case we started activation and
        // some columns reserved memory during Activate.
        this->build_table_buffer.reset();

        time_pt start;
        this->metrics.repartitioning_inactive_n_pop_chunks +=
            this->build_table_buffer_chunked->chunks.size();
        while (!this->build_table_buffer_chunked->chunks.empty()) {
            start = start_timer();
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked->PopChunk();
            this->metrics.repartitioning_inactive_pop_chunk_time +=
                end_timer(start);

            // Compute partitioning hashes
            start = start_timer();
            std::shared_ptr<uint32_t[]> build_table_partitioning_hashes_chunk =
                hash_keys_table(build_table_chunk, this->n_keys,
                                SEED_HASH_PARTITION, false, false, dict_hashes);
            this->metrics.repartitioning_part_hashing_time += end_timer(start);
            this->metrics.repartitioning_part_hashing_nrows +=
                build_table_chunk->nrows();

            // Put the build data in the sub partitions.
            append_partition1.resize(build_table_nrows_chunk, false);
            for (int64_t i_row = 0; i_row < build_table_nrows_chunk; i_row++) {
                append_partition1[i_row] = new_part1->is_in_partition(
                    build_table_partitioning_hashes_chunk[i_row]);
            }

            start = start_timer();
            new_part1->build_table_buffer_chunked->AppendBatch(
                build_table_chunk, append_partition1);

            append_partition1.flip();
            std::vector<bool>& append_partition2 = append_partition1;

            new_part2->build_table_buffer_chunked->AppendBatch(
                build_table_chunk, append_partition2);
            this->metrics.repartitioning_inactive_append_time +=
                end_timer(start);
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
    size_t build_table_nrows = this->build_table_buffer->data_table->nrows();
    size_t join_hashes_cur_len = build_table_join_hashes_->size();

    // TODO: Do this processing in batches of 4K rows (for handling inactive
    // partition case where we will do this for the entire table)!!

    size_t n_unhashed_rows = build_table_nrows - join_hashes_cur_len;
    if (n_unhashed_rows > 0) {
        ScopedTimer ht_hashing_timer(this->metrics.build_ht_hashing_time);
        // Compute hashes for the batch:
        std::unique_ptr<uint32_t[]> join_hashes = hash_keys_table(
            this->build_table_buffer->data_table, this->n_keys, SEED_HASH_JOIN,
            /*is_parallel*/ false,
            /*global_dict_needed*/ false, /*dict_hashes*/ nullptr,
            /*start_row_offset*/ join_hashes_cur_len);
        // Append the hashes:
        build_table_join_hashes_->insert(build_table_join_hashes_->end(),
                                         join_hashes.get(),
                                         join_hashes.get() + n_unhashed_rows);
        ht_hashing_timer.finalize();
        this->metrics.build_ht_hashing_nrows += n_unhashed_rows;
    }

    // Create reference variables for easier usage.
    auto& num_rows_in_group_ = this->num_rows_in_group_guard.value();
    auto& build_row_to_group_map_ = this->build_row_to_group_map_guard.value();
    auto& build_hash_table_ = this->build_hash_table_guard.value();

    // Add all the rows in the build_table_buffer that haven't
    // already been added to the hash table.
    ScopedTimer ht_insert_timer(this->metrics.build_ht_insert_time);
    while (this->curr_build_size <
           this->build_table_buffer->data_table->nrows()) {
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
        this->metrics.build_ht_insert_nrows++;
    }
    ht_insert_timer.finalize();
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
#ifndef _WIN32
        std::partial_sum(num_rows_in_group_->cbegin(),
                         num_rows_in_group_->cend(),
                         groups_offsets_->begin() + 1);
#else
        // std::partial_sum requires pointer_to method in pinnable_ptr on
        // Windows, which seems difficult to implement correctly. Use a loop
        // instead.
        size_t sum = num_rows_in_group_->at(0);
        for (size_t i = 1; i < num_groups; ++i) {
            (*groups_offsets_)[i] = sum;
            sum += num_rows_in_group_->at(i);
        }
        (*groups_offsets_)[num_groups] = sum;
#endif
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
    size_t build_table_rows = this->build_table_buffer->data_table->nrows();
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
        ScopedTimer append_timer(this->metrics.build_appends_active_time);
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        // Now append the rows. This will always succeed since we've
        // reserved space upfront.
        this->build_table_buffer->UnsafeAppendBatch(in_table);
        append_timer.finalize();
        this->metrics.build_appends_active_nrows += in_table->nrows();
        // Compute the hashes and add rows to the hash table now.
        this->BuildHashTable();

        /// Commit "transaction". Only update this after all the rows have
        /// been appended to build_table_buffer, the hash table _and_
        /// build_table_join_hashes.
        this->build_safely_appended_nrows = this->curr_build_size;
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        this->build_table_buffer_chunked->AppendBatch(in_table);
        this->metrics.build_appends_inactive_time += end_timer(start);
        this->metrics.build_appends_inactive_nrows += in_table->nrows();
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
        ScopedTimer append_timer(this->metrics.build_appends_active_time);
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        // Now append the rows. This will always succeed since we've
        // reserved space upfront.
        uint64_t append_rows_sum = std::accumulate(
            append_rows.begin(), append_rows.end(), (uint64_t)0);
        this->build_table_buffer->UnsafeAppendBatch(in_table, append_rows,
                                                    append_rows_sum);
        append_timer.finalize();
        this->metrics.build_appends_active_nrows += append_rows_sum;
        // Compute the hashes and add rows to the hash table now.
        this->BuildHashTable();

        /// Commit "transaction". Only update this after all the rows have
        /// been appended to build_table_buffer, the hash table _and_
        /// build_table_join_hashes.
        this->build_safely_appended_nrows = this->curr_build_size;
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        uint64_t append_rows_sum = std::accumulate(
            append_rows.begin(), append_rows.end(), (uint64_t)0);
        this->build_table_buffer_chunked->AppendBatch(in_table, append_rows,
                                                      append_rows_sum, 0);
        this->metrics.build_appends_inactive_time += end_timer(start);
        this->metrics.build_appends_inactive_nrows += append_rows_sum;
    }
}

bool JoinPartition::IsUniqueBuildPartition() {
    return this->build_table_buffer->data_table->nrows() ==
           this->curr_build_size;
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
    ScopedTimer apt(this->metrics.build_finalize_activate_time);
    this->ActivatePartition();
    apt.finalize();
    // Make sure all rows from build_table_buffer have been inserted
    // into the hash table. This is idempotent.
    this->BuildHashTable();
    // Finalize the groups. This step is idempotent.
    ScopedTimer fgt(this->metrics.build_finalize_groups_time);
    this->FinalizeGroups();
    fgt.finalize();
    if (this->build_table_outer) {
        // This step is idempotent by definition.
        this->build_table_matched_guard.value()->resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer->data_table->nrows()),
            0);
    }
}

void JoinPartition::InitProbeInputBuffer() {
    if (this->probe_table_buffer_chunked != nullptr) {
        // Already initialized. We only initialize on the first
        // iteration.
        return;
    }
    this->probe_table_buffer_chunked = std::make_unique<ChunkedTableBuilder>(
        this->probe_table_schema, this->probe_table_dict_builders,
        INACTIVE_PARTITION_TABLE_CHUNK_SIZE,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}

void JoinPartition::AppendInactiveProbeBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::vector<bool>& append_rows, const int64_t in_table_start_offset,
    int64_t table_nrows) {
    assert(in_table_start_offset >= 0);
    auto probe_table_buffer_join_hashes_(
        bodo::pin(this->probe_table_buffer_join_hashes));
    if (table_nrows == -1) {
        // Convert default of -1 to all rows in the table (starting from
        // 'in_table_start_offset').
        table_nrows = in_table->nrows() - in_table_start_offset;
    }
    assert(table_nrows >= 0);
    assert(static_cast<int64_t>(append_rows.size()) == table_nrows);
    for (size_t i_row = 0; i_row < static_cast<size_t>(table_nrows); i_row++) {
        if (append_rows[i_row]) {
            probe_table_buffer_join_hashes_->push_back(join_hashes[i_row]);
        }
    }
    this->probe_table_buffer_chunked->AppendBatch(in_table, append_rows,
                                                  in_table_start_offset);
}

/**
 * @brief Helper function for join_probe_consume_batch and
 * FinalizeProbeForInactivePartition to probe the
 * build table and return group ids.
 *
 * NOTE: Assumes that the row is in the partition.
 * NOTE: Inlined since it's called inside loops.
 *
 * @param ht Partition's hash table to probe.
 *  NOTE: This function assumes that the partition is already pinned.
 * @param i_row Row index in partition->probe_table to probe.
 * @return group_id for the row. Similar to build table convention, 0 means not
 * in build table (ids start from 1).
 */
inline size_t handle_probe_input_for_partition(
    const bodo::pin_guard<bodo::pinnable<JoinPartition::hash_table_t>>& ht,
    size_t i_row) {
    auto iter = ht->find(-i_row - 1);
    const size_t group_id = (iter == ht->end()) ? 0 : iter->second;
    return group_id;
}

/**
 * @brief Helper function for join_probe_consume_batch and
 * FinalizeProbeForInactivePartition to update 'build_idxs'
 * and 'probe_idxs' and produce output when they are "full" (to avoid OOM).
 * It also updates the 'build_table_matched'
 * bitmap of the partition in the `build_table_outer` case.
 *
 * NOTE: Assumes that the row is in the partition.
 * NOTE: Inlined since it's called inside loops.
 *
 * @tparam build_table_outer
 * @tparam probe_table_outer
 * @tparam non_equi_condition
 * @tparam is_anti_join
 * @param cond_func Condition function to use. `nullptr` for the
 * all-equality conditions case.
 * @param[in, out] partition Partition that this row belongs to.
 *  NOTE: This function assumes that the partition is already pinned.
 * @param i_row Row index in partition->probe_table to produce the output.
 * @param batch_start_row Offset to skip when looking up in group_ids.
 * @param group_ids Vector of group ids for each row.
 * @param[in, out] build_idxs Build table indices for the output. This will
 * be updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will
 * be updated in place.
 *
 * These parameters are output of get_gen_cond_data_ptrs on the
 * build and probe table and are only relevant for the condition function
 * case:
 * @param build_table_info_ptrs
 * @param probe_table_info_ptrs
 * @param build_col_ptrs
 * @param probe_col_ptrs
 * @param build_null_bitmaps
 * @param probe_null_bitmaps
 *
 * These parameters are for output generation with AppendJoinOutput (see
 * join_probe_consume_batch for more details):
 * @param output_buffer
 * @param build_table
 * @param probe_table
 * @param build_kept_cols
 * @param probe_kept_cols
 * @param[in, out] append_time -- Increment this with the time spent in
 * AppendJoinOutput.
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool is_anti_join>
inline void produce_probe_output(
    cond_expr_fn_t cond_func, JoinPartition* partition, const size_t i_row,
    const size_t batch_start_row, const bodo::vector<int64_t>& group_ids,
    bodo::vector<int64_t>& build_idxs, bodo::vector<int64_t>& probe_idxs,
    std::vector<array_info*>& build_table_info_ptrs,
    std::vector<array_info*>& probe_table_info_ptrs,
    std::vector<void*>& build_col_ptrs, std::vector<void*>& probe_col_ptrs,
    std::vector<void*>& build_null_bitmaps,
    std::vector<void*>& probe_null_bitmaps,
    const std::shared_ptr<ChunkedTableBuilder>& output_buffer,
    const std::shared_ptr<table_info>& build_table,
    const std::shared_ptr<table_info>& probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols,
    HashJoinMetrics::time_t& append_time, bool is_mark_join = false) {
    const int64_t group_id = group_ids[i_row - batch_start_row];

    // -1 means ignore row (e.g. shouldn't be processed on this rank)
    if (group_id == -1) {
        return;
    }

    // 0 means not found in build table
    if (group_id == 0) {
        if constexpr (probe_table_outer) {
            // Add unmatched rows from probe table to output table
            build_idxs.push_back(-1);
            probe_idxs.push_back(i_row);
        }
        return;
    }

    if constexpr (is_anti_join && probe_table_outer) {
        // In the anti join case, if we have a match in the build table, we
        // don't want to output anything for this probe row.
        return;
    }

    // TODO Pass pinned groups_offsets vector instead of pinning for each
    // row.
    auto& partition_groups_offsets_ = partition->groups_offsets_guard.value();
    const size_t group_start_idx = (*partition_groups_offsets_)[group_id - 1];
    const size_t group_end_idx = (*partition_groups_offsets_)[group_id - 1 + 1];
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
        if constexpr (non_equi_condition) {
            // Check for matches with the non-equality portion.
            bool match =
                cond_func(probe_table_info_ptrs.data(),
                          build_table_info_ptrs.data(), probe_col_ptrs.data(),
                          build_col_ptrs.data(), probe_null_bitmaps.data(),
                          build_null_bitmaps.data(), i_row, j_build);
            if (!match) {
                continue;
            }
            has_match = true;
        }
        if constexpr (build_table_outer) {
            SetBitTo(partition_build_table_matched_->data(), j_build, true);
        }
        if constexpr (!is_anti_join) {
            build_idxs.push_back(j_build);
            probe_idxs.push_back(i_row);
        }

        // Produce output in a chunked builder periodically to avoid OOM
        // (not used in mark join case to avoid appending duplicate output rows)
        if (build_idxs.size() >= static_cast<size_t>(STREAMING_BATCH_SIZE) &&
            !is_mark_join) {
            time_pt start_append = start_timer();
            output_buffer->AppendJoinOutput(
                build_table, probe_table, build_idxs, probe_idxs,
                build_kept_cols, probe_kept_cols, is_mark_join);
            append_time += end_timer(start_append);
            build_idxs.clear();
            probe_idxs.clear();
        }
    }
    // non-equality condition only branch
    if (!has_match && probe_table_outer) {
        // Add unmatched rows from probe table to output table
        // NOTE: build_idxs.size() check like above isn't required here since
        // in the worst case, this would add as many rows as the probe batch,
        // which is always a small number.
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
    if constexpr (requires_reduction) {
        auto pin = *build_table_matched_;
        MPI_Allreduce_bool_or({pin.data(), pin.size()});
    }
    int n_pes, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Add unmatched rows from build table to output table
    for (size_t i_row = 0;
         i_row < partition->build_table_buffer->data_table->nrows(); i_row++) {
        if ((!requires_reduction ||
             ((i_row % n_pes) == static_cast<size_t>(my_rank)))) {
            bool has_match = GetBit(build_table_matched_->data(), i_row);
            // TODO Add the same check here to materialize early if build_idxs
            // grows beyond STREAMING_BATCH_SIZE.
            if (!has_match) {
                build_idxs.push_back(i_row);
                probe_idxs.push_back(-1);
            }
        }
    }
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool is_anti_join>
void JoinPartition::FinalizeProbeForInactivePartition(
    cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols,
    const std::shared_ptr<ChunkedTableBuilder>& output_buffer) {
    assert(this->pinned_);
    // This should always be 0 here but just in case
    this->probe_table_hashes_offset = 0;

    bodo::vector<int64_t> group_ids;
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

    if constexpr (non_equi_condition) {
        get_gen_cond_data_ptrs(this->build_table_buffer->data_table,
                               &build_table_info_ptrs, &build_col_ptrs,
                               &build_null_bitmaps);
    }

    auto probe_table_buffer_join_hashes_(
        bodo::pin(this->probe_table_buffer_join_hashes));

    // Loop over chunked probe table's buffers/hashes and process one at a
    // time At each loop iteration we mutate `this->probe_table`, so all
    // `i_row` and `probe_idxs` indices refer only to the local chunk.
    // Therefore, adding unmatched rows from the probe table to the output
    // must be done at each iteration, while that probe table is in memory.
    // Adding unmatched rows from the build table should be done at the end.
    this->probe_table_hashes = probe_table_buffer_join_hashes_->data();
    this->probe_table_buffer_chunked->Finalize();
    time_pt start_pop, start_ht_probe, start_produce_probe;
    HashJoinMetrics::time_t append_time = 0;
    this->metrics.probe_inactive_pop_chunk_n_chunks +=
        this->probe_table_buffer_chunked->chunks.size();
    // For ease of reference
    const auto& ht = this->build_hash_table_guard.value();
    while (!this->probe_table_buffer_chunked->chunks.empty()) {
        start_pop = start_timer();
        auto [probe_table_chunk, probe_table_nrows] =
            this->probe_table_buffer_chunked->PopChunk();
        this->metrics.probe_inactive_pop_chunk_time += end_timer(start_pop);
        this->probe_table = std::move(probe_table_chunk);

        if constexpr (non_equi_condition) {
            get_gen_cond_data_ptrs(this->probe_table, &probe_table_info_ptrs,
                                   &probe_col_ptrs, &probe_null_bitmaps);
        }

        start_ht_probe = start_timer();
        group_ids.resize(this->probe_table->nrows());
        for (size_t i_row = 0; i_row < this->probe_table->nrows(); i_row++) {
            group_ids[i_row] = handle_probe_input_for_partition(ht, i_row);
        }
        this->metrics.ht_probe_time += end_timer(start_ht_probe);
        append_time = 0;
        start_produce_probe = start_timer();
        for (size_t i_row = 0; i_row < this->probe_table->nrows(); i_row++) {
            produce_probe_output<build_table_outer, probe_table_outer,
                                 non_equi_condition, is_anti_join>(
                cond_func, this, i_row, 0, group_ids, build_idxs, probe_idxs,
                build_table_info_ptrs, probe_table_info_ptrs, build_col_ptrs,
                probe_col_ptrs, build_null_bitmaps, probe_null_bitmaps,
                output_buffer, this->build_table_buffer->data_table,
                this->probe_table, build_kept_cols, probe_kept_cols,
                append_time, this->is_mark_join);
        }
        this->metrics.produce_probe_out_idxs_time +=
            end_timer(start_produce_probe) - append_time;
        group_ids.clear();

        output_buffer->AppendJoinOutput(
            this->build_table_buffer->data_table, this->probe_table, build_idxs,
            probe_idxs, build_kept_cols, probe_kept_cols, this->is_mark_join);
        build_idxs.clear();
        probe_idxs.clear();

        this->probe_table_hashes += probe_table_nrows;
        this->probe_table.reset();
    }

    // Add unmatched rows from build table to output table
    if (build_table_outer) {
        time_pt start_build_outer = start_timer();
        // If an inactive partition exists, this means that the build side is
        // distributed and therefore no reduction is required.
        generate_build_table_outer_rows_for_partition<
            /*requires_reduction*/ false>(this, build_idxs, probe_idxs);
        this->metrics.build_outer_output_idx_time +=
            end_timer(start_build_outer);

        output_buffer->AppendJoinOutput(this->build_table_buffer->data_table,
                                        this->dummy_probe_table, build_idxs,
                                        probe_idxs, build_kept_cols,
                                        probe_kept_cols, this->is_mark_join);
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
        this->build_table_buffer_chunked->Finalize();

        // Do a single ReserveTable call to allocate all required space in a
        // single call:
        this->build_table_buffer->ReserveTable(
            *(this->build_table_buffer_chunked));

        // This will work without error because we've already allocated
        // all the required space:
        while (!this->build_table_buffer_chunked->chunks.empty()) {
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked->PopChunk();
            this->build_table_buffer->UnsafeAppendBatch(build_table_chunk);
        }

        // Mark this partition as activated once we've moved the data
        // from the chunked buffer to a contiguous buffer:
        this->is_active = true;
        this->build_safely_appended_nrows =
            this->build_table_buffer->data_table->nrows();

        // Free the chunked buffer state entirely since it's not needed anymore.
        this->build_table_buffer_chunked.reset();
    }
}

void JoinPartition::pin() {
    if (!this->pinned_) {
        this->build_table_buffer->pin();
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
        this->pinned_ = true;
    }
}

void JoinPartition::unpin() {
    if (this->pinned_) {
        this->build_table_buffer->unpin();
        this->build_table_join_hashes_guard.reset();
        this->build_hash_table_guard.reset();
        this->num_rows_in_group_guard.reset();
        this->build_row_to_group_map_guard.reset();
        this->groups_guard.reset();
        this->groups_offsets_guard.reset();
        this->build_table_matched_guard.reset();
        this->pinned_ = false;
    }
}

#pragma endregion  // JoinPartition
/* ------------------------------------------------------------------------ */

/* --------------------------- JoinState ---------------------------------- */
#pragma region  // JoinState

JoinState::JoinState(const std::shared_ptr<bodo::Schema> build_table_schema_,
                     const std::shared_ptr<bodo::Schema> probe_table_schema_,
                     uint64_t n_keys_, bool build_table_outer_,
                     bool probe_table_outer_, bool force_broadcast_,
                     cond_expr_fn_t cond_func_, bool build_parallel_,
                     bool probe_parallel_, int64_t output_batch_size_,
                     int64_t sync_iter_, int64_t op_id_, bool is_na_equal_,
                     bool is_mark_join_)
    : build_table_schema(build_table_schema_),
      probe_table_schema(probe_table_schema_),
      n_keys(n_keys_),
      cond_func(cond_func_),
      build_table_outer(build_table_outer_),
      probe_table_outer(probe_table_outer_),
      force_broadcast(force_broadcast_),
      is_na_equal(is_na_equal_),
      is_mark_join(is_mark_join_),
      build_parallel(build_parallel_),
      probe_parallel(probe_parallel_),
      output_batch_size(output_batch_size_),
      dummy_probe_table(alloc_table(probe_table_schema_)),
      op_id(op_id_) {
    this->key_dict_builders.resize(this->n_keys);

    // Create dictionary builders for key columns:
    for (uint64_t i = 0; i < this->n_keys; i++) {
        this->key_dict_builders[i] = create_dict_builder_for_array(
            this->build_table_schema->column_types[i]->copy(), true);
        // Also set this as the dictionary of the dummy probe table
        // for consistency, else there will be issues during output
        // generation.
        set_array_dict_from_builder(this->dummy_probe_table->columns[i],
                                    this->key_dict_builders[i]);

        if (this->build_table_schema->column_types[i]->array_type ==
            bodo_array_type::DICT) {
            if (this->probe_table_schema->column_types[i]->array_type !=
                bodo_array_type::DICT) {
                throw std::runtime_error(
                    "HashJoinState: Key column array types don't match "
                    "between build and probe tables!");
            }
        }
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        build_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in build table:
    for (size_t i = this->n_keys; i < this->build_table_schema->ncols(); i++) {
        build_table_non_key_dict_builders.emplace_back(
            create_dict_builder_for_array(
                this->build_table_schema->column_types[i]->copy(), false));
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        probe_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in probe table:
    for (size_t i = this->n_keys; i < this->probe_table_schema->ncols(); i++) {
        std::shared_ptr<DictionaryBuilder> builder =
            create_dict_builder_for_array(
                this->probe_table_schema->column_types[i]->copy(), false);
        probe_table_non_key_dict_builders.emplace_back(builder);
        // Also set this as the dictionary of the dummy probe table
        // for consistency, else there will be issues during output
        // generation.
        set_array_dict_from_builder(this->dummy_probe_table->columns[i],
                                    builder);
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
    std::shared_ptr<bodo::Schema> out_schema = std::make_shared<bodo::Schema>();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    dict_builders.reserve(probe_kept_cols.size() + build_kept_cols.size());

    for (uint64_t i_col : probe_kept_cols) {
        std::unique_ptr<bodo::DataType> col_type =
            this->probe_table_schema->column_types[i_col]->copy();
        // In the build outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        if (this->build_table_outer) {
            col_type = col_type->to_nullable_type();
        }
        out_schema->append_column(std::move(col_type));
        dict_builders.push_back(this->probe_table_dict_builders[i_col]);
    }

    // Add the mark output column if this is a mark join.
    if (this->is_mark_join) {
        if (!build_kept_cols.empty()) {
            throw std::runtime_error(
                "JoinState::InitOutputBuffer: Mark join should not output "
                "build table columns.");
        }
        out_schema->append_column(std::make_unique<bodo::DataType>(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL));
        dict_builders.push_back(nullptr);
    }

    for (uint64_t i_col : build_kept_cols) {
        std::unique_ptr<bodo::DataType> col_type =
            this->build_table_schema->column_types[i_col]->copy();
        // In the probe outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        if (this->probe_table_outer) {
            col_type = col_type->to_nullable_type();
        }
        out_schema->append_column(std::move(col_type));
        dict_builders.push_back(this->build_table_dict_builders[i_col]);
    }

    this->output_buffer = std::make_shared<ChunkedTableBuilder>(
        out_schema, dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}

std::shared_ptr<table_info> JoinState::UnifyBuildTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table) {
    return unify_dictionary_arrays_helper(
        in_table, this->build_table_dict_builders, this->n_keys, false);
}

std::shared_ptr<table_info> JoinState::UnifyProbeTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table,
    bool only_transpose_existing_on_key_cols) {
    return unify_dictionary_arrays_helper(
        in_table, this->probe_table_dict_builders, this->n_keys,
        only_transpose_existing_on_key_cols);
}

void JoinState::ReportBuildStageMetrics(std::vector<MetricBase>& metrics_out) {
    metrics_out.reserve(metrics_out.size() + 2);
    metrics_out.emplace_back(TimerMetric(
        "min_max_update_time", this->metrics.build_min_max_update_time));
    metrics_out.emplace_back(TimerMetric(
        "min_max_finalize_time", this->metrics.build_min_max_finalize_time));
}

void JoinState::ReportProbeStageMetrics(std::vector<MetricBase>& metrics_out) {
    assert(this->probe_input_finalized);
    if (this->op_id == -1) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 4);

    // Get time spent appending to, total number of rows appended to, and max
    // reached size of the output buffer
    metrics_out.emplace_back(
        TimerMetric("output_append_time", this->output_buffer->append_time));
    MetricBase::StatValue output_total_size = this->output_buffer->total_size;
    metrics_out.emplace_back(
        StatMetric("output_total_nrows", output_total_size));
    MetricBase::StatValue output_total_rem =
        this->output_buffer->total_remaining;
    metrics_out.emplace_back(
        StatMetric("output_total_nrows_rem_at_finalize", output_total_rem));
    MetricBase::StatValue output_peak_nrows =
        this->output_buffer->max_reached_size;
    metrics_out.emplace_back(
        StatMetric("output_peak_nrows", output_peak_nrows));
}

#pragma endregion  // JoinState
/* ------------------------------------------------------------------------ */

/* ---------------------------- HashJoinState ----------------------------- */
#pragma region  // HashJoinState

HashJoinState::HashJoinState(
    const std::shared_ptr<bodo::Schema> build_table_schema_,
    const std::shared_ptr<bodo::Schema> probe_table_schema_, uint64_t n_keys_,
    bool build_table_outer_, bool probe_table_outer_, bool force_broadcast_,
    cond_expr_fn_t cond_func_, bool build_parallel_, bool probe_parallel_,
    int64_t output_batch_size_, int64_t sync_iter_, int64_t op_id_,
    int64_t op_pool_size_bytes, size_t max_partition_depth_, bool is_na_equal_,
    bool is_mark_join_)
    : JoinState(build_table_schema_, probe_table_schema_, n_keys_,
                build_table_outer_, probe_table_outer_, force_broadcast_,
                cond_func_, build_parallel_, probe_parallel_,
                output_batch_size_, sync_iter_, op_id_, is_na_equal_,
                is_mark_join_),
      // Create the operator buffer pool
      op_pool(std::make_unique<bodo::OperatorBufferPool>(
          op_id_,
          ((op_pool_size_bytes == -1)
               ? static_cast<uint64_t>(
                     bodo::BufferPool::Default()->get_memory_size_bytes() *
                     JOIN_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL)
               : op_pool_size_bytes),
          bodo::BufferPool::Default(),
          JOIN_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD)),
      op_mm(bodo::buffer_memory_manager(op_pool.get())),
      op_scratch_pool(
          std::make_unique<bodo::OperatorScratchPool>(this->op_pool.get())),
      op_scratch_mm(bodo::buffer_memory_manager(this->op_scratch_pool.get())),
      max_partition_depth(max_partition_depth_),
      build_shuffle_state(build_table_schema_, this->build_table_dict_builders,
                          this->n_keys, this->build_iter, this->sync_iter,
                          op_id_),
      probe_shuffle_state(probe_table_schema_, this->probe_table_dict_builders,
                          this->n_keys, this->probe_iter, this->sync_iter,
                          op_id_),
      // Create a build buffer for NA values to skip the hash table.
      build_na_key_buffer(build_table_schema_, this->build_table_dict_builders,
                          output_batch_size_,
                          DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES) {
    // Turn partitioning on by default.
    bool enable_partitioning = true;

    if (!this->build_parallel) {
        // For now, we will allow re-partitioning only in the case where the
        // build side is distributed. If we allow re-partitioning in the
        // replicated build side case, we must assume that the partitioning
        // state is identical on all ranks. This might not always be true.
        // Therefore, we will turn off re-partitioning in the replicated
        // build case altogether.
        // XXX Revisit this in the future if needed.
        enable_partitioning = false;
    } else {
        // Force enable/disable partitioning if env var set. This is
        // primarily for unit testing purposes.
        char* enable_partitioning_env_ =
            std::getenv("BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING");
        if (enable_partitioning_env_) {
            if (std::strcmp(enable_partitioning_env_, "0") == 0) {
                enable_partitioning = false;
            } else if (std::strcmp(enable_partitioning_env_, "1") == 0) {
                enable_partitioning = true;
            } else {
                throw std::runtime_error(
                    "HashJoinState::HashJoinState: "
                    "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING set to "
                    "unsupported value: " +
                    std::string(enable_partitioning_env_));
            }
        } else if (!this->op_pool->is_spilling_enabled()) {
            // There's no point in repartitioning when spilling is not
            // available anyway.
            enable_partitioning = false;
        }
    }

    if (!enable_partitioning) {
        this->DisablePartitioning();
    }

    if (char* debug_partitioning_env_ =
            std::getenv("BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING")) {
        this->debug_partitioning = !std::strcmp(debug_partitioning_env_, "1");
    }

    // Create the initial partition
    this->partitions.emplace_back(std::make_shared<JoinPartition>(
        0, 0, build_table_schema_, probe_table_schema_, n_keys_,
        build_table_outer_, probe_table_outer_, this->build_table_dict_builders,
        this->probe_table_dict_builders,
        /*is_active*/ true, this->metrics, this->op_pool.get(), this->op_mm,
        this->op_scratch_pool.get(), this->op_scratch_mm, this->is_na_equal,
        this->is_mark_join));

    this->global_bloom_filter = create_bloom_filter();

    CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &this->shuffle_comm),
              "HashJoinState: MPI error on MPI_Comm_dup:");

    if (char* unique_vals_limit_env_ =
            std::getenv("BODO_JOIN_UNIQUE_VALUES_LIMIT")) {
        this->unique_values_limit = (size_t)(std::atol(unique_vals_limit_env_));
    }

    // Allocate the min/max array values for each key column
    for (size_t key_col = 0; key_col < n_keys; key_col++) {
        std::unique_ptr<bodo::DataType>& dtype =
            probe_table_schema_->column_types[key_col];
        if (this->IsValidRuntimeJoinFilterMinMaxColumn(dtype)) {
            if (dtype->array_type == bodo_array_type::NUMPY ||
                dtype->array_type == bodo_array_type::NULLABLE_INT_BOOL) {
                this->min_max_values.emplace_back(
                    alloc_nullable_array_all_nulls(
                        2, probe_table_schema_->column_types[key_col]->c_type));
            } else {
                assert(dtype->array_type == bodo_array_type::STRING ||
                       dtype->array_type == bodo_array_type::DICT);
                std::shared_ptr<array_info> str_arr =
                    alloc_string_array(Bodo_CTypes::STRING, 2, 0);
                this->min_max_values.emplace_back(str_arr);
                str_arr->set_null_bit<bodo_array_type::STRING>(0, false);
                str_arr->set_null_bit<bodo_array_type::STRING>(1, false);
                // zero out offsets because there is no data initially
                offset_t* offsets =
                    str_arr->data2<bodo_array_type::STRING, offset_t>();
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
            }
        } else {
            this->min_max_values.emplace_back(std::nullopt);
        }
        if (IsValidRuntimeJoinFilterUniqueValuesColumn(dtype)) {
            std::unordered_set<int64_t> unique_set;
            this->unique_values.emplace_back(unique_set);
        } else {
            this->unique_values.emplace_back(std::nullopt);
        }
    }
    build_dict_hit_bitmap.resize(n_keys);
}

bool JoinState::IsValidRuntimeJoinFilterMinMaxColumn(
    std::unique_ptr<bodo::DataType>& dtype) {
    switch (dtype->array_type) {
        case bodo_array_type::NUMPY:
        case bodo_array_type::NULLABLE_INT_BOOL: {
            switch (dtype->c_type) {
                case Bodo_CTypes::INT8:
                case Bodo_CTypes::INT16:
                case Bodo_CTypes::INT32:
                case Bodo_CTypes::INT64:
                case Bodo_CTypes::UINT8:
                case Bodo_CTypes::UINT16:
                case Bodo_CTypes::UINT32:
                case Bodo_CTypes::UINT64:
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::FLOAT32:
                case Bodo_CTypes::FLOAT64:
                case Bodo_CTypes::DATETIME: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        }
        // For now, we do not allow computing the min/max on regular
        // string columns. However, we do have support for computing
        // them on string arrays as a sub-routine of the min/max
        // computations on dictionary encoded arrays.
        // case bodo_array_type::STRING:
        case bodo_array_type::DICT: {
            return true;
        }
        default: {
            return false;
        }
    }
}

bool HashJoinState::IsValidRuntimeJoinFilterUniqueValuesColumn(
    std::unique_ptr<bodo::DataType>& dtype) {
    switch (dtype->array_type) {
        case bodo_array_type::NUMPY:
        case bodo_array_type::NULLABLE_INT_BOOL: {
            switch (dtype->c_type) {
                case Bodo_CTypes::INT8:
                case Bodo_CTypes::INT16:
                case Bodo_CTypes::INT32:
                case Bodo_CTypes::INT64:
                case Bodo_CTypes::UINT8:
                case Bodo_CTypes::UINT16:
                case Bodo_CTypes::UINT32:
                case Bodo_CTypes::UINT64:
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::FLOAT32:
                case Bodo_CTypes::FLOAT64:
                case Bodo_CTypes::DATETIME: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        }
        default: {
            return false;
        }
    }
}

void HashJoinState::SplitPartition(size_t idx) {
    if (this->partitions[idx]->get_num_top_bits() >=
        this->max_partition_depth) {
        // TODO Eventually, this should lead to falling back
        // to nested loop join for this partition.
        // (https://bodo.atlassian.net/browse/BSE-535).
        throw std::runtime_error(
            "HashJoinState::SplitPartition: Cannot split partition beyond max "
            "partition depth of: " +
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
    time_pt start_split = start_timer();
    if (this->partitions[idx]->is_active_partition()) {
        new_partitions = this->partitions[idx]->SplitPartition<true>();
    } else {
        new_partitions = this->partitions[idx]->SplitPartition<false>();
    }
    this->metrics.repartitioning_time += end_timer(start_split);
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
        0, 0, this->build_table_schema, this->probe_table_schema, this->n_keys,
        this->build_table_outer, this->probe_table_outer,
        this->build_table_dict_builders, this->probe_table_dict_builders,
        /*is_active*/ true, this->metrics, this->op_pool.get(), this->op_mm,
        this->op_scratch_pool.get(), this->op_scratch_mm, this->is_na_equal,
        this->is_mark_join));
}

void HashJoinState::AppendBuildBatchHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch<true>(in_table);
        return;
    }
    time_pt start_part_check = start_timer();
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
    this->metrics.build_input_partition_check_time +=
        end_timer(start_part_check);
    this->metrics.build_input_partition_check_nrows += in_table->nrows();

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
            this->metrics.n_repartitions_in_appends++;
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

    time_pt start_part_check = start_timer();
    std::vector<std::vector<bool>> append_rows_by_partition;
    append_rows_by_partition.resize(this->partitions.size());
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] = std::vector<bool>(in_table->nrows());
    }

    size_t append_rows_sum = 0;
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            append_rows_sum++;
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
    this->metrics.build_input_partition_check_time +=
        end_timer(start_part_check);
    this->metrics.build_input_partition_check_nrows += append_rows_sum;
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
            this->metrics.n_repartitions_in_appends++;
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
                probe_idxs, build_kept_cols, probe_kept_cols,
                this->is_mark_join);
            // We don't need to clear probe_idxs since in the next iteration
            // we will just resize it (and contents don't need to be
            // changed).
            build_idxs.clear();
            // 'build_na_table_chunk' will go out of scope and be freed
            // automatically here.
        }
    }
}

std::string HashJoinState::GetPartitionStateString() const {
    std::string partition_state = "[";
    for (const auto& partition : this->partitions) {
        if (partition != nullptr) {
            size_t num_top_bits = partition->get_num_top_bits();
            uint32_t top_bitmask = partition->get_top_bitmask();
            partition_state +=
                fmt::format("({0}, {1:#b})", num_top_bits, top_bitmask);
        } else {
            // In case we call this function after the partition has been
            // free-d.
            partition_state += "(UNKNOWN)";
        }
        partition_state += ",";
    }
    partition_state += "]";
    return partition_state;
}

void HashJoinState::ReportBuildStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    assert(this->build_input_finalized);
    if (this->op_id == -1) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 128);

    metrics_out.emplace_back(
        StatMetric("bcast_join", this->metrics.is_build_bcast_join, true));
    metrics_out.emplace_back(
        TimerMetric("bcast_time", this->metrics.build_bcast_time));
    metrics_out.emplace_back(StatMetric(
        "bloom_filter_enabled", this->metrics.bloom_filter_enabled, true));
    metrics_out.emplace_back(TimerMetric(
        "appends_active_time", this->metrics.build_appends_active_time));
    metrics_out.emplace_back(StatMetric(
        "appends_active_nrows", this->metrics.build_appends_active_nrows));
    metrics_out.emplace_back(TimerMetric(
        "appends_inactive_time", this->metrics.build_appends_inactive_time));
    metrics_out.emplace_back(StatMetric(
        "appends_inactive_nrows", this->metrics.build_appends_inactive_nrows));
    metrics_out.emplace_back(
        TimerMetric("input_partition_check_time",
                    this->metrics.build_input_partition_check_time));
    metrics_out.emplace_back(
        StatMetric("input_partition_check_nrows",
                   this->metrics.build_input_partition_check_nrows));
    metrics_out.emplace_back(
        TimerMetric("ht_hashing_time", this->metrics.build_ht_hashing_time));
    metrics_out.emplace_back(
        StatMetric("ht_hashing_nrows", this->metrics.build_ht_hashing_nrows));
    metrics_out.emplace_back(
        TimerMetric("ht_insert_time", this->metrics.build_ht_insert_time));
    metrics_out.emplace_back(
        StatMetric("ht_insert_nrows", this->metrics.build_ht_insert_nrows));
    metrics_out.emplace_back(
        StatMetric("n_partitions", this->metrics.n_partitions));
    metrics_out.emplace_back(
        StatMetric("n_rows", this->metrics.build_nrows, !this->build_parallel));
    metrics_out.emplace_back(StatMetric(
        "n_groups", this->metrics.build_n_groups, !this->build_parallel));
    metrics_out.emplace_back(StatMetric(
        "n_repartitions_in_appends", this->metrics.n_repartitions_in_appends));
    metrics_out.emplace_back(
        StatMetric("n_repartitions_in_finalize",
                   this->metrics.n_repartitions_in_finalize));
    metrics_out.emplace_back(TimerMetric("repartitioning_time_total",
                                         this->metrics.repartitioning_time));
    metrics_out.emplace_back(
        TimerMetric("repartitioning_part_hashing_time",
                    this->metrics.repartitioning_part_hashing_time));
    metrics_out.emplace_back(
        StatMetric("repartitioning_part_hashing_nrows",
                   this->metrics.repartitioning_part_hashing_nrows));
    metrics_out.emplace_back(
        TimerMetric("repartitioning_active_part1_append_time",
                    this->metrics.repartitioning_active_part1_append_time));
    metrics_out.emplace_back(
        StatMetric("repartitioning_active_part1_append_nrows",
                   this->metrics.repartitioning_active_part1_append_nrows));
    metrics_out.emplace_back(
        TimerMetric("repartitioning_active_part2_append_time",
                    this->metrics.repartitioning_active_part2_append_time));
    metrics_out.emplace_back(
        StatMetric("repartitioning_active_part2_append_nrows",
                   this->metrics.repartitioning_active_part2_append_nrows));
    metrics_out.emplace_back(
        TimerMetric("repartitioning_inactive_append_time",
                    this->metrics.repartitioning_inactive_append_time));
    metrics_out.emplace_back(
        TimerMetric("repartitioning_inactive_pop_chunk_time",
                    this->metrics.repartitioning_inactive_pop_chunk_time));
    metrics_out.emplace_back(
        StatMetric("repartitioning_inactive_n_pop_chunks",
                   this->metrics.repartitioning_inactive_n_pop_chunks));
    metrics_out.emplace_back(
        TimerMetric("finalize_time_total", this->metrics.build_finalize_time));
    metrics_out.emplace_back(TimerMetric(
        "finalize_groups_time", this->metrics.build_finalize_groups_time));
    metrics_out.emplace_back(TimerMetric(
        "finalize_activate_time", this->metrics.build_finalize_activate_time));
    metrics_out.emplace_back(
        TimerMetric("input_part_hashing_time",
                    this->metrics.build_input_part_hashing_time));
    metrics_out.emplace_back(TimerMetric("bloom_filter_add_time",
                                         this->metrics.bloom_filter_add_time));
    metrics_out.emplace_back(
        TimerMetric("bloom_filter_union_reduction_time",
                    this->metrics.bloom_filter_union_reduction_time));
    metrics_out.emplace_back(StatMetric("max_partition_size_bytes",
                                        this->metrics.max_partition_size_bytes,
                                        !this->build_parallel));
    metrics_out.emplace_back(StatMetric(
        "total_partitions_size_bytes",
        this->metrics.total_partitions_size_bytes, !this->build_parallel));
    metrics_out.emplace_back(StatMetric("final_op_pool_size_bytes",
                                        this->metrics.final_op_pool_size_bytes,
                                        !this->build_parallel));
    metrics_out.emplace_back(
        TimerMetric("filter_na_time", this->metrics.build_filter_na_time));
    metrics_out.emplace_back(StatMetric(
        "filter_na_output_nrows", this->metrics.build_filter_na_output_nrows));
    metrics_out.emplace_back(BlobMetric("final_partitioning_state",
                                        this->metrics.final_partitioning_state,
                                        !this->build_parallel));
    metrics_out.emplace_back(
        TimerMetric("unique_values_update_time",
                    this->metrics.build_unique_values_update_time));
    metrics_out.emplace_back(
        TimerMetric("unique_values_finalize_time",
                    this->metrics.build_unique_values_finalize_time));

    // Get shuffle stats from build shuffle state
    this->build_shuffle_state.ExportMetrics(metrics_out);

    // Get time spent appending to and number of rows in build_na_key_buffer
    metrics_out.emplace_back(TimerMetric(
        "na_buffer_append_time", this->build_na_key_buffer.append_time));
    MetricBase::StatValue na_buffer_nrows =
        this->build_na_key_buffer.total_size;
    metrics_out.emplace_back(StatMetric("na_buffer_nrows", na_buffer_nrows));
    // Determine if the hash table is unique on this rank
    metrics_out.emplace_back(
        StatMetric("build_unique_on_rank", this->metrics.is_build_unique));

    // Get and combine metrics from dict-builders
    DictBuilderMetrics key_dict_builder_metrics;
    DictBuilderMetrics non_key_dict_builder_metrics;
    MetricBase::StatValue n_key_dict_builders = 0;
    MetricBase::StatValue n_non_key_dict_builders = 0;
    for (size_t i = 0; i < this->build_table_dict_builders.size(); i++) {
        const auto& dict_builder = this->build_table_dict_builders[i];
        if (dict_builder != nullptr) {
            if (i < this->n_keys) {
                key_dict_builder_metrics.add_metrics(
                    dict_builder->GetMetrics());
                n_key_dict_builders++;
            } else {
                non_key_dict_builder_metrics.add_metrics(
                    dict_builder->GetMetrics());
                n_non_key_dict_builders++;
            }
        }
    }
    metrics_out.emplace_back(
        StatMetric("n_key_dict_builders", n_key_dict_builders, true));
    key_dict_builder_metrics.add_to_metrics(metrics_out, "key_dict_builders_");
    metrics_out.emplace_back(
        StatMetric("n_non_key_dict_builders", n_non_key_dict_builders, true));
    non_key_dict_builder_metrics.add_to_metrics(metrics_out,
                                                "non_key_dict_builders_");

    // Save a snapshot of the dict-builder metrics of the key columns.
    this->metrics.key_dict_builder_metrics_build_stage_snapshot =
        key_dict_builder_metrics;

    JoinState::ReportBuildStageMetrics(metrics_out);
}

void HashJoinState::FinalizeBuild() {
    time_pt start_finalize = start_timer();
    // Free build shuffle buffer, etc.
    this->build_shuffle_state.Finalize();

    // TODO Finalize the CTBs of inactive partitions before
    // finalizing any partition (will reduce pinned memory).

    // Finalize the NA buffer now that we've seen all the input.
    this->build_na_key_buffer.Finalize();

    // Track if the build table is unique across all partitions on
    // this rank.
    bool is_build_unique = true;

    // Finalize all the partitions and split them as needed:
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
                this->metrics.build_nrows +=
                    this->partitions[i_part]
                        ->build_table_buffer->data_table->nrows();
                this->metrics.build_n_groups +=
                    this->partitions[i_part]
                        ->groups_offsets_guard.value()
                        ->size() -
                    1;
                is_build_unique &=
                    this->partitions[i_part]->IsUniqueBuildPartition();
                // The partition size is roughly the number of bytes pinned
                // through the OperatorBufferPool at this point. All other
                // partitions are either unfinalized (i.e. they don't have any
                // memory allocated directly through the op-pool) or are
                // unpinned.
                size_t est_partition_size = this->op_pool->bytes_pinned();
                this->metrics.total_partitions_size_bytes += est_partition_size;
                this->metrics.max_partition_size_bytes = std::max(
                    static_cast<size_t>(this->metrics.max_partition_size_bytes),
                    est_partition_size);
                // Unpin the partition once we're done.
                this->partitions[i_part]->unpin();
                if (this->debug_partitioning) {
                    std::cerr << "[DEBUG] HashJoinState::FinalizeBuild: "
                                 "Successfully finalized partition "
                              << i_part << ". Estimated partition size: "
                              << BytesToHumanReadableString(est_partition_size)
                              << std::endl;
                }
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
                this->metrics.n_repartitions_in_finalize++;
                this->SplitPartition(i_part);
            }
        }
    }
    // Globally determine if all ranks are empty.
    bool local_empty_build =
        this->partitions.empty() ||
        this->partitions[0]->build_table_buffer->data_table->nrows() == 0;
    if (this->build_parallel) {
        CHECK_MPI(MPI_Allreduce(&local_empty_build, &this->global_build_empty,
                                1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD),
                  "HashJoinState::FinalizeBuild: MPI error on MPI_Allreduce:");
    } else {
        this->global_build_empty = local_empty_build;
    }

    // The estimated required size of the pool is at least the size of the
    // biggest partition.
    size_t est_req_pool_size = this->metrics.max_partition_size_bytes;
    // At this point, all partitions are unpinned, so op_pool->bytes_pinned() is
    // the amount of bytes that *cannot* be unpinned (they go through malloc),
    // so we need to account for it. It is likely that we are over-counting
    // these since they might be part of the max_partition_size already, but its
    // impact should be minimal. This shouldn't be large in practice since these
    // buffers must be <12KiB and there are only a finite number of buffers.
    est_req_pool_size += this->op_pool->bytes_pinned();
    // Add a 5% headroom (or 16MiB, whichever is lower).
    est_req_pool_size +=
        std::min(OP_POOL_EST_MAX_HEADROOM,
                 static_cast<size_t>(std::ceil(OP_POOL_EST_HEADROOM_FRACTION *
                                               est_req_pool_size)));

    if (this->debug_partitioning) {
        std::cerr << "[DEBUG] HashJoinState::FinalizeBuild: Total number of "
                     "partitions: "
                  << this->partitions.size()
                  << ". Estimated max partition size: "
                  << BytesToHumanReadableString(
                         this->metrics.max_partition_size_bytes)
                  << ". Total size of all partitions: "
                  << BytesToHumanReadableString(
                         this->metrics.total_partitions_size_bytes)
                  << ". Estimated required size of Op-Pool: "
                  << BytesToHumanReadableString(est_req_pool_size) << "."
                  << std::endl;
    }

    // Update the query profile information.
    this->metrics.n_partitions = this->partitions.size();
    this->metrics.bloom_filter_enabled =
        (this->global_bloom_filter != nullptr) ? 1 : 0;
    this->metrics.is_build_unique = is_build_unique;

    // We won't be making any new allocations, so we can move
    // the error threshold to 1.0 now. This is required for reducing
    // the budget since that's predicated on having entire budget available.
    this->op_pool->SetErrorThreshold(1.0);
    // If the est_req_pool_size is lower than the original budget, report this
    // new number to the OperatorComptroller.
    if (est_req_pool_size < this->op_pool->get_operator_budget_bytes()) {
        this->op_pool->SetBudget(est_req_pool_size);
    }
    this->metrics.final_op_pool_size_bytes =
        this->op_pool->get_operator_budget_bytes();
    this->metrics.final_partitioning_state = this->GetPartitionStateString();
    this->metrics.build_finalize_time += end_timer(start_finalize);

    JoinState::FinalizeBuild();
}

bool HashJoinState::RuntimeFilter(
    std::shared_ptr<table_info> in_table,
    std::shared_ptr<array_info> row_bitmask,
    const std::vector<int64_t>& join_key_idxs,
    const std::vector<bool>& process_col_bitmask) {
    assert(this->n_keys == join_key_idxs.size());
    assert(this->n_keys == process_col_bitmask.size());
    assert(this->build_input_finalized);
    // Cannot filter if probe is on the outer side.
    // The planner should never generate these and the compiler
    // would raise an exception if it did, so we just add an assert for now.
    assert(!this->probe_table_outer);

    size_t n_bytes = ::arrow::bit_util::BytesForBits(in_table->nrows());
    // If none of the entries in join_key_idxs are -1, then also apply
    // the bloom filter if one is available.
    bool apply_bloom_filter = (this->global_bloom_filter != nullptr);
    for (int64_t idx : join_key_idxs) {
        apply_bloom_filter &= (idx != -1);
    }

    // Key arrays for computing the hashes for the bloom filter.
    std::vector<std::shared_ptr<array_info>> key_arrs_for_bf_hashes;
    if (apply_bloom_filter) {
        key_arrs_for_bf_hashes.reserve(this->n_keys);
    }

    bool applied_any_filter = false;

    // If join_key_idxs[i] is not -1 and process_col_bitmask[i] is true,
    // then apply a column level filter if one is available.
    for (size_t i = 0; i < join_key_idxs.size(); i++) {
        if (join_key_idxs[i] == -1) {
            assert(!apply_bloom_filter);
            continue;
        }
        if (process_col_bitmask[i] || apply_bloom_filter) {
            std::shared_ptr<array_info>& in_arr =
                in_table->columns[join_key_idxs[i]];
            std::shared_ptr<array_info> unified_arr;

            if (this->build_table_dict_builders[i] != nullptr) {
                // If a dict builder exists:
                //  - If only applying column level filter
                //    - If DICT, then transpose and filter out nulls
                //    - If STRING, then skip transpose
                //    - If NESTED, then skip transpose
                //  - If applying bloom filter
                //    - If DICT, then transpose for correct hash calculation
                //    - If STRING, then skip transpose and calculate
                //      hashes directly on the strings
                //    - If NESTED, then transpose for correctness.
                if (in_arr->arr_type == bodo_array_type::DICT) {
                    assert(
                        this->build_table_schema->column_types[i]->array_type ==
                        bodo_array_type::DICT);
                    unified_arr =
                        this->build_table_dict_builders[i]->TransposeExisting(
                            in_arr);
                    // Filter out nulls in the column if we should filter based
                    // on this column. This will filter out both: entries that
                    // didn't exist in the dictionary and regular nulls. The
                    // former is what we want and the latter is a safe
                    // side-effect since we want to filter out nulls anyway
                    // (null keys cannot match with anything in the probe_inner
                    // case).
                    if (process_col_bitmask[i] &&
                        unified_arr->null_bitmask<bodo_array_type::DICT>()) {
                        const uint8_t* out_arr_bitmask =
                            (uint8_t*)(unified_arr->null_bitmask<
                                       bodo_array_type::DICT>());
                        for (size_t i = 0; i < n_bytes; i++) {
                            row_bitmask
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                        uint8_t>()[i] &= out_arr_bitmask[i];
                        }
                        applied_any_filter = true;
                    }
                } else if (in_arr->arr_type == bodo_array_type::STRING) {
                    assert(
                        this->build_table_schema->column_types[i]->array_type ==
                        bodo_array_type::DICT);
                    // Technically, we could do TransposeExisting on this to
                    // convert this to a DICT. However that would require doing
                    // a hashmap lookup on every element (potentially billions),
                    // which would be too expensive for a best-effort filter.
                    // For now, we will simply utilize the bloom filter which
                    // can be used in this case since the strings can be hashed
                    // as is. We may change this in the future.
                    unified_arr = in_arr;
                } else if (is_nested_arr_type(in_arr->arr_type)) {
                    assert(
                        in_arr->arr_type ==
                        this->build_table_schema->column_types[i]->array_type);
                    if (apply_bloom_filter) {
                        unified_arr = this->build_table_dict_builders[i]
                                          ->UnifyDictionaryArray(in_arr);
                    } else {
                        unified_arr = in_arr;
                    }
                } else {
                    throw std::runtime_error(fmt::format(
                        "HashJoinState::RuntimeFilter: Expected input array "
                        "type corresponding to key at index {} to be "
                        "DICT/STRING/STRUCT/MAP/ARRAY_ITEM, but got {} "
                        "instead!",
                        i, GetArrType_as_string(in_arr->arr_type)));
                }
            } else {
                // The types should match exactly. Even when
                // the key column is STRING and the input was originally a DICT,
                // the compiler should've casted it to a STRING before calling
                // this function.
                if (in_arr->arr_type !=
                    this->build_table_schema->column_types[i]->array_type) {
                    throw std::runtime_error(fmt::format(
                        "HashJoinState::RuntimeFilter: Expected input array "
                        "type corresponding to key at index {} to be {}, but "
                        "got {} instead!",
                        i,
                        GetArrType_as_string(
                            this->build_table_schema->column_types[i]
                                ->array_type),
                        GetArrType_as_string(in_arr->arr_type)));
                }
                if (in_arr->dtype !=
                    this->build_table_schema->column_types[i]->c_type) {
                    throw std::runtime_error(fmt::format(
                        "HashJoinState::RuntimeFilter: Expected input data "
                        "type corresponding to key at index {} to be {}, but "
                        "got {} instead!",
                        i,
                        GetDtype_as_string(
                            this->build_table_schema->column_types[i]->c_type),
                        GetDtype_as_string(in_arr->dtype)));
                }
                // If not a dict-encoded array, there's no additional
                // filtering we can do even if process_col_bitmask[i] is true.
                // This can change in the future, e.g. we may maintain column
                // level hash-sets or bloom filters or range information for
                // filtering.
                unified_arr = in_arr;
            }

            // We will need the unified array for calculating the hashes
            // for the bloom filter.
            if (apply_bloom_filter) {
                key_arrs_for_bf_hashes.emplace_back(unified_arr);
            }
        }
    }
    if (apply_bloom_filter) {
        std::shared_ptr<table_info> unified_in_table_only_keys =
            std::make_shared<table_info>(key_arrs_for_bf_hashes);
        // Compute partitioning hashes.
        // NOTE: In the case where the input column is STRING and the join key
        // is originally DICT, we can hash the input strings directly since they
        // would match the dictionary hashes.
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
            dict_hashes = this->GetDictionaryHashesForKeys();
        time_pt start_hash = start_timer();
        std::shared_ptr<uint32_t[]> in_table_hashes_partition =
            hash_keys_table(unified_in_table_only_keys, this->n_keys,
                            SEED_HASH_PARTITION, false, true, dict_hashes);
        this->metrics.join_filter_bloom_filter_hashing_time +=
            end_timer(start_hash);
        this->metrics.join_filter_bloom_filter_hashing_nrows +=
            unified_in_table_only_keys->nrows();
        time_pt start_bloom = start_timer();
        for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
            bool keep =
                arrow::bit_util::GetBit(
                    row_bitmask
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                    i_row) &&
                this->global_bloom_filter->Find(
                    in_table_hashes_partition[i_row]);
            arrow::bit_util::SetBitTo(
                row_bitmask
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                i_row, keep);
        }
        this->metrics.join_filter_bloom_filter_probe_time +=
            end_timer(start_bloom);
        this->metrics.join_filter_bloom_filter_probe_nrows += in_table->nrows();
        applied_any_filter = true;
    }

    if (applied_any_filter) {
        this->metrics.num_runtime_filter_applied_rows += in_table->nrows();
    }

    return applied_any_filter;
}

void HashJoinState::ReportProbeStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    assert(this->probe_input_finalized);
    if (this->op_id == -1) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 64);

    metrics_out.emplace_back(StatMetric(
        "nrows_processed", this->metrics.num_processed_probe_table_rows));
    metrics_out.emplace_back(
        TimerMetric("append_inactive_partitions_time",
                    this->metrics.append_probe_inactive_partitions_time));
    metrics_out.emplace_back(
        StatMetric("append_inactive_partitions_nrows",
                   this->metrics.append_probe_inactive_partitions_nrows));
    metrics_out.emplace_back(
        TimerMetric("inactive_partition_check_time",
                    this->metrics.probe_inactive_partition_check_time));
    metrics_out.emplace_back(
        TimerMetric("ht_probe_time", this->metrics.ht_probe_time));
    metrics_out.emplace_back(
        TimerMetric("produce_probe_out_idxs_time",
                    this->metrics.produce_probe_out_idxs_time));
    metrics_out.emplace_back(
        TimerMetric("build_outer_output_idx_time",
                    this->metrics.build_outer_output_idx_time));
    metrics_out.emplace_back(TimerMetric(
        "join_hashing_time", this->metrics.probe_join_hashing_time));
    metrics_out.emplace_back(TimerMetric(
        "part_hashing_time", this->metrics.probe_part_hashing_time));
    metrics_out.emplace_back(
        TimerMetric("finalize_inactive_partitions_total_time",
                    this->metrics.finalize_probe_inactive_partitions_time));
    metrics_out.emplace_back(TimerMetric(
        "finalize_inactive_partitions_pin_partition_time",
        this->metrics.finalize_probe_inactive_partitions_pin_partition_time));
    metrics_out.emplace_back(
        TimerMetric("inactive_pop_chunk_time",
                    this->metrics.probe_inactive_pop_chunk_time));
    metrics_out.emplace_back(
        StatMetric("inactive_pop_chunk_n_chunks",
                   this->metrics.probe_inactive_pop_chunk_n_chunks));
    metrics_out.emplace_back(
        TimerMetric("filter_na_time", this->metrics.probe_filter_na_time));
    metrics_out.emplace_back(StatMetric(
        "filter_na_output_nrows", this->metrics.probe_filter_na_output_nrows));
    if (this->probe_table_outer) {
        metrics_out.emplace_back(
            StatMetric("probe_outer_bloom_filter_misses",
                       this->metrics.probe_outer_bloom_filter_misses));
    } else {
        metrics_out.emplace_back(
            TimerMetric("join_filter_bloom_filter_hashing_time",
                        this->metrics.join_filter_bloom_filter_hashing_time));
        metrics_out.emplace_back(
            StatMetric("join_filter_bloom_filter_hashing_nrows",
                       this->metrics.join_filter_bloom_filter_hashing_nrows));
        metrics_out.emplace_back(
            TimerMetric("join_filter_bloom_filter_probe_time",
                        this->metrics.join_filter_bloom_filter_probe_time));
        metrics_out.emplace_back(
            StatMetric("join_filter_bloom_filter_probe_nrows",
                       this->metrics.join_filter_bloom_filter_probe_nrows));
        metrics_out.emplace_back(
            StatMetric("num_runtime_filter_applied_rows",
                       this->metrics.num_runtime_filter_applied_rows));
    }

    // Get shuffle stats from probe shuffle state
    this->probe_shuffle_state.ExportMetrics(metrics_out);

    MetricBase::StatValue na_counter = this->probe_na_counter;
    metrics_out.emplace_back(StatMetric("na_counter", na_counter));

    // Get and combine metrics from dict-builders
    DictBuilderMetrics key_dict_builder_metrics;
    DictBuilderMetrics non_key_dict_builder_metrics;
    MetricBase::StatValue n_key_dict_builders = 0;
    MetricBase::StatValue n_non_key_dict_builders = 0;
    for (size_t i = 0; i < this->probe_table_dict_builders.size(); i++) {
        const auto& dict_builder = this->probe_table_dict_builders[i];
        if (dict_builder != nullptr) {
            if (i < this->n_keys) {
                key_dict_builder_metrics.add_metrics(
                    dict_builder->GetMetrics());
                n_key_dict_builders++;
            } else {
                non_key_dict_builder_metrics.add_metrics(
                    dict_builder->GetMetrics());
                n_non_key_dict_builders++;
            }
        }
    }

    // Subtract the metrics from the build stage snapshot.
    // This is important since the key columns' DictBuilders are shared
    // by the build and probe stages, so we need to be able to distinguish
    // between the metrics from the build stage and those from the probe stage.
    key_dict_builder_metrics.subtract_metrics(
        this->metrics.key_dict_builder_metrics_build_stage_snapshot);
    metrics_out.emplace_back(
        StatMetric("n_key_dict_builders", n_key_dict_builders, true));
    key_dict_builder_metrics.add_to_metrics(metrics_out, "key_dict_builders_");
    metrics_out.emplace_back(
        StatMetric("n_non_key_dict_builders", n_non_key_dict_builders, true));
    non_key_dict_builder_metrics.add_to_metrics(metrics_out,
                                                "non_key_dict_builders_");

    JoinState::ReportProbeStageMetrics(metrics_out);
}

void HashJoinState::FinalizeProbe() {
    // Free the probe shuffle buffer's memory:
    this->probe_shuffle_state.Finalize();

    // Give all the budget back in case other operators in this pipeline can
    // benefit from it.
    // XXX We can change it to an assert once they're enabled.
    if (this->op_pool->bytes_pinned() != 0 ||
        this->op_pool->bytes_allocated() != 0) {
        throw std::runtime_error(
            "HashJoinState::FinalizeProbe: Number of pinned bytes (" +
            std::to_string(this->op_pool->bytes_pinned()) +
            ") or allocated bytes (" +
            std::to_string(this->op_pool->bytes_allocated()) + ") is not 0!");
    }
    this->op_pool->SetErrorThreshold(1.0);
    this->op_pool->SetBudget(0);

    JoinState::FinalizeProbe();
}

void HashJoinState::InitProbeInputBuffers() {
    // We only need to initialize the probe input buffers of non-0th
    // partitions. The 0th partition doesn't buffer the input since
    // we produce output directly.
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->InitProbeInputBuffer();
    }
}

void HashJoinState::AppendProbeBatchToInactivePartition(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows, const int64_t in_table_start_offset,
    int64_t table_nrows) {
    assert(in_table_start_offset >= 0);
    time_pt start_part_check = start_timer();
    std::vector<std::vector<bool>> append_rows_by_partition;
    append_rows_by_partition.resize(this->partitions.size());
    if (table_nrows == -1) {
        // Convert default of -1 to all rows in the table (starting from
        // 'in_table_start_offset').
        table_nrows = in_table->nrows() - in_table_start_offset;
    }
    assert(table_nrows >= 0);
    assert(static_cast<int64_t>(append_rows.size()) == table_nrows);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] =
            std::vector<bool>(table_nrows, false);
    }

    for (size_t i_row = 0; i_row < static_cast<size_t>(table_nrows); i_row++) {
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
    this->metrics.probe_inactive_partition_check_time +=
        end_timer(start_part_check);
    time_pt start_append = start_timer();
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendInactiveProbeBatch(
            in_table, join_hashes, append_rows_by_partition[i_part],
            in_table_start_offset, table_nrows);
    }
    this->metrics.append_probe_inactive_partitions_time +=
        end_timer(start_append);
    this->metrics.append_probe_inactive_partitions_nrows += table_nrows;
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool is_anti_join>
void HashJoinState::FinalizeProbeForInactivePartitions(
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    time_pt start_finalize = start_timer();
    time_pt start_pin;
    for (size_t i = 1; i < this->partitions.size(); i++) {
        if (this->debug_partitioning) {
            std::cerr
                << "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: "
                   "Starting probe finalization for partition "
                << i << "." << std::endl;
        }
        // Multiple partitions should only exist when the build table is
        // distributed.
        assert(this->build_parallel);
        // Pin the partition
        start_pin = start_timer();
        this->partitions[i]->pin();
        this->metrics.finalize_probe_inactive_partitions_pin_partition_time +=
            end_timer(start_pin);
        this->partitions[i]
            ->FinalizeProbeForInactivePartition<
                build_table_outer, probe_table_outer, non_equi_condition,
                is_anti_join>(this->cond_func, build_kept_cols, probe_kept_cols,
                              this->output_buffer);
        // Free the partition
        this->partitions[i].reset();
        if (this->debug_partitioning) {
            std::cerr
                << "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: "
                   "Finalized probe for partition "
                << i << "." << std::endl;
        }
    }
    this->metrics.finalize_probe_inactive_partitions_time +=
        end_timer(start_finalize);
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

uint64_t HashJoinState::op_pool_budget_bytes() const {
    return this->op_pool->get_operator_budget_bytes();
}

uint64_t HashJoinState::op_pool_bytes_pinned() const {
    return this->op_pool->bytes_pinned();
}

uint64_t HashJoinState::op_pool_bytes_allocated() const {
    return this->op_pool->bytes_allocated();
}

/**
 * @brief Filter NA values from input of build/probe consume batch and write to
 * output buffer in case of outer join.
 *
 * @tparam table_outer build/probe table is on outer side of join
 * @tparam is_probe input is probe input (helps write to output buffer properly)
 * @param in_table input batch to build/probe
 * @param n_keys number of key columns in input
 * @param table_parallel target (build/probe) table is parallel
 * @param other_table_parallel The other non-target table (build/probe) is
 * parallel
 * @param na_counter counter of build/probe NAs so far (helps write output in
 * replicated case)
 * @param na_out_buffer output buffer for writing NA rows
 * @param build_table build table in case of probe input since necessary for
 * output generation (only used in probe case)
 * @param build_kept_cols Which columns to generate in the output on the
 * build side (only used in probe case)
 * @param probe_kept_cols Which columns to generate in the output on the
 * probe side (only used in probe case)
 * @return std::shared_ptr<table_info> input table with NAs filtered out
 */
template <bool table_outer, bool is_probe>
std::shared_ptr<table_info> filter_na_values(
    std::shared_ptr<table_info> in_table, uint64_t n_keys, bool table_parallel,
    bool other_table_parallel, size_t& na_counter,
    ChunkedTableBuilder& na_out_buffer, std::shared_ptr<table_info> build_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols) {
    bodo::vector<bool> not_na(in_table->nrows(), true);
    bool can_have_na = false;
    for (uint64_t i = 0; i < n_keys; i++) {
        // Determine which columns can contain NA/contain NA
        const std::shared_ptr<array_info>& col = in_table->columns[i];
        if (col->can_contain_na()) {
            can_have_na = true;
            bodo::vector<bool> col_not_na = col->get_notna_vector();
            // Do an elementwise logical and to update not_na
            std::transform(not_na.begin(), not_na.end(),  // NOLINT
                           col_not_na.begin(), not_na.begin(),
                           std::logical_and<>());
        }
    }
    if (!can_have_na) {
        // No NA values, just return.
        return in_table;
    }

    // Retrieve table takes a list of columns. Convert the boolean array.
    bodo::vector<int64_t> idx_list;
    // For appending NAs in outer join (build case).
    std::vector<bool> append_nas;
    if constexpr (!is_probe) {
        append_nas.resize(in_table->nrows(), false);
    }

    // For appending NAs in outer join (probe case).
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    // If we have a replicated build table without a replicated probe table (or
    // vice versa). then we only add a fraction of NA rows to the output.
    // Otherwise we add all rows.
    const bool add_all = table_parallel || !other_table_parallel;
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for (size_t i = 0; i < in_table->nrows(); i++) {
        if (not_na[i]) {
            idx_list.emplace_back(i);
        } else if (table_outer) {
            // If table is replicated but output is not
            // replicated evenly divide NA values across all ranks.
            bool append_na = add_all || ((na_counter % n_pes) ==
                                         static_cast<size_t>(myrank));
            if (is_probe) {
                if (append_na) {
                    build_idxs.push_back(-1);
                    probe_idxs.push_back(i);
                }
            } else {
                append_nas[i] = append_na;
            }
            na_counter++;
        }
    }
    if (idx_list.size() == in_table->nrows()) {
        // No NA values, skip the copy.
        return in_table;
    } else {
        if constexpr (table_outer) {
            // If have an outer join we must push the NA values directly to
            // the output, not just filter them.
            if constexpr (is_probe) {
                na_out_buffer.AppendJoinOutput(
                    build_table, in_table, build_idxs, probe_idxs,
                    build_kept_cols, probe_kept_cols);
            } else {
                na_out_buffer.AppendBatch(in_table, append_nas);
            }
        }
        return RetrieveTable(std::move(in_table), std::move(idx_list));
    }
}

void HashJoinState::DisablePartitioning() {
    this->op_pool->DisableThresholdEnforcement();
}

#pragma endregion  // HashJoinState
/* ------------------------------------------------------------------------ */

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#define CHECK_ARROW(expr, msg)                                             \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("Error in join build: ") + msg + \
                              " " + expr.ToString();                       \
        throw std::runtime_error(err_msg);                                 \
    }

/**
 * Helper for UpdateKeysMinMax on numeric columns with supported types.
 *
 * @param[in] min_max_arr: the array storing the accumulated min/max values
 * (min in row 0, max in row 1) for the current column.
 * @param[in] in_arr: the column of numeric data that is being inserted
 * to update the min/max values stored in min_max_arr.
 */
template <Bodo_CTypes::CTypeEnum DType, typename ScalarType>
void update_build_min_max_state_numeric(
    std::shared_ptr<array_info>& min_max_arr,
    const std::shared_ptr<array_info>& in_arr) {
    using T = typename dtype_to_type<DType>::type;

    // Convert the bodo array to an arrow array
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    auto arrow_arr = bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), in_arr,
                                         false, "", time_unit, false,
                                         bodo::default_buffer_memory_manager());

    // Use the vectorized arrow kernel to compute the minimum & maximum,
    // then extract the corresponding arrow scalars.
    auto minMaxRes = arrow::compute::MinMax(arrow_arr);
    CHECK_ARROW(minMaxRes.status(), "Error in computing min/max");
    std::shared_ptr<arrow::Scalar> minMax =
        std::move(minMaxRes.ValueOrDie()).scalar();
    std::shared_ptr<arrow::StructScalar> minMaxStruct =
        std::static_pointer_cast<arrow::StructScalar>(minMax);
    std::shared_ptr<arrow::Scalar> min =
        minMaxStruct->field(arrow::FieldRef("min")).ValueOrDie();
    std::shared_ptr<arrow::Scalar> max =
        minMaxStruct->field(arrow::FieldRef("max")).ValueOrDie();

    // Skip if the scalars are nulls, since there is no relevant
    // data to insert.
    if (min->is_valid && max->is_valid) {
        // Extract the min scalar and potentially update the value
        // stored in row #0 of the corresponding array.
        auto min_scalar = static_pointer_cast<ScalarType>(min);
        T min_value = min_scalar->value;
        T existing_min_value = getv<T>(min_max_arr, 0);
        if (!min_max_arr->get_null_bit(0) || min_value < existing_min_value) {
            getv<T>(min_max_arr, 0) = min_value;
            min_max_arr->set_null_bit(0, true);
        }
        // Extract the max scalar and potentially update the value
        // stored in row #1 of the corresponding array.
        auto max_scalar = static_pointer_cast<ScalarType>(max);
        T max_value = max_scalar->value;
        T existing_max_value = getv<T>(min_max_arr, 1);
        if (!min_max_arr->get_null_bit(1) || max_value > existing_max_value) {
            getv<T>(min_max_arr, 1) = max_value;
            min_max_arr->set_null_bit(1, true);
        }
    }
}

void JoinState::UpdateKeysMinMax(const std::shared_ptr<array_info>& arr,
                                 size_t col_idx) {
#define MIN_MAX_NUMERIC_DTYPE_CASE(DType, ScalarType)          \
    case DType: {                                              \
        update_build_min_max_state_numeric<DType, ScalarType>( \
            min_max_values[col_idx].value(), arr);             \
        break;                                                 \
    }
    if (min_max_values[col_idx].has_value()) {
        switch (arr->arr_type) {
            case bodo_array_type::NUMPY:
            case bodo_array_type::NULLABLE_INT_BOOL: {
                switch (arr->dtype) {
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::INT8,
                                               arrow::Int8Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::UINT8,
                                               arrow::UInt8Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::INT16,
                                               arrow::Int16Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::UINT16,
                                               arrow::UInt16Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::INT32,
                                               arrow::Int32Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::UINT32,
                                               arrow::UInt32Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::INT64,
                                               arrow::Int64Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::UINT64,
                                               arrow::UInt64Scalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::FLOAT32,
                                               arrow::FloatScalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::FLOAT64,
                                               arrow::DoubleScalar);
                    MIN_MAX_NUMERIC_DTYPE_CASE(
                        Bodo_CTypes::DATE,
                        arrow::DateScalar<arrow::Date32Type>);
                    MIN_MAX_NUMERIC_DTYPE_CASE(Bodo_CTypes::DATETIME,
                                               arrow::TimestampScalar);
                    default: {
                        throw std::runtime_error(
                            "Unsupported dtype for min/max runtime join "
                            "filter: " +
                            GetDtype_as_string(arr->dtype));
                        break;
                    }
                }
                break;
            }
            case bodo_array_type::STRING: {
                // This path is currently only ever invoked during the
                // finalization step of Min/Max on dictionary-encoded strings
                UpdateKeysMinMaxString(arr, col_idx);
                break;
            }
            case bodo_array_type::DICT: {
                UpdateKeysMinMaxDict(arr, col_idx);
                break;
            }
            default: {
                throw std::runtime_error(
                    "Unsupported array type for min/max runtime join "
                    "filter: " +
                    GetArrType_as_string(arr->arr_type));
                break;
            }
        }
    }
#undef MIN_MAX_NUMERIC_DTYPE_CASE
}

void JoinState::UpdateKeysMinMaxString(
    const std::shared_ptr<array_info>& in_arr, size_t col_idx) {
    assert(in_arr->arr_type == bodo_array_type::STRING);
    // Convert the bodo array to an arrow array
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    auto arrow_arr = bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), in_arr,
                                         false, "", time_unit, false,
                                         bodo::default_buffer_memory_manager());

    // Use the vectorized arrow kernel to compute the minimum & maximum,
    // then extract the corresponding arrow scalars.
    auto minMaxRes = arrow::compute::MinMax(arrow_arr);
    CHECK_ARROW(minMaxRes.status(), "Error in computing min/max");
    std::shared_ptr<arrow::Scalar> minMax =
        std::move(minMaxRes.ValueOrDie()).scalar();
    std::shared_ptr<arrow::StructScalar> minMaxStruct =
        std::static_pointer_cast<arrow::StructScalar>(minMax);
    std::shared_ptr<arrow::Scalar> min =
        minMaxStruct->field(arrow::FieldRef("min")).ValueOrDie();
    std::shared_ptr<arrow::Scalar> max =
        minMaxStruct->field(arrow::FieldRef("max")).ValueOrDie();

    // Skip if the scalars are nulls, since there is no relevant
    // data to insert.
    if (min->is_valid && max->is_valid) {
        std::string min_str_res;
        std::string max_str_res;

        // Extract the min and max scalars from the new batch as string views
        auto min_string_scalar =
            std::static_pointer_cast<arrow::StringScalar>(min);
        std::string_view min_str(
            reinterpret_cast<const char*>(min_string_scalar->value->data()),
            min_string_scalar->value->size());
        auto max_string_scalar =
            std::static_pointer_cast<arrow::StringScalar>(max);
        std::string_view max_str(
            reinterpret_cast<const char*>(max_string_scalar->value->data()),
            max_string_scalar->value->size());

        std::shared_ptr<array_info>& min_max_arr =
            min_max_values[col_idx].value();
        char* data = min_max_arr->data1<bodo_array_type::STRING, char>();
        offset_t* offsets =
            min_max_arr->data2<bodo_array_type::STRING, offset_t>();
        offset_t mid = offsets[1];
        offset_t end = offsets[2];

        // Extract the min string from the existing data
        // and use it to calculate the overall min.
        if (min_max_arr->get_null_bit<bodo_array_type::STRING>(0)) {
            std::string min_str_existing(data, mid);
            min_str_res =
                (min_str < min_str_existing) ? min_str : min_str_existing;
        } else {
            min_str_res = min_str;
        }

        // Extract the max string from the existing data
        // and use it to calculate the overall max.
        if (min_max_arr->get_null_bit<bodo_array_type::STRING>(1)) {
            std::string max_str_existing(data + mid, end - mid);
            max_str_res =
                (max_str > max_str_existing) ? max_str : max_str_existing;
        } else {
            max_str_res = max_str;
        }

        // Build a new string array from the overall min/max. The null
        // bitmap vector is set to all-1 since all of the rows must
        // now be non-null.
        size_t min_size = min_str_res.size();
        size_t max_size = max_str_res.size();
        size_t total_chars = min_size + max_size;
        min_max_values[col_idx] =
            alloc_string_array(Bodo_CTypes::STRING, 2, total_chars);
        char* final_data = min_max_values[col_idx]
                               .value()
                               ->data1<bodo_array_type::STRING, char>();
        offset_t* final_offsets =
            min_max_values[col_idx]
                .value()
                ->data2<bodo_array_type::STRING, offset_t>();
        memcpy(final_data, min_str_res.data(), min_size);
        memcpy(final_data + min_size, max_str_res.data(), max_size);
        final_offsets[0] = 0;
        final_offsets[1] = min_size;
        final_offsets[2] = total_chars;
    }
}

void JoinState::UpdateKeysMinMaxDict(const std::shared_ptr<array_info>& in_arr,
                                     size_t col_idx) {
    assert(in_arr->arr_type == bodo_array_type::DICT);
    assert(build_table_dict_builders[col_idx] != nullptr);
    if (!is_matching_dictionary(
            in_arr->child_arrays[0],
            build_table_dict_builders[col_idx]->dict_buff->data_array)) {
        throw std::runtime_error(
            "JoinState::UpdateKeysMinMaxDict: Input dictionary not "
            "unified!");
    }
    std::shared_ptr<array_info>& dictionary = in_arr->child_arrays[0];
    size_t n_strings = dictionary->length;
    if (build_dict_hit_bitmap[col_idx].size() < n_strings) {
        build_dict_hit_bitmap[col_idx].resize(n_strings, false);
    }
    std::shared_ptr<array_info>& indices = in_arr->child_arrays[1];
    dict_indices_t* index_buffer =
        indices->data1<bodo_array_type::NULLABLE_INT_BOOL, dict_indices_t>();
    size_t n_rows = indices->length;
    for (size_t row = 0; row < n_rows; row++) {
        if (indices->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row)) {
            dict_indices_t index = index_buffer[row];
            build_dict_hit_bitmap[col_idx][index] = true;
        }
    }
}

// Does the parallel finalization of the min/max values for each key column to
// ensure that ever rank has the global min/max of the key columns before any
// runtime join filters.
void JoinState::FinalizeKeysMinMax() {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Gather the min/max arrays onto every rank
    std::shared_ptr<table_info> dummy_table = std::make_shared<table_info>();
    for (size_t col_idx = 0; col_idx < min_max_values.size(); col_idx++) {
        if (min_max_values[col_idx].has_value()) {
            if (this->build_table_schema->column_types[col_idx]->array_type ==
                bodo_array_type::DICT) {
                // If we are computing the result on a dictionary encoded array,
                // we need to use the bitmasks and dictionary builders to
                // materialize the strings that were encountered at least once
                // during build time.
                std::shared_ptr<array_info>& dict_values =
                    build_table_dict_builders[col_idx]->dict_buff->data_array;
                std::vector<int64_t> indices_to_keep;
                for (size_t i = 0; i < build_dict_hit_bitmap[col_idx].size();
                     i++) {
                    if (build_dict_hit_bitmap[col_idx][i]) {
                        indices_to_keep.push_back((int64_t)i);
                    }
                }
                std::shared_ptr<array_info> dictionary_subset =
                    RetrieveArray_SingleColumn(dict_values, indices_to_keep);
                // Insert the filtered string data as if it were a regular
                // min/max batch.
                UpdateKeysMinMax(dictionary_subset, col_idx);
            }
            std::shared_ptr<array_info> existing_values =
                min_max_values[col_idx].value();
            // If this column is one of the ones with min max values stored
            // in it, do an allgather so every rank has all the min/max
            // values.
            std::shared_ptr<array_info> combined_arr = gather_array(
                existing_values, true, build_parallel, 0, n_pes, myrank);

            // Insert the combined data as if it were a regular batch update,
            // ensuring every rank has the same min/max values.
            UpdateKeysMinMax(combined_arr, col_idx);
        }
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
void update_keys_unique_values_numeric_helper(
    HashJoinState* join_state, const std::shared_ptr<array_info>& arr,
    size_t col_idx) {
    using T = typename dtype_to_type<DType>::type;
    using MT = typename type_to_max_type<T>::type;
    if (join_state->unique_values[col_idx].has_value() && arr->length > 0) {
        std::unordered_set<int64_t>& unique_set =
            join_state->unique_values[col_idx].value();
        T* data_arr = arr->data1<ArrType, T>();
        for (size_t row = 0; row < arr->length; row++) {
            if (non_null_at<ArrType, T, DType>(*arr, row)) {
                MT upcasted_value = static_cast<MT>(data_arr[row]);
                int64_t as_int64 =
                    bit_preserving_cast<MT, int64_t>(upcasted_value);
                unique_set.insert(as_int64);
            }
        }
        if (unique_set.size() > join_state->unique_values_limit) {
            join_state->unique_values[col_idx] = std::nullopt;
        }
    }
}

void HashJoinState::UpdateKeysUniqueValues(
    const std::shared_ptr<array_info>& arr, size_t col_idx) {
#define UPDATE_NUMERIC_UNIQUE_KEYS(ArrType, DType)                          \
    case DType: {                                                           \
        update_keys_unique_values_numeric_helper<ArrType, DType>(this, arr, \
                                                                 col_idx);  \
        break;                                                              \
    }
    switch (arr->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            switch (arr->dtype) {
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::INT8);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::INT16);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::INT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::INT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::UINT8);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::UINT16);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::UINT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::UINT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::FLOAT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::FLOAT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::DATE);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                           Bodo_CTypes::TIME);
                default: {
                    break;
                }
            }
            break;
        }
        case bodo_array_type::NUMPY: {
            switch (arr->dtype) {
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::INT8);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::INT16);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::INT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::INT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::UINT8);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::UINT16);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::UINT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::UINT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::FLOAT32);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::FLOAT64);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::DATE);
                UPDATE_NUMERIC_UNIQUE_KEYS(bodo_array_type::NUMPY,
                                           Bodo_CTypes::TIME);
                default: {
                    break;
                }
            }
            break;
        }
        default: {
            break;
        }
    }
#undef UPDATE_NUMERIC_UNIQUE_KEYS
}

void HashJoinState::FinalizeKeysUniqueValues() {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::shared_ptr<table_info> dummy_table = std::make_shared<table_info>();
    for (size_t col_idx = 0; col_idx < unique_values.size(); col_idx++) {
        bool local_has_unique = unique_values[col_idx].has_value();
        bool global_has_unique;
        CHECK_MPI(MPI_Allreduce(&local_has_unique, &global_has_unique, 1,
                                MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD),
                  "HashJoinState::FinalizeKeysUniqueValues: MPI error "
                  "on MPI_Allreduce:");
        if (global_has_unique) {
            std::unordered_set<int64_t>& existing_values =
                unique_values[col_idx].value();
            std::shared_ptr<array_info> existing_array =
                alloc_nullable_array_no_nulls(existing_values.size(),
                                              Bodo_CTypes::INT64);
            int64_t* data =
                existing_array
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, int64_t>();
            size_t row = 0;
            for (int64_t elem : existing_values) {
                data[row] = elem;
                row++;
            }
            // If this column is one of the ones with min max values stored
            // in it, do an allgather so every rank has all the unique values
            // values.
            std::shared_ptr<array_info> combined_arr = gather_array(
                existing_array, true, build_parallel, 0, n_pes, myrank);

            // Insert the combined data as if it were a regular batch update,
            // ensuring every rank has the same unique values.
            UpdateKeysUniqueValues(combined_arr, col_idx);
        } else {
            unique_values[col_idx] = std::nullopt;
        }
    }
}

/**
 * @brief Get global is_last flag given local is_last using ibarrier
 *
 * @param local_is_last Whether we're done on this rank.
 * @param[in] join_state Join state used to get the distributed information
 * and the sync_iter.
 * @return true We don't need to have any more iterations on this rank.
 * @return false We may need to have more iterations on this rank.
 */
bool stream_join_sync_is_last(bool local_is_last, JoinState* join_state) {
    if (join_state->global_is_last) {
        return true;
    }

    // We must synchronize if either we have a distributed build or an
    // LEFT/FULL OUTER JOIN where probe is distributed.
    if ((join_state->build_parallel ||
         (join_state->build_table_outer && join_state->probe_parallel)) &&
        local_is_last) {
        if (!join_state->is_last_barrier_started) {
            CHECK_MPI(MPI_Ibarrier(join_state->shuffle_comm,
                                   &join_state->is_last_request),
                      "stream_join_sync_is_last: MPI error on MPI_Ibarrier:");
            join_state->is_last_barrier_started = true;
            return false;
        } else {
            int flag = 0;
            CHECK_MPI(MPI_Test(&join_state->is_last_request, &flag,
                               MPI_STATUS_IGNORE),
                      "stream_join_sync_is_last: MPI error on MPI_Test:");
            if (flag) {
                join_state->global_is_last = true;
            }
            return flag;
        }
    } else {
        // If we have a broadcast join or a replicated input we don't need to be
        // synchronized because there is no shuffle.
        return local_is_last;
    }
}

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
                              bool use_bloom_filter, bool local_is_last) {
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
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // TODO: remove
    if (join_state->build_iter == 0) {
        join_state->build_shuffle_state.Initialize(
            in_table, join_state->build_parallel, join_state->shuffle_comm);
    }

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

    // Prune any rows with NA keys (if matching SQL behavior).
    // If this is an build_table_outer = False,
    // then we can prune these rows from the table entirely. If
    // build_table_outer = True then we can skip adding these rows to the
    // hash table (as they can't match), but must write them to the Join
    // output.
    // TODO: Have outer join skip the build table/avoid shuffling.
    if (!join_state->is_na_equal) {
        time_pt start_filter = start_timer();
        if (join_state->build_table_outer) {
            in_table = filter_na_values<true, false>(
                std::move(in_table), join_state->n_keys,
                join_state->build_parallel, join_state->probe_parallel,
                join_state->build_na_counter, join_state->build_na_key_buffer,
                nullptr, std::vector<uint64_t>(), std::vector<uint64_t>());
        } else {
            in_table = filter_na_values<false, false>(
                std::move(in_table), join_state->n_keys,
                join_state->build_parallel, join_state->probe_parallel,
                join_state->build_na_counter, join_state->build_na_key_buffer,
                nullptr, std::vector<uint64_t>(), std::vector<uint64_t>());
        }
        join_state->metrics.build_filter_na_time += end_timer(start_filter);
        join_state->metrics.build_filter_na_output_nrows += in_table->nrows();
    }

    if (!join_state->probe_table_outer) {
        // If this is not an outer probe, use the latest batch
        // to process the min/max values for each key column and
        // update the min_max_values vector.
        time_pt start_min_max = start_timer();
        for (size_t col_idx = 0; col_idx < join_state->n_keys; col_idx++) {
            join_state->UpdateKeysMinMax(in_table->columns[col_idx], col_idx);
        }
        join_state->metrics.build_min_max_update_time +=
            end_timer(start_min_max);
        // And do the same for the unique values.
        time_pt start_unique_timer = start_timer();
        for (size_t col_idx = 0; col_idx < join_state->n_keys; col_idx++) {
            join_state->UpdateKeysUniqueValues(in_table->columns[col_idx],
                                               col_idx);
        }
        join_state->metrics.build_unique_values_update_time +=
            end_timer(start_unique_timer);
    }

    // Get hashes of the new batch (different hashes for partitioning and
    // hash table to reduce conflict)
    // NOTE: Partition hashes need to be consistent across ranks so need to use
    // dictionary hashes. Since we are using dictionary hashes, we don't need
    // dictionaries to be global. In fact, hash_keys_table will ignore the
    // dictionaries entirely when dict_hashes are provided.
    time_pt start_part_hash = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        join_state->build_parallel, false, dict_hashes);
    join_state->metrics.build_input_part_hashing_time +=
        end_timer(start_part_hash);

    // Add to the bloom filter.
    if (use_bloom_filter) {
        time_pt start_bloom = start_timer();
        join_state->global_bloom_filter->AddAll(batch_hashes_partition, 0,
                                                in_table->nrows());
        join_state->metrics.bloom_filter_add_time += end_timer(start_bloom);
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

    if (join_state->build_parallel) {
        // 'append_row_to_build_table' is all 'true' when build side is
        // replicated, so there's no point in appending to the shuffle buffer.
        append_row_to_build_table.flip();
        std::vector<bool>& append_row_to_shuffle_table =
            append_row_to_build_table;
        join_state->build_shuffle_state.AppendBatch(
            in_table, append_row_to_shuffle_table);
    }

    batch_hashes_partition.reset();
    in_table.reset();

    if (join_state->build_parallel) {
        std::optional<std::shared_ptr<table_info>> new_data_ =
            join_state->build_shuffle_state.ShuffleIfRequired(local_is_last);
        if (new_data_.has_value()) {
            std::shared_ptr<table_info> new_data = new_data_.value();
            // NOTE: Partition hashes need to be consistent across ranks, so
            // need to use dictionary hashes. Since we are using dictionary
            // hashes, we don't need dictionaries to be global. In fact,
            // hash_keys_table will ignore the dictionaries entirely when
            // dict_hashes are provided.
            dict_hashes = join_state->GetDictionaryHashesForKeys();
            start_part_hash = start_timer();
            std::shared_ptr<uint32_t[]> batch_hashes_partition =
                hash_keys_table(new_data, join_state->n_keys,
                                SEED_HASH_PARTITION, /*is_parallel*/ true,
                                /*global_dict_needed*/ false, dict_hashes);
            join_state->metrics.build_input_part_hashing_time +=
                end_timer(start_part_hash);
            // Add new batch of data to partitions (bulk insert)
            join_state->AppendBuildBatch(new_data, batch_hashes_partition);
            batch_hashes_partition.reset();
        }
    }

    // Make is_last global
    bool is_last = stream_join_sync_is_last(
        local_is_last && join_state->build_shuffle_state.SendRecvEmpty(),
        join_state);

    // If this is not an outer probe, finalize the min/max
    // values for each key by shuffling across multiple ranks.
    // This is done before the broadcast handling since
    // the accumulated min/max state may be parallel but
    // broadcast handling may change the join_state->build_parallel flag.
    if (is_last && !join_state->probe_table_outer) {
        // If this is not an outer probe, finalize the min/max
        // values for each key by shuffling across multiple ranks.
        // This is done before the broadcast handling since
        // the finalization deals with the parallel handling
        // of the accumulated min/max state.
        time_pt start_min_max = start_timer();
        join_state->FinalizeKeysMinMax();
        join_state->metrics.build_min_max_finalize_time +=
            end_timer(start_min_max);
        // Do the same for the unique values
        time_pt start_unique_values = start_timer();
        join_state->FinalizeKeysUniqueValues();
        join_state->metrics.build_unique_values_finalize_time +=
            end_timer(start_unique_values);
    }

    // If the build table is small enough, broadcast it to all ranks
    // so the probe table can be joined locally.
    // NOTE: broadcasting build table is incorrect if the probe table is
    // replicated.
    // TODO: Simplify this logic into helper functions
    // and/or move to FinalizeBuild?
    if (is_last && join_state->build_parallel && join_state->probe_parallel) {
        // Only consider a broadcast join if we have a single partition
        bool single_partition = join_state->partitions.size() == 1;
        CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &single_partition, 1, MPI_C_BOOL,
                                MPI_LAND, MPI_COMM_WORLD),
                  "join_build_consume_batch: MPI error on MPI_Allreduce:");
        if (single_partition) {
            int64_t global_table_size = table_global_memory_size(
                join_state->partitions[0]->build_table_buffer->data_table);
            global_table_size += table_global_memory_size(
                join_state->build_shuffle_state.table_buffer->data_table);
            if (join_state->force_broadcast ||
                global_table_size < get_bcast_join_threshold()) {
                if (join_state->debug_partitioning && myrank == 0) {
                    std::cerr
                        << fmt::format(
                               "[DEBUG] HashJoin (Op ID: {}): Converting to a "
                               "broadcast hash join since the global table "
                               "size ({}) is lower than the threshold",
                               join_state->op_id,
                               BytesToHumanReadableString(global_table_size))
                        << std::endl;
                }
                time_pt start_bcast = start_timer();
                // Mark the build side as replicated.
                join_state->build_parallel = false;
                // Now that we'll have a single partition, disable
                // partitioning altogether. This essentially
                // disables threshold enforcement during any
                // AppendBuildTable or FinalizeBuild calls.
                join_state->DisablePartitioning();

                // We have decided to do a broadcast join. To do this we
                // will execute the following steps:
                // 1. Broadcast the table across all ranks with allgatherv.
                // 2. Clear the existing JoinPartition state. This is
                // necessary because the allgatherv includes rows that we
                // have already processed and we need to avoid processing
                // them twice.
                // 3. Insert the entire table into the new partition.
                join_state->metrics.is_build_bcast_join = 1;

                // Step 1: Broadcast the table.

                // Gather the partition data.
                std::shared_ptr<table_info> gathered_table = gather_table(
                    join_state->partitions[0]->build_table_buffer->data_table,
                    -1, /*all_gather*/ true, true);

                gathered_table =
                    join_state->UnifyBuildTableDictionaryArrays(gathered_table);

                // Step 2: Clear the existing JoinPartition state
                join_state->ResetPartitions();

                // Step 3: Insert the broadcast table.

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
                join_state->metrics.build_bcast_time += end_timer(start_bcast);
            }
        }
    }

    // Finalize build on all partitions if it's the last input batch.
    if (is_last) {
        if (use_bloom_filter && join_state->build_parallel) {
            // Make the bloom filter global.
            time_pt start_bloom = start_timer();
            join_state->global_bloom_filter->union_reduction();
            join_state->metrics.bloom_filter_union_reduction_time +=
                end_timer(start_bloom);
        }

        if (join_state->build_parallel && !join_state->probe_table_outer) {
            // Ensure key dict builders have all global dictionary values
            // so we can filter the probe input based on the dictionaries
            // We don't need to worry about nested dict builders
            // since they aren't used for filtering
            for (size_t i = 0; i < join_state->n_keys; ++i) {
                std::shared_ptr<DictionaryBuilder>& key_dict_builder =
                    join_state->key_dict_builders[i];
                if (join_state->build_table_schema->column_types[i]
                        ->array_type != bodo_array_type::DICT) {
                    continue;
                }
                std::shared_ptr<array_info> empty_dict_array =
                    create_dict_string_array(
                        key_dict_builder->dict_buff->data_array,
                        alloc_nullable_array(0, Bodo_CTypes::INT32));
                make_dictionary_global_and_unique(empty_dict_array, true);
                key_dict_builder->UnifyDictionaryArray(empty_dict_array);
            }
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
          bool non_equi_condition, bool use_bloom_filter, bool is_anti_join>
bool join_probe_consume_batch(HashJoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              const std::vector<uint64_t> build_kept_cols,
                              const std::vector<uint64_t> probe_kept_cols,
                              bool local_is_last) {
    assert(join_state->build_input_finalized);
    if (join_state->probe_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "join_probe_consume_batch: Received non-empty in_table "
                "after the probe was already finalized!");
        }
        // No processing left.
        // When probe is finalized global is_last has been seen so no need
        // for additional synchronization
        return true;
    }
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (local_is_last && join_state->local_is_last_debug_counter == 0) {
        std::cout << fmt::format(
                         "[DEBUG]:{} HashJoin (Op ID: {}): Rank {} received "
                         "local_is_last = true on probe iteration {}",
                         __LINE__, join_state->op_id, myrank,
                         join_state->probe_iter)
                  << std::endl;
        join_state->local_is_last_debug_counter = 1;
    }

    if (join_state->probe_iter == 0) {
        join_state->is_last_request = MPI_REQUEST_NULL;
        join_state->is_last_barrier_started = false;
        join_state->global_is_last = false;
        join_state->probe_shuffle_state.Initialize(
            in_table, join_state->probe_parallel, join_state->shuffle_comm);
        // Initialize the probe_table_buffer_chunked of partitions
        // 1 and onwards.
        join_state->InitProbeInputBuffers();
    }

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
    // In the probe inner case, we don't need to expand the dictionaries of the
    // key columns since if a value doesn't already exist in them, it cannot
    // ever match anyway. Setting only_transpose_existing_on_key_cols = true
    // will only transpose the values that already exist in the dict-builder and
    // set the rest as null. Due to the JoinFilter(s), these values should've
    // already gotten filtered out, so we don't need to call filter_na_values to
    // filter them out. 'only_transpose_existing_on_key_cols' just ensures that
    // the dict-builder isn't needlessly expanded. NOTE: The case where the
    // input type is STRING but the key type is DICT (and therefore the
    // column-level JoinFilter would've been a NOP) cannot happen since the
    // compiler would've converted the key to a simple STRING type in that case.
    in_table = join_state->UnifyProbeTableDictionaryArrays(
        in_table,
        /*only_transpose_existing_on_key_cols*/ !probe_table_outer);

    // Prune rows with NA keys (necessary to avoid NA imbalance after shuffle).
    // In case of probe_table_outer = True, write NAs directly to output.
    // We only need to do this in the probe_table_outer case since in the inner
    // case, the planner will automatically generate IS NOT NULL filters
    // and push them down as much as possible. It won't do so in the outer case
    // since the rows need to be preserved and we need to handle them here.
    if constexpr (probe_table_outer) {
        time_pt start_filter = start_timer();
        in_table = filter_na_values<probe_table_outer, true>(
            std::move(in_table), join_state->n_keys, join_state->probe_parallel,
            join_state->build_parallel, join_state->probe_na_counter,
            *(join_state->output_buffer),
            active_partition->build_table_buffer->data_table, build_kept_cols,
            probe_kept_cols);
        join_state->metrics.probe_filter_na_time += end_timer(start_filter);
    }
    join_state->metrics.probe_filter_na_output_nrows += in_table->nrows();

    active_partition->probe_table = in_table;

    // Determine if a shuffle could be required.
    const bool shuffle_possible =
        join_state->build_parallel && join_state->probe_parallel;

    // Compute join hashes
    time_pt start_join_hash = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
        in_table, join_state->n_keys, SEED_HASH_JOIN, shuffle_possible, false);
    join_state->metrics.probe_join_hashing_time += end_timer(start_join_hash);
    active_partition->probe_table_hashes = batch_hashes_join.get();

    // Compute partitioning hashes:
    // NOTE: partition hashes need to be consistent across ranks so need to
    // use dictionary hashes
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = join_state->GetDictionaryHashesForKeys();
    time_pt start_part_hash = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        shuffle_possible, true, dict_hashes);
    join_state->metrics.probe_part_hashing_time += end_timer(start_part_hash);

    bodo::vector<int64_t> group_ids;
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    // Fetch the raw array pointers from the arrays for passing
    // to the non-equijoin condition
    std::vector<array_info*> build_table_info_ptrs, probe_table_info_ptrs;
    // Vectors for data
    std::vector<void*> build_col_ptrs, probe_col_ptrs;
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
    if constexpr (non_equi_condition) {
        std::tie(build_table_info_ptrs, build_col_ptrs, build_null_bitmaps) =
            get_gen_cond_data_ptrs(
                active_partition->build_table_buffer->data_table);
        std::tie(probe_table_info_ptrs, probe_col_ptrs, probe_null_bitmaps) =
            get_gen_cond_data_ptrs(active_partition->probe_table);
    }

    // probe hash table
    std::vector<bool> append_to_probe_shuffle_buffer(in_table->nrows(), false);
    std::vector<bool> append_to_probe_inactive_partition(in_table->nrows(),
                                                         false);
    std::shared_ptr<std::vector<bool>> append_to_mark_output = nullptr;

    time_pt start_ht_probe = start_timer();
    group_ids.resize(in_table->nrows());
    const auto& active_partition_ht =
        active_partition.get()->build_hash_table_guard.value();
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        // If just build_parallel = False then we have a broadcast join on
        // the build side. So process all rows.
        // If just probe_parallel = False and build_parallel = True then we
        // still need to check batch_hashes_partition to know which rows to
        // process.
        bool process_on_rank =
            !join_state->build_parallel ||
            hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank;
        // Check the bloom filter if we would need to shuffle data
        // or if we would process the row.
        bool check_bloom_filter = shuffle_possible || process_on_rank;
        // Check bloom filter.
        // NOTE: We only need to do this in the probe_outer case since in the
        // inner case, the optimizer generates RuntimeFilter calls before the
        // probe step (and pushes them down as far as possible). It won't do so
        // in the outer case since the rows need to be preserved, so we need to
        // apply it here.
        if (use_bloom_filter && check_bloom_filter && probe_table_outer &&
            // We use batch_hashes_partition to use consistent hashing
            // across ranks for dict-encoded string arrays
            (!join_state->global_bloom_filter->Find(
                batch_hashes_partition[i_row]))) {
            join_state->metrics.probe_outer_bloom_filter_misses++;
            group_ids[i_row] = 0;
        } else if (process_on_rank) {
            join_state->metrics.num_processed_probe_table_rows++;
            if (active_partition->is_in_partition(
                    batch_hashes_partition[i_row])) {
                group_ids[i_row] = handle_probe_input_for_partition(
                    active_partition_ht, i_row);
            } else {
                append_to_probe_inactive_partition[i_row] = true;
                group_ids[i_row] = -1;
            }
        } else if (shuffle_possible) {
            append_to_probe_shuffle_buffer[i_row] = true;
            group_ids[i_row] = -1;
        } else {
            group_ids[i_row] = -1;
        }
    }
    join_state->metrics.ht_probe_time += end_timer(start_ht_probe);
    HashJoinMetrics::time_t append_time = 0;
    time_pt start_produce_probe = start_timer();
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        produce_probe_output<build_table_outer, probe_table_outer,
                             non_equi_condition, is_anti_join>(
            join_state->cond_func, active_partition.get(), i_row, 0, group_ids,
            build_idxs, probe_idxs, build_table_info_ptrs,
            probe_table_info_ptrs, build_col_ptrs, probe_col_ptrs,
            build_null_bitmaps, probe_null_bitmaps, join_state->output_buffer,
            active_partition->build_table_buffer->data_table, in_table,
            build_kept_cols, probe_kept_cols, append_time,
            join_state->is_mark_join);
    }
    join_state->metrics.produce_probe_out_idxs_time +=
        end_timer(start_produce_probe) - append_time;
    group_ids.clear();

    if (join_state->partitions.size() > 1) {
        // Skip this in the single-partition case:
        join_state->AppendProbeBatchToInactivePartition(
            in_table, batch_hashes_join, batch_hashes_partition,
            append_to_probe_inactive_partition);
    }

    if (shuffle_possible) {
        // Skip this when shuffle isn't possible.
        join_state->probe_shuffle_state.AppendBatch(
            in_table, append_to_probe_shuffle_buffer);
    }

    if (join_state->is_mark_join) {
        append_to_mark_output =
            std::make_shared<std::vector<bool>>(in_table->nrows());
        for (size_t i = 0; i < in_table->nrows(); i++) {
            (*append_to_mark_output)[i] =
                !(append_to_probe_inactive_partition[i] ||
                  append_to_probe_shuffle_buffer[i]);
        }
    }

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
        active_partition->build_table_buffer->data_table, std::move(in_table),
        build_idxs, probe_idxs, build_kept_cols, probe_kept_cols,
        join_state->is_mark_join, append_to_mark_output);
    build_idxs.clear();
    probe_idxs.clear();

    if (shuffle_possible) {
        if (local_is_last && join_state->local_is_last_debug_counter == 1) {
            std::cout
                << fmt::format(
                       "[DEBUG]:{} HashJoin (Op ID: {}): Rank {} received "
                       "local_is_last = true on probe iteration {}",
                       __LINE__, join_state->op_id, myrank,
                       join_state->probe_iter)
                << std::endl;
            join_state->local_is_last_debug_counter = 2;
        }

        std::optional<std::shared_ptr<table_info>> new_data_ =
            join_state->probe_shuffle_state.ShuffleIfRequired(local_is_last);

        if (local_is_last && join_state->local_is_last_debug_counter == 2) {
            std::cout
                << fmt::format(
                       "[DEBUG]:{} HashJoin (Op ID: {}): Rank {} received "
                       "local_is_last = true on probe iteration {}",
                       __LINE__, join_state->op_id, myrank,
                       join_state->probe_iter)
                << std::endl;
            join_state->local_is_last_debug_counter = 3;
        }

        if (new_data_.has_value()) {
            std::shared_ptr<table_info> new_data = new_data_.value();
            active_partition->probe_table = new_data;
            // NOTE: partition hashes need to be consistent across ranks
            // so need to use dictionary hashes
            std::shared_ptr<
                bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
                new_data_dict_hashes = join_state->GetDictionaryHashesForKeys();
            // Fetch the raw array pointers from the arrays for passing
            // to the non-equijoin condition
            std::vector<array_info*> build_table_info_ptrs,
                probe_table_info_ptrs;
            // Vectors for data
            std::vector<void*> build_col_ptrs, probe_col_ptrs;
            // Vectors for null bitmaps for fast null checking from the
            // cfunc
            std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
            if (non_equi_condition) {
                std::tie(build_table_info_ptrs, build_col_ptrs,
                         build_null_bitmaps) =
                    get_gen_cond_data_ptrs(
                        active_partition->build_table_buffer->data_table);
                std::tie(probe_table_info_ptrs, probe_col_ptrs,
                         probe_null_bitmaps) =
                    get_gen_cond_data_ptrs(active_partition->probe_table);
            }

            append_time = 0;

            if (local_is_last && join_state->local_is_last_debug_counter == 3) {
                std::cout
                    << fmt::format(
                           "[DEBUG]:{} HashJoin (Op ID: {}): Rank {} received "
                           "local_is_last = true on probe iteration {}",
                           __LINE__, join_state->op_id, myrank,
                           join_state->probe_iter)
                    << std::endl;
                join_state->local_is_last_debug_counter = 4;
            }

            for (size_t batch_start_row = 0;
                 batch_start_row < new_data->nrows();
                 batch_start_row += STREAMING_BATCH_SIZE) {
                size_t nrows = std::min<size_t>(
                    STREAMING_BATCH_SIZE, new_data->nrows() - batch_start_row);
                // Probe hash table with new data
                start_join_hash = start_timer();
                std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
                    new_data, join_state->n_keys, SEED_HASH_JOIN,
                    shuffle_possible, false, nullptr, batch_start_row, nrows);
                join_state->metrics.probe_join_hashing_time +=
                    end_timer(start_join_hash);
                active_partition->probe_table_hashes = batch_hashes_join.get();
                // Since we only have the hashes for this batch we need an
                // offset to the row when indexing into the hashtable
                active_partition->probe_table_hashes_offset = batch_start_row;
                join_state->metrics.num_processed_probe_table_rows += nrows;

                if (join_state->is_mark_join) {
                    append_to_mark_output->clear();
                    append_to_mark_output->resize(nrows, true);
                }

                if (join_state->partitions.size() > 1) {
                    std::cout << "RANK " << myrank << " NUMBER OF PARTITIONS: "
                              << join_state->partitions.size() << std::endl;
                    // NOTE: Partition hashes need to be consistent across
                    // ranks, so need to use dictionary hashes. Since we are
                    // using dictionary hashes, we don't need dictionaries to be
                    // global. In fact, hash_keys_table will ignore the
                    // dictionaries entirely when dict_hashes are provided. They
                    // are only needed when there is more than 1 partition.
                    start_part_hash = start_timer();
                    std::shared_ptr<uint32_t[]> batch_hashes_partition =
                        hash_keys_table(new_data, join_state->n_keys,
                                        SEED_HASH_PARTITION, shuffle_possible,
                                        /*global_dict_needed*/ false,
                                        new_data_dict_hashes, batch_start_row,
                                        nrows);
                    join_state->metrics.probe_part_hashing_time +=
                        end_timer(start_part_hash);

                    // Initialize bit-vector to false.
                    append_to_probe_inactive_partition.resize(nrows, false);
                    start_ht_probe = start_timer();
                    group_ids.resize(nrows);
                    for (size_t i_row = 0; i_row < nrows; i_row++) {
                        if (active_partition->is_in_partition(
                                batch_hashes_partition[i_row])) {
                            group_ids[i_row] = handle_probe_input_for_partition(
                                active_partition_ht,
                                // Add offset to get the actual row index in
                                // the full table:
                                i_row + batch_start_row);
                        } else {
                            append_to_probe_inactive_partition[i_row] = true;
                            group_ids[i_row] = -1;
                        }
                    }
                    join_state->metrics.ht_probe_time +=
                        end_timer(start_ht_probe);
                    append_time = 0;
                    start_produce_probe = start_timer();
                    for (size_t i_row = 0; i_row < nrows; i_row++) {
                        produce_probe_output<build_table_outer,
                                             probe_table_outer,
                                             non_equi_condition, is_anti_join>(
                            join_state->cond_func, active_partition.get(),
                            i_row + batch_start_row, batch_start_row, group_ids,
                            build_idxs, probe_idxs, build_table_info_ptrs,
                            probe_table_info_ptrs, build_col_ptrs,
                            probe_col_ptrs, build_null_bitmaps,
                            probe_null_bitmaps, join_state->output_buffer,
                            active_partition->build_table_buffer->data_table,
                            new_data, build_kept_cols, probe_kept_cols,
                            append_time, join_state->is_mark_join);
                    }
                    join_state->metrics.produce_probe_out_idxs_time +=
                        end_timer(start_produce_probe) - append_time;
                    group_ids.clear();
                    join_state->AppendProbeBatchToInactivePartition(
                        new_data, batch_hashes_join, batch_hashes_partition,
                        append_to_probe_inactive_partition,
                        /*in_table_start_offset*/ batch_start_row,
                        /*table_nrows*/ nrows);
                    if (join_state->is_mark_join) {
                        for (size_t i = 0; i < nrows; i++) {
                            (*append_to_mark_output)[i] =
                                !append_to_probe_inactive_partition[i];
                        }
                    }
                    // Reset the bit-vector. This is important for correctness.
                    append_to_probe_inactive_partition.clear();
                } else {
                    // Fast path for the single partition case:
                    start_ht_probe = start_timer();
                    group_ids.resize(nrows);
                    for (size_t i_row = 0; i_row < nrows; i_row++) {
                        group_ids[i_row] = handle_probe_input_for_partition(
                            active_partition_ht,
                            // Add offset to get the actual row index in the
                            // full table:
                            i_row + batch_start_row);
                    }
                    join_state->metrics.ht_probe_time +=
                        end_timer(start_ht_probe);
                    append_time = 0;
                    start_produce_probe = start_timer();
                    for (size_t i_row = 0; i_row < nrows; i_row++) {
                        produce_probe_output<build_table_outer,
                                             probe_table_outer,
                                             non_equi_condition, is_anti_join>(
                            join_state->cond_func, active_partition.get(),
                            i_row + batch_start_row, batch_start_row, group_ids,
                            build_idxs, probe_idxs, build_table_info_ptrs,
                            probe_table_info_ptrs, build_col_ptrs,
                            probe_col_ptrs, build_null_bitmaps,
                            probe_null_bitmaps, join_state->output_buffer,
                            active_partition->build_table_buffer->data_table,
                            new_data, build_kept_cols, probe_kept_cols,
                            append_time, join_state->is_mark_join);
                    }
                    join_state->metrics.produce_probe_out_idxs_time +=
                        end_timer(start_produce_probe) - append_time;
                    group_ids.clear();
                }

                // Reset active partition state
                active_partition->probe_table_hashes = nullptr;
                active_partition->probe_table_hashes_offset = 0;

                batch_hashes_join.reset();
                batch_hashes_partition.reset();

                join_state->output_buffer->AppendJoinOutput(
                    active_partition->build_table_buffer->data_table, new_data,
                    build_idxs, probe_idxs, build_kept_cols, probe_kept_cols,
                    join_state->is_mark_join, append_to_mark_output);
                build_idxs.clear();
                probe_idxs.clear();
            }
            active_partition->probe_table = nullptr;
        }
    }

    if (local_is_last && join_state->local_is_last_debug_counter == 4) {
        std::cout << fmt::format(
                         "[DEBUG]:{} HashJoin (Op ID: {}): Rank {} received "
                         "local_is_last = true on probe iteration {}",
                         __LINE__, join_state->op_id, myrank,
                         join_state->probe_iter)
                  << std::endl;
        join_state->local_is_last_debug_counter = 5;
    }

    if (local_is_last && !join_state->probe_shuffle_state.SendRecvEmpty()) {
        if (join_state->probe_iter % 10000 == 0) {
            std::cout << "RANK" << myrank << " Join Iteration "
                      << join_state->probe_iter
                      << " probe_shuffle_state not empty!"
                      << " Recv states empty: "
                      << join_state->probe_shuffle_state.RecvEmpty()
                      << " Send states empty: "
                      << join_state->probe_shuffle_state.SendEmpty()
                      << std::endl;
        }
    } else if (local_is_last &&
               join_state->probe_shuffle_state.SendRecvEmpty()) {
        if (join_state->probe_iter > 0 && join_state->op_id == 5) {
            std::cout << "RANK" << myrank << " Join Iteration "
                      << join_state->probe_iter << " probe_shuffle_state empty!"
                      << std::endl;
        }
    }

    // Make is_last global
    bool is_last = stream_join_sync_is_last(
        local_is_last && join_state->probe_shuffle_state.SendRecvEmpty(),
        join_state);

    if constexpr (!build_table_outer) {
        join_state->global_probe_reduce_done = true;
    }

    if (is_last && build_table_outer && !join_state->global_probe_reduce_done) {
        // We need a reduction of build misses if the probe table is
        // distributed and the build table is not.
        bool build_needs_reduction =
            join_state->probe_parallel && !join_state->build_parallel;

        if (build_needs_reduction) {
            // start reduction
            if (!join_state->probe_reduce_started) {
                std::cout << "RANK" << myrank
                          << " Starting build table matched reduction"
                          << std::endl;

                auto& build_table_matched_ =
                    active_partition.get()->build_table_matched_guard.value();
                MPI_Datatype mpi_type = get_MPI_typ(Bodo_CTypes::UINT8);
                CHECK_MPI(
                    MPI_Iallreduce(MPI_IN_PLACE, build_table_matched_->data(),
                                   build_table_matched_->size(), mpi_type,
                                   MPI_BOR, MPI_COMM_WORLD,
                                   &join_state->probe_reduce_request),
                    "join_probe_consume_batch: MPI error on MPI_Iallreduce:");
                join_state->probe_reduce_started = true;
            } else {
                int flag = 0;
                CHECK_MPI(MPI_Test(&join_state->probe_reduce_request, &flag,
                                   MPI_STATUS_IGNORE),
                          "join_probe_consume_batch: MPI error on MPI_Test:");
                if (flag) {
                    join_state->global_probe_reduce_done = true;
                }
            }
        } else {
            join_state->global_probe_reduce_done = true;
        }

        if (join_state->global_probe_reduce_done) {
            // Add unmatched rows from build table to output table
            time_pt start_build_outer = start_timer();
            if (build_needs_reduction) {
                generate_build_table_outer_rows_for_partition<true>(
                    active_partition.get(), build_idxs, probe_idxs);
            } else {
                generate_build_table_outer_rows_for_partition<false>(
                    active_partition.get(), build_idxs, probe_idxs);
            }
            join_state->metrics.build_outer_output_idx_time +=
                end_timer(start_build_outer);

            // Use the dummy probe table since all indices are -1
            join_state->output_buffer->AppendJoinOutput(
                active_partition->build_table_buffer->data_table,
                join_state->dummy_probe_table, build_idxs, probe_idxs,
                build_kept_cols, probe_kept_cols, join_state->is_mark_join);
            build_idxs.clear();
            probe_idxs.clear();
        }
    }

    join_state->probe_iter++;

    bool fully_done = is_last && join_state->global_probe_reduce_done;

    if (fully_done) {
        // TODO Free the shuffle state here to free the memory early.
        // TODO Finalize the input probe buffers of inactive partitions before
        // finalizing any other partition.

        // Free the 0th partition:
        join_state->partitions[0].reset();
        active_partition.reset();
        // Finalize and produce output from the inactive partitions.
        // This will pin the partitions (one at a time), generate all the
        // output from it and then free it.
        join_state->FinalizeProbeForInactivePartitions<
            build_table_outer, probe_table_outer, non_equi_condition,
            is_anti_join>(build_kept_cols, probe_kept_cols);
        // Finalize the probe step:
        join_state->FinalizeProbe();
    }
    return fully_done;
}

/** Add template definitions for join_probe_consume_batch for anti-join, since
only used in the DataFrame library code and not available to the compiler here.
Generated using:

In [36]: temp = """
    ...: template bool join_probe_consume_batch<{}, {}, {}, {}, true>(
    ...:     HashJoinState* join_state,
    ...:                               std::shared_ptr<table_info> in_table,
    ...:                               const std::vector<uint64_t>
build_kept_cols,
    ...:                               const std::vector<uint64_t>
probe_kept_cols,
    ...:                               bool local_is_last);
    ...: """

In [37]: for b1 in ("true", "false"):
    ...:     for b2 in ("true", "false"):
    ...:         for b3 in ("true", "false"):
    ...:             for b4 in ("true", "false"):
    ...:                 print(temp.format(b1, b2, b3, b4))
*/

template bool join_probe_consume_batch<true, true, true, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, true, true, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, true, false, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, true, false, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, false, true, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, false, true, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, false, false, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<true, false, false, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, true, true, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, true, true, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, true, false, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, true, false, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, false, true, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, false, true, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, false, false, true, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

template bool join_probe_consume_batch<false, false, false, false, true>(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool local_is_last);

/**
 * @brief Initialize a new streaming join state for specified array types
 * and number of keys (called from Python)
 *
 * @param arr_c_types array types of build table columns (Bodo_CTypes ints)
 * @param n_arrs number of build table columns
 * @param n_keys number of join keys
 * @param interval_build_columns_arr array of column indices that are
 * interval columns in the build table.
 * @param num_interval_build_columns number of interval columns in the build
 * side.
 * @param build_table_outer whether to produce left outer join
 * @param probe_table_outer whether to produce right outer join
 * @param cond_func pointer to function that evaluates non-equality
 * condition. If there is no non-equality condition, this should be NULL.
 * @param build_parallel whether the build table is distributed
 * @param probe_parallel whether the probe table is distributed
 * @param force_broadcast Should we broadcast the build side regardless
 * of size.
 * @param output_batch_size Batch size for reading output.
 * @param op_pool_size_bytes Size of the operator buffer pool for this join
 * operator. This is only applicable for the hash join case at this time.
 * If it's set to -1, we will use a fixed portion of the total available
 * memory (based on JOIN_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL).
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(
    int64_t operator_id, int8_t* build_arr_c_types,
    int8_t* build_arr_array_types, int n_build_arrs, int8_t* probe_arr_c_types,
    int8_t* probe_arr_array_types, int n_probe_arrs, uint64_t n_keys,
    int64_t* interval_build_columns_arr, int64_t num_interval_build_columns,
    bool build_table_outer, bool probe_table_outer, bool force_broadcast,
    cond_expr_fn_t cond_func, bool build_parallel, bool probe_parallel,
    int64_t output_batch_size, int64_t sync_iter, int64_t op_pool_size_bytes) {
    // If the memory budget has not been explicitly set, then ask the
    // OperatorComptroller for the budget.
    if (op_pool_size_bytes == -1) {
        op_pool_size_bytes =
            OperatorComptroller::Default()->GetOperatorBudget(operator_id);
    }
    // nested loop join is required if there are no equality keys
    if (n_keys == 0) {
        std::vector<int64_t> interval_build_columns(
            interval_build_columns_arr,
            interval_build_columns_arr + num_interval_build_columns);
        return new NestedLoopJoinState(
            bodo::Schema::Deserialize(
                std::vector<int8_t>(build_arr_array_types,
                                    build_arr_array_types + n_build_arrs),
                std::vector<int8_t>(build_arr_c_types,
                                    build_arr_c_types + n_build_arrs)),
            bodo::Schema::Deserialize(
                std::vector<int8_t>(probe_arr_array_types,
                                    probe_arr_array_types + n_probe_arrs),
                std::vector<int8_t>(probe_arr_c_types,
                                    probe_arr_c_types + n_probe_arrs)),
            build_table_outer, probe_table_outer, interval_build_columns,
            force_broadcast, cond_func, build_parallel, probe_parallel,
            output_batch_size, sync_iter, operator_id);
    }

    return new HashJoinState(
        bodo::Schema::Deserialize(
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs)),
        bodo::Schema::Deserialize(
            std::vector<int8_t>(probe_arr_array_types,
                                probe_arr_array_types + n_probe_arrs),
            std::vector<int8_t>(probe_arr_c_types,
                                probe_arr_c_types + n_probe_arrs)),
        n_keys, build_table_outer, probe_table_outer, force_broadcast,
        cond_func, build_parallel, probe_parallel, output_batch_size, sync_iter,
        operator_id, op_pool_size_bytes);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param join_state_ join state pointer
 * @param in_table build table batch
 * @param is_last is last batch locally
 * @param[out] request_input whether to request input rows from preceding
 * operators
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool join_build_consume_batch_py_entry(JoinState* join_state_,
                                       table_info* in_table, bool is_last,
                                       bool* request_input) {
    // Request input rows from preceding operators by default
    *request_input = true;

    join_state_->metrics.build_input_row_count += in_table->nrows();
    // nested loop join is required if there are no equality keys
    if (join_state_->IsNestedLoopJoin()) {
        is_last = nested_loop_join_build_consume_batch_py_entry(
            (NestedLoopJoinState*)join_state_, in_table, is_last);
    } else {
        HashJoinState* join_state = (HashJoinState*)join_state_;
        try {
            bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
            is_last = join_build_consume_batch(
                join_state, std::unique_ptr<table_info>(in_table),
                has_bloom_filter, is_last);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return false;
        }
        *request_input = !join_state->build_shuffle_state.BuffersFull();
    }
    if (is_last && (join_state_->op_id != -1)) {
        try {
            // Build doesn't output anything, so it's output row count is 0.
            QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                QueryProfileCollector::MakeOperatorStageID(
                    join_state_->op_id, QUERY_PROFILE_BUILD_STAGE_ID),
                0);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    }
    return is_last;
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
        join_state_->metrics.probe_input_row_count += input_table->nrows();

        // nested loop join is required if there are no equality keys
        if (join_state_->IsNestedLoopJoin()) {
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
            has_non_equi_cond_exp, use_bloom_filter_exp, false>(            \
            join_state, std::move(input_table), std::move(build_kept_cols), \
            std::move(probe_kept_cols), is_last);                           \
    }
#endif

            bool contain_non_equi_cond = join_state->cond_func != nullptr;

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
            if (join_state->probe_shuffle_state.BuffersFull()) {
                *request_input = false;
            }
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
            join_state_->metrics.probe_output_row_count += chunk_size;
        }
        // This is the last output if we've already seen all input (i.e.
        // is_last) and there's no more output remaining in the output_buffer:
        *out_is_last =
            is_last && join_state_->output_buffer->total_remaining == 0;

        // If this is the last output, submit the row count stats for the query
        // profile.
        if (*out_is_last && (join_state_->op_id != -1)) {
            QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                QueryProfileCollector::MakeOperatorStageID(
                    join_state_->op_id, QUERY_PROFILE_PROBE_STAGE_ID),
                join_state_->metrics.probe_output_row_count);
        }
        return out_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint for runtime join filter.
 *
 * @param join_state_ Pointer to the join state.
 * @param in_table_ The table to filter.
 * @param[in, out] row_bitmask_arr The bitmask whether row is included in the
 * current set. Note that this will be modified in place. Expects
 * array_type=NULLABLE_INTO_BOOL, dtype=BOOL.
 * @param join_key_idxs_ An array of length join_state->n_keys specifying the
 * indices in the input table that correspond to the join keys.
 * @param join_key_idxs_len Length of join_key_idxs_. This must be equal to
 * join_state->n_keys.
 * @param process_col_bitmask_ An array of boolean specifying whether a column
 * level filter for the i'th key should be applied.
 * @param process_col_bitmask_len Length of process_col_bitmask_. This must be
 * equal to join_state->n_keys.
 * @param applied_any_filter Whether any filters have been applied in previous
 * RTJFs.
 * @return bool whether any filters were applied.
 */
bool runtime_join_filter_py_entry(JoinState* join_state_, table_info* in_table_,
                                  array_info* row_bitmask_arr,
                                  int64_t* join_key_idxs_,
                                  int64_t join_key_idxs_len,
                                  bool* process_col_bitmask_,
                                  int64_t process_col_bitmask_len,
                                  bool applied_any_filter) {
    try {
        assert(join_key_idxs_len == process_col_bitmask_len);
        assert(join_key_idxs_len == static_cast<int64_t>(join_state_->n_keys));
        assert(row_bitmask_arr->arr_type ==
                   bodo_array_type::NULLABLE_INT_BOOL &&
               row_bitmask_arr->dtype == Bodo_CTypes::_BOOL);

        if (join_state_->IsNestedLoopJoin()) {
            // Filters are only supported for equi hash joins.
            return applied_any_filter;
        } else {
            HashJoinState* join_state = (HashJoinState*)join_state_;
            auto in_table_ptr = std::unique_ptr<table_info>(in_table_);
            auto row_bitmask_ptr = std::unique_ptr<array_info>(row_bitmask_arr);
            std::vector<int64_t> join_key_idxs(
                join_key_idxs_, join_key_idxs_ + join_key_idxs_len);
            std::vector<bool> process_col_bitmask(
                process_col_bitmask_,
                process_col_bitmask_ + process_col_bitmask_len);
            bool result = join_state->RuntimeFilter(
                std::move(in_table_ptr), std::move(row_bitmask_ptr),
                join_key_idxs, process_col_bitmask);
            return result || applied_any_filter;
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
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
    if (join_state_->IsNestedLoopJoin()) {
        NestedLoopJoinState* join_state = (NestedLoopJoinState*)join_state_;
        delete join_state;
    } else {
        HashJoinState* join_state = (HashJoinState*)join_state_;
        delete join_state;
    }
}

uint64_t get_op_pool_budget_bytes(JoinState* join_state) {
    return ((HashJoinState*)join_state)->op_pool_budget_bytes();
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
        if (static_cast<size_t>(idx) >= partitions.size()) {
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
        if (static_cast<size_t>(idx) >= partitions.size()) {
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

/**
 * Helper for get_runtime_join_filter_min_max_py_entrypt on string/dictionary
 * columns.
 */
PyObject* get_runtime_join_filter_min_max_string(JoinState* join_state,
                                                 int64_t key_idx, bool is_min) {
    size_t row = is_min ? 0 : 1;
    if (join_state->min_max_values[key_idx].has_value() &&
        join_state->min_max_values[key_idx].value()->get_null_bit(row)) {
        char* data = join_state->min_max_values[key_idx]
                         .value()
                         ->data1<bodo_array_type::STRING, char>();
        offset_t* offsets = join_state->min_max_values[key_idx]
                                .value()
                                ->data2<bodo_array_type::STRING, offset_t>();
        offset_t start = offsets[row];
        offset_t end = offsets[row + 1];
        return PyUnicode_FromStringAndSize(data + start, end - start);
    } else {
        return Py_None;
    }
}

/**
 * Python entrypoint to fetch the minimum or maximum value of a specific
 * join key as a PyObject so it can be used an a runtime join filter. If
 * there is no value to fetch, returns a None PyPObject.
 *
 * @param[in] join_state: the state object containing any min/max information
 *            from the build table.
 * @param[in] key_idx: the column index of the join key being requested.
 * @param[in] is_min: if true, fetches the min. If false, fetches the max.
 * @param[in] precision: the precision to use for types like Time.
 */
PyObject* get_runtime_join_filter_min_max_py_entrypt(JoinState* join_state,
                                                     int64_t key_idx,
                                                     bool is_min,
                                                     int64_t precision) {
    try {
        std::unique_ptr<bodo::DataType>& col_type =
            join_state->build_table_schema->column_types[key_idx];
        assert(join_state->build_input_finalized);
#define RTJF_MIN_MAX_NUMERIC_CASE(dtype, ctype, py_func)                       \
    case dtype: {                                                              \
        using T = typename dtype_to_type<dtype>::type;                         \
        size_t row = is_min ? 0 : 1;                                           \
        if (join_state->min_max_values[key_idx].has_value() &&                 \
            join_state->min_max_values[key_idx].value()->get_null_bit(row)) {  \
            T val = getv<T>(join_state->min_max_values[key_idx].value(), row); \
            return py_func((ctype)val);                                        \
        } else {                                                               \
            return Py_None;                                                    \
        }                                                                      \
    }
        switch (col_type->array_type) {
            case bodo_array_type::NUMPY:
            case bodo_array_type::NULLABLE_INT_BOOL: {
                switch (col_type->c_type) {
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::INT8, int64_t,
                                              PyLong_FromLong);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::INT16, int64_t,
                                              PyLong_FromLong);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::INT32, int64_t,
                                              PyLong_FromLong);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::INT64, int64_t,
                                              PyLong_FromLong);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::UINT8, size_t,
                                              PyLong_FromSsize_t);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::UINT16, size_t,
                                              PyLong_FromSsize_t);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::UINT32, size_t,
                                              PyLong_FromSsize_t);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::UINT64, size_t,
                                              PyLong_FromSsize_t);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::FLOAT32, double,
                                              PyFloat_FromDouble);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::FLOAT64, double,
                                              PyFloat_FromDouble);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::DATE, size_t,
                                              py_date_from_int);
                    RTJF_MIN_MAX_NUMERIC_CASE(Bodo_CTypes::DATETIME, size_t,
                                              py_timestamp_from_int);
                    default: {
                        // If the column is not a supported nullable/numpy
                        // dtype for min/max pushdown, return None
                        // back to Python so it knows not to insert the filter.
                        return Py_None;
                    }
                }
                case bodo_array_type::STRING:
                case bodo_array_type::DICT: {
                    return get_runtime_join_filter_min_max_string(
                        join_state, key_idx, is_min);
                }
                default: {
                    return Py_None;
                }
            }
        }
#undef RTJF_MIN_MAX_NUMERIC_CASE
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Determine if the finalized build table is empty.
 * @param[in] join_state: the state object containing build table information.
 * @return bool whether the build table is empty.
 */
bool is_empty_build_table_py_entrypt(JoinState* join_state) {
    return join_state->global_build_empty;
}

/**
 * Function to be called before get_runtime_join_filter_unique_values_py_entrypt
 * to determine if the list of unique values even exists.
 *
 * @param[in] join_state: the state object containing any unique value
 *            information from the build table.
 * @param[in] key_idx: the column index of the join key being requested.
 */
bool has_runtime_join_filter_unique_values_py_entrypt(JoinState* join_state_,
                                                      int64_t key_idx) {
    HashJoinState* join_state = (HashJoinState*)join_state_;
    return (key_idx < (int64_t)(join_state->n_keys)) &&
           (join_state->build_input_finalized) &&
           (join_state->unique_values[key_idx].has_value());
}

/**
 * Python entrypoint to fetch the unique values of a specific
 * join key as a PyObject so it can be used an a runtime join filter. If
 * there is no value to fetch, throws an exception.
 *
 * @param[in] join_state: the state object containing any unique value
 *            information from the build table.
 * @param[in] key_idx: the column index of the join key being requested.
 */
PyObject* get_runtime_join_filter_unique_values_py_entrypt(
    JoinState* join_state_, int64_t key_idx, bool is_min) {
    try {
        auto result = PyList_New(0);
        HashJoinState* join_state = (HashJoinState*)join_state_;
        std::unique_ptr<bodo::DataType>& col_type =
            join_state->build_table_schema->column_types[key_idx];
        assert(key_idx < static_cast<int64_t>(join_state->n_keys));
        assert(join_state->build_input_finalized);
        assert(join_state->unique_values[key_idx].has_value());
        std::unordered_set<int64_t>& unique_vals_set =
            join_state->unique_values[key_idx].value();
#define RTJF_UNIQUE_VALUES_NUMERIC_CASE(dtype, ctype, pyfunc)          \
    case dtype: {                                                      \
        for (int64_t val : unique_vals_set) {                          \
            ctype as_ctype = bit_preserving_cast<int64_t, ctype>(val); \
            PyObject* obj = pyfunc(as_ctype);                          \
            PyList_Append(result, obj);                                \
            Py_DECREF(obj);                                            \
        }                                                              \
        break;                                                         \
    }
        switch (col_type->array_type) {
            case bodo_array_type::NUMPY:
            case bodo_array_type::NULLABLE_INT_BOOL: {
                switch (col_type->c_type) {
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::INT8, int64_t,
                                                    PyLong_FromLong);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::INT16, int64_t,
                                                    PyLong_FromLong);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::INT32, int64_t,
                                                    PyLong_FromLong);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::INT64, int64_t,
                                                    PyLong_FromLong);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::UINT8, size_t,
                                                    PyLong_FromSsize_t);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::UINT16, size_t,
                                                    PyLong_FromSsize_t);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::UINT32, size_t,
                                                    PyLong_FromSsize_t);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::UINT64, size_t,
                                                    PyLong_FromSsize_t);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::FLOAT32,
                                                    double, PyFloat_FromDouble);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::FLOAT64,
                                                    double, PyFloat_FromDouble);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(Bodo_CTypes::DATE, size_t,
                                                    py_date_from_int);
                    RTJF_UNIQUE_VALUES_NUMERIC_CASE(
                        Bodo_CTypes::DATETIME, size_t, py_timestamp_from_int);
                    default: {
                        throw std::runtime_error(
                            "get_runtime_join_filter_unique_values_py_entrypt: "
                            "Unsupported dtype type " +
                            GetDtype_as_string(col_type->c_type));
                    }
                }
                break;
            }
            default: {
                throw std::runtime_error(
                    "get_runtime_join_filter_unique_values_py_entrypt: "
                    "Unsupported array type " +
                    GetArrType_as_string(col_type->array_type));
            }
        }
        return result;
#undef RTJF_UNIQUE_NUMERIC_CASE
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMODINIT_FUNC PyInit_stream_join_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_join_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, join_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, join_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, join_probe_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, runtime_join_filter_py_entry);
    SetAttrStringFromVoidPtr(m, delete_join_state);
    SetAttrStringFromVoidPtr(m, nested_loop_join_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, generate_array_id);
    SetAttrStringFromVoidPtr(m, get_op_pool_budget_bytes);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_pinned);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_allocated);
    SetAttrStringFromVoidPtr(m, get_num_partitions);
    SetAttrStringFromVoidPtr(m, get_partition_num_top_bits_by_idx);
    SetAttrStringFromVoidPtr(m, get_partition_top_bitmask_by_idx);
    SetAttrStringFromVoidPtr(m, get_runtime_join_filter_min_max_py_entrypt);
    SetAttrStringFromVoidPtr(m, is_empty_build_table_py_entrypt);
    SetAttrStringFromVoidPtr(m,
                             has_runtime_join_filter_unique_values_py_entrypt);
    SetAttrStringFromVoidPtr(m,
                             get_runtime_join_filter_unique_values_py_entrypt);

    return m;
}
