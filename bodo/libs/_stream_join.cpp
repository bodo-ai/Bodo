#include "_stream_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"

/* --------------------------- Helper Functions --------------------------- */

/**
 * @brief Get the dict builders from a table build buffer object.
 *
 * @param table_build_buffer TableBuildBuffer to get the dict builders from.
 * @return std::vector<std::shared_ptr<DictionaryBuilder>>
 */
std::vector<std::shared_ptr<DictionaryBuilder>>
get_dict_builders_from_table_build_buffer(
    TableBuildBuffer& table_build_buffer) {
    size_t n_cols = table_build_buffer.data_table->columns.size();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        dict_builders[i] = table_build_buffer.array_buffers[i].dict_builder;
    }
    return dict_builders;
}

/* ------------------------------------------------------------------------ */

/* --------------------------- HashHashJoinTable -------------------------- */

uint32_t HashHashJoinTable::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        return this->join_partition->build_table_join_hashes[iRow];
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

inline bool JoinPartition::is_in_partition(const uint32_t& hash) {
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

std::vector<std::shared_ptr<JoinPartition>> JoinPartition::SplitPartition(
    size_t num_levels) {
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

    // Rebuild build_arr_c_types, build_arr_array_types, probe_arr_c_types,
    // probe_arr_array_types and dict_builders from the build and probe buffers.
    // Note that for now we will share the dict builders between all partitions,
    // but this might change in the future.
    auto [build_arr_c_types, build_arr_array_types] =
        get_dtypes_arr_types_from_table(this->build_table_buffer.data_table);
    auto [probe_arr_c_types, probe_arr_array_types] =
        get_dtypes_arr_types_from_table(this->probe_table_buffer.data_table);
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders =
        get_dict_builders_from_table_build_buffer(this->build_table_buffer);
    std::vector<std::shared_ptr<DictionaryBuilder>> probe_table_dict_builders =
        get_dict_builders_from_table_build_buffer(this->probe_table_buffer);
    // Get dictionary hashes from the dict-builders of build table.
    // Dictionaries of key columns are shared between build and probe tables,
    // so using either is fine.
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = std::make_shared<
            bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>();
    dict_hashes->reserve(this->n_keys);
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (build_table_dict_builders[i] == nullptr) {
            dict_hashes->push_back(nullptr);
        } else {
            dict_hashes->emplace_back(
                build_table_dict_builders[i]->GetDictionaryHashes());
        }
    }

    // Create the two new partitions. These will differ on the next bit.
    std::shared_ptr<JoinPartition> new_part1 = std::make_shared<JoinPartition>(
        this->num_top_bits + 1, (this->top_bitmask << 1), build_arr_c_types,
        build_arr_array_types, probe_arr_c_types, probe_arr_array_types,
        this->n_keys, this->build_table_outer, this->probe_table_outer,
        build_table_dict_builders, probe_table_dict_builders);
    std::shared_ptr<JoinPartition> new_part2 = std::make_shared<JoinPartition>(
        this->num_top_bits + 1, (this->top_bitmask << 1) + 1, build_arr_c_types,
        build_arr_array_types, probe_arr_c_types, probe_arr_array_types,
        this->n_keys, this->build_table_outer, this->probe_table_outer,
        build_table_dict_builders, probe_table_dict_builders);

    // Reserve space in build_table_buffer and hash vectors for both
    // partitions.
    // XXX This is over provisioning, so need to figure out how
    // to restrict that.
    new_part1->build_table_buffer.ReserveTable(
        this->build_table_buffer.data_table);
    new_part1->build_table_join_hashes.reserve(
        this->build_table_join_hashes.size());
    new_part2->build_table_buffer.ReserveTable(
        this->build_table_buffer.data_table);
    new_part2->build_table_join_hashes.reserve(
        this->build_table_join_hashes.size());

    // Compute partitioning hashes:
    std::shared_ptr<uint32_t[]> build_table_partitioning_hashes =
        hash_keys_table(this->build_table_buffer.data_table, this->n_keys,
                        SEED_HASH_PARTITION, false, false, dict_hashes);

    // Put the build data in the sub partitions.
    // XXX Might be faster to pre-calculate the required
    // sizes, build a bitmap, pre-allocated space and then append
    // into the new partitions.
    std::vector<bool> append_partion1(
        this->build_table_buffer.data_table->nrows(), false);
    for (size_t i_row = 0; i_row < build_table_buffer.data_table->nrows();
         i_row++) {
        append_partion1[i_row] =
            new_part1->is_in_partition(build_table_partitioning_hashes[i_row]);
    }

    // Copy the hash values to the new partitions.
    for (size_t i_row = 0; i_row < build_table_buffer.data_table->nrows();
         i_row++) {
        if (append_partion1[i_row]) {
            new_part1->build_table_join_hashes.push_back(
                this->build_table_join_hashes[i_row]);
        } else {
            new_part2->build_table_join_hashes.push_back(
                this->build_table_join_hashes[i_row]);
        }
    }

    new_part1->build_table_buffer.AppendBatch(
        this->build_table_buffer.data_table, append_partion1);

    append_partion1.flip();
    std::vector<bool>& append_partion2 = append_partion1;

    new_part2->build_table_buffer.AppendBatch(
        this->build_table_buffer.data_table, append_partion2);

    // Splitting happens at build time, so the probe buffers should
    // be empty.

    return {new_part1, new_part2};
}

void JoinPartition::BuildHashTable() {
    for (size_t i_row = this->curr_build_size;
         i_row < this->build_table_buffer.data_table->nrows(); i_row++) {
        this->InsertLastRowIntoMap();
        this->curr_build_size++;
    }
}

void JoinPartition::ReserveBuildTable(
    const std::shared_ptr<table_info>& in_table) {
    this->build_table_buffer.ReserveTable(in_table);
}

void JoinPartition::ReserveProbeTable(
    const std::shared_ptr<table_info>& in_table) {
    this->probe_table_buffer.ReserveTable(in_table);
}

inline void JoinPartition::InsertLastRowIntoMap() {
    std::vector<std::vector<size_t>>& groups = this->groups;
    size_t& group_id = this->build_table[this->curr_build_size];
    // group_id==0 means key doesn't exist in map
    if (group_id == 0) {
        // Update the value of group_id stored in the hash map
        // as well since its pass by reference.
        group_id = groups.size() + 1;
        groups.emplace_back();
    }
    groups[group_id - 1].emplace_back(this->curr_build_size);
}

// XXX Currently the implementation for the active and inactive case are
// very similar, but this will change in the future, hence the template
// variable.
template <bool is_active>
void JoinPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes) {
    this->build_table_join_hashes.insert(this->build_table_join_hashes.end(),
                                         join_hashes.get(),
                                         join_hashes.get() + in_table->nrows());
    this->build_table_buffer.AppendBatch(in_table);
    for (size_t i_row = 0; is_active && i_row < in_table->nrows(); i_row++) {
        this->InsertLastRowIntoMap();
        this->curr_build_size++;
    }
}

template <bool is_active>
void JoinPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::vector<bool>& append_rows) {
    this->build_table_buffer.AppendBatch(in_table, append_rows);
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            this->build_table_join_hashes.push_back(join_hashes[i_row]);
            if (is_active) {
                this->InsertLastRowIntoMap();
                this->curr_build_size++;
            }
        }
    }
}

void JoinPartition::FinalizeBuild() {
    // XXX Currently, there's a single table_buffer, so we don't
    // need to concat, etc.
    // TODO Add steps to pin the buffers and concat.
    // TODO Add logic to identify that the buffer is too large
    // and that we need to re-partition.
    this->BuildHashTable();
    if (this->build_table_outer) {
        this->build_table_matched.resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer.data_table->nrows()),
            0);
    }
    // TODO Unpin all state
}

void JoinPartition::AppendInactiveProbeBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::vector<bool>& append_rows) {
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            this->probe_table_buffer_join_hashes.push_back(join_hashes[i_row]);
        }
    }
    this->probe_table_buffer.AppendBatch(in_table, append_rows);
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
 * @param cond_func Condition function to use. `nullptr` for the all equality
 * conditions case.
 * @param[in, out] partition Partition that this row belongs to.
 * @param i_row Row index in partition->probe_table to produce the output for,
 * @param[in, out] build_idxs Build table indices for the output. This will be
 * updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will be
 * updated in place.
 *
 * The rest of the parameters are output of get_gen_cond_data_ptrs on the build
 * and probe table and are only relevant for the condition function case:
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
    auto iter = partition->build_table.find(-i_row - 1);
    if (iter == partition->build_table.end()) {
        if (probe_table_outer) {
            // Add unmatched rows from probe table to output table
            build_idxs.push_back(-1);
            probe_idxs.push_back(i_row);
        }
        return;
    }
    const std::vector<size_t>& group = partition->groups[iter->second - 1];
    // Initialize to true for pure hash join so the final branch
    // is non-equality condition only.
    bool has_match = !non_equi_condition;
    for (size_t j_build : group) {
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
            SetBitTo(partition->build_table_matched, j_build, 1);
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
 * all the build records that didn't match, and adds them to the output (with
 * NULL on the probe side).
 * @Tparam requires_reduction Whether the build matches require a reduction
 * because the probe table is distributed but the build table is replicated.
 *
 * @param partition Join partition to produce the output for.
 * @param[in, out] build_idxs Build table indices for the output. This will be
 * updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will be
 * updated in place.
 */
template <bool requires_reduction>
void generate_build_table_outer_rows_for_partition(
    JoinPartition* partition, bodo::vector<int64_t>& build_idxs,
    bodo::vector<int64_t>& probe_idxs) {
    if (requires_reduction) {
        MPI_Allreduce_bool_or(partition->build_table_matched);
    }
    int n_pes, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Add unmatched rows from build table to output table
    for (size_t i_row = 0;
         i_row < partition->build_table_buffer.data_table->nrows(); i_row++) {
        if ((!requires_reduction || ((i_row % n_pes) == my_rank))) {
            bool has_match = GetBit(partition->build_table_matched, i_row);
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
    // XXX Currently, there's a single table_buffer, but in the future,
    // there'll probably be multiple buffers.
    this->probe_table = this->probe_table_buffer.data_table;
    this->probe_table_hashes = this->probe_table_buffer_join_hashes.data();
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    // TODO Pin the build table, etc.

    // Fetch the raw array pointers from the arrays for passing
    // to the non-equijoin condition
    std::vector<array_info*> build_table_info_ptrs, probe_table_info_ptrs;
    // Vectors for data
    std::vector<void*> build_col_ptrs, probe_col_ptrs;
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
    if (non_equi_condition) {
        std::tie(build_table_info_ptrs, build_col_ptrs, build_null_bitmaps) =
            get_gen_cond_data_ptrs(this->build_table_buffer.data_table);
        std::tie(probe_table_info_ptrs, probe_col_ptrs, probe_null_bitmaps) =
            get_gen_cond_data_ptrs(this->probe_table);
    }

    for (size_t i_row = 0; i_row < this->probe_table->nrows(); i_row++) {
        // TODO Add steps to pin the selected buffer.
        handle_probe_input_for_partition<build_table_outer, probe_table_outer,
                                         non_equi_condition>(
            cond_func, this, i_row, build_idxs, probe_idxs,
            build_table_info_ptrs, probe_table_info_ptrs, build_col_ptrs,
            probe_col_ptrs, build_null_bitmaps, probe_null_bitmaps);
    }
    // TODO Free the selected buffer.

    if (build_table_outer) {
        // Add unmatched rows from build table to output table
        if (build_needs_reduction) {
            generate_build_table_outer_rows_for_partition<true>(
                this, build_idxs, probe_idxs);
        } else {
            generate_build_table_outer_rows_for_partition<false>(
                this, build_idxs, probe_idxs);
        }
    }

    output_buffer->AppendJoinOutput(this->build_table_buffer.data_table,
                                    this->probe_table, build_idxs, probe_idxs,
                                    build_kept_cols, probe_kept_cols);
    build_idxs.clear();
    probe_idxs.clear();
    this->probe_table = nullptr;
    this->probe_table_hashes = nullptr;

    // TODO Unpin/Free all state
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
                     int64_t output_batch_size_)
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
 * @param only_keys only unify key columns
 * @return std::shared_ptr<table_info> input table with dictionaries unified
 * with build table dictionaries.
 */
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
                             int64_t output_batch_size_,
                             uint64_t shuffle_sync_iter,
                             size_t max_partition_depth_)
    : JoinState(build_arr_c_types, build_arr_array_types, probe_arr_c_types,
                probe_arr_array_types, n_keys_, build_table_outer_,
                probe_table_outer_, cond_func_, build_parallel_,
                probe_parallel_, output_batch_size_),
      max_partition_depth(max_partition_depth_),
      build_iter(0),
      probe_iter(0),
      shuffle_sync_iter(shuffle_sync_iter),
      join_event("HashJoin") {
    this->build_shuffle_buffer =
        TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                         this->build_table_dict_builders);
    this->probe_shuffle_buffer =
        TableBuildBuffer(probe_arr_c_types, probe_arr_array_types,
                         this->probe_table_dict_builders);
    // Create a build buffer for NA values to skip the hash table.
    this->build_na_key_buffer =
        TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                         this->build_table_dict_builders);

    // Create the initial partition
    this->partitions.emplace_back(std::make_shared<JoinPartition>(
        0, 0, build_arr_c_types, build_arr_array_types, probe_arr_c_types,
        probe_arr_array_types, n_keys_, build_table_outer_, probe_table_outer_,
        this->build_table_dict_builders, this->probe_table_dict_builders));

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

    // Call SplitPartition on the current active partition
    std::vector<std::shared_ptr<JoinPartition>> new_partitions =
        this->partitions[idx]->SplitPartition();
    // Remove the current partition (this should release its memory)
    this->partitions.erase(this->partitions.begin() + idx);
    // Insert the new partitions in its place
    this->partitions.insert(this->partitions.begin() + idx,
                            new_partitions.begin(), new_partitions.end());
    // Rebuild the hash table for the _new_ active partition
    this->partitions[idx]->BuildHashTable();

    // TODO Check if the new active partition needs to be split up further.
    // XXX Might not be required if we split proactively and there isn't
    // a single hot key (in which case we need to fall back to nested loop
    // join for this partition).
}

void HashJoinState::ResetPartitions() {
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders =
        get_dict_builders_from_table_build_buffer(
            this->partitions[0]->build_table_buffer);
    std::vector<std::shared_ptr<DictionaryBuilder>> probe_table_dict_builders =
        get_dict_builders_from_table_build_buffer(
            this->partitions[0]->probe_table_buffer);

    std::shared_ptr<JoinPartition> new_partition =
        std::make_shared<JoinPartition>(
            0, 0, this->build_arr_c_types, this->build_arr_array_types,
            this->probe_arr_c_types, this->probe_arr_array_types, this->n_keys,
            this->build_table_outer, this->probe_table_outer,
            build_table_dict_builders, probe_table_dict_builders);
    this->partitions.clear();
    this->partitions.emplace_back(new_partition);
}

void HashJoinState::ReserveBuildTable(
    const std::shared_ptr<table_info>& in_table) {
    // XXX This currently over allocates, so will need to be adjusted based
    // on whether or not it's an active partition or not.
    for (auto& partition : this->partitions) {
        partition->ReserveBuildTable(in_table);
    }
}

void HashJoinState::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch<true>(in_table, join_hashes);
        return;
    }

    std::vector<std::vector<bool>> append_rows_by_partition(
        this->partitions.size());
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        append_rows_by_partition[i_part] = std::vector<bool>(in_table->nrows());
    }

    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        bool found_partition = false;

        // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize partition
        // search by storing a tree representation of the partition space.
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
    this->partitions[0]->AppendBuildBatch<true>(in_table, join_hashes,
                                                append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendBuildBatch(
            in_table, join_hashes, append_rows_by_partition[i_part]);
    }
}

void HashJoinState::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch<true>(in_table, join_hashes,
                                                    append_rows);
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
    this->partitions[0]->AppendBuildBatch<true>(in_table, join_hashes,
                                                append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->AppendBuildBatch(
            in_table, join_hashes, append_rows_by_partition[i_part]);
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
        // Create the idxs.
        int64_t n_rows = this->build_na_key_buffer.data_table->nrows();
        std::vector<int64_t> build_idxs(n_rows);
        std::vector<int64_t> probe_idxs(n_rows, -1);
        for (int64_t i = 0; i < n_rows; i++) {
            build_idxs[i] = i;
        }
        this->output_buffer->AppendJoinOutput(
            this->build_na_key_buffer.data_table, this->dummy_probe_table,
            build_idxs, probe_idxs, build_kept_cols, probe_kept_cols);
        // TODO: Free memory. This is never used again.
    }
}

void HashJoinState::FinalizeBuild() {
    // TODO Add steps to make sure only one partition is pinned at a time.
    // TODO Add logic to check if partition is too big and needs to be
    // repartitioned.
    // TODO Free shuffle buffer, etc.
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->FinalizeBuild();
    }
    JoinState::FinalizeBuild();
}

void HashJoinState::ReserveProbeTableForInactivePartitions(
    const std::shared_ptr<table_info>& in_table) {
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->ReserveProbeTable(in_table);
    }
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
                    "Couldn't "
                    "find "
                    "any matching partition for row!");
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
        this->partitions[i]
            ->FinalizeProbeForInactivePartition<
                build_table_outer, probe_table_outer, non_equi_condition>(
                this->cond_func, build_kept_cols, probe_kept_cols,
                build_needs_reduction, this->output_buffer);
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
    // then we only add a fraction of NA rows to the output. Otherwise we add
    // all rows.
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
            // If have an outer join we must push the NA values directly to the
            // output, not just filter them.
            join_state->build_na_key_buffer.AppendBatch(in_table, append_nas);
        }
        return RetrieveTable(std::move(in_table), std::move(idx_list));
    }
}

/* ------------------------------------------------------------------------ */

/**
 * @brief consume build table batch in streaming join (insert into hash
 * table)
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void join_build_consume_batch(HashJoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              bool use_bloom_filter, bool is_last) {
    if (join_state->build_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "join_build_consume_batch: Received non-empty in_table after "
                "the build was already finalized!");
        }
        // Nothing left to do for build
        return;
    }
    int n_pes, myrank;
    auto iterationEvent(join_state->join_event.iteration());
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Unify dictionaries to allow consistent hashing and fast key comparison
    // using indices
    // NOTE: key columns in build_table_buffer (of all partitions),
    // probe_table_buffers (of all partitions), build_shuffle_buffer and
    // probe_shuffle_buffer use the same dictionary object for consistency.
    // Non-key DICT columns of build_table_buffer and build_shuffle_buffer also
    // share their dictionaries and will also be unified.
    in_table = join_state->UnifyBuildTableDictionaryArrays(in_table);

    // Dictionary hashes for the key columns which will be used for
    // the partitioning hashes:
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = join_state->GetDictionaryHashesForKeys();

    // Prune any rows with NA keys. If this is an build_table_outer = False,
    // then we can prune these rows from the table entirely. If
    // build_table_outer = True then we can skip adding these rows to the hash
    // table (as they can't match), but must write them to the Join output.
    // TODO: Have outer join skip the build table/avoid shuffling.
    if (join_state->build_table_outer) {
        in_table = filter_na_values<true>(join_state, std::move(in_table),
                                          join_state->n_keys);
    } else {
        in_table = filter_na_values<false>(join_state, std::move(in_table),
                                           join_state->n_keys);
    }

    // Get hashes of the new batch (different hashes for partitioning and hash
    // table to reduce conflict)
    // NOTE: Partition hashes need to be consistent across ranks so need to use
    // dictionary hashes. Since we are using dictionary hashes, we don't
    // need dictionaries to be global. In fact, hash_keys_table will ignore
    // the dictionaries entirely when dict_hashes are provided.
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        join_state->build_parallel, false, dict_hashes);
    std::shared_ptr<uint32_t[]> batch_hashes_join =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_JOIN,
                        join_state->build_parallel,
                        /*global_dict_needed*/ false);

    // Insert batch into the correct partition.
    // TODO[BSE-441]: see if appending all selected rows upfront to the build
    // buffer is faster (a call like AppendSelectedRows that takes a bool vector
    // from partitioning and appends all selected input rows)
    // TODO[BSE-441]: tune initial buffer buffer size and expansion strategy
    // using heuristics (e.g. SQL planner statistics)
    join_state->ReserveBuildTable(in_table);
    join_state->build_shuffle_buffer.ReserveTable(in_table);
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

    join_state->AppendBuildBatch(in_table, batch_hashes_join,
                                 batch_hashes_partition,
                                 append_row_to_build_table);

    append_row_to_build_table.flip();
    std::vector<bool>& append_row_to_shuffle_table = append_row_to_build_table;
    join_state->build_shuffle_buffer.AppendBatch(in_table,
                                                 append_row_to_shuffle_table);

    batch_hashes_partition.reset();
    batch_hashes_join.reset();
    in_table.reset();

    // For now, we will allow re-partitioning only in the case where the build
    // side is distributed. If we allow re-partitioning in the replicated build
    // side case, we must assume that the partitioning state is identical on all
    // ranks. This might not always be true.
    // XXX Revisit this in the future if needed.
    if (!is_last && join_state->build_parallel &&
        join_state->partitions[0]->is_near_full()) {
        join_state->SplitPartition(0);
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
        MPI_Allreduce(MPI_IN_PLACE, &single_partition, 1, MPI_C_BOOL, MPI_LAND,
                      MPI_COMM_WORLD);
        if (single_partition) {
            int64_t global_table_size = table_global_memory_size(
                join_state->partitions[0]->build_table_buffer.data_table);
            global_table_size += table_global_memory_size(
                join_state->build_shuffle_buffer.data_table);
            if (global_table_size < get_bcast_join_threshold()) {
                join_state->build_parallel = false;

                // We have decided to do a broadcast join. To do this we will
                // execute the following steps:
                //
                // 1. Combine the shuffle buffer into the existing partition.
                // This is necessary so we can shuffle a single table.
                //
                // 2. Broadcast the table across all ranks with allgatherv.
                //
                // 3. Clear the existing JoinPartition state. This is necessary
                // because the allgatherv includes rows that we have already
                // processed and we need to avoid processing them twice.
                //
                // 4. Insert the entire table into the new partition.

                // Step 1: Combine the shuffle buffer into the existing
                // partition

                // Append all the shuffle data to the partition. This allows
                // us to just shuffle 1 table.
                // Dictionary hashes for the key columns which will be used
                // for the partitioning hashes:
                dict_hashes = join_state->GetDictionaryHashesForKeys();

                batch_hashes_partition =
                    hash_keys_table(join_state->build_shuffle_buffer.data_table,
                                    join_state->n_keys, SEED_HASH_PARTITION,
                                    false, false, dict_hashes);
                batch_hashes_join =
                    hash_keys_table(join_state->build_shuffle_buffer.data_table,
                                    join_state->n_keys, SEED_HASH_JOIN, false,
                                    /*global_dict_needed*/ false);

                join_state->ReserveBuildTable(
                    join_state->build_shuffle_buffer.data_table);
                join_state->AppendBuildBatch(
                    join_state->build_shuffle_buffer.data_table,
                    batch_hashes_join, batch_hashes_partition);

                // Free the hashes
                batch_hashes_partition.reset();
                batch_hashes_join.reset();
                // Reset the build shuffle buffer. This will also
                // reset the dictionaries to point to the shared dictionaries
                // and reset the dictionary related flags.
                // This is crucial for correctness.
                join_state->build_shuffle_buffer.Reset();

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
                batch_hashes_join = hash_keys_table(
                    gathered_table, join_state->n_keys, SEED_HASH_JOIN, false,
                    /*global_dict_needed*/ false);

                join_state->ReserveBuildTable(gathered_table);
                if (use_bloom_filter) {
                    join_state->global_bloom_filter->AddAll(
                        batch_hashes_partition, 0, gathered_table->nrows());
                }
                join_state->AppendBuildBatch(gathered_table, batch_hashes_join,
                                             batch_hashes_partition);
                batch_hashes_partition.reset();
                batch_hashes_join.reset();
                gathered_table.reset();
            }
        }
    }
    if (shuffle_this_iter(join_state->build_parallel, is_last,
                          join_state->build_shuffle_buffer.data_table,
                          join_state->build_iter,
                          join_state->shuffle_sync_iter)) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->build_shuffle_buffer.data_table;
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
        join_state->build_shuffle_buffer.Reset();

        // unify dictionaries to allow consistent hashing and fast key
        // comparison using indices
        new_data = join_state->UnifyBuildTableDictionaryArrays(new_data);
        // compute hashes of the new data
        std::shared_ptr<uint32_t[]> batch_hashes_join =
            hash_keys_table(new_data, join_state->n_keys, SEED_HASH_JOIN,
                            join_state->build_parallel, false);
        dict_hashes = join_state->GetDictionaryHashesForKeys();
        // NOTE: Partition hashes need to be consistent across ranks, so need to
        // use dictionary hashes. Since we are using dictionary hashes, we don't
        // need dictionaries to be global. In fact, hash_keys_table will ignore
        // the dictionaries entirely when dict_hashes are provided.
        std::shared_ptr<uint32_t[]> batch_hashes_partition =
            hash_keys_table(new_data, join_state->n_keys, SEED_HASH_PARTITION,
                            join_state->build_parallel,
                            /*global_dict_needed*/ false, dict_hashes);

        // Add new batch of data to partitions (bulk insert)
        join_state->ReserveBuildTable(new_data);
        join_state->AppendBuildBatch(new_data, batch_hashes_join,
                                     batch_hashes_partition);
        batch_hashes_join.reset();
        batch_hashes_partition.reset();
    }
    // Finalize build on all partitions if it's the last input batch.
    if (is_last) {
        if (use_bloom_filter && join_state->build_parallel) {
            // Make the bloom filter global.
            join_state->global_bloom_filter->union_reduction();
        }
        // TODO: clear build_shuffle_buffer memory
        join_state->FinalizeBuild();
    }
    join_state->build_iter++;
}

/**
 * @brief consume probe table batch in streaming join.
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the build
 * side.
 * @param probe_kept_cols Which columns to generate in the output on the probe
 * side.
 * @param is_last is last batch
 * @param parallel parallel flag
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool use_bloom_filter>
void join_probe_consume_batch(HashJoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              const std::vector<uint64_t> build_kept_cols,
                              const std::vector<uint64_t> probe_kept_cols,
                              const bool is_last) {
    if (join_state->probe_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "join_probe_consume_batch: Received non-empty in_table after "
                "the probe was already finalized!");
        }
        // No processing left.
        return;
    }

    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Update active partition state (temporarily) for hashing and
    // comparison functions.
    std::shared_ptr<JoinPartition>& active_partition =
        join_state->partitions[0];

    // Unify dictionaries to allow consistent hashing and fast key comparison
    // using indices
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

    join_state->probe_shuffle_buffer.ReserveTable(in_table);
    // XXX This is temporary until we have proper buffers.
    join_state->ReserveProbeTableForInactivePartitions(in_table);

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
        // If just build_parallel = False then we have a broadcast join on the
        // build side. So process all rows.
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
            // We use batch_hashes_partition to use consistent hashing across
            // ranks for dict-encoded string arrays
            if (!join_state->global_bloom_filter->Find(
                    batch_hashes_partition[i_row])) {
                if (probe_table_outer) {
                    // Add unmatched rows from probe table to output table
                    build_idxs.push_back(-1);
                    probe_idxs.push_back(i_row);
                }
                continue;
            }
        }
        if (process_on_rank) {
            // TODO Add a fast path without this check for the single partition
            // case.
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
    join_state->probe_shuffle_buffer.AppendBatch(
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
                          join_state->probe_shuffle_buffer.data_table,
                          join_state->probe_iter,
                          join_state->shuffle_sync_iter)) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->probe_shuffle_buffer.data_table;
        // NOTE: shuffle hashes need to be consistent with partition hashes
        // above
        std::shared_ptr<uint32_t[]> shuffle_hashes = hash_keys_table(
            shuffle_table, join_state->n_keys, SEED_HASH_PARTITION,
            shuffle_possible, false, dict_hashes);
        // make dictionaries global for shuffle
        for (size_t i = 0; i < shuffle_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = shuffle_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr, /*is_parallel*/ true);
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
        join_state->probe_shuffle_buffer.Reset();

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

        // XXX This is temporary until we have proper buffers.
        join_state->ReserveProbeTableForInactivePartitions(new_data);

        append_to_probe_inactive_partition.resize(new_data->nrows(), false);
        for (size_t i_row = 0; i_row < new_data->nrows(); i_row++) {
            if (use_bloom_filter) {
                // We use partition hashes to use consistent hashing across
                // ranks for dict-encoded string arrays
                if (!join_state->global_bloom_filter->Find(
                        batch_hashes_partition[i_row])) {
                    if (probe_table_outer) {
                        // Add unmatched rows from probe table to output
                        // table
                        build_idxs.push_back(-1);
                        probe_idxs.push_back(i_row);
                    }
                    continue;
                }
            }
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
        join_state->FinalizeProbeForInactivePartitions<
            build_table_outer, probe_table_outer, non_equi_condition>(
            build_kept_cols, probe_kept_cols);
        // Finalize the probe side
        join_state->FinalizeProbe();
    }
}

/**
 * @brief Initialize a new streaming join state for specified array types and
 * number of keys (called from Python)
 *
 * @param arr_c_types array types of build table columns (Bodo_CTypes ints)
 * @param n_arrs number of build table columns
 * @param n_keys number of join keys
 * @param build_table_outer whether to produce left outer join
 * @param probe_table_outer whether to produce right outer join
 * @param cond_func pointer to function that evaluates non-equality condition.
 *                  If there is no non-equality condition, this should be NULL.
 * @param build_parallel whether the build table is distributed
 * @param probe_parallel whether the probe table is distributed
 * @param output_batch_size Batch size for reading output.
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(
    int8_t* build_arr_c_types, int8_t* build_arr_array_types, int n_build_arrs,
    int8_t* probe_arr_c_types, int8_t* probe_arr_array_types, int n_probe_arrs,
    uint64_t n_keys, bool build_table_outer, bool probe_table_outer,
    cond_expr_fn_t cond_func, bool build_parallel, bool probe_parallel,
    int64_t output_batch_size, uint64_t shuffle_sync_iter) {
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
            probe_parallel, output_batch_size);
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
        probe_parallel, output_batch_size, shuffle_sync_iter);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param join_state_ join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void join_build_consume_batch_py_entry(JoinState* join_state_,
                                       table_info* in_table, bool is_last) {
    // nested loop join is required if there are no equality keys
    if (join_state_->n_keys == 0) {
        nested_loop_join_build_consume_batch_py_entry(
            (NestedLoopJoinState*)join_state_, in_table, is_last);
        return;
    }

    HashJoinState* join_state = (HashJoinState*)join_state_;

    try {
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
        join_build_consume_batch(join_state,
                                 std::unique_ptr<table_info>(in_table),
                                 has_bloom_filter, is_last);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Python wrapper to consume probe table batch and produce output table
 * batch
 *
 * @param join_state_ join state pointer
 * @param in_table probe table batch
 * @param kept_build_col_nums indices of kept columns in build table
 * @param num_kept_build_cols Length of kept_build_col_nums
 * @param kept_probe_col_nums indices of kept columns in probe table
 * @param num_kept_probe_cols Length of kept_probe_col_nums
 * @param[out] total_rows Store the number of rows in the output batch in case
 *        all columns are dead.
 * @param is_last is last batch
 * @return table_info* output table batch
 */
table_info* join_probe_consume_batch_py_entry(
    JoinState* join_state_, table_info* in_table, uint64_t* kept_build_col_nums,
    int64_t num_kept_build_cols, uint64_t* kept_probe_col_nums,
    int64_t num_kept_probe_cols, int64_t* total_rows, bool is_last,
    bool* out_is_last) {
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
            nested_loop_join_probe_consume_batch(
                (NestedLoopJoinState*)join_state_, std::move(input_table),
                std::move(build_kept_cols), std::move(probe_kept_cols),
                is_last);
        } else {
            HashJoinState* join_state = (HashJoinState*)join_state_;

#ifndef CONSUME_PROBE_BATCH
#define CONSUME_PROBE_BATCH(build_table_outer, probe_table_outer,              \
                            has_non_equi_cond, use_bloom_filter,               \
                            build_table_outer_exp, probe_table_outer_exp,      \
                            has_non_equi_cond_exp, use_bloom_filter_exp)       \
    if (build_table_outer == build_table_outer_exp &&                          \
        probe_table_outer == probe_table_outer_exp &&                          \
        has_non_equi_cond == has_non_equi_cond_exp &&                          \
        use_bloom_filter == use_bloom_filter_exp) {                            \
        join_probe_consume_batch<build_table_outer_exp, probe_table_outer_exp, \
                                 has_non_equi_cond_exp, use_bloom_filter_exp>( \
            join_state, std::move(input_table), std::move(build_kept_cols),    \
            std::move(probe_kept_cols), is_last);                              \
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
        auto [out_table, chunk_size] =
            join_state_->output_buffer->PopChunk(/*force_return*/ is_last);
        *total_rows = chunk_size;
        // This is the last output if we've already seen all input (i.e.
        // is_last) and there's no more output remaining in the output_buffer:
        *out_is_last =
            is_last && join_state_->output_buffer->total_remaining == 0;
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief delete join state (called from Python after probe loop is finished)
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
    return m;
}
