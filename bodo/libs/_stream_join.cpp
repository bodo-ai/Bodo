#include "_stream_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_shuffle.h"

/* --------------------------- Helper Functions --------------------------- */

/**
 * @brief Get the dtypes and arr types from an existing table
 *
 * @param table Reference table
 * @return std::tuple<std::vector<int8_t>, std::vector<int8_t>>
 */
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

    // Determine if NA columns should match. They should always
    // match when populating the hash map with the build table.
    // When comparing the build and probe tables this depends on
    // is_na_equal.
    // TODO: Eliminate groups with NA columns with is_na_equal=False
    // from the hashmap.
    bool set_na_equal = (is_build_A && is_build_B);
    bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB, this->n_keys,
                              set_na_equal);
    return test;
}

/* ------------------------------------------------------------------------ */

/* -------------------------- DictionaryBuilder --------------------------- */

std::shared_ptr<array_info> DictionaryBuilder::UnifyDictionaryArray(
    const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != bodo_array_type::DICT) {
        throw std::runtime_error("UnifyDictionaryArray: DICT array expected");
    }

    std::shared_ptr<array_info> batch_dict = in_arr->child_arrays[0];
    this->dict_buff->ReserveArray(batch_dict);

    // Check/update dictionary hash table and create transpose map
    std::vector<dict_indices_t> transpose_map;
    transpose_map.reserve(batch_dict->length);
    char* data = batch_dict->data1();
    offset_t* offsets = (offset_t*)batch_dict->data2();
    for (size_t i = 0; i < batch_dict->length; i++) {
        // handle nulls in the dictionary
        if (!batch_dict->get_null_bit(i)) {
            transpose_map.emplace_back(-1);
            continue;
        }
        offset_t start_offset = offsets[i];
        offset_t end_offset = offsets[i + 1];
        int64_t len = end_offset - start_offset;
        std::string_view val(&data[start_offset], len);
        // get existing index if already in hash table
        if (this->dict_str_to_ind->contains(val)) {
            dict_indices_t ind = this->dict_str_to_ind->find(val)->second;
            transpose_map.emplace_back(ind);
        } else {
            // insert into hash table if not exists
            dict_indices_t ind = this->dict_str_to_ind->size();
            // TODO: remove std::string() after upgrade to C++23
            (*(this->dict_str_to_ind))[std::string(val)] = ind;
            transpose_map.emplace_back(ind);
            this->dict_buff->AppendRow(batch_dict, i);
            if (this->is_key) {
                uint32_t hash;
                hash_string_32(&data[start_offset], (const int)len,
                               SEED_HASH_PARTITION, &hash);
                this->dict_hashes->emplace_back(hash);
            }
        }
    }

    // create output batch array with common dictionary and new transposed
    // indices
    const std::shared_ptr<array_info>& in_indices_arr = in_arr->child_arrays[1];
    std::shared_ptr<array_info> out_indices_arr =
        alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
    dict_indices_t* in_inds = (dict_indices_t*)in_indices_arr->data1();
    dict_indices_t* out_inds = (dict_indices_t*)out_indices_arr->data1();
    for (size_t i = 0; i < in_indices_arr->length; i++) {
        if (!in_indices_arr->get_null_bit(i)) {
            out_indices_arr->set_null_bit(i, false);
            out_inds[i] = -1;
            continue;
        }
        dict_indices_t ind = transpose_map[in_inds[i]];
        out_inds[i] = ind;
        if (ind == -1) {
            out_indices_arr->set_null_bit(i, false);
        } else {
            out_indices_arr->set_null_bit(i, true);
        }
    }
    return std::make_shared<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
        out_indices_arr->length, std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>(
            {dict_buff->data_array, out_indices_arr}),
        0, 0, 0, false,
        /*_has_deduped_local_dictionary=*/true, false);
}

std::shared_ptr<bodo::vector<uint32_t>>
DictionaryBuilder::GetDictionaryHashes() {
    return this->dict_hashes;
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
    for (size_t i_row = 0; i_row < build_table_buffer.data_table->nrows();
         i_row++) {
        if (new_part1->is_in_partition(
                build_table_partitioning_hashes[i_row])) {
            // Copy to partition 1 buffers:
            new_part1->build_table_buffer.AppendRow(
                this->build_table_buffer.data_table, i_row);
            new_part1->build_table_join_hashes.push_back(
                this->build_table_join_hashes[i_row]);
        } else {
            // Copy to partition 2 buffers:
            new_part2->build_table_buffer.AppendRow(
                this->build_table_buffer.data_table, i_row);
            new_part2->build_table_join_hashes.push_back(
                this->build_table_join_hashes[i_row]);
        }
    }

    // Splitting happens at build time, so the probe buffers should
    // be empty.

    return {new_part1, new_part2};
}

void JoinPartition::BuildHashTable() {
    for (size_t i_row = this->curr_build_size;
         i_row < this->build_table_buffer.data_table->nrows(); i_row++) {
        this->build_table.emplace(this->curr_build_size, this->curr_build_size);
        this->curr_build_size++;
    }
}

void JoinPartition::ReserveBuildTable(
    const std::shared_ptr<table_info>& in_table) {
    this->build_table_join_hashes.reserve(this->build_table_join_hashes.size() +
                                          in_table->nrows());
    this->build_table_buffer.ReserveTable(in_table);
}

void JoinPartition::ReserveProbeTable(
    const std::shared_ptr<table_info>& in_table) {
    this->probe_table_buffer_join_hashes.reserve(
        this->probe_table_buffer_join_hashes.size() + in_table->nrows());
    this->probe_table_buffer.ReserveTable(in_table);
}

// XXX Currently the implementation for the active and inactive case are
// very similar, but this will change in the future, hence the template
// variable.
template <bool is_active>
void JoinPartition::AppendBuildRow(const std::shared_ptr<table_info>& in_table,
                                   int64_t row_ind, const uint32_t& join_hash) {
    this->build_table_join_hashes.emplace_back(join_hash);
    this->build_table_buffer.AppendRow(in_table, row_ind);
    if (is_active) {
        this->build_table.emplace(this->curr_build_size, this->curr_build_size);
        this->curr_build_size++;
    }
}

void JoinPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    this->build_table_join_hashes.insert(this->build_table_join_hashes.end(),
                                         join_hashes.get(),
                                         join_hashes.get() + in_table->nrows());
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        this->build_table_buffer.AppendRow(in_table, i_row);
        this->build_table.emplace(this->curr_build_size, this->curr_build_size);
        this->curr_build_size++;
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
            this->build_table_buffer.data_table->nrows(), false);
    }
    // TODO Unpin all state
}

void JoinPartition::AppendInactiveProbeRow(
    const std::shared_ptr<table_info>& in_table, int64_t row_ind,
    const uint32_t& join_hash) {
    this->probe_table_buffer_join_hashes.push_back(join_hash);
    this->probe_table_buffer.AppendRow(in_table, row_ind);
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
    auto range = partition->build_table.equal_range(-i_row - 1);
    if (probe_table_outer && range.first == range.second) {
        // Add unmatched rows from probe table to output table
        build_idxs.push_back(-1);
        probe_idxs.push_back(i_row);
        return;
    }
    // Initialize to true for pure hash join so the final branch
    // is non-equality condition only.
    bool has_match = !non_equi_condition;
    for (auto it = range.first; it != range.second; ++it) {
        if (non_equi_condition) {
            // Check for matches with the non-equality portion.
            bool match =
                cond_func(build_table_info_ptrs.data(),
                          probe_table_info_ptrs.data(), build_col_ptrs.data(),
                          probe_col_ptrs.data(), build_null_bitmaps.data(),
                          probe_null_bitmaps.data(), it->second, i_row);
            if (!match) {
                continue;
            }
            has_match = true;
        }
        if (build_table_outer) {
            partition->build_table_matched[it->second] = true;
        }
        build_idxs.push_back(it->second);
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
 *
 * @param partition Join partition to produce the output for.
 * @param[in, out] build_idxs Build table indices for the output. This will be
 * updated in place.
 * @param[in, out] probe_idxs Probe table indices for the output. This will be
 * updated in place.
 */
void generate_build_table_outer_rows_for_partition(
    const JoinPartition* partition, bodo::vector<int64_t>& build_idxs,
    bodo::vector<int64_t>& probe_idxs) {
    // Add unmatched rows from build table to output table
    for (size_t i_row = 0; i_row < partition->build_table_matched.size();
         i_row++) {
        if (!partition->build_table_matched[i_row]) {
            build_idxs.push_back(i_row);
            probe_idxs.push_back(-1);
        }
    }
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
JoinPartition::FinalizeProbeForInactivePartition(
    cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
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
        generate_build_table_outer_rows_for_partition(this, build_idxs,
                                                      probe_idxs);
    }

    std::shared_ptr<table_info> build_out_table =
        RetrieveTable(this->build_table_buffer.data_table, build_idxs,
                      build_kept_cols, probe_table_outer);
    // XXX Use std::move instead?
    std::shared_ptr<table_info> probe_out_table = RetrieveTable(
        this->probe_table, probe_idxs, probe_kept_cols, build_table_outer);
    build_idxs.clear();
    probe_idxs.clear();
    this->probe_table = nullptr;
    this->probe_table_hashes = nullptr;

    // TODO Unpin/Free all state

    return std::make_tuple(build_out_table, probe_out_table);
}

/* ------------------------------------------------------------------------ */

/* ---------------------------- HashJoinState ----------------------------- */

HashJoinState::HashJoinState(std::vector<int8_t> build_arr_c_types,
                             std::vector<int8_t> build_arr_array_types,
                             std::vector<int8_t> probe_arr_c_types,
                             std::vector<int8_t> probe_arr_array_types,
                             uint64_t n_keys_, bool build_table_outer_,
                             bool probe_table_outer_, cond_expr_fn_t _cond_func,
                             size_t max_partition_depth_)
    : JoinState(n_keys_, build_table_outer_, probe_table_outer_, _cond_func),
      max_partition_depth(max_partition_depth_),
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
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING, 0, 0);
            this->key_dict_builders[i] =
                std::make_shared<DictionaryBuilder>(std::move(dict), true);
        } else {
            this->key_dict_builders[i] = nullptr;
        }
    }

    // Create dictionary builders for non-key columns in build table:
    for (size_t i = this->n_keys; i < build_arr_array_types.size(); i++) {
        if (build_arr_array_types[i] == bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING, 0, 0);
            this->build_table_non_key_dict_builders.emplace_back(
                std::make_shared<DictionaryBuilder>(dict, false));
        } else {
            this->build_table_non_key_dict_builders.emplace_back(nullptr);
        }
    }

    // Create dictionary builders for non-key columns in probe table:
    for (size_t i = this->n_keys; i < probe_arr_array_types.size(); i++) {
        if (probe_arr_array_types[i] == bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING, 0, 0);
            this->probe_table_non_key_dict_builders.emplace_back(
                std::make_shared<DictionaryBuilder>(dict, false));
        } else {
            this->probe_table_non_key_dict_builders.emplace_back(nullptr);
        }
    }

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(), this->key_dict_builders.begin(),
        this->key_dict_builders.end());
    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(),
        this->build_table_non_key_dict_builders.begin(),
        this->build_table_non_key_dict_builders.end());

    this->probe_table_dict_builders.insert(
        this->probe_table_dict_builders.end(), this->key_dict_builders.begin(),
        this->key_dict_builders.end());
    this->probe_table_dict_builders.insert(
        this->probe_table_dict_builders.end(),
        this->probe_table_non_key_dict_builders.begin(),
        this->probe_table_non_key_dict_builders.end());

    this->build_shuffle_buffer =
        TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                         this->build_table_dict_builders);
    this->probe_shuffle_buffer =
        TableBuildBuffer(probe_arr_c_types, probe_arr_array_types,
                         this->probe_table_dict_builders);

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

void HashJoinState::ReserveBuildTable(
    const std::shared_ptr<table_info>& in_table) {
    // XXX This currently over allocates, so will need to be adjusted based
    // on whether or not it's an active partition or not.
    for (auto& partition : this->partitions) {
        partition->ReserveBuildTable(in_table);
    }
}

void HashJoinState::AppendBuildRow(const std::shared_ptr<table_info>& in_table,
                                   int64_t row_ind, const uint32_t& join_hash,
                                   const uint32_t& partitioning_hash) {
    // If there's a single partition, we can skip the check entirely,
    // else we check if the row is in the active partition.
    if (this->partitions.size() == 1 ||
        this->partitions[0]->is_in_partition(partitioning_hash)) {
        this->partitions[0]->AppendBuildRow<true>(in_table, row_ind, join_hash);
    } else {  // If not in active partition, find the correct inactive partition
              // and append it there
        bool found_partition = false;

        // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize partition
        // search by storing a tree representation of the partition space.
        for (size_t i_part = 1;
             (i_part < this->partitions.size() && !found_partition); i_part++) {
            if (this->partitions[i_part]->is_in_partition(partitioning_hash)) {
                this->partitions[i_part]->AppendBuildRow<false>(
                    in_table, row_ind, join_hash);
                found_partition = true;
            }
        }
        if (!found_partition) {
            throw std::runtime_error(
                "HashJoinState::AppendBuildRow: Couldn't find any matching "
                "partition for row!");
        }
    }
}

void HashJoinState::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& join_hashes,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->AppendBuildBatch(in_table, join_hashes,
                                              partitioning_hashes);
    } else {
        for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
            this->AppendBuildRow(in_table, i_row, join_hashes[i_row],
                                 partitioning_hashes[i_row]);
        }
    }
}

void HashJoinState::FinalizeBuild() {
    // TODO Add steps to make sure only one partition is pinned at a time.
    // TODO Add logic to check if partition is too big and needs to be
    // repartitioned.
    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->FinalizeBuild();
    }
}

void HashJoinState::ReserveProbeTableForInactivePartitions(
    const std::shared_ptr<table_info>& in_table) {
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->ReserveProbeTable(in_table);
    }
}

void HashJoinState::AppendProbeRowToInactivePartition(
    const std::shared_ptr<table_info>& in_table, int64_t row_ind,
    const uint32_t& join_hash, const uint32_t& partitioning_hash) {
    bool found_partition = false;

    // TODO (https://bodo.atlassian.net/browse/BSE-472) Optimize partition
    // search by storing a tree representation of the partition space.
    for (size_t i_part = 1;
         (i_part < this->partitions.size() && !found_partition); i_part++) {
        if (this->partitions[i_part]->is_in_partition(partitioning_hash)) {
            this->partitions[i_part]->AppendInactiveProbeRow(in_table, row_ind,
                                                             join_hash);
            found_partition = true;
        }
    }
    if (!found_partition) {
        throw std::runtime_error(
            "HashJoinState::AppendProbeRowToInactivePartition: Couldn't find "
            "any matching partition for row!");
    }
}

template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
HashJoinState::FinalizeProbeForInactivePartitions(
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    if (this->partitions.size() <= 1) {
        return std::make_tuple(nullptr, nullptr);
    }

    std::vector<std::shared_ptr<table_info>> build_out_tables;
    std::vector<std::shared_ptr<table_info>> probe_out_tables;
    build_out_tables.reserve(this->partitions.size() - 1);
    probe_out_tables.reserve(this->partitions.size() - 1);

    for (size_t i = 1; i < this->partitions.size(); i++) {
        auto [build_out_table, probe_out_table] =
            this->partitions[i]
                ->FinalizeProbeForInactivePartition<
                    build_table_outer, probe_table_outer, non_equi_condition>(
                    this->cond_func, build_kept_cols, probe_kept_cols);
        build_out_tables.push_back(build_out_table);
        probe_out_tables.push_back(probe_out_table);
    }

    std::shared_ptr<table_info> build_out_table =
        concat_tables(build_out_tables);
    build_out_tables.clear();
    std::shared_ptr<table_info> probe_out_table =
        concat_tables(probe_out_tables);
    probe_out_tables.clear();

    return std::make_tuple(build_out_table, probe_out_table);
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

std::shared_ptr<table_info> HashJoinState::UnifyBuildTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    return unify_dictionary_arrays_helper(
        in_table, this->build_table_dict_builders, this->n_keys, only_keys);
}

std::shared_ptr<table_info> HashJoinState::UnifyProbeTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    return unify_dictionary_arrays_helper(
        in_table, this->probe_table_dict_builders, this->n_keys, only_keys);
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
                              bool use_bloom_filter, bool is_last,
                              bool parallel) {
    int n_pes, myrank;
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

    // Get hashes of the new batch (different hashes for partitioning and hash
    // table to reduce conflict)
    // NOTE: Partition hashes need to be consistent across ranks so need to use
    // dictionary hashes. Since we are using dictionary hashes, we don't
    // need dictionaries to be global. In fact, hash_keys_table will ignore
    // the dictionaries entirely when dict_hashes are provided.
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        parallel, false, dict_hashes);
    std::shared_ptr<uint32_t[]> batch_hashes_join =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_JOIN, parallel,
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
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank ||
            !parallel) {
            join_state->AppendBuildRow(in_table, i_row,
                                       batch_hashes_join[i_row],
                                       batch_hashes_partition[i_row]);
        } else {
            join_state->build_shuffle_buffer.AppendRow(in_table, i_row);
        }
    }
    batch_hashes_partition.reset();
    batch_hashes_join.reset();
    in_table.reset();

    if (!is_last && join_state->partitions[0]->is_near_full()) {
        join_state->SplitPartition(0);
    }

    if (is_last && parallel) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->build_shuffle_buffer.data_table;
        // NOTE: shuffle hashes need to be consistent with partition hashes
        // above
        std::shared_ptr<uint32_t[]> shuffle_hashes =
            hash_keys_table(shuffle_table, join_state->n_keys,
                            SEED_HASH_PARTITION, parallel, true, dict_hashes);
        // make dictionaries global for shuffle
        for (size_t i = 0; i < shuffle_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = shuffle_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr, parallel);
            }
        }
        mpi_comm_info comm_info_table(shuffle_table->columns);
        comm_info_table.set_counts(shuffle_hashes, parallel);
        std::shared_ptr<table_info> new_data =
            shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                                 comm_info_table, parallel);
        shuffle_hashes.reset();
        // TODO: clear build_shuffle_buffer memory

        // unify dictionaries to allow consistent hashing and fast key
        // comparison using indices
        new_data = join_state->UnifyBuildTableDictionaryArrays(new_data);
        // compute hashes of the new data
        std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
            new_data, join_state->n_keys, SEED_HASH_JOIN, parallel, false);
        dict_hashes = join_state->GetDictionaryHashesForKeys();
        // NOTE: Partition hashes need to be consistent across ranks, so need to
        // use dictionary hashes. Since we are using dictionary hashes, we don't
        // need dictionaries to be global. In fact, hash_keys_table will ignore
        // the dictionaries entirely when dict_hashes are provided.
        std::shared_ptr<uint32_t[]> batch_hashes_partition = hash_keys_table(
            new_data, join_state->n_keys, SEED_HASH_PARTITION, parallel,
            /*global_dict_needed*/ false, dict_hashes);

        // Add new batch of data to partitions (bulk insert)
        join_state->ReserveBuildTable(new_data);
        join_state->AppendBuildBatch(new_data, batch_hashes_join,
                                     batch_hashes_partition);
        batch_hashes_join.reset();
        batch_hashes_partition.reset();
        // Finalize the bloom filter
        if (use_bloom_filter) {
            // Make the bloom filter global.
            join_state->global_bloom_filter->union_reduction();
        }
    }

    // Finalize build on all partitions if it's the last input batch.
    if (is_last) {
        join_state->FinalizeBuild();
    }
}

/**
 * @brief consume probe table batch in streaming join and produce output table
 * batch
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the build
 * side.
 * @param probe_kept_cols Which columns to generate in the output on the probe
 * side.
 * @param is_last is last batch
 * @param parallel parallel flag
 * @return std::shared_ptr<table_info> output table batch
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition, bool use_bloom_filter>
std::shared_ptr<table_info> join_probe_consume_batch(
    HashJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, const bool is_last,
    const bool parallel) {
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

    // Compute join hashes
    std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
        in_table, join_state->n_keys, SEED_HASH_JOIN, parallel, false);
    active_partition->probe_table_hashes = batch_hashes_join.get();

    // Compute partitioning hashes:
    // NOTE: partition hashes need to be consistent across ranks so need to
    // use dictionary hashes
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = join_state->GetDictionaryHashesForKeys();
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_PARTITION,
                        parallel, true, dict_hashes);

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
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        // Check bloom filter
        if (use_bloom_filter) {
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
        if (!parallel ||
            hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank) {
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
                join_state->AppendProbeRowToInactivePartition(
                    in_table, i_row, batch_hashes_join[i_row],
                    batch_hashes_partition[i_row]);
            }
        } else {
            join_state->probe_shuffle_buffer.AppendRow(in_table, i_row);
        }
    }

    // Reset active partition state
    active_partition->probe_table = nullptr;
    active_partition->probe_table_hashes = nullptr;

    // Free hash memory
    batch_hashes_partition.reset();
    batch_hashes_join.reset();

    // create output table using build and probe table indices (columns
    // appended side by side)
    std::shared_ptr<table_info> build_out_table =
        RetrieveTable(active_partition->build_table_buffer.data_table,
                      build_idxs, build_kept_cols, probe_table_outer);
    std::shared_ptr<table_info> probe_out_table = RetrieveTable(
        std::move(in_table), probe_idxs, probe_kept_cols, build_table_outer);
    build_idxs.clear();
    probe_idxs.clear();

    if (is_last && (build_table_outer || parallel)) {
        std::vector<std::shared_ptr<table_info>> build_out_tables(
            {build_out_table});
        std::vector<std::shared_ptr<table_info>> probe_out_tables(
            {probe_out_table});
        if (parallel) {
            // shuffle data of other ranks
            std::shared_ptr<table_info> shuffle_table =
                join_state->probe_shuffle_buffer.data_table;
            // NOTE: shuffle hashes need to be consistent with partition hashes
            // above
            std::shared_ptr<uint32_t[]> shuffle_hashes = hash_keys_table(
                shuffle_table, join_state->n_keys, SEED_HASH_PARTITION,
                parallel, true, dict_hashes);
            // make dictionaries global for shuffle
            for (size_t i = 0; i < shuffle_table->ncols(); i++) {
                std::shared_ptr<array_info> arr = shuffle_table->columns[i];
                if (arr->arr_type == bodo_array_type::DICT) {
                    make_dictionary_global_and_unique(arr, parallel);
                }
            }
            mpi_comm_info comm_info_table(shuffle_table->columns);
            comm_info_table.set_counts(shuffle_hashes, parallel);
            std::shared_ptr<table_info> new_data =
                shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                                     comm_info_table, parallel);
            shuffle_hashes.reset();
            // TODO: clear probe_shuffle_buffer memory

            // unify dictionaries to allow consistent hashing and fast key
            // comparison using indices NOTE: only key arrays need unified since
            // probe_shuffle_buffer isn't used anymore
            new_data = join_state->UnifyProbeTableDictionaryArrays(
                new_data, /*only_keys*/ true);

            // NOTE: partition hashes need to be consistent across ranks so need
            // to use dictionary hashes
            std::shared_ptr<
                bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
                new_data_dict_hashes = join_state->GetDictionaryHashesForKeys();
            // NOTE: Partition hashes need to be consistent across ranks, so
            // need to use dictionary hashes. Since we are using dictionary
            // hashes, we don't need dictionaries to be global. In fact,
            // hash_keys_table will ignore the dictionaries entirely when
            // dict_hashes are provided.
            std::shared_ptr<uint32_t[]> batch_hashes_partition =
                hash_keys_table(
                    new_data, join_state->n_keys, SEED_HASH_PARTITION, parallel,
                    /*global_dict_needed*/ false, new_data_dict_hashes);

            // probe hash table with new data
            std::shared_ptr<uint32_t[]> batch_hashes_join = hash_keys_table(
                new_data, join_state->n_keys, SEED_HASH_JOIN, parallel, false);
            active_partition->probe_table = new_data;
            active_partition->probe_table_hashes = batch_hashes_join.get();

            // Fetch the raw array pointers from the arrays for passing
            // to the non-equijoin condition
            std::vector<array_info*> build_table_info_ptrs,
                probe_table_info_ptrs;
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
                if (active_partition->is_in_partition(
                        batch_hashes_partition[i_row])) {
                    handle_probe_input_for_partition<build_table_outer,
                                                     probe_table_outer,
                                                     non_equi_condition>(
                        join_state->cond_func, active_partition.get(), i_row,
                        build_idxs, probe_idxs, build_table_info_ptrs,
                        probe_table_info_ptrs, build_col_ptrs, probe_col_ptrs,
                        build_null_bitmaps, probe_null_bitmaps);
                } else {
                    join_state->AppendProbeRowToInactivePartition(
                        new_data, i_row, batch_hashes_join[i_row],
                        batch_hashes_partition[i_row]);
                }
            }

            // Reset active partition state
            active_partition->probe_table_hashes = nullptr;
            active_partition->probe_table = nullptr;
            batch_hashes_join.reset();
            batch_hashes_partition.reset();
            build_out_tables.emplace_back(
                RetrieveTable(active_partition->build_table_buffer.data_table,
                              build_idxs, build_kept_cols, probe_table_outer));
            build_idxs.clear();
            probe_out_tables.emplace_back(
                RetrieveTable(std::move(new_data), probe_idxs, probe_kept_cols,
                              build_table_outer));
            probe_idxs.clear();
        }

        if (build_table_outer) {
            // Add unmatched rows from build table to output table
            generate_build_table_outer_rows_for_partition(
                active_partition.get(), build_idxs, probe_idxs);
            build_out_tables.emplace_back(
                RetrieveTable(active_partition->build_table_buffer.data_table,
                              build_idxs, build_kept_cols, probe_table_outer));
            build_idxs.clear();
            // Use the dummy probe table since all indices are -1
            probe_out_tables.emplace_back(
                RetrieveTable(join_state->dummy_probe_table, probe_idxs,
                              probe_kept_cols, build_table_outer));
            probe_idxs.clear();
        }

        // Create output table using build and probe table indices (columns
        // appended)

        // append new build data
        build_out_table = concat_tables(build_out_tables);
        build_out_tables.clear();

        // append new probe data
        probe_out_table = concat_tables(probe_out_tables);
        probe_out_tables.clear();
    }

    if (is_last) {
        auto [build_out_table_inactive, probe_out_table_inactive] =
            join_state->FinalizeProbeForInactivePartitions<
                build_table_outer, probe_table_outer, non_equi_condition>(
                build_kept_cols, probe_kept_cols);
        // Since it can return nullptr
        if (build_out_table_inactive && probe_out_table_inactive) {
            std::vector<std::shared_ptr<table_info>> build_out_tables(
                {build_out_table, build_out_table_inactive});
            build_out_table = concat_tables(build_out_tables);
            build_out_tables.clear();
            build_out_table_inactive.reset();

            std::vector<std::shared_ptr<table_info>> probe_out_tables(
                {probe_out_table, probe_out_table_inactive});
            probe_out_table = concat_tables(probe_out_tables);
            probe_out_tables.clear();
            probe_out_table_inactive.reset();
        }
    }

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.insert(out_arrs.end(), build_out_table->columns.begin(),
                    build_out_table->columns.end());
    out_arrs.insert(out_arrs.end(), probe_out_table->columns.begin(),
                    probe_out_table->columns.end());
    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (size_t i = 0; i < arr_c_types.size(); i++) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)arr_array_types[i];
        Bodo_CTypes::CTypeEnum dtype = (Bodo_CTypes::CTypeEnum)arr_c_types[i];

        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, 0, 0));
    }
    return std::make_shared<table_info>(arrays);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table) {
    std::vector<std::shared_ptr<array_info>> arrays;
    for (size_t i = 0; i < table->ncols(); i++) {
        bodo_array_type::arr_type_enum arr_type = table->columns[i]->arr_type;
        Bodo_CTypes::CTypeEnum dtype = table->columns[i]->dtype;
        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, 0, 0));
    }
    return std::make_shared<table_info>(arrays);
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
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(
    int8_t* build_arr_c_types, int8_t* build_arr_array_types, int n_build_arrs,
    int8_t* probe_arr_c_types, int8_t* probe_arr_array_types, int n_probe_arrs,
    uint64_t n_keys, bool build_table_outer, bool probe_table_outer,
    cond_expr_fn_t cond_func) {
    // nested loop join is required if there are no equality keys
    if (n_keys == 0) {
        return new NestedLoopJoinState(
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs),
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            build_table_outer, probe_table_outer, cond_func);
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
        n_keys, build_table_outer, probe_table_outer, cond_func);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param join_state_ join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void join_build_consume_batch_py_entry(JoinState* join_state_,
                                       table_info* in_table, bool is_last,
                                       bool parallel) {
    // nested loop join is required if there are no equality keys
    if (join_state_->n_keys == 0) {
        nested_loop_join_build_consume_batch_py_entry(
            (NestedLoopJoinState*)join_state_, in_table, is_last, parallel);
        return;
    }

    HashJoinState* join_state = (HashJoinState*)join_state_;

    try {
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
        join_build_consume_batch(join_state,
                                 std::unique_ptr<table_info>(in_table),
                                 has_bloom_filter, is_last, parallel);
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
 * @param is_last is last batch
 * @param is_parallel parallel flag
 * @return table_info* output table batch
 */
table_info* join_probe_consume_batch_py_entry(
    JoinState* join_state_, table_info* in_table, uint64_t* kept_build_col_nums,
    int64_t num_kept_build_cols, uint64_t* kept_probe_col_nums,
    int64_t num_kept_probe_cols, bool is_last, bool* out_is_last,
    bool parallel) {
    // nested loop join is required if there are no equality keys
    if (join_state_->n_keys == 0) {
        return nested_loop_join_probe_consume_batch_py_entry(
            (NestedLoopJoinState*)join_state_, in_table, kept_build_col_nums,
            num_kept_build_cols, kept_probe_col_nums, num_kept_probe_cols,
            is_last, out_is_last, parallel);
    }

    HashJoinState* join_state = (HashJoinState*)join_state_;

#ifndef CONSUME_PROBE_BATCH
#define CONSUME_PROBE_BATCH(build_table_outer, probe_table_outer,            \
                            has_non_equi_cond, use_bloom_filter,             \
                            build_table_outer_exp, probe_table_outer_exp,    \
                            has_non_equi_cond_exp, use_bloom_filter_exp)     \
    if (build_table_outer == build_table_outer_exp &&                        \
        probe_table_outer == probe_table_outer_exp &&                        \
        has_non_equi_cond == has_non_equi_cond_exp &&                        \
        use_bloom_filter == use_bloom_filter_exp) {                          \
        out = join_probe_consume_batch<                                      \
            build_table_outer_exp, probe_table_outer_exp,                    \
            has_non_equi_cond_exp, use_bloom_filter_exp>(                    \
            join_state, std::unique_ptr<table_info>(in_table),               \
            std::move(build_kept_cols), std::move(probe_kept_cols), is_last, \
            parallel);                                                       \
    }
#endif

    try {
        // TODO: Actually output out_is_last based on is_last + the state
        // of the output buffer.
        *out_is_last = is_last;
        std::shared_ptr<table_info> out;
        bool contain_non_equi_cond = join_state->cond_func != NULL;

        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
        std::vector<uint64_t> build_kept_cols(
            kept_build_col_nums, kept_build_col_nums + num_kept_build_cols);
        std::vector<uint64_t> probe_kept_cols(
            kept_probe_col_nums, kept_probe_col_nums + num_kept_probe_cols);

        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, true, true, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, true, true, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, true, false, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, true, false, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, false, true, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, false, true, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, false, false, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, true, false, false, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, true, true, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, true, true, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, true, false, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, true, false, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, false, true, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, false, true, false)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, false, false, true)
        CONSUME_PROBE_BATCH(
            join_state->build_table_outer, join_state->probe_table_outer,
            contain_non_equi_cond, has_bloom_filter, false, false, false, false)
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
#undef CONSUME_PROBE_BATCH
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
    SetAttrStringFromVoidPtr(m, nested_loop_join_probe_consume_batch_py_entry);
    return m;
}
