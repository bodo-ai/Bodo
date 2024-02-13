#include "_stream_groupby.h"
#include <fmt/format.h>
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_groupby_common.h"
#include "_memory_budget.h"
#include "_shuffle.h"
#include "_window_compute.h"

#define MAX_SHUFFLE_TABLE_SIZE 50 * 1024 * 1024
#define MAX_SHUFFLE_HASHTABLE_SIZE 50 * 1024 * 1024

/* --------------------------- HashGroupbyTable --------------------------- */

template <bool is_local>
uint32_t HashGroupbyTable<is_local>::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        const bodo::vector<uint32_t>& build_hashes =
            is_local ? this->groupby_partition->build_table_groupby_hashes
                     : this->groupby_shuffle_state->groupby_hashes;

        return build_hashes[iRow];
    } else {
        const std::shared_ptr<uint32_t[]>& in_hashes =
            is_local ? this->groupby_partition->in_table_hashes
                     : this->groupby_shuffle_state->in_table_hashes;
        return in_hashes[-iRow - 1];
    }
}

/* ------------------------------------------------------------------------ */

/* ------------------------- KeyEqualGroupbyTable ------------------------- */

template <bool is_local>
bool KeyEqualGroupbyTable<is_local>::operator()(const int64_t iRowA,
                                                const int64_t iRowB) const {
    const std::shared_ptr<table_info>& build_table =
        is_local ? this->groupby_partition->build_table_buffer->data_table
                 : this->groupby_shuffle_state->table_buffer->data_table;
    const std::shared_ptr<table_info>& in_table =
        is_local ? this->groupby_partition->in_table
                 : this->groupby_shuffle_state->in_table;

    bool is_build_A = iRowA >= 0;
    bool is_build_B = iRowB >= 0;

    size_t jRowA = is_build_A ? iRowA : -iRowA - 1;
    size_t jRowB = is_build_B ? iRowB : -iRowB - 1;

    const std::shared_ptr<table_info>& table_A =
        is_build_A ? build_table : in_table;
    const std::shared_ptr<table_info>& table_B =
        is_build_B ? build_table : in_table;

    bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB, this->n_keys,
                              /*is_na_equal=*/true);
    return test;
}

/* ------------------------------------------------------------------------ */

/* ------------------ Update -> Combine -> Eval helpers ------------------- */
#pragma region  // Update -> Combine -> Eval helpers

/**
 * @brief Helper for 'get_update_table' to get the grouping_infos
 * for creating the update table.
 *
 * @tparam is_acc_case Is the function being called in the
 * accumulating code path. This allows us to specialize certain operations for
 * the large input case.
 * @param in_table Input batch table in the agg case. Entire input table in the
 * acc case.
 * @param n_keys Number of key columns for this groupby operation.
 * @param req_extended_group_info Whether we need to collect extended group
 * information.
 * @param pool Memory pool to use for allocations during the execution of this
 * function. See description of 'get_update_table' for a more detailed
 * explanation.
 * @return std::vector<grouping_info> Grouping Infos for the update step.
 *  This is a vector with a single 'grouping_info' object.
 */
template <bool is_acc_case>
std::vector<grouping_info> get_grouping_infos_for_update_table(
    std::shared_ptr<table_info> in_table, const uint64_t n_keys,
    const bool req_extended_group_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    // Allocate the memory for hashes through the pool.
    std::shared_ptr<uint32_t[]> batch_hashes_groupby =
        bodo::make_shared_arr<uint32_t>(in_table->nrows(), pool);
    // Compute and fill hashes into allocated memory.
    hash_keys_table(batch_hashes_groupby, in_table, n_keys,
                    SEED_HASH_GROUPBY_SHUFFLE, false);

    std::vector<std::shared_ptr<table_info>> tables = {in_table};

    size_t nunique_hashes = 0;
    if (is_acc_case) {
        // In the accumulating code path case, we have the entire input, so
        // it's better to get an actual estimate using HLL.
        // The HLL only uses ~1MiB of memory, so we don't really need it to
        // go through the pool.
        nunique_hashes = get_nunique_hashes(
            batch_hashes_groupby, in_table->nrows(), /*is_parallel*/ false);
    } else {
        // In the case of streaming groupby, we don't need to estimate the
        // number of unique hashes. We can just use the number of rows in the
        // input table since the batches are so small. This has been tested to
        // be faster than estimating the number of unique hashes based on
        // previous batches as well as using HLL.
        nunique_hashes = in_table->nrows();
    }

    std::vector<grouping_info> grp_infos;

    if (req_extended_group_info) {
        // TODO[BSE-578]: set to true when handling cumulative operations that
        // need the list of NA row indexes.
        get_group_info_iterate(tables, batch_hashes_groupby, nunique_hashes,
                               grp_infos, n_keys, /*consider_missing*/ false,
                               /*key_dropna*/ false, /*is_parallel*/ false,
                               pool);
    } else {
        get_group_info(tables, batch_hashes_groupby, nunique_hashes, grp_infos,
                       n_keys, /*check_for_null_keys*/ true,
                       /*key_dropna*/ false, /*is_parallel*/ false, pool);
    }

    // get_group_info_iterate / get_group_info always reset the pointer,
    // so this should be a NOP, but we're adding it just to be safe.
    batch_hashes_groupby.reset();

    grouping_info& grp_info = grp_infos[0];
    grp_info.mode = 1;
    return grp_infos;
}

/**
 * @brief Call groupby update function on new input batch data and return the
 * output update table
 *
 * @tparam is_acc_case Is the function being called in the
 * accumulating code path. This allows us to specialize certain operations for
 * the large input case.
 * @param in_table Input batch table in the agg case. Entire input table in the
 * acc case.
 * @param n_keys Number of key columns for this groupby operation.
 * @param col_sets ColSets to use for the update.
 * @param f_in_offsets Contains the offsets into f_in_cols.
 * @param f_in_cols List of physical column indices for the different
 * aggregation functions.
 * @param req_extended_group_info Whether we need to collect extended group
 * information.
 * @param pool Memory pool to use for allocations during the execution of this
 * function. In the accumulate code path case, this is the operator buffer pool.
 * In the agg case, this is the default BufferPool. This is because in the
 * accumulate case, we call this function on the entire partition, so we need to
 * track and enforce memory usage to be able to re-partition and retry in case
 * of failure. In the agg case, this function is called on an input batch (~4K
 * rows). We can treat the execution on this small batch as basically scratch
 * usage, for which the BufferPool is sufficient and using the
 * Operator-Buffer-Pool (which would enforce thresholds) could be problematic
 * (it would re-partition the partition which has no bearing on this input
 * batch).
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<table_info> Output update table
 */
template <bool is_acc_case>
std::shared_ptr<table_info> get_update_table(
    std::shared_ptr<table_info> in_table, const uint64_t n_keys,
    const std::vector<std::shared_ptr<BasicColSet>>& col_sets,
    const std::vector<int32_t>& f_in_offsets,
    const std::vector<int32_t>& f_in_cols, const bool req_extended_group_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    // Empty function set means drop_duplicates operation, which doesn't require
    // update. Drop-duplicates only goes through the agg path, so this is safe.
    if (col_sets.size() == 0) {
        assert(!is_acc_case);
        return in_table;
    }

    // similar to update() function of GroupbyPipeline:
    // https://github.com/Bodo-inc/Bodo/blob/58f995dec2507a84afefbb27af01d67bd40fabb4/bodo/libs/_groupby.cpp#L546

    // Get the grouping_info:
    std::vector<grouping_info> grp_infos =
        get_grouping_infos_for_update_table<is_acc_case>(
            in_table, n_keys, req_extended_group_info, pool);
    grouping_info& grp_info = grp_infos[0];
    std::vector<std::shared_ptr<table_info>> tables = {in_table};

    size_t num_groups = grp_info.num_groups;
    int64_t update_col_len = num_groups;
    std::shared_ptr<table_info> update_table = std::make_shared<table_info>();
    alloc_init_keys(tables, update_table, grp_infos, n_keys, num_groups, pool,
                    mm);

    for (size_t i = 0; i < col_sets.size(); i++) {
        const std::shared_ptr<BasicColSet>& col_set = col_sets[i];

        // set input columns of ColSet to new batch data
        std::vector<std::shared_ptr<array_info>> input_cols;
        for (size_t input_ind = (size_t)f_in_offsets[i];
             input_ind < (size_t)f_in_offsets[i + 1]; input_ind++) {
            input_cols.push_back(in_table->columns[f_in_cols[input_ind]]);
        }
        col_set->setInCol(input_cols);
        std::vector<std::shared_ptr<array_info>> list_arr;
        // Regarding alloc_out_if_no_combine:
        //   - If this is the ACC case, i.e. there's no combine step,
        //     we don't need to allocate the output column yet.
        //     This is important to be able to use f_running_value_offsets
        //     as is.
        //   - If this is the AGG case, i.e. there's a combine step,
        //     this parameter doesn't matter.
        col_set->alloc_update_columns(update_col_len, list_arr,
                                      /*alloc_out_if_no_combine*/ false, pool,
                                      mm);
        for (auto& e_arr : list_arr) {
            update_table->columns.push_back(e_arr);
        }
        col_set->update(grp_infos, pool, mm);
        col_set->clear();
    }
    return update_table;
}

/**
 * @brief Get group numbers for input table and update build table with new
 * groups if any.
 *
 * @tparam is_local Whether we are updating build table of a local partition or
 * the shuffle build table values
 * @param[in, out] build_table_buffer Table with running values for the groups.
 * This will be appended to in place.
 * @param[in, out] build_hashes Hashes for groups in build_table_buffer. This
 * will be appended to in place.
 * @param[in, out] build_hash_table Hash table for finding group number. We
 * append to this in place.
 * @param[in, out] next_group Next group ID to use for any groups that don't
 * already exist in the build-table. This will be incremented in place.
 * @param n_keys Number of key columns.
 * @param[in, out] grp_info Group info object to update in place for the input
 * batch.
 * @param in_table Input batch table.
 * @param batch_hashes_groupby Hashes of input batch
 * @param i_row Row number of input batch to use
 */
template <bool is_local>
inline void update_groups_helper(
    TableBuildBuffer& build_table_buffer, bodo::vector<uint32_t>& build_hashes,
    grpby_hash_table_t<is_local>& build_hash_table, int64_t& next_group,
    const uint64_t n_keys, grouping_info& grp_info,
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby, size_t i_row) {
    bodo::vector<int64_t>& row_to_group = grp_info.row_to_group;
    // TODO[BSE-578]: update group_to_first_row, group_to_first_row etc. if
    // necessary
    int64_t group;

    if (auto group_iter = build_hash_table.find(-i_row - 1);
        group_iter != build_hash_table.end()) {
        // update existing group
        group = group_iter->second;
    } else {
        // add new group
        build_table_buffer.AppendRowKeys(in_table, i_row, n_keys);
        build_table_buffer.IncrementSizeDataColumns(n_keys);
        build_hashes.emplace_back(batch_hashes_groupby[i_row]);
        group = next_group++;
        build_hash_table[group] = group;
    }
    row_to_group[i_row] = group;
}

/**
 * @brief Call groupby combine function on new update data and aggregate with
 * existing build table
 *
 * @param in_table Input update table
 * @param grp_info Row to group mapping info (for input table)
 * @param build_table Build table with running values. This will be updated in
 * place.
 * @param f_running_value_offsets Contains the offsets into f_in_cols.
 * @param col_sets ColSets to use for the combine.
 * @param init_start_row Starting offset of rows in build table that need output
 * data initialization (created by new groups introduced by this batch)
 */
void combine_input_table_helper(
    std::shared_ptr<table_info> in_table, const grouping_info& grp_info,
    const std::shared_ptr<table_info>& build_table,
    const std::vector<int32_t>& f_running_value_offsets,
    const std::vector<std::shared_ptr<BasicColSet>>& col_sets,
    int64_t init_start_row) {
    for (size_t i_colset = 0; i_colset < col_sets.size(); i_colset++) {
        const std::shared_ptr<BasicColSet>& col_set = col_sets[i_colset];
        std::vector<std::shared_ptr<array_info>> in_update_cols;
        std::vector<std::shared_ptr<array_info>> out_combine_cols;
        for (size_t col_ind = (size_t)f_running_value_offsets[i_colset];
             col_ind < (size_t)f_running_value_offsets[i_colset + 1];
             col_ind++) {
            in_update_cols.push_back(in_table->columns[col_ind]);
            out_combine_cols.push_back(build_table->columns[col_ind]);
        }
        col_set->setUpdateCols(in_update_cols);
        col_set->setCombineCols(out_combine_cols);
        col_set->combine({grp_info}, init_start_row);
        col_set->clear();
    }
}

/**
 * @brief Calls groupby eval() functions of groupby operations on running values
 * to compute final output.
 *
 * @tparam is_acc_case Is the function being called in the
 * accumulating code path. This decides whether we set the update columns
 * or the combine columns in the ColSets. This also decides whether we set the
 * output columns from separate_out_cols.
 * @param f_running_value_offsets Contains the offsets into f_in_cols.
 * @param col_sets ColSets to use for the combine.
 * @param build_table Build table with running values. This may be updated in
 * place.
 * @param n_keys Number of key columns.
 * @param separate_out_cols Output columns for colsets that require separate
 * output columns
 * @param pool Memory pool to use for allocations during the execution of this
 * function. In the accumulate code path case, this is the operator buffer pool.
 * In the agg case, this is the default BufferPool. This is because in the
 * accumulate case, we call this function on the entire partition, so we need to
 * track and enforce memory usage to be able to re-partition and retry in case
 * of failure. In the agg case, this function is called on an input batch (~4K
 * rows). We can treat the execution on this small batch as basically scratch
 * usage, for which the BufferPool is sufficient and using the
 * Operator-Buffer-Pool (which would enforce thresholds) could be problematic
 * (it would re-partition the partition which has no bearing on this input
 * batch).
 * @param mm Memory manager associated with the pool.
 */
template <bool is_acc_case>
std::shared_ptr<table_info> eval_groupby_funcs_helper(
    const std::vector<int32_t>& f_running_value_offsets,
    const std::vector<std::shared_ptr<BasicColSet>>& col_sets,
    std::shared_ptr<table_info> build_table, const uint64_t n_keys,
    std::shared_ptr<table_info> separate_out_cols,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    // TODO(njriasan): Move eval computation directly into the output
    // buffer.
    std::shared_ptr<table_info> out_table = std::make_shared<table_info>();
    // Add the key columns to the output table:
    out_table->columns.assign(build_table->columns.begin(),
                              build_table->columns.begin() + n_keys);

    size_t sep_col_idx = 0;
    for (size_t i_colset = 0; i_colset < col_sets.size(); i_colset++) {
        const std::shared_ptr<BasicColSet>& col_set = col_sets[i_colset];
        if (is_acc_case) {
            // In the ACC case, the ColSets are created with "do_combine=false",
            // so we need to set the update columns and not the combine columns.
            // We don't need to call setOutputColumn() either since it
            // allocates the output column through the pool automatically if
            // needed.
            std::vector<std::shared_ptr<array_info>> out_update_cols;
            for (size_t col_ind = (size_t)f_running_value_offsets[i_colset];
                 col_ind < (size_t)f_running_value_offsets[i_colset + 1];
                 col_ind++) {
                out_update_cols.push_back(build_table->columns[col_ind]);
            }
            col_set->setUpdateCols(out_update_cols);
        } else {
            // In the AGG case, we call eval on running values that have been
            // incrementally combined over several input batches:
            std::vector<std::shared_ptr<array_info>> out_combine_cols;
            for (size_t col_ind = (size_t)f_running_value_offsets[i_colset];
                 col_ind < (size_t)f_running_value_offsets[i_colset + 1];
                 col_ind++) {
                out_combine_cols.push_back(build_table->columns[col_ind]);
            }
            col_set->setCombineCols(out_combine_cols);
            // Set the separate out col if this col-set needs it:
            if (col_set->getSeparateOutputColumnType().size() != 0) {
                col_set->setOutputColumn(
                    separate_out_cols->columns[sep_col_idx]);
                sep_col_idx++;
            }
        }

        // calling eval() doesn't require grouping info.
        // TODO(ehsan): refactor eval not take grouping info input.
        grouping_info dummy_grp_info;
        col_set->eval(dummy_grp_info, pool, mm);
        const std::vector<std::shared_ptr<array_info>> out_cols =
            col_set->getOutputColumns();
        out_table->columns.insert(out_table->columns.end(), out_cols.begin(),
                                  out_cols.end());
        col_set->clear();
    }
    return out_table;
}

#pragma endregion  // Update -> Combine -> Eval helpers
/* ------------------------------------------------------------------------ */

/* --------------------- Min Row-Number Filter Helpers -------------------- */
#pragma region  // Min Row-Number Filter Helpers

/**
 * @brief Helper function to validate that the GroupbyState arguments
 * are as expected in the MRNF case.
 *
 * @param ftypes There should be a single function.
 * @param f_in_cols Must include all columns expect the partitioning columns.
 * @param f_in_offsets Must be length 2 and must be [0, f_in_cols.size()].
 * @param mrnf_sort_asc Must be the same length as mrnf_sort_na.
 * @param mrnf_sort_na Must be the same length as mrnf_sort_asc.
 * @param mrnf_sort_cols_to_keep Must be the same length as mrnf_sort_asc.
 * @param mrnf_part_cols_to_keep Must be of length n_keys.
 * @param in_arr_n_cols Number of columns in the input table.
 * @param n_keys Number of partitioning keys.
 * @param caller Name of the caller function. Used in runtime-errors.
 */
void validate_mrnf_args(const std::vector<int32_t>& ftypes,
                        const std::vector<int32_t>& f_in_cols,
                        const std::vector<int32_t>& f_in_offsets,
                        const std::vector<bool>& mrnf_sort_asc,
                        const std::vector<bool>& mrnf_sort_na,
                        const std::vector<bool>& mrnf_sort_cols_to_keep,
                        const std::vector<bool>& mrnf_part_cols_to_keep,
                        size_t in_arr_n_cols, uint64_t n_keys,
                        std::string caller) {
    if (ftypes.size() != 1) {
        throw std::runtime_error(
            fmt::format("{}: Min Row-Number Filter cannot be "
                        "combined with other aggregate functions.",
                        caller));
    }
    if ((mrnf_sort_asc.size() == 0) ||
        (mrnf_sort_asc.size() != mrnf_sort_na.size()) ||
        (mrnf_sort_asc.size() != mrnf_sort_cols_to_keep.size())) {
        throw std::runtime_error(
            fmt::format("{}: Min Row-Number Filter arguments "
                        "are not the correct size.",
                        caller));
    }
    if (mrnf_part_cols_to_keep.size() != n_keys) {
        throw std::runtime_error(
            fmt::format("{}: Size of bitmask "
                        "specifying partition columns to retain is not the "
                        "correct size. Expected {} but got {} instead.",
                        caller, n_keys, mrnf_part_cols_to_keep.size()));
    }
    // f_in_cols should be:
    //   [<n_keys>, <n_keys+1>, ..., <in_arr_n_cols-1>]
    if (f_in_cols.size() != (in_arr_n_cols - n_keys)) {
        throw std::runtime_error(
            fmt::format("{}: f_in_cols expected to "
                        "have {} entries in the Min Row-Number Filter "
                        "case, but got {} instead.",
                        caller, (in_arr_n_cols - n_keys), f_in_cols.size()));
    }
    // f_in_offsets should be: [0, <f_in_cols.size()>]
    if ((f_in_offsets.size() != 2) || (f_in_offsets[0] != 0) ||
        (f_in_offsets[1] != static_cast<int32_t>(f_in_cols.size()))) {
        throw std::runtime_error(
            fmt::format("{}: f_in_offsets is not as expected "
                        "for Min Row-Number Filter!",
                        caller));
    }
}

/**
 * @brief Helper function to get a bitmask specifying the columns
 * to keep in the output in the Min Row-Number Filter case.
 * We assume that the columns are in the order:
 * - Partition columns
 * - Order-By columns
 * - Remaining columns
 * We will keep the partition columns corresponding to
 * true bits in the mrnf_part_cols_to_keep bitmask.
 * Similarly for the order-by columns using mrnf_sort_cols_to_keep.
 * All remaining columns will be retained.
 * The output is essentially:
 * mrnf_part_cols_to_keep.extend(mrnf_sort_cols_to_keep).extend(
 *  [True] * (n_cols - len(mrnf_part_cols_to_keep) -
 *  len(mrnf_sort_cols_to_keep))
 * )
 *
 * @param mrnf_part_cols_to_keep Bitmask of the partition columns to retain.
 * @param mrnf_sort_cols_to_keep Bitmask of the order-by columns to retain.
 * @param n_cols Total number of columns.
 * @return std::vector<bool> Combined bitmask of the columns to retain.
 */
std::vector<bool> get_mrnf_cols_to_keep_bitmask(
    const std::vector<bool>& mrnf_part_cols_to_keep,
    const std::vector<bool>& mrnf_sort_cols_to_keep, const size_t n_cols) {
    std::vector<bool> cols_to_keep(n_cols, false);
    // Set bitmask to true for partition columns to keep based on
    // mrnf_part_cols_to_keep.
    for (size_t i = 0; i < mrnf_part_cols_to_keep.size(); i++) {
        if (mrnf_part_cols_to_keep[i]) {
            cols_to_keep[i] = true;
        }
    }
    // Set bitmask to true for order-by columns to keep based on
    // mrnf_sort_cols_to_keep.
    for (size_t i = 0; i < mrnf_sort_cols_to_keep.size(); i++) {
        if (mrnf_sort_cols_to_keep[i]) {
            cols_to_keep[mrnf_part_cols_to_keep.size() + i] = true;
        }
    }
    // Set bitmask to true for all the remaining columns.
    for (size_t i =
             (mrnf_part_cols_to_keep.size() + mrnf_sort_cols_to_keep.size());
         i < n_cols; i++) {
        cols_to_keep[i] = true;
    }
    return cols_to_keep;
}

#pragma endregion  // Min Row-Number Filter Helpers
/* ------------------------------------------------------------------------ */

/* --------------------------- GroupbyPartition --------------------------- */
#pragma region  // GroupbyPartition

GroupbyPartition::GroupbyPartition(
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
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm_)
    : build_table_schema(std::move(build_table_schema_)),
      build_table_dict_builders(build_table_dict_builders_),
      build_hash_table(std::make_unique<hash_table_t>(
          0, HashGroupbyTable<true>(this, nullptr),
          KeyEqualGroupbyTable<true>(this, nullptr, n_keys_),
          op_scratch_pool_)),
      build_table_groupby_hashes(op_pool_),
      separate_out_cols_schema(std::move(separate_out_cols_schema_)),
      col_sets(col_sets_),
      f_in_offsets(f_in_offsets_),
      f_in_cols(f_in_cols_),
      f_running_value_offsets(f_running_value_offsets_),
      num_top_bits(num_top_bits_),
      top_bitmask(top_bitmask_),
      n_keys(n_keys_),
      accumulate_before_update(accumulate_before_update_),
      req_extended_group_info(req_extended_group_info_),
      is_active(is_active_),
      op_pool(op_pool_),
      op_mm(op_mm_),
      op_scratch_pool(op_scratch_pool_),
      op_scratch_mm(op_scratch_mm_) {
    if (this->is_active) {
        this->build_table_buffer = std::make_unique<TableBuildBuffer>(
            this->build_table_schema, this->build_table_dict_builders,
            this->op_pool, this->op_mm);
        this->separate_out_cols = std::make_unique<TableBuildBuffer>(
            this->separate_out_cols_schema,
            std::vector<std::shared_ptr<DictionaryBuilder>>(
                this->separate_out_cols_schema->ncols(), nullptr),
            // The out columns can be freed before repartitioning and should
            // therefore be considered scratch.
            this->op_scratch_pool, this->op_scratch_mm);
    } else {
        this->build_table_buffer_chunked =
            std::make_unique<ChunkedTableBuilder>(
                this->build_table_schema, this->build_table_dict_builders,
                INACTIVE_PARTITION_TABLE_CHUNK_SIZE,
                DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
    }
}

inline bool GroupbyPartition::is_in_partition(const uint32_t& hash) const {
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

inline void GroupbyPartition::RebuildHashTableFromBuildBuffer() {
    // First compute the join hashes:
    size_t build_table_nrows = this->build_table_buffer->data_table->nrows();
    size_t hashes_cur_len = this->build_table_groupby_hashes.size();
    size_t n_unhashed_rows = build_table_nrows - hashes_cur_len;
    if (n_unhashed_rows > 0) {
        // Compute hashes for the un-hashed rows.
        // TODO: Do this processing in batches of 4K/batch_size rows (for
        // handling inactive partition case where we will do this for the
        // entire table)!
        std::unique_ptr<uint32_t[]> hashes = hash_keys_table(
            this->build_table_buffer->data_table, this->n_keys,
            SEED_HASH_GROUPBY_SHUFFLE,
            /*is_parallel*/ false,
            /*global_dict_needed*/ false, /*dict_hashes*/ nullptr,
            /*start_row_offset*/ hashes_cur_len);
        // Append the hashes:
        this->build_table_groupby_hashes.insert(
            this->build_table_groupby_hashes.end(), hashes.get(),
            hashes.get() + n_unhashed_rows);
    }

    // Add entries to the hash table. All rows in the build_table_buffer
    // are guaranteed to be unique groups, so we can just map group ->
    // group.
    while (this->next_group < static_cast<int64_t>(build_table_nrows)) {
        (*(this->build_hash_table))[this->next_group] = this->next_group;
        this->next_group++;
    }
}

template <bool is_active>
void GroupbyPartition::UpdateGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby) {
    if (is_active) {
        /// Start "transaction":

        // Idempotent. This will be a NOP unless we're re-trying this step
        // after a partition split.
        this->RebuildHashTableFromBuildBuffer();

        // set state batch input
        this->in_table = in_table;
        this->in_table_hashes = batch_hashes_groupby;

        // Reserve space in buffers for potential new groups. This will be a
        // NOP if we already have sufficient space. Note that if any of the
        // keys/running values are strings, they always go through the
        // accumulate path.
        // TODO[BSE-616]: support variable size output like strings
        this->build_table_buffer->ReserveTable(in_table);
        this->separate_out_cols->ReserveTableSize(in_table->nrows());

        // Fill row group numbers in grouping_info to reuse existing
        // infrastructure.
        // We set group=-1 for rows that don't belong to the current buffer
        // (e.g. row belongs to shuffle buffer but we are processing local
        // buffer) for them to be ignored in combine step later.
        grouping_info grp_info;
        grp_info.row_to_group.resize(in_table->nrows(), -1);
        // Get current size of the buffers to know starting offset of new
        // keys which need output data column initialization.
        // update_groups_helper will update this->next_group in place,
        // so we need to cache it beforehand.
        int64_t init_start_row = this->next_group;

        // Add new groups and get group mappings for input batch. This will
        // make allocations that could invoke the threshold enforcement
        // error.
        for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
            update_groups_helper(
                *(this->build_table_buffer), this->build_table_groupby_hashes,
                *(this->build_hash_table), this->next_group, this->n_keys,
                grp_info, in_table, batch_hashes_groupby, i_row);
        }

        // Increment separate_out_cols size so aggfunc_out_initialize correctly
        // initializes the columns
        this->separate_out_cols->IncrementSize(std::max<uint64_t>(
            this->next_group - this->separate_out_cols->data_table->nrows(),
            (uint64_t)0));

        // Combine existing (and new) keys using the input batch.
        // Since we're not passing in anything that can access the op-pool,
        // this shouldn't make any additional allocations that go through
        // the Operator Pool and hence cannot invoke the threshold
        // enforcement error.
        combine_input_table_helper(
            in_table, grp_info, this->build_table_buffer->data_table,
            this->f_running_value_offsets, this->col_sets, init_start_row);

        /// Commit "transaction". Only update this after all the groups
        /// have been updated and combined and after the hash table,
        /// the build buffer and hashes are all up to date.
        this->build_safely_appended_groups = this->next_group;

        // Reset temporary references
        this->in_table.reset();
        this->in_table_hashes.reset();
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked->AppendBatch(in_table);
    }
}

template <bool is_active>
void GroupbyPartition::UpdateGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
    const std::vector<bool>& append_rows) {
    if (is_active) {
        /// Start "transaction":

        // Idempotent. This will be a NOP unless we're re-trying this step
        // after a partition split.
        this->RebuildHashTableFromBuildBuffer();

        // set state batch input
        this->in_table = in_table;
        this->in_table_hashes = batch_hashes_groupby;

        // Reserve space in buffers for potential new groups. This will be a
        // NOP if we already have sufficient space. Note that if any of the
        // keys/running values are strings, they always go through the
        // accumulate path.
        // TODO[BSE-616]: support variable size output like strings
        this->build_table_buffer->ReserveTable(in_table);
        this->separate_out_cols->ReserveTableSize(in_table->nrows());
        // Fill row group numbers in grouping_info to reuse existing
        // infrastructure.
        // We set group=-1 for rows that don't belong to the current buffer
        // (e.g. row belongs to shuffle buffer but we are processing local
        // buffer) for them to be ignored in combine step later.
        grouping_info grp_info;
        grp_info.row_to_group.resize(in_table->nrows(), -1);
        // Get current size of the buffers to know starting offset of new
        // keys which need output data column initialization.
        // update_groups_helper will update this->next_group in place,
        // so we need to cache it beforehand.
        int64_t init_start_row = this->next_group;

        // Add new groups and get group mappings for input batch. This will
        // make allocations that could invoke the threshold enforcement
        // error.
        for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
            if (append_rows[i_row]) {
                update_groups_helper(*(this->build_table_buffer),
                                     this->build_table_groupby_hashes,
                                     *(this->build_hash_table),
                                     this->next_group, this->n_keys, grp_info,
                                     in_table, batch_hashes_groupby, i_row);
            }
        }

        // Increment separate_out_cols size so aggfunc_out_initialize correctly
        // initializes the columns
        this->separate_out_cols->IncrementSize(std::max<uint64_t>(
            this->next_group - this->separate_out_cols->data_table->nrows(),
            (uint64_t)0));

        // Combine existing (and new) keys using the input batch.
        // Since we're not passing in anything that can access the op-pool,
        // this shouldn't make any additional allocations that go through
        // the Operator Pool and hence cannot invoke the threshold
        // enforcement error.
        combine_input_table_helper(
            in_table, grp_info, this->build_table_buffer->data_table,
            this->f_running_value_offsets, this->col_sets, init_start_row);

        /// Commit "transaction". Only update this after all the groups
        /// have been updated and combined and after the hash table,
        /// the build buffer and hashes are all up to date.
        this->build_safely_appended_groups = this->next_group;

        // Reset temporary references
        this->in_table.reset();
        this->in_table_hashes.reset();
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked->AppendBatch(in_table, append_rows);
    }
}

template <bool is_active>
void GroupbyPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table) {
    if (is_active) {
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        // Now append the rows. This will always
        // succeed since we've
        // reserved space upfront.
        this->build_table_buffer->UnsafeAppendBatch(in_table);
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked->AppendBatch(in_table);
    }
}

template <bool is_active>
void GroupbyPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    if (is_active) {
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        // Now append the rows. This will always
        // succeed since we've
        // reserved space upfront.
        this->build_table_buffer->UnsafeAppendBatch(in_table, append_rows);
    } else {
        // Append into the ChunkedTableBuilder
        this->build_table_buffer_chunked->AppendBatch(in_table, append_rows);
    }
}

template <bool is_active>
std::vector<std::shared_ptr<GroupbyPartition>> GroupbyPartition::SplitPartition(
    size_t num_levels) {
    if (num_levels != 1) {
        throw std::runtime_error(
            "GroupbyPartition::SplitPartition: We currently only support "
            "splitting a partition into 2 at a time.");
    }
    constexpr size_t uint32_bits = sizeof(uint32_t) * CHAR_BIT;
    if (this->num_top_bits >= (uint32_bits - 1)) {
        throw std::runtime_error(
            "Cannot split the partition further. Out of hash bits.");
    }

    // Release hash-table memory.
    this->build_hash_table.reset();
    // Release separate out cols buffer.
    this->separate_out_cols.reset();

    // Ensure that the scratch mem usage is 0.
    // If we're splitting an active partition, all ColSet mem
    // usage should've already been cleared before calling this
    // and we've just cleared the hash-table and separate_out_cols
    // which are all the allocations that go through the scratch
    // memory.
    // If we're splitting an inactive partition, there should
    // be no memory usage from the Op-Pool (main or scratch)
    // anyway.
    assert(this->op_scratch_pool->bytes_pinned() == 0);

    // Get dictionary hashes from the dict-builders of build table.
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
    std::shared_ptr<GroupbyPartition> new_part1 =
        std::make_shared<GroupbyPartition>(
            this->num_top_bits + 1, (this->top_bitmask << 1),
            this->build_table_schema, this->separate_out_cols_schema,
            this->n_keys, this->build_table_dict_builders, this->col_sets,
            this->f_in_offsets, this->f_in_cols, this->f_running_value_offsets,
            is_active, this->accumulate_before_update,
            this->req_extended_group_info, this->op_pool, this->op_mm,
            this->op_scratch_pool, this->op_scratch_mm);

    std::shared_ptr<GroupbyPartition> new_part2 =
        std::make_shared<GroupbyPartition>(
            this->num_top_bits + 1, (this->top_bitmask << 1) + 1,
            this->build_table_schema, this->separate_out_cols_schema,
            this->n_keys, this->build_table_dict_builders, this->col_sets,
            this->f_in_offsets, this->f_in_cols, this->f_running_value_offsets,
            false, this->accumulate_before_update,
            this->req_extended_group_info, this->op_pool, this->op_mm,
            this->op_scratch_pool, this->op_scratch_mm);

    std::vector<bool> append_partition1;
    if (is_active) {
        // In the active case, partition this->build_table_buffer directly

        // Compute partitioning hashes
        std::shared_ptr<uint32_t[]> build_table_partitioning_hashes =
            hash_keys_table(this->build_table_buffer->data_table, this->n_keys,
                            SEED_HASH_PARTITION, false, false, dict_hashes);

        // Put the build data in the new partitions.
        append_partition1.resize(this->build_table_buffer->data_table->nrows(),
                                 false);

        // In the agg case, we will only append the entries until
        // build_safely_appended_groups. If there are more entries in the
        // build_table_buffer, this means we triggered this partition split
        // in the middle of an UpdateGroupsAndCombine step. That means that
        // those entries weren't added safely and will be retried.
        // Therefore, we will skip those entries here (and leave their
        // default to false).
        size_t rows_to_insert =
            this->accumulate_before_update
                ? this->build_table_buffer->data_table->nrows()
                : this->build_safely_appended_groups;
        for (size_t i_row = 0; i_row < rows_to_insert; i_row++) {
            append_partition1[i_row] = new_part1->is_in_partition(
                build_table_partitioning_hashes[i_row]);
        }

        // Calculate number of rows going to 1st new partition
        uint64_t append_partition1_sum = std::accumulate(
            append_partition1.begin(), append_partition1.end(), (uint64_t)0);

        if (!this->accumulate_before_update) {
            // In the AGG case, we also need to populate the hashes.

            // Reserve space in hashes vector. This doesn't inhibit
            // exponential growth since we're only doing it at the start.
            // Future appends will still allow for regular exponential
            // growth.
            new_part1->build_table_groupby_hashes.reserve(
                append_partition1_sum);

            // Copy the hash values to the new active partition. We might
            // not have hashes for every row, so copy over whatever we can.
            // Subsequent RebuildHashTableFromBuildBuffer steps will compute
            // the rest. We drop the hashes that would go to the new
            // inactive partition for now and we will re-compute them later
            // when needed.
            size_t n_hashes_to_copy_over =
                std::min(this->build_safely_appended_groups,
                         this->build_table_groupby_hashes.size());
            for (size_t i_row = 0; i_row < n_hashes_to_copy_over; i_row++) {
                if (append_partition1[i_row]) {
                    new_part1->build_table_groupby_hashes.push_back(
                        this->build_table_groupby_hashes[i_row]);
                }
            }
        }

        // Reserve space for append (append_partition1 already accounts for
        // build_safely_appended_groups)
        new_part1->build_table_buffer->ReserveTable(
            this->build_table_buffer->data_table, append_partition1,
            append_partition1_sum);
        new_part1->build_table_buffer->UnsafeAppendBatch(
            this->build_table_buffer->data_table, append_partition1,
            append_partition1_sum);

        // Reserve space for output columns (NOP in the ACC case since there are
        // no separate out columns)
        new_part1->separate_out_cols->ReserveTableSize(append_partition1_sum);
        new_part1->separate_out_cols->IncrementSize(append_partition1_sum);

        if (!this->accumulate_before_update) {
            // Update safely appended group count for the new active
            // partition in the AGG case.
            new_part1->build_safely_appended_groups = append_partition1_sum;
            // XXX Technically, we could store similar information even in
            // the new inactive partition. That would let us add some number
            // of rows to the build-table directly upfront during
            // ActivatePartition. e.g. In this case, we could tell the
            // inactive partition that the first
            // (build_safely_appended_groups - append_partition1_sum) many
            // rows are guaranteed to be unique. The information could even
            // be propagated in the case we're splitting an inactive
            // partition (keep track of how many of the first
            // build_safely_appended_groups rows are going to the new
            // partitions).
        }

        append_partition1.flip();
        std::vector<bool>& append_partition2 = append_partition1;

        if (!this->accumulate_before_update) {
            // The rows between this->build_safely_appended_groups
            // and this->build_table_buffer.data_table->nrows() shouldn't
            // be copied over to either partition:
            for (size_t i = this->build_safely_appended_groups;
                 i < append_partition2.size(); i++) {
                append_partition2[i] = false;
            }
        }

        new_part2->build_table_buffer_chunked->AppendBatch(
            this->build_table_buffer->data_table, append_partition2);

        // We do not rebuild the hash table here (for new_part1 which is the
        // new active partition). That needs to be handled by the caller.
    } else {
        // In the inactive case, partition build_table_buffer_chunked chunk
        // by chunk
        this->build_table_buffer_chunked->Finalize();
        // Just in case we started the activation and some columns
        // reserved memory during ActivatePartition.
        this->build_table_buffer.reset();
        this->build_table_groupby_hashes.resize(0);
        this->build_table_groupby_hashes.shrink_to_fit();

        while (!this->build_table_buffer_chunked->chunks.empty()) {
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked->PopChunk();
            // Compute partitioning hashes.
            // TODO XXX Allocate the hashes buffer once (set size to CTB's
            // chunk-size) and reuse it across all chunks.
            std::shared_ptr<uint32_t[]> build_table_partitioning_hashes_chunk =
                hash_keys_table(build_table_chunk, this->n_keys,
                                SEED_HASH_PARTITION, false, false, dict_hashes);

            append_partition1.resize(build_table_nrows_chunk, false);
            for (int64_t i_row = 0; i_row < build_table_nrows_chunk; i_row++) {
                append_partition1[i_row] = new_part1->is_in_partition(
                    build_table_partitioning_hashes_chunk[i_row]);
            }

            new_part1->build_table_buffer_chunked->AppendBatch(
                build_table_chunk, append_partition1);

            append_partition1.flip();
            std::vector<bool>& append_partition2 = append_partition1;

            new_part2->build_table_buffer_chunked->AppendBatch(
                build_table_chunk, append_partition2);
        }
    }

    return {new_part1, new_part2};
}

void GroupbyPartition::ClearBuildState() {
    this->build_hash_table.reset();
    this->build_table_groupby_hashes.resize(0);
    this->build_table_groupby_hashes.shrink_to_fit();
    this->build_table_buffer.reset();
    this->separate_out_cols.reset();
    this->in_table.reset();
    this->in_table_hashes.reset();
}

void GroupbyPartition::ActivatePartition() {
    if (this->is_active) {
        return;
    }

    // Finalize the chunked table builder:
    this->build_table_buffer_chunked->Finalize();

    // Initialize build_table_buffer
    this->build_table_buffer = std::make_unique<TableBuildBuffer>(
        this->build_table_schema, this->build_table_dict_builders,
        this->op_pool, this->op_mm);

    // Initialize separate output columns
    // NOTE: separate_out_cols cannot be STRING or DICT arrays.
    this->separate_out_cols = std::make_unique<TableBuildBuffer>(
        this->separate_out_cols_schema,
        std::vector<std::shared_ptr<DictionaryBuilder>>(
            this->separate_out_cols_schema->ncols(), nullptr),
        this->op_scratch_pool, this->op_scratch_mm);

    if (this->accumulate_before_update) {
        /// Concatenate all build chunks into contiguous build buffer

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

        // Free the chunked buffer state entirely since it's not needed anymore.
        this->build_table_buffer_chunked.reset();
    } else {
        // Just call UpdateGroupsAndCombine on each chunk. Note that
        // we cannot pop the chunks since we need to keep them around
        // in case we need to re-partition and retry.
        for (const auto& chunk : *(this->build_table_buffer_chunked)) {
            // By definition, this is a small chunk, so we don't need to
            // track this allocation and can consider this scratch memory
            // usage.
            // TODO XXX Cache the allocation for these hashes by making
            // an allocation for the chunk size (active_chunk_capacity)
            // and reusing that buffer for all chunks.
            std::shared_ptr<uint32_t[]> chunk_hashes_groupby = hash_keys_table(
                chunk, this->n_keys, SEED_HASH_GROUPBY_SHUFFLE, false, false);
            // Treat the partition as active temporarily.
            // This step can fail. If it does, we can repartition and retry
            // safely since the CTB still has all the original data.
            this->UpdateGroupsAndCombine</*is_active*/ true>(
                chunk, chunk_hashes_groupby);
        }

        // If we were able to insert all chunks, we have successfully
        // activated the partition and can now reset the CTB (which will
        // free all the chunks):
        this->build_table_buffer_chunked->Reset();
        this->build_table_buffer_chunked.reset();
    }

    // Mark this partition as activated once we've moved the data
    // from the chunked buffer to a contiguous buffer:
    this->is_active = true;
}

void GroupbyPartition::FinalizeMrnf(
    const std::vector<bool>& mrnf_part_cols_to_keep,
    const std::vector<bool>& mrnf_sort_cols_to_keep,
    const std::shared_ptr<ChunkedTableBuilder>& output_buffer) {
    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    this->ActivatePartition();

    // MRNF always uses the ACC path.
    std::vector<grouping_info> grp_infos =
        get_grouping_infos_for_update_table</*is_acc_case*/ true>(
            this->build_table_buffer->data_table, n_keys,
            this->req_extended_group_info, this->op_scratch_pool);
    grouping_info& grp_info = grp_infos[0];

    // Construct a vector with the order-by columns.
    std::vector<std::shared_ptr<array_info>> orderby_arrs(
        this->build_table_buffer->data_table->columns.begin() + this->n_keys,
        this->build_table_buffer->data_table->columns.begin() + this->n_keys +
            mrnf_sort_cols_to_keep.size());

    // In the MRNF case, there's a single colset.
    assert(this->col_sets.size() == 1);
    const std::shared_ptr<BasicColSet>& col_set = this->col_sets[0];
    std::vector<std::shared_ptr<array_info>> list_arr;
    // Compute the output index column using this colset:
    col_set->setInCol(orderby_arrs);
    col_set->alloc_update_columns(grp_info.num_groups, list_arr,
                                  /*alloc_out_if_no_combine*/ false,
                                  this->op_scratch_pool, this->op_scratch_mm);
    col_set->update(grp_infos, this->op_scratch_pool, this->op_scratch_mm);
    const std::vector<std::shared_ptr<array_info>> out_cols =
        col_set->getOutputColumns();
    const std::shared_ptr<array_info>& idx_col = out_cols[0];
    col_set->clear();

    // Create an insertion bitmask for the output.
    // XXX Use bodo::vector instead?
    std::vector<bool> out_bitmask(this->build_table_buffer->data_table->nrows(),
                                  false);
    for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
        int64_t row_one_idx = getv<int64_t>(idx_col, group_idx);
        out_bitmask[row_one_idx] = true;
    }

    // Use mrnf_part_cols_to_keep and mrnf_sort_cols_to_keep to determine
    // the columns to skip from build_table_buffer.
    size_t n_cols = this->build_table_buffer->data_table->columns.size();
    std::vector<bool> cols_to_keep_bitmask = get_mrnf_cols_to_keep_bitmask(
        mrnf_part_cols_to_keep, mrnf_sort_cols_to_keep, n_cols);
    std::vector<std::shared_ptr<array_info>> cols_to_keep;

    for (size_t i = 0; i < n_cols; i++) {
        if (cols_to_keep_bitmask[i]) {
            cols_to_keep.push_back(
                this->build_table_buffer->data_table->columns[i]);
        }
    }
    std::shared_ptr<table_info> data_table_w_cols_to_keep =
        std::make_shared<table_info>(cols_to_keep);

    // Append this "pruned" table to the output buffer using the bitmask.
    output_buffer->AppendBatch(data_table_w_cols_to_keep, out_bitmask);

    // Since we have added the output to the output buffer, we don't need the
    // build state anymore and can release that memory.
    this->ClearBuildState();
}

std::shared_ptr<table_info> GroupbyPartition::Finalize() {
    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    this->ActivatePartition();

    std::shared_ptr<table_info> out_table;
    if (accumulate_before_update) {
        // NOTE: Since the allocations made during the update and eval steps are
        // not required for re-partitioning (in case there is one), we will use
        // the scratch portion of the pool for them.

        // Get update table with the running values:
        std::shared_ptr<table_info> update_table =
            get_update_table</*is_acc_case*/ true>(
                this->build_table_buffer->data_table, this->n_keys,
                this->col_sets, this->f_in_offsets, this->f_in_cols,
                this->req_extended_group_info, this->op_scratch_pool,
                this->op_scratch_mm);
        // Call eval on these running values to get the final output.
        out_table = eval_groupby_funcs_helper</*is_acc_case*/ true>(
            this->f_running_value_offsets, this->col_sets, update_table,
            this->n_keys, this->separate_out_cols->data_table,
            this->op_scratch_pool, this->op_scratch_mm);
    } else {
        // Note that we don't need to call RebuildHashTableFromBuildBuffer
        // here. If the partition was inactive, ActivatePartition could've
        // failed. However, the partition would've remained inactive, so it
        // will be retried until it succeeds. When it eventually does, we're
        // guaranteed that the hash-table is up to date. If the partition
        // was already active, "ActivatePartition" would've been a NOP and
        // couldn't have failed anyway. The hash-table is guaranteed to be
        // up to date since UpdateGroupsAndCombine always brings it up to
        // date. The only place where the hash table is not up to date is
        // after splitting an active partition. However, that can never
        // happen during Finalize. Even if we somehow end up in a situation
        // where the hash-table is not up to date, that's fine since
        // eval_groupby_funcs_helper doesn't need the hash table at all
        // (assuming the rows in build-table-buffer are all unique, which
        // they should be since this is an active partition).

        // Call eval() on running values to get final output.
        // Since we're not passing in anything that can access the op-pool,
        // this shouldn't make any additional allocations that go through
        // the Operator Pool and hence cannot invoke the threshold
        // enforcement error.
        std::shared_ptr<table_info> combine_data =
            this->build_table_buffer->data_table;
        out_table = eval_groupby_funcs_helper</*is_acc_case*/ false>(
            this->f_running_value_offsets, this->col_sets, combine_data,
            this->n_keys, this->separate_out_cols->data_table);
    }

    // Since we have generated the output, we don't need the build state
    // anymore, so we can release that memory.
    this->ClearBuildState();

    return out_table;
}

#pragma endregion  // GroupbyPartition
/* ------------------------------------------------------------------------ */

/* -------------------- GroupbyIncrementalShuffleState -------------------- */
#pragma region  // GroupbyIncrementalShuffleState

GroupbyIncrementalShuffleState::GroupbyIncrementalShuffleState(
    const std::shared_ptr<bodo::Schema> shuffle_table_schema_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
    int64_t op_id_, const bool nunique_only_)
    : IncrementalShuffleState(shuffle_table_schema_, dict_builders_, n_keys_,
                              curr_iter_, sync_freq_, op_id_),
      hash_table(std::make_unique<shuffle_hash_table_t>(
          0, HashGroupbyTable<false>(nullptr, this),
          KeyEqualGroupbyTable<false>(nullptr, this, this->n_keys))),
      nunique_only(nunique_only_) {}

void GroupbyIncrementalShuffleState::Finalize() {
    this->hash_table.reset();
    this->groupby_hashes.resize(0);
    this->groupby_hashes.shrink_to_fit();
    IncrementalShuffleState::Finalize();
}

std::tuple<
    std::shared_ptr<table_info>,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
    std::shared_ptr<uint32_t[]>>
GroupbyIncrementalShuffleState::GetShuffleTableAndHashes() {
    auto [shuffle_table, dict_hashes, shuffle_hashes] =
        IncrementalShuffleState::GetShuffleTableAndHashes();
    // drop shuffle table duplicate rows if there are a lot of duplicates
    // only possible for nunique-only cases
    if (this->nunique_only) {
        int64_t shuffle_nrows = shuffle_table->nrows();

        // estimate number of uniques using key/value hashes
        std::shared_ptr<uint32_t[]> key_value_hashes =
            std::make_unique<uint32_t[]>(shuffle_nrows);
        // reusing shuffle_hashes for keys to make the initial check cheaper
        // for code path without drop duplicates
        std::memcpy(key_value_hashes.get(), shuffle_hashes.get(),
                    sizeof(uint32_t) * shuffle_nrows);
        for (size_t col = this->n_keys; col < shuffle_table->ncols(); col++) {
            hash_array_combine(key_value_hashes, shuffle_table->columns[col],
                               shuffle_nrows, SEED_HASH_PARTITION,
                               /*global_dict_needed=*/false,
                               /*is_parallel*/ true);
        }
        size_t nunique_keyval_hashes = get_nunique_hashes(
            key_value_hashes, shuffle_nrows, /*is_parallel*/ true);

        // drop duplicates if output will be less than half the size of
        // input (rough heuristic, TODO: tune)
        if ((2 * nunique_keyval_hashes) < static_cast<size_t>(shuffle_nrows)) {
            shuffle_table = drop_duplicates_table_inner(
                shuffle_table, shuffle_table->ncols(), 0, 1, false, false,
                /*drop_duplicates_dict=*/false, key_value_hashes);
            shuffle_hashes = hash_keys_table(
                shuffle_table, this->n_keys, SEED_HASH_PARTITION,
                /*is_parallel*/ true, false, dict_hashes);
        }
    }
    return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes);
}

void GroupbyIncrementalShuffleState::ResetAfterShuffle() {
    if (this->hash_table->get_allocator().size() > MAX_SHUFFLE_HASHTABLE_SIZE) {
        // If the shuffle hash table is too large, reset it.
        // This shouldn't happen often in practice, but is a safeguard.
        this->hash_table.reset();
        this->hash_table = std::make_unique<shuffle_hash_table_t>(
            0, HashGroupbyTable<false>(nullptr, this),
            KeyEqualGroupbyTable<false>(nullptr, this, this->n_keys));
    }
    if (this->groupby_hashes.get_allocator().size() > MAX_SHUFFLE_TABLE_SIZE) {
        // If the shuffle hashes vector is too large, reallocate it to the
        // maximum size
        this->groupby_hashes.resize(MAX_SHUFFLE_TABLE_SIZE / sizeof(uint32_t));
        this->groupby_hashes.shrink_to_fit();
    }
    this->next_group = 0;
    this->hash_table->clear();
    this->groupby_hashes.resize(0);
    IncrementalShuffleState::ResetAfterShuffle();
}

#pragma endregion  // GroupbyIncrementalShuffleState
/* ------------------------------------------------------------------------ */

/* ----------------------------- GroupbyState ----------------------------- */
#pragma region  // GroupbyState

GroupbyState::GroupbyState(const std::unique_ptr<bodo::Schema>& in_schema_,
                           std::vector<int32_t> ftypes,
                           std::vector<int32_t> f_in_offsets_,
                           std::vector<int32_t> f_in_cols_, uint64_t n_keys_,
                           std::vector<bool> mrnf_sort_asc_vec_,
                           std::vector<bool> mrnf_sort_na_pos_,
                           std::vector<bool> mrnf_part_cols_to_keep_,
                           std::vector<bool> mrnf_sort_cols_to_keep_,
                           int64_t output_batch_size_, bool parallel_,
                           int64_t sync_iter_, int64_t op_id_,
                           int64_t op_pool_size_bytes)
    :  // Create the operator buffer pool
      op_pool(std::make_unique<bodo::OperatorBufferPool>(
          op_id_,
          ((op_pool_size_bytes == -1)
               ? static_cast<uint64_t>(
                     bodo::BufferPool::Default()->get_memory_size_bytes() *
                     GROUPBY_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL)
               : op_pool_size_bytes),
          bodo::BufferPool::Default(),
          GROUPBY_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD)),
      op_mm(bodo::buffer_memory_manager(op_pool.get())),
      op_scratch_pool(
          std::make_unique<bodo::OperatorScratchPool>(this->op_pool.get())),
      op_scratch_mm(bodo::buffer_memory_manager(this->op_scratch_pool.get())),
      // Get the max partition depth from env var if set. This is primarily
      // for unit testing purposes. If it's not set, use the default.
      max_partition_depth(std::getenv("BODO_STREAM_GROUPBY_MAX_PARTITION_DEPTH")
                              ? std::atoi(std::getenv(
                                    "BODO_STREAM_GROUPBY_MAX_PARTITION_DEPTH"))
                              : GROUPBY_DEFAULT_MAX_PARTITION_DEPTH),
      n_keys(n_keys_),
      parallel(parallel_),
      output_batch_size(output_batch_size_),
      f_in_offsets(std::move(f_in_offsets_)),
      f_in_cols(std::move(f_in_cols_)),
      mrnf_sort_asc(std::move(mrnf_sort_asc_vec_)),
      mrnf_sort_na(std::move(mrnf_sort_na_pos_)),
      mrnf_part_cols_to_keep(std::move(mrnf_part_cols_to_keep_)),
      mrnf_sort_cols_to_keep(std::move(mrnf_sort_cols_to_keep_)),
      sync_iter(sync_iter_),
      groupby_event("Groupby") {
    // Partitioning is enabled by default:
    bool enable_partitioning = true;

    // Force enable/disable partitioning if env var set. This is
    // primarily for unit testing purposes.
    char* enable_partitioning_env_ =
        std::getenv("BODO_STREAM_GROUPBY_ENABLE_PARTITIONING");
    if (enable_partitioning_env_) {
        if (std::strcmp(enable_partitioning_env_, "0") == 0) {
            enable_partitioning = false;
        } else if (std::strcmp(enable_partitioning_env_, "1") == 0) {
            enable_partitioning = true;
        } else {
            throw std::runtime_error(
                "GroupbyState::GroupbyState: "
                "BODO_STREAM_GROUPBY_ENABLE_PARTITIONING set to "
                "unsupported value: " +
                std::string(enable_partitioning_env_));
        }
    } else if (!this->op_pool->is_spilling_enabled()) {
        // There's no point in repartitioning when spilling is not
        // available anyway.
        enable_partitioning = false;
    }

    if (!enable_partitioning) {
        this->DisablePartitioning();
    }

    if (char* debug_partitioning_env_ =
            std::getenv("BODO_DEBUG_STREAM_GROUPBY_PARTITIONING")) {
        this->debug_partitioning = !std::strcmp(debug_partitioning_env_, "1");
    }

    // Add key column types to running value buffer types (same type as
    // input)
    std::shared_ptr<bodo::Schema> build_table_schema =
        std::make_shared<bodo::Schema>();
    for (size_t i = 0; i < n_keys; ++i) {
        build_table_schema->append_column(in_schema_->column_types[i]->copy());
    }
    std::unique_ptr<bodo::Schema> separate_out_cols_schema =
        std::make_unique<bodo::Schema>();

    // Get offsets of update and combine columns for each function since
    // some functions have multiple update/combine columns
    this->f_running_value_offsets.push_back(n_keys);
    int32_t curr_running_value_offset = n_keys;

    for (size_t i = 0; i < ftypes.size(); i++) {
        int ftype = ftypes[i];
        // NOTE: adding all functions that need accumulating inputs for now
        // but they may not be supported in streaming groupby yet
        // Should be kept in sync with
        // https://github.com/Bodo-inc/Bodo/blob/56c77832aea4c4d5b33fd5cf631b201f4157f73a/BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/rel/core/AggregateBase.kt#L58
        if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::cumsum ||
            ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummin ||
            ftype == Bodo_FTypes::cummax || ftype == Bodo_FTypes::shift ||
            ftype == Bodo_FTypes::transform || ftype == Bodo_FTypes::ngroup ||
            ftype == Bodo_FTypes::window || ftype == Bodo_FTypes::listagg ||
            ftype == Bodo_FTypes::nunique || ftype == Bodo_FTypes::head ||
            ftype == Bodo_FTypes::gen_udf ||
            ftype == Bodo_FTypes::min_row_number_filter) {
            this->accumulate_before_update = true;
        }
        if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::cumsum ||
            ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummin ||
            ftype == Bodo_FTypes::cummax || ftype == Bodo_FTypes::shift ||
            ftype == Bodo_FTypes::transform || ftype == Bodo_FTypes::ngroup ||
            ftype == Bodo_FTypes::window || ftype == Bodo_FTypes::listagg ||
            ftype == Bodo_FTypes::nunique) {
            this->req_extended_group_info = true;
        }
        if (ftype == Bodo_FTypes::min_row_number_filter) {
            this->mrnf_only = true;
        }
    }

    // Validate MRNF arguments in the MRNF case:
    if (this->mrnf_only) {
        validate_mrnf_args(ftypes, this->f_in_cols, this->f_in_offsets,
                           this->mrnf_sort_asc, this->mrnf_sort_na,
                           this->mrnf_sort_cols_to_keep,
                           this->mrnf_part_cols_to_keep, in_schema_->ncols(),
                           this->n_keys, "GroupbyState::GroupbyState");
    }

    // TODO[BSE-578]: handle all necessary ColSet parameters for BodoSQL
    // groupby functions
    std::shared_ptr<array_info> index_col = nullptr;

    // Currently, all SQL aggregations that we support excluding count(*)
    // drop or ignore na values durring computation. Since count(*) maps to
    // size, and skip_na_data has no effect on that aggregation, we can
    // safely set skip_na_data to true for all SQL aggregations. There is an
    // issue to fix this behavior so that use_sql_rules trumps the value of
    // skip_na_data: https://bodo.atlassian.net/browse/BSE-841
    bool skip_na_data = true;
    bool use_sql_rules = true;

    // First, get the input column types for each function.
    std::vector<std::vector<std::shared_ptr<array_info>>> local_input_cols_vec(
        ftypes.size());
    std::vector<std::vector<std::unique_ptr<bodo::DataType>>> in_arr_types_vec(
        ftypes.size());
    for (size_t i = 0; i < ftypes.size(); i++) {
        // Get the input columns, array types, and dtypes for the current
        // function
        std::vector<std::shared_ptr<array_info>>& local_input_cols =
            local_input_cols_vec.at(i);
        std::vector<std::unique_ptr<bodo::DataType>>& in_arr_types =
            in_arr_types_vec.at(i);
        for (size_t logical_input_ind = (size_t)f_in_offsets[i];
             logical_input_ind < (size_t)f_in_offsets[i + 1];
             logical_input_ind++) {
            size_t physical_input_ind = (size_t)f_in_cols[logical_input_ind];
            // set dummy input columns in ColSet since will be replaced by
            // input batches
            local_input_cols.push_back(nullptr);
            in_arr_types.push_back(
                (in_schema_->column_types[physical_input_ind])->copy());
        }
    }

    // Next, perform a check on the running value and output types.
    // If any of them are of type string,
    // set accumulate_before_update to true.
    for (size_t i = 0; i < ftypes.size(); i++) {
        std::vector<std::shared_ptr<array_info>>& local_input_cols =
            local_input_cols_vec.at(i);
        std::vector<std::unique_ptr<bodo::DataType>>& in_arr_types =
            in_arr_types_vec.at(i);
        std::vector<std::unique_ptr<bodo::DataType>> in_arr_types_copy;
        for (const auto& t : in_arr_types) {
            in_arr_types_copy.push_back(t->copy());
        }

        std::unique_ptr<bodo::Schema> running_values_schema =
            this->getRunningValueColumnTypes(
                local_input_cols, std::move(in_arr_types_copy), ftypes[i]);

        auto seperate_out_cols =
            this->getSeparateOutputColumns(local_input_cols, ftypes[i]);
        std::set<bodo_array_type::arr_type_enum> force_acc_types = {
            bodo_array_type::STRING, bodo_array_type::DICT,
            bodo_array_type::ARRAY_ITEM, bodo_array_type::STRUCT,
            bodo_array_type::MAP};
        for (const auto& t : (running_values_schema->column_types)) {
            if (force_acc_types.contains(t->array_type)) {
                this->accumulate_before_update = true;
                break;
            }
        }

        if (seperate_out_cols.size() != 0) {
            for (auto t : seperate_out_cols) {
                if (force_acc_types.contains(std::get<0>(t))) {
                    this->accumulate_before_update = true;
                    break;
                }
            }
        }
    }

    // Finally, now that we know if we need to accumulate all values before
    // update, do one last iteration to actually create each of the col_sets
    bool do_combine = !this->accumulate_before_update;
    for (size_t i = 0; i < ftypes.size(); i++) {
        std::vector<std::shared_ptr<array_info>>& local_input_cols =
            local_input_cols_vec.at(i);
        std::vector<std::unique_ptr<bodo::DataType>>& in_arr_types =
            in_arr_types_vec.at(i);
        std::vector<std::unique_ptr<bodo::DataType>> in_arr_types_copy;
        for (const auto& t : in_arr_types) {
            in_arr_types_copy.push_back(t->copy());
        }

        std::shared_ptr<BasicColSet> col_set = makeColSet(
            local_input_cols, index_col, ftypes[i], do_combine, skip_na_data, 0,
            // In the streaming multi-partition scenario, it's
            // safer to mark things as *not* parallel to avoid
            // any synchronization and hangs.
            {0}, 0, /*is_parallel*/ false, this->mrnf_sort_asc,
            this->mrnf_sort_na, {nullptr}, 0, nullptr, nullptr, 0, nullptr,
            use_sql_rules);

        // get update/combine type info to initialize build state
        std::unique_ptr<bodo::Schema> running_values_schema =
            col_set->getRunningValueColumnTypes(
                std::make_shared<bodo::Schema>(std::move(in_arr_types_copy)));
        size_t n_running_value_types = running_values_schema->ncols();

        if (!this->accumulate_before_update) {
            build_table_schema->append_schema(std::move(running_values_schema));

            // Determine what separate output columns are necessary.
            // This is only required in the AGG case.
            auto separate_out_col_type = col_set->getSeparateOutputColumnType();
            if (separate_out_col_type.size() != 0) {
                if (separate_out_col_type.size() != 1) {
                    throw std::runtime_error(
                        "GroupbyState::GroupbyState Colsets with multiple "
                        "separate output columns not supported");
                }
                separate_out_cols_schema->append_column(
                    std::get<0>(separate_out_col_type[0]),
                    std::get<1>(separate_out_col_type[0]));
            }
        }

        curr_running_value_offset += n_running_value_types;
        this->f_running_value_offsets.push_back(curr_running_value_offset);

        this->col_sets.push_back(col_set);
    }

    // See if all ColSet functions are nunique, which enables optimization of
    // dropping duplicate shuffle table rows before shuffle
    this->nunique_only = true;
    for (size_t i = 0; i < ftypes.size(); i++) {
        if (ftypes[i] != Bodo_FTypes::nunique) {
            this->nunique_only = false;
        }
    }

    // build buffer types are same as input if just accumulating batches
    if (this->accumulate_before_update) {
        build_table_schema = std::make_shared<bodo::Schema>(*in_schema_);
    }

    this->key_dict_builders.resize(this->n_keys);

    // Create dictionary builders for key columns:
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (build_table_schema->column_types[i]->array_type ==
            bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array_top_level(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            this->key_dict_builders[i] =
                std::make_shared<DictionaryBuilder>(dict, true);
        } else {
            this->key_dict_builders[i] = nullptr;
        }
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        build_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in build table:
    for (size_t i = this->n_keys; i < build_table_schema->column_types.size();
         i++) {
        if (build_table_schema->column_types[i]->array_type ==
            bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array_top_level(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            build_table_non_key_dict_builders.emplace_back(
                std::make_shared<DictionaryBuilder>(dict, false));
        } else {
            build_table_non_key_dict_builders.emplace_back(nullptr);
        }
    }

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(), this->key_dict_builders.begin(),
        this->key_dict_builders.end());

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(),
        build_table_non_key_dict_builders.begin(),
        build_table_non_key_dict_builders.end());

    this->shuffle_state = std::make_unique<GroupbyIncrementalShuffleState>(
        build_table_schema, this->build_table_dict_builders, this->n_keys,
        this->build_iter, this->sync_iter, op_id_, this->nunique_only);

    this->partitions.emplace_back(std::make_shared<GroupbyPartition>(
        0, 0, build_table_schema, std::move(separate_out_cols_schema),
        this->n_keys, this->build_table_dict_builders, this->col_sets,
        this->f_in_offsets, this->f_in_cols, this->f_running_value_offsets,
        /*is_active*/ true, this->accumulate_before_update,
        this->req_extended_group_info, this->op_pool.get(), this->op_mm,
        this->op_scratch_pool.get(), this->op_scratch_mm));
    this->partition_state.emplace_back(std::make_pair<size_t, uint32_t>(0, 0));

    // Reserve space upfront. The output-batch-size is typically the same
    // as the input batch size.
    this->append_row_to_build_table.reserve(output_batch_size_);
}

std::unique_ptr<bodo::Schema> GroupbyState::getRunningValueColumnTypes(
    std::vector<std::shared_ptr<array_info>> local_input_cols,
    std::vector<std::unique_ptr<bodo::DataType>>&& in_dtypes, int ftype) {
    std::shared_ptr<BasicColSet> col_set =
        makeColSet(local_input_cols,     // in_cols
                   nullptr,              // index_col
                   ftype,                // ftype
                   true,                 // do_combine
                   true,                 // skip_na_data
                   0,                    // period
                   {0},                  // transform_funcs
                   0,                    // n_udf
                   false,                // parallel
                   this->mrnf_sort_asc,  // window_ascending
                   this->mrnf_sort_na,   // window_na_position
                   {nullptr},            // window_args
                   0,                    // n_input_cols
                   nullptr,              // udf_n_redvars
                   nullptr,              // udf_table
                   0,                    // udf_table_idx
                   nullptr,              // nunique_table
                   true                  // use_sql_rules
        );

    // get update/combine type info to initialize build state
    std::vector<std::unique_ptr<bodo::DataType>> in_arr_types_copy;
    for (const auto& t : in_dtypes) {
        in_arr_types_copy.push_back(t->copy());
    }
    return col_set->getRunningValueColumnTypes(
        std::make_shared<bodo::Schema>(std::move(in_arr_types_copy)));
}

std::vector<std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
GroupbyState::getSeparateOutputColumns(
    std::vector<std::shared_ptr<array_info>> local_input_cols, int ftype) {
    std::shared_ptr<BasicColSet> col_set =
        makeColSet(local_input_cols,     // in_cols
                   nullptr,              // index_col
                   ftype,                // ftype
                   true,                 // do_combine
                   true,                 // skip_na_data
                   0,                    // period
                   {0},                  // transform_funcs
                   0,                    // n_udf
                   false,                // parallel
                   this->mrnf_sort_asc,  // window_ascending
                   this->mrnf_sort_na,   // window_na_position
                   {nullptr},            // window_args
                   0,                    // n_input_cols
                   nullptr,              // udf_n_redvars
                   nullptr,              // udf_table
                   0,                    // udf_table_idx
                   nullptr,              // nunique_table
                   true                  // use_sql_rules
        );

    auto seperate_out_cols = col_set->getSeparateOutputColumnType();
    return seperate_out_cols;
}

void GroupbyState::DisablePartitioning() {
    if (this->partitioning_enabled) {
        this->op_pool->DisableThresholdEnforcement();
        this->partitioning_enabled = false;
    }
}

void GroupbyState::EnablePartitioning() {
    if (!this->partitioning_enabled) {
        this->op_pool->EnableThresholdEnforcement();
        this->partitioning_enabled = true;
    }
}

void GroupbyState::SplitPartition(size_t idx) {
    if (this->partitions[idx]->get_num_top_bits() >=
        this->max_partition_depth) {
        // TODO Eventually, this should lead to falling back
        // to nested loop join for this partition.
        // (https://bodo.atlassian.net/browse/BSE-535).
        throw std::runtime_error(
            "GroupbyState::SplitPartition: Cannot split partition beyond "
            "max partition depth of: " +
            std::to_string(max_partition_depth));
    }

    if (this->debug_partitioning) {
        std::cerr << "[DEBUG] Splitting partition " << idx << "." << std::endl;
    }

    // Temporarily disable threshold enforcement during partition
    // split.
    this->op_pool->DisableThresholdEnforcement();

    // Call SplitPartition on the idx'th partition:
    std::vector<std::shared_ptr<GroupbyPartition>> new_partitions;
    if (this->partitions[idx]->is_active_partition()) {
        new_partitions = this->partitions[idx]->SplitPartition<true>();
    } else {
        new_partitions = this->partitions[idx]->SplitPartition<false>();
    }
    // Remove the current partition (this should release its memory)
    this->partitions.erase(this->partitions.begin() + idx);
    this->partition_state.erase(this->partition_state.begin() + idx);
    // Insert the new partitions in its place
    this->partitions.insert(this->partitions.begin() + idx,
                            new_partitions.begin(), new_partitions.end());
    std::vector<std::pair<size_t, uint32_t>> new_partitions_state;
    for (auto& partition : new_partitions) {
        new_partitions_state.emplace_back(std::make_pair<size_t, uint32_t>(
            partition->get_num_top_bits(), partition->get_top_bitmask()));
    }
    this->partition_state.insert(this->partition_state.begin() + idx,
                                 new_partitions_state.begin(),
                                 new_partitions_state.end());

    // Re-enable threshold enforcement now that we have split the
    // partition successfully.
    this->op_pool->EnableThresholdEnforcement();

    // TODO Check if the new active partition needs to be split up further.
    // XXX Might not be required if we split proactively and there isn't
    // a single hot key (in which case we need to fall back to sorted
    // aggregation for this partition).
}

void GroupbyState::UpdateGroupsAndCombineHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->UpdateGroupsAndCombine<true>(in_table,
                                                          batch_hashes_groupby);
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
                "GroupbyState::UpdateGroupsAndCombine: Couldn't find "
                "any matching partition for row!");
        }
    }
    this->partitions[0]->UpdateGroupsAndCombine<true>(
        in_table, batch_hashes_groupby, append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->UpdateGroupsAndCombine<false>(
            in_table, batch_hashes_groupby, append_rows_by_partition[i_part]);
    }
}

void GroupbyState::UpdateGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby) {
    while (true) {
        try {
            this->UpdateGroupsAndCombineHelper(in_table, partitioning_hashes,
                                               batch_hashes_groupby);
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
                std::cerr << "[DEBUG] GroupbyState::UpdateGroupsAndCombine[3]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }

            // Note that we don't need to call ClearColSetsState here since
            // the ColSets are only used in the combine step which shouldn't
            // raise a threshold enforcement error.
            this->SplitPartition(0);
        }
    }
}

void GroupbyState::UpdateGroupsAndCombineHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
    const std::vector<bool>& append_rows) {
    if (this->partitions.size() == 1) {
        // Fast path for the single partition case
        this->partitions[0]->UpdateGroupsAndCombine<true>(
            in_table, batch_hashes_groupby, append_rows);
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
                    break;
                }
            }
            if (!found_partition) {
                throw std::runtime_error(
                    "GroupbyState::UpdateGroupsAndCombine: Couldn't find "
                    "any matching partition for row!");
            }
        }
    }

    this->partitions[0]->UpdateGroupsAndCombine<true>(
        in_table, batch_hashes_groupby, append_rows_by_partition[0]);
    for (size_t i_part = 1; i_part < this->partitions.size(); i_part++) {
        this->partitions[i_part]->UpdateGroupsAndCombine<false>(
            in_table, batch_hashes_groupby, append_rows_by_partition[i_part]);
    }
}

void GroupbyState::UpdateGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
    const std::vector<bool>& append_rows) {
    while (true) {
        try {
            this->UpdateGroupsAndCombineHelper(in_table, partitioning_hashes,
                                               batch_hashes_groupby,
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
                std::cerr << "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }

            // Note that we don't need to call ClearColSetsState here since
            // the ColSets are only used in the combine step which shouldn't
            // raise a threshold enforcement error.
            this->SplitPartition(0);
        }
    }
}

void GroupbyState::UpdateShuffleGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
    const std::vector<bool>& not_append_rows) {
    // set state batch input
    this->shuffle_state->in_table = in_table;
    this->shuffle_state->in_table_hashes = batch_hashes_groupby;
    // Reserve space in buffers for potential new groups.
    // Note that if any of the keys/running values are strings, they always
    // go through the accumulate path.
    this->shuffle_state->table_buffer->ReserveTable(in_table);

    // Fill row group numbers in grouping_info to reuse existing
    // infrastructure.
    // We set group=-1 for rows that don't belong to the current buffer
    // (e.g. row belongs to shuffle buffer but we are processing local
    // buffer) for them to be ignored in combine step later.
    grouping_info shuffle_grp_info;
    shuffle_grp_info.row_to_group.resize(in_table->nrows(), -1);

    // Get current size of the buffers to know starting offset of new
    // keys which need output data column initialization.
    // update_groups_helper will update this->shuffle_next_group in place,
    // so we need to cache it beforehand.
    int64_t shuffle_init_start_row = this->shuffle_state->next_group;

    // Add new groups and get group mappings for input batch
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (!not_append_rows[i_row]) {  // double-negative to avoid
                                        // recomputation
            update_groups_helper(*(this->shuffle_state->table_buffer),
                                 this->shuffle_state->groupby_hashes,
                                 *(this->shuffle_state->hash_table),
                                 this->shuffle_state->next_group, this->n_keys,
                                 shuffle_grp_info, in_table,
                                 batch_hashes_groupby, i_row);
        }
    }
    // Combine existing (and new) keys using the input batch
    combine_input_table_helper(in_table, shuffle_grp_info,
                               this->shuffle_state->table_buffer->data_table,
                               this->f_running_value_offsets, this->col_sets,
                               shuffle_init_start_row);

    int64_t new_groups =
        this->shuffle_state->next_group - shuffle_init_start_row;
    this->shuffle_state->UpdateAppendBatchSize(in_table->nrows(), new_groups);

    // Reset temporary references
    this->shuffle_state->in_table.reset();
    this->shuffle_state->in_table_hashes.reset();
}

void GroupbyState::AppendBuildBatchHelper(
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
                "GroupbyState::AppendBuildBatch: Couldn't find "
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

void GroupbyState::AppendBuildBatch(
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
                std::cerr << "[DEBUG] GroupbyState::AppendBuildBatch[2]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }

            this->SplitPartition(0);
        }
    }
}

void GroupbyState::AppendBuildBatchHelper(
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
                    "GroupbyState::AppendBuildBatch: Couldn't find "
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

void GroupbyState::AppendBuildBatch(
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
                std::cerr << "[DEBUG] GroupbyState::AppendBuildBatch[3]: "
                             "Encountered OperatorPoolThresholdExceededError."
                          << std::endl;
            }

            this->SplitPartition(0);
        }
    }
}

void GroupbyState::InitOutputBufferMrnf(
    const std::shared_ptr<table_info>& dummy_build_table) {
    assert(this->mrnf_only);

    // Skip if already initialized.
    if (this->output_buffer != nullptr) {
        return;
    }

    size_t n_cols = dummy_build_table->columns.size();
    std::vector<bool> cols_to_keep_bitmask = get_mrnf_cols_to_keep_bitmask(
        this->mrnf_part_cols_to_keep, this->mrnf_sort_cols_to_keep, n_cols);

    // List of column types in the output.
    std::vector<std::unique_ptr<bodo::DataType>> output_column_types;
    for (size_t i = 0; i < n_cols; i++) {
        if (cols_to_keep_bitmask[i]) {
            output_column_types.push_back(
                dummy_build_table->columns[i]->data_type());
            // We can re-use existing dictionaries.
            this->out_dict_builders.push_back(
                this->build_table_dict_builders[i]);
        }
    }

    // Build the output schema from the output column types.
    std::shared_ptr<bodo::Schema> output_schema =
        std::make_shared<bodo::Schema>(std::move(output_column_types));
    // Initialize the output buffer using this schema.
    this->output_buffer = std::make_shared<ChunkedTableBuilder>(
        output_schema, this->out_dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}

void GroupbyState::InitOutputBuffer(
    const std::shared_ptr<table_info>& dummy_table) {
    assert(!this->mrnf_only);
    auto schema = dummy_table->schema();

    // This is not initialized until this point. Resize it to the required
    // size.
    this->out_dict_builders.resize(dummy_table->columns.size(), nullptr);

    // Keys are always the first columns in groupby output and match input
    // array types and dictionaries for DICT arrays. See
    // https://github.com/Bodo-inc/Bodo/blob/f94ab6d2c78e3a536a8383ddf71956cc238fccc8/bodo/libs/_groupby_common.cpp#L604
    for (size_t i = 0; i < this->n_keys; i++) {
        this->out_dict_builders[i] = this->build_table_dict_builders[i];
    }
    // Non-key columns may have different type and/or dictionaries from
    // input arrays
    for (size_t i = this->n_keys; i < dummy_table->ncols(); i++) {
        if (dummy_table->columns[i]->arr_type == bodo_array_type::DICT) {
            std::shared_ptr<array_info> dict = alloc_array_top_level(
                0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
            this->out_dict_builders[i] =
                std::make_shared<DictionaryBuilder>(dict, false);
        }
    }
    this->output_buffer = std::make_shared<ChunkedTableBuilder>(
        std::move(schema), this->out_dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}

std::shared_ptr<table_info> GroupbyState::UnifyBuildTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type != bodo_array_type::DICT ||
            (only_keys && (i >= this->n_keys))) {
            out_arr = in_arr;
        } else {
            out_arr = this->build_table_dict_builders[i]->UnifyDictionaryArray(
                in_arr);
        }
        out_arrs.emplace_back(out_arr);
    }

    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> GroupbyState::UnifyOutputDictionaryArrays(
    const std::shared_ptr<table_info>& out_table) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(out_table->ncols());
    for (size_t i = 0; i < out_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = out_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        // Output key columns have the same dictionary as inputs and don't
        // need unification
        if (in_arr->arr_type != bodo_array_type::DICT || (i < this->n_keys)) {
            out_arr = in_arr;
        } else {
            out_arr = this->out_dict_builders[i]->UnifyDictionaryArray(in_arr);
        }
        out_arrs.emplace_back(out_arr);
    }

    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
GroupbyState::GetDictionaryHashesForKeys() {
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

void GroupbyState::ClearColSetsStates() {
    for (const std::shared_ptr<BasicColSet>& col_set : this->col_sets) {
        col_set->clear();
    }
}

void GroupbyState::ClearBuildState() {
    this->col_sets.clear();
    // this->out_dict_builders retains references
    // to the DictionaryBuilders required for the output
    // buffer, so clearing these is safe.
    this->build_table_dict_builders.clear();
    this->key_dict_builders.clear();
    this->append_row_to_build_table.resize(0);
    this->append_row_to_build_table.shrink_to_fit();
}

void GroupbyState::FinalizeBuild() {
    // Clear the shuffle state since it is longer required.
    this->shuffle_state->Finalize();

    for (size_t i_part = 0; i_part < this->partitions.size(); i_part++) {
        // TODO Add logic to check if partition is too big
        // (build_table_buffer size + approximate hash table size) and needs
        // to be repartitioned upfront.

        while (true) {
            bool exception_caught = true;
            bool orig_partitioning_enabled = this->partitioning_enabled;
            std::shared_ptr<table_info> output_table;
            try {
                // If partitioning is enabled and this partition is at max
                // partition depth, then temporarily disable partitioning. If it
                // wouldn't have run into a threshold enforcement error, there's
                // no side-effects. If it would've, then it would've tried to
                // split the partition which would've simply raised a
                // runtime-error and halted the execution. In this situation,
                // it's better to let it use as much memory as is available. If
                // it fails, we're no worse off than before. However, there's a
                // chance that this will succeed. Overall, we're better off
                // doing this. Note that this is only until we implement a
                // proper fallback mechanism such as a sorted aggregation.
                if ((this->partitions[i_part]->get_num_top_bits() ==
                     this->max_partition_depth) &&
                    orig_partitioning_enabled) {
                    this->DisablePartitioning();
                    if (this->debug_partitioning) {
                        // Log a warning
                        std::cerr
                            << "[DEBUG] WARNING: Disabling partitioning and "
                               "threshold enforcement temporarily to finalize "
                               "partition "
                            << i_part
                            << " which is at max allowed partition depth ("
                            << this->max_partition_depth
                            << "). This may invoke the OOM killer."
                            << std::endl;
                    }
                }

                // Finalize the partition and get output from it.
                // TODO: Write output directly into the GroupybyState's
                // output buffer instead of returning the output.
                if (this->mrnf_only) {
                    // Initialize output buffer if this is the first
                    // partition.
                    if (i_part == 0) {
                        this->InitOutputBufferMrnf(
                            this->partitions[i_part]
                                ->build_table_buffer->data_table);
                    }
                    this->partitions[i_part]->FinalizeMrnf(
                        this->mrnf_part_cols_to_keep,
                        this->mrnf_sort_cols_to_keep, this->output_buffer);
                } else {
                    output_table = this->partitions[i_part]->Finalize();
                }
                exception_caught = false;
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
                        << "[DEBUG] GroupbyState::FinalizeBuild: "
                           "Encountered OperatorPoolThresholdExceededError "
                           "while finalizing partition "
                        << i_part << "." << std::endl;
                }

                // In case the error happened during a ColSet operation, we
                // want to clear their states before retrying.
                this->ClearColSetsStates();
                this->SplitPartition(i_part);
            }

            if (!exception_caught) {
                // Since we have generated the output, we don't need the
                // partition anymore, so we can release that memory.
                this->partitions[i_part].reset();
                if (!this->mrnf_only) {
                    if (i_part == 0) {
                        this->InitOutputBuffer(output_table);
                    }
                    // XXX TODO UnifyOutputDictionaryArrays needs a version that
                    // can take the shared_ptr without reference and free
                    // individual columns early.
                    output_table =
                        this->UnifyOutputDictionaryArrays(output_table);
                    this->output_buffer->AppendBatch(output_table);
                    output_table.reset();
                }
                if (this->debug_partitioning) {
                    std::cerr << "[DEBUG] GroupbyState::FinalizeBuild: "
                                 "Successfully finalized partition "
                              << i_part << "." << std::endl;
                }
                // If we disabled partitioning (and it was originally enabled),
                // turn it back on. Note that we only need to check this in the
                // case that it succeeded (i.e. exception_caught = false). If we
                // disabled partitioning, OperatorPoolThresholdExceededError
                // couldn't have been thrown and that's the only error we catch.
                // Note that re-enabling partitioning cannot raise the
                // OperatorPoolThresholdExceededError here since we have already
                // free-d all allocations that were made since temporarily
                // disabling partitioning (the entire partition and the output
                // table have been freed). The output was pushed into the
                // output_buffer which is not tracked by the OperatorPool.
                // Therefore, if we were below the threshold then, we must be
                // below the threshold now as well.
                if (orig_partitioning_enabled) {
                    this->EnablePartitioning();
                }
                break;
            }
        }
    }

    if (this->debug_partitioning) {
        std::cerr << "[DEBUG] GroupbyState::FinalizeBuild: Total number of "
                     "partitions: "
                  << this->partitions.size() << "." << std::endl;
    }

    this->output_buffer->Finalize();
    // Release the ColSets, etc.
    this->ClearBuildState();
    this->build_input_finalized = true;
}

uint64_t GroupbyState::op_pool_bytes_pinned() const {
    return this->op_pool->bytes_pinned();
}

uint64_t GroupbyState::op_pool_bytes_allocated() const {
    return this->op_pool->bytes_allocated();
}

#pragma endregion  // GroupbyState
/* ------------------------------------------------------------------------ */

/**
 * @brief consume build table batch in streaming groupby (insert into hash
 * table and update running values)
 *
 * @param groupby_state groupby state pointer
 * @param in_table build table batch
 * @param is_last is last batch (in this pipeline) locally
 * @param is_final_pipeline Is this the final pipeline. Only relevant for the
 * Union-Distinct case where this is called in multiple pipelines. For regular
 * groupby, this should always be true. We only call FinalizeBuild in the last
 * pipeline.
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool groupby_agg_build_consume_batch(GroupbyState* groupby_state,
                                     std::shared_ptr<table_info> in_table,
                                     bool is_last,
                                     const bool is_final_pipeline) {
    // High level workflow (reusing as much of existing groupby
    // infrastructure as possible):
    // 1. Get update values from input. Example with sum function:
    //      A   B
    //      1   3
    //      2   1
    //      1   1
    //    becomes:
    //      A   B
    //      1   4
    //      2   1
    // 2. Get group numbers for each input row from local and shuffle build
    // tables.
    //    This inserts a new group to the table if it doesn't exist.
    // 3. Combine update values with local and shuffle build tables.

    int n_pes, myrank;
    // trace performance
    auto iterationEvent(groupby_state->groupby_event.iteration());
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Unify dictionaries keys to allow consistent hashing and fast key
    // comparison using indices
    in_table = groupby_state->UnifyBuildTableDictionaryArrays(in_table, true);
    // We don't pass the op-pool here since this is operation on a small
    // batch and we consider this "scratch" usage essentially.
    in_table = get_update_table</*is_acc_case*/ false>(
        in_table, groupby_state->n_keys, groupby_state->col_sets,
        groupby_state->f_in_offsets, groupby_state->f_in_cols,
        groupby_state->req_extended_group_info);

    if (groupby_state->build_iter == 0) {
        groupby_state->shuffle_state->Initialize(in_table,
                                                 groupby_state->parallel);
    }

    // Make is_last global
    is_last = stream_sync_is_last(is_last, groupby_state->build_iter,
                                  groupby_state->sync_iter);

    // Dictionary hashes for the key columns which will be used for
    // the partitioning hashes:
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = groupby_state->GetDictionaryHashesForKeys();

    std::shared_ptr<uint32_t[]> batch_hashes_groupby =
        hash_keys_table(in_table, groupby_state->n_keys,
                        SEED_HASH_GROUPBY_SHUFFLE, false, false);
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, groupby_state->n_keys, SEED_HASH_PARTITION,
                        groupby_state->parallel, false, dict_hashes);

    // Use the cached allocation:
    std::vector<bool>& append_row_to_build_table =
        groupby_state->append_row_to_build_table;
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        append_row_to_build_table.push_back(
            (!groupby_state->parallel ||
             (hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank)));
    }

    // Fill row group numbers in grouping_info to reuse existing
    // infrastructure. We set group=-1 for rows that don't belong to the
    // current buffer (e.g. row belongs to shuffle buffer but we are
    // processing local buffer) for them to be ignored in combine step
    // later.
    groupby_state->UpdateGroupsAndCombine(in_table, batch_hashes_partition,
                                          batch_hashes_groupby,
                                          append_row_to_build_table);

    // Do the same for the shuffle groups:
    groupby_state->UpdateShuffleGroupsAndCombine(
        in_table, batch_hashes_partition, batch_hashes_groupby,
        append_row_to_build_table);

    // Reset the bitmask for the next iteration:
    append_row_to_build_table.resize(0);

    if (groupby_state->parallel) {
        std::optional<std::shared_ptr<table_info>> new_data_ =
            groupby_state->shuffle_state->ShuffleIfRequired(is_last);
        if (new_data_.has_value()) {
            std::shared_ptr<table_info> new_data = new_data_.value();
            dict_hashes = groupby_state->GetDictionaryHashesForKeys();
            batch_hashes_groupby = hash_keys_table(
                new_data, groupby_state->n_keys, SEED_HASH_GROUPBY_SHUFFLE,
                groupby_state->parallel, /*global_dict_needed*/ false);
            batch_hashes_partition =
                hash_keys_table(new_data, groupby_state->n_keys,
                                SEED_HASH_PARTITION, groupby_state->parallel,
                                /*global_dict_needed*/ false, dict_hashes);

            groupby_state->UpdateGroupsAndCombine(
                new_data, batch_hashes_partition, batch_hashes_groupby);
        }
    }

    if (is_last && is_final_pipeline) {
        groupby_state->FinalizeBuild();
    }

    groupby_state->build_iter++;

    // XXX Could reset the shuffle state (including setting build_iter to 0)
    // here if it's the last iteration of non-final pipeline.
    // This would free the buffer and give back memory to the system until we
    // enter the next pipeline.
    // It would also force it to re-evaluate the sync frequency in the new
    // pipeline, which is generally good since the data pattern might change
    // between sources.
    // (https://bodo.atlassian.net/browse/BSE-2091)

    return is_last;
}

/**
 * @brief consume build table batch in streaming groupby by just
 * accumulating rows (used in cases where at least one groupby function
 * requires all group data upfront)
 *
 * @param groupby_state groupby state pointer
 * @param in_table build table batch
 * @param is_last is last batch locally
 * @param is_final_pipeline Is this the final pipeline. This should always be
 * true. We provide this parameter for consistency with the
 * incremental-aggregation code-path.
 * @return updated is_last
 */
bool groupby_acc_build_consume_batch(GroupbyState* groupby_state,
                                     std::shared_ptr<table_info> in_table,
                                     bool is_last,
                                     const bool is_final_pipeline) {
    assert(is_final_pipeline);
    int n_pes, myrank;
    // trace performance
    auto iterationEvent(groupby_state->groupby_event.iteration());
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (groupby_state->build_iter == 0) {
        groupby_state->shuffle_state->Initialize(in_table,
                                                 groupby_state->parallel);
    }

    is_last = stream_sync_is_last(is_last, groupby_state->build_iter,
                                  groupby_state->sync_iter);

    // We require that all dictionary keys/values are unified before update
    in_table = groupby_state->UnifyBuildTableDictionaryArrays(in_table, false);
    // Dictionary hashes for the key columns which will be used for
    // the partitioning hashes:
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = groupby_state->GetDictionaryHashesForKeys();

    // Append input rows to local or shuffle buffer:
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, groupby_state->n_keys, SEED_HASH_PARTITION,
                        groupby_state->parallel, false, dict_hashes);

    // Use cached allocation:
    std::vector<bool>& append_row_to_build_table =
        groupby_state->append_row_to_build_table;
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        append_row_to_build_table.push_back(
            (!groupby_state->parallel ||
             hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank));
    }
    groupby_state->AppendBuildBatch(in_table, batch_hashes_partition,
                                    append_row_to_build_table);

    batch_hashes_partition.reset();

    append_row_to_build_table.flip();
    std::vector<bool>& append_row_to_shuffle_table = append_row_to_build_table;
    groupby_state->shuffle_state->AppendBatch(in_table,
                                              append_row_to_shuffle_table);

    // Reset for next iteration:
    append_row_to_build_table.resize(0);

    // Shuffle data of other ranks and append received data to local buffer
    if (groupby_state->parallel) {
        std::optional<std::shared_ptr<table_info>> new_data_ =
            groupby_state->shuffle_state->ShuffleIfRequired(is_last);
        if (new_data_.has_value()) {
            std::shared_ptr<table_info> new_data = new_data_.value();
            // Dictionary hashes for the key columns which will be used for
            // the partitioning hashes:
            dict_hashes = groupby_state->GetDictionaryHashesForKeys();

            // Append input rows to local or shuffle buffer:
            batch_hashes_partition = hash_keys_table(
                new_data, groupby_state->n_keys, SEED_HASH_PARTITION,
                groupby_state->parallel, false, dict_hashes);

            // XXX Technically, we don't need the partition hashes if
            // there's just one partition. We could move the hash computation
            // inside AppendBuildBatch and only do it if there are multiple
            // partitions.
            groupby_state->AppendBuildBatch(new_data, batch_hashes_partition);

            batch_hashes_partition.reset();
        }
    }

    // Compute output when all input batches are accumulated
    if (is_last && is_final_pipeline) {
        groupby_state->FinalizeBuild();
    }

    groupby_state->build_iter++;

    // XXX Could reset the shuffle state (including setting build_iter to 0)
    // here if it's the last iteration of non-final pipeline.
    // This would free the buffer and give back memory to the system until we
    // enter the next pipeline.
    // It would also force it to re-evaluate the sync frequency in the new
    // pipeline, which is generally good since the data pattern might change
    // between sources.
    // (https://bodo.atlassian.net/browse/BSE-2091)

    return is_last;
}

/**
 * @brief return output of groupby computation
 *
 * @param groupby_state groupby state pointer
 * @param produce_output flag to indicate if output should be produced
 * @return std::tuple<std::shared_ptr<table_info>, bool> output data batch
 * and flag for last batch
 */
std::tuple<std::shared_ptr<table_info>, bool> groupby_produce_output_batch(
    GroupbyState* groupby_state, bool produce_output) {
    if (!produce_output) {
        return std::tuple(groupby_state->output_buffer->dummy_output_chunk,
                          false);
    }
    // TODO[BSE-645]: Prune unused columns at this point.
    // Note: We always finalize the active chunk the build step so we don't
    // need to finalize here.
    auto [out_table, chunk_size] = groupby_state->output_buffer->PopChunk();
    // TODO[BSE-645]: Include total_rows to handle the dead key case
    // *total_rows = chunk_size;
    bool is_last = groupby_state->output_buffer->total_remaining == 0;
    return std::tuple(out_table, is_last);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param groupby_state groupby state pointer
 * @param in_table build table batch
 * @param is_last is last batch (in this pipeline) locally
 * @param is_final_pipeline Is this the final pipeline. Only relevant for the
 * Union-Distinct case where this is called in multiple pipelines. For regular
 * groupby, this should always be true. We only call FinalizeBuild in the last
 * pipeline.
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool groupby_build_consume_batch_py_entry(GroupbyState* groupby_state,
                                          table_info* in_table, bool is_last,
                                          const bool is_final_pipeline) {
    try {
        if (groupby_state->accumulate_before_update) {
            return groupby_acc_build_consume_batch(
                groupby_state, std::unique_ptr<table_info>(in_table), is_last,
                is_final_pipeline);
        } else {
            return groupby_agg_build_consume_batch(
                groupby_state, std::unique_ptr<table_info>(in_table), is_last,
                is_final_pipeline);
        }

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return false;
}

/**
 * @brief Python wrapper to produce output table
 * batch
 *
 * @param groupby_state groupby state pointer
 * @param[out] out_is_last is last batch
 * @param produce_output whether to produce output
 * @return table_info* output table batch
 */
table_info* groupby_produce_output_batch_py_entry(GroupbyState* groupby_state,
                                                  bool* out_is_last,
                                                  bool produce_output) {
    try {
        bool is_last;
        std::shared_ptr<table_info> out;
        std::tie(out, is_last) =
            groupby_produce_output_batch(groupby_state, produce_output);
        *out_is_last = is_last;
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief Initialize a new streaming groubpy state for specified array types
 * and number of keys (called from Python)
 *
 * @param build_arr_c_types array types of build table columns (Bodo_CTypes
 * ints)
 * @param build_arr_array_types array types of build table columns
 * (bodo_array_type ints)
 * @param n_build_arrs number of build table columns
 * @param n_keys number of groupby keys
 * @param mrnf_sort_asc Boolean bitmask specifying sort-direction for MRNF
 *  order-by columns. It should be 'mrnf_n_sort_keys' elements long.
 * @param mrnf_sort_na Boolean bitmask specifying whether nulls should be
 *  considered 'last' for the order-by columns of MRNF. It should be
 * 'mrnf_n_sort_keys' elements long.
 * @param mrnf_n_sort_keys Number of MRNF order-by columns. If this is
 *  >0, we will use MRNF specific code.
 * @param mrnf_part_cols_to_keep Bitmask specifying the partition columns
 * to keep in the MRNF case. In the MRNF case, it must have n_keys elements.
 * Otherwise, it can be nullptr since it won't be read.
 * @param mrnf_sort_cols_to_keep Bitmask specifying the order-by columns
 * to keep in the MRNF case. In the MRNF case, it must have mrnf_n_sort_keys
 * elements. Otherwise, it can be nullptr since it won't be read.
 * @param output_batch_size Batch size for reading output.
 * @param op_pool_size_bytes Size of the operator buffer pool for this join
 * operator. If it's set to -1, we will get the budget from the operator
 * comptroller.
 * @return GroupbyState* groupby state to return to Python
 */
GroupbyState* groupby_state_init_py_entry(
    int64_t operator_id, int8_t* build_arr_c_types,
    int8_t* build_arr_array_types, int n_build_arrs, int32_t* ftypes,
    int32_t* f_in_offsets, int32_t* f_in_cols, int n_funcs, uint64_t n_keys,
    bool* mrnf_sort_asc, bool* mrnf_sort_na, uint64_t mrnf_n_sort_keys,
    bool* mrnf_part_cols_to_keep, bool* mrnf_sort_cols_to_keep,
    int64_t output_batch_size, bool parallel, int64_t sync_iter,
    int64_t op_pool_size_bytes) {
    // If the memory budget has not been explicitly set, then ask the
    // OperatorComptroller for the budget.
    if (op_pool_size_bytes == -1) {
        op_pool_size_bytes =
            OperatorComptroller::Default()->GetOperatorBudget(operator_id);
    }

    // Create vectors for the MRNF arguments from the raw pointer arrays.
    std::vector<bool> mrnf_sort_asc_vec(mrnf_n_sort_keys, false);
    std::vector<bool> mrnf_sort_na_vec(mrnf_n_sort_keys, false);
    std::vector<bool> mrnf_part_cols_to_keep_vec(n_keys, true);
    std::vector<bool> mrnf_sort_cols_to_keep_vec(mrnf_n_sort_keys, true);
    if (mrnf_n_sort_keys > 0) {
        for (size_t i = 0; i < n_keys; i++) {
            mrnf_part_cols_to_keep_vec[i] = mrnf_part_cols_to_keep[i];
        }
        for (size_t i = 0; i < mrnf_n_sort_keys; i++) {
            mrnf_sort_asc_vec[i] = mrnf_sort_asc[i];
            mrnf_sort_na_vec[i] = mrnf_sort_na[i];
            mrnf_sort_cols_to_keep_vec[i] = mrnf_sort_cols_to_keep[i];
        }
    }

    return new GroupbyState(
        bodo::Schema::Deserialize(
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs)),

        std::vector<int32_t>(ftypes, ftypes + n_funcs),
        std::vector<int32_t>(f_in_offsets, f_in_offsets + n_funcs + 1),
        std::vector<int32_t>(f_in_cols, f_in_cols + f_in_offsets[n_funcs]),
        n_keys, mrnf_sort_asc_vec, mrnf_sort_na_vec, mrnf_part_cols_to_keep_vec,
        mrnf_sort_cols_to_keep_vec, output_batch_size, parallel, sync_iter,
        operator_id, op_pool_size_bytes);
}

/**
 * @brief delete groupby state (called from Python after output loop is
 * finished)
 *
 * @param groupby_state groupby state pointer to delete
 */
void delete_groupby_state(GroupbyState* groupby_state) { delete groupby_state; }

uint64_t get_op_pool_bytes_pinned(GroupbyState* groupby_state) {
    return groupby_state->op_pool_bytes_pinned();
}

uint64_t get_op_pool_bytes_allocated(GroupbyState* groupby_state) {
    return groupby_state->op_pool_bytes_allocated();
}

uint32_t get_num_partitions(GroupbyState* groupby_state) {
    return groupby_state->partition_state.size();
}

uint32_t get_partition_num_top_bits_by_idx(GroupbyState* groupby_state,
                                           int64_t idx) {
    try {
        std::vector<std::pair<size_t, uint32_t>>& partition_state =
            groupby_state->partition_state;
        if (static_cast<size_t>(idx) >= partition_state.size()) {
            throw std::runtime_error(
                "get_partition_num_top_bits_by_idx: partition index " +
                std::to_string(idx) + " out of bound: " + std::to_string(idx) +
                " >= " + std::to_string(partition_state.size()));
        }
        return partition_state[idx].first;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

uint32_t get_partition_top_bitmask_by_idx(GroupbyState* groupby_state,
                                          int64_t idx) {
    try {
        std::vector<std::pair<size_t, uint32_t>>& partition_state =
            groupby_state->partition_state;
        if (static_cast<size_t>(idx) >= partition_state.size()) {
            throw std::runtime_error(
                "get_partition_top_bitmask_by_idx: partition index " +
                std::to_string(idx) + " out of bound: " + std::to_string(idx) +
                " >= " + std::to_string(partition_state.size()));
        }
        return partition_state[idx].second;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

PyMODINIT_FUNC PyInit_stream_groupby_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_groupby_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, groupby_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, groupby_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, groupby_produce_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_groupby_state);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_pinned);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_allocated);
    SetAttrStringFromVoidPtr(m, get_num_partitions);
    SetAttrStringFromVoidPtr(m, get_partition_num_top_bits_by_idx);
    SetAttrStringFromVoidPtr(m, get_partition_top_bitmask_by_idx);
    return m;
}

#undef MAX_SHUFFLE_HASHTABLE_SIZE
#undef MAX_SHUFFLE_TABLE_SIZE
