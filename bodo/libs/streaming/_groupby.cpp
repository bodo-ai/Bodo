#include "_groupby.h"

#include <fmt/format.h>
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <list>
#include <memory>
#include <tuple>
#include "../_array_hash.h"
#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_chunked_table_builder.h"
#include "../_dict_builder.h"
#include "../_distributed.h"
#include "../_memory_budget.h"
#include "../_query_profile_collector.h"
#include "../_shuffle.h"
#include "../_table_builder.h"
#include "../_utils.h"
#include "../groupby/_groupby_col_set.h"
#include "../groupby/_groupby_common.h"
#include "../groupby/_groupby_ftypes.h"
#include "../groupby/_groupby_groups.h"
#include "_shuffle.h"
#include "arrow/util/bit_util.h"

#define MAX_SHUFFLE_TABLE_SIZE 50 * 1024 * 1024
#define MAX_SHUFFLE_HASHTABLE_SIZE 50 * 1024 * 1024

std::string get_aggregation_type_string(AggregationType type) {
    switch (type) {
        case AggregationType::AGGREGATE:
            return "AGGREGATE";
        case AggregationType::MRNF:
            return "MIN ROW NUMBER FILTER";
        case AggregationType::WINDOW:
            return "WINDOW";
        default:
            throw std::runtime_error("Unsupported aggregation type!");
    }
}

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
 * @param[in, out] metrics Metrics to add to.
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
    GroupbyMetrics::GetGroupInfoMetrics& metrics,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    // Allocate the memory for hashes through the pool.
    // We don't need a ScopedTimer since the allocation will either fail
    // instantly or not at all.
    time_pt start = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_groupby =
        bodo::make_shared_arr<uint32_t>(in_table->nrows(), pool);
    // Compute and fill hashes into allocated memory.
    hash_keys_table(batch_hashes_groupby.get(), in_table, n_keys,
                    SEED_HASH_GROUPBY_SHUFFLE, false);
    metrics.hashing_time += end_timer(start);
    metrics.hashing_nrows += in_table->nrows();

    std::vector<std::shared_ptr<table_info>> tables = {in_table};

    size_t nunique_hashes = 0;
    if (is_acc_case) {
        // In the accumulating code path case, we have the entire input, so
        // it's better to get an actual estimate using HLL.
        // The HLL only uses ~1MiB of memory, so we don't really need it to
        // go through the pool.
        start = start_timer();
        nunique_hashes = get_nunique_hashes(
            batch_hashes_groupby, in_table->nrows(), /*is_parallel*/ false);
        metrics.hll_time += end_timer(start);
        metrics.hll_nrows += in_table->nrows();
    } else {
        // In the case of streaming groupby, we don't need to estimate the
        // number of unique hashes. We can just use the number of rows in the
        // input table since the batches are so small. This has been tested to
        // be faster than estimating the number of unique hashes based on
        // previous batches as well as using HLL.
        nunique_hashes = in_table->nrows();
    }

    std::vector<grouping_info> grp_infos;

    ScopedTimer grouping_timer(metrics.grouping_time);
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
    grouping_timer.finalize();
    metrics.grouping_nrows += in_table->nrows();

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
 * @param[in, out] metrics Metrics to add to.
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
    GroupbyMetrics::AggUpdateMetrics& metrics,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    // Empty function set means drop_duplicates operation, which doesn't require
    // update. Drop-duplicates only goes through the agg path, so this is safe.
    if (col_sets.size() == 0) {
        assert(!is_acc_case);
        // XXX Because of re-partitioning, we might still want to drop the
        // duplicates to avoid storing too much data.
        return in_table;
    }

    // similar to update() function of GroupbyPipeline:
    // https://github.com/bodo-ai/Bodo/blob/58f995dec2507a84afefbb27af01d67bd40fabb4/bodo/libs/_groupby.cpp#L546

    // Get the grouping_info:
    std::vector<grouping_info> grp_infos =
        get_grouping_infos_for_update_table<is_acc_case>(
            in_table, n_keys, req_extended_group_info, metrics.grouping_metrics,
            pool);
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
        ScopedTimer update_timer(metrics.colset_update_time);
        col_set->update(grp_infos, pool, mm);
        update_timer.finalize();
        metrics.colset_update_nrows += in_table->nrows();
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
 * @param in_arr_n_cols Number of columns in the input table.
 * @param n_keys Number of partitioning keys.
 * @param caller Name of the caller function. Used in runtime-errors.
 */
void validate_mrnf_args(const std::vector<int32_t>& ftypes,
                        const std::vector<int32_t>& f_in_cols,
                        const std::vector<int32_t>& f_in_offsets,
                        const std::vector<bool>& mrnf_sort_asc,
                        const std::vector<bool>& mrnf_sort_na,
                        size_t in_arr_n_cols, uint64_t n_keys,
                        std::string caller) {
    if (ftypes.size() != 1) {
        throw std::runtime_error(
            fmt::format("{}: Min Row-Number Filter cannot be "
                        "combined with other aggregate functions.",
                        caller));
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
 * @brief Helper function to compute the output bitmask for the Min Row-Number
 * Filter (MRNF) case.
 * @param in_table The input table to compute the MRNF bitmask for.
 * @param colset The colset to use for computing the output bitmask.
 * @param n_sort_cols The number of columns that will be used for the sort.
 * @param n_keys The number of keys in the input table.
 * @param[in, out] metrics Metrics to add to.
 * @param pool The buffer pool to use for allocations.
 * @param mm The memory manager to use for
 * @return std::tuple<std::unique_ptr<uint8_t[]>, size_t> Bitmask of rows to
 * keep, Number of rows in the output (i.e. number of set bits in the output
 * bitmask)
 */
std::tuple<std::unique_ptr<uint8_t[]>, size_t> compute_local_mrnf(
    std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<BasicColSet>& colset, size_t n_sort_cols,
    size_t n_keys, GroupbyMetrics::AggUpdateMetrics& metrics,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    // MRNF always uses the ACC path.
    std::vector<grouping_info> grp_infos =
        get_grouping_infos_for_update_table</*is_acc_case*/ true>(
            in_table, n_keys,
            /*req_extended_group_info=*/false, metrics.grouping_metrics, pool);
    grouping_info& grp_info = grp_infos[0];
    // Construct a vector with the order-by columns.
    std::vector<std::shared_ptr<array_info>> orderby_arrs(
        in_table->columns.begin() + n_keys,
        in_table->columns.begin() + n_keys + n_sort_cols);

    std::vector<std::shared_ptr<array_info>> list_arr;
    // Compute the output index column using this colset:
    colset->setInCol(orderby_arrs);
    colset->alloc_update_columns(grp_info.num_groups, list_arr,
                                 /*alloc_out_if_no_combine*/ false, pool, mm);
    ScopedTimer update_timer(metrics.colset_update_time);
    colset->update(grp_infos, pool, mm);
    update_timer.finalize();
    metrics.colset_update_nrows += in_table->nrows();
    const std::vector<std::shared_ptr<array_info>> out_cols =
        colset->getOutputColumns();
    const std::shared_ptr<array_info>& idx_col = out_cols[0];
    colset->clear();

    // Create an insertion bitmask for the output.
    // XXX Use bodo::vector instead?
    std::unique_ptr<uint8_t[]> out_bitmask = std::make_unique<uint8_t[]>(
        arrow::bit_util::BytesForBits(in_table->columns[0]->length));
    memset(out_bitmask.get(), 0,
           arrow::bit_util::BytesForBits(in_table->columns[0]->length));

    if (idx_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
            int64_t row_one_idx =
                getv<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(idx_col,
                                                                  group_idx);
            arrow::bit_util::SetBit(out_bitmask.get(), row_one_idx);
        }
    } else {
        assert(idx_col->arr_type == bodo_array_type::NUMPY);
        for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
            int64_t row_one_idx =
                getv<int64_t, bodo_array_type::NUMPY>(idx_col, group_idx);
            arrow::bit_util::SetBit(out_bitmask.get(), row_one_idx);
        }
    }
    size_t num_groups = idx_col->length;
    return std::make_tuple(std::move(out_bitmask), num_groups);
}

#pragma endregion  // Min Row-Number Filter Helpers
/* ------------------------------------------------------------------------ */

/* ---------------------------- Window Helpers ---------------------------- */
#pragma region  // Window Helpers

/**
 * @brief Helper function to output result for 1 or more window function.
 * @param in_table The input table to compute the MRNF bitmask for.
 * @param colset The colset to use for computing the output bitmask.
 * @param n_sort_cols The number of columns that will be used for the sort.
 * @param n_keys The number of keys in the input table.
 * @param[in, out] metrics Metrics to add to.
 * @param pool The buffer pool to use for allocations.
 * @param mm Memory manager corresponding to the pool.
 * @return std::vector<std::shared_ptr<array_info>> Output array_infos for the
 * window functions.
 */
std::vector<std::shared_ptr<array_info>> compute_local_window(
    std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builders,
    const std::shared_ptr<BasicColSet>& colset, size_t n_sort_cols,
    size_t n_keys, GroupbyMetrics::AggUpdateMetrics& metrics,
    std::vector<int32_t> f_in_offsets, std::vector<int32_t> f_in_cols,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    // Window always uses the ACC path.
    std::vector<grouping_info> grp_infos =
        get_grouping_infos_for_update_table</*is_acc_case*/ true>(
            in_table, n_keys,
            /*req_extended_group_info=*/false, metrics.grouping_metrics, pool);
    grouping_info& grp_info = grp_infos[0];
    // Construct a vector with the order-by columns.
    std::vector<std::shared_ptr<array_info>> orderby_input_arrs(
        in_table->columns.begin() + n_keys,
        in_table->columns.begin() + n_keys + n_sort_cols);

    size_t num_funcs = f_in_offsets.size() - 1;

    // also add any extra input columns for functions such as lead to be passed
    // into window_computation
    for (size_t i = 0; i < num_funcs; i++) {
        for (size_t j = f_in_offsets[i]; j < (size_t)f_in_offsets[i + 1]; j++) {
            size_t in_col_idx = f_in_cols[j];
            orderby_input_arrs.push_back(in_table->columns[in_col_idx]);
        }
    }

    std::vector<std::shared_ptr<array_info>> list_arr;
    // Compute the output index column using this colset:
    colset->setInCol(orderby_input_arrs);
    colset->setOutDictBuilders(out_dict_builders);
    colset->alloc_update_columns(grp_info.num_groups, list_arr,
                                 /*alloc_out_if_no_combine*/ false, pool, mm);
    ScopedTimer update_timer(metrics.colset_update_time);
    // This does the actual window computation via window_computation
    colset->update(grp_infos, pool, mm);
    update_timer.finalize();
    metrics.colset_update_nrows += in_table->nrows();
    const std::vector<std::shared_ptr<array_info>> out_cols =
        colset->getOutputColumns();
    colset->clear();
    return out_cols;
}

#pragma endregion  // Window Helpers
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
    GroupbyMetrics& metrics_, bodo::OperatorBufferPool* op_pool_,
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
      metrics(metrics_),
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

inline bool GroupbyPartition::is_in_partition(
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
        time_pt start_hash = start_timer();
        std::unique_ptr<uint32_t[]> hashes = hash_keys_table(
            this->build_table_buffer->data_table, this->n_keys,
            SEED_HASH_GROUPBY_SHUFFLE,
            /*is_parallel*/ false,
            /*global_dict_needed*/ false, /*dict_hashes*/ nullptr,
            /*start_row_offset*/ hashes_cur_len);
        this->metrics.rebuild_ht_hashing_time += end_timer(start_hash);
        this->metrics.rebuild_ht_hashing_nrows += n_unhashed_rows;
        // Append the hashes:
        this->build_table_groupby_hashes.insert(
            this->build_table_groupby_hashes.end(), hashes.get(),
            hashes.get() + n_unhashed_rows);
    }

    // Add entries to the hash table. All rows in the build_table_buffer
    // are guaranteed to be unique groups, so we can just map group ->
    // group.
    ScopedTimer insert_timer(this->metrics.rebuild_ht_insert_time);
    this->metrics.rebuild_ht_insert_nrows +=
        (build_table_nrows - this->next_group);
    while (this->next_group < static_cast<int64_t>(build_table_nrows)) {
        (*(this->build_hash_table))[this->next_group] = this->next_group;
        this->next_group++;
    }
    insert_timer.finalize();
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
        ScopedTimer update_timer(this->metrics.update_logical_ht_time);
        this->build_table_buffer->ReserveTable(in_table);
        this->separate_out_cols->ReserveTableSize(in_table->nrows());
        this->metrics.update_logical_ht_nrows += in_table->nrows();

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

        // Increment separate_out_cols size so aggfunc_out_initialize
        // correctly initializes the columns
        this->separate_out_cols->IncrementSize(std::max<uint64_t>(
            this->next_group - this->separate_out_cols->data_table->nrows(),
            (uint64_t)0));
        update_timer.finalize();

        // Combine existing (and new) keys using the input batch.
        // Since we're not passing in anything that can access the op-pool,
        // this shouldn't make any additional allocations that go through
        // the Operator Pool and hence cannot invoke the threshold
        // enforcement error.
        time_pt start_combine = start_timer();
        combine_input_table_helper(
            in_table, grp_info, this->build_table_buffer->data_table,
            this->f_running_value_offsets, this->col_sets, init_start_row);
        this->metrics.combine_input_time += end_timer(start_combine);
        this->metrics.combine_input_nrows += in_table->nrows();

        /// Commit "transaction". Only update this after all the groups
        /// have been updated and combined and after the hash table,
        /// the build buffer and hashes are all up to date.
        this->build_safely_appended_groups = this->next_group;

        // Reset temporary references
        this->in_table.reset();
        this->in_table_hashes.reset();
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        this->build_table_buffer_chunked->AppendBatch(in_table);
        this->metrics.appends_inactive_time += end_timer(start);
        this->metrics.appends_inactive_nrows += in_table->nrows();
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
        ScopedTimer update_timer(this->metrics.update_logical_ht_time);
        this->build_table_buffer->ReserveTable(in_table);
        size_t append_rows_sum =
            std::count(append_rows.begin(), append_rows.end(), true);
        this->separate_out_cols->ReserveTableSize(append_rows_sum);
        this->metrics.update_logical_ht_nrows += append_rows_sum;
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
        update_timer.finalize();

        // Combine existing (and new) keys using the input batch.
        // Since we're not passing in anything that can access the op-pool,
        // this shouldn't make any additional allocations that go through
        // the Operator Pool and hence cannot invoke the threshold
        // enforcement error.
        time_pt start_combine = start_timer();
        combine_input_table_helper(
            in_table, grp_info, this->build_table_buffer->data_table,
            this->f_running_value_offsets, this->col_sets, init_start_row);
        this->metrics.combine_input_time += end_timer(start_combine);
        this->metrics.combine_input_nrows += append_rows_sum;

        /// Commit "transaction". Only update this after all the groups
        /// have been updated and combined and after the hash table,
        /// the build buffer and hashes are all up to date.
        this->build_safely_appended_groups = this->next_group;

        // Reset temporary references
        this->in_table.reset();
        this->in_table_hashes.reset();
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        size_t num_append_rows =
            std::accumulate(append_rows.begin(), append_rows.end(), (size_t)0);
        this->build_table_buffer_chunked->AppendBatch(in_table, append_rows,
                                                      num_append_rows, 0);
        this->metrics.appends_inactive_time += end_timer(start);
        this->metrics.appends_inactive_nrows += num_append_rows;
    }
}

template <bool is_active>
void GroupbyPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table) {
    if (is_active) {
        ScopedTimer append_timer(this->metrics.appends_active_time);
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        this->metrics.appends_active_nrows += in_table->nrows();
        // Now append the rows. This will always
        // succeed since we've
        // reserved space upfront.
        this->build_table_buffer->UnsafeAppendBatch(in_table);
        append_timer.finalize();
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        this->build_table_buffer_chunked->AppendBatch(in_table);
        this->metrics.appends_inactive_time += end_timer(start);
        this->metrics.appends_inactive_nrows += in_table->nrows();
    }
}

template <bool is_active>
void GroupbyPartition::AppendBuildBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    if (is_active) {
        ScopedTimer append_timer(this->metrics.appends_active_time);
        // Reserve space. This will be a NOP if we already
        // have sufficient space.
        this->build_table_buffer->ReserveTable(in_table);
        uint64_t append_rows_sum = std::accumulate(
            append_rows.begin(), append_rows.end(), (uint64_t)0);
        this->metrics.appends_active_nrows += append_rows_sum;
        // Now append the rows. This will always
        // succeed since we've reserved space upfront.
        this->build_table_buffer->UnsafeAppendBatch(in_table, append_rows,
                                                    append_rows_sum);
        append_timer.finalize();
    } else {
        // Append into the ChunkedTableBuilder
        time_pt start = start_timer();
        size_t num_append_rows =
            std::accumulate(append_rows.begin(), append_rows.end(), (size_t)0);
        this->build_table_buffer_chunked->AppendBatch(in_table, append_rows,
                                                      num_append_rows, 0);
        this->metrics.appends_inactive_time += end_timer(start);
        this->metrics.appends_inactive_nrows += num_append_rows;
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
            this->req_extended_group_info, this->metrics, this->op_pool,
            this->op_mm, this->op_scratch_pool, this->op_scratch_mm);

    std::shared_ptr<GroupbyPartition> new_part2 =
        std::make_shared<GroupbyPartition>(
            this->num_top_bits + 1, (this->top_bitmask << 1) + 1,
            this->build_table_schema, this->separate_out_cols_schema,
            this->n_keys, this->build_table_dict_builders, this->col_sets,
            this->f_in_offsets, this->f_in_cols, this->f_running_value_offsets,
            false, this->accumulate_before_update,
            this->req_extended_group_info, this->metrics, this->op_pool,
            this->op_mm, this->op_scratch_pool, this->op_scratch_mm);

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

        start = start_timer();
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
        this->metrics.repartitioning_active_part1_append_time +=
            end_timer(start);
        this->metrics.repartitioning_active_part1_append_nrows +=
            append_partition1_sum;

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

        start = start_timer();
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
        this->metrics.repartitioning_active_part2_append_time +=
            end_timer(start);
        this->metrics.repartitioning_active_part2_append_nrows +=
            (rows_to_insert - append_partition1_sum);

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
        this->metrics.repartitioning_inactive_pop_chunk_n_chunks +=
            this->build_table_buffer_chunked->chunks.size();

        while (!this->build_table_buffer_chunked->chunks.empty()) {
            time_pt start = start_timer();
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked->PopChunk();
            this->metrics.repartitioning_inactive_pop_chunk_time +=
                end_timer(start);
            // Compute partitioning hashes.
            // TODO XXX Allocate the hashes buffer once (set size to CTB's
            // chunk-size) and reuse it across all chunks.
            start = start_timer();
            std::shared_ptr<uint32_t[]> build_table_partitioning_hashes_chunk =
                hash_keys_table(build_table_chunk, this->n_keys,
                                SEED_HASH_PARTITION, false, false, dict_hashes);
            this->metrics.repartitioning_part_hashing_time += end_timer(start);
            this->metrics.repartitioning_part_hashing_nrows +=
                build_table_chunk->nrows();

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
        ScopedTimer append_timer(this->metrics.appends_active_time);
        this->build_table_buffer->ReserveTable(
            *(this->build_table_buffer_chunked));
        append_timer.finalize();

        time_pt start;
        this->metrics.finalize_activate_pin_chunk_n_chunks +=
            this->build_table_buffer_chunked->chunks.size();
        // This will work without error because we've already allocated
        // all the required space:
        while (!this->build_table_buffer_chunked->chunks.empty()) {
            start = start_timer();
            auto [build_table_chunk, build_table_nrows_chunk] =
                this->build_table_buffer_chunked->PopChunk();
            this->metrics.finalize_activate_pin_chunk_time += end_timer(start);

            start = start_timer();
            this->build_table_buffer->UnsafeAppendBatch(build_table_chunk);
            this->metrics.appends_active_time += end_timer(start);
            this->metrics.appends_active_nrows += build_table_nrows_chunk;
        }

        // Free the chunked buffer state entirely since it's not needed anymore.
        this->build_table_buffer_chunked.reset();
    } else {
        // Just call UpdateGroupsAndCombine on each chunk. Note that
        // we cannot pop the chunks since we need to keep them around
        // in case we need to re-partition and retry.
        time_pt start_pin = start_timer();
        time_pt start_hash;
        for (const auto& chunk : *(this->build_table_buffer_chunked)) {
            this->metrics.finalize_activate_pin_chunk_time +=
                end_timer(start_pin);
            this->metrics.finalize_activate_pin_chunk_n_chunks++;
            // By definition, this is a small chunk, so we don't need to
            // track this allocation and can consider this scratch memory
            // usage.
            // TODO XXX Cache the allocation for these hashes by making
            // an allocation for the chunk size (active_chunk_capacity)
            // and reusing that buffer for all chunks.
            start_hash = start_timer();
            std::shared_ptr<uint32_t[]> chunk_hashes_groupby = hash_keys_table(
                chunk, this->n_keys, SEED_HASH_GROUPBY_SHUFFLE, false, false);
            this->metrics.finalize_activate_groupby_hashing_time +=
                end_timer(start_hash);
            // Treat the partition as active temporarily.
            // This step can fail. If it does, we can repartition and retry
            // safely since the CTB still has all the original data.
            this->UpdateGroupsAndCombine</*is_active*/ true>(
                chunk, chunk_hashes_groupby);
            start_pin = start_timer();
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
    const std::vector<bool>& cols_to_keep_bitmask, size_t n_sort_keys,
    ChunkedTableBuilder& output_buffer) {
    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    this->ActivatePartition();

    // MRNF only supports a single colset.
    assert(this->col_sets.size() == 1);
    ScopedTimer mrnf_timer(this->metrics.finalize_compute_mrnf_time);
    auto [out_bitmask, n_bits_set] = compute_local_mrnf(
        this->build_table_buffer->data_table, this->col_sets[0], n_sort_keys,
        this->n_keys, this->metrics.finalize_update_metrics,
        this->op_scratch_pool, this->op_scratch_mm);
    mrnf_timer.finalize();

    // cols_to_keep_bitmask to determine
    // the columns to skip from build_table_buffer.
    size_t n_cols = this->build_table_buffer->data_table->columns.size();
    std::vector<std::shared_ptr<array_info>> cols_to_keep;

    assert(cols_to_keep_bitmask.size() == n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        if (cols_to_keep_bitmask[i]) {
            cols_to_keep.push_back(
                this->build_table_buffer->data_table->columns[i]);
        }
    }
    std::shared_ptr<table_info> data_table_w_cols_to_keep =
        std::make_shared<table_info>(cols_to_keep);

    // Append this "pruned" table to the output buffer using the bitmask.
    output_buffer.AppendBatch(data_table_w_cols_to_keep, std::move(out_bitmask),
                              n_bits_set, 0);

    // Since we have added the output to the output buffer, we don't need the
    // build state anymore and can release that memory.
    this->ClearBuildState();
}

void GroupbyPartition::FinalizeWindow(
    const std::vector<bool>& cols_to_keep_bitmask, size_t n_sort_keys,
    ChunkedTableBuilder& output_buffer,
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builder,
    std::vector<int32_t> f_in_offsets, std::vector<int32_t> f_in_cols) {
    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    this->ActivatePartition();

    // Window only supports a single colset.
    assert(this->col_sets.size() == 1);
    ScopedTimer window_timer(this->metrics.finalize_window_compute_time);
    std::vector<std::shared_ptr<array_info>> window_cols = compute_local_window(
        this->build_table_buffer->data_table, out_dict_builder,
        this->col_sets[0], n_sort_keys, this->n_keys,
        this->metrics.finalize_update_metrics, f_in_offsets, f_in_cols,
        this->op_scratch_pool, this->op_scratch_mm);
    window_timer.finalize();

    // Use partition_by_cols_to_keep and order_by_cols_to_keep to determine
    // the columns to skip from build_table_buffer.
    size_t n_cols = this->build_table_buffer->data_table->columns.size();
    std::vector<std::shared_ptr<array_info>> cols_to_keep;

    assert(cols_to_keep_bitmask.size() == n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        if (cols_to_keep_bitmask[i]) {
            cols_to_keep.push_back(
                this->build_table_buffer->data_table->columns[i]);
        }
    }
    for (auto& it : window_cols) {
        cols_to_keep.push_back(it);
    }
    std::shared_ptr<table_info> data_table_w_cols_to_keep =
        std::make_shared<table_info>(cols_to_keep);

    // Append the table to the output buffer.
    output_buffer.AppendBatch(data_table_w_cols_to_keep);

    // Since we have added the output to the output buffer, we don't need the
    // build state anymore and can release that memory.
    this->ClearBuildState();
}

std::shared_ptr<table_info> GroupbyPartition::Finalize() {
    // Make sure this partition is active. This is idempotent
    // and hence a NOP if the partition is already active.
    ScopedTimer activate_timer(this->metrics.finalize_activate_partition_time);
    this->ActivatePartition();
    activate_timer.finalize();

    std::shared_ptr<table_info> out_table;
    if (accumulate_before_update) {
        // NOTE: Since the allocations made during the update and eval steps are
        // not required for re-partitioning (in case there is one), we will use
        // the scratch portion of the pool for them.

        // Get update table with the running values:
        ScopedTimer update_timer(this->metrics.finalize_get_update_table_time);
        std::shared_ptr<table_info> update_table =
            get_update_table</*is_acc_case*/ true>(
                this->build_table_buffer->data_table, this->n_keys,
                this->col_sets, this->f_in_offsets, this->f_in_cols,
                this->req_extended_group_info,
                this->metrics.finalize_update_metrics, this->op_scratch_pool,
                this->op_scratch_mm);
        update_timer.finalize();
        // Call eval on these running values to get the final output.
        this->metrics.finalize_eval_nrows += update_table->nrows();
        ScopedTimer eval_timer(this->metrics.finalize_eval_time);
        out_table = eval_groupby_funcs_helper</*is_acc_case*/ true>(
            this->f_running_value_offsets, this->col_sets, update_table,
            this->n_keys, this->separate_out_cols->data_table,
            this->op_scratch_pool, this->op_scratch_mm);
        eval_timer.finalize();
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
        this->metrics.finalize_eval_nrows += combine_data->nrows();
        time_pt start_eval = start_timer();
        out_table = eval_groupby_funcs_helper</*is_acc_case*/ false>(
            this->f_running_value_offsets, this->col_sets, combine_data,
            this->n_keys, this->separate_out_cols->data_table);
        this->metrics.finalize_eval_time += end_timer(start_eval);
    }

    // Since we have generated the output, we don't need the build state
    // anymore, so we can release that memory.
    this->ClearBuildState();

    return out_table;
}

#pragma endregion  // GroupbyPartition
/* ------------------------------------------------------------------------ */

/* ------------------- GroupbyIncrementalShuffleMetrics ------------------- */
#pragma region  // GroupbyIncrementalShuffleMetrics

void GroupbyIncrementalShuffleMetrics::add_to_metrics(
    std::vector<MetricBase>& metrics, bool accumulate_before_update) {
    metrics.emplace_back(StatMetric("shuffle_n_ht_resets", this->n_ht_reset));
    metrics.emplace_back(
        StatMetric("shuffle_peak_ht_size_bytes", this->peak_ht_size_bytes));
    metrics.emplace_back(
        StatMetric("shuffle_n_hashes_reset", this->n_hashes_reset));
    metrics.emplace_back(StatMetric("shuffle_peak_hashes_size_bytes",
                                    this->peak_hashes_size_bytes));
    metrics.emplace_back(TimerMetric("shuffle_nunique_hll_hashing_time",
                                     this->nunique_hll_hashing_time));
    metrics.emplace_back(TimerMetric("shuffle_hll_time", this->hll_time));
    metrics.emplace_back(
        StatMetric("shuffle_n_local_reductions", this->n_local_reductions));
    metrics.emplace_back(StatMetric("shuffle_local_reduction_input_nrows",
                                    this->local_reduction_input_nrows));
    metrics.emplace_back(StatMetric("shuffle_local_reduction_output_nrows",
                                    this->local_reduction_output_nrows));
    metrics.emplace_back(TimerMetric("shuffle_local_reduction_time",
                                     this->local_reduction_time));
    metrics.emplace_back(
        TimerMetric("shuffle_local_reduction_mrnf_colset_update_time",
                    this->local_reduction_mrnf_metrics.colset_update_time));
    metrics.emplace_back(TimerMetric(
        "shuffle_local_reduction_mrnf_hashing_time",
        this->local_reduction_mrnf_metrics.grouping_metrics.hashing_time));
    metrics.emplace_back(TimerMetric(
        "shuffle_local_reduction_mrnf_grouping_time",
        this->local_reduction_mrnf_metrics.grouping_metrics.grouping_time));
    metrics.emplace_back(TimerMetric(
        "shuffle_local_reduction_mrnf_hll_time",
        this->local_reduction_mrnf_metrics.grouping_metrics.hll_time));
    if (!accumulate_before_update) {
        metrics.emplace_back(StatMetric("shuffle_n_possible_ht_reductions",
                                        this->n_possible_shuffle_reductions));
        metrics.emplace_back(
            StatMetric("shuffle_n_ht_reductions", this->n_shuffle_reductions));
        metrics.emplace_back(TimerMetric("shuffle_update_logical_ht_time",
                                         this->shuffle_update_logical_ht_time));
        metrics.emplace_back(TimerMetric("shuffle_combine_input_time",
                                         this->shuffle_combine_input_time));
        metrics.emplace_back(
            StatMetric("shuffle_update_logical_ht_and_combine_nrows",
                       this->shuffle_update_logical_ht_and_combine_nrows));
        metrics.emplace_back(
            TimerMetric("shuffle_hll_time", this->shuffle_hll_time));
        metrics.emplace_back(StatMetric("shuffle_n_pre_reduction_buffer_reset",
                                        this->n_pre_reduction_buffer_reset));
        metrics.emplace_back(StatMetric("shuffle_n_pre_reduction_hashes_reset",
                                        this->n_pre_reduction_hashes_reset));
        // Note: We need to keep the name distinct from the
        // IncrementalShuffleBuffer name.
        metrics.emplace_back(TimerMetric("shuffle_agg_buffer_append_time",
                                         this->shuffle_agg_buffer_append_time));
    }
}

#pragma endregion  // GroupbyIncrementalShuffleMetrics
/* ------------------------------------------------------------------------ */

/* -------------------- GroupbyIncrementalShuffleState -------------------- */
#pragma region  // GroupbyIncrementalShuffleState

#define DEFAULT_AGG_REDUCTION_THRESHOLD 0.85

GroupbyIncrementalShuffleState::GroupbyIncrementalShuffleState(
    const std::shared_ptr<bodo::Schema> shuffle_table_schema_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
    const uint64_t mrnf_n_sort_cols_, const uint64_t n_keys_,
    const uint64_t& curr_iter_, int64_t& sync_freq_, int64_t op_id_,
    const bool nunique_only_, const AggregationType agg_type_,
    const std::vector<int32_t>& f_running_value_offsets_,
    const bool accumulate_before_update_)
    : IncrementalShuffleState(shuffle_table_schema_, dict_builders_, n_keys_,
                              curr_iter_, sync_freq_, op_id_),
      hash_table(std::make_unique<shuffle_hash_table_t>(
          0, HashGroupbyTable<false>(nullptr, this),
          KeyEqualGroupbyTable<false>(nullptr, this, this->n_keys))),
      pre_reduction_table_buffer(std::make_unique<TableBuildBuffer>(
          this->schema, this->dict_builders)),
      col_sets(col_sets_),
      mrnf_n_sort_cols(mrnf_n_sort_cols_),
      nunique_only(nunique_only_),
      agg_type(agg_type_),
      f_running_value_offsets(f_running_value_offsets_),
      accumulate_before_update(accumulate_before_update_) {
    running_reduction_hll = hll::HyperLogLog(HLL_SIZE);
    double agg_reduction_threshold = DEFAULT_AGG_REDUCTION_THRESHOLD;
    char* agg_reduction_threshold_env_ =
        std::getenv("BODO_STREAM_GROUPBY_SHUFFLE_REDUCTION_THRESHOLD");
    if (agg_reduction_threshold_env_) {
        agg_reduction_threshold = std::stod(agg_reduction_threshold_env_);
        if (agg_reduction_threshold < 0.0) {
            agg_reduction_threshold = 0.0;
        } else if (agg_reduction_threshold > 1.0) {
            agg_reduction_threshold = 1.0;
        }
    }
    this->agg_reduction_threshold = agg_reduction_threshold;
}

#undef DEFAULT_AGG_REDUCTION_THRESHOLD

void GroupbyIncrementalShuffleState::Finalize() {
    this->hash_table.reset();
    this->groupby_hashes.resize(0);
    this->groupby_hashes.shrink_to_fit();
    IncrementalShuffleState::Finalize();
}

void GroupbyIncrementalShuffleState::UpdateGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby) {
    // set state batch input
    this->in_table = in_table;
    this->in_table_hashes = batch_hashes_groupby;
    time_pt start = start_timer();
    // Reserve space in buffers for potential new groups.
    // Note that if any of the running values are strings, they always
    // go through the accumulate path.
    this->table_buffer->ReserveTable(in_table);

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
    int64_t shuffle_init_start_row = this->next_group;

    // Add new groups and get group mappings for input batch
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        update_groups_helper(*(this->table_buffer), this->groupby_hashes,
                             *(this->hash_table), this->next_group,
                             this->n_keys, shuffle_grp_info, in_table,
                             batch_hashes_groupby, i_row);
    }
    size_t row_count = in_table->nrows();
    this->metrics.shuffle_update_logical_ht_time += end_timer(start);
    start = start_timer();
    // Combine existing (and new) keys using the input batch
    combine_input_table_helper(
        in_table, shuffle_grp_info, this->table_buffer->data_table,
        this->f_running_value_offsets, this->col_sets, shuffle_init_start_row);
    this->metrics.shuffle_combine_input_time += end_timer(start);
    this->metrics.shuffle_update_logical_ht_and_combine_nrows += row_count;

    int64_t new_groups = this->next_group - shuffle_init_start_row;
    this->UpdateAppendBatchSize(in_table->nrows(), new_groups);

    // Reset temporary references
    this->in_table.reset();
    this->in_table_hashes.reset();
}

bool GroupbyIncrementalShuffleState::ShouldShuffleAfterProcessing(
    bool is_last) {
    if (this->accumulate_before_update) {
        // Just look at the default buffer handling.
        return IncrementalShuffleState::ShouldShuffleAfterProcessing(is_last);
    } else {
        // Check if we need to do a reduction before shuffling.
        bool must_shuffle =
            is_last &&
            (this->table_buffer->data_table->nrows() > 0 ||
             this->pre_reduction_table_buffer->data_table->nrows() > 0);
        int64_t hash_table_size = table_local_memory_size(
            this->table_buffer->data_table, /*include_dict_size*/ false);
        int64_t pre_reduction_table_size = table_local_memory_size(
            this->pre_reduction_table_buffer->data_table,
            /*include_dict_size*/ false);
        bool exceed_table_size = (hash_table_size + pre_reduction_table_size) >=
                                 this->shuffle_threshold;
        bool should_shuffle = must_shuffle || exceed_table_size;
        if (should_shuffle &&
            this->pre_reduction_table_buffer->data_table->nrows() > 0) {
            // Indicate that we checked if we should insert into the hash table.
            this->metrics.n_possible_shuffle_reductions++;
            time_pt start = start_timer();
            // We need to check if we should perform a local reduction
            // on the data. To do this we check a running HLL.
            this->running_reduction_hll.addAll(this->pre_reduction_hashes);
            double hll_estimate = this->running_reduction_hll.estimate();
            this->metrics.shuffle_hll_time += end_timer(start);
            // Only look at the estimated new unique values from the
            // pre-reduction table.
            double insert_estimate =
                hll_estimate - this->table_buffer->data_table->nrows();
            double expected_uniqueness =
                insert_estimate /
                this->pre_reduction_table_buffer->data_table->nrows();
            bool insert_into_hash_table =
                expected_uniqueness < this->agg_reduction_threshold;
            // Decide if we should insert into the hash table.
            if (insert_into_hash_table) {
                // Indicate that we have decided to insert into the hash table.
                this->metrics.n_shuffle_reductions++;
                // Copy hashes to be API compliant with our hash table.
                // TODO: Remove.
                std::shared_ptr<uint32_t[]> hashes =
                    std::make_shared<uint32_t[]>(
                        this->pre_reduction_hashes.size());
                memcpy(hashes.get(), this->pre_reduction_hashes.data(),
                       sizeof(uint32_t) * this->pre_reduction_hashes.size());
                this->UpdateGroupsAndCombine(
                    this->pre_reduction_table_buffer->data_table, hashes);
                // Recompute should_shuffle since we are inserting into the hash
                // table.
                should_shuffle =
                    IncrementalShuffleState::ShouldShuffleAfterProcessing(
                        is_last);
            } else {
                // We need to combine the tables.
                if (this->table_buffer->data_table->nrows() == 0) {
                    // Swap the tables if the hash table is empty to avoid
                    // an unnecessary copy.
                    std::swap(this->table_buffer,
                              this->pre_reduction_table_buffer);
                } else {
                    time_pt start = start_timer();
                    this->table_buffer->ReserveTable(
                        this->pre_reduction_table_buffer->data_table);
                    this->table_buffer->UnsafeAppendBatch(
                        this->pre_reduction_table_buffer->data_table);
                    this->metrics.shuffle_agg_buffer_append_time +=
                        end_timer(start);
                }
            }
            this->pre_reduction_table_buffer->Reset();
            this->pre_reduction_hashes.resize(0);
            if (should_shuffle) {
                // Reset the HLL only when we are emptying the hash table.
                // We maintain the HLL information across multiple reductions
                // so we can look at uniqueness information across both the
                // hash table and the pre-reduction table.
                this->running_reduction_hll.clear();
            }
        }
        return should_shuffle;
    }
}

std::tuple<
    std::shared_ptr<table_info>,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
    std::shared_ptr<uint32_t[]>, std::unique_ptr<uint8_t[]>>
GroupbyIncrementalShuffleState::GetShuffleTableAndHashes() {
    auto [shuffle_table, dict_hashes, shuffle_hashes, always_null] =
        IncrementalShuffleState::GetShuffleTableAndHashes();
    assert(always_null == nullptr);
    // drop shuffle table duplicate rows if there are a lot of duplicates
    // only possible for nunique-only cases
    if (this->nunique_only || this->agg_type == AggregationType::MRNF) {
        int64_t shuffle_nrows = shuffle_table->nrows();
        std::shared_ptr<uint32_t[]> hashes;
        if (this->nunique_only) {
            // estimate number of uniques using key/value hashes
            time_pt start_hash = start_timer();
            std::shared_ptr<uint32_t[]> key_value_hashes =
                std::make_unique<uint32_t[]>(shuffle_nrows);
            // reusing shuffle_hashes for keys to make the initial check cheaper
            // for code path without drop duplicates
            std::memcpy(key_value_hashes.get(), shuffle_hashes.get(),
                        sizeof(uint32_t) * shuffle_nrows);
            for (size_t col = this->n_keys; col < shuffle_table->ncols();
                 col++) {
                hash_array_combine(key_value_hashes.get(),
                                   shuffle_table->columns[col], shuffle_nrows,
                                   SEED_HASH_PARTITION,
                                   /*global_dict_needed=*/false,
                                   /*is_parallel*/ true);
            }
            this->metrics.nunique_hll_hashing_time += end_timer(start_hash);
            hashes = key_value_hashes;
        } else {
            hashes = shuffle_hashes;
        }

        time_pt start_hll = start_timer();
        size_t nunique_keyval_hashes =
            get_nunique_hashes(hashes, shuffle_nrows, /*is_parallel*/ true);
        this->metrics.hll_time += end_timer(start_hll);

        // local reduction if output will be less than half the size of
        // input (rough heuristic, TODO: tune)
        if ((2 * nunique_keyval_hashes) < static_cast<size_t>(shuffle_nrows)) {
            this->metrics.n_local_reductions++;
            this->metrics.local_reduction_input_nrows += shuffle_nrows;
            time_pt start_loc_red = start_timer();
            std::unique_ptr<uint8_t[]> row_inclusion_bitmask;
            size_t n_groups = 0;
            if (this->nunique_only) {
                // drop duplicates
                bodo::vector<int64_t> ListIdx = drop_duplicates_table_helper(
                    shuffle_table, shuffle_table->ncols(), 0, 1, false, false,
                    /*drop_duplicates_dict=*/false, hashes);
                row_inclusion_bitmask = std::make_unique<uint8_t[]>(
                    arrow::bit_util::BytesForBits(shuffle_table->nrows()));
                memset(row_inclusion_bitmask.get(), 0,
                       arrow::bit_util::BytesForBits(shuffle_table->nrows()));
                for (auto idx : ListIdx) {
                    arrow::bit_util::SetBit(row_inclusion_bitmask.get(), idx);
                }
                n_groups = ListIdx.size();
            } else {
                // MRNF
                std::tie(row_inclusion_bitmask, n_groups) = compute_local_mrnf(
                    shuffle_table, this->col_sets[0], this->mrnf_n_sort_cols,
                    this->n_keys, this->metrics.local_reduction_mrnf_metrics);
            }
            this->metrics.local_reduction_time += end_timer(start_loc_red);
            this->metrics.local_reduction_output_nrows += n_groups;
            return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes,
                                   std::move(row_inclusion_bitmask));
        }
    }
    return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes, nullptr);
}

void GroupbyIncrementalShuffleState::ResetAfterShuffle() {
    size_t ht_size = this->hash_table->get_allocator().size();
    this->metrics.peak_ht_size_bytes =
        std::max(static_cast<MetricBase::StatValue>(ht_size),
                 this->metrics.peak_ht_size_bytes);
    if (ht_size > MAX_SHUFFLE_HASHTABLE_SIZE) {
        // If the shuffle hash table is too large, reset it.
        // This shouldn't happen often in practice, but is a safeguard.
        this->hash_table.reset();
        this->hash_table = std::make_unique<shuffle_hash_table_t>(
            0, HashGroupbyTable<false>(nullptr, this),
            KeyEqualGroupbyTable<false>(nullptr, this, this->n_keys));
        this->metrics.n_ht_reset++;
    }
    size_t hashes_size = this->groupby_hashes.get_allocator().size();
    this->metrics.peak_hashes_size_bytes =
        std::max(static_cast<MetricBase::StatValue>(hashes_size),
                 this->metrics.peak_hashes_size_bytes);
    if (hashes_size > MAX_SHUFFLE_TABLE_SIZE) {
        // If the shuffle hashes vector is too large, reallocate it to the
        // maximum size
        this->groupby_hashes.resize(MAX_SHUFFLE_TABLE_SIZE / sizeof(uint32_t));
        this->groupby_hashes.shrink_to_fit();
        this->metrics.n_hashes_reset++;
    }
    if (!this->accumulate_before_update) {
        // Check if we need to reset our pre-reduction buffers.
        size_t pre_reduction_hashes_size =
            this->pre_reduction_hashes.get_allocator().size();
        if (pre_reduction_hashes_size > MAX_SHUFFLE_TABLE_SIZE) {
            // If the hashes vector is too large, reallocate it to the
            // maximum size
            this->pre_reduction_hashes.resize(MAX_SHUFFLE_TABLE_SIZE /
                                              sizeof(uint32_t));
            this->groupby_hashes.shrink_to_fit();
            this->metrics.n_pre_reduction_hashes_reset++;
        }
        size_t capacity = this->pre_reduction_table_buffer->EstimatedSize();
        int64_t buffer_used_size = table_local_memory_size(
            this->pre_reduction_table_buffer->data_table, false);
        if (capacity >
                (SHUFFLE_BUFFER_CUTOFF_MULTIPLIER * this->shuffle_threshold) &&
            (capacity * SHUFFLE_BUFFER_MIN_UTILIZATION) > buffer_used_size) {
            this->pre_reduction_table_buffer =
                std::make_unique<TableBuildBuffer>(this->schema,
                                                   this->dict_builders);
            this->metrics.n_pre_reduction_buffer_reset++;
        }
    }

    this->next_group = 0;
    this->hash_table->clear();
    this->groupby_hashes.resize(0);
    IncrementalShuffleState::ResetAfterShuffle();
}

#pragma endregion  // GroupbyIncrementalShuffleState
/* ------------------------------------------------------------------------ */

/* -------------------------- GroupbyOutputState -------------------------- */
#pragma region  // GroupbyOutputState

GroupbyOutputState::GroupbyOutputState(
    const std::shared_ptr<bodo::Schema>& schema,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    size_t chunk_size, size_t max_resize_count_for_variable_size_dtypes,
    bool enable_work_stealing_)
    : dict_builders(dict_builders_),
      buffer(schema, dict_builders, chunk_size,
             max_resize_count_for_variable_size_dtypes),
      enable_work_stealing(enable_work_stealing_) {
    MPI_Comm_size(MPI_COMM_WORLD, &(this->n_pes));
    MPI_Comm_rank(MPI_COMM_WORLD, &(this->myrank));

    // TODO Remove this once we make this async.
    char* work_stealing_sync_iter_env_ =
        std::getenv("BODO_STREAM_GROUPBY_OUTPUT_WORK_STEALING_SYNC_ITER");
    if (work_stealing_sync_iter_env_) {
        this->work_stealing_sync_iter = std::stoi(work_stealing_sync_iter_env_);
    }

    char* work_stealing_threshold_timer_env_ = std::getenv(
        "BODO_STREAM_GROUPBY_OUTPUT_WORK_STEALING_TIME_THRESHOLD_SECONDS");
    if (work_stealing_threshold_timer_env_) {
        // Convert from seconds to microseconds.
        this->work_stealing_timer_threshold_us =
            std::stoi(work_stealing_threshold_timer_env_) * 1000 * 1000;
    }

    char* work_stealing_debug_env_ =
        std::getenv("BODO_STREAM_GROUPBY_DEBUG_OUTPUT_WORK_STEALING");
    if (work_stealing_debug_env_) {
        this->debug_work_stealing =
            (std::strcmp(work_stealing_debug_env_, "1") == 0);
    }

    // TODO Remove this and do a heuristic calculation instead.
    char* work_stealing_max_send_recv_batches_env_ = std::getenv(
        "BODO_STREAM_GROUPBY_WORK_STEALING_MAX_SEND_RECV_BATCHES_PER_RANK");
    if (work_stealing_max_send_recv_batches_env_) {
        this->work_stealing_max_batched_send_recv_per_rank =
            std::stoi(work_stealing_max_send_recv_batches_env_);
    }

    // Use a 50% threshold by default.
    double num_ranks_frac_ = 0.5;
    char* work_stealing_percent_ranks_done_threshold_env_ = std::getenv(
        "BODO_STREAM_GROUPBY_WORK_STEALING_PERCENT_RANKS_DONE_THRESHOLD");
    if (work_stealing_percent_ranks_done_threshold_env_) {
        num_ranks_frac_ = static_cast<double>(
            std::stoi(work_stealing_percent_ranks_done_threshold_env_) / 100.0);
    }
    this->work_stealing_num_ranks_done_threshold =
        std::floor(num_ranks_frac_ * this->n_pes);
    // The threshold should be at least 1.
    this->work_stealing_num_ranks_done_threshold = std::max(
        static_cast<uint64_t>(1), this->work_stealing_num_ranks_done_threshold);

    if (this->enable_work_stealing) {
        CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &this->mpi_comm),
                  "GroupbyOutputState::GroupbyOutputState: MPI error on "
                  "MPI_Comm_dup:");
    }
}

GroupbyOutputState::~GroupbyOutputState() {
    if (this->enable_work_stealing) {
        // Make sure "done" message is sent to avoid MPI state issues
        MPI_Wait(&this->done_request, MPI_STATUS_IGNORE);
        if (this->myrank == 0) {
            // Make sure all "done" messages are received on rank 0 to avoid MPI
            // state issues
            for (int i = 0; i < this->n_pes; i++) {
                if (!this->done_received[i]) {
                    MPI_Wait(&this->done_recv_requests[i], MPI_STATUS_IGNORE);
                }
            }
        }
        MPI_Comm_free(&this->mpi_comm);
    }
}

void GroupbyOutputState::Finalize() {
    if (this->enable_work_stealing) {
        // If work stealing is enabled, we might need to add more output to the
        // buffer later on, so we will only finalize the active chunk for now.
        this->buffer.FinalizeActiveChunk();
    } else {
        this->buffer.Finalize();
    }
}

std::tuple<std::shared_ptr<table_info>, bool> GroupbyOutputState::PopBatch(
    const bool produce_output) {
    std::shared_ptr<table_info> out_batch;
    if (!produce_output) {
        out_batch = this->buffer.dummy_output_chunk;
    } else {
        int64_t chunk_size;
        // TODO[BSE-645]: Prune unused columns at this point.
        // Note: We always finalize the active chunk the build step so we don't
        // need to finalize here.
        std::tie(out_batch, chunk_size) = this->buffer.PopChunk();
    }

    // NOTE: work stealing is called after buffer.PopChunk() to know updated
    // total_remaining value for "done" message before returning with
    // out_is_last=true
    this->StealWorkIfNeeded();

    // If work stealing is disabled or is done (was done this iter), then we
    // just need to check if our local output buffer is empty). If work stealing
    // is not done yet, then we will only output is_last when all ranks are done
    // (since work-stealing may happen in the future).
    bool out_is_last =
        (!this->enable_work_stealing || this->work_stealing_done) &&
        (this->buffer.total_remaining == 0);
    return std::tuple(out_batch, out_is_last);
}

void GroupbyOutputState::StealWorkIfNeeded() {
    // If work-stealing not allowed, or if it has already happened, simply
    // return.
    if (!this->enable_work_stealing) {
        return;
    }

    // Parallel work stealing decision approach:
    // Rank 0 broadcasts a work stealing "command" to all ranks eventually,
    // once and only once.
    // It decides whether to perform work redistribution by tracking how many
    // ranks are done and using timers. All ranks send a "done" message to rank
    // 0 when their buffers are empty. The actual work redistribution is
    // synchronous and uses blocking collectives for simplicity since it happens
    // once on all ranks.

    // Start work stealing "command" bcast on non-zero ranks in the first
    // iteration
    if (this->myrank != 0 && !this->steal_work_bcast_started) {
        CHECK_MPI(
            MPI_Ibcast(&this->should_steal_work, 1, MPI_C_BOOL, 0,
                       this->mpi_comm, &this->steal_work_bcast_request),
            "GroupbyOutputState::StealWorkIfNeeded: MPI error on MPI_Ibcast:");
        this->steal_work_bcast_started = true;
    }

    // Send "done" signal to rank 0 (all ranks send irrespective of work
    // stealing status to simplify the code)
    if (this->buffer.total_remaining == 0 && !this->done_sent) {
        this->done_sent = true;
        CHECK_MPI(
            MPI_Isend(&this->done_sent, 1, MPI_C_BOOL, 0, 0, this->mpi_comm,
                      &this->done_request),
            "GroupbyOutputState::StealWorkIfNeeded: MPI error on MPI_Isend:");
    }

    // Return if work stealing command bcast already received from rank 0
    if (this->work_stealing_done) {
        return;
    }

    if (this->myrank == 0) {
        this->manage_work_stealing_rank_0();
    }
    // Test every 10 iterations to reduce MPI call overheads
    else if ((this->iter + 1) % 10 == 0) {
        int flag = 0;
        CHECK_MPI(
            MPI_Test(&this->steal_work_bcast_request, &flag, MPI_STATUS_IGNORE),
            "GroupbyOutputState::StealWorkIfNeeded: MPI error on MPI_Test:");
        if (flag) {
            this->steal_work_bcast_done = true;
        }
    }

    // Process received work stealing command (perform work redistribution if
    // rank 0 decided)
    if (this->steal_work_bcast_done) {
        if (this->should_steal_work) {
            if (this->myrank == 0 && this->debug_work_stealing) {
                std::cerr
                    << "[DEBUG][GroupbyOutputState] Starting work stealing."
                    << std::endl;
            }
            this->RedistributeWork();
            this->performed_work_redistribution = true;
            // Finalize output buffer now that all work-stealing is done.
            this->buffer.Finalize();
            if (this->myrank == 0 && this->debug_work_stealing) {
                std::cerr
                    << "[DEBUG][GroupbyOutputState] Done with work stealing."
                    << std::endl;
            }
        }
        this->work_stealing_done = true;
    }
}

void GroupbyOutputState::manage_work_stealing_rank_0() {
    // Post Irecv for done messages from all ranks in first iteration
    if (!this->recvs_posted) {
        this->done_recv_buff = std::make_unique<bool[]>(this->n_pes);
        for (int i = 0; i < this->n_pes; i++) {
            MPI_Request recv_req;
            CHECK_MPI(MPI_Irecv(&this->done_recv_buff[i], 1, MPI_C_BOOL, i, 0,
                                this->mpi_comm, &recv_req),
                      "GroupbyOutputState::manage_work_stealing_rank_0: "
                      "MPI error on MPI_Irecv:");
            this->done_recv_requests.push_back(recv_req);
        }
        done_received.resize(n_pes, false);
        this->recvs_posted = true;
    }

    if (this->command_sent) {
        return;
    }

    // Avoid MPI_Test call overheads in every iteration
    if ((this->iter + 1) % this->work_stealing_sync_iter != 0) {
        return;
    }

    // Check for new done messages
    for (int i = 0; i < this->n_pes; i++) {
        if (!this->done_received[i]) {
            int flag = 0;
            CHECK_MPI(MPI_Test(&this->done_recv_requests[i], &flag,
                               MPI_STATUS_IGNORE),
                      "GroupbyOutputState::manage_work_stealing_rank_0: "
                      "MPI error on MPI_Test:");
            if (flag) {
                this->done_received[i] = true;
                this->num_ranks_done++;
            }
        }
    }

    // All ranks done, broadcast "no work redistribution command"
    if (this->num_ranks_done == this->n_pes) {
        assert(this->should_steal_work == false);
        CHECK_MPI(
            MPI_Ibcast(&this->should_steal_work, 1, MPI_C_BOOL, 0,
                       this->mpi_comm, &this->steal_work_bcast_request),
            "GroupbyOutputState::manage_work_stealing_rank_0: MPI error on "
            "MPI_Ibcast:");
        this->steal_work_bcast_done = true;
        this->command_sent = true;
        return;
    }

    // If timer hasn't started and if the threshold fraction of ranks are done,
    // start the timer. Store the timer on rank0.
    if (!this->work_steal_timer_started &&
        (static_cast<uint64_t>(this->num_ranks_done) >=
         this->work_stealing_num_ranks_done_threshold)) {
        this->work_steal_timer_started = true;
        this->work_steal_start_time = start_timer();
        if (this->debug_work_stealing) {
            std::cerr << "[DEBUG][GroupbyOutputState] Started work-stealing "
                         "timer since "
                      << this->num_ranks_done << " of " << this->n_pes
                      << " ranks are done outputting." << std::endl;
        }
        this->metrics.n_ranks_done_at_timer_start = this->num_ranks_done;
    }

    // If timer has already started, and rank 0 determines it's above some
    // threshold, then we will move work around.
    if (this->work_steal_timer_started &&
        end_timer(this->work_steal_start_time) >
            this->work_stealing_timer_threshold_us) {
        this->should_steal_work = true;
        this->metrics.n_ranks_done_before_work_redistribution =
            this->num_ranks_done;
        CHECK_MPI(
            MPI_Ibcast(&this->should_steal_work, 1, MPI_C_BOOL, 0,
                       this->mpi_comm, &this->steal_work_bcast_request),
            "GroupbyOutputState::manage_work_stealing_rank_0: MPI error on "
            "MPI_Ibcast:");
        this->steal_work_bcast_done = true;
        this->command_sent = true;
    }
}

std::vector<std::vector<size_t>> GroupbyOutputState::determine_redistribution(
    const std::vector<uint64_t>& num_batches_ranks) {
    int n_pes = num_batches_ranks.size();
    const uint64_t num_batches_global =
        std::reduce(num_batches_ranks.begin(), num_batches_ranks.end());
    // How far from their "fair" distribution are these ranks.
    std::vector<int64_t> diff_from_fair(n_pes);
    // Group the ranks into "senders" and "receivers".
    std::list<int> senders;
    std::list<int> receivers;
    for (int i = 0; i < n_pes; i++) {
        int64_t fair_batches =
            dist_get_node_portion(num_batches_global, n_pes, i);
        int64_t diff =
            fair_batches - static_cast<int64_t>(num_batches_ranks[i]);
        diff_from_fair[i] = diff;
        if (diff > 0) {
            receivers.emplace_back(i);
        } else if (diff < 0) {
            senders.emplace_back(i);
        }
    }

    const auto f = [&](const int& first, const int& second) -> bool {
        int64_t a = std::abs(diff_from_fair[first]);
        int64_t b = std::abs(diff_from_fair[second]);
        // Handle tie-breaks deterministically.
        return (a == b) ? (first < second) : (a < b);
    };
    // This orders the senders from from least to most data to send.
    senders.sort(f);
    // This orders the receivers from least to most capacity to
    // receive.
    receivers.sort(f);

    std::vector<std::vector<size_t>> batches_to_send(n_pes);
    for (auto& x : batches_to_send) {
        x.resize(n_pes, 0);
    }

    // Loop through the "senders" in descending order. For every "sender":
    // - Let's say the number of "receivers" (with non-0 excess capacity) is
    // n_potential_receivers and the minimum remaining capacity within these
    // groups is min_rem_capacity. Let's say the number or rows remaining to
    // send it n_rows_to_send. Assign min(min_rem_capacity,
    // n_rows_to_send/n_potential_receivers) to every receiver. If all rows
    // for this rank are assigned, then remove "sender" from the list.
    // Otherwise, repeat this process until all rows are assigned.
    // XXX TODO Make this assignment more topology aware by prioritizing sending
    // to ranks on the same shared memory host to reduce overall communication
    // overheads.
    while (senders.size() > 0) {
        int sender = senders.back();
        int min_capacity_receiver = receivers.front();
        int64_t min_recv_capacity = diff_from_fair[min_capacity_receiver];
        int64_t n_batches_to_send = -diff_from_fair[sender];
        if (n_batches_to_send <= static_cast<int64_t>(receivers.size())) {
            int64_t rem_batches = n_batches_to_send;
            std::vector<int> receivers_to_remove;
            for (const int& receiver : receivers) {
                batches_to_send[sender][receiver] += 1;
                diff_from_fair[sender] += 1;
                diff_from_fair[receiver] -= 1;
                rem_batches--;
                if (diff_from_fair[receiver] == 0) {
                    receivers_to_remove.push_back(receiver);
                }
                if (rem_batches == 0) {
                    break;
                }
            }
            for (auto x : receivers_to_remove) {
                receivers.remove(x);
            }
        } else {
            int64_t batches_ = std::min(
                min_recv_capacity,
                static_cast<int64_t>(n_batches_to_send / receivers.size()));
            std::vector<int> receivers_to_remove;
            for (const int& receiver : receivers) {
                batches_to_send[sender][receiver] += batches_;
                diff_from_fair[sender] += batches_;
                diff_from_fair[receiver] -= batches_;
                if (diff_from_fair[receiver] == 0) {
                    receivers_to_remove.push_back(receiver);
                }
            }
            for (auto x : receivers_to_remove) {
                receivers.remove(x);
            }
        }

        if (diff_from_fair[sender] == 0) {
            senders.pop_back();
        }
    }
    return batches_to_send;
}

std::vector<std::vector<size_t>>
GroupbyOutputState::determine_batched_send_counts(
    std::vector<std::vector<size_t>>& batches_to_send_overall,
    const size_t max_batches_send_recv_per_rank) {
    int n_pes = batches_to_send_overall.size();
    std::vector<size_t> batches_to_recv_this_iter(n_pes, 0);
    std::vector<std::vector<size_t>> to_send_this_iter(n_pes);
    for (auto& x : to_send_this_iter) {
        x.resize(n_pes, 0);
    }

    for (int sender = 0; sender < n_pes; sender++) {
        const size_t total_batches_left_to_send =
            std::reduce(batches_to_send_overall[sender].begin(),
                        batches_to_send_overall[sender].end());
        size_t batches_left_to_send_this_iter = std::min(
            max_batches_send_recv_per_rank, total_batches_left_to_send);

        for (int receiver = 0; receiver < n_pes; receiver++) {
            if (batches_left_to_send_this_iter == 0) {
                break;
            }
            const size_t max_batches_can_receive_this_iter =
                max_batches_send_recv_per_rank -
                batches_to_recv_this_iter[receiver];
            if (batches_to_send_overall[sender][receiver] > 0 &&
                max_batches_can_receive_this_iter > 0) {
                size_t batches_ =
                    std::min({batches_left_to_send_this_iter,
                              batches_to_send_overall[sender][receiver],
                              max_batches_can_receive_this_iter});
                batches_left_to_send_this_iter -= batches_;
                batches_to_send_overall[sender][receiver] -= batches_;
                to_send_this_iter[sender][receiver] += batches_;
                batches_to_recv_this_iter[receiver] += batches_;
            }
        }
    }

    // An alternative approach could be to do this:
    // For each receiver:
    // - List all its senders and how many batches they must send to this rank.
    // Take min(min_send_from_any_sender, N/n_senders) many batches from each
    // sender. If we still have capacity to receive, then remove the senders
    // that no longer have anything to send to this rank and repeat the process.
    // This is similar to the logic in 'determine_redistribution' and has better
    // "spread" in each shuffle iteration.

    return to_send_this_iter;
}

bool GroupbyOutputState::needs_another_shuffle(
    const std::vector<std::vector<size_t>>& batches_to_send) {
    for (const auto& x : batches_to_send) {
        for (const auto& y : x) {
            if (y > 0) {
                return true;
            }
        }
    }
    return false;
}

std::tuple<std::shared_ptr<table_info>, size_t>
GroupbyOutputState::redistribute_batches_helper(
    const std::vector<std::vector<size_t>>& to_send_this_iter,
    std::shared_ptr<TableBuildBuffer>& redistribution_tbb) {
    // While any rank has any data left to send to any other rank:
    // - Insert the relevant number of chunks into the TBB.
    // - Simultaneously, maintain a counter of rows to send to each
    // ranks based on the row counts of the chunks.
    time_pt start_data_prep = start_timer();
    const std::vector<size_t>& send_batch_counts =
        to_send_this_iter[this->myrank];
    size_t total_send_row_count = 0;
    std::vector<size_t> send_row_counts(this->n_pes, 0);
    for (int recv_rank = 0; recv_rank < this->n_pes; recv_rank++) {
        for (size_t i = 0; i < send_batch_counts[recv_rank]; i++) {
            auto [chunk, chunk_row_count] = this->buffer.PopChunk();
            send_row_counts[recv_rank] += chunk_row_count;
            total_send_row_count += chunk_row_count;
            redistribution_tbb->ReserveTable(chunk);
            redistribution_tbb->UnsafeAppendBatch(chunk);
        }
    }
    this->metrics.shuffle_data_prep_time += end_timer(start_data_prep);

    std::shared_ptr<table_info> shuffle_table = redistribution_tbb->data_table;
    // - Unify dictionaries for dict-encoded arrays.
    // TODO: Optimize to only do this once instead of per shuffle batch.
    time_pt start_dict_unif = start_timer();
    for (size_t i = 0; i < shuffle_table->ncols(); i++) {
        std::shared_ptr<array_info> arr = shuffle_table->columns[i];
        if (arr->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(arr,
                                              /*is_parallel*/ true);
        }
    }
    this->metrics.shuffle_dict_unification_time += end_timer(start_dict_unif);

    // - Setup shuffle_hashes to be the destination rank for each row.
    std::shared_ptr<uint32_t[]> shuffle_hashes =
        std::make_shared<uint32_t[]>(total_send_row_count);
    size_t offset = 0;
    for (int i = 0; i < this->n_pes; i++) {
        size_t start = offset;
        size_t end = offset + send_row_counts[i];
        for (size_t j = start; j < end; j++) {
            shuffle_hashes[j] = i;
        }
        offset = end;
    }

    time_pt start_shuffle = start_timer();
    mpi_comm_info comm_info_table(shuffle_table->columns, shuffle_hashes,
                                  /*is_parallel*/ true, /*filter*/ nullptr,
                                  /*keep_row_bitmask*/ nullptr,
                                  /*keep_filter_misses*/ false);

    // - Append the received input into the output buffer.
    // XXX Technically the shuffle_table already has the data laid out
    // by rank, so the copy it will make internally is wasteful and
    // something we should optimize for.
    std::shared_ptr<table_info> recv_data =
        shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                             comm_info_table, /*is_parallel*/ true);
    this->metrics.shuffle_time += end_timer(start_shuffle);

    // - Reset the shuffle TBB for the next iteration.
    redistribution_tbb->Reset();

    return std::make_tuple(recv_data, total_send_row_count);
}

void GroupbyOutputState::RedistributeWork() {
    assert(!this->work_stealing_done);
    time_pt start = start_timer();

    // 1. Gather data about remaining number of batches on every rank.
    uint64_t num_batches = this->buffer.chunks.size();
    std::vector<uint64_t> num_batches_ranks(this->n_pes);
    CHECK_MPI(
        MPI_Allgather(&num_batches, 1, MPI_UINT64_T, num_batches_ranks.data(),
                      1, MPI_UINT64_T, MPI_COMM_WORLD),
        "GroupbyOutputState::RedistributeWork: MPI error on MPI_Allgather:");

    // 2. Use a deterministic algorithm to assign the number of
    // batches/rows that each rank needs to send to every other rank.
    // TODO To reduce the size, this could only contain sub-vectors for the
    // "senders", which should be relatively few (guaranteed to be <50% at
    // the very least).
    time_pt start_det_redis = start_timer();
    std::vector<std::vector<size_t>> batches_to_send =
        GroupbyOutputState::determine_redistribution(num_batches_ranks);
    this->metrics.determine_redistribution_time += end_timer(start_det_redis);

    if (this->work_stealing_max_batched_send_recv_per_rank == -1) {
        // XXX TODO This could be modified to use the actual table size instead,
        // but it's good enough as a starting point.
        int64_t shuffle_threshold = get_shuffle_threshold();
        int64_t estimated_batch_size =
            get_row_bytes(this->buffer.dummy_output_chunk->schema()) *
            this->buffer.active_chunk_capacity;
        this->work_stealing_max_batched_send_recv_per_rank =
            std::ceil(static_cast<double>(shuffle_threshold) /
                      static_cast<double>(estimated_batch_size));
    }

    // 3. Use existing shuffle APIs to move the data around, but do this
    // a few batches at a time. During any one round of data transfer, each
    // sender will send at most N batches (overall) and any receiver must
    // receive at most N batches (overall) to maintain good memory pressure.

    // Temporary TBB to store intermediate data during the redistribution.
    std::shared_ptr<TableBuildBuffer> redistribution_tbb =
        std::make_shared<TableBuildBuffer>(
            this->buffer.dummy_output_chunk->schema(), this->dict_builders);

    // TODO Replace 'needs_another_shuffle' with a simpler check,
    // potentially in 'determine_batched_send_counts'.
    while (GroupbyOutputState::needs_another_shuffle(batches_to_send)) {
        this->metrics.num_shuffles++;
        time_pt start_det_batch_redis = start_timer();
        std::vector<std::vector<size_t>> to_send_this_iter =
            GroupbyOutputState::determine_batched_send_counts(
                batches_to_send,
                this->work_stealing_max_batched_send_recv_per_rank);
        this->metrics.determine_batched_send_counts_time +=
            end_timer(start_det_batch_redis);

        auto [recv_data, send_row_count] =
            redistribute_batches_helper(to_send_this_iter, redistribution_tbb);
        this->metrics.num_recv_rows += recv_data->nrows();
        this->metrics.num_sent_rows += send_row_count;
        if (recv_data->nrows() > 0) {
            // Unify with local dict-builders and append into the CTB.
            time_pt start_append = start_timer();
            this->buffer.UnifyDictionariesAndAppend(recv_data);
            this->metrics.shuffle_output_append_time += end_timer(start_append);
        }
    }
    // Release the shuffle buffer
    redistribution_tbb.reset();

    if (this->debug_work_stealing) {
        if (this->myrank == 0) {
            std::cerr << "[DEBUG][GroupbyOutputState] Performed "
                      << this->metrics.num_shuffles
                      << " shuffles to redistribute data." << std::endl;
        }
        if (this->metrics.num_recv_rows > 0) {
            std::cerr << "[DEBUG][GroupbyOutputState] Received "
                      << this->metrics.num_recv_rows
                      << " rows from other ranks during redistribution."
                      << std::endl;
        }
        if (this->metrics.num_sent_rows > 0) {
            std::cerr << "[DEBUG][GroupbyOutputState] Sent "
                      << this->metrics.num_sent_rows
                      << " rows to other ranks during redistribution."
                      << std::endl;
        }
    }
    this->metrics.redistribute_work_total_time += end_timer(start);
}

void GroupbyOutputState::ExportMetrics(std::vector<MetricBase>& metrics) {
    MetricBase::StatValue work_stealing_enabled =
        (this->enable_work_stealing ? 1 : 0);
    metrics.emplace_back(
        StatMetric("work_stealing_enabled", work_stealing_enabled, true));

    if (!this->enable_work_stealing) {
        return;
    }

    MetricBase::StatValue started_work_stealing_timer =
        (this->work_steal_timer_started ? 1 : 0);
    metrics.emplace_back(StatMetric("started_work_stealing_timer",
                                    started_work_stealing_timer, true));
    metrics.emplace_back(StatMetric("n_ranks_done_at_timer_start",
                                    this->metrics.n_ranks_done_at_timer_start,
                                    true));
    MetricBase::StatValue performed_work_redistribution =
        (this->performed_work_redistribution ? 1 : 0);
    metrics.emplace_back(StatMetric("performed_work_redistribution",
                                    performed_work_redistribution, true));

    if (!this->performed_work_redistribution) {
        return;
    }

    MetricBase::StatValue max_batches_send_recv_per_rank =
        this->work_stealing_max_batched_send_recv_per_rank;
    metrics.emplace_back(StatMetric("max_batches_send_recv_per_rank",
                                    max_batches_send_recv_per_rank, true));
    metrics.emplace_back(StatMetric(
        "n_ranks_done_before_work_redistribution",
        this->metrics.n_ranks_done_before_work_redistribution, true));
    metrics.emplace_back(
        StatMetric("num_shuffles", this->metrics.num_shuffles, true));
    metrics.emplace_back(
        TimerMetric("redistribute_work_total_time",
                    this->metrics.redistribute_work_total_time));
    metrics.emplace_back(
        TimerMetric("determine_redistribution_time",
                    this->metrics.determine_redistribution_time));
    metrics.emplace_back(
        TimerMetric("determine_batched_send_counts_time",
                    this->metrics.determine_batched_send_counts_time));
    metrics.emplace_back(
        StatMetric("num_recv_rows", this->metrics.num_recv_rows));
    metrics.emplace_back(
        StatMetric("num_sent_rows", this->metrics.num_sent_rows));
    metrics.emplace_back(TimerMetric("shuffle_data_prep_time",
                                     this->metrics.shuffle_data_prep_time));
    metrics.emplace_back(
        TimerMetric("shuffle_dict_unification_time",
                    this->metrics.shuffle_dict_unification_time));
    metrics.emplace_back(
        TimerMetric("shuffle_time", this->metrics.shuffle_time));
    metrics.emplace_back(TimerMetric("shuffle_output_append_time",
                                     this->metrics.shuffle_output_append_time));
}

#pragma endregion  // GroupbyOutputState
/* ------------------------------------------------------------------------ */

/* ----------------------------- GroupbyState ----------------------------- */
#pragma region  // GroupbyState

GroupbyState::GroupbyState(
    const std::unique_ptr<bodo::Schema>& in_schema_,
    std::vector<int32_t> ftypes, std::vector<int32_t> window_ftypes_,
    std::vector<int32_t> f_in_offsets_, std::vector<int32_t> f_in_cols_,
    uint64_t n_keys_, std::vector<bool> sort_asc_vec_,
    std::vector<bool> sort_na_pos_, std::vector<bool> cols_to_keep_bitmask_,
    std::shared_ptr<table_info> window_args, int64_t output_batch_size_,
    bool parallel_, int64_t sync_iter_, int64_t op_id_,
    int64_t op_pool_size_bytes_, bool allow_any_work_stealing,
    std::optional<std::vector<std::shared_ptr<DictionaryBuilder>>>
        key_dict_builders_,
    bool use_sql_rules, bool pandas_drop_na_,
    std::shared_ptr<table_info> udf_out_types,
    std::vector<stream_udf_t*> udf_cfuncs)
    :  // Create the operator buffer pool
      op_pool(std::make_unique<bodo::OperatorBufferPool>(
          op_id_,
          ((op_pool_size_bytes_ == -1)
               ? static_cast<uint64_t>(
                     bodo::BufferPool::Default()->get_memory_size_bytes() *
                     GROUPBY_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL)
               : op_pool_size_bytes_),
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
      pandas_drop_na(pandas_drop_na_),
      f_in_offsets(std::move(f_in_offsets_)),
      f_in_cols(std::move(f_in_cols_)),
      sort_asc(std::move(sort_asc_vec_)),
      sort_na(std::move(sort_na_pos_)),
      cols_to_keep_bitmask(std::move(cols_to_keep_bitmask_)),
      sync_iter(sync_iter_),
      op_id(op_id_) {
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

    for (int ftype : ftypes) {
        // NOTE: adding all functions that need accumulating inputs for now
        // but they may not be supported in streaming groupby yet
        // Should be kept in sync with
        // https://github.com/bodo-ai/Bodo/blob/56c77832aea4c4d5b33fd5cf631b201f4157f73a/BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/rel/core/AggregateBase.kt#L58
        if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::mode ||
            ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod ||
            ftype == Bodo_FTypes::cummin || ftype == Bodo_FTypes::cummax ||
            ftype == Bodo_FTypes::shift || ftype == Bodo_FTypes::transform ||
            ftype == Bodo_FTypes::ngroup || ftype == Bodo_FTypes::window ||
            ftype == Bodo_FTypes::listagg || ftype == Bodo_FTypes::nunique ||
            ftype == Bodo_FTypes::head || ftype == Bodo_FTypes::stream_udf ||
            ftype == Bodo_FTypes::min_row_number_filter) {
            this->accumulate_before_update = true;
        }
        if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::mode ||
            ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod ||
            ftype == Bodo_FTypes::cummin || ftype == Bodo_FTypes::cummax ||
            ftype == Bodo_FTypes::shift || ftype == Bodo_FTypes::transform ||
            ftype == Bodo_FTypes::ngroup || ftype == Bodo_FTypes::window ||
            ftype == Bodo_FTypes::listagg || ftype == Bodo_FTypes::nunique) {
            this->req_extended_group_info = true;
        }
        if (ftype == Bodo_FTypes::window) {
            this->agg_type = AggregationType::WINDOW;
        } else if (ftype == Bodo_FTypes::min_row_number_filter) {
            this->agg_type = AggregationType::MRNF;
        } else {
            this->agg_type = AggregationType::AGGREGATE;
        }
    }

    // Validate MRNF arguments in the MRNF case:
    if (this->agg_type == AggregationType::MRNF) {
        validate_mrnf_args(ftypes, this->f_in_cols, this->f_in_offsets,
                           this->sort_asc, this->sort_na, in_schema_->ncols(),
                           this->n_keys, "GroupbyState::GroupbyState");
    }

    if (allow_any_work_stealing) {
        if (this->agg_type == AggregationType::WINDOW) {
            char* disable_output_work_stealing_env_ =
                std::getenv("BODO_STREAM_WINDOW_DISABLE_OUTPUT_WORK_STEALING");
            if (disable_output_work_stealing_env_) {
                this->enable_output_work_stealing_window =
                    (std::strcmp(disable_output_work_stealing_env_, "0") == 0);
            }
        } else {
            char* enable_output_work_stealing_env_ =
                std::getenv("BODO_STREAM_GROUPBY_ENABLE_OUTPUT_WORK_STEALING");
            if (enable_output_work_stealing_env_) {
                this->enable_output_work_stealing_groupby =
                    (std::strcmp(enable_output_work_stealing_env_, "1") == 0);
            }
        }
    } else {
        // Work stealing isn't safe (e.g. grouping sets)
        this->enable_output_work_stealing_window = false;
        this->enable_output_work_stealing_groupby = false;
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
    // For Pandas aggregations, skip_na_data is set to true to match JIT:
    // https://github.com/bodo-ai/Bodo/blob/302e4f06f4f6ff3746f1f113fcae83ab2dc6e6dc/bodo/ir/aggregate.py#L522
    bool skip_na_data = true;
    if (!ftypes.empty() && ftypes[0] == Bodo_FTypes::window) {
        // Handle a collection of window functions.
        this->accumulate_before_update = true;

        // First, get the input column types for each function.
        size_t n_window_funcs = window_ftypes_.size();
        std::vector<std::shared_ptr<array_info>> local_input_cols_vec;
        std::vector<std::vector<std::unique_ptr<bodo::DataType>>>
            in_arr_types_vec(n_window_funcs);
        for (size_t i = 0; i < n_window_funcs; i++) {
            // Get the input columns, array types, and dtypes for the current
            // function
            std::vector<std::unique_ptr<bodo::DataType>>& in_arr_types =
                in_arr_types_vec.at(i);
            for (size_t logical_input_ind = (size_t)f_in_offsets[i];
                 logical_input_ind < (size_t)f_in_offsets[i + 1];
                 logical_input_ind++) {
                size_t physical_input_ind =
                    (size_t)f_in_cols[logical_input_ind];
                // set dummy input columns in ColSet since will be replaced by
                // input batches
                local_input_cols_vec.push_back(nullptr);
                in_arr_types.push_back(
                    (in_schema_->column_types[physical_input_ind])->copy());
            }
        }
        std::vector<int64_t> window_funcs;
        for (auto it : window_ftypes_) {
            window_funcs.push_back(static_cast<int64_t>(it));
        }
        std::vector<std::unique_ptr<bodo::DataType>> in_arr_types_copy;
        for (const auto& it : in_arr_types_vec) {
            for (const auto& t : it) {
                in_arr_types_copy.push_back(t->copy());
            }
        }
        std::shared_ptr<BasicColSet> col_set = makeColSet(
            local_input_cols_vec, index_col, Bodo_FTypes::window, false,
            skip_na_data, 0,
            // In the streaming multi-partition scenario, it's
            // safer to mark things as *not* parallel to avoid
            // any synchronization and hangs.
            window_funcs, 0, /*is_parallel*/ false, this->sort_asc,
            this->sort_na, window_args, f_in_cols.size(), nullptr, nullptr, 0,
            nullptr, use_sql_rules, std::move(in_arr_types_vec));

        // get update/combine type info to initialize build state
        std::unique_ptr<bodo::Schema> running_values_schema =
            col_set->getRunningValueColumnTypes(
                std::make_shared<bodo::Schema>(std::move(in_arr_types_copy)));
        size_t n_running_value_types = running_values_schema->ncols();
        curr_running_value_offset += n_running_value_types;
        this->f_running_value_offsets.push_back(curr_running_value_offset);

        this->col_sets.push_back(col_set);
    } else {
        // First, get the input column types for each function.
        std::vector<std::vector<std::shared_ptr<array_info>>>
            local_input_cols_vec(ftypes.size());
        std::vector<std::vector<std::unique_ptr<bodo::DataType>>>
            in_arr_types_vec(ftypes.size());
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
                size_t physical_input_ind =
                    (size_t)f_in_cols[logical_input_ind];
                // set dummy input columns in ColSet since will be replaced by
                // input batches
                local_input_cols.push_back(nullptr);
                in_arr_types.push_back(
                    (in_schema_->column_types.at(physical_input_ind))->copy());
            }
        }

        // Track number of UDFs i.e. groupby.agg(), used for creating the UDF
        // Colsets.
        int udf_idx = 0;

        // Handle non-window functions.
        // Perform a check on the running value and output types.
        // If any of them are of type string, set accumulate_before_update to
        // true.
        for (size_t i = 0; i < ftypes.size(); i++) {
            std::vector<std::shared_ptr<array_info>>& local_input_cols =
                local_input_cols_vec.at(i);
            std::vector<std::unique_ptr<bodo::DataType>>& in_arr_types =
                in_arr_types_vec.at(i);
            std::vector<std::unique_ptr<bodo::DataType>> in_arr_types_copy;
            for (const auto& t : in_arr_types) {
                in_arr_types_copy.push_back(t->copy());
            }

            std::shared_ptr<table_info> udf_out_type = nullptr;
            if (ftypes[i] == Bodo_FTypes::stream_udf) {
                std::vector<std::shared_ptr<array_info>> out_col = {
                    udf_out_types->columns[udf_idx]};
                udf_out_type = std::make_shared<table_info>(out_col);
            }

            std::unique_ptr<bodo::Schema> running_values_schema =
                this->getRunningValueColumnTypes(
                    local_input_cols, std::move(in_arr_types_copy), ftypes[i],
                    0, window_args, udf_out_type);

            auto seperate_out_cols = this->getSeparateOutputColumns(
                local_input_cols, ftypes[i], 0, udf_out_type);

            if (ftypes[i] == Bodo_FTypes::stream_udf) {
                udf_idx++;
            }

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

        udf_idx = 0;

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

            stream_udf_t* udf_cfunc = nullptr;
            std::shared_ptr<table_info> udf_out_type = nullptr;
            if (ftypes[i] == Bodo_FTypes::stream_udf) {
                udf_cfunc = udf_cfuncs[udf_idx];
                std::vector<std::shared_ptr<array_info>> out_col = {
                    udf_out_types->columns[udf_idx]};
                udf_out_type = std::make_shared<table_info>(out_col);
            }

            std::shared_ptr<BasicColSet> col_set =
                makeColSet(local_input_cols, index_col, ftypes[i], do_combine,
                           skip_na_data, 0,
                           // In the streaming multi-partition scenario, it's
                           // safer to mark things as *not* parallel to avoid
                           // any synchronization and hangs.
                           {}, 0, /*is_parallel*/ false, this->sort_asc,
                           this->sort_na, window_args, 0, nullptr, udf_out_type,
                           0, nullptr, use_sql_rules, {}, udf_cfunc);

            if (ftypes[i] == Bodo_FTypes::stream_udf) {
                udf_idx++;
            }
            // get update/combine type info to initialize build state
            std::unique_ptr<bodo::Schema> running_values_schema =
                col_set->getRunningValueColumnTypes(
                    std::make_shared<bodo::Schema>(
                        std::move(in_arr_types_copy)));
            size_t n_running_value_types = running_values_schema->ncols();

            if (!this->accumulate_before_update) {
                build_table_schema->append_schema(
                    std::move(running_values_schema));

                // Determine what separate output columns are necessary.
                // This is only required in the AGG case.
                auto separate_out_col_type =
                    col_set->getSeparateOutputColumnType();
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
    }

    // Allocate the histogram if we are taking the accumulate path.
    this->compute_histogram = this->accumulate_before_update;
    // Always compute the max number of partitions to protect against the worst
    // case.
    this->num_histogram_bits =
        this->compute_histogram ? this->max_partition_depth : 0;
    this->histogram_buckets.resize((1 << this->num_histogram_bits), 0);

    // See if all ColSet functions are nunique, which enables optimization of
    // dropping duplicate shuffle table rows before shuffle
    this->nunique_only = (ftypes.size() > 0);
    for (int ftype : ftypes) {
        if (ftype != Bodo_FTypes::nunique) {
            this->nunique_only = false;
        }
    }

    // build buffer types are same as input if just accumulating batches
    if (this->accumulate_before_update) {
        build_table_schema = std::make_shared<bodo::Schema>(*in_schema_);
    }

    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;
    if (key_dict_builders_.has_value()) {
        // Enable sharing dictionary builders across group by used by multiple
        // grouping sets.
        key_dict_builders = key_dict_builders_.value();
        ASSERT(key_dict_builders.size() == this->n_keys);
    } else {
        // Create dictionary builders for key columns (if not provided by
        // caller
        key_dict_builders.resize(this->n_keys);

        // Create dictionary builders for key columns:
        for (uint64_t i = 0; i < this->n_keys; i++) {
            key_dict_builders[i] = create_dict_builder_for_array(
                build_table_schema->column_types[i]->copy(), true);
        }
    }

    std::vector<std::shared_ptr<DictionaryBuilder>>
        build_table_non_key_dict_builders;
    // Create dictionary builders for non-key columns in build table:
    for (size_t i = this->n_keys; i < build_table_schema->column_types.size();
         i++) {
        build_table_non_key_dict_builders.emplace_back(
            create_dict_builder_for_array(
                build_table_schema->column_types[i]->copy(), false));
    }

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(), key_dict_builders.begin(),
        key_dict_builders.end());

    this->build_table_dict_builders.insert(
        this->build_table_dict_builders.end(),
        build_table_non_key_dict_builders.begin(),
        build_table_non_key_dict_builders.end());

    this->shuffle_state = std::make_unique<GroupbyIncrementalShuffleState>(
        build_table_schema, this->build_table_dict_builders, this->col_sets,
        this->sort_na.size(), this->n_keys, this->build_iter, this->sync_iter,
        op_id_, this->nunique_only, this->agg_type,
        this->f_running_value_offsets, this->accumulate_before_update);

    this->partitions.emplace_back(std::make_shared<GroupbyPartition>(
        0, 0, build_table_schema, std::move(separate_out_cols_schema),
        this->n_keys, this->build_table_dict_builders, this->col_sets,
        this->f_in_offsets, this->f_in_cols, this->f_running_value_offsets,
        /*is_active*/ true, this->accumulate_before_update,
        this->req_extended_group_info, this->metrics, this->op_pool.get(),
        this->op_mm, this->op_scratch_pool.get(), this->op_scratch_mm));
    this->partition_state.emplace_back(std::make_pair<size_t, uint32_t>(0, 0));

    // Reserve space upfront. The output-batch-size is typically the same
    // as the input batch size.
    this->append_row_to_build_table.reserve(output_batch_size_);

    if (this->op_id != -1) {
        std::vector<MetricBase> metrics;
        metrics.reserve(4);
        MetricBase::BlobValue agg_type =
            get_aggregation_type_string(this->agg_type);
        metrics.emplace_back(BlobMetric("aggregation_type", agg_type, true));
        MetricBase::StatValue is_nunique_only = this->nunique_only ? 1 : 0;
        metrics.emplace_back(
            StatMetric("is_nunique_only", is_nunique_only, true));
        MetricBase::BlobValue acc_or_agg =
            this->accumulate_before_update ? "ACC" : "AGG";
        metrics.emplace_back(BlobMetric("acc_or_agg", acc_or_agg, true));
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }
    this->curr_stage_id++;

    CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &this->shuffle_comm),
              "GroupbyState::GroupbyState: MPI error on MPI_Comm_dup:");
}

std::unique_ptr<bodo::Schema> GroupbyState::getRunningValueColumnTypes(
    std::vector<std::shared_ptr<array_info>> local_input_cols,
    std::vector<std::unique_ptr<bodo::DataType>>&& in_dtypes, int ftype,
    int window_ftype, std::shared_ptr<table_info> window_args,
    std::shared_ptr<table_info> udf_output_type) {
    std::shared_ptr<BasicColSet> col_set =
        makeColSet(local_input_cols,  // in_cols
                   nullptr,           // index_col
                   ftype,             // ftype
                   true,              // do_combine
                   true,              // skip_na_data
                   0,                 // period
                   {window_ftype},    // transform_funcs
                   0,                 // n_udf
                   false,             // parallel
                   this->sort_asc,    // window_ascending
                   this->sort_na,     // window_na_position
                   window_args,       // window_args
                   0,                 // n_input_cols
                   nullptr,           // udf_n_redvars
                   udf_output_type,   // udf_table
                   0,                 // udf_table_idx
                   nullptr,           // nunique_table
                   true               // use_sql_rules
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
    std::vector<std::shared_ptr<array_info>> local_input_cols, int ftype,
    int window_ftype, std::shared_ptr<table_info> udf_output_type) {
    std::shared_ptr<BasicColSet> col_set =
        makeColSet(local_input_cols,  // in_cols
                   nullptr,           // index_col
                   ftype,             // ftype
                   true,              // do_combine
                   true,              // skip_na_data
                   0,                 // period
                   {window_ftype},    // transform_funcs
                   0,                 // n_udf
                   false,             // parallel
                   this->sort_asc,    // window_ascending
                   this->sort_na,     // window_na_position
                   nullptr,           // window_args
                   0,                 // n_input_cols
                   nullptr,           // udf_n_redvars
                   udf_output_type,   // udf_table
                   0,                 // udf_table_idx
                   nullptr,           // nunique_table
                   true               // use_sql_rules
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
    time_pt start_split = start_timer();
    if (this->partitions[idx]->is_active_partition()) {
        new_partitions = this->partitions[idx]->SplitPartition<true>();
    } else {
        new_partitions = this->partitions[idx]->SplitPartition<false>();
    }
    this->metrics.repartitioning_time += end_timer(start_split);
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

std::string GroupbyState::GetPartitionStateString() const {
    std::string partition_state = "[";
    for (const auto& i : this->partition_state) {
        size_t num_top_bits = i.first;
        uint32_t top_bitmask = i.second;
        partition_state +=
            fmt::format("({0}, {1:#b}),", num_top_bits, top_bitmask);
    }
    partition_state += "]";
    return partition_state;
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
                "GroupbyState::UpdateGroupsAndCombine: Couldn't find "
                "any matching partition for row!");
        }
    }
    this->metrics.input_partition_check_time += end_timer(start_part_check);
    this->metrics.input_partition_check_nrows += in_table->nrows();
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
            this->metrics.n_repartitions_in_append++;
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

    time_pt start_part_check = start_timer();
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
    this->metrics.input_partition_check_time += end_timer(start_part_check);
    this->metrics.input_partition_check_nrows += in_table->nrows();

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
            this->metrics.n_repartitions_in_append++;
            this->SplitPartition(0);
        }
    }
}

void GroupbyState::UpdateShuffleGroupsAndCombine(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& batch_hashes_groupby,
    const std::vector<bool>& append_rows) {
    // Just insert into the buffer. We check about the hash table later.
    time_pt start = start_timer();
    this->shuffle_state->pre_reduction_table_buffer->ReserveTable(in_table);
    this->shuffle_state->pre_reduction_table_buffer->UnsafeAppendBatch(
        in_table, append_rows);
    // Insert the hashes.
    // TODO: Preallocate the hashes to the max size?
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (append_rows[i_row]) {
            this->shuffle_state->pre_reduction_hashes.push_back(
                batch_hashes_groupby[i_row]);
        }
    }
    this->shuffle_state->metrics.shuffle_agg_buffer_append_time +=
        end_timer(start);
}

void GroupbyState::AppendBuildBatchHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes) {
    // Update the histogram buckets, regardless of how many partitions
    // there are.
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        this->histogram_buckets[hash_to_bucket(partitioning_hashes[i_row],
                                               this->num_histogram_bits)] += 1;
    }

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
                "GroupbyState::AppendBuildBatch: Couldn't find "
                "any matching partition for row!");
        }
    }
    this->metrics.input_partition_check_time += end_timer(start_part_check);
    this->metrics.input_partition_check_nrows += in_table->nrows();

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
            this->metrics.n_repartitions_in_append++;
            this->SplitPartition(0);
        }
    }
}

void GroupbyState::AppendBuildBatchHelper(
    const std::shared_ptr<table_info>& in_table,
    const std::shared_ptr<uint32_t[]>& partitioning_hashes,
    const std::vector<bool>& append_rows) {
    // Update the histogram buckets, regardless of how many partitions
    // there are.
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        this->histogram_buckets[hash_to_bucket(partitioning_hashes[i_row],
                                               this->num_histogram_bits)] +=
            append_rows[i_row] ? 1 : 0;
    }

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
    this->metrics.input_partition_check_time += end_timer(start_part_check);
    this->metrics.input_partition_check_nrows += in_table->nrows();

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
            this->metrics.n_repartitions_in_append++;
            this->SplitPartition(0);
        }
    }
}

void GroupbyState::InitOutputBufferMrnf(
    const std::shared_ptr<table_info>& dummy_build_table) {
    ASSERT(this->agg_type == AggregationType::MRNF);

    // Skip if already initialized.
    if (this->output_state != nullptr) {
        return;
    }

    size_t n_cols = dummy_build_table->columns.size();

    // List of column types in the output.
    std::vector<std::unique_ptr<bodo::DataType>> output_column_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> output_dict_builders;
    for (size_t i = 0; i < n_cols; i++) {
        if (this->cols_to_keep_bitmask[i]) {
            output_column_types.push_back(
                dummy_build_table->columns[i]->data_type());
            // We can re-use existing dictionaries.
            output_dict_builders.push_back(this->build_table_dict_builders[i]);
        }
    }

    // Build the output schema from the output column types.
    std::shared_ptr<bodo::Schema> output_schema =
        std::make_shared<bodo::Schema>(std::move(output_column_types));
    // Initialize the output state using this schema.
    this->output_state = std::make_shared<GroupbyOutputState>(
        output_schema, output_dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        /*enable_work_stealing_*/ this->parallel &&
            this->enable_output_work_stealing_groupby);
}

void GroupbyState::InitOutputBufferWindow(
    const std::shared_ptr<table_info>& dummy_build_table) {
    ASSERT(this->agg_type == AggregationType::WINDOW);
    ASSERT(this->col_sets.size() == 1);
    const std::shared_ptr<BasicColSet>& col_set = this->col_sets[0];
    const std::vector<int64_t> window_funcs = col_set->getFtypes();
    // Skip if already initialized.
    if (this->output_state != nullptr) {
        return;
    }

    size_t n_cols = dummy_build_table->columns.size();

    // List of column types in the output.
    std::vector<std::unique_ptr<bodo::DataType>> output_column_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> output_dict_builders;
    for (size_t i = 0; i < n_cols; i++) {
        if (this->cols_to_keep_bitmask[i]) {
            output_column_types.push_back(
                dummy_build_table->columns[i]->data_type());
            // We can re-use existing dictionaries.
            output_dict_builders.push_back(this->build_table_dict_builders[i]);
        }
    }

    // Append the window function column types.
    const std::vector<std::unique_ptr<bodo::DataType>> window_output_types =
        col_set->getOutputTypes();

    for (size_t i = 0; i < window_output_types.size(); i++) {
        output_column_types.push_back(window_output_types[i]->copy());
        // if lead lag then we need to set the dict builder. It can just be a
        // copy of the input
        if (window_funcs[i] == Bodo_FTypes::lead ||
            window_funcs[i] == Bodo_FTypes::lag) {
            size_t in_col_idx = f_in_cols[f_in_offsets[i]];
            output_dict_builders.push_back(
                this->build_table_dict_builders[in_col_idx]);
        } else {
            output_dict_builders.push_back(nullptr);
        }
    }

    // Build the output schema from the output column types.
    std::shared_ptr<bodo::Schema> output_schema =
        std::make_shared<bodo::Schema>(std::move(output_column_types));
    // Initialize the output buffer using this schema.
    this->output_state = std::make_shared<GroupbyOutputState>(
        output_schema, output_dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        /*enable_work_stealing_*/ this->parallel &&
            this->enable_output_work_stealing_window);
}

void GroupbyState::InitOutputBuffer(
    const std::shared_ptr<table_info>& dummy_table) {
    ASSERT(this->agg_type == AggregationType::AGGREGATE);
    auto schema = dummy_table->schema();

    std::vector<std::shared_ptr<DictionaryBuilder>> output_dict_builders(
        dummy_table->columns.size(), nullptr);

    // Keys are always the first columns in groupby output and match input
    // array types and dictionaries for DICT arrays. See
    // https://github.com/bodo-ai/Bodo/blob/f94ab6d2c78e3a536a8383ddf71956cc238fccc8/bodo/libs/_groupby_common.cpp#L604
    for (size_t i = 0; i < this->n_keys; i++) {
        output_dict_builders[i] = this->build_table_dict_builders[i];
    }
    // Non-key columns may have different type and/or dictionaries from
    // input arrays
    for (size_t i = this->n_keys; i < dummy_table->ncols(); i++) {
        output_dict_builders[i] =
            create_dict_builder_for_array(dummy_table->columns[i], false);
    }
    this->output_state = std::make_shared<GroupbyOutputState>(
        std::move(schema), output_dict_builders,
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        /*enable_work_stealing*/ this->parallel &&
            this->enable_output_work_stealing_groupby);
}

std::shared_ptr<table_info> GroupbyState::UnifyBuildTableDictionaryArrays(
    const std::shared_ptr<table_info>& in_table, bool only_keys) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (this->build_table_dict_builders[i] == nullptr ||
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
    assert(this->agg_type == AggregationType::AGGREGATE);
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(out_table->ncols());
    for (size_t i = 0; i < out_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = out_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        // Output key columns have the same dictionary as inputs and don't
        // need unification
        if (this->output_state->dict_builders[i] == nullptr ||
            (i < this->n_keys)) {
            out_arr = in_arr;
        } else {
            out_arr =
                this->output_state->dict_builders[i]->UnifyDictionaryArray(
                    in_arr);
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
        if (this->build_table_dict_builders[i] == nullptr) {
            (*dict_hashes)[i] = nullptr;
        } else {
            (*dict_hashes)[i] =
                this->build_table_dict_builders[i]->GetDictionaryHashes();
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
    this->append_row_to_build_table.resize(0);
    this->append_row_to_build_table.shrink_to_fit();
}

bool GroupbyState::GetGlobalIsLast(bool local_is_last) {
    if (this->global_is_last) {
        return true;
    }

    if (this->parallel && local_is_last) {
        if (!this->is_last_barrier_started) {
            CHECK_MPI(
                MPI_Ibarrier(this->shuffle_comm, &this->is_last_request),
                "GroupbyState::GetGlobalIsLast: MPI error on MPI_Ibarrier:");
            this->is_last_barrier_started = true;
            return false;
        } else {
            int flag = 0;
            CHECK_MPI(
                MPI_Test(&this->is_last_request, &flag, MPI_STATUS_IGNORE),
                "GroupbyState::GetGlobalIsLast: MPI error on MPI_Test:");
            if (flag) {
                this->global_is_last = true;
            }
            return flag;
        }
    } else {
        // If we have a replicated input we don't need to be
        // synchronized because there is no shuffle.
        return local_is_last;
    }
}

void GroupbyState::ReportAndResetBuildMetrics(bool is_final) {
    std::vector<MetricBase> metrics;
    metrics.reserve(128);

    metrics.emplace_back(StatMetric("n_repartitions_in_append",
                                    this->metrics.n_repartitions_in_append));
    metrics.emplace_back(StatMetric("n_repartitions_in_finalize",
                                    this->metrics.n_repartitions_in_finalize));
    metrics.emplace_back(TimerMetric("repartitioning_time_total",
                                     this->metrics.repartitioning_time));
    metrics.emplace_back(
        TimerMetric("repartitioning_part_hashing_time",
                    this->metrics.repartitioning_part_hashing_time));
    metrics.emplace_back(
        StatMetric("repartitioning_part_hashing_nrows",
                   this->metrics.repartitioning_part_hashing_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_active_part1_append_time",
                    this->metrics.repartitioning_active_part1_append_time));
    metrics.emplace_back(
        StatMetric("repartitioning_active_part1_append_nrows",
                   this->metrics.repartitioning_active_part1_append_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_active_part2_append_time",
                    this->metrics.repartitioning_active_part2_append_time));
    metrics.emplace_back(
        StatMetric("repartitioning_active_part2_append_nrows",
                   this->metrics.repartitioning_active_part2_append_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_inactive_pop_chunk_time",
                    this->metrics.repartitioning_inactive_pop_chunk_time));
    metrics.emplace_back(
        StatMetric("repartitioning_inactive_pop_chunk_n_chunks",
                   this->metrics.repartitioning_inactive_pop_chunk_n_chunks));
    metrics.emplace_back(
        TimerMetric("repartitioning_inactive_append_time",
                    this->metrics.repartitioning_inactive_append_time));

    if (!this->accumulate_before_update) {
        // In the pre-agg case, the number of input rows is the same as the
        // total number of local input rows.
        metrics.emplace_back(TimerMetric("pre_agg_total_time",
                                         this->metrics.pre_agg_total_time));
        metrics.emplace_back(
            TimerMetric("pre_agg_colset_update_time",
                        this->metrics.pre_agg_metrics.colset_update_time));
        metrics.emplace_back(TimerMetric(
            "pre_agg_hashing_time",
            this->metrics.pre_agg_metrics.grouping_metrics.hashing_time));
        metrics.emplace_back(TimerMetric(
            "pre_agg_grouping_time",
            this->metrics.pre_agg_metrics.grouping_metrics.grouping_time));
        metrics.emplace_back(TimerMetric(
            "pre_agg_hll_time",
            this->metrics.pre_agg_metrics.grouping_metrics.hll_time));
        metrics.emplace_back(StatMetric("pre_agg_output_nrows",
                                        this->metrics.pre_agg_output_nrows));
        metrics.emplace_back(
            TimerMetric("input_groupby_hashing_time",
                        this->metrics.input_groupby_hashing_time));
        metrics.emplace_back(TimerMetric(
            "rebuild_ht_hashing_time", this->metrics.rebuild_ht_hashing_time));
        metrics.emplace_back(
            StatMetric("rebuild_ht_hashing_nrows",
                       this->metrics.rebuild_ht_hashing_nrows));
        metrics.emplace_back(TimerMetric("rebuild_ht_insert_time",
                                         this->metrics.rebuild_ht_insert_time));
        metrics.emplace_back(StatMetric("rebuild_ht_insert_nrows",
                                        this->metrics.rebuild_ht_insert_nrows));
        metrics.emplace_back(TimerMetric("update_logical_ht_time",
                                         this->metrics.update_logical_ht_time));
        metrics.emplace_back(StatMetric("update_logical_ht_nrows",
                                        this->metrics.update_logical_ht_nrows));
        metrics.emplace_back(TimerMetric("combine_input_time",
                                         this->metrics.combine_input_time));
        metrics.emplace_back(StatMetric("combine_input_nrows",
                                        this->metrics.combine_input_nrows));
    } else {
        metrics.emplace_back(TimerMetric("appends_active_time",
                                         this->metrics.appends_active_time));
        metrics.emplace_back(StatMetric("appends_active_nrows",
                                        this->metrics.appends_active_nrows));
    }

    metrics.emplace_back(TimerMetric("input_part_hashing_time",
                                     this->metrics.input_part_hashing_time));
    metrics.emplace_back(
        StatMetric("input_hashing_nrows", this->metrics.input_hashing_nrows));
    metrics.emplace_back(TimerMetric("input_partition_check_time",
                                     this->metrics.input_partition_check_time));
    metrics.emplace_back(StatMetric("input_partition_check_nrows",
                                    this->metrics.input_partition_check_nrows));
    metrics.emplace_back(TimerMetric("appends_inactive_time",
                                     this->metrics.appends_inactive_time));
    metrics.emplace_back(StatMetric("appends_inactive_nrows",
                                    this->metrics.appends_inactive_nrows));

    if (is_final) {
        // Final number of partitions
        metrics.emplace_back(
            StatMetric("n_partitions", this->metrics.n_partitions));
        MetricBase::BlobValue final_partitioning_state =
            this->GetPartitionStateString();
        metrics.emplace_back(
            BlobMetric("final_partitioning_state", final_partitioning_state));
        metrics.emplace_back(
            TimerMetric("finalize_time_total", this->metrics.finalize_time));
        if (!this->accumulate_before_update) {
            metrics.emplace_back(TimerMetric(
                "finalize_activate_groupby_hashing_time",
                this->metrics.finalize_activate_groupby_hashing_time));
        } else {
            if (this->agg_type == AggregationType::MRNF) {
                metrics.emplace_back(
                    TimerMetric("finalize_compute_mrnf_time",
                                this->metrics.finalize_compute_mrnf_time));
            } else if (this->agg_type == AggregationType::WINDOW) {
                metrics.emplace_back(
                    TimerMetric("finalize_window_compute_time",
                                this->metrics.finalize_window_compute_time));
            } else {
                metrics.emplace_back(
                    TimerMetric("finalize_get_update_table_time",
                                this->metrics.finalize_get_update_table_time));
            }
            metrics.emplace_back(TimerMetric(
                "finalize_colset_update_time",
                this->metrics.finalize_update_metrics.colset_update_time));
            metrics.emplace_back(StatMetric(
                "finalize_colset_update_nrows",
                this->metrics.finalize_update_metrics.colset_update_nrows));
            metrics.emplace_back(TimerMetric(
                "finalize_hashing_time", this->metrics.finalize_update_metrics
                                             .grouping_metrics.hashing_time));
            metrics.emplace_back(StatMetric(
                "finalize_hashing_nrows", this->metrics.finalize_update_metrics
                                              .grouping_metrics.hashing_nrows));
            metrics.emplace_back(TimerMetric(
                "finalize_grouping_time", this->metrics.finalize_update_metrics
                                              .grouping_metrics.grouping_time));
            metrics.emplace_back(
                StatMetric("finalize_grouping_nrows",
                           this->metrics.finalize_update_metrics
                               .grouping_metrics.grouping_nrows));
            metrics.emplace_back(TimerMetric(
                "finalize_hll_time", this->metrics.finalize_update_metrics
                                         .grouping_metrics.hll_time));
            metrics.emplace_back(StatMetric(
                "finalize_hll_nrows", this->metrics.finalize_update_metrics
                                          .grouping_metrics.hll_nrows));
        }

        metrics.emplace_back(TimerMetric("finalize_eval_time",
                                         this->metrics.finalize_eval_time));
        metrics.emplace_back(StatMetric("finalize_eval_nrows",
                                        this->metrics.finalize_eval_nrows));
        metrics.emplace_back(
            TimerMetric("finalize_activate_partition_time",
                        this->metrics.finalize_activate_partition_time));
        metrics.emplace_back(
            TimerMetric("finalize_activate_pin_chunk_time",
                        this->metrics.finalize_activate_pin_chunk_time));
        metrics.emplace_back(
            StatMetric("finalize_activate_pin_chunk_n_chunks",
                       this->metrics.finalize_activate_pin_chunk_n_chunks));
    }

    // Shuffle metrics
    this->shuffle_state->ExportMetrics(metrics);

    // Dict Builders Stats
    if (this->agg_type == AggregationType::MRNF ||
        this->agg_type == AggregationType::WINDOW) {
        // All dict builders are shared between the build buffer and output
        // buffer.
        // NOTE: When window functions can output string arrays this will need
        // to be updated.
        assert(is_final);
        DictBuilderMetrics dict_builder_metrics;
        MetricBase::StatValue n_dict_builders = 0;
        for (const auto& dict_builder : this->build_table_dict_builders) {
            if (dict_builder != nullptr) {
                dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
                n_dict_builders++;
            }
        }
        metrics.emplace_back(
            StatMetric("n_dict_builders", n_dict_builders, true));
        dict_builder_metrics.add_to_metrics(metrics, "dict_builders_");
    } else {
        DictBuilderMetrics key_dict_builder_metrics;
        DictBuilderMetrics non_key_build_dict_builder_metrics;
        MetricBase::StatValue n_key_dict_builders = 0;
        MetricBase::StatValue n_non_key_build_dict_builders = 0;
        for (size_t i = 0; i < this->build_table_dict_builders.size(); i++) {
            const auto& dict_builder = this->build_table_dict_builders[i];
            if (dict_builder != nullptr) {
                if (i < this->n_keys) {
                    key_dict_builder_metrics.add_metrics(
                        dict_builder->GetMetrics());
                    n_key_dict_builders++;
                } else {
                    non_key_build_dict_builder_metrics.add_metrics(
                        dict_builder->GetMetrics());
                    n_non_key_build_dict_builders++;
                }
            }
        }

        // Create a copy before we modify it.
        DictBuilderMetrics key_dict_builder_metrics_copy =
            key_dict_builder_metrics;
        // Subtract metrics from previous stage to get the delta for this stage.
        key_dict_builder_metrics.subtract_metrics(
            this->key_dict_builder_metrics_prev_stage_snapshot);
        // Set the snapshot to the new values.
        this->key_dict_builder_metrics_prev_stage_snapshot =
            key_dict_builder_metrics_copy;
        key_dict_builder_metrics.add_to_metrics(metrics, "key_dict_builders_");
        non_key_build_dict_builder_metrics.add_to_metrics(
            metrics, "non_key_build_dict_builders_");

        if (is_final) {
            metrics.emplace_back(
                StatMetric("n_key_dict_builders", n_key_dict_builders, true));
            metrics.emplace_back(StatMetric("n_non_key_build_dict_builders",
                                            n_non_key_build_dict_builders,
                                            true));
            DictBuilderMetrics non_key_output_dict_builder_metrics;
            MetricBase::StatValue n_non_key_output_dict_builders = 0;
            for (size_t i = this->n_keys;
                 i < this->output_state->dict_builders.size(); i++) {
                const auto& dict_builder = this->output_state->dict_builders[i];
                if (dict_builder != nullptr) {
                    non_key_output_dict_builder_metrics.add_metrics(
                        dict_builder->GetMetrics());
                    n_non_key_output_dict_builders++;
                }
            }
            metrics.emplace_back(StatMetric("n_non_key_output_dict_builders",
                                            n_non_key_output_dict_builders,
                                            true));
            non_key_output_dict_builder_metrics.add_to_metrics(
                metrics, "non_key_output_dict_builders_");
        }
    }

    if (is_final) {
        // Output buffer append time and total size.
        metrics.emplace_back(TimerMetric(
            "output_append_time", this->output_state->buffer.append_time));
        MetricBase::StatValue output_total_size =
            this->output_state->buffer.total_size;
        metrics.emplace_back(
            StatMetric("output_total_nrows", output_total_size));
        MetricBase::StatValue output_n_chunks =
            this->output_state->buffer.chunks.size();
        metrics.emplace_back(StatMetric("output_n_chunks", output_n_chunks));
    }

    if (this->op_id != -1) {
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }

    // Reset metrics
    this->metrics = GroupbyMetrics();
    this->shuffle_state->ResetMetrics();
}

void GroupbyState::ReportOutputMetrics() {
    std::vector<MetricBase> metrics;
    metrics.reserve(32);

    this->output_state->ExportMetrics(metrics);

    if (this->op_id != -1) {
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }
}

bool GroupbyState::MaxPartitionExceedsThreshold(size_t num_bits,
                                                uint32_t bitmask,
                                                double threshold) {
    int64_t max_bucket_size = 0;
    int64_t total_bucket_size = 0;
    uint64_t shift_amount = this->max_partition_depth - num_bits;
    // All buckets in a partition will be clustered together.
    // Find the bounds to investigate.
    uint64_t lower_bound_partition_zero_inclusive = bitmask << shift_amount;
    uint64_t upper_bound_partition_zero_exclusive = (num_bits + 1)
                                                    << shift_amount;
    for (uint64_t i = lower_bound_partition_zero_inclusive;
         i < upper_bound_partition_zero_exclusive; i++) {
        int64_t bucket_size = this->histogram_buckets[i];
        total_bucket_size += bucket_size;
        max_bucket_size = std::max(max_bucket_size, bucket_size);
    }

    if (total_bucket_size > 0) {
        double ratio = static_cast<double>(max_bucket_size) /
                       static_cast<double>(total_bucket_size);
        return ratio > threshold;
    } else {
        return false;
    }
}

void GroupbyState::FinalizeBuild() {
    std::cout
        << "[DEBUG] GroupbyState::FinalizeBuild: Finalizing build phase with "
           " "
        << this->partitions.size() << " partitions." << std::endl;
    time_pt start_finalize = start_timer();
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
                bool at_max_partition_depth =
                    this->partitions[i_part]->get_num_top_bits() ==
                    this->max_partition_depth;
                // Check if based on having a histogram we can predict the
                // impact of partitioning. If we know that repartitioning will
                // never reduce a partition beyond 90% of the current row count
                // then just try compute the whole thing.
                bool bucket_disabled_partitioning = false;
                if (!at_max_partition_depth && this->compute_histogram) {
                    if (this->debug_partitioning) {
                        std::cerr
                            << "[DEBUG] GroupbyState::FinalizeBuild: Checking "
                               "histogram buckets to disable partitioning"
                            << std::endl;
                    }
                    // We require num_histogram_bits to be max_partition_depth,
                    // so we have an exact mapping of buckets -> partition.
                    ASSERT(this->num_histogram_bits ==
                           this->max_partition_depth);
                    bucket_disabled_partitioning =
                        this->MaxPartitionExceedsThreshold(
                            this->partitions[i_part]->get_num_top_bits(),
                            this->partitions[i_part]->get_top_bitmask(), 0.9);
                }
                if (orig_partitioning_enabled &&
                    (at_max_partition_depth || bucket_disabled_partitioning)) {
                    this->DisablePartitioning();
                    if (this->debug_partitioning) {
                        if (at_max_partition_depth) {
                            // Log a warning
                            std::cerr
                                << "[DEBUG] WARNING: Disabling partitioning "
                                   "and "
                                   "threshold enforcement temporarily to "
                                   "finalize "
                                   "partition "
                                << i_part
                                << " which is at max allowed partition depth ("
                                << this->max_partition_depth
                                << "). This may invoke the OOM killer."
                                << std::endl;
                        } else {
                            std::cerr
                                << "[DEBUG] WARNING: Disabling partitioning "
                                   "and "
                                   "threshold enforcement temporarily to "
                                   "finalize "
                                   "partition "
                                << i_part
                                << " which is determined based on the "
                                   "histogram "
                                << "to retain at least 90% of its data after "
                                << "repartitioning. This may invoke the OOM "
                                   "killer."
                                << std::endl;
                        }
                    }
                }

                // Finalize the partition and get output from it.
                // TODO: Write output directly into the GroupybyState's
                // output buffer instead of returning the output.
                if (this->agg_type == AggregationType::MRNF) {
                    // Initialize output buffer if this is the first
                    // partition.
                    if (i_part == 0) {
                        this->InitOutputBufferMrnf(
                            this->partitions[i_part]
                                ->build_table_buffer->data_table);
                    }
                    this->partitions[i_part]->FinalizeMrnf(
                        this->cols_to_keep_bitmask, this->sort_na.size(),
                        this->output_state->buffer);
                } else if (this->agg_type == AggregationType::WINDOW) {
                    // Initialize output buffer if this is the first
                    // partition.
                    if (i_part == 0) {
                        this->InitOutputBufferWindow(
                            this->partitions[i_part]
                                ->build_table_buffer->data_table);
                    }
                    this->partitions[i_part]->FinalizeWindow(
                        this->cols_to_keep_bitmask, this->sort_na.size(),
                        this->output_state->buffer,
                        this->output_state->dict_builders, this->f_in_offsets,
                        this->f_in_cols);
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
                this->metrics.n_repartitions_in_finalize++;
                this->SplitPartition(i_part);
            }

            if (!exception_caught) {
                // Since we have generated the output, we don't need the
                // partition anymore, so we can release that memory.
                this->partitions[i_part].reset();
                if (this->agg_type == AggregationType::AGGREGATE) {
                    if (i_part == 0) {
                        this->InitOutputBuffer(output_table);
                    }
                    // XXX TODO UnifyOutputDictionaryArrays needs a version that
                    // can take the shared_ptr without reference and free
                    // individual columns early.
                    output_table =
                        this->UnifyOutputDictionaryArrays(output_table);
                    this->output_state->buffer.AppendBatch(output_table);
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
    this->metrics.n_partitions = this->partitions.size();
    this->output_state->Finalize();
    // Release the ColSets, etc.
    this->ClearBuildState();
    this->build_input_finalized = true;
    this->metrics.finalize_time += end_timer(start_finalize);
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
 * @brief Filter out NA keys in groupby to match pandas drop_na=True.
 * Reference:
 * https://github.com/bodo-ai/Bodo/blob/bed3fb5908472ebc80a24ccc514e241fedbada37/bodo/libs/streaming/_join.cpp#L2537
 *
 * @param in_table input batch to groupby.
 * @param n_keys number of key columns in input
 * @return std::shared_ptr<table_info> input table with NAs filtered out
 */
std::shared_ptr<table_info> filter_na_keys(std::shared_ptr<table_info> in_table,
                                           uint64_t n_keys) {
    bodo::vector<bool> not_na(in_table->nrows(), true);
    bool contains_na_keys = false;
    for (uint64_t i = 0; i < n_keys; i++) {
        // Determine which columns can contain NA/contain NA
        const std::shared_ptr<array_info>& col = in_table->columns[i];
        if (col->can_contain_na()) {
            bodo::vector<bool> col_not_na = col->get_notna_vector();
            // Do an elementwise logical and to update not_na
            for (size_t i = 0; i < in_table->nrows(); i++) {
                not_na[i] = not_na[i] && col_not_na[i];
                contains_na_keys = contains_na_keys || !not_na[i];
            }
        }
    }

    if (!contains_na_keys) {
        // No NA values, skip the copy.
        return in_table;
    } else {
        // Retrieve table takes a list of columns. Convert the boolean array.
        bodo::vector<int64_t> idx_list;

        for (size_t i = 0; i < in_table->nrows(); i++) {
            if (not_na[i]) {
                idx_list.emplace_back(i);
            }
        }
        return RetrieveTable(std::move(in_table), std::move(idx_list));
    }
}

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
                                     bool local_is_last,
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
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Unify dictionaries keys to allow consistent hashing and fast key
    // comparison using indices
    in_table = groupby_state->UnifyBuildTableDictionaryArrays(in_table, true);
    // We don't pass the op-pool here since this is operation on a small
    // batch and we consider this "scratch" usage essentially.
    time_pt start_pre_agg = start_timer();
    in_table = get_update_table</*is_acc_case*/ false>(
        in_table, groupby_state->n_keys, groupby_state->col_sets,
        groupby_state->f_in_offsets, groupby_state->f_in_cols,
        groupby_state->req_extended_group_info,
        groupby_state->metrics.pre_agg_metrics);
    groupby_state->metrics.pre_agg_total_time += end_timer(start_pre_agg);
    groupby_state->metrics.pre_agg_output_nrows += in_table->nrows();

    if (groupby_state->build_iter == 0) {
        groupby_state->shuffle_state->Initialize(
            in_table, groupby_state->parallel, groupby_state->shuffle_comm);
    }

    // Dictionary hashes for the key columns which will be used for
    // the partitioning hashes:
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = groupby_state->GetDictionaryHashesForKeys();

    time_pt start_hash = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_groupby =
        hash_keys_table(in_table, groupby_state->n_keys,
                        SEED_HASH_GROUPBY_SHUFFLE, false, false);
    groupby_state->metrics.input_groupby_hashing_time += end_timer(start_hash);
    start_hash = start_timer();
    std::shared_ptr<uint32_t[]> batch_hashes_partition =
        hash_keys_table(in_table, groupby_state->n_keys, SEED_HASH_PARTITION,
                        groupby_state->parallel, false, dict_hashes);
    groupby_state->metrics.input_part_hashing_time += end_timer(start_hash);
    groupby_state->metrics.input_hashing_nrows += in_table->nrows();

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

    append_row_to_build_table.flip();
    // Do the same for the shuffle groups:
    groupby_state->UpdateShuffleGroupsAndCombine(in_table, batch_hashes_groupby,
                                                 append_row_to_build_table);

    // Reset the bitmask for the next iteration:
    append_row_to_build_table.resize(0);

    if (groupby_state->parallel) {
        std::optional<std::shared_ptr<table_info>> new_data_ =
            groupby_state->shuffle_state->ShuffleIfRequired(local_is_last);
        if (new_data_.has_value()) {
            std::shared_ptr<table_info> new_data = new_data_.value();
            dict_hashes = groupby_state->GetDictionaryHashesForKeys();
            start_hash = start_timer();
            batch_hashes_groupby = hash_keys_table(
                new_data, groupby_state->n_keys, SEED_HASH_GROUPBY_SHUFFLE,
                groupby_state->parallel, /*global_dict_needed*/ false);
            groupby_state->metrics.input_groupby_hashing_time +=
                end_timer(start_hash);
            start_hash = start_timer();
            batch_hashes_partition =
                hash_keys_table(new_data, groupby_state->n_keys,
                                SEED_HASH_PARTITION, groupby_state->parallel,
                                /*global_dict_needed*/ false, dict_hashes);
            groupby_state->metrics.input_part_hashing_time +=
                end_timer(start_hash);
            groupby_state->metrics.input_hashing_nrows += new_data->nrows();

            groupby_state->UpdateGroupsAndCombine(
                new_data, batch_hashes_partition, batch_hashes_groupby);
        }
    }

    // Make is_last global
    bool is_last = groupby_state->GetGlobalIsLast(
        local_is_last && groupby_state->shuffle_state->SendRecvEmpty());

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
                                     bool local_is_last,
                                     const bool is_final_pipeline) {
    assert(is_final_pipeline);
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (groupby_state->build_iter == 0) {
        groupby_state->shuffle_state->Initialize(
            in_table, groupby_state->parallel, groupby_state->shuffle_comm);
    }

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
            groupby_state->shuffle_state->ShuffleIfRequired(local_is_last);
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
            // there's just one partition and we aren't computing the
            // histogram. We could move the hash computation
            // inside AppendBuildBatch and only do it if there are multiple
            // partitions.
            groupby_state->AppendBuildBatch(new_data, batch_hashes_partition);

            batch_hashes_partition.reset();
        }
    }

    // Make is_last global
    bool is_last = groupby_state->GetGlobalIsLast(
        local_is_last && groupby_state->shuffle_state->SendRecvEmpty());

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
    auto [batch, is_last] =
        groupby_state->output_state->PopBatch(produce_output);
    groupby_state->output_state->iter++;
    return std::make_tuple(batch, is_last);
}

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
                                 bool* request_input) {
    // Request input rows from preceding operators by default
    *request_input = true;
    if (groupby_state->build_input_finalized) {
        // If the build input has been finalized, we should not be
        // consuming any more build input.
        return true;
    }
    groupby_state->metrics.build_input_row_count += input_table->nrows();

    // Filter NA keys in Pandas case.
    if (groupby_state->pandas_drop_na) {
        input_table =
            filter_na_keys(std::move(input_table), groupby_state->n_keys);
    }

    if (groupby_state->accumulate_before_update) {
        is_last = groupby_acc_build_consume_batch(
            groupby_state, std::move(input_table), is_last, is_final_pipeline);
    } else {
        is_last = groupby_agg_build_consume_batch(
            groupby_state, std::move(input_table), is_last, is_final_pipeline);
    }
    *request_input = !groupby_state->shuffle_state->BuffersFull();

    if (is_last) {
        // Report and reset metrics
        groupby_state->ReportAndResetBuildMetrics(is_final_pipeline);
        groupby_state->curr_stage_id++;
        if (is_final_pipeline) {
            // groupby_state->out_dict_builders retains references
            // to the DictionaryBuilders required for the output
            // buffer, so clearing these is safe.
            // We cannot free these during FinalizeBuild since we need these
            // for the metric calculation. Once we've collected the metrics,
            // these are safe to release.
            assert(groupby_state->build_input_finalized);
            groupby_state->build_table_dict_builders.clear();
        }
    }

    return is_last;
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
 * @param[out] request_input whether to request input rows from preceding
 * operators.
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool groupby_build_consume_batch_py_entry(GroupbyState* groupby_state,
                                          table_info* in_table, bool is_last,
                                          const bool is_final_pipeline,
                                          bool* request_input) {
    try {
        std::unique_ptr<table_info> input_table(in_table);
        return groupby_build_consume_batch(groupby_state,
                                           std::move(input_table), is_last,
                                           is_final_pipeline, request_input);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief Python wrapper to consume build table batch across all groupby
 * states in the grouping sets object.
 * @param groupby_state groupby state pointer
 * @param in_table build table batch
 * @param is_last is last batch (in this pipeline) locally
 * @param[out] request_input whether to request input rows from preceding
 * operators.
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool grouping_sets_build_consume_batch_py_entry(
    GroupingSetsState* grouping_sets_state, table_info* in_table, bool is_last,
    bool* request_input) {
    try {
        std::unique_ptr<table_info> input_table(in_table);
        std::pair<bool, bool> result = grouping_sets_state->ConsumeBuildBatch(
            std::move(input_table), is_last);
        *request_input = result.second;
        return result.first;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

std::pair<bool, bool> GroupingSetsState::ConsumeBuildBatch(
    std::shared_ptr<table_info> input_table, bool is_last) {
    bool global_is_last = true;
    bool global_request_input = true;
    for (size_t i = 0; i < this->groupby_states.size(); i++) {
        bool local_request_input;
        std::shared_ptr<table_info> pruned_table =
            ProjectTable(input_table, this->input_columns_remaps[i]);
        bool local_is_last = groupby_build_consume_batch(
            this->groupby_states[i].get(), pruned_table, is_last, true,
            &local_request_input);
        global_is_last = global_is_last && local_is_last;
        global_request_input = global_request_input && local_request_input;
    }
    return std::make_pair(global_is_last, global_request_input);
}

/**
 * @brief Resets non-blocking is_last sync state after each union pipeline when
 * using groupby
 *
 * @param groupby_state streaming groupby state
 */
void end_union_consume_pipeline_py_entry(GroupbyState* groupby_state) {
    groupby_state->is_last_request = MPI_REQUEST_NULL;
    groupby_state->is_last_barrier_started = false;
    groupby_state->global_is_last = false;
}

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
    GroupbyState* groupby_state, bool* out_is_last, bool produce_output) {
    bool is_last;
    std::shared_ptr<table_info> out;
    std::tie(out, is_last) =
        groupby_produce_output_batch(groupby_state, produce_output);
    *out_is_last = is_last;
    groupby_state->metrics.output_row_count += out->nrows();
    if (is_last) {
        if (groupby_state->op_id != -1) {
            QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                QueryProfileCollector::MakeOperatorStageID(
                    groupby_state->op_id, groupby_state->curr_stage_id),
                groupby_state->metrics.output_row_count);
        }
        groupby_state->ReportOutputMetrics();
    }
    return out;
}

/**
 * @brief Python wrapper to produce output table
 * batch.
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
        std::shared_ptr<table_info> out = groupby_produce_output_batch_wrapper(
            groupby_state, out_is_last, produce_output);
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

#define GROUPING_SETS_OUTPUT_STAGE 2

/**
 * @brief Python wrapper to produce an output table
 * batch
 *
 * @param grouping_sets_state grouping sets state pointer
 * @param[out] out_is_last is last batch
 * @param produce_output whether to produce output
 * @return table_info* output table batch
 */
table_info* grouping_sets_produce_output_batch_py_entry(
    GroupingSetsState* grouping_sets_state, bool* out_is_last,
    bool produce_output) {
    try {
        bool is_last;
        std::shared_ptr<table_info> out;
        std::tie(out, is_last) =
            grouping_sets_state->ProduceOutputBatch(produce_output);
        *out_is_last = is_last;
        if (is_last) {
            if (grouping_sets_state->op_id != -1) {
                QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                    QueryProfileCollector::MakeOperatorStageID(
                        grouping_sets_state->op_id, GROUPING_SETS_OUTPUT_STAGE),
                    grouping_sets_state->metrics.output_row_count);
            }
        }
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

#undef GROUPING_SETS_OUTPUT_STAGE

std::pair<std::shared_ptr<table_info>, bool>
GroupingSetsState::ProduceOutputBatch(bool produce_output) {
    bool state_is_last;
    std::shared_ptr<table_info> out_table =
        groupby_produce_output_batch_wrapper(
            this->groupby_states[this->current_output_idx].get(),
            &state_is_last, produce_output);
    // Remap the kept and generate nulls.
    const std::vector<int64_t>& remap_vector =
        this->output_columns_remaps[this->current_output_idx];
    const std::vector<int64_t>& nulls_vector =
        this->missing_output_columns_remaps[this->current_output_idx];
    const std::vector<int64_t>& grouping_indices = this->grouping_output_idxs;
    const std::vector<int64_t>& grouping_values =
        this->grouping_values[this->current_output_idx];
    std::vector<std::shared_ptr<array_info>> final_columns(
        remap_vector.size() + nulls_vector.size() + grouping_indices.size());
    size_t table_size = out_table->nrows();
    for (size_t i = 0; i < remap_vector.size(); i++) {
        final_columns[remap_vector[i]] = out_table->columns[i];
    }
    for (int64_t idx : nulls_vector) {
        bodo_array_type::arr_type_enum arr_typ =
            this->keys_schema->column_types[idx]->array_type;
        const Bodo_CTypes::CTypeEnum c_typ =
            this->keys_schema->column_types[idx]->c_type;
        std::shared_ptr<array_info> null_arr =
            alloc_all_null_array_top_level(table_size, arr_typ, c_typ);
        // Reuse the dictionary builder to avoid unnecessary
        // transposing/unification.
        if (this->key_dict_builders[idx] != nullptr) {
            if (arr_typ == bodo_array_type::DICT) {
                null_arr->child_arrays[0] =
                    this->key_dict_builders[idx]->dict_buff->data_array;
            } else {
                throw std::runtime_error(
                    "Unsupported dictionary builder for null array");
            }
        }
        final_columns[idx] = null_arr;
    }
    // Append the grouping function.
    for (size_t i = 0; i < grouping_indices.size(); i++) {
        std::shared_ptr<array_info> grouping_arr =
            alloc_numpy(table_size, Bodo_CTypes::INT64);
        int64_t* grouping_data =
            (int64_t*)grouping_arr->data1<bodo_array_type::NUMPY>();
        int64_t grouping_value = grouping_values[i];
        for (size_t j = 0; j < table_size; j++) {
            grouping_data[j] = grouping_value;
        }
        final_columns[grouping_indices[i]] = grouping_arr;
    }
    std::shared_ptr<table_info> final_table =
        std::make_shared<table_info>(final_columns, table_size);
    this->metrics.output_row_count += final_table->nrows();
    if (state_is_last) {
        if (this->current_output_idx == 0) {
            // Keep the last group by state around to simplify generating the
            // empty output.
            this->finalized_output = true;
        } else {
            this->current_output_idx--;
        }
    }
    return std::make_pair(final_table, this->finalized_output);
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
 * @param ftypes function types (Bodo_FTypes ints)
 * @param window_ftypes window function types (Bodo_FTypes ints)
 * @param n_keys number of groupby keys
 * @param sort_asc Boolean bitmask specifying sort-direction for MRNF
 *  order-by columns. It should be 'mrnf_n_sort_keys' elements long.
 * @param sort_na Boolean bitmask specifying whether nulls should be
 *  considered 'last' for the order-by columns of MRNF. It should be
 * 'mrnf_n_sort_keys' elements long.
 * @param n_sort_keys Number of MRNF order-by columns. If this is
 *  >0, we will use MRNF or windowspecific code.
 * @param cols_to_keep Bitmask of columns to keep in the output. Should be the
 * same length as the number of columns in the input.
 * @param output_batch_size Batch size for reading output.
 * @param op_pool_size_bytes Size of the operator buffer pool for this join
 * operator. If it's set to -1, we will get the budget from the operator
 * comptroller.
 * @return GroupbyState* groupby state to return to Python
 */
GroupbyState* groupby_state_init_py_entry(
    int64_t operator_id, int8_t* build_arr_c_types,
    int8_t* build_arr_array_types, int n_build_arrs, int32_t* ftypes,
    int32_t* window_ftypes, int32_t* f_in_offsets, int32_t* f_in_cols,
    int n_funcs, uint64_t n_keys, bool* sort_asc, bool* sort_na,
    uint64_t n_sort_keys, bool* cols_to_keep, table_info* window_args_,
    int64_t output_batch_size, bool parallel, int64_t sync_iter,
    int64_t op_pool_size_bytes) {
    // If the memory budget has not been explicitly set, then ask the
    // OperatorComptroller for the budget.
    if (op_pool_size_bytes == -1) {
        op_pool_size_bytes =
            OperatorComptroller::Default()->GetOperatorBudget(operator_id);
    }

    // The true number of funcs is 1 if doing a window operation, since each
    // window ftype should correspond to a single ::window in the ftypes vector.
    size_t true_n_funcs =
        (n_funcs > 0 && ftypes[0] == Bodo_FTypes::window) ? 1 : n_funcs;

    std::unique_ptr<bodo::Schema> in_schema = bodo::Schema::Deserialize(
        std::vector<int8_t>(build_arr_array_types,
                            build_arr_array_types + n_build_arrs),
        std::vector<int8_t>(build_arr_c_types,
                            build_arr_c_types + n_build_arrs));
    size_t n_inputs = in_schema->column_types.size();

    // Create vectors for the MRNF arguments from the raw pointer arrays.
    std::vector<bool> sort_asc_vec(n_sort_keys, false);
    std::vector<bool> sort_na_vec(n_sort_keys, false);
    std::vector<bool> cols_to_keep_vec(n_inputs, true);
    if (true_n_funcs == 1 && (ftypes[0] == Bodo_FTypes::min_row_number_filter ||
                              ftypes[0] == Bodo_FTypes::window)) {
        for (size_t i = 0; i < (size_t)n_inputs; i++) {
            cols_to_keep_vec[i] = cols_to_keep[i];
        }
        for (size_t i = 0; i < n_sort_keys; i++) {
            sort_na_vec[i] = sort_na[i];
        }
        for (size_t i = 0; i < n_sort_keys; i++) {
            sort_asc_vec[i] = sort_asc[i];
        }
    }
    std::shared_ptr<table_info> window_args(window_args_);

    return new GroupbyState(
        in_schema, std::vector<int32_t>(ftypes, ftypes + true_n_funcs),
        std::vector<int32_t>(window_ftypes, window_ftypes + n_funcs),
        std::vector<int32_t>(f_in_offsets, f_in_offsets + n_funcs + 1),
        std::vector<int32_t>(f_in_cols, f_in_cols + f_in_offsets[n_funcs]),
        n_keys, sort_asc_vec, sort_na_vec, cols_to_keep_vec, window_args,
        output_batch_size, parallel, sync_iter, operator_id,
        op_pool_size_bytes);
}

/**
 * @brief Get the grouping value for a particular group by state
 * based on which keys are which keys are "live" in the group by.
 * GROUPING places a 1 bit when a key is missing in the order given
 * by its argument order. So GROUPING(KEY1, KEY0, KEY2) would be
 * 0b110 if we group by KEY2 in a particular grouping set, 0b110 if
 * we group by KEY0, and 0b011 if we group by KEY1.
 *
 * @param live_keys vector of booleans indicating which keys are live.
 * @param arg_data array of grouping arguments
 * @param start_idx The starting index of the grouping arguments.
 * @param end_idx The ending index of the grouping arguments (exclusive).
 * @return int64_t The value of grouping.
 */
int64_t get_grouping_value(const std::vector<bool>& live_keys,
                           int32_t* arg_data, int32_t start_idx,
                           int32_t end_idx) {
    int64_t output = 0;
    for (int i = start_idx; i < end_idx; i++) {
        output <<= 1;
        int32_t key = arg_data[i];
        if (!live_keys[key]) {
            output |= 1;
        }
    }
    return output;
}

/**
 * @brief Python wrapper to create and initialize a new streaming grouping sets
 * state object. This object will also create and manage several groupby states.
 * @param operator_id The operator ID of the grouping sets operator.
 * @param sub_operator_ids The operator IDs for each groupby state being
 * generated.
 * @param build_arr_c_types The array types of the build table columns
 * (Bodo_CTypes ints).
 * @param build_arr_array_types The array types of the build table columns
 * (bodo_array_type ints).
 * @param n_build_arrs The length of build_arr_c_types and
 * build_arr_array_types.
 * @param grouping_sets_data The grouping sets data.
 * @param grouping_sets_offsets The grouping sets offsets. Groupby i uses the
 * values between grouping_sets_offsets[i] and grouping_sets_offsets[i+1]
 * (exclusive).
 * @param n_grouping_sets The number of grouping sets.
 * @param ftypes The function types (Bodo_FTypes ints).
 * @param f_in_offsets The offsets into the f_in_cols array for each function.
 * These are shared across all group by states.
 * @param f_in_cols The column indices for each function. These need to be
 * remapped for each group by state.
 * @param n_funcs The number of functions.
 * @param n_keys The number of total group by keys across all grouping sets.
 * @param output_batch_size The batch size for reading output.
 * @param parallel Whether to run in parallel.
 * @param sync_iter The synchronization iteration.
 *
 */
GroupingSetsState* grouping_sets_state_init_py_entry(
    int64_t operator_id, int64_t* sub_operator_ids, int8_t* build_arr_c_types,
    int8_t* build_arr_array_types, int32_t n_build_arrs,
    int32_t* grouping_sets_data, int32_t* grouping_sets_offsets,
    int n_grouping_sets, int32_t* ftypes, int32_t* f_in_offsets,
    int32_t* f_in_cols, int n_funcs, uint64_t n_keys, int64_t output_batch_size,
    bool parallel, int64_t sync_iter) {
    try {
        // Generate the total schema for being able to fetch subsets.
        std::unique_ptr<bodo::Schema> total_schema = bodo::Schema::Deserialize(
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs));
        // Determine the number of build columns because decimal arrays
        // can require additional entries.
        size_t n_build_columns = total_schema->ncols();
        // Generate the general keys schema for remapping the output from
        // grouping sets.
        std::unique_ptr<bodo::Schema> keys_schema =
            total_schema->Project(n_keys);

        // Generate the key dictionary builders for all group by states.
        std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders(
            n_keys);
        // Create dictionary builders for key columns:
        for (uint64_t i = 0; i < n_keys; i++) {
            key_dict_builders[i] = create_dict_builder_for_array(
                keys_schema->column_types[i]->copy(), true);
        }
        // Remove any ftypes for grouping.
        std::vector<int32_t> ftypes_vector;
        std::vector<int32_t> f_in_cols_vector;
        std::vector<int32_t> f_in_offsets_vector;
        f_in_offsets_vector.push_back(f_in_offsets[0]);
        std::vector<int64_t> grouping_output_idxs;
        int64_t num_skipped_columns = 0;
        for (int i = 0; i < n_funcs; i++) {
            if (ftypes[i] != Bodo_FTypes::grouping) {
                int64_t f_types_start = f_in_offsets[i];
                int64_t f_types_end = f_in_offsets[i + 1];
                for (int64_t j = f_types_start; j < f_types_end; j++) {
                    f_in_cols_vector.push_back(f_in_cols[j]);
                }
                ftypes_vector.push_back(ftypes[i]);
                f_in_offsets_vector.push_back(f_in_offsets[i + 1] -
                                              num_skipped_columns);
            } else {
                grouping_output_idxs.push_back(i +
                                               static_cast<int64_t>(n_keys));
                num_skipped_columns += f_in_offsets[i + 1] - f_in_offsets[i];
            }
        }

        // Generate a grouping state for each grouping set and perform any
        // necessary remapping.
        std::vector<std::unique_ptr<GroupbyState>> groupby_states;
        // Generate the column index remapping for each build input.
        std::vector<std::vector<int64_t>> input_columns_remaps;
        std::vector<std::vector<int64_t>> output_columns_remaps;
        // Generate the vector of null keys for remapping the outputs.
        std::vector<std::vector<int64_t>> skipped_columns_remaps;
        // Collect grouping values
        std::vector<std::vector<int64_t>> total_grouping_values;
        for (int i = 0; i < n_grouping_sets; i++) {
            // Compute the bitmask for which keys need to be dropped.
            std::vector<bool> kept_columns(n_build_columns, false);
            for (int64_t j = grouping_sets_offsets[i];
                 j < grouping_sets_offsets[i + 1]; j++) {
                kept_columns[grouping_sets_data[j]] = true;
            }
            for (uint64_t j = n_keys;
                 j < static_cast<uint64_t>(n_build_columns); j++) {
                kept_columns[j] = true;
            }

            // Remap the build information to skip any extra keys.
            std::vector<int64_t> kept_input_column_idxs;
            std::vector<std::unique_ptr<bodo::DataType>> kept_column_types;
            for (size_t j = 0; j < n_build_columns; j++) {
                if (kept_columns[j]) {
                    kept_column_types.push_back(
                        total_schema->column_types[j]->copy());
                    kept_input_column_idxs.push_back(j);
                }
            }

            int32_t num_grouping_keys =
                grouping_sets_offsets[i + 1] - grouping_sets_offsets[i];
            int32_t num_skipped_keys =
                static_cast<int32_t>(n_keys) - num_grouping_keys;

            // Remap f_in_cols for this grouping set. We need some special
            // handling for key columns (the key columns for this grouping set
            // as well as any key columns in any of the grouping sets) that are
            // also data column. Their mapping needs to be handled carefully to
            // ensure there's no unnecessary duplication.
            std::vector<int32_t> remapped_f_in_cols;
            // This is used for caching the remapping computation. This also
            // ensures that we insert additional values into 'kept_column_types'
            // and 'kept_input_column_idxs' exactly once. Note that we do *not*
            // update 'kept_columns' here. This is because we later use
            // 'kept_columns' for checking if a key column is part of this
            // grouping set's key columns.
            std::unordered_map<int32_t, int32_t> remapping_cache;

            for (const int32_t orig_idx : f_in_cols_vector) {
                int32_t new_idx = -1;
                if (remapping_cache.contains(orig_idx)) {
                    new_idx = remapping_cache[orig_idx];
                } else {
                    if (static_cast<uint64_t>(orig_idx) < n_keys) {
                        // This is a key column of *some* grouping set
                        if (kept_columns[orig_idx]) {
                            // Part of this grouping set's keys already, so we
                            // just need to find its index
                            auto it = std::ranges::find(kept_input_column_idxs,

                                                        orig_idx);
                            ASSERT(it != kept_input_column_idxs.end());
                            new_idx = std::distance(
                                kept_input_column_idxs.begin(), it);
                        } else {
                            // This is a key column, but not for this grouping
                            // set, so we need to add this as a data column.
                            new_idx = kept_input_column_idxs.size();
                            kept_input_column_idxs.push_back(orig_idx);
                            kept_column_types.push_back(
                                total_schema->column_types[orig_idx]->copy());
                        }
                    } else {
                        // Regular data column
                        ASSERT(orig_idx - num_skipped_keys >= 0);
                        new_idx = (orig_idx - num_skipped_keys);
                    }
                    // Add to remapping cache
                    remapping_cache[orig_idx] = new_idx;
                }
                ASSERT(new_idx != -1);
                remapped_f_in_cols.push_back(new_idx);
            }

            input_columns_remaps.push_back(kept_input_column_idxs);
            // Remap the output columns
            std::vector<int64_t> kept_output_column_idxs;
            std::vector<int64_t> skipped_columns;
            // Track the keys that are used so all group by states can share
            // dictionary builders.
            std::vector<std::shared_ptr<DictionaryBuilder>>
                local_key_dict_builders;
            for (size_t j = 0; j < n_keys; j++) {
                if (kept_columns[j]) {
                    // Note that some key column, that is not a key column for
                    // this particular grouping set, may still be a data column
                    // for this grouping set. However, 'kept_column' doesn't
                    // know about that, which means that we're technically
                    // losing some potential to reuse dictionary builders.
                    // TODO This is something we could optimize. This would also
                    // require allowing GroupbyState to optionally take in
                    // DictionaryBuilders for its data columns.
                    kept_output_column_idxs.push_back(j);
                    local_key_dict_builders.push_back(key_dict_builders[j]);
                } else {
                    skipped_columns.push_back(j);
                }
            }
            std::vector<int64_t> grouping_values;
            for (int32_t j = 0; j < n_funcs; j++) {
                if (ftypes[j] != Bodo_FTypes::grouping) {
                    kept_output_column_idxs.push_back(j + n_keys);
                } else {
                    // Note that some key column that is not a key column for
                    // this particular grouping set, may still be a data column
                    // for this grouping set. However, it shouldn't be part of
                    // the group value. 'kept_columns' doesn't know about these,
                    // so it produces the correct output.
                    grouping_values.push_back(get_grouping_value(
                        kept_columns, f_in_cols, f_in_offsets[j],
                        f_in_offsets[j + 1]));
                }
            }
            total_grouping_values.push_back(grouping_values);
            output_columns_remaps.push_back(kept_output_column_idxs);
            skipped_columns_remaps.push_back(skipped_columns);
            int64_t sub_operator_id = sub_operator_ids[i];
            int64_t op_pool_size_bytes =
                OperatorComptroller::Default()->GetOperatorBudget(
                    sub_operator_id);
            // Create the groupby state.
            groupby_states.push_back(std::make_unique<GroupbyState>(
                std::make_unique<bodo::Schema>(std::move(kept_column_types)),
                ftypes_vector, /*window_ftypes_*/ std::vector<int32_t>{},
                f_in_offsets_vector, remapped_f_in_cols, num_grouping_keys,
                /*sort_asc_vec_*/ std::vector<bool>{},
                /*sort_na_pos_*/ std::vector<bool>{},
                /*cols_to_keep_bitmask_*/ std::vector<bool>{}, nullptr,
                output_batch_size, parallel, sync_iter, sub_operator_id,
                op_pool_size_bytes, /*allow_any_work_stealing*/ false,
                local_key_dict_builders));
        }
        return new GroupingSetsState(
            std::move(keys_schema), std::move(groupby_states),
            input_columns_remaps, output_columns_remaps, skipped_columns_remaps,
            grouping_output_idxs, total_grouping_values, key_dict_builders,
            operator_id);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief delete groupby state (called from Python after output loop is
 * finished)
 *
 * @param groupby_state groupby state pointer to delete
 */
void delete_groupby_state(GroupbyState* groupby_state) { delete groupby_state; }

/**
 * @brief delete a grouping sets state object (called from Python after the
 * output loop is finished).
 * @param grouping_sets_state The grouping set state pointer to delete
 */
void delete_grouping_sets_state(GroupingSetsState* grouping_sets_state) {
    delete grouping_sets_state;
}

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
    MOD_DEF(m, "stream_groupby_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, groupby_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, grouping_sets_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, groupby_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, grouping_sets_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, groupby_produce_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, grouping_sets_produce_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_groupby_state);
    SetAttrStringFromVoidPtr(m, delete_grouping_sets_state);
    SetAttrStringFromVoidPtr(m, end_union_consume_pipeline_py_entry);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_pinned);
    SetAttrStringFromVoidPtr(m, get_op_pool_bytes_allocated);
    SetAttrStringFromVoidPtr(m, get_num_partitions);
    SetAttrStringFromVoidPtr(m, get_partition_num_top_bits_by_idx);
    SetAttrStringFromVoidPtr(m, get_partition_top_bitmask_by_idx);
    return m;
}

#undef MAX_SHUFFLE_HASHTABLE_SIZE
#undef MAX_SHUFFLE_TABLE_SIZE
