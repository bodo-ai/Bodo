#include "_groupby_col_set.h"

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "../window/_window_compute.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"
#include "_groupby_mode.h"
#include "_groupby_udf.h"
#include "_groupby_update.h"

/**
 * This file creates the "col set" infrastructure. A col set is the
 * explicit implementation of the groupby compute steps for various
 * groups of functions. For example, simple aggregations (e.g.
 * sum, min, max) will share a col set and then transformation
 * operations will share a col set. The operations defined in
 * the groupby col set will largely mirror the general infrastructure.
 *
 */

// ############################## BasicColSet ##############################

BasicColSet::BasicColSet(std::shared_ptr<array_info> in_col, int ftype,
                         bool combine_step, bool use_sql_rules)
    : in_col(std::move(in_col)),
      ftype(ftype),
      combine_step(combine_step),
      use_sql_rules(use_sql_rules) {}

BasicColSet::~BasicColSet() = default;

void BasicColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
    int64_t num_categories = in_col->num_categories;
    std::tie(arr_type, dtype) =
        get_groupby_output_dtype(ftype, arr_type, dtype);
    out_cols.push_back(alloc_array_top_level(
        num_groups, 1, 1, arr_type, dtype, -1, 0, num_categories, false, false,
        false, pool, std::move(mm)));
}

void BasicColSet::update(const std::vector<grouping_info>& grp_infos,
                         bodo::IBufferPool* const pool,
                         std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
    do_apply_to_column(in_col, update_cols[0], aux_cols, grp_infos[0], ftype,
                       pool, std::move(mm));
}

typename std::vector<std::shared_ptr<array_info>>::iterator
BasicColSet::update_after_shuffle(
    typename std::vector<std::shared_ptr<array_info>>::iterator& it) {
    for (auto& update_col : update_cols) {
        update_col = *(it++);
    }
    return it;
}

void BasicColSet::combine(const grouping_info& grp_info,
                          int64_t init_start_row) {
    int combine_ftype = get_combine_func(ftype);
    std::vector<std::shared_ptr<array_info>> aux_cols(combine_cols.begin() + 1,
                                                      combine_cols.end());
    for (auto col : combine_cols) {
        aggfunc_output_initialize(col, combine_ftype, use_sql_rules,
                                  init_start_row);
    }

    do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                       combine_ftype);
}

void BasicColSet::eval(const grouping_info& grp_info,
                       bodo::IBufferPool* const pool,
                       std::shared_ptr<::arrow::MemoryManager> mm) {}

const std::vector<std::shared_ptr<array_info>> BasicColSet::getOutputColumns() {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (combine_step) {
        mycols = &combine_cols;
    } else {
        mycols = &update_cols;
    }

    std::shared_ptr<array_info> out_col = mycols->at(0);
    return {out_col};
}

// ############################### Size ##############################

SizeColSet::SizeColSet(bool combine_step, bool use_sql_rules)
    : BasicColSet(nullptr, Bodo_FTypes::size, combine_step, use_sql_rules) {}
SizeColSet::~SizeColSet() = default;

void SizeColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    out_cols.push_back(alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::INT64, -1, 0, 0,
        false, false, false, pool, std::move(mm)));
}

void SizeColSet::setInCol(
    std::vector<std::shared_ptr<array_info>> new_in_cols) { /*NOP*/ }

void SizeColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
    do_apply_size(update_cols[0], grp_infos[0]);
}

// ############################## First ##############################

FirstColSet::FirstColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                         bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::first, combine_step, use_sql_rules) {}

FirstColSet::~FirstColSet() = default;

void FirstColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    if (in_col->arr_type == bodo_array_type::ARRAY_ITEM) {
        // For ARRAY_ITEM array, allocate a dummy inner array for now since the
        // true array item array cannot be computed until later.
        std::shared_ptr<array_info> inner_arr =
            alloc_numpy(0, Bodo_CTypes::INT8, pool, mm);
        std::shared_ptr<array_info> out_col =
            alloc_array_item(num_groups, inner_arr, 0, pool, mm);
        out_cols.push_back(out_col);
    } else if (in_col->arr_type == bodo_array_type::MAP) {
        // For MAP array, allocate dummy inner arrays for
        // now since the true array cannot be computed until later
        std::shared_ptr<array_info> inner_arr =
            alloc_numpy(0, Bodo_CTypes::INT8, pool, mm);
        std::shared_ptr<array_info> inner_item_arr =
            alloc_array_item(num_groups, inner_arr, 0, pool, mm);
        std::shared_ptr<array_info> out_col =
            alloc_map(num_groups, inner_item_arr);
        out_cols.push_back(out_col);
    } else if (in_col->arr_type == bodo_array_type::STRUCT) {
        // For STRUCT array, allocate dummy inner arrays for
        // now since the true array cannot be computed until later for
        // string/nested arrays
        std::vector<std::shared_ptr<array_info>> child_arrays;
        for (auto child : in_col->child_arrays) {
            child_arrays.push_back(alloc_numpy(0, Bodo_CTypes::INT8, pool, mm));
        }
        std::shared_ptr<array_info> out_col =
            alloc_struct(num_groups, child_arrays, 0, pool, mm);
        out_cols.push_back(out_col);
    } else {
        BasicColSet::alloc_running_value_columns(num_groups, out_cols, pool,
                                                 mm);
    }
    if (in_col->arr_type == bodo_array_type::NUMPY) {
        std::shared_ptr<array_info> bitmask = alloc_nullable_array_no_nulls(
            num_groups, Bodo_CTypes::_BOOL, 0, pool, std::move(mm));
        memset(bitmask->data1<bodo_array_type::NUMPY>(), 0,
               arrow::bit_util::BytesForBits(num_groups));
        out_cols.push_back(bitmask);
    }
}

void FirstColSet::update(const std::vector<grouping_info>& grp_infos,
                         bodo::IBufferPool* const pool,
                         std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols(update_cols.begin() + 1,
                                                      update_cols.end());
    aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
    do_apply_to_column(in_col, update_cols[0], aux_cols, grp_infos[0], ftype,
                       pool, std::move(mm));
}

void FirstColSet::combine(const grouping_info& grp_info,
                          int64_t init_start_row) {
    int combine_ftype = get_combine_func(ftype);
    std::vector<std::shared_ptr<array_info>> aux_cols(combine_cols.begin() + 1,
                                                      combine_cols.end());
    aggfunc_output_initialize(combine_cols[0], combine_ftype, use_sql_rules,
                              init_start_row);
    if (aux_cols.size() == 1) {
        arrow::bit_util::ClearBitmap((uint8_t*)combine_cols[1]->data1(),
                                     init_start_row, combine_cols[1]->length);
        arrow::bit_util::SetBitmap((uint8_t*)combine_cols[1]->null_bitmask(),
                                   init_start_row, combine_cols[1]->length);
    }
    do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                       combine_ftype);
}

// ############################## Mean ##############################

MeanColSet::MeanColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                       bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::mean, combine_step, use_sql_rules) {}

MeanColSet::~MeanColSet() = default;

void MeanColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> c1 = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
        mm);  // for sum and result
    std::shared_ptr<array_info> c2 = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::UINT64, -1, 0, 0, false, false, false, pool,
        std::move(mm));  // for counts
    out_cols.push_back(c1);
    out_cols.push_back(c2);
}

void MeanColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {this->update_cols[1]};
    aggfunc_output_initialize(this->update_cols[0], this->ftype, use_sql_rules);
    aggfunc_output_initialize(this->update_cols[1], this->ftype, use_sql_rules);
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype, pool, std::move(mm));
}

void MeanColSet::combine(const grouping_info& grp_info,
                         int64_t init_start_row) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    aggfunc_output_initialize(this->combine_cols[0], this->ftype, use_sql_rules,
                              init_start_row);
    // Initialize the output as mean to match the nullable behavior.

    aggfunc_output_initialize(this->combine_cols[1], this->ftype, use_sql_rules,
                              init_start_row);
    do_apply_to_column(this->update_cols[0], this->combine_cols[0], aux_cols,
                       grp_info, Bodo_FTypes::sum);
    do_apply_to_column(this->update_cols[1], this->combine_cols[1], aux_cols,
                       grp_info, Bodo_FTypes::sum);
}

void MeanColSet::eval(const grouping_info& grp_info,
                      bodo::IBufferPool* const pool,
                      std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    if (this->combine_step) {
        do_apply_to_column(this->combine_cols[1], this->combine_cols[0],
                           aux_cols, grp_info, Bodo_FTypes::mean_eval, pool,
                           std::move(mm));
    } else {
        do_apply_to_column(this->update_cols[1], this->update_cols[0], aux_cols,
                           grp_info, Bodo_FTypes::mean_eval, pool,
                           std::move(mm));
    }
}

// ############################## IdxMin/IdxMax ##############################

IdxMinMaxColSet::IdxMinMaxColSet(std::shared_ptr<array_info> in_col,
                                 std::shared_ptr<array_info> _index_col,
                                 int ftype, bool combine_step,
                                 bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules),
      index_col(std::move(_index_col)) {}

IdxMinMaxColSet::~IdxMinMaxColSet() = default;

void IdxMinMaxColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // output column containing index values. dummy for now. will be
    // assigned the real data at the end of update()
    std::shared_ptr<array_info> out_col = alloc_array_top_level(
        num_groups, 1, 1, index_col->arr_type, index_col->dtype, -1, 0, 0,
        false, false, false, pool, mm);
    // create array to store min/max value
    std::shared_ptr<array_info> max_col = alloc_array_top_level(
        num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, -1, 0, 0,
        false, false, false, pool, std::move(mm));  // for min/max

    out_cols.push_back(out_col);
    out_cols.push_back(max_col);
}

void IdxMinMaxColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    this->alloc_running_value_columns(num_groups, out_cols, pool, mm);
    this->update_cols = out_cols;

    // create array to store index position of min/max value
    std::shared_ptr<array_info> index_pos_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, -1, 0, 0,
        false, false, false, pool, std::move(mm));

    update_cols.push_back(index_pos_col);
}

void IdxMinMaxColSet::update(const std::vector<grouping_info>& grp_infos,
                             bodo::IBufferPool* const pool,
                             std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> index_pos_col = this->update_cols[2];
    std::vector<std::shared_ptr<array_info>> aux_cols = {index_pos_col};
    if (this->ftype == Bodo_FTypes::idxmax) {
        aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::max,
                                  use_sql_rules);
    } else {
        // Bodo_FTypes::idxmin
        aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::min,
                                  use_sql_rules);
    }
    aggfunc_output_initialize(index_pos_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero init

    do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                       grp_infos[0], this->ftype, pool, mm);

    std::shared_ptr<array_info> real_out_col = RetrieveArray_SingleColumn_arr(
        index_col, index_pos_col, false, pool, std::move(mm));
    std::shared_ptr<array_info> out_col = this->update_cols[0];
    *out_col = std::move(*real_out_col);
    this->update_cols.pop_back();
}

void IdxMinMaxColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    alloc_running_value_columns(num_groups, out_cols);
    this->combine_cols = out_cols;

    // create array to store index position of min/max value
    std::shared_ptr<array_info> index_pos_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64);

    combine_cols.push_back(index_pos_col);
}

void IdxMinMaxColSet::combine(const grouping_info& grp_info,
                              int64_t init_start_row) {
    std::shared_ptr<array_info> index_pos_col = this->combine_cols[2];
    std::vector<std::shared_ptr<array_info>> aux_cols = {index_pos_col};
    if (this->ftype == Bodo_FTypes::idxmax) {
        aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::max,
                                  use_sql_rules, init_start_row);
    } else {
        // Bodo_FTypes::idxmin
        aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::min,
                                  use_sql_rules, init_start_row);
    }
    aggfunc_output_initialize(index_pos_col, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);  // zero init
    do_apply_to_column(this->update_cols[1], this->combine_cols[1], aux_cols,
                       grp_info, this->ftype);

    std::shared_ptr<array_info> real_out_col =
        RetrieveArray_SingleColumn_arr(this->update_cols[0], index_pos_col);
    std::shared_ptr<array_info> out_col = this->combine_cols[0];
    *out_col = std::move(*real_out_col);
    this->combine_cols.pop_back();
}

// ############################## BoolXorColSet ##############################

BoolXorColSet::BoolXorColSet(std::shared_ptr<array_info> in_col, int ftype,
                             bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

BoolXorColSet::~BoolXorColSet() = default;

void BoolXorColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> one_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::_BOOL, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> two_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::_BOOL, -1, 0, 0, false, false, false, pool, std::move(mm));
    aggfunc_output_initialize(one_col, this->ftype, use_sql_rules);
    aggfunc_output_initialize(two_col, this->ftype, use_sql_rules);
    out_cols.push_back(one_col);
    out_cols.push_back(two_col);
}

void BoolXorColSet::update(const std::vector<grouping_info>& grp_infos,
                           bodo::IBufferPool* const pool,
                           std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {this->update_cols[1]};
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype, pool, std::move(mm));
}

void BoolXorColSet::combine(const grouping_info& grp_info,
                            int64_t init_start_row) {
    std::shared_ptr<array_info> one_col_in = this->update_cols[0];
    std::shared_ptr<array_info> two_col_in = this->update_cols[1];
    std::shared_ptr<array_info> one_col_out = this->combine_cols[0];
    std::shared_ptr<array_info> two_col_out = this->combine_cols[1];
    aggfunc_output_initialize(one_col_out, Bodo_FTypes::boolxor_agg,
                              use_sql_rules, init_start_row);
    aggfunc_output_initialize(two_col_out, Bodo_FTypes::boolxor_agg,
                              use_sql_rules, init_start_row);
    boolxor_combine(one_col_in, two_col_in, one_col_out, two_col_out, grp_info);
}

void BoolXorColSet::eval(const grouping_info& grp_info,
                         bodo::IBufferPool* const pool,
                         std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }
    // Perform the evaluation step with one_col used as the output column
    // and two_col as the aux column
    std::vector<std::shared_ptr<array_info>> aux_cols = {mycols->at(1)};
    do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                       Bodo_FTypes::boolxor_eval, pool, std::move(mm));
}

std::unique_ptr<bodo::Schema> BoolXorColSet::getRunningValueColumnTypes(
    const std::shared_ptr<bodo::Schema>& in_schema) const {
    std::vector<std::unique_ptr<bodo::DataType>> datatypes;
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL));
    return std::make_unique<bodo::Schema>(std::move(datatypes));
}

// ############################## Var/Std ##############################

VarStdColSet::VarStdColSet(std::shared_ptr<array_info> in_col, int ftype,
                           bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

VarStdColSet::~VarStdColSet() = default;

void VarStdColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> count_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::UINT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> mean_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m2_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
        std::move(mm));
    aggfunc_output_initialize(count_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(mean_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m2_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(count_col);
    out_cols.push_back(mean_col);
    out_cols.push_back(m2_col);
}

void VarStdColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Starting index for the loop where we copy the running value column into
    // update_cols. 1 if we are not doing a combine and therefore allocating
    // out_col, 0 otherwise.
    int init_start = 0;

    // If I am not doing a combine, allocate the ouput as well
    // This is needed due to some technical debt with transform/UDF colsets
    if (!this->combine_step && alloc_out_if_no_combine) {
        // need to create output column now
        std::shared_ptr<array_info> col = alloc_array_top_level(
            num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
            mm);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->out_col = col;
        init_start++;
    }
    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = init_start; i < out_cols.size(); i++) {
        this->update_cols.push_back(out_cols.at(i));
    }
}

void VarStdColSet::update(const std::vector<grouping_info>& grp_infos,
                          bodo::IBufferPool* const pool,
                          std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {
        this->update_cols[0], this->update_cols[1], this->update_cols[2]};
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype, pool, std::move(mm));
}

void VarStdColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    this->out_col = col;
    out_cols.push_back(col);

    this->alloc_running_value_columns(num_groups, out_cols);

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = 1; i < out_cols.size(); i++) {
        this->combine_cols.push_back(out_cols.at(i));
    }
}

void VarStdColSet::combine(const grouping_info& grp_info,
                           int64_t init_start_row) {
    const std::shared_ptr<array_info>& count_col_in = this->update_cols[0];
    const std::shared_ptr<array_info>& mean_col_in = this->update_cols[1];
    const std::shared_ptr<array_info>& m2_col_in = this->update_cols[2];
    const std::shared_ptr<array_info>& count_col_out = this->combine_cols[0];
    const std::shared_ptr<array_info>& mean_col_out = this->combine_cols[1];
    const std::shared_ptr<array_info>& m2_col_out = this->combine_cols[2];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                mean_col_out, m2_col_out, grp_info);
}

void VarStdColSet::eval(const grouping_info& grp_info,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(0), mycols->at(1), mycols->at(2)};

    // allocate output if not done already (streaming groupby doesn't call
    // alloc_combine_columns)
    if (this->out_col == nullptr) {
        this->out_col = alloc_array_top_level(
            mycols->at(0)->length, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

    switch (this->ftype) {
        case Bodo_FTypes::var_pop: {
            do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                               Bodo_FTypes::var_pop_eval, pool, std::move(mm));
            break;
        }
        case Bodo_FTypes::std_pop: {
            do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                               Bodo_FTypes::std_pop_eval, pool, std::move(mm));
            break;
        }
        case Bodo_FTypes::var: {
            do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                               Bodo_FTypes::var_eval, pool, std::move(mm));
            break;
        }
        case Bodo_FTypes::std: {
            do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                               Bodo_FTypes::std_eval, pool, std::move(mm));
            break;
        }
    }
}

std::unique_ptr<bodo::Schema> VarStdColSet::getRunningValueColumnTypes(
    const std::shared_ptr<bodo::Schema>& in_schema) const {
    // var/std's update columns are always uint64 for count and float64 for
    // mean and m2 data. See VarStdColSet::alloc_running_value_columns()

    std::vector<std::unique_ptr<bodo::DataType>> datatypes;
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::UINT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    return std::make_unique<bodo::Schema>(std::move(datatypes));
}

// ############################## Skew ##############################

SkewColSet::SkewColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

SkewColSet::~SkewColSet() = default;

void SkewColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> count_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::UINT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m1_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m2_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m3_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
        std::move(mm));
    aggfunc_output_initialize(count_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m1_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m2_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m3_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(count_col);
    out_cols.push_back(m1_col);
    out_cols.push_back(m2_col);
    out_cols.push_back(m3_col);
}

void SkewColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Starting index for the loop where we copy the running value column into
    // update_cols. 1 if we are not doing a combine and therefore allocating
    // out_col, 0 otherwise.
    int init_start = 0;

    // If I am not doing a combine, allocate the ouput as well
    // This is needed due to some technical debt with transform/UDF colsets
    if (!this->combine_step && alloc_out_if_no_combine) {
        // need to create output column now
        std::shared_ptr<array_info> col = alloc_array_top_level(
            num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
            mm);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->out_col = col;
        init_start++;
    }
    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = init_start; i < out_cols.size(); i++) {
        this->update_cols.push_back(out_cols.at(i));
    }
}

void SkewColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {
        this->update_cols[0], this->update_cols[1], this->update_cols[2],
        this->update_cols[3]};
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype, pool, std::move(mm));
}

void SkewColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    this->out_col = col;
    out_cols.push_back(col);

    this->alloc_running_value_columns(num_groups, out_cols);

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = 1; i < out_cols.size(); i++) {
        this->combine_cols.push_back(out_cols.at(i));
    }
}

void SkewColSet::combine(const grouping_info& grp_info,
                         int64_t init_start_row) {
    const std::shared_ptr<array_info>& count_col_in = this->update_cols[0];
    const std::shared_ptr<array_info>& m1_col_in = this->update_cols[1];
    const std::shared_ptr<array_info>& m2_col_in = this->update_cols[2];
    const std::shared_ptr<array_info>& m3_col_in = this->update_cols[3];
    const std::shared_ptr<array_info>& count_col_out = this->combine_cols[0];
    const std::shared_ptr<array_info>& m1_col_out = this->combine_cols[1];
    const std::shared_ptr<array_info>& m2_col_out = this->combine_cols[2];
    const std::shared_ptr<array_info>& m3_col_out = this->combine_cols[3];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m1_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m3_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    skew_combine(count_col_in, m1_col_in, m2_col_in, m3_col_in, count_col_out,
                 m1_col_out, m2_col_out, m3_col_out, grp_info);
}

void SkewColSet::eval(const grouping_info& grp_info,
                      bodo::IBufferPool* const pool,
                      std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    // allocate output if not done already (streaming groupby doesn't call
    // alloc_combine_columns)
    if (this->out_col == nullptr) {
        this->out_col = alloc_array_top_level(
            mycols->at(0)->length, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(0), mycols->at(1), mycols->at(2), mycols->at(3)};
    do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                       Bodo_FTypes::skew_eval, pool, std::move(mm));
}

std::unique_ptr<bodo::Schema> SkewColSet::getRunningValueColumnTypes(
    const std::shared_ptr<bodo::Schema>& in_schema) const {
    // Skew's update columns are always uint64 for count and float64 for
    // m1/m2/m3 data. See SkewColSet::alloc_running_value_columns()
    std::vector<std::unique_ptr<bodo::DataType>> datatypes;
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::UINT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    return std::make_unique<bodo::Schema>(std::move(datatypes));
}

// ############################## Listagg ##############################

ListAggColSet::ListAggColSet(
    std::shared_ptr<array_info> in_col, std::shared_ptr<array_info> sep_col,
    std::vector<std::shared_ptr<array_info>> _orderby_cols,
    std::vector<bool> _window_ascending, std::vector<bool> _window_na_position)
    : BasicColSet(in_col, Bodo_FTypes::listagg, false, true),
      orderby_cols(std::move(_orderby_cols)),
      window_ascending(std::move(_window_ascending)),
      window_na_position(std::move(_window_na_position)) {
    std::shared_ptr<array_info> string_array;
    if (sep_col->arr_type == bodo_array_type::arr_type_enum::DICT) {
        string_array = sep_col->child_arrays[0];

    } else if (sep_col->arr_type == bodo_array_type::arr_type_enum::STRING) {
        // If NUMBA_DEVELOPER_MODE, output stderr message
        if (std::getenv("NUMBA_DEVELOPER_MODE") != nullptr) {
            std::cerr
                << "Internal error in ListAggColSet constructor: Separator "
                   "array is not dictionary type.\n";
        }

        string_array = sep_col;
    } else {
        throw std::runtime_error(
            "Internal error in ListAggColSet constructor: Separator array must "
            "always be "
            "dictionary type.");
    }

    char* data_in = string_array->data1<bodo_array_type::STRING>();
    offset_t* offsets_in =
        (offset_t*)string_array->data2<bodo_array_type::STRING>();
    offset_t end_offset = offsets_in[1];
    std::string substr(&data_in[0], end_offset);
    this->listagg_sep = substr;
}

ListAggColSet::~ListAggColSet() = default;

void ListAggColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Because we can't do a proper allocation until after the shuffle has
    // ocurred, we allocate a dummy update column here. We then overwrite the
    // internally stored value in the update function.
    out_cols.push_back(alloc_string_array(Bodo_CTypes::STRING, 0, 0, -1, 0,
                                          false, false, false, pool,
                                          std::move(mm)));
}

// Struct that contains templated helper functions
// for string/dict encoded strings in listagg update
template <bool is_dict>
struct listagg_groupby_utils {
    inline static void get_data_and_offsets(
        const std::shared_ptr<array_info> in_col, char** data_in_ptr,
        offset_t** offsets_in_ptr,
        std::shared_ptr<array_info>* dict_indices_ptr);
    inline static bool get_null_bit(std::shared_ptr<array_info> in_col,
                                    std::shared_ptr<array_info> dict_indices,
                                    size_t i);
    inline static int32_t get_offset_idx(
        std::shared_ptr<array_info> dict_indices, size_t i);
};

template <>
struct listagg_groupby_utils<true> {
    inline static void get_data_and_offsets(
        const std::shared_ptr<array_info> in_col, char** data_in_ptr,
        offset_t** offsets_in_ptr,
        std::shared_ptr<array_info>* dict_indices_ptr) {
        *dict_indices_ptr = in_col->child_arrays[1];
        std::shared_ptr<array_info> string_array = in_col->child_arrays[0];

        *data_in_ptr = string_array->data1<bodo_array_type::STRING>();
        *offsets_in_ptr =
            (offset_t*)string_array->data2<bodo_array_type::STRING>();
    }
    inline static bool get_null_bit(std::shared_ptr<array_info> in_col,
                                    std::shared_ptr<array_info> dict_indices,
                                    size_t i) {
        return dict_indices->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
            i);
    }
    inline static int32_t get_offset_idx(
        std::shared_ptr<array_info> dict_indices, size_t i) {
        return getv<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(dict_indices,
                                                                 i);
    }
};

template <>
struct listagg_groupby_utils<false> {
    inline static void get_data_and_offsets(
        const std::shared_ptr<array_info> in_col, char** data_in_ptr,
        offset_t** offsets_in_ptr,
        std::shared_ptr<array_info>* dict_indices_ptr) {
        *data_in_ptr = in_col->data1();
        *offsets_in_ptr = (offset_t*)in_col->data2();
    }
    inline static bool get_null_bit(std::shared_ptr<array_info> in_col,
                                    std::shared_ptr<array_info> dict_indices,
                                    size_t i) {
        return in_col->get_null_bit<bodo_array_type::STRING>(i);
    }
    inline static int32_t get_offset_idx(
        std::shared_ptr<array_info> dict_indices, size_t i) {
        return i;
    }
};

// Helper function that returns the order in which to traverse the data elements
// in the input column for listagg update.
// (This essentially returns an argsort)
std::shared_ptr<array_info> get_traversal_order(
    const std::shared_ptr<array_info> in_col,
    const std::vector<bool> window_ascending,
    const std::vector<std::shared_ptr<array_info>> orderby_cols,
    const std::vector<bool> window_na_position,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    int64_t n_sort_keys = orderby_cols.size();
    int64_t num_rows = in_col->length;

    // Append an index column so we can find the original
    // index in the out array.
    std::shared_ptr<array_info> idx_arr =
        alloc_numpy(num_rows, Bodo_CTypes::INT64, pool, mm);
    for (int64_t i = 0; i < num_rows; i++) {
        getv<int64_t, bodo_array_type::NUMPY>(idx_arr, i) = i;
    }

    if (n_sort_keys == 0) {
        return idx_arr;
    }

    // Create a new table. We want to sort the table first by
    // the groups and second by the orderby_arr.
    std::shared_ptr<table_info> sort_table = std::make_shared<table_info>();

    for (std::shared_ptr<array_info> orderby_arr : orderby_cols) {
        sort_table->columns.push_back(orderby_arr);
    }

    sort_table->columns.push_back(idx_arr);

    std::vector<int64_t> window_ascending_real(n_sort_keys);
    std::vector<int64_t> window_na_position_real(n_sort_keys);

    // Initialize the group ordering
    // Ignore the value for the separator argument and data argument
    // present at the beginning.
    // (they are dummy values, required to get the rest of this to work)
    for (int64_t i = 0; i < n_sort_keys; i++) {
        window_ascending_real[i] = window_ascending[i + 2];
        window_na_position_real[i] = window_na_position[i + 2];
    }

    // new initializes to 0
    //  int64_t* dead_keys = std::vector<int64_t>(n_sort_keys + 1, 1).data();
    std::vector<int64_t> dead_keys(n_sort_keys);
    for (int64_t i = 0; i < n_sort_keys; i++) {
        // Mark all keys as dead.
        dead_keys[i] = 1;
    }
    // Sort the table
    // XXX: We don't need the entire chunk of data sorted,
    // just the final column. We could do a partial sort to avoid
    // the overhead of sorting the orderby columns in the future.
    std::shared_ptr<table_info> iter_table = sort_values_table_local(
        sort_table, n_sort_keys, window_ascending_real.data(),
        window_na_position_real.data(), dead_keys.data(),
        // TODO: set this correctly
        false /* This is just used for tracing */, pool, std::move(mm));
    // All keys are dead so the sorted_idx is column 0.
    std::shared_ptr<array_info> sorted_idx = iter_table->columns[0];
    return sorted_idx;
}

/**
 * Helper function that does the bulk of the work of update for listagg. The
 * high level algorithm is as follows: 0. Find the expected order of the input
 * array (used to traverse the input array in the final step)
 * 1. Find the expected length of each output string (one output string for each
 * group)
 * 2. Determine the offsets of each output string (used to set the offsets of
 * the output string array)
 * 3. Allocate the output array, and set the offsets accordingly.
 * 4. Traverse the input array, copying the data + separator into the output
 * array. The traversal order of the input array is dependent on the order
 * obtained in step 0
 */
template <bool is_dict>
void listagg_update_helper(
    const std::vector<grouping_info>& grp_infos,
    const std::shared_ptr<array_info>& in_col,
    const std::vector<std::shared_ptr<array_info>>& update_cols,
    const std::vector<bool>& window_ascending,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& window_na_position, const std::string listagg_sep,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    const grouping_info grp_info = grp_infos.at(0);
    const size_t listagg_sep_len = listagg_sep.length();
    const size_t num_groups = grp_info.num_groups;

    // Handle the dict/str cases
    // These two arguments are initialized in both cases
    char* data_in = nullptr;
    offset_t* offsets_in = nullptr;
    // dict_indices is only used for dict array, uninitialized in string case
    std::shared_ptr<array_info> dict_indices = nullptr;

    listagg_groupby_utils<is_dict>::get_data_and_offsets(
        in_col, &data_in, &offsets_in, &dict_indices);

    //----------Step 0, find the traversal order of the input array----------
    std::shared_ptr<array_info> traversal_order = get_traversal_order(
        in_col, window_ascending, orderby_cols, window_na_position, pool, mm);

    //----------Step 1, Find the expected length of each output string----------

    // This variable is confusingly named. During step 1, it will contain
    // the length values for each output string. In step 2,
    // we will re-use this array to store the offsets for each output string
    // num_groups + 1 is needed to store the final offset value, but is unused
    // during step 1
    // In the final step, we will use this array to store the offsets of the
    // current end of each of the output strings.
    bodo::vector<offset_t> str_offsets(num_groups + 1, 0, pool);

    // Initialize bool array to False
    bodo::vector<bool> seen_non_nulls =
        bodo::vector<bool>(num_groups, false, pool);

    // First, we need to figure out the length of each of the output strings
    // for each group we store this information in str_offsets
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = grp_info.row_to_group[i];

        if (i_grp == -1) {
            continue;
        }

        bool null_bit = listagg_groupby_utils<is_dict>::get_null_bit(
            in_col, dict_indices, i);

        if (null_bit) {
            int32_t offsets_idx =
                listagg_groupby_utils<is_dict>::get_offset_idx(dict_indices, i);

            offset_t len =
                offsets_in[offsets_idx + 1] - offsets_in[offsets_idx];

            // seen_non_nulls[i_grp]
            if (seen_non_nulls[i_grp]) {
                // if this is not the first string in the group, we need to
                // add the separator length to the length of the string
                len += listagg_sep_len;
            }
            seen_non_nulls[i_grp] = true;
            str_offsets[i_grp + 1] += len;
        }
    }

    //----------Step 2, Determine the offsets of each output string----------

    // Convert the length information into the actual offset information
    // y computing the sum of each prefix of the lengths
    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());

    size_t total_output_size = str_offsets[num_groups];

    //----------Step 3, Allocate the output array, and set the offsets
    // accordingly----------
    std::shared_ptr<array_info> real_out_arr = alloc_string_array(
        Bodo_CTypes::CTypeEnum::STRING, num_groups, total_output_size, -1, 0,
        false, false, false, pool, mm);
    char* data_out = real_out_arr->data1<bodo_array_type::STRING>();

    offset_t* offsets_out =
        (offset_t*)real_out_arr->data2<bodo_array_type::STRING>();

    // copy to the output offset array
    // This should be num_groups + 1. If you look inside the allocate,
    // string function, the allocated offset length is 1+number of strings.
    // The last element should be the offset of the end of the last string.
    memcpy(offsets_out, str_offsets.data(),
           (num_groups + 1) * sizeof(offset_t));

    //----------Step 4, Traverse the input array, copying the data + separator
    // into the output array.----------

    const char* listagg_sep_c_str = listagg_sep.c_str();

    // Initialize bool array to False
    // Used to track if we should insert the separator string or not
    bodo::vector<bool> seen_non_nulls_2 =
        bodo::vector<bool>(num_groups, false, pool);

    // copy characters to output
    for (size_t j = 0; j < in_col->length; j++) {
        size_t i = getv<int64_t>(traversal_order, j);

        int64_t i_grp = grp_info.row_to_group[i];
        bool null_bit = listagg_groupby_utils<is_dict>::get_null_bit(
            in_col, dict_indices, i);

        // NOTE: i_grp != -1  can happen when the group key is null and,
        // we still want to do group operation on it (i.e.
        // groupby(dropna=False))
        if ((i_grp != -1) && null_bit) {
            int32_t offsets_idx =
                listagg_groupby_utils<is_dict>::get_offset_idx(dict_indices, i);

            offset_t input_string_len =
                offsets_in[offsets_idx + 1] - offsets_in[offsets_idx];

            if (seen_non_nulls_2[i_grp]) {
                // if this is not the first string in the group, we need to
                // add the separator the output string string

                memcpy(&data_out[str_offsets[i_grp]], listagg_sep_c_str,
                       listagg_sep_len);
                memcpy(&data_out[str_offsets[i_grp] + listagg_sep_len],
                       data_in + offsets_in[offsets_idx], input_string_len);

                str_offsets[i_grp] += input_string_len + listagg_sep_len;
            } else {
                memcpy(&data_out[str_offsets[i_grp]],
                       data_in + offsets_in[offsets_idx], input_string_len);
                str_offsets[i_grp] += input_string_len;
            }
            seen_non_nulls_2[i_grp] = true;
        }
    }

    std::shared_ptr<array_info> out_col = update_cols[0];
    *out_col = std::move(*real_out_arr);
}

void ListAggColSet::update(const std::vector<grouping_info>& grp_infos,
                           bodo::IBufferPool* const pool,
                           std::shared_ptr<::arrow::MemoryManager> mm) {
    if (this->combine_step) {
        throw std::runtime_error(
            "Internal error in ListAggColSet::update: listAgg must always "
            "shuffle before update");
    } else {
        if (grp_infos.size() != 1) {
            throw std::runtime_error(
                "Internal error in ListAggColSet::update: grp_infos length "
                "does not equal 1. Each ListAggColSet should handle only one "
                "group");
        }
        if (in_col->arr_type == bodo_array_type::arr_type_enum::STRING) {
            listagg_update_helper<false>(
                grp_infos, this->in_col, this->update_cols,
                this->window_ascending, this->orderby_cols,
                this->window_na_position, this->listagg_sep, pool,
                std::move(mm));
        } else if (in_col->arr_type == bodo_array_type::arr_type_enum::DICT) {
            listagg_update_helper<true>(
                grp_infos, this->in_col, this->update_cols,
                this->window_ascending, this->orderby_cols,
                this->window_na_position, this->listagg_sep, pool,
                std::move(mm));
        } else {
            throw std::runtime_error(
                "Internal error in ListAggColSet::update: input "
                "data column must be of type string or dict encoded string. "
                "Found: " +
                GetArrType_as_string(in_col->arr_type));
        }
    }
}

// Since we don't do combine, output column is always at update_cols[0]
std::shared_ptr<array_info> ListAggColSet::getOutputColumn() {
    std::shared_ptr<array_info> out_col = update_cols.at(0);
    return out_col;
}

// ############################## ArrayAgg ##############################

ArrayAggColSet::ArrayAggColSet(
    std::shared_ptr<array_info> in_col,
    std::vector<std::shared_ptr<array_info>> _orderby_cols,
    std::vector<bool> _ascending, std::vector<bool> _na_position, int ftype,
    bool _is_parallel)
    : BasicColSet(in_col, ftype, false, true),
      orderby_cols(std::move(_orderby_cols)),
      ascending(std::move(_ascending)),
      na_position(std::move(_na_position)),
      is_distinct(ftype == Bodo_FTypes::array_agg_distinct),
      is_parallel(_is_parallel) {}

ArrayAggColSet::~ArrayAggColSet() = default;

void ArrayAggColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
    // Need to allocate a dummy inner column for now since we cannot allocate
    // the real array until we know the sizes; the real inner array will be
    // handled later.
    std::shared_ptr inner_arr =
        alloc_array_top_level(0, 1, 1, in_col->arr_type, dtype, -1, 0, 0, false,
                              false, false, pool, mm);
    out_cols.push_back(
        alloc_array_item(num_groups, inner_arr, 0, pool, std::move(mm)));
}

void ArrayAggColSet::update(const std::vector<grouping_info>& grp_infos,
                            bodo::IBufferPool* const pool,
                            std::shared_ptr<::arrow::MemoryManager> mm) {
    if (this->combine_step) {
        throw std::runtime_error(
            "Internal error in ArrayAggColSet::update: array_agg must always "
            "shuffle before update");
    } else {
        if (grp_infos.size() != 1) {
            throw std::runtime_error(
                "Internal error in ArrayAggColSet::update: grp_infos length "
                "does not equal 1. Each ArrayAggColSet should handle only one "
                "group");
        }
        array_agg_computation(in_col, update_cols[0], orderby_cols, ascending,
                              na_position, grp_infos[0], is_parallel,
                              is_distinct, pool, std::move(mm));
    }
}

// Since we don't do combine, output column is always at update_cols[0]
const std::vector<std::shared_ptr<array_info>>
ArrayAggColSet::getOutputColumns() {
    return {update_cols[0]};
}

// ############################## ArrayAgg ##############################

ObjectAggColSet::ObjectAggColSet(std::shared_ptr<array_info> _key_col,
                                 std::shared_ptr<array_info> _val_col,
                                 bool _is_parallel)
    : BasicColSet(_key_col, Bodo_FTypes::object_agg, false, true),
      key_col(_key_col),
      val_col(std::move(_val_col)),
      is_parallel(_is_parallel) {}

ObjectAggColSet::~ObjectAggColSet() = default;

void ObjectAggColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> child_arrays;
    // Allocate dummy array for the struct array containing the keys and values
    // since they will be replaced later.
    std::shared_ptr inner_arr = alloc_numpy(0, Bodo_CTypes::INT64);
    out_cols.push_back(alloc_map(
        num_groups,
        alloc_array_item(num_groups, inner_arr, 0, pool, std::move(mm))));
}

void ObjectAggColSet::update(const std::vector<grouping_info>& grp_infos,
                             bodo::IBufferPool* const pool,
                             std::shared_ptr<::arrow::MemoryManager> mm) {
    if (this->combine_step) {
        throw std::runtime_error(
            "Internal error in ObjectAggColSet::update: array_agg must always "
            "shuffle before update");
    } else {
        if (grp_infos.size() != 1) {
            throw std::runtime_error(
                "Internal error in ObjectAggColSet::update: grp_infos length "
                "does not equal 1. Each ObjectAggColSet should handle only one "
                "group");
        }
        object_agg_computation(key_col, val_col, update_cols[0], grp_infos[0],
                               is_parallel, pool, std::move(mm));
    }
}

// Since we don't do combine, output column is always at update_cols[0]
const std::vector<std::shared_ptr<array_info>>
ObjectAggColSet::getOutputColumns() {
    return {update_cols[0]};
}

// ############################## Kurtosis ##############################

KurtColSet::KurtColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

KurtColSet::~KurtColSet() = default;

void KurtColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Starting index for the loop where we copy the running value column into
    // update_cols. 1 if we are not doing a combine and therefore allocating
    // out_col, 0 otherwise.
    int init_start = 0;

    // If I am not doing a combine, allocate the ouput as well
    // This is needed due to some technical debt with transform/UDF colsets
    if (!this->combine_step && alloc_out_if_no_combine) {
        // need to create output column now
        std::shared_ptr<array_info> col = alloc_array_top_level(
            num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
            mm);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->out_col = col;
        init_start++;
    }
    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = init_start; i < out_cols.size(); i++) {
        this->update_cols.push_back(out_cols.at(i));
    }
}

void KurtColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> count_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::UINT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m1_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m2_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m3_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
    std::shared_ptr<array_info> m4_col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool,
        std::move(mm));
    aggfunc_output_initialize(count_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m1_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m2_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m3_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m4_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(count_col);
    out_cols.push_back(m1_col);
    out_cols.push_back(m2_col);
    out_cols.push_back(m3_col);
    out_cols.push_back(m4_col);
}

void KurtColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {
        this->update_cols[0], this->update_cols[1], this->update_cols[2],
        this->update_cols[3], this->update_cols[4]};
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype, pool, std::move(mm));
}

void KurtColSet::eval(const grouping_info& grp_info,
                      bodo::IBufferPool* const pool,
                      std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    // allocate output if not done already (streaming groupby doesn't call
    // alloc_combine_columns)
    if (this->out_col == nullptr) {
        this->out_col = alloc_array_top_level(
            mycols->at(0)->length, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
            Bodo_CTypes::FLOAT64, -1, 0, 0, false, false, false, pool, mm);
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(0), mycols->at(1), mycols->at(2), mycols->at(3),
        mycols->at(4)};
    do_apply_to_column(this->out_col, this->out_col, aux_cols, grp_info,
                       Bodo_FTypes::kurt_eval, pool, std::move(mm));
}

void KurtColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col = alloc_array_top_level(
        num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
        Bodo_CTypes::FLOAT64);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    this->out_col = col;
    out_cols.push_back(col);

    this->alloc_running_value_columns(num_groups, out_cols);

    // Add every value to update cols, except the first one
    //(the output column)
    for (size_t i = 1; i < out_cols.size(); i++) {
        this->combine_cols.push_back(out_cols.at(i));
    }
}

void KurtColSet::combine(const grouping_info& grp_info,
                         int64_t init_start_row) {
    const std::shared_ptr<array_info>& count_col_in = this->update_cols[0];
    const std::shared_ptr<array_info>& m1_col_in = this->update_cols[1];
    const std::shared_ptr<array_info>& m2_col_in = this->update_cols[2];
    const std::shared_ptr<array_info>& m3_col_in = this->update_cols[3];
    const std::shared_ptr<array_info>& m4_col_in = this->update_cols[4];
    const std::shared_ptr<array_info>& count_col_out = this->combine_cols[0];
    const std::shared_ptr<array_info>& m1_col_out = this->combine_cols[1];
    const std::shared_ptr<array_info>& m2_col_out = this->combine_cols[2];
    const std::shared_ptr<array_info>& m3_col_out = this->combine_cols[3];
    const std::shared_ptr<array_info>& m4_col_out = this->combine_cols[4];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m1_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m3_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    aggfunc_output_initialize(m4_col_out, Bodo_FTypes::count, use_sql_rules,
                              init_start_row);
    kurt_combine(count_col_in, m1_col_in, m2_col_in, m3_col_in, m4_col_in,
                 count_col_out, m1_col_out, m2_col_out, m3_col_out, m4_col_out,
                 grp_info);
}

std::unique_ptr<bodo::Schema> KurtColSet::getRunningValueColumnTypes(
    const std::shared_ptr<bodo::Schema>& in_schema) const {
    // Kurt's update columns are always uint64 for count and float64 for
    // m1/m2/m3/m4 data. See KurtColSet::alloc_running_value_columns()
    std::vector<std::unique_ptr<bodo::DataType>> datatypes;
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::UINT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    datatypes.push_back(std::make_unique<bodo::DataType>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
    return std::make_unique<bodo::Schema>(std::move(datatypes));
}

// ############################## UDF ##############################

UdfColSet::UdfColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                     std::shared_ptr<table_info> udf_table, int udf_table_idx,
                     int n_redvars, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::udf, combine_step, use_sql_rules),
      udf_table(std::move(udf_table)),
      udf_table_idx(udf_table_idx),
      n_redvars(n_redvars) {}

UdfColSet::~UdfColSet() = default;

void UdfColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    throw std::runtime_error(
        "Internal error in UdfColSet::alloc_running_value_columns: UdfColSet "
        "should never be call alloc_running_value_columns");
}

void UdfColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    int offset = 0;

    if (this->combine_step) {
        offset = 1;
    }
    // for update table we only need redvars (skip first column which is
    // output column)
    for (int i = udf_table_idx + offset; i < udf_table_idx + 1 + n_redvars;
         i++) {
        // we get the type from the udf dummy table that was passed
        // to C++ library
        bodo_array_type::arr_type_enum arr_type =
            udf_table->columns[i]->arr_type;
        Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
        int64_t num_categories = udf_table->columns[i]->num_categories;
        out_cols.push_back(alloc_array_top_level(
            num_groups, 1, 1, arr_type, dtype, -1, 0, num_categories, false,
            false, false, pool, mm));

        if (!this->combine_step) {
            this->update_cols.push_back(out_cols.back());
        }
    }
}

void UdfColSet::update(const std::vector<grouping_info>& grp_infos,
                       bodo::IBufferPool* const pool,
                       std::shared_ptr<::arrow::MemoryManager> mm) {
    // do nothing because this is done in JIT-compiled code (invoked
    // from GroupbyPipeline once for all udf columns sets)
}

typename std::vector<std::shared_ptr<array_info>>::iterator
UdfColSet::update_after_shuffle(
    typename std::vector<std::shared_ptr<array_info>>::iterator& it) {
    // UdfColSet doesn't keep the update cols, return the updated iterator
    return it + n_redvars;
}

void UdfColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    for (int i = udf_table_idx; i < udf_table_idx + 1 + n_redvars; i++) {
        // we get the type from the udf dummy table that was passed
        // to C++ library
        bodo_array_type::arr_type_enum arr_type =
            udf_table->columns[i]->arr_type;
        Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
        int64_t num_categories = udf_table->columns[i]->num_categories;
        out_cols.push_back(alloc_array_top_level(num_groups, 1, 1, arr_type,
                                                 dtype, -1, 0, num_categories));
        this->combine_cols.push_back(out_cols.back());
    }
}

void UdfColSet::combine(const grouping_info& grp_info, int64_t init_start_row) {
    // do nothing because this is done in JIT-compiled code (invoked
    // from GroupbyPipeline once for all udf columns sets)
}

void UdfColSet::eval(const grouping_info& grp_info,
                     bodo::IBufferPool* const pool,
                     std::shared_ptr<::arrow::MemoryManager> mm) {
    // do nothing because this is done in JIT-compiled code (invoked
    // from GroupbyPipeline once for all udf columns sets)
}

// ############################## GeneralUDF ##############################

GeneralUdfColSet::GeneralUdfColSet(std::shared_ptr<array_info> in_col,
                                   std::shared_ptr<table_info> udf_table,
                                   int udf_table_idx, bool use_sql_rules)
    : UdfColSet(in_col, false, udf_table, udf_table_idx, 0, use_sql_rules) {}

GeneralUdfColSet::~GeneralUdfColSet() = default;

void GeneralUdfColSet::fill_in_columns(
    const std::shared_ptr<table_info>& general_in_table,
    const grouping_info& grp_info) const {
    std::shared_ptr<array_info> in_col = this->in_col;
    bodo::vector<bodo::vector<int64_t>> group_rows(grp_info.num_groups);
    // get the rows in each group
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = grp_info.row_to_group[i];
        group_rows[i_grp].push_back(i);
    }
    // retrieve one column per group from the input column, add it
    // to the general UDF input table
    for (size_t i = 0; i < grp_info.num_groups; i++) {
        std::shared_ptr<array_info> col =
            RetrieveArray_SingleColumn(in_col, group_rows[i]);
        general_in_table->columns.push_back(col);
    }
}

// ############################## StreaminglUDF ##############################
StreamingUDFColSet::StreamingUDFColSet(std::shared_ptr<array_info> in_col,
                                       std::shared_ptr<table_info> udf_table,
                                       int udf_table_idx, stream_udf_t* func,
                                       bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::stream_udf, false, use_sql_rules),
      udf_table(std::move(udf_table)),
      udf_table_idx(udf_table_idx),
      func(func) {}

StreamingUDFColSet::~StreamingUDFColSet() = default;

std::unique_ptr<bodo::Schema> StreamingUDFColSet::getRunningValueColumnTypes(
    const std::shared_ptr<bodo::Schema>& in_schema) const {
    std::vector<int> col_idxs;
    col_idxs.push_back(udf_table_idx);
    return udf_table->schema()->Project(col_idxs);
}

void StreamingUDFColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Allocate a dummy array column, actual column will be filled in by update
    bodo_array_type::arr_type_enum arr_type =
        udf_table->columns[udf_table_idx]->arr_type;
    Bodo_CTypes::CTypeEnum dtype = udf_table->columns[udf_table_idx]->dtype;
    auto update_col = alloc_array_top_level(0, 0, 0, arr_type, dtype);
    out_cols.push_back(std::move(update_col));
}

void StreamingUDFColSet::update(const std::vector<grouping_info>& grp_infos,
                                bodo::IBufferPool* const pool,
                                std::shared_ptr<::arrow::MemoryManager> mm) {
    const grouping_info& grp_info = grp_infos[0];
    std::shared_ptr<array_info> in_col = this->in_col;
    bodo::vector<bodo::vector<int64_t>> group_rows(grp_info.num_groups, pool);

    if (!in_col->length) {
        return;
    }

    // get the rows in each group
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = grp_info.row_to_group[i];
        group_rows[i_grp].push_back(i);
    }

    // TODO: make out_arrs a bodo::vector for large number of groups case
    std::vector<std::shared_ptr<array_info>> out_arrs(grp_info.num_groups);
    for (size_t i = 0; i < grp_info.num_groups; i++) {
        bodo::vector<int64_t> row_idxs = group_rows[i];
        std::shared_ptr<array_info> in_group_arr =
            RetrieveArray_SingleColumn(in_col, row_idxs);
        array_info* out_arr_result = func(in_group_arr.get());

        if (!out_arr_result) {
            throw std::runtime_error(
                "Groupby.agg(): An error occured while executing user defined "
                "function.");
        }
        std::shared_ptr<array_info> out_arr(out_arr_result);

        out_arrs[i] = out_arr;
    }

    std::shared_ptr<array_info> real_out_col = concat_arrays(out_arrs);

    // Replace the dummy update column.
    std::shared_ptr<array_info> out_col = this->update_cols[0];
    *out_col = std::move(*real_out_col);
}

// ############################## Percentile ##############################

PercentileColSet::PercentileColSet(std::shared_ptr<array_info> in_col,
                                   std::shared_ptr<array_info> percentile_col,
                                   bool _interpolate, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::percentile_disc, false, use_sql_rules),
      percentile(getv<double>(percentile_col, 0)),
      interpolate(_interpolate) {
    if (percentile_col->dtype == Bodo_CTypes::INT32 &&
        getv<int>(percentile_col, 0) == 1) {
        // The percentile 1 should be coerced to 1.0.
        this->percentile = 1.0;
    }
}

PercentileColSet::~PercentileColSet() = default;

void PercentileColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));
    this->update_cols = out_cols;
    if (this->in_col->dtype == Bodo_CTypes::DECIMAL && this->interpolate) {
        // For decimal, compute the scale and precision of the output array
        // The input precision and scale are increased by 3, capped at 38.
        // This means that an input scale of 36 or more is invalid.

        // Note we only want to readjust if we are interpolating.
        // PERCENTILE_DISC does not require this adjustment.
        int new_scale = this->in_col->scale + 3;
        int new_precision = std::min(38, this->in_col->precision + 3);
        if (new_scale > 37) {
            std::string err_msg = "Scale " +
                                  std::to_string(this->in_col->scale) +
                                  " is too large for PERCENTILE operation";
            throw std::runtime_error(err_msg);
        }
        this->update_cols[0]->scale = new_scale;
        this->update_cols[0]->precision = new_precision;
    }
}

void PercentileColSet::update(const std::vector<grouping_info>& grp_infos,
                              bodo::IBufferPool* const pool,
                              std::shared_ptr<::arrow::MemoryManager> mm) {
    percentile_computation(this->in_col, this->update_cols[0], this->percentile,
                           this->interpolate, grp_infos[0], pool);
}

// ############################## Median ##############################

MedianColSet::MedianColSet(std::shared_ptr<array_info> in_col,
                           bool _skip_na_data, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::median, false, use_sql_rules),
      skip_na_data(_skip_na_data) {}

MedianColSet::~MedianColSet() = default;

void MedianColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));
    this->update_cols = out_cols;
    if (this->in_col->dtype == Bodo_CTypes::DECIMAL) {
        // For decimal, compute the scale and precision of the output array
        // The input precision and scale are increased by 3, capped at 38.
        // This means that an input scale of 36 or more is invalid.
        int new_scale = this->in_col->scale + 3;
        int new_precision = std::min(38, this->in_col->precision + 3);
        if (new_scale > 37) {
            std::string err_msg = "Scale " +
                                  std::to_string(this->in_col->scale) +
                                  " is too large for MEDIAN operation";
            throw std::runtime_error(err_msg);
        }
        this->update_cols[0]->scale = new_scale;
        this->update_cols[0]->precision = new_precision;
    }
}

void MedianColSet::update(const std::vector<grouping_info>& grp_infos,
                          bodo::IBufferPool* const pool,
                          std::shared_ptr<::arrow::MemoryManager> mm) {
    median_computation(this->in_col, this->update_cols[0], grp_infos[0],
                       this->skip_na_data, use_sql_rules, pool);
}

// ############################## Mode ##############################

ModeColSet::ModeColSet(std::shared_ptr<array_info> in_col, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::mode, false, use_sql_rules) {}

ModeColSet::~ModeColSet() = default;

void ModeColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
    mode_computation(this->in_col, this->update_cols[0], grp_infos[0], pool,
                     std::move(mm));
}

// ############################## NUnique ##############################

NUniqueColSet::NUniqueColSet(std::shared_ptr<array_info> in_col,
                             bool _skip_na_data,
                             std::shared_ptr<table_info> nunique_table,
                             bool do_combine, bool _is_parallel,
                             bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::nunique, do_combine, use_sql_rules),
      skip_na_data(_skip_na_data),
      my_nunique_table(std::move(nunique_table)),
      is_parallel(_is_parallel) {}

NUniqueColSet::~NUniqueColSet() = default;

void NUniqueColSet::update(const std::vector<grouping_info>& grp_infos,
                           bodo::IBufferPool* const pool,
                           std::shared_ptr<::arrow::MemoryManager> mm) {
    // to support nunique for dictionary-encoded arrays we only need to
    // perform the nunqiue operation on the indices array(child_arrays[1]),
    // which is a int32_t numpy array.
    std::shared_ptr<array_info> input_col =
        this->in_col->arr_type == bodo_array_type::DICT
            ? this->in_col->child_arrays[1]
            : this->in_col;
    // TODO: check nunique with pivot_table operation
    if (my_nunique_table != nullptr) {
        // use the grouping_info that corresponds to my nunique
        // table
        aggfunc_output_initialize(this->update_cols[0], Bodo_FTypes::sum,
                                  use_sql_rules);  // zero initialize
        nunique_computation(std::move(input_col), this->update_cols[0],
                            grp_infos[my_nunique_table->id], this->skip_na_data,
                            is_parallel, pool);
    } else {
        // use default grouping_info
        nunique_computation(std::move(input_col), this->update_cols[0],
                            grp_infos[0], this->skip_na_data, is_parallel,
                            pool);
    }
}

// ############################## CumOp ##############################

CumOpColSet::CumOpColSet(std::shared_ptr<array_info> in_col, int ftype,
                         bool _skip_na_data, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules),
      skip_na_data(_skip_na_data) {}

CumOpColSet::~CumOpColSet() = default;

void CumOpColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // NOTE: output size of cum ops is the same as input size
    //       (NOT the number of groups)
    bodo_array_type::arr_type_enum out_type = this->in_col->arr_type;
    if (out_type == bodo_array_type::DICT) {
        // for dictionary-encoded input the arrtype of the output is
        // regular string
        out_type = bodo_array_type::STRING;
    }
    out_cols.push_back(alloc_array_top_level(
        this->in_col->length, 1, 1, out_type, this->in_col->dtype, -1, 0,
        this->in_col->num_categories, false, false, false, pool,
        std::move(mm)));
}

void CumOpColSet::update(const std::vector<grouping_info>& grp_infos,
                         bodo::IBufferPool* const pool,
                         std::shared_ptr<::arrow::MemoryManager> mm) {
    cumulative_computation(this->in_col, this->update_cols[0], grp_infos[0],
                           this->ftype, this->skip_na_data, pool,
                           std::move(mm));
}

// ############################## Shift ##############################

ShiftColSet::ShiftColSet(std::shared_ptr<array_info> in_col, int ftype,
                         int64_t _periods, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules), periods(_periods) {}
ShiftColSet::~ShiftColSet() = default;

void ShiftColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // NOTE: output size of shift is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array_top_level(
        this->in_col->length, 1, 1, this->in_col->arr_type, this->in_col->dtype,
        -1, 0, this->in_col->num_categories, false, false, false, pool,
        std::move(mm)));
}

void ShiftColSet::update(const std::vector<grouping_info>& grp_infos,
                         bodo::IBufferPool* const pool,
                         std::shared_ptr<::arrow::MemoryManager> mm) {
    shift_computation(this->in_col, this->update_cols[0], grp_infos[0],
                      this->periods, pool, std::move(mm));
}

// ############################## Transform ##############################

TransformColSet::TransformColSet(std::shared_ptr<array_info> in_col, int ftype,
                                 int _func_num, bool do_combine,
                                 bool is_parallel, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules),
      is_parallel(is_parallel),
      transform_func(_func_num) {
    transform_op_col =
        makeColSet({in_col}, nullptr, transform_func, do_combine, false, 0,
                   {transform_func}, 0, is_parallel, {false}, {false},
                   {nullptr}, 0, nullptr, nullptr, 0, nullptr, use_sql_rules);
}

TransformColSet::~TransformColSet() = default;

void TransformColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Get output column type based on transform_func and its in_col
    // datatype
    auto arr_type = this->in_col->arr_type;
    auto dtype = this->in_col->dtype;
    int64_t num_categories = this->in_col->num_categories;
    std::tie(arr_type, dtype) =
        get_groupby_output_dtype(transform_func, arr_type, dtype);
    // NOTE: output size of transform is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array_top_level(
        this->in_col->length, 1, 1, arr_type, dtype, -1, 0, num_categories,
        false, false, false, pool, std::move(mm)));
}

void TransformColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    const bool alloc_out_if_no_combine, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Allocate child column that does the actual computation
    std::vector<std::shared_ptr<array_info>> list_arr;

    transform_op_col->alloc_update_columns(num_groups, list_arr,
                                           alloc_out_if_no_combine, pool, mm);

    this->alloc_running_value_columns(num_groups, out_cols, pool,
                                      std::move(mm));
    this->update_cols.push_back(out_cols.back());
}

void TransformColSet::update(const std::vector<grouping_info>& grp_infos,
                             bodo::IBufferPool* const pool,
                             std::shared_ptr<::arrow::MemoryManager> mm) {
    transform_op_col->update(grp_infos);
    aggfunc_output_initialize(this->update_cols[0], transform_func,
                              use_sql_rules);
}

void TransformColSet::eval(const grouping_info& grp_info,
                           bodo::IBufferPool* const pool,
                           std::shared_ptr<::arrow::MemoryManager> mm) {
    // Needed to get final result for transform operation on
    // transform_op_col
    transform_op_col->eval(grp_info, pool, mm);
    // copy_values need to know type of the data it'll copy.
    // Hence we use switch case on the column dtype
    std::vector<std::shared_ptr<array_info>> out_cols;

    // getOutputColumns for this ColSet is guaranteed to return a vector of size
    // 1
    const std::shared_ptr<array_info> child_out_col =
        this->transform_op_col->getOutputColumns().at(0);

    assert(this->update_cols.size() == 1);
    copy_values_transform(this->update_cols[0], child_out_col, grp_info,
                          this->is_parallel, pool, std::move(mm));
}

// ############################## Head ##############################

HeadColSet::HeadColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules) {}

HeadColSet::~HeadColSet() = default;

void HeadColSet::alloc_running_value_columns(
    size_t update_col_len, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // NOTE: output size of head is dependent on number of rows to
    // get from each group. This is computed in
    // GroupbyPipeline::update().
    out_cols.push_back(alloc_array_top_level(
        update_col_len, 1, 1, this->in_col->arr_type, this->in_col->dtype, -1,
        0, this->in_col->num_categories, false, false, false, pool,
        std::move(mm)));
}

void HeadColSet::update(const std::vector<grouping_info>& grp_infos,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    head_computation(this->in_col, this->update_cols[0], head_row_list, pool,
                     std::move(mm));
}

void HeadColSet::set_head_row_list(bodo::vector<int64_t>& row_list) {
    head_row_list = row_list;
}

// ########################## Streaming MRNF ##########################

StreamingMRNFColSet::StreamingMRNFColSet(std::vector<bool>& _asc,
                                         std::vector<bool>& _na_pos,
                                         bool use_sql_rules)
    : BasicColSet(nullptr, Bodo_FTypes::min_row_number_filter,
                  /*combine_step*/ false, use_sql_rules),
      asc(_asc),
      na_pos(_na_pos) {
    // Decide the array type (NULLABLE or NUMPY) of the index column
    // to allocate and the ftype to use during the update step.
    std::tie(this->update_ftype, this->update_idx_arr_type) =
        get_update_ftype_idx_arr_type_for_mrnf(this->asc.size(), this->asc,
                                               this->na_pos);
}

StreamingMRNFColSet::~StreamingMRNFColSet() = default;

void StreamingMRNFColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Allocate intermediate buffer to store the index of the min element for
    // each group.
    out_cols.push_back(alloc_array_top_level(
        num_groups, 1, 1, this->update_idx_arr_type, Bodo_CTypes::UINT64, -1, 0,
        0, false, false, false, pool, std::move(mm)));
}

void StreamingMRNFColSet::update(const std::vector<grouping_info>& grp_infos,
                                 bodo::IBufferPool* const pool,
                                 std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> idx_col = this->update_cols[0];
    const grouping_info& grp_info = grp_infos[0];
    // Use the common helper (between streaming and non-streaming) to populate
    // 'idx_col' based on the order-by columns.
    min_row_number_filter_no_sort(idx_col, this->orderby_cols, grp_info,
                                  this->asc, this->na_pos, this->update_ftype,
                                  this->use_sql_rules, pool, std::move(mm));
}

// ############################## Window ##############################

WindowColSet::WindowColSet(
    std::vector<std::shared_ptr<array_info>>& in_cols,
    std::vector<int64_t> _window_funcs, std::vector<bool>& _asc,
    std::vector<bool>& _na_pos, std::shared_ptr<table_info> _window_args,
    int _n_input_cols, bool _is_parallel, bool use_sql_rules,
    std::vector<std::vector<std::unique_ptr<bodo::DataType>>> _in_arr_types_vec)
    :  // Note the inputCol in BasicColSet is not used by
       // WindowColSet
      BasicColSet(nullptr, Bodo_FTypes::window, false, use_sql_rules),
      input_cols(in_cols),
      window_funcs(std::move(_window_funcs)),
      asc(_asc),
      na_pos(_na_pos),
      window_args(std::move(_window_args)),
      n_input_cols(_n_input_cols),
      is_parallel(_is_parallel),
      in_arr_types_vec(std::move(_in_arr_types_vec)) {}

WindowColSet::~WindowColSet() = default;

void WindowColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t window_col_offset = input_cols.size() - n_input_cols;
    // Allocate one output column for each window function call
    for (int64_t window_func : window_funcs) {
        // arr_type and dtype are assigned dummy default values.
        // This is simple to ensure they are initialized but should
        // always be modified by get_groupby_output_dtype.
        bodo_array_type::arr_type_enum arr_type =
            bodo_array_type::NULLABLE_INT_BOOL;
        Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::INT64;
        // If using an ftype that requires an input array, start
        // the arr_type and dtype as that.
        if (window_func == Bodo_FTypes::conditional_true_event ||
            window_func == Bodo_FTypes::conditional_change_event ||
            window_func == Bodo_FTypes::ratio_to_report ||
            window_func == Bodo_FTypes::mean ||
            window_func == Bodo_FTypes::var ||
            window_func == Bodo_FTypes::std ||
            window_func == Bodo_FTypes::var_pop ||
            window_func == Bodo_FTypes::std_pop ||
            window_func == Bodo_FTypes::count_if ||
            window_func == Bodo_FTypes::first ||
            window_func == Bodo_FTypes::last ||
            window_func == Bodo_FTypes::any_value) {
            arr_type = input_cols[window_col_offset]->arr_type;
            dtype = input_cols[window_col_offset]->dtype;
            window_col_offset++;
        }
        // Certain functions can have numpy input arrays but
        // nullable output arrays because of window frames
        if (window_func == Bodo_FTypes::first ||
            window_func == Bodo_FTypes::last ||
            window_func == Bodo_FTypes::mean ||
            window_func == Bodo_FTypes::var ||
            window_func == Bodo_FTypes::std ||
            window_func == Bodo_FTypes::var_pop ||
            window_func == Bodo_FTypes::std_pop) {
            if (arr_type == bodo_array_type::NUMPY) {
                arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        }
        std::tie(arr_type, dtype) =
            get_groupby_output_dtype(window_func, arr_type, dtype);
        std::shared_ptr<array_info> c;
        // String & Dictionary arrays are not allocated until the end, so
        // a dummy column is created at this stage
        if (arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::DICT) {
            c = alloc_string_array(Bodo_CTypes::STRING, 0, 0, 0, 0, false,
                                   false, false, pool, mm);
        } else {
            c = alloc_array_top_level(this->input_cols[0]->length, -1, -1,
                                      arr_type, dtype, -1, 0, 0, false, false,
                                      false, pool, mm);
            aggfunc_output_initialize(c, window_func, use_sql_rules);
        }
        out_cols.push_back(c);
    }
}

void WindowColSet::setOutDictBuilders(
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builders) {
    this->out_dict_builders = out_dict_builders;
}

void WindowColSet::update(const std::vector<grouping_info>& grp_infos,
                          bodo::IBufferPool* const pool,
                          std::shared_ptr<::arrow::MemoryManager> mm) {
    window_computation(this->input_cols, window_funcs, this->update_cols,
                       this->out_dict_builders, grp_infos[0], asc, na_pos,
                       window_args, n_input_cols, is_parallel, use_sql_rules,
                       pool, mm);
}

std::vector<std::unique_ptr<bodo::DataType>> WindowColSet::getOutputTypes() {
    std::vector<std::unique_ptr<bodo::DataType>> output_types;
    for (size_t i = 0; i < window_funcs.size(); i++) {
        int64_t window_func = window_funcs[i];
        switch (window_func) {
            case Bodo_FTypes::dense_rank:
            case Bodo_FTypes::rank:
            case Bodo_FTypes::row_number:
            case Bodo_FTypes::cume_dist:
            case Bodo_FTypes::percent_rank:
            case Bodo_FTypes::ntile: {
                bodo_array_type::arr_type_enum arr_type =
                    bodo_array_type::NULLABLE_INT_BOOL;
                Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::INT64;
                std::tie(arr_type, dtype) =
                    get_groupby_output_dtype(window_func, arr_type, dtype);
                output_types.push_back(
                    std::make_unique<bodo::DataType>(arr_type, dtype));
                break;
            }
            case Bodo_FTypes::lead:
            case Bodo_FTypes::lag: {
                // TODO: cast in_arr_type to nullable if default is null
                std::unique_ptr<bodo::DataType> in_arr_type =
                    in_arr_types_vec[i][0]->copy();
                output_types.push_back(std::move(in_arr_type));
                break;
            }
            default:
                throw std::runtime_error(
                    "Window function is not supported in streaming");
        }
    }
    return output_types;
}

const std::vector<std::shared_ptr<array_info>>
WindowColSet::getOutputColumns() {
    return update_cols;
}

// ############################## Ngroup ##############################

NgroupColSet::NgroupColSet(std::shared_ptr<array_info> in_col,
                           bool _is_parallel, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::ngroup, false, use_sql_rules),
      is_parallel(_is_parallel) {}

NgroupColSet::~NgroupColSet() = default;

void NgroupColSet::alloc_running_value_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    bodo_array_type::arr_type_enum arr_type = this->in_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = this->in_col->dtype;
    int64_t num_categories = this->in_col->num_categories;
    std::tie(arr_type, dtype) =
        get_groupby_output_dtype(this->ftype, arr_type, dtype);
    // NOTE: output size of ngroup is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array_top_level(
        this->in_col->length, 1, 1, arr_type, dtype, -1, 0, num_categories,
        false, false, false, pool, std::move(mm)));
}

void NgroupColSet::update(const std::vector<grouping_info>& grp_infos,
                          bodo::IBufferPool* const pool,
                          std::shared_ptr<::arrow::MemoryManager> mm) {
    ngroup_computation(this->in_col, this->update_cols[0], grp_infos[0],
                       is_parallel);
}

std::unique_ptr<BasicColSet> makeColSet(
    std::vector<std::shared_ptr<array_info>> in_cols,
    std::shared_ptr<array_info> index_col, int ftype, bool do_combine,
    bool skip_na_data, int64_t periods, std::vector<int64_t> transform_funcs,
    int n_udf, bool is_parallel, std::vector<bool> window_ascending,
    std::vector<bool> window_na_position,
    std::shared_ptr<table_info> window_args, int n_input_cols,
    int* udf_n_redvars, std::shared_ptr<table_info> udf_table,
    int udf_table_idx, std::shared_ptr<table_info> nunique_table,
    bool use_sql_rules,
    std::vector<std::vector<std::unique_ptr<bodo::DataType>>> in_arr_types_vec,
    stream_udf_t* udf_cfunc) {
    BasicColSet* colset;

    if (ftype != Bodo_FTypes::size && ftype != Bodo_FTypes::window &&
        in_cols.size() == 0) {
        throw std::runtime_error(fmt::format(
            "Only 'size' and 'window' can have no input columns. Provided "
            "ftype: {} must have one or more input columns.",
            ftype));
    }
    if ((ftype != Bodo_FTypes::window &&
         ftype != Bodo_FTypes::min_row_number_filter &&
         ftype != Bodo_FTypes::listagg && ftype != Bodo_FTypes::array_agg &&
         ftype != Bodo_FTypes::array_agg_distinct &&
         ftype != Bodo_FTypes::percentile_cont &&
         ftype != Bodo_FTypes::percentile_disc &&
         ftype != Bodo_FTypes::object_agg) &&
        in_cols.size() > 1) {
        throw std::runtime_error(
            "Only listagg, array_agg, percentile_cont, percentile_disc, "
            "object_agg, window functions and min_row_number_filter can have "
            "multiple input columns.");
    }
    switch (ftype) {
        case Bodo_FTypes::udf:
            colset = new UdfColSet(in_cols[0], do_combine, std::move(udf_table),
                                   udf_table_idx, udf_n_redvars[n_udf],
                                   use_sql_rules);
            break;
        case Bodo_FTypes::gen_udf:
            colset = new GeneralUdfColSet(in_cols[0], std::move(udf_table),
                                          udf_table_idx, use_sql_rules);
            break;
        case Bodo_FTypes::stream_udf:
            colset =
                new StreamingUDFColSet(in_cols[0], std::move(udf_table),
                                       udf_table_idx, udf_cfunc, use_sql_rules);
            break;
        case Bodo_FTypes::percentile_disc:
        case Bodo_FTypes::percentile_cont:
            colset = new PercentileColSet(in_cols[0], in_cols[1],
                                          ftype == Bodo_FTypes::percentile_cont,
                                          use_sql_rules);
            break;
        case Bodo_FTypes::median:
            colset = new MedianColSet(in_cols[0], skip_na_data, use_sql_rules);
            break;
        case Bodo_FTypes::mode:
            colset = new ModeColSet(in_cols[0], use_sql_rules);
            break;
        case Bodo_FTypes::nunique:
            colset = new NUniqueColSet(in_cols[0], skip_na_data,
                                       std::move(nunique_table), do_combine,
                                       is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::cummin:
        case Bodo_FTypes::cummax:
        case Bodo_FTypes::cumprod:
            colset =
                new CumOpColSet(in_cols[0], ftype, skip_na_data, use_sql_rules);
            break;
        case Bodo_FTypes::mean:
            colset = new MeanColSet(in_cols[0], do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::boolxor_agg:
            colset =
                new BoolXorColSet(in_cols[0], ftype, do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::var_pop:
        case Bodo_FTypes::std_pop:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            colset =
                new VarStdColSet(in_cols[0], ftype, do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::skew:
            colset =
                new SkewColSet(in_cols[0], ftype, do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::kurtosis:
            colset =
                new KurtColSet(in_cols[0], ftype, do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::idxmin:
        case Bodo_FTypes::idxmax:
            colset = new IdxMinMaxColSet(in_cols[0], index_col, ftype,
                                         do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::shift:
            colset = new ShiftColSet(in_cols[0], ftype, periods, use_sql_rules);
            break;
        case Bodo_FTypes::transform:
            colset =
                new TransformColSet(in_cols[0], ftype, transform_funcs[0],
                                    do_combine, is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::head:
            colset = new HeadColSet(in_cols[0], ftype, use_sql_rules);
            break;
        case Bodo_FTypes::ngroup:
            colset = new NgroupColSet(in_cols[0], is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::min_row_number_filter:
            colset = new StreamingMRNFColSet(window_ascending,
                                             window_na_position, use_sql_rules);
            break;
        case Bodo_FTypes::window:
            colset = new WindowColSet(
                in_cols, transform_funcs, window_ascending, window_na_position,
                window_args, n_input_cols, is_parallel, use_sql_rules,
                std::move(in_arr_types_vec));
            break;
        case Bodo_FTypes::listagg:

            colset = new ListAggColSet(
                // data column
                in_cols[0],
                // separator column
                in_cols[1],
                // Remaining columns are the columns to order by
                std::vector<std::shared_ptr<array_info>>(in_cols.begin() + 2,
                                                         in_cols.end()),
                window_ascending, window_na_position);
            break;
        case Bodo_FTypes::array_agg:
        case Bodo_FTypes::array_agg_distinct:
            colset = new ArrayAggColSet(
                // data column
                in_cols[0],
                // Remaining columns are the columns to order by
                std::vector<std::shared_ptr<array_info>>(in_cols.begin() + 1,
                                                         in_cols.end()),
                window_ascending, window_na_position, ftype, is_parallel);
            break;
        case Bodo_FTypes::object_agg:
            colset = new ObjectAggColSet(
                // key column
                in_cols[0],
                // value column
                in_cols[1], is_parallel);
            break;
        case Bodo_FTypes::first:
            colset = new FirstColSet(in_cols[0], do_combine, use_sql_rules);
            break;
        case Bodo_FTypes::size:
            colset = new SizeColSet(do_combine, use_sql_rules);
            break;
        default:
            colset =
                new BasicColSet(in_cols[0], ftype, do_combine, use_sql_rules);
    }
    return std::unique_ptr<BasicColSet>(colset);
}
