// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_col_set.h"
#include "_array_utils.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"
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

BasicColSet::BasicColSet(std::shared_ptr<array_info> in_col, int ftype,
                         bool combine_step, bool use_sql_rules)
    : in_col(in_col),
      ftype(ftype),
      combine_step(combine_step),
      use_sql_rules(use_sql_rules) {}

BasicColSet::~BasicColSet() {}

void BasicColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
    int64_t num_categories = in_col->num_categories;
    // calling this modifies arr_type and dtype
    get_groupby_output_dtype(ftype, arr_type, dtype);
    out_cols.push_back(
        alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
    update_cols.push_back(out_cols.back());
}

void BasicColSet::update(const std::vector<grouping_info>& grp_infos) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
    do_apply_to_column(in_col, update_cols[0], aux_cols, grp_infos[0], ftype);
}

typename std::vector<std::shared_ptr<array_info>>::iterator
BasicColSet::update_after_shuffle(
    typename std::vector<std::shared_ptr<array_info>>::iterator& it) {
    for (size_t i_col = 0; i_col < update_cols.size(); i_col++) {
        update_cols[i_col] = *(it++);
    }
    return it;
}

void BasicColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    int combine_ftype = get_combine_func(ftype);
    for (auto col : update_cols) {
        bodo_array_type::arr_type_enum arr_type = col->arr_type;
        // Combine may remap the dtype.
        Bodo_CTypes::CTypeEnum dtype = col->dtype;
        int64_t num_categories = col->num_categories;
        // calling this modifies arr_type and dtype
        get_groupby_output_dtype(combine_ftype, arr_type, dtype);
        out_cols.push_back(
            alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
        combine_cols.push_back(out_cols.back());
    }
}

void BasicColSet::combine(const grouping_info& grp_info) {
    int combine_ftype = get_combine_func(ftype);
    std::vector<std::shared_ptr<array_info>> aux_cols(combine_cols.begin() + 1,
                                                      combine_cols.end());
    for (auto col : combine_cols) {
        aggfunc_output_initialize(col, combine_ftype, use_sql_rules);
    }
    do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                       combine_ftype);
}

void BasicColSet::eval(const grouping_info& grp_info) {}

std::shared_ptr<array_info> BasicColSet::getOutputColumn() {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (combine_step) {
        mycols = &combine_cols;
    } else {
        mycols = &update_cols;
    }
    std::shared_ptr<array_info> out_col = mycols->at(0);
    return out_col;
}

MeanColSet::MeanColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                       bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::mean, combine_step, use_sql_rules) {}

MeanColSet::~MeanColSet() {}

void MeanColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> c1 =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0,
                    0);  // for sum and result
    std::shared_ptr<array_info> c2 =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::UINT64, 0,
                    0);  // for counts
    out_cols.push_back(c1);
    out_cols.push_back(c2);
    this->update_cols.push_back(c1);
    this->update_cols.push_back(c2);
}

void MeanColSet::update(const std::vector<grouping_info>& grp_infos) {
    std::vector<std::shared_ptr<array_info>> aux_cols = {this->update_cols[1]};
    aggfunc_output_initialize(this->update_cols[0], this->ftype, use_sql_rules);
    aggfunc_output_initialize(this->update_cols[1], this->ftype, use_sql_rules);
    do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                       grp_infos[0], this->ftype);
}

void MeanColSet::combine(const grouping_info& grp_info) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    aggfunc_output_initialize(this->combine_cols[0], this->ftype,
                              use_sql_rules);
    // Initialize the output as mean to match the nullable behavior.
    aggfunc_output_initialize(this->combine_cols[1], this->ftype,
                              use_sql_rules);
    do_apply_to_column(this->update_cols[0], this->combine_cols[0], aux_cols,
                       grp_info, Bodo_FTypes::sum);
    do_apply_to_column(this->update_cols[1], this->combine_cols[1], aux_cols,
                       grp_info, Bodo_FTypes::sum);
}

void MeanColSet::eval(const grouping_info& grp_info) {
    std::vector<std::shared_ptr<array_info>> aux_cols;
    if (this->combine_step) {
        do_apply_to_column(this->combine_cols[1], this->combine_cols[0],
                           aux_cols, grp_info, Bodo_FTypes::mean_eval);
    } else {
        do_apply_to_column(this->update_cols[1], this->update_cols[0], aux_cols,
                           grp_info, Bodo_FTypes::mean_eval);
    }
}

IdxMinMaxColSet::IdxMinMaxColSet(std::shared_ptr<array_info> in_col,
                                 std::shared_ptr<array_info> _index_col,
                                 int ftype, bool combine_step,
                                 bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules),
      index_col(_index_col) {}

IdxMinMaxColSet::~IdxMinMaxColSet() {}

void IdxMinMaxColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // output column containing index values. dummy for now. will be
    // assigned the real data at the end of update()
    std::shared_ptr<array_info> out_col = alloc_array(
        num_groups, 1, 1, index_col->arr_type, index_col->dtype, 0, 0);
    // create array to store min/max value
    std::shared_ptr<array_info> max_col = alloc_array(
        num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0,
        0);  // for min/max
    // create array to store index position of min/max value
    std::shared_ptr<array_info> index_pos_col = alloc_array(
        num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, 0, 0);
    out_cols.push_back(out_col);
    out_cols.push_back(max_col);
    this->update_cols.push_back(out_col);
    this->update_cols.push_back(max_col);
    this->update_cols.push_back(index_pos_col);
}

void IdxMinMaxColSet::update(const std::vector<grouping_info>& grp_infos) {
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
                       grp_infos[0], this->ftype);

    std::shared_ptr<array_info> real_out_col =
        RetrieveArray_SingleColumn_arr(index_col, index_pos_col);
    std::shared_ptr<array_info> out_col = this->update_cols[0];
    *out_col = std::move(*real_out_col);
    this->update_cols.pop_back();
}

void IdxMinMaxColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // output column containing index values. dummy for now. will be
    // assigned the real data at the end of combine()
    std::shared_ptr<array_info> out_col = alloc_array(
        num_groups, 1, 1, index_col->arr_type, index_col->dtype, 0, 0);
    // create array to store min/max value
    std::shared_ptr<array_info> max_col = alloc_array(
        num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0,
        0);  // for min/max
    // create array to store index position of min/max value
    std::shared_ptr<array_info> index_pos_col = alloc_array(
        num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, 0, 0);
    out_cols.push_back(out_col);
    out_cols.push_back(max_col);
    this->combine_cols.push_back(out_col);
    this->combine_cols.push_back(max_col);
    this->combine_cols.push_back(index_pos_col);
}

void IdxMinMaxColSet::combine(const grouping_info& grp_info) {
    std::shared_ptr<array_info> index_pos_col = this->combine_cols[2];
    std::vector<std::shared_ptr<array_info>> aux_cols = {index_pos_col};
    if (this->ftype == Bodo_FTypes::idxmax) {
        aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::max,
                                  use_sql_rules);
    } else {
        // Bodo_FTypes::idxmin
        aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::min,
                                  use_sql_rules);
    }
    aggfunc_output_initialize(index_pos_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero init
    do_apply_to_column(this->update_cols[1], this->combine_cols[1], aux_cols,
                       grp_info, this->ftype);

    std::shared_ptr<array_info> real_out_col =
        RetrieveArray_SingleColumn_arr(this->update_cols[0], index_pos_col);
    std::shared_ptr<array_info> out_col = this->combine_cols[0];
    *out_col = std::move(*real_out_col);
    this->combine_cols.pop_back();
}

VarStdColSet::VarStdColSet(std::shared_ptr<array_info> in_col, int ftype,
                           bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

VarStdColSet::~VarStdColSet() {}

void VarStdColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    if (!this->combine_step) {
        // need to create output column now
        std::shared_ptr<array_info> col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0, 0);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->update_cols.push_back(col);
    }
    std::shared_ptr<array_info> count_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::UINT64, 0, 0);
    std::shared_ptr<array_info> mean_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m2_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    aggfunc_output_initialize(count_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(mean_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    aggfunc_output_initialize(m2_col, Bodo_FTypes::count,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(count_col);
    out_cols.push_back(mean_col);
    out_cols.push_back(m2_col);
    this->update_cols.push_back(count_col);
    this->update_cols.push_back(mean_col);
    this->update_cols.push_back(m2_col);
}

void VarStdColSet::update(const std::vector<grouping_info>& grp_infos) {
    if (!this->combine_step) {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[1], this->update_cols[2], this->update_cols[3]};
        do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                           grp_infos[0], this->ftype);
    } else {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[0], this->update_cols[1], this->update_cols[2]};
        do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                           grp_infos[0], this->ftype);
    }
}

void VarStdColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0,
                    0);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(col);
    this->combine_cols.push_back(col);
    BasicColSet::alloc_combine_columns(num_groups, out_cols);
}

void VarStdColSet::combine(const grouping_info& grp_info) {
    std::shared_ptr<array_info> count_col_in = this->update_cols[0];
    std::shared_ptr<array_info> mean_col_in = this->update_cols[1];
    std::shared_ptr<array_info> m2_col_in = this->update_cols[2];
    std::shared_ptr<array_info> count_col_out = this->combine_cols[1];
    std::shared_ptr<array_info> mean_col_out = this->combine_cols[2];
    std::shared_ptr<array_info> m2_col_out = this->combine_cols[3];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules);
    var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                mean_col_out, m2_col_out, grp_info);
}

void VarStdColSet::eval(const grouping_info& grp_info) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(1), mycols->at(2), mycols->at(3)};
    switch (this->ftype) {
        case Bodo_FTypes::var_pop: {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::var_pop_eval);
            break;
        }
        case Bodo_FTypes::std_pop: {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::std_pop_eval);
            break;
        }
        case Bodo_FTypes::var: {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::var_eval);
            break;
        }
        case Bodo_FTypes::std: {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::std_eval);
            break;
        }
    }
}

SkewColSet::SkewColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

SkewColSet::~SkewColSet() {}

void SkewColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    if (!this->combine_step) {
        // need to create output column now
        std::shared_ptr<array_info> col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0, 0);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->update_cols.push_back(col);
    }
    std::shared_ptr<array_info> count_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::UINT64, 0, 0);
    std::shared_ptr<array_info> m1_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m2_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m3_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
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
    this->update_cols.push_back(count_col);
    this->update_cols.push_back(m1_col);
    this->update_cols.push_back(m2_col);
    this->update_cols.push_back(m3_col);
}

void SkewColSet::update(const std::vector<grouping_info>& grp_infos) {
    if (!this->combine_step) {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[1], this->update_cols[2], this->update_cols[3],
            this->update_cols[4]};
        do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                           grp_infos[0], this->ftype);
    } else {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[0], this->update_cols[1], this->update_cols[2],
            this->update_cols[3]};
        do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                           grp_infos[0], this->ftype);
    }
}

void SkewColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0,
                    0);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(col);
    this->combine_cols.push_back(col);
    BasicColSet::alloc_combine_columns(num_groups, out_cols);
}

void SkewColSet::combine(const grouping_info& grp_info) {
    std::shared_ptr<array_info> count_col_in = this->update_cols[0];
    std::shared_ptr<array_info> m1_col_in = this->update_cols[1];
    std::shared_ptr<array_info> m2_col_in = this->update_cols[2];
    std::shared_ptr<array_info> m3_col_in = this->update_cols[3];
    std::shared_ptr<array_info> count_col_out = this->combine_cols[1];
    std::shared_ptr<array_info> m1_col_out = this->combine_cols[2];
    std::shared_ptr<array_info> m2_col_out = this->combine_cols[3];
    std::shared_ptr<array_info> m3_col_out = this->combine_cols[4];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m1_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m3_col_out, Bodo_FTypes::count, use_sql_rules);
    skew_combine(count_col_in, m1_col_in, m2_col_in, m3_col_in, count_col_out,
                 m1_col_out, m2_col_out, m3_col_out, grp_info);
}

void SkewColSet::eval(const grouping_info& grp_info) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(1), mycols->at(2), mycols->at(3), mycols->at(4)};
    do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                       Bodo_FTypes::skew_eval);
}

KurtColSet::KurtColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool combine_step, bool use_sql_rules)
    : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}

KurtColSet::~KurtColSet() {}

void KurtColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    if (!this->combine_step) {
        // need to create output column now
        std::shared_ptr<array_info> col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0, 0);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->update_cols.push_back(col);
    }
    std::shared_ptr<array_info> count_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::UINT64, 0, 0);
    std::shared_ptr<array_info> m1_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m2_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m3_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
    std::shared_ptr<array_info> m4_col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0, 0);
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
    this->update_cols.push_back(count_col);
    this->update_cols.push_back(m1_col);
    this->update_cols.push_back(m2_col);
    this->update_cols.push_back(m3_col);
    this->update_cols.push_back(m4_col);
}

void KurtColSet::update(const std::vector<grouping_info>& grp_infos) {
    if (!this->combine_step) {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[1], this->update_cols[2], this->update_cols[3],
            this->update_cols[4], this->update_cols[5]};
        do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                           grp_infos[0], this->ftype);
    } else {
        std::vector<std::shared_ptr<array_info>> aux_cols = {
            this->update_cols[0], this->update_cols[1], this->update_cols[2],
            this->update_cols[3], this->update_cols[4]};
        do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                           grp_infos[0], this->ftype);
    }
}

void KurtColSet::eval(const grouping_info& grp_info) {
    std::vector<std::shared_ptr<array_info>>* mycols;
    if (this->combine_step) {
        mycols = &this->combine_cols;
    } else {
        mycols = &this->update_cols;
    }

    std::vector<std::shared_ptr<array_info>> aux_cols = {
        mycols->at(1), mycols->at(2), mycols->at(3), mycols->at(4),
        mycols->at(5)};
    do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                       Bodo_FTypes::kurt_eval);
}

void KurtColSet::alloc_combine_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    std::shared_ptr<array_info> col =
        alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::FLOAT64, 0,
                    0);  // for result
    // Initialize as ftype to match nullable behavior
    aggfunc_output_initialize(col, this->ftype,
                              use_sql_rules);  // zero initialize
    out_cols.push_back(col);
    this->combine_cols.push_back(col);
    BasicColSet::alloc_combine_columns(num_groups, out_cols);
}

void KurtColSet::combine(const grouping_info& grp_info) {
    std::shared_ptr<array_info> count_col_in = this->update_cols[0];
    std::shared_ptr<array_info> m1_col_in = this->update_cols[1];
    std::shared_ptr<array_info> m2_col_in = this->update_cols[2];
    std::shared_ptr<array_info> m3_col_in = this->update_cols[3];
    std::shared_ptr<array_info> m4_col_in = this->update_cols[4];
    std::shared_ptr<array_info> count_col_out = this->combine_cols[1];
    std::shared_ptr<array_info> m1_col_out = this->combine_cols[2];
    std::shared_ptr<array_info> m2_col_out = this->combine_cols[3];
    std::shared_ptr<array_info> m3_col_out = this->combine_cols[4];
    std::shared_ptr<array_info> m4_col_out = this->combine_cols[5];
    aggfunc_output_initialize(count_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m1_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m3_col_out, Bodo_FTypes::count, use_sql_rules);
    aggfunc_output_initialize(m4_col_out, Bodo_FTypes::count, use_sql_rules);
    kurt_combine(count_col_in, m1_col_in, m2_col_in, m3_col_in, m4_col_in,
                 count_col_out, m1_col_out, m2_col_out, m3_col_out, m4_col_out,
                 grp_info);
}

UdfColSet::UdfColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                     std::shared_ptr<table_info> udf_table, int udf_table_idx,
                     int n_redvars, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::udf, combine_step, use_sql_rules),
      udf_table(udf_table),
      udf_table_idx(udf_table_idx),
      n_redvars(n_redvars) {}

UdfColSet::~UdfColSet() {}

void UdfColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    int offset = 0;
    if (this->combine_step)
        offset = 1;
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
        out_cols.push_back(
            alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
        if (!this->combine_step) {
            this->update_cols.push_back(out_cols.back());
        }
    }
}

void UdfColSet::update(const std::vector<grouping_info>& grp_infos) {
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
        out_cols.push_back(
            alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
        this->combine_cols.push_back(out_cols.back());
    }
}

void UdfColSet::combine(const grouping_info& grp_info) {
    // do nothing because this is done in JIT-compiled code (invoked
    // from GroupbyPipeline once for all udf columns sets)
}

void UdfColSet::eval(const grouping_info& grp_info) {
    // do nothing because this is done in JIT-compiled code (invoked
    // from GroupbyPipeline once for all udf columns sets)
}

GeneralUdfColSet::GeneralUdfColSet(std::shared_ptr<array_info> in_col,
                                   std::shared_ptr<table_info> udf_table,
                                   int udf_table_idx, bool use_sql_rules)
    : UdfColSet(in_col, false, udf_table, udf_table_idx, 0, use_sql_rules) {}

GeneralUdfColSet::~GeneralUdfColSet() {}

void GeneralUdfColSet::fill_in_columns(
    std::shared_ptr<table_info> general_in_table,
    const grouping_info& grp_info) const {
    std::shared_ptr<array_info> in_col = this->in_col;
    std::vector<std::vector<int64_t>> group_rows(grp_info.num_groups);
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

MedianColSet::MedianColSet(std::shared_ptr<array_info> in_col, bool _skipna,
                           bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::median, false, use_sql_rules),
      skipna(_skipna) {}

MedianColSet::~MedianColSet() {}

void MedianColSet::update(const std::vector<grouping_info>& grp_infos) {
    median_computation(this->in_col, this->update_cols[0], grp_infos[0],
                       this->skipna, use_sql_rules);
}

NUniqueColSet::NUniqueColSet(std::shared_ptr<array_info> in_col, bool _dropna,
                             std::shared_ptr<table_info> nunique_table,
                             bool do_combine, bool _is_parallel,
                             bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::nunique, do_combine, use_sql_rules),
      dropna(_dropna),
      my_nunique_table(nunique_table),
      is_parallel(_is_parallel) {}

NUniqueColSet::~NUniqueColSet() {}

void NUniqueColSet::update(const std::vector<grouping_info>& grp_infos) {
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
        nunique_computation(input_col, this->update_cols[0],
                            grp_infos[my_nunique_table->id], dropna,
                            is_parallel);
    } else {
        // use default grouping_info
        nunique_computation(input_col, this->update_cols[0], grp_infos[0],
                            dropna, is_parallel);
    }
}

CumOpColSet::CumOpColSet(std::shared_ptr<array_info> in_col, int ftype,
                         bool _skipna, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules), skipna(_skipna) {}

CumOpColSet::~CumOpColSet() {}

void CumOpColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // NOTE: output size of cum ops is the same as input size
    //       (NOT the number of groups)
    bodo_array_type::arr_type_enum out_type = this->in_col->arr_type;
    if (out_type == bodo_array_type::DICT) {
        // for dictionary-encoded input the arrtype of the output is
        // regular string
        out_type = bodo_array_type::STRING;
    }
    out_cols.push_back(alloc_array(this->in_col->length, 1, 1, out_type,
                                   this->in_col->dtype, 0,
                                   this->in_col->num_categories));
    this->update_cols.push_back(out_cols.back());
}

void CumOpColSet::update(const std::vector<grouping_info>& grp_infos) {
    cumulative_computation(this->in_col, this->update_cols[0], grp_infos[0],
                           this->ftype, this->skipna);
}

ShiftColSet::ShiftColSet(std::shared_ptr<array_info> in_col, int ftype,
                         int64_t _periods, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules), periods(_periods) {}
ShiftColSet::~ShiftColSet() {}

void ShiftColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // NOTE: output size of shift is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array(this->in_col->length, 1, 1,
                                   this->in_col->arr_type, this->in_col->dtype,
                                   0, this->in_col->num_categories));
    this->update_cols.push_back(out_cols.back());
}

void ShiftColSet::update(const std::vector<grouping_info>& grp_infos) {
    shift_computation(this->in_col, this->update_cols[0], grp_infos[0],
                      this->periods);
}

TransformColSet::TransformColSet(std::shared_ptr<array_info> in_col, int ftype,
                                 int _func_num, bool do_combine,
                                 bool is_parallel, bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules),
      is_parallel(is_parallel),
      transform_func(_func_num) {
    transform_op_col =
        makeColSet({in_col}, nullptr, transform_func, do_combine, false, 0,
                   transform_func, 0, is_parallel, {false}, {false}, nullptr,
                   nullptr, 0, nullptr, use_sql_rules);
}

TransformColSet::~TransformColSet() {}

void TransformColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // Allocate child column that does the actual computation
    std::vector<std::shared_ptr<array_info>> list_arr;
    transform_op_col->alloc_update_columns(num_groups, list_arr);

    // Get output column type based on transform_func and its in_col
    // datatype
    auto arr_type = this->in_col->arr_type;
    auto dtype = this->in_col->dtype;
    int64_t num_categories = this->in_col->num_categories;
    get_groupby_output_dtype(transform_func, arr_type, dtype);
    // NOTE: output size of transform is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array(this->in_col->length, 1, 1, arr_type, dtype,
                                   0, num_categories));
    this->update_cols.push_back(out_cols.back());
}

void TransformColSet::update(const std::vector<grouping_info>& grp_infos) {
    transform_op_col->update(grp_infos);
    aggfunc_output_initialize(this->update_cols[0], transform_func,
                              use_sql_rules);
}

void TransformColSet::eval(const grouping_info& grp_info) {
    // Needed to get final result for transform operation on
    // transform_op_col
    transform_op_col->eval(grp_info);
    // copy_values need to know type of the data it'll copy.
    // Hence we use switch case on the column dtype
    std::shared_ptr<array_info> child_out_col =
        this->transform_op_col->getOutputColumn();
    copy_values_transform(this->update_cols[0], child_out_col, grp_info,
                          this->is_parallel);
}

HeadColSet::HeadColSet(std::shared_ptr<array_info> in_col, int ftype,
                       bool use_sql_rules)
    : BasicColSet(in_col, ftype, false, use_sql_rules) {}

HeadColSet::~HeadColSet() {}

void HeadColSet::alloc_update_columns(
    size_t update_col_len, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // NOTE: output size of head is dependent on number of rows to
    // get from each group. This is computed in
    // GroupbyPipeline::update().
    out_cols.push_back(alloc_array(update_col_len, 1, 1, this->in_col->arr_type,
                                   this->in_col->dtype, 0,
                                   this->in_col->num_categories));
    this->update_cols.push_back(out_cols.back());
}

void HeadColSet::update(const std::vector<grouping_info>& grp_infos) {
    head_computation(this->in_col, this->update_cols[0], head_row_list);
}
void HeadColSet::set_head_row_list(std::vector<int64_t> row_list) {
    head_row_list = row_list;
}

WindowColSet::WindowColSet(std::vector<std::shared_ptr<array_info>>& in_cols,
                           int64_t _window_func, std::vector<bool>& _asc,
                           std::vector<bool>& _na_pos, bool _is_parallel,
                           bool use_sql_rules)
    :  // Note the inputCol in BasicColSet is not used by
       // WindowColSet
      BasicColSet(nullptr, Bodo_FTypes::window, false, use_sql_rules),
      input_cols(in_cols),
      window_func(_window_func),
      asc(_asc),
      na_pos(_na_pos),
      is_parallel(_is_parallel) {}

WindowColSet::~WindowColSet() {}

void WindowColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    // arr_type and dtype are assigned dummy default values.
    // This is simple to ensure they are initialized but should
    // always be modified by get_groupby_output_dtype.
    bodo_array_type::arr_type_enum arr_type =
        bodo_array_type::NULLABLE_INT_BOOL;
    Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::INT64;
    // calling this modifies arr_type and dtype
    // Output dtype is based on the window function.
    get_groupby_output_dtype(this->window_func, arr_type, dtype);
    // NOTE: output size of ngroup is the same as input size
    //       (NOT the number of groups)
    std::shared_ptr<array_info> c = alloc_array(this->input_cols[0]->length, -1,
                                                -1, arr_type, dtype, 0, -1);
    aggfunc_output_initialize(c, window_func, use_sql_rules);
    out_cols.push_back(c);
    this->update_cols.push_back(c);
}

void WindowColSet::update(const std::vector<grouping_info>& grp_infos) {
    window_computation(this->input_cols, window_func, this->update_cols[0],
                       grp_infos[0], asc, na_pos, is_parallel, use_sql_rules);
}

NgroupColSet::NgroupColSet(std::shared_ptr<array_info> in_col,
                           bool _is_parallel, bool use_sql_rules)
    : BasicColSet(in_col, Bodo_FTypes::ngroup, false, use_sql_rules),
      is_parallel(_is_parallel) {}

NgroupColSet::~NgroupColSet() {}

void NgroupColSet::alloc_update_columns(
    size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
    bodo_array_type::arr_type_enum arr_type = this->in_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = this->in_col->dtype;
    int64_t num_categories = this->in_col->num_categories;
    // calling this modifies arr_type and dtype
    get_groupby_output_dtype(this->ftype, arr_type, dtype);
    // NOTE: output size of ngroup is the same as input size
    //       (NOT the number of groups)
    out_cols.push_back(alloc_array(this->in_col->length, 1, 1, arr_type, dtype,
                                   0, num_categories));
    this->update_cols.push_back(out_cols.back());
}

void NgroupColSet::update(const std::vector<grouping_info>& grp_infos) {
    ngroup_computation(this->in_col, this->update_cols[0], grp_infos[0],
                       is_parallel);
}

std::unique_ptr<BasicColSet> makeColSet(
    std::vector<std::shared_ptr<array_info>> in_cols,
    std::shared_ptr<array_info> index_col, int ftype, bool do_combine,
    bool skipna, int64_t periods, int64_t transform_func, int n_udf,
    bool is_parallel, std::vector<bool> window_ascending,
    std::vector<bool> window_na_position, int* udf_n_redvars,
    std::shared_ptr<table_info> udf_table, int udf_table_idx,
    std::shared_ptr<table_info> nunique_table, bool use_sql_rules) {
    BasicColSet* colset;
    if (ftype != Bodo_FTypes::window && in_cols.size() != 1) {
        throw std::runtime_error(
            "Only window functions can have multiple input "
            "columns");
    }
    switch (ftype) {
        case Bodo_FTypes::udf:
            colset =
                new UdfColSet(in_cols[0], do_combine, udf_table, udf_table_idx,
                              udf_n_redvars[n_udf], use_sql_rules);
            break;
        case Bodo_FTypes::gen_udf:
            colset = new GeneralUdfColSet(in_cols[0], udf_table, udf_table_idx,
                                          use_sql_rules);
            break;
        case Bodo_FTypes::median:
            colset = new MedianColSet(in_cols[0], skipna, use_sql_rules);
            break;
        case Bodo_FTypes::nunique:
            colset = new NUniqueColSet(in_cols[0], skipna, nunique_table,
                                       do_combine, is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::cummin:
        case Bodo_FTypes::cummax:
        case Bodo_FTypes::cumprod:
            colset = new CumOpColSet(in_cols[0], ftype, skipna, use_sql_rules);
            break;
        case Bodo_FTypes::mean:
            colset = new MeanColSet(in_cols[0], do_combine, use_sql_rules);
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
                new TransformColSet(in_cols[0], ftype, transform_func,
                                    do_combine, is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::head:
            colset = new HeadColSet(in_cols[0], ftype, use_sql_rules);
            break;
        case Bodo_FTypes::ngroup:
            colset = new NgroupColSet(in_cols[0], is_parallel, use_sql_rules);
            break;
        case Bodo_FTypes::window:
            colset = new WindowColSet(in_cols, transform_func, window_ascending,
                                      window_na_position, is_parallel,
                                      use_sql_rules);
            break;
        default:
            colset =
                new BasicColSet(in_cols[0], ftype, do_combine, use_sql_rules);
    }
    return std::unique_ptr<BasicColSet>(colset);
}
