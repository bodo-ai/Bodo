// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_COL_SET_H_INCLUDED
#define _GROUPBY_COL_SET_H_INCLUDED

#include "_bodo_common.h"
#include "_groupby.h"
#include "_groupby_common.h"
#include "_groupby_update.h"

/**
 * This file declares the functions used to create "col sets"
 * in the groupby infrastructure. A col set is the explicit
 * implementation of the groupby compute steps for various
 * groups of functions. For example, simple aggregations (e.g.
 * sum, min, max) will share a col set and then transformation
 * operations will share a col set. The operations defined in
 * the groupby col set will largely mirror the general infrastructure.
 *
 */

/*
 * This is the base column set class which is used by most operations (like
 * sum, prod, count, etc.). Several subclasses also rely on some of the methods
 * of this base class.
 */
class BasicColSet {
   public:
    /**
     * Construct column set corresponding to function of type ftype applied to
     * the input column in_col
     * @param in_col input column of groupby associated with this column set
     * @param ftype function associated with this column set
     * @param combine_step tells the column set whether GroupbyPipeline is going
     * to perform a combine operation or not. If false, this means that either
     *        shuffling is not necessary or that it will be done at the
     *        beginning of the pipeline.
     * @param use_sql_rules tells the column set whether to use SQL or Pandas
     * rules
     */
    BasicColSet(std::shared_ptr<array_info> in_col, int ftype,
                bool combine_step, bool use_sql_rules);

    virtual ~BasicColSet();

    /**
     * Allocate my columns for update step.
     * @param number of groups found in the input table
     * @param[in,out] vector of columns of update table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos);

    /**
     * When GroupbyPipeline shuffles the table after update, the column set
     * needs to be updated with the columns from the new shuffled table. This
     * method is called by GroupbyPipeline with an iterator pointing to my
     * first column. The column set will update its columns and return an
     * iterator pointing to the next set of columns.
     * @param iterator pointing to the first column in this column set
     */
    virtual typename std::vector<std::shared_ptr<array_info>>::iterator
    update_after_shuffle(
        typename std::vector<std::shared_ptr<array_info>>::iterator& it);

    /**
     * Allocate my columns for combine step.
     * @param number of groups found in the input table (which is the update
     * table)
     * @param[in,out] vector of columns of combine table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    /**
     * Perform combine step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     * @param init_start_row index of first row for initializing output (used in
     * streaming groupby)
     */
    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    /**
     * Perform eval step for this column set. This will fill the output column
     * with the final result of the aggregation operation corresponding to this
     * column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void eval(const grouping_info& grp_info);

    /**
     * Obtain the final output columns resulting from the groupby operation on
     * this column set by appending them into a provided vector. This will free
     * all other intermediate or auxiliary
     * columns (if any) used by the column set (like reduction variables).
     */
    virtual void addOutputColumns(
        std::vector<std::shared_ptr<array_info>>& out_cols);

    /**
     * @brief Set input columns of this ColSet (used in streaming groupby for
     * processing new batches)
     *
     * @param new_in_cols new input columns
     */
    void setInCol(std::vector<std::shared_ptr<array_info>> new_in_cols) {
        // TODO[BSE-578]: implement setInCol() for other colsets that can have
        // more input columns
        in_col = new_in_cols[0];
    }

    /**
     * @brief Get update column types for this ColSet function with the given
     * input array types.
     * Should match arrays allocated with alloc_update_columns().
     *
     * @param in_arr_types input array types
     * @param in_dtypes input array dtypes
     * @return std::tuple<std::vector<bodo_array_type::arr_type_enum>,
     * std::vector<Bodo_CTypes::CTypeEnum>> update column array types and dtypes
     */
    virtual std::tuple<std::vector<bodo_array_type::arr_type_enum>,
                       std::vector<Bodo_CTypes::CTypeEnum>>
    getUpdateColumnTypes(
        const std::vector<bodo_array_type::arr_type_enum>& in_arr_types,
        const std::vector<Bodo_CTypes::CTypeEnum>& in_dtypes) {
        // TODO[BSE-578]: implement getUpdateColumnTypes() for other colsets
        // that can have more update columns
        std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
            out_arr_type =
                get_groupby_output_dtype(ftype, in_arr_types[0], in_dtypes[0]);
        return std::tuple(
            std::vector<bodo_array_type::arr_type_enum>{
                std::get<0>(out_arr_type)},
            std::vector<Bodo_CTypes::CTypeEnum>{std::get<1>(out_arr_type)});
    }

    /**
     * @brief Get combine column types for this ColSet function with the given
     * update array types.
     * Should match arrays allocated with alloc_combine_columns().
     *
     * @param update_arr_types update array types
     * @param update_dtypes update dtypes
     * @return std::tuple<std::vector<bodo_array_type::arr_type_enum>,
     * std::vector<Bodo_CTypes::CTypeEnum>> combine column array types and
     * dtypes
     */
    std::tuple<std::vector<bodo_array_type::arr_type_enum>,
               std::vector<Bodo_CTypes::CTypeEnum>>
    getCombineColumnTypes(
        const std::vector<bodo_array_type::arr_type_enum>& update_arr_types,
        const std::vector<Bodo_CTypes::CTypeEnum>& update_dtypes) {
        // TODO[BSE-578]: implement getCombineColumnTypes() for other colsets
        // that can have more update columns
        int combine_ftype = get_combine_func(ftype);
        std::vector<bodo_array_type::arr_type_enum> out_arr_types;
        std::vector<Bodo_CTypes::CTypeEnum> out_dtypes;
        for (size_t i = 0; i < update_arr_types.size(); i++) {
            std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
                out_arr_type_info = get_groupby_output_dtype(
                    combine_ftype, update_arr_types[i], update_dtypes[i]);
            out_arr_types.push_back(std::get<0>(out_arr_type_info));
            out_dtypes.push_back(std::get<1>(out_arr_type_info));
        }

        return std::tuple(out_arr_types, out_dtypes);
    }

    /**
     * @brief clear the update column vector to allow calling update with new
     * data batch (used in streaming groupby)
     *
     */
    void clearUpdateCols() { update_cols.clear(); }

    /**
     * @brief Set update columns to allow calling combine with new data batch
     * (used in streaming groupby)
     *
     * @param update_cols_
     */
    void setUpdateCols(std::vector<std::shared_ptr<array_info>> update_cols_) {
        update_cols = update_cols_;
    }

    /**
     * @brief Set combine columns to allow producing combine results in the
     * given buffer (used in streaming groupby)
     *
     * @param combine_cols_
     */
    void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) {
        combine_cols = combine_cols_;
    }

   protected:
    std::shared_ptr<array_info>
        in_col;  // the input column (from groupby input table) to which
                 // this column set corresponds to
    int ftype;
    bool combine_step;   // GroupbyPipeline is going to perform a combine
                         // operation or not
    bool use_sql_rules;  // Use SQL rules for aggregation or Pandas?
    std::vector<std::shared_ptr<array_info>>
        update_cols;  // columns for update step
    std::vector<std::shared_ptr<array_info>>
        combine_cols;  // columns for combine step
};

/**
 * Column Set for the mean operation
 *
 */
class MeanColSet : public BasicColSet {
   public:
    MeanColSet(std::shared_ptr<array_info> in_col, bool combine_step,
               bool use_sql_rules);
    virtual ~MeanColSet();
    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);
    virtual void update(const std::vector<grouping_info>& grp_infos);
    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);
    virtual void eval(const grouping_info& grp_info);

    std::tuple<std::vector<bodo_array_type::arr_type_enum>,
               std::vector<Bodo_CTypes::CTypeEnum>>
    getUpdateColumnTypes(
        const std::vector<bodo_array_type::arr_type_enum>& in_arr_types,
        const std::vector<Bodo_CTypes::CTypeEnum>& in_dtypes) override {
        // Mean's update columns are always float64 for sum data and uint64 for
        // count data. See MeanColSet::alloc_update_columns()
        return std::tuple(
            std::vector<bodo_array_type::arr_type_enum>{
                bodo_array_type::NULLABLE_INT_BOOL,
                bodo_array_type::NULLABLE_INT_BOOL},
            std::vector<Bodo_CTypes::CTypeEnum>{Bodo_CTypes::FLOAT64,
                                                Bodo_CTypes::UINT64});
    }
};

/**
 * @brief WindowColSet column set for window operations.
 *
 */
class WindowColSet : public BasicColSet {
   public:
    /**
     * Construct Window column set
     * @param in_cols input columns of groupby associated with this column set.
     * There are the columns that we will sort on.
     * @param _window_funcs: What function(s) are we computing.
     * @param _asc: Are the sort columns ascending on the input column.
     * @param _na_pos: Are NAs last in the sort columns
     * @param _windiw_args: Any additional window arguments
     * @param _is_parallel: flag to identify whether data is distributed
     * @param use_sql_rules: Do we use SQL or Pandas null handling rules.
     *
     */
    WindowColSet(std::vector<std::shared_ptr<array_info>>& in_cols,
                 std::vector<int64_t> _window_funcs, std::vector<bool>& _asc,
                 std::vector<bool>& _na_pos, std::vector<void*>& _window_args,
                 int n_input_cols, bool _is_parallel, bool use_sql_rules);
    virtual ~WindowColSet();

    /**
     * Allocate column for update step.
     * @param num_groups: number of groups found in the input table
     * @param[in,out] out_cols: vector of columns of update table. This method
     * adds columns to this vector.
     * NOTE: the added column is an integer array with same length as
     * input column regardless of input column types (i.e num_groups is not used
     * in this case)
     */
    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);
    /**
     * Perform update step for this column set. This first shuffles
     * the data based on the orderby condition + group columns and
     * then computes the window function. If this is a parallel operations
     * then we must update the shuffle info so the reverse shuffle will
     * be correct. If this is a serial operation then we need to execute
     * a local reverse shuffle.
     * @param grp_infos: grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos);

    /**
     * Obtain the final output columns resulting from the groupby operation on
     * this column set by appending them into a provided vector.
     */
    virtual void addOutputColumns(
        std::vector<std::shared_ptr<array_info>>& out_cols);

   private:
    std::vector<std::shared_ptr<array_info>> input_cols;
    std::vector<int64_t> window_funcs;
    std::vector<bool> asc;
    std::vector<bool> na_pos;
    std::vector<void*> window_args;
    int n_input_cols;
    bool is_parallel;  // whether input column data is distributed or
                       // replicated
};

/**
 * @brief Colset for idxmin and idxmax operations
 *
 */
class IdxMinMaxColSet : public BasicColSet {
   public:
    IdxMinMaxColSet(std::shared_ptr<array_info> in_col,
                    std::shared_ptr<array_info> _index_col, int ftype,
                    bool combine_step, bool use_sql_rules);

    virtual ~IdxMinMaxColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

   private:
    std::shared_ptr<array_info> index_col;
};

/**
 * @brief Colset for boolxor_agg
 *
 */
class BoolXorColSet : public BasicColSet {
   public:
    BoolXorColSet(std::shared_ptr<array_info> in_col, int ftype,
                  bool combine_step, bool use_sql_rules);

    virtual ~BoolXorColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    virtual void eval(const grouping_info& grp_info);
};

/**
 * @brief Colset for Variance and Standard deviation operations.
 *
 */
class VarStdColSet : public BasicColSet {
   public:
    VarStdColSet(std::shared_ptr<array_info> in_col, int ftype,
                 bool combine_step, bool use_sql_rules);

    virtual ~VarStdColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    virtual void eval(const grouping_info& grp_info);

    void addOutputColumns(
        std::vector<std::shared_ptr<array_info>>& out_cols) override {
        out_cols.push_back(this->out_col);
    }

    std::tuple<std::vector<bodo_array_type::arr_type_enum>,
               std::vector<Bodo_CTypes::CTypeEnum>>
    getUpdateColumnTypes(
        const std::vector<bodo_array_type::arr_type_enum>& in_arr_types,
        const std::vector<Bodo_CTypes::CTypeEnum>& in_dtypes) override;

   private:
    std::shared_ptr<array_info> out_col = nullptr;
};

/**
 * @brief Colset for Skew operation.
 *
 */
class SkewColSet : public BasicColSet {
   public:
    SkewColSet(std::shared_ptr<array_info> in_col, int ftype, bool combine_step,
               bool use_sql_rules);

    virtual ~SkewColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    virtual void eval(const grouping_info& grp_info);
};

/**
 * @brief Colset for ListAgg operation.
 */
class ListAggColSet : public BasicColSet {
   public:
    ListAggColSet(std::shared_ptr<array_info> in_col,
                  std::shared_ptr<array_info> sep_col,
                  std::vector<std::shared_ptr<array_info>> orderby_cols,
                  std::vector<bool> window_ascending,
                  std::vector<bool> window_na_position);

    virtual ~ListAggColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual std::shared_ptr<array_info> getOutputColumn();

   private:
    std::string listagg_sep;
    std::vector<std::shared_ptr<array_info>> orderby_cols;
    std::vector<bool> window_ascending;
    std::vector<bool> window_na_position;
};

/**
 * @brief Colset for Kurtosis operation.
 *
 */
class KurtColSet : public BasicColSet {
   public:
    KurtColSet(std::shared_ptr<array_info> in_col, int ftype, bool combine_step,
               bool use_sql_rules);

    virtual ~KurtColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    virtual void eval(const grouping_info& grp_info);
};

/**
 * @brief Colset for UDF operations that have been optimized
 * in their compilation.
 */
class UdfColSet : public BasicColSet {
   public:
    UdfColSet(std::shared_ptr<array_info> in_col, bool combine_step,
              std::shared_ptr<table_info> udf_table, int udf_table_idx,
              int n_redvars, bool use_sql_rules);

    virtual ~UdfColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    virtual typename std::vector<std::shared_ptr<array_info>>::iterator
    update_after_shuffle(
        typename std::vector<std::shared_ptr<array_info>>::iterator& it);

    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0);

    virtual void eval(const grouping_info& grp_info);

   private:
    std::shared_ptr<table_info>
        udf_table;      // the table containing type info for UDF columns
    int udf_table_idx;  // index to my information in the udf table
    int n_redvars;      // number of redvar columns this UDF uses
};

/**
 * @brief Colset for general UDF operations that could not be optimized
 * in their compilation.
 */
class GeneralUdfColSet : public UdfColSet {
   public:
    GeneralUdfColSet(std::shared_ptr<array_info> in_col,
                     std::shared_ptr<table_info> udf_table, int udf_table_idx,
                     bool use_sql_rules);

    virtual ~GeneralUdfColSet();

    /**
     * Fill in the input table for general UDF cfunc. See udf_general_fn
     * and aggregate.py::gen_general_udf_cb for more information.
     */
    void fill_in_columns(const std::shared_ptr<table_info>& general_in_table,
                         const grouping_info& grp_info) const;
};

/**
 * @brief ColSet for Median operations.
 *
 */
class MedianColSet : public BasicColSet {
   public:
    MedianColSet(std::shared_ptr<array_info> in_col, bool _skipna,
                 bool use_sql_rules);

    virtual ~MedianColSet();

    virtual void update(const std::vector<grouping_info>& grp_infos);

   private:
    bool skipna;
};

/**
 * @brief ColSet for Mode operations.
 *
 */
class ModeColSet : public BasicColSet {
   public:
    ModeColSet(std::shared_ptr<array_info> in_col, bool _is_parallel,
               bool use_sql_rules);

    virtual ~ModeColSet();

    void update(const std::vector<grouping_info>& grp_infos) override;

   private:
    bool is_parallel;
};

/**
 * @brief ColSet for Nunique operations.
 *
 */
class NUniqueColSet : public BasicColSet {
   public:
    NUniqueColSet(std::shared_ptr<array_info> in_col, bool _dropna,
                  std::shared_ptr<table_info> nunique_table, bool do_combine,
                  bool _is_parallel, bool use_sql_rules);

    virtual ~NUniqueColSet();

    virtual void update(const std::vector<grouping_info>& grp_infos);

   private:
    bool dropna;
    std::shared_ptr<table_info> my_nunique_table = nullptr;
    bool is_parallel;
};

/**
 * @brief ColSet for cumulative operations.
 *
 */
class CumOpColSet : public BasicColSet {
   public:
    CumOpColSet(std::shared_ptr<array_info> in_col, int ftype, bool _skipna,
                bool use_sql_rules);

    virtual ~CumOpColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

   private:
    bool skipna;
};

/**
 * @brief ColSet for shift operations.
 *
 */
class ShiftColSet : public BasicColSet {
   public:
    ShiftColSet(std::shared_ptr<array_info> in_col, int ftype, int64_t _periods,
                bool use_sql_rules);

    virtual ~ShiftColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

   private:
    int64_t periods;
};

/**
 * @brief Colset for transform operations.
 *
 */
class TransformColSet : public BasicColSet {
   public:
    TransformColSet(std::shared_ptr<array_info> in_col, int ftype,
                    int _func_num, bool do_combine, bool is_parallel,
                    bool use_sql_rules);

    virtual ~TransformColSet();

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    // Call corresponding groupby function operation to compute
    // transform_op_col column.
    virtual void update(const std::vector<grouping_info>& grp_infos);

    // Fill the output column by copying values from the transform_op_col column
    virtual void eval(const grouping_info& grp_info);

   private:
    bool is_parallel;
    int64_t transform_func;
    std::unique_ptr<BasicColSet> transform_op_col;
};

/**
 * @brief Column set for the head operation.
 *
 */
class HeadColSet : public BasicColSet {
   public:
    HeadColSet(std::shared_ptr<array_info> in_col, int ftype,
               bool use_sql_rules);

    virtual ~HeadColSet();

    virtual void alloc_update_columns(
        size_t update_col_len,
        std::vector<std::shared_ptr<array_info>>& out_cols);

    virtual void update(const std::vector<grouping_info>& grp_infos);

    void set_head_row_list(bodo::vector<int64_t>& row_list);

   private:
    bodo::vector<int64_t> head_row_list;
};

/**
 * @brief NgroupColSet column set for ngroup operation
 */
class NgroupColSet : public BasicColSet {
   public:
    /**
     * Construct Ngroup column set
     * @param in_col input column of groupby associated with this column set
     * @param _is_parallel: flag to identify whether data is distributed or
     * replicated across ranks
     */
    NgroupColSet(std::shared_ptr<array_info> in_col, bool _is_parallel,
                 bool use_sql_rules);

    virtual ~NgroupColSet();

    /**
     * Allocate column for update step.
     * @param num_groups: number of groups found in the input table
     * @param[in,out] out_cols: vector of columns of update table. This method
     * adds columns to this vector.
     * NOTE: the added column is an integer array with same length as
     * input column regardless of input column types (i.e num_groups is not used
     * in this case)
     */
    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols);

    /**
     * Perform update step for this column set. compute and fill my columns with
     * the result of the ngroup operation.
     * @param grp_infos: grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos);

   private:
    bool is_parallel;  // whether input column data is distributed or
                       // replicated.
};

/**
 * Construct and return a column set based on the ftype.
 * @param[in] in_col Vector of input columns upon which the function will
 * be computed. All functions should have at most 1 input column except for
 * the window functions which can have more.
 * @param[in] index_col The index column used by some operations. If
 * unused this will be a nullptr.
 * @param ftype function type associated with this column set.
 * @param do_combine whether GroupbyPipeline will perform combine operation
 *        or not.
 * @param skipna option used for nunique, cumsum, cumprod, cummin, cummax
 * @param periods option used for shift
 * @param transform_func option used for identifying transform function
 *        (currently groupby operation that are already supported)
 * @param is_parallel is the groupby implementation distributed?
 * @param window_ascending For the window ftype is each orderby column in the
 * window ascending?
 * @param window_na_position For the window ftype does each orderby column have
 * NA values last?
 * @param[in] udf_n_redvars For groupby udf functions these are the reduction
 * variables. For other operations this will be a nullptr.
 * @param[in] udf_table For groupby udf functions this is the table of used
 * columns.
 * @param udf_table_idx For groupby udf functions this is the column number to
 * select from udf_table for this operation.
 * @param[in] nunique_table For nunique this is a special table used for special
 * handling
 * @param use_sql_rules Should we use SQL rules for NULL handling/initial
 * values.
 * @return A pointer to the created col set.
 */
std::unique_ptr<BasicColSet> makeColSet(
    std::vector<std::shared_ptr<array_info>> in_cols,
    std::shared_ptr<array_info> index_col, int ftype, bool do_combine,
    bool skipna, int64_t periods, std::vector<int64_t> transform_funcs,
    int n_udf, bool is_parallel, std::vector<bool> window_ascending,
    std::vector<bool> window_na_position, std::vector<void*> window_args,
    int n_input_cols, int* udf_n_redvars = nullptr,
    std::shared_ptr<table_info> udf_table = nullptr, int udf_table_idx = 0,
    std::shared_ptr<table_info> nunique_table = nullptr,
    bool use_sql_rules = false);

#endif  // _GROUPBY_COL_SET_H_INCLUDED
