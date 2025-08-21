#pragma once

#include <memory>
#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "_groupby.h"
#include "_groupby_common.h"
#include "_groupby_udf.h"

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

/**
 * Function pointer for window computation operations.
 */
typedef void (*window_computation_fn)(
    std::vector<std::shared_ptr<array_info>>& orderby_arrs,
    std::vector<int64_t> window_funcs,
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builders,
    grouping_info const& grp_info, const std::vector<bool>& asc_vect,
    const std::vector<bool>& na_pos_vect,
    const std::shared_ptr<table_info> window_args, int n_input_cols,
    bool is_parallel, bool use_sql_rules, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm);

/**
 * @brief Helper function to determine the correct ftype and array-type for
 * index column to use during the update step of Min Row-Number Filter.
 *
 * @param n_orderby_arrs Number of order-by columns for the MRNF.
 * @param asc_vec Bitmask specifying the sort direction for the order-by
 * columns.
 * @param na_pos_vec Bitmask specifying whether nulls should be considered
 * 'last' in the order-by columns.
 * @return std::tuple<int64_t, bodo_array_type::arr_type_enum> Tuple of the
 * ftype and array-type for the index column.
 */
std::tuple<int64_t, bodo_array_type::arr_type_enum>
get_update_ftype_idx_arr_type_for_mrnf(size_t n_orderby_arrs,
                                       const std::vector<bool>& asc_vec,
                                       const std::vector<bool>& na_pos_vec);

/**
 * @brief Primary implementation of MRNF.
 * The function updates 'idx_col' in place and writes the index
 * of the output row corresponding to each group.
 * This is used by both the streaming MRNF implementation as well
 * as the non-streaming window implementation.
 * Note that this doesn't make any assumptions about the sorted-ness
 * of the data, i.e. it computes the minimum row per group based on
 * the order-by columns. If the data is known to be already sorted, use
 * the specialized 'min_row_number_filter_window_computation_already_sorted'
 * implementation instead.
 *
 * @param[in, out] idx_col Column with indices of the output rows. This will be
 * updated in place.
 * @param orderby_cols The columns used in the order by clause of the query.
 * @param grp_info Grouping information for the rows in the table.
 * @param asc Bitmask specifying the sort direction for the order-by
 * columns.
 * @param na_pos Bitmask specifying whether nulls should be considered
 * 'last' in the order-by columns.
 * @param update_ftype The ftype to use for update. This is the output
 * from 'get_update_ftype_idx_arr_type_for_mrnf'.
 * @param use_sql_rules Should initialization functions obey SQL semantics?
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 */
void min_row_number_filter_no_sort(
    const std::shared_ptr<array_info>& idx_col,
    std::vector<std::shared_ptr<array_info>>& orderby_cols,
    grouping_info const& grp_info, const std::vector<bool>& asc,
    const std::vector<bool>& na_pos, int update_ftype, bool use_sql_rules,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

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
     * Allocate running value columns for the update/combine steps, and places
     * them in out_cols. Note that this does not modify any fields in the colset
     * itself; The caller is responsible for calling setUpdateCols or
     * setCombineCols on the output of this function to set the update/combine
     * columns of this colset.
     *
     *
     * Allocated columns should always be identical to those allocated in
     * alloc_update_columns/alloc_combine_columns.
     *
     * @param num_groups number of groups found in the input table
     * @param[in,out] out_cols vector of columns for the update/combine step.
     * This method adds columns to this vector.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    /**
     * Allocates the running value columns for the update step, places
     * them in out_cols, and sets the update_cols field of this colset.
     *
     * Allocated columns should always be identical to those allocated in
     * alloc_combine_columns/alloc_running_value_columns.
     *
     * May do some additional misc initialization work for specific older
     * column sets that require it (std, var, skew, etc.), but this
     * Should not be overwritten moving forward.
     *
     * TODO: Move misc init work to a separate method and remove this method
     * in favor of always calling alloc_running_value_columns
     *
     * @param num_groups number of groups found in the input table
     * @param[in,out] out_cols vector of columns for the update step.
     * This method adds columns to this vector.
     * @param alloc_out_if_no_combine If we won't be doing a combine,
     * allocate the output column as well. This is required for some ColSets
     * such as VarStd, Skew, etc. This is true by default. In streaming groupby,
     * even in the ACC (accumulate_before_update) path, we don't want to
     * allocate the output column since we do that separately.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) {
        this->alloc_running_value_columns(num_groups, out_cols, pool,
                                          std::move(mm));
        this->update_cols = out_cols;
    }

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void update(
        const std::vector<grouping_info>& grp_infos,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

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
     * Allocates the running value columns for the combine step, places
     * them in out_cols, and sets the combine_cols field of this colset.
     *
     * Allocated columns should always be identical to those allocated in
     * alloc_update_columns/alloc_running_value_columns.
     *
     * May do some additional misc initialization work for specific older
     * column sets that require it (std, var, skew, etc.), but this
     * Should not be overwritten moving forward.
     *
     * TODO: Move misc init work to a separate method and remove this method
     * in favor of always calling alloc_running_value_columns
     *
     * @param num_groups number of groups found in the input table
     * @param[in,out] out_cols vector of columns for the combine step.
     * This method adds columns to this vector.
     */
    virtual void alloc_combine_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols) {
        alloc_running_value_columns(num_groups, out_cols);
        this->combine_cols = out_cols;
    }

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
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void eval(
        const grouping_info& grp_info,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    /**
     * Obtain the final output columns resulting from the groupby operation on
     * this column set. This will free all other intermediate or auxiliary
     * columns (if any) used by the column set (like reduction variables).
     * @return constant vector of output columns
     */
    virtual const std::vector<std::shared_ptr<array_info>> getOutputColumns();

    /**
     * @brief Set input columns of this ColSet (used in streaming groupby for
     * processing new batches)
     *
     * @param new_in_cols new input columns
     */
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) {
        // TODO[BSE-578]: implement setInCol() for other colsets that can have
        // more input columns
        in_col = new_in_cols[0];
    }

    /**
     * @brief For window functions
     *
     * @return std::vector<int32_t>
     */
    virtual const std::vector<int64_t> getFtypes() {
        throw std::runtime_error(
            "getOutputTypes() not implemented for this colset");
    }

    /**
     * @brief For window col set
     *
     * @param out_dict_builder The dictionary builders for the output columns
     */
    virtual void setOutDictBuilders(
        std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builder) {
        throw std::runtime_error(
            "setOutDictBuilders not implemented for this colset");
    }

    /**
     * @brief Get combine/update column types for this ColSet function with the
     * given input array types. Should match arrays allocated with
     * alloc_update_or_combine_columns().
     *
     * @param in_arr_types input array types
     * @param in_dtypes input array dtypes
     * @return std::unique_ptr<bodo::Schema> update column array types and
     * dtypes
     */
    virtual std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const {
        // TODO[BSE-578]: implement getRunningValueColumnTypes() for other
        // colsets that can have more update columns
        // place holder arr typ or dtype in the case there are no input columns
        bodo_array_type::arr_type_enum arr_typ = bodo_array_type::NUMPY;
        Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::INT8;
        if (in_schema->column_types.size() > 0) {
            arr_typ = in_schema->column_types[0]->array_type;
            dtype = in_schema->column_types[0]->c_type;
        }

        std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
            out_arr_type = get_groupby_output_dtype(ftype, arr_typ, dtype);
        std::vector<std::unique_ptr<bodo::DataType>> datatypes;
        datatypes.push_back(std::make_unique<bodo::DataType>(
            std::get<0>(out_arr_type), std::get<1>(out_arr_type)));
        return std::make_unique<bodo::Schema>(std::move(datatypes));
    }
    /**
     * @brief Set update columns to allow calling combine with new data batch
     * (used in streaming groupby)
     *
     * @param update_cols_
     */
    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) {
        update_cols = update_cols_;
    }

    /**
     * @brief Set combine columns to allow producing combine results in the
     * given buffer (used in streaming groupby)
     *
     * @param combine_cols_
     */
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) {
        combine_cols = combine_cols_;
    }

    /**
     * @brief Clear all state of this column set (used in streaming groupby)
     */
    virtual void clear() {
        this->update_cols.clear();
        this->combine_cols.clear();
        this->in_col.reset();
    }

    /**
     * @brief Returns a vector of array types needed for separate output columns
     * or an empty vector if none are needed
     */
    virtual std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumnType() {
        return {};
    }

    /**
     * @brief Returns a vector of the output types for the compute column(s).
     * This is currently only supported for window.
     */
    virtual std::vector<std::unique_ptr<bodo::DataType>> getOutputTypes() {
        throw std::runtime_error(
            "getOutputTypes() not implemented for this colset");
    }

    /**
     * @brief Set output column for this column set if this
     * colset uses output columns (used in streaming groupby)
     */
    virtual void setOutputColumn(std::shared_ptr<array_info> out_col_) {
        throw std::runtime_error(
            "setOutputColumn() this colset does not use separate output "
            "columns");
    }

   protected:
    std::shared_ptr<array_info>
        in_col;  // the input column (from groupby input table) to which
                 // this column set corresponds to
    const int ftype;
    const bool combine_step;   // GroupbyPipeline is going to perform a combine
                               // operation or not
    const bool use_sql_rules;  // Use SQL rules for aggregation or Pandas?
    std::vector<std::shared_ptr<array_info>>
        update_cols;  // columns for update step
    std::vector<std::shared_ptr<array_info>>
        combine_cols;  // columns for combine step
};

/**
 * @brief Column Set for the SIZE operation.
 *
 */
class SizeColSet : public BasicColSet {
   public:
    // NOTE: We do not require an input column for Size since the grouping
    // information will be sufficient for computing the output.
    SizeColSet(bool combine_step, bool use_sql_rules);
    virtual ~SizeColSet();

    /**
     * @brief Allocate a NUMPY INT64 array for storing the running output.
     *
     * @param num_groups number of groups found in the input table
     * @param[in,out] out_cols vector of columns for the update/combine step.
     * This method adds columns to this vector.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    /**
     * @brief For Size, input columns are not required, so this will be a NOP.
     *
     * @param new_in_cols
     */
    void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override;

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the Size aggregation operation.
     * @param grouping info calculated by GroupbyPipeline
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool,
                std::shared_ptr<::arrow::MemoryManager> mm) override;
};

/**
 * Column Set for the FIRST operation
 *
 */
class FirstColSet : public BasicColSet {
   public:
    FirstColSet(std::shared_ptr<array_info> in_col, bool combine_step,
                bool use_sql_rules);
    virtual ~FirstColSet();
    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override {
        assert(in_schema->ncols() == 1);
        auto out_schema = std::make_unique<bodo::Schema>(*in_schema);
        for (size_t i = 0; i < in_schema->ncols(); ++i) {
            if (in_schema->column_types[i]->array_type ==
                bodo_array_type::NUMPY) {
                out_schema->append_column(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::_BOOL);
            }
        }
        return out_schema;
    }

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0) override;

    virtual void update(
        const std::vector<grouping_info>& grp_infos,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;
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
    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;
    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;
    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;
    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override {
        // Mean's update columns are always float64 for sum data and uint64 for
        // count data. See MeanColSet::alloc_update_columns()

        std::vector<std::unique_ptr<bodo::DataType>> datatypes;
        datatypes.push_back(std::make_unique<bodo::DataType>(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64));
        datatypes.push_back(std::make_unique<bodo::DataType>(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::UINT64));
        return std::make_unique<bodo::Schema>(std::move(datatypes));
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
     * @param in_cols input columns of groupby associated with this column
     * set. There are the columns that we will sort on.
     * @param _window_funcs: What function(s) are we computing.
     * @param _asc: Are the sort columns ascending on the input column.
     * @param _na_pos: Are NAs last in the sort columns
     * @param _window_args: Any additional window arguments
     * @param _is_parallel: flag to identify whether data is distributed
     * @param use_sql_rules: Do we use SQL or Pandas null handling rules.
     *
     */
    WindowColSet(std::vector<std::shared_ptr<array_info>>& in_cols,
                 std::vector<int64_t> _window_funcs, std::vector<bool>& _asc,
                 std::vector<bool>& _na_pos,
                 std::shared_ptr<table_info> _window_args, int n_input_cols,
                 bool _is_parallel, bool use_sql_rules,
                 std::vector<std::vector<std::unique_ptr<bodo::DataType>>>
                     _in_arr_types_vec);
    virtual ~WindowColSet();

    /**
     * Allocate column for update step.
     * @param num_groups: number of groups found in the input table
     * @param[in,out] out_cols: vector of columns of update table. This
     * method adds columns to this vector. NOTE: the added column is an
     * integer array with same length as input column regardless of input
     * column types (i.e num_groups is not used in this case)
     */
    virtual void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;
    /**
     * Perform update step for this column set. This first shuffles
     * the data based on the orderby condition + group columns and
     * then computes the window function. If this is a parallel operations
     * then we must update the shuffle info so the reverse shuffle will
     * be correct. If this is a serial operation then we need to execute
     * a local reverse shuffle.
     * @param grp_infos: grouping info calculated by GroupbyPipeline
     * @param pool Memory pool to use for allocations during the execution
     * of this function (not actually used)
     * @param mm Memory manager associated with the pool (not actually used)
     */
    virtual void update(
        const std::vector<grouping_info>& grp_infos,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    /**
     * Obtain the final output columns resulting from the groupby operation
     * on this column set.
     * @return constant vector of output columns
     */
    virtual const std::vector<std::shared_ptr<array_info>> getOutputColumns()
        override;

    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "WindowColSet only supports the accumulate streaming path");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        this->input_cols = new_in_cols;
    }

    virtual const std::vector<int64_t> getFtypes() override {
        return this->window_funcs;
    }

    virtual void clear() override {
        BasicColSet::clear();
        this->input_cols.clear();
    }

    virtual std::vector<std::unique_ptr<bodo::DataType>> getOutputTypes()
        override;

    virtual void setOutDictBuilders(
        std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builder)
        override;

   private:
    std::vector<std::shared_ptr<array_info>> input_cols;
    const std::vector<int64_t> window_funcs;
    const std::vector<bool> asc;
    const std::vector<bool> na_pos;
    const std::shared_ptr<table_info> window_args;
    const int n_input_cols;
    const bool is_parallel;  // whether input column data is distributed or
                             // replicated
    const std::vector<std::vector<std::unique_ptr<bodo::DataType>>>
        in_arr_types_vec;
    std::vector<std::shared_ptr<DictionaryBuilder>> out_dict_builders;

    // Function pointer for window_computation() from stream_window_cpp
    // module. Loaded lazily only when needed to avoid loading a large
    // binary that slows down import and worker spin up time.
    window_computation_fn window_computation_func;
};

/**
 * @brief ColSet for the specialized Min-Row Filter
 * case in streaming groupby.
 *
 */
class StreamingMRNFColSet : public BasicColSet {
   public:
    /**
     * @brief Construct a new MRNF col set.
     *
     * @param _asc Are the sort columns ascending on the input column.
     * @param _na_pos Are NAs last in the sort columns
     * @param use_sql_rules Do we use SQL or Pandas null handling rules.
     */
    StreamingMRNFColSet(std::vector<bool>& _asc, std::vector<bool>& _na_pos,
                        bool use_sql_rules);
    virtual ~StreamingMRNFColSet();

    /**
     * @brief Set the orderby columns for computation.
     *
     * @param new_in_cols Vector of the orderby columns.
     *  These won't be modified.
     */
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) {
        this->orderby_cols = new_in_cols;
    }

    /**
     * @brief Allocate intermediate buffer to store the index of the min element
     * for each group. This will be used in the update step.
     *
     * @param num_groups Number of groups found in the input table.
     * @param[in, out] out_cols The allocated column will be appended to this
     * vector.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    /**
     * @brief Perform the update step. This will populate the index
     * column (update_cols[0]) with the index of the output row
     * for each group.
     *
     * @param grp_infos Information about the groups.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    virtual void update(
        const std::vector<grouping_info>& grp_infos,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    /**
     * @brief Get the output columns. In this case, this is simply
     * the index column containing the index of the output row
     * for each group.
     *
     * @return const std::vector<std::shared_ptr<array_info>> Vector
     *  with a single column containing the indices of the output rows.
     */
    virtual const std::vector<std::shared_ptr<array_info>> getOutputColumns() {
        return this->update_cols;
    }

    /**
     * @brief Clear all state of this column set (used in streaming groupby).
     * This will release the references to the order-by columns, any update
     * columns that have been allocated, etc.
     *
     */
    virtual void clear() {
        BasicColSet::clear();
        this->orderby_cols.clear();
    }

   private:
    /// Static state:
    const std::vector<bool> asc;
    const std::vector<bool> na_pos;
    // Ftype to use during update. This is decided based on the number of
    // order-by columns, 'asc' and 'na'.
    int64_t update_ftype;
    // Array type of the index column to allocate during update.
    // This will be either NULLABLE or NUMPY.
    bodo_array_type::arr_type_enum update_idx_arr_type;

    /// Ephemeral streaming state:
    std::vector<std::shared_ptr<array_info>> orderby_cols;
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

    virtual void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    virtual void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    virtual void update(
        const std::vector<grouping_info>& grp_infos,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    virtual void alloc_combine_columns(
        size_t num_groups,
        std::vector<std::shared_ptr<array_info>>& out_cols) override;

    virtual void combine(const grouping_info& grp_info,
                         int64_t init_start_row = 0) override;

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "IdxMinMaxColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "IdxMinMaxColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "IdxMinMaxColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "IdxMinMaxColSet not implemented for streaming groupby");
    }

   private:
    const std::shared_ptr<array_info> index_col;
};

/**
 * @brief Colset for boolxor_agg
 *
 */
class BoolXorColSet : public BasicColSet {
   public:
    BoolXorColSet(std::shared_ptr<array_info> in_col, int ftype,
                  bool combine_step, bool use_sql_rules);

    ~BoolXorColSet() override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;

    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override;
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

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void alloc_combine_columns(
        size_t num_groups,
        std::vector<std::shared_ptr<array_info>>& out_cols) override;

    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;

    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    const std::vector<std::shared_ptr<array_info>> getOutputColumns() override {
        return {out_col};
    }

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override;

    virtual void clear() override {
        BasicColSet::clear();
        this->out_col.reset();
    }

    virtual std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumnType() override {
        return {{bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64}};
    }

    virtual void setOutputColumn(
        std::shared_ptr<array_info> out_col_) override {
        if (out_col_ == nullptr) {
            throw std::runtime_error("out_col_ is null");
        }
        if (out_col_->dtype != Bodo_CTypes::FLOAT64) {
            throw std::runtime_error("out_col_ is not FLOAT64");
        }
        if (out_col_->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
            throw std::runtime_error("out_col_ is not NULLABLE_INT_BOOL");
        }
        this->out_col = out_col_;
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

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

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void alloc_combine_columns(
        size_t num_groups,
        std::vector<std::shared_ptr<array_info>>& out_cols) override;

    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;

    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    const std::vector<std::shared_ptr<array_info>> getOutputColumns() override {
        return {out_col};
    }

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override;

    virtual void clear() override {
        BasicColSet::clear();
        this->out_col.reset();
    }

    virtual std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumnType() override {
        return {std::make_pair(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT64)};
    }
    virtual void setOutputColumn(
        std::shared_ptr<array_info> out_col_) override {
        if (out_col_ == nullptr) {
            throw std::runtime_error("out_col_ is null");
        }
        if (out_col_->dtype != Bodo_CTypes::FLOAT64) {
            throw std::runtime_error("out_col_ is not FLOAT64");
        }
        if (out_col_->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
            throw std::runtime_error("out_col_ is not NULLABLE_INT_BOOL");
        }
        this->out_col = out_col_;
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

   private:
    std::shared_ptr<array_info> out_col = nullptr;
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

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    std::shared_ptr<array_info> getOutputColumn();

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "ListAggColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "ListAggColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "ListAggColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "ListAggColSet not implemented for streaming groupby");
    }

   private:
    std::string listagg_sep;
    const std::vector<std::shared_ptr<array_info>> orderby_cols;
    const std::vector<bool> window_ascending;
    const std::vector<bool> window_na_position;
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

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void alloc_combine_columns(
        size_t num_groups,
        std::vector<std::shared_ptr<array_info>>& out_cols) override;

    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;

    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    const std::vector<std::shared_ptr<array_info>> getOutputColumns() override {
        return {out_col};
    }

    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override;

    virtual void clear() override {
        BasicColSet::clear();
        this->out_col.reset();
    }

    virtual std::vector<
        std::pair<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>>
    getSeparateOutputColumnType() override {
        return {{bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64}};
    }
    virtual void setOutputColumn(
        std::shared_ptr<array_info> out_col_) override {
        if (out_col_ == nullptr) {
            throw std::runtime_error("out_col_ is null");
        }
        if (out_col_->dtype != Bodo_CTypes::FLOAT64) {
            throw std::runtime_error("out_col_ is not FLOAT64");
        }
        if (out_col_->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
            throw std::runtime_error("out_col_ is not NULLABLE_INT_BOOL");
        }
        this->out_col = out_col_;
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(this->out_col, this->ftype,
                                  use_sql_rules);  // zero initialize
    }

   private:
    std::shared_ptr<array_info> out_col = nullptr;
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

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    typename std::vector<std::shared_ptr<array_info>>::iterator
    update_after_shuffle(
        typename std::vector<std::shared_ptr<array_info>>::iterator& it)
        override;

    void alloc_combine_columns(
        size_t num_groups,
        std::vector<std::shared_ptr<array_info>>& out_cols) override;

    void combine(const grouping_info& grp_info,
                 int64_t init_start_row = 0) override;

    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "UDFColSet not implemented for streaming groupby");
    }
    void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "UDFColSet not implemented for streaming groupby");
    }
    void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "UDFColSet not implemented for streaming groupby");
    }
    void clear() override {
        throw std::runtime_error(
            "UDFColSet not implemented for streaming groupby");
    }

   private:
    const std::shared_ptr<table_info>
        udf_table;            // the table containing type info for UDF columns
    const int udf_table_idx;  // index to my information in the udf table
    const int n_redvars;      // number of redvar columns this UDF uses
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

class StreamingUDFColSet : public BasicColSet {
   public:
    StreamingUDFColSet(std::shared_ptr<array_info> in_col,
                       std::shared_ptr<table_info> out_table,
                       stream_udf_t* func, bool use_sql_rules);

    virtual ~StreamingUDFColSet();

    /**
     * @brief Get the Running Value Column Types based on udf_table and
     * udf_table_idx.
     *
     * @param in_schema Input schema (unused)
     * @return std::unique_ptr<bodo::Schema>
     */
    std::unique_ptr<bodo::Schema> getRunningValueColumnTypes(
        const std::shared_ptr<bodo::Schema>& in_schema) const override;

    /**
     * @brief Allocate dummy update columns, actual update columns are created
     * inside of update.
     */
    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    /**
     * @brief Call a cfunc for each group in grp_info and concatenate results.
     * update will allocate the update column based on the results.
     */
    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void setInCol(std::vector<std::shared_ptr<array_info>>) override;

    void clear() override;

   private:
    const std::shared_ptr<table_info>
        out_table;  // Table containing a single column of UDF output type.
    std::shared_ptr<table_info>
        in_table;        // Table containing input columns for the UDF.
    stream_udf_t* func;  // Callback for computing the UDF on a single group.
};

/**
 * @brief ColSet for Percentile operations.
 *
 */
class PercentileColSet : public BasicColSet {
   public:
    PercentileColSet(std::shared_ptr<array_info> in_col,
                     std::shared_ptr<array_info> percentile_col,
                     bool _interpolate, bool use_sql_rules);

    virtual ~PercentileColSet();

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "PercentileColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "PercentileColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "PercentileColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "PercentileColSet not implemented for streaming groupby");
    }

   private:
    double percentile;
    const bool interpolate;
};

/**
 * @brief ColSet for Median operations.
 *
 */
class MedianColSet : public BasicColSet {
   public:
    MedianColSet(std::shared_ptr<array_info> in_col, bool _skip_na_data,
                 bool use_sql_rules);

    virtual ~MedianColSet();

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

   private:
    const bool skip_na_data;
};

/**
 * @brief ColSet for Mode operations.
 *
 */
class ModeColSet : public BasicColSet {
   public:
    ModeColSet(std::shared_ptr<array_info> in_col, bool use_sql_rules);

    virtual ~ModeColSet();

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;
};

/**
 * @brief ColSet for ARRAY_AGG.
 *
 */
class ArrayAggColSet : public BasicColSet {
   public:
    ArrayAggColSet(std::shared_ptr<array_info> in_col,
                   std::vector<std::shared_ptr<array_info>> orderby_cols,
                   std::vector<bool> ascending, std::vector<bool> na_position,
                   int ftype, bool _is_parallel);

    ~ArrayAggColSet() override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    const std::vector<std::shared_ptr<array_info>> getOutputColumns() override;
    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "ArrayAggColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "ArrayAggColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "ArrayAggColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "ArrayAggColSet not implemented for streaming groupby");
    }

   private:
    const std::vector<std::shared_ptr<array_info>> orderby_cols;
    const std::vector<bool> ascending;
    const std::vector<bool> na_position;
    const bool is_distinct;
    const bool is_parallel;
};

/**
 * @brief ColSet for OBJECT_AGG.
 *
 */
class ObjectAggColSet : public BasicColSet {
   public:
    ObjectAggColSet(std::shared_ptr<array_info> _key_col,
                    std::shared_ptr<array_info> _val_col, bool _is_parallel);

    ~ObjectAggColSet() override;

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    const std::vector<std::shared_ptr<array_info>> getOutputColumns() override;
    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "ObjectAggColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "ObjectAggColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "ObjectAggColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "ObjectAggColSet not implemented for streaming groupby");
    }

   private:
    const std::shared_ptr<array_info> key_col;
    const std::shared_ptr<array_info> val_col;
    const bool is_parallel;
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

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    virtual void clear() override {
        BasicColSet::clear();
        this->my_nunique_table.reset();
    }

   private:
    const bool skip_na_data;
    std::shared_ptr<table_info> my_nunique_table = nullptr;
    const bool is_parallel;
};

/**
 * @brief ColSet for cumulative operations.
 *
 */
class CumOpColSet : public BasicColSet {
   public:
    CumOpColSet(std::shared_ptr<array_info> in_col, int ftype,
                bool _skip_na_data, bool use_sql_rules);

    virtual ~CumOpColSet();

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "CumOpColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "CumOpColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "CumOpColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "CumOpColSet not implemented for streaming groupby");
    }

   private:
    const bool skip_na_data;
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

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "ShiftColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "ShiftColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "ShiftColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "ShiftColSet not implemented for streaming groupby");
    }

   private:
    const int64_t periods;
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

    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void alloc_update_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        const bool alloc_out_if_no_combine = true,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    // Call corresponding groupby function operation to compute
    // transform_op_col column.
    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    // Fill the output column by copying values from the transform_op_col
    // column
    void eval(const grouping_info& grp_info,
              bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
              std::shared_ptr<::arrow::MemoryManager> mm =
                  bodo::default_buffer_memory_manager()) override;

    virtual void setUpdateCols(
        std::vector<std::shared_ptr<array_info>> update_cols_) override {
        throw std::runtime_error(
            "TransformColSet not implemented for streaming groupby");
    }
    virtual void setCombineCols(
        std::vector<std::shared_ptr<array_info>> combine_cols_) override {
        throw std::runtime_error(
            "TransformColSet not implemented for streaming groupby");
    }
    virtual void setInCol(
        std::vector<std::shared_ptr<array_info>> new_in_cols) override {
        throw std::runtime_error(
            "TransformColSet not implemented for streaming groupby");
    }
    virtual void clear() override {
        throw std::runtime_error(
            "TransformColSet not implemented for streaming groupby");
    }

   private:
    const bool is_parallel;
    const int64_t transform_func;
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

    void alloc_running_value_columns(
        size_t update_col_len,
        std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

    void set_head_row_list(bodo::vector<int64_t>& row_list);

    virtual void clear() override {
        BasicColSet::clear();
        this->head_row_list.clear();
    }

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
     * @param[in,out] out_cols: vector of columns of update table. This
     * method adds columns to this vector. NOTE: the added column is an
     * integer array with same length as input column regardless of input
     * column types (i.e num_groups is not used in this case)
     */
    void alloc_running_value_columns(
        size_t num_groups, std::vector<std::shared_ptr<array_info>>& out_cols,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager()) override;

    void update(const std::vector<grouping_info>& grp_infos,
                bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
                std::shared_ptr<::arrow::MemoryManager> mm =
                    bodo::default_buffer_memory_manager()) override;

   private:
    const bool is_parallel;  // whether input column data is distributed or
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
 * @param skip_na_data option used for nunique, cumsum, cumprod, cummin,
 * cummax
 * @param periods option used for shift
 * @param transform_func option used for identifying transform function
 *        (currently groupby operation that are already supported)
 * @param is_parallel is the groupby implementation distributed?
 * @param window_ascending For the window ftype is each orderby column in
 * the window ascending?
 * @param window_na_position For the window ftype does each orderby column
 * have NA values last?
 * @param window_args A table of scalar arguments for window functions where
 * each argument is contained in a column with a single row.
 * @param[in] udf_n_redvars For groupby udf functions these are the
 * reduction variables. For other operations this will be a nullptr.
 * @param[in] udf_table For groupby udf functions this is the table of used
 * columns.
 * @param udf_table_idx For groupby udf functions this is the column number
 * to select from udf_table for this operation.
 * @param[in] nunique_table For nunique this is a special table used for
 * special handling
 * @param use_sql_rules Should we use SQL rules for NULL handling/initial
 * values.
 * @param[in] in_arr_types_vec A vector contains vectors of input types for each
 * function.
 * @return A pointer to the created col set.
 */
std::unique_ptr<BasicColSet> makeColSet(
    std::vector<std::shared_ptr<array_info>> in_cols,
    std::shared_ptr<array_info> index_col, int ftype, bool do_combine,
    bool skip_na_data, int64_t periods, std::vector<int64_t> transform_funcs,
    int n_udf, bool is_parallel, std::vector<bool> window_ascending,
    std::vector<bool> window_na_position,
    std::shared_ptr<table_info> window_args, int n_input_cols,
    int* udf_n_redvars = nullptr,
    std::shared_ptr<table_info> udf_table = nullptr, int udf_table_idx = 0,
    std::shared_ptr<table_info> nunique_table = nullptr,
    bool use_sql_rules = false,
    std::vector<std::vector<std::unique_ptr<bodo::DataType>>> in_arr_types_vec =
        {},
    stream_udf_t* udf_cfunc = nullptr);
