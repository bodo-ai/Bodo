#include "_groupby.h"
#include <map>
#include <string>
#include <utility>

#include "../_array_hash.h"
#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_dict_builder.h"
#include "../_distributed.h"
#include "../_shuffle.h"
#include "_groupby_col_set.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"
#include "_groupby_groups.h"
#include "_groupby_mpi_exscan.h"
#include "_groupby_udf.h"

/*
 An instance of GroupbyPipeline class manages a groupby operation. In a
 groupby operation, an arbitrary number of functions can be applied to each
 input column. The functions can vary between input columns. Each combination
 of (input column, function) is an operation that produces a column in the
 output table. The computation of each (input column, function) pair is
 encapsulated in what is called a "column set" (for lack of a better name).
 There are different column sets for different types of operations (e.g. var,
 mean, median, udfs, basic operations...). Each column set creates,
 initializes, operates on and manages the arrays needed to perform its
 computation. Different column set types may require different number of
 columns and dtypes.

 The main control flow of groupby is in
 GroupbyPipeline::run(). It invokes update, shuffle, combine and eval steps
 (as needed), and these steps iterate through the column sets and invoke
 their operations. We will refer to this as the standard groupby. The one
 exception is that if we decide to shuffle before update then the shuffle is
 done in the constructor and not run().

 Here is a more explicit definition of each of these steps:

 Update(): Update is the process of performing an aggregation function
 per group on each rank's chunk of data.

 shuffle(): Shuffle is the process of collecting data across ranks so all
 remaining values for one group are located on the same rank. There may be some
 special handling for performance where groups are intentionally kept split.

 combine(): Combine is the process of generating a global result for each group.
 It only runs if we execute update() and then shuffle(). Running update()
 before shuffle means that we may have computed the result for each group on
 multiple ranks, so we need to "combine" these results into a single output.
 This process is very similar to update() in its implementation, often only
 changing the function to an appropriate function to combine the groups (see
 get_combine_func).

 eval(): Eval is the process of doing one more pass over the final groups to
 "transform" the data into the desired output. For example, to compute "mean" we
 locally compute two separate values, the sum of values and the count and only
 output the actual mean once the results are fully aggregated.

 One crucial decision that complicates the code path is whether or not to a do a
 local update() before the shuffle. For operations that support updating before
 shuffling we make the decision about whether or not to compute a local update
 by estimating the number of unique values. If there are too many unique values
 then we conclude its not worthwhile to do a local update.

 In addition to the standard groupby we also have an MPI EXSCAN path for
 implementing cumulative functions. If we have only cumulative operations then
 we can use explicit MPI operations to reduce shuffle costs.

 Here is an example for each of the four main steps in the standard group by
 using COUNT.

 update(): For count the aggregation function is to count the number of non-null
 elements. This mean we each group we increment a counter whenever the data in
 a column is not null.

 shuffle(): For count we either combine data across ranks so that every value in
 the same group is located on the same rank. If we have already run the update()
 this means we are shuffling the current count for each group. If we have not
 run the update() then we are shuffling the original data.

 combine(): For count we combine the counts across ranks. This is done by
 summing the existing counts so we are doing a SUM operation here instead of the
 original count operation.

 eval(): Count already has the final output in a single array. This is a no-op.
*/
class GroupbyPipeline {
   public:
    GroupbyPipeline(
        std::shared_ptr<table_info> _in_table, int64_t _num_keys,
        int8_t* _ncols_per_func, int8_t* _n_window_calls_per_func,
        int64_t num_funcs, std::shared_ptr<table_info> _dispatch_table,
        std::shared_ptr<table_info> _dispatch_info, bool input_has_index,
        bool _is_parallel, int* ftypes, int* func_offsets, int* _udf_nredvars,
        std::shared_ptr<table_info> _udf_table, udf_table_op_fn update_cb,
        udf_table_op_fn combine_cb, udf_eval_fn eval_cb,
        udf_general_fn general_udfs_cb, bool skip_na_data, int64_t periods,
        int64_t* transform_funcs, int64_t _head_n, bool _return_key,
        bool _return_index, bool _key_dropna, bool* window_ascending,
        bool* window_na_position, std::shared_ptr<table_info> window_args,
        int8_t* n_window_args_per_func, int* n_input_cols_per_func,
        bool _maintain_input_size, int64_t _n_shuffle_keys, bool _use_sql_rules)
        : orig_in_table(_in_table),
          in_table(_in_table),
          num_keys(_num_keys),
          ncols_per_func(_ncols_per_func),
          n_window_calls_per_func(_n_window_calls_per_func),
          dispatch_table(std::move(_dispatch_table)),
          dispatch_info(std::move(_dispatch_info)),
          is_parallel(_is_parallel),
          return_key(_return_key),
          return_index(_return_index),
          key_dropna(_key_dropna),
          udf_table(std::move(_udf_table)),
          udf_n_redvars(_udf_nredvars),
          head_n(_head_n),
          maintain_input_size(_maintain_input_size),
          n_shuffle_keys(_n_shuffle_keys),
          use_sql_rules(_use_sql_rules) {
        tracing::Event ev("GroupbyPipeline()", is_parallel);
        udf_info = {.udf_table_dummy = udf_table,
                    .update = update_cb,
                    .combine = combine_cb,
                    .eval = eval_cb,
                    .general_udf = general_udfs_cb};
        // if true, the last column is the index on input and output.
        // this is relevant only to cumulative operations like cumsum
        // and transform.
        int index_i = int(input_has_index);
        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        bool has_udf = false;
        nunique_op = false;
        int nunique_count = 0;
        for (int i = 0; i < num_funcs; i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::gen_udf && is_parallel) {
                shuffle_before_update = true;
            }
            if (ftype == Bodo_FTypes::udf) {
                has_udf = true;
            }
            if (ftype == Bodo_FTypes::head) {
                head_op = true;
                if (is_parallel) {
                    shuffle_before_update = true;
                }
                break;
            }
            if (ftype == Bodo_FTypes::nunique) {
                nunique_op = true;
                req_extended_group_info = true;
                nunique_count++;
            } else if (ftype == Bodo_FTypes::median ||
                       ftype == Bodo_FTypes::mode ||
                       ftype == Bodo_FTypes::cumsum ||
                       ftype == Bodo_FTypes::cumprod ||
                       ftype == Bodo_FTypes::cummin ||
                       ftype == Bodo_FTypes::cummax ||
                       ftype == Bodo_FTypes::shift ||
                       ftype == Bodo_FTypes::transform ||
                       ftype == Bodo_FTypes::ngroup ||
                       ftype == Bodo_FTypes::window ||
                       ftype == Bodo_FTypes::listagg ||
                       ftype == Bodo_FTypes::array_agg ||
                       ftype == Bodo_FTypes::array_agg_distinct ||
                       ftype == Bodo_FTypes::percentile_cont ||
                       ftype == Bodo_FTypes::percentile_disc ||
                       ftype == Bodo_FTypes::object_agg) {
                // these operations first require shuffling the data to
                // gather all rows with the same key in the same process
                if (is_parallel) {
                    shuffle_before_update = true;
                }
                // these operations require extended group info
                req_extended_group_info = true;
                if (ftype == Bodo_FTypes::cumsum ||
                    ftype == Bodo_FTypes::cummin ||
                    ftype == Bodo_FTypes::cumprod ||
                    ftype == Bodo_FTypes::cummax) {
                    cumulative_op = true;
                } else if (ftype == Bodo_FTypes::shift) {
                    shift_op = true;
                } else if (ftype == Bodo_FTypes::transform) {
                    transform_op = true;
                } else if (ftype == Bodo_FTypes::ngroup) {
                    ngroup_op = true;
                } else if (ftype == Bodo_FTypes::window) {
                    window_op = true;
                }
                break;
            }
        }
        // In case of ngroup: previous loop will be skipped
        // As num_funcs will be 0 since ngroup output is single column
        // regardless of number of input and key columns
        // So, set flags for ngroup here.
        if (num_funcs == 0 && ftypes[0] == Bodo_FTypes::ngroup) {
            ngroup_op = true;
            // these operations first require shuffling the data to
            // gather all rows with the same key in the same process
            if (is_parallel) {
                shuffle_before_update = true;
            }
            // these operations require extended group info
            req_extended_group_info = true;
        }
        if (nunique_op) {
            if (nunique_count == num_funcs) {
                nunique_only = true;
            }
            ev.add_attribute("nunique_only", nunique_only);
        }

        // if gb.head and data are distributed, last column is key-sort column.
        int head_i = int(head_op && is_parallel);
        // Add key-sorting-column for gb.head() to sort output at the end
        // this is relevant only if data is distributed.
        if (head_i) {
            add_head_key_sort_column();
        }
        size_t num_input_cols = in_table->ncols() - index_i - head_i;
        for (size_t icol = 0; icol < num_input_cols; icol++) {
            std::shared_ptr<array_info> a = in_table->columns[icol];
            if (a->arr_type == bodo_array_type::DICT) {
                // Convert the local dictionary to global for hashing purposes
                make_dictionary_global_and_unique(a, is_parallel);
            }
        }

        // get hashes of keys
        // NOTE: this has to be num_keys and not n_shuffle_keys
        // to avoid having a far off estimated nunique_hashes
        // which could lead to having large chance of map insertion collisions.
        // See [BE-3371] for more context.
        hashes = hash_keys_table(in_table, num_keys, SEED_HASH_PARTITION,
                                 is_parallel);
        size_t nunique_hashes_global = 0;
        // get estimate of number of unique hashes to guide optimization.
        // if shuffle_before_update=true we are going to shuffle everything
        // first so we don't need statistics of current hashes
        if (is_parallel && !shuffle_before_update) {
            if (nunique_op) {
                // nunique_hashes_global is currently only used for gb.nunique
                // heuristic
                std::tie(nunique_hashes, nunique_hashes_global) =
                    get_nunique_hashes_global(hashes, in_table->nrows(),
                                              is_parallel);
            } else {
                nunique_hashes =
                    get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
            }
        } else if (!is_parallel) {
            nunique_hashes =
                get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
        }

        // compute statistics, to determine if we want to shuffle before update.
        if (is_parallel && (dispatch_table == nullptr) && !has_udf &&
            !shuffle_before_update) {
            // If the estimated number of groups (given by nunique_hashes)
            // is similar to the number of input rows, then it's better to
            // shuffle first instead of doing a local reduction

            // TODO To do this with UDF functions we need to generate
            // two versions of UDFs at compile time (one for
            // shuffle_before_update=true and one for
            // shuffle_before_update=false)

            int shuffle_before_update_local = 0;
            double local_expected_avg_group_size;
            if (nunique_hashes == 0) {
                local_expected_avg_group_size = 1.0;
            } else {
                local_expected_avg_group_size =
                    in_table->nrows() / double(nunique_hashes);
            }
            // XXX what threshold is best? Here we say on average we expect
            // every group to shrink.
            if (local_expected_avg_group_size <= 2.0) {
                shuffle_before_update_local = 1;
            }
            ev.add_attribute("local_expected_avg_group_size",
                             local_expected_avg_group_size);
            ev.add_attribute("shuffle_before_update_local",
                             shuffle_before_update_local);
            // global count of ranks that decide to shuffle before update
            int shuffle_before_update_count;
            CHECK_MPI(MPI_Allreduce(&shuffle_before_update_local,
                                    &shuffle_before_update_count, 1, MPI_INT,
                                    MPI_SUM, MPI_COMM_WORLD),
                      "GroupbyPipeline::GroupbyPipeline: MPI error on "
                      "MPI_Allreduce:");
            // TODO Need a better threshold or cost model to decide when
            // to shuffle: https://bodo.atlassian.net/browse/BE-1140
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            if (shuffle_before_update_count >= num_ranks * 0.5) {
                shuffle_before_update = true;
            }
        }

        // NOTE: This path will shuffle the data.
        if (shuffle_before_update) {
            // If we are using a subset of keys we need to use the hash function
            // based on the actual number of shuffle keys. Note: This shouldn't
            // matter in the other cases because we will recompute the hashes
            // based on the number of shuffle keys if we update then shuffle and
            // num_keys == n_shuffle_keys for nunique.
            if (num_keys != n_shuffle_keys) {
                hashes.reset();
                hashes = hash_keys_table(in_table, n_shuffle_keys,
                                         SEED_HASH_PARTITION, is_parallel);
            }

            // Code below is equivalent to:
            // std::shared_ptr<table_info> in_table = shuffle_table(in_table,
            // num_keys) We do this more complicated construction because we may
            // need the hashes and comm_info later.
            comm_info_ptr = std::make_shared<mpi_comm_info>(
                in_table->columns, hashes, is_parallel);
            in_table = shuffle_table_kernel(std::move(in_table), hashes,
                                            *comm_info_ptr, is_parallel);
            has_reverse_shuffle = cumulative_op || shift_op || transform_op ||
                                  ngroup_op || window_op;
            if (!has_reverse_shuffle) {
                hashes.reset();
            } else {
                // preserve input table hashes for reverse shuffle at the end
                in_hashes = hashes;
            }
            hashes = nullptr;
        } else if (nunique_op && is_parallel) {
            // **NOTE**: gb_nunique_preprocess can set
            // shuffle_before_update=true in some cases
            gb_nunique_preprocess(ftypes, num_funcs, nunique_hashes_global);
        }

        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;

        std::shared_ptr<array_info> index_col = nullptr;
        if (input_has_index) {
            // if gb.head() exclude head_op column as well (if data is
            // distributed).
            index_col =
                in_table->columns[in_table->columns.size() - 1 - head_i];
        }

        // construct the column sets, one for each (input_columns, func) pair.
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_columns, func) pair. Here
        // input_columns are the columns required to compute the function once,
        // typically 1 column.
        int k = 0;
        n_udf = 0;

        // minor note k should equal num_funcs at the end of the loop in all
        // situations,
        //  EXCEPT for aggregations that don't take input columns as arguments,
        //  (Size, count, head, etc)
        //  since num_keys == num_input_cols in that case.
        for (uint64_t i = num_keys; i < num_input_cols;
             k++) {  // for each data column
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            int8_t num_used_cols = ncols_per_func[k];
            int8_t nwindow_colls = n_window_calls_per_func[k];
            int n_input_cols = n_input_cols_per_func[k];
            for (int j = start; j != end;
                 j++) {  // for each function applied to this column
                // Copy the columns because we pass the input vector by
                // reference.
                std::vector<std::shared_ptr<array_info>> input_cols;
                std::vector<int64_t> transform_funcs_vect;
                std::vector<bool> window_ascending_vect;
                std::vector<bool> window_na_position_vect;
                for (int m = 0; m < nwindow_colls; m++) {
                    transform_funcs_vect.push_back(
                        transform_funcs[(i - num_keys) + m]);
                }

                for (int m = 0; m < num_used_cols; m++) {
                    // There is no ascending or na_position for the key columns
                    window_ascending_vect.push_back(
                        window_ascending[(i - num_keys) + m]);
                    window_na_position_vect.push_back(
                        window_na_position[(i - num_keys) + m]);
                }
                if (ftypes[j] == Bodo_FTypes::nunique &&
                    (nunique_tables.size() > 0)) {
                    for (int m = 0; m < num_used_cols; m++) {
                        input_cols.push_back(
                            nunique_tables[i + m]->columns[num_keys]);
                    }
                    col_sets.push_back(makeColSet(
                        input_cols, index_col, ftypes[j], do_combine,
                        skip_na_data, periods, transform_funcs_vect, n_udf,
                        is_parallel, window_ascending_vect,
                        window_na_position_vect, window_args, n_input_cols,
                        udf_n_redvars, udf_table, udf_table_idx,
                        nunique_tables[i], use_sql_rules));
                } else {
                    for (int m = 0; m < num_used_cols; m++) {
                        input_cols.push_back(in_table->columns[i + m]);
                    }
                    col_sets.push_back(
                        makeColSet(input_cols, index_col, ftypes[j], do_combine,
                                   skip_na_data, periods, transform_funcs_vect,
                                   n_udf, is_parallel, window_ascending_vect,
                                   window_na_position_vect, window_args,
                                   n_input_cols, udf_n_redvars, udf_table,
                                   udf_table_idx, nullptr, use_sql_rules));
                }
                if (ftypes[j] == Bodo_FTypes::udf ||
                    ftypes[j] == Bodo_FTypes::gen_udf) {
                    udf_table_idx += (1 + udf_n_redvars[n_udf]);
                    n_udf++;
                    if (ftypes[j] == Bodo_FTypes::gen_udf) {
                        gen_udf_col_sets.push_back(
                            std::dynamic_pointer_cast<GeneralUdfColSet>(
                                col_sets.back()));
                    }
                }
                ev.add_attribute("g_column_ftype_" + std::to_string(j),
                                 ftypes[j]);
            }
            // Increment the input columns.
            i += num_used_cols;
        }

        // This is needed if aggregation was just size/ngroup operation, it will
        // skip loop (ncols = num_keys + index_i)
        if (col_sets.size() == 0 && (ftypes[0] == Bodo_FTypes::size ||
                                     ftypes[0] == Bodo_FTypes::ngroup)) {
            col_sets.push_back(makeColSet(
                {in_table->columns[0]}, index_col, ftypes[0], do_combine,
                skip_na_data, periods, {0}, n_udf, is_parallel, {false},
                {false}, {nullptr}, 0, udf_n_redvars, udf_table, udf_table_idx,
                nullptr, use_sql_rules));
        }
        // Add key-sort column and index to col_sets
        // to apply head_computation on them as well.
        if (head_op && return_index) {
            // index-column
            col_sets.push_back(makeColSet(
                {index_col}, index_col, Bodo_FTypes::head, do_combine,
                skip_na_data, periods, {0}, n_udf, is_parallel, {false},
                {false}, {nullptr}, 0, udf_n_redvars, udf_table, udf_table_idx,
                nullptr, use_sql_rules));
            if (head_i) {
                col_sets.push_back(makeColSet(
                    {in_table->columns[in_table->columns.size() - 1]},
                    index_col, Bodo_FTypes::head, do_combine, skip_na_data,
                    periods, {0}, n_udf, is_parallel, {false}, {false},
                    {nullptr}, 0, udf_n_redvars, udf_table, udf_table_idx,
                    nullptr, use_sql_rules));
            }
        }
        in_table->id = 0;
        ev.add_attribute("g_shuffle_before_update",
                         static_cast<size_t>(shuffle_before_update));
        ev.add_attribute("g_do_combine", static_cast<size_t>(do_combine));
    }

    ~GroupbyPipeline() {
        if (hashes) {
            hashes.reset();
        }
    }
    /**
     * @brief
     * Create key-sort column used to sort table at the end.
     * Set its values and add as the last column in in_table.
     * Column values is in range(start, start+nrows).
     * Each rank will compute its range by identifying
     * start/end index of its set of rows.
     * @return ** void
     */
    void add_head_key_sort_column() {
        std::shared_ptr<array_info> head_sort_col =
            alloc_array_top_level(in_table->nrows(), 1, 1,
                                  bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
        int64_t num_ranks = dist_get_size();
        int64_t my_rank = dist_get_rank();
        // Gather the number of rows on every rank
        int64_t num_rows = in_table->nrows();
        std::vector<int64_t> num_rows_ranks(num_ranks);
        CHECK_MPI(
            MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                          MPI_INT64_T, MPI_COMM_WORLD),
            "GroupbyPipeline::add_head_key_sort_column: MPI error on "
            "MPI_Allgather:");

        // Determine the start/end row number of each rank
        int64_t rank_start_row, rank_end_row;
        rank_end_row = std::accumulate(num_rows_ranks.begin(),
                                       num_rows_ranks.begin() + my_rank + 1, 0);
        rank_start_row = rank_end_row - num_rows;
        // generate start/end range
        for (int64_t i = 0; i < num_rows; i++) {
            uint64_t& val =
                getv<uint64_t, bodo_array_type::NUMPY>(head_sort_col, i);
            val = rank_start_row + i;
        }
        in_table->columns.push_back(head_sort_col);
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    std::shared_ptr<table_info> run(int64_t* n_out_rows) {
        // If data is already shuffled (shuffle_before_update = True),
        // update() is a reduction on all of the data in the group.
        // Otherwise, update is performed on current chunk of data
        // and data will be shuffled after if it's parallel.
        update();
        if (shuffle_before_update) {
            if (in_table != orig_in_table) {
                // in_table is temporary table created in C++
                in_table.reset();
            }
        }
        if (is_parallel && !shuffle_before_update) {
            shuffle();
            combine();
        }
        eval();
        // For gb.head() operation, if data is distributed,
        // sort table based on head_sort_col column.
        if (head_op && is_parallel) {
            sort_gb_head_output();
        }
        return getOutputTable(n_out_rows);
    }
    /**
     * @brief
     * 1. Put head_sort_col at the beginning of the table.
     * 2. Sort table based on this column.
     * 3. Remove head_sort_col.
     */
    void sort_gb_head_output() {
        // Move sort column to the front.
        std::vector<std::shared_ptr<array_info>>::iterator pos =
            cur_table->columns.end() - 1;
        std::rotate(cur_table->columns.begin(), pos, pos + 1);
        // whether to put NaN first or last.
        // Does not matter in this case (no NaN, values are range(nrows))
        int64_t asc_pos = 1;
        int64_t zero = 0;
        cur_table = sort_values_table(cur_table, 1, &asc_pos, &asc_pos, &zero,
                                      nullptr, nullptr, is_parallel);
        // Remove key-sort column
        cur_table->columns.erase(cur_table->columns.begin());
    }

   private:
    int64_t compute_head_row_list(grouping_info const& grp_info,
                                  bodo::vector<int64_t>& head_row_list) {
        // keep track of how many rows found per group so far.
        bodo::vector<int64_t> nrows_per_grp(grp_info.num_groups);
        int64_t count = 0;  // how many rows found so far
        uint64_t iRow = 0;  // index looping over all rows
        for (iRow = 0; iRow < in_table->nrows(); iRow++) {
            int64_t igrp = grp_info.row_to_group[iRow];
            if (igrp != -1 && nrows_per_grp[igrp] < head_n) {
                nrows_per_grp[igrp]++;
                head_row_list.push_back(iRow);
                count++;
            }
        }
        return count;
    }
    /**
     * The update step groups rows in the input table based on keys, and
     * aggregates them based on the function to be applied to the columns.
     * More specifically, it will invoke the update method of each column set.
     */
    void update() {
        tracing::Event ev("update", is_parallel);
        std::vector<std::shared_ptr<table_info>> tables;
        // If nunique_only and nunique_tables.size() > 0 then all of the input
        // data is in nunique_tables
        if (!(nunique_only && nunique_tables.size() > 0)) {
            tables.push_back(in_table);
        }

        for (auto& nunique_table : nunique_tables) {
            tables.push_back(nunique_table.second);
        }

        if (req_extended_group_info) {
            const bool consider_missing = cumulative_op || shift_op ||
                                          transform_op || ngroup_op ||
                                          window_op;
            get_group_info_iterate(tables, hashes, nunique_hashes, grp_infos,
                                   this->num_keys, consider_missing, key_dropna,
                                   is_parallel);
        } else {
            get_group_info(tables, hashes, nunique_hashes, grp_infos,
                           this->num_keys, true, key_dropna, is_parallel);
        }
        grouping_info& grp_info = grp_infos[0];
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.mode = 1;
        num_groups = grp_info.num_groups;
        int64_t update_col_len = num_groups;
        bodo::vector<int64_t> head_row_list;
        if (head_op) {
            update_col_len = compute_head_row_list(grp_infos[0], head_row_list);
        }

        // Now if we have multiple tables, this step recombines them into
        // a single update table. There could be multiple tables if different
        // operations shuffle at different times. For example nunique + sum
        // in test_711.py e2e tests.
        update_table = cur_table = std::make_shared<table_info>();

        if (cumulative_op || shift_op || transform_op || head_op || ngroup_op ||
            window_op) {
            num_keys = 0;  // there are no key columns in output of cumulative
                           // operations
        } else {
            alloc_init_keys(tables, update_table, grp_infos, num_keys,
                            num_groups);
        }
        for (auto col_set : col_sets) {
            std::vector<std::shared_ptr<array_info>> list_arr;
            std::cout << "calling alloc update col" << std::endl;
            col_set->alloc_update_columns(update_col_len, list_arr);
            for (auto& e_arr : list_arr) {
                update_table->columns.push_back(e_arr);
            }
            auto head_col = std::dynamic_pointer_cast<HeadColSet>(col_set);
            if (head_col) {
                head_col->set_head_row_list(head_row_list);
            }
            col_set->update(grp_infos);
        }

        // gb.head() already added the index to the tables columns.
        // This is need to do head_computation on it as well.
        // since it will not be the same length as the in_table.
        if (!head_op && return_index) {
            update_table->columns.push_back(
                copy_array(in_table->columns.back()));
        }

        if (n_udf > 0) {
            int n_gen_udf = gen_udf_col_sets.size();
            std::cout << "n gen udfs " << n_gen_udf << std::endl;
            std::stringstream ss;
            DEBUG_PrintTable(ss, update_table);
            std::cout << ss.str() << std::endl;
            if (n_udf > n_gen_udf) {
                // regular UDFs
                udf_info.update(in_table.get(), update_table.get(),
                                grp_info.row_to_group.data());
            }
            if (n_gen_udf > 0) {
                std::shared_ptr<table_info> general_in_table =
                    std::make_shared<table_info>();
                for (auto udf_col_set : gen_udf_col_sets) {
                    udf_col_set->fill_in_columns(general_in_table, grp_info);
                }
                udf_info.general_udf(grp_info.num_groups,
                                     general_in_table.get(),
                                     update_table.get());
            }
        }
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        tracing::Event ev("shuffle", is_parallel);
        int64_t num_shuffle_keys = n_shuffle_keys;
        // If we do a reverse shuffle there is no benefit to keeping the shuffle
        // keys as the data doesn't stay shuffled. nunique is heavily optimized
        // so we cannot yet use a subset of keys to shuffle
        if (has_reverse_shuffle || nunique_op) {
            num_shuffle_keys = num_keys;
        }
        ev.add_attribute("passed_n_shuffle_keys", n_shuffle_keys);
        ev.add_attribute("num_shuffle_keys", num_shuffle_keys);
        std::shared_ptr<table_info> shuf_table = shuffle_table(
            std::move(update_table), num_shuffle_keys, is_parallel);

        update_table = cur_table = shuf_table;

        // update column sets with columns from shuffled table
        auto it = update_table->columns.begin() + num_keys;
        for (auto col_set : col_sets) {
            it = col_set->update_after_shuffle(it);
        }
    }

    /**
     * The combine step is performed after update and shuffle. It groups rows
     * in shuffled table based on keys, and aggregates them based on the
     * function to be applied to the columns. More specifically, it will invoke
     * the combine method of each column set.
     */
    void combine() {
        tracing::Event ev("combine", is_parallel);
        grp_infos.clear();
        std::vector<std::shared_ptr<table_info>> tables = {update_table};
        get_group_info(tables, hashes, nunique_hashes, grp_infos,
                       this->num_keys, false, key_dropna, is_parallel);
        grouping_info& grp_info = grp_infos[0];
        num_groups = grp_info.num_groups;
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.mode = 2;

        combine_table = cur_table = std::make_shared<table_info>();
        alloc_init_keys({update_table}, combine_table, grp_infos, num_keys,
                        num_groups);
        std::vector<std::shared_ptr<array_info>> list_arr;
        for (auto& col_set : col_sets) {
            std::vector<std::shared_ptr<array_info>> list_arr;
            col_set->alloc_combine_columns(num_groups, list_arr);
            for (auto& e_arr : list_arr) {
                combine_table->columns.push_back(e_arr);
            }
            col_set->combine(grp_info);
        }
        if (n_udf > 0) {
            udf_info.combine(update_table.get(), combine_table.get(),
                             grp_info.row_to_group.data());
        }
        update_table.reset();
    }

    /**
     * The eval step generates the final result (output column) for each column
     * set. It call the eval method of each column set.
     */
    void eval() {
        tracing::Event ev("eval", is_parallel);
        for (auto col_set : col_sets) {
            col_set->eval(grp_infos[0]);
        }
        // only regular UDFs need eval step
        if (n_udf - gen_udf_col_sets.size() > 0) {
            udf_info.eval(cur_table.get());
        }
    }

    /**
     * Returns the final output table which is the result of the groupby.
     */
    std::shared_ptr<table_info> getOutputTable(int64_t* n_out_rows) {
        if (maintain_input_size) {
            // These operations are all defined to maintain the same
            // length as the input.
            *n_out_rows = orig_in_table->nrows();
        } else {
            *n_out_rows = cur_table->nrows();
        }

        std::shared_ptr<table_info> out_table = std::make_shared<table_info>();

        if (return_key) {
            out_table->columns.assign(cur_table->columns.begin(),
                                      cur_table->columns.begin() + num_keys);
        }

        // gb.head() with distributed data sorted the table so col_sets no
        // longer reflects final
        // output columns.
        if (head_op && is_parallel) {
            for (uint64_t i = 0; i < cur_table->ncols(); i++) {
                out_table->columns.push_back(cur_table->columns[i]);
            }
        } else {
            for (std::shared_ptr<BasicColSet> col_set : col_sets) {
                const std::vector<std::shared_ptr<array_info>> out_cols =
                    col_set->getOutputColumns();
                out_table->columns.insert(out_table->columns.end(),
                                          out_cols.begin(), out_cols.end());
            }
            // gb.head() already added index to out_table.
            if (!head_op && return_index) {
                out_table->columns.push_back(cur_table->columns.back());
            }
        }

        if ((cumulative_op || shift_op || transform_op || ngroup_op ||
             window_op) &&
            is_parallel) {
            std::shared_ptr<table_info> revshuf_table =
                reverse_shuffle_table_kernel(std::move(out_table),
                                             *comm_info_ptr);
            in_hashes.reset();
            out_table = revshuf_table;
        }

        return out_table;
    }

    /**
     * We enter this algorithm at the beginning of the groupby pipeline, if
     * there are gb.nunique operations and there is no other operation that
     * requires shuffling before update. This algorithm decides, for each
     * nunique column, whether all ranks drop duplicates locally for that column
     * based on average local cardinality estimates across all ranks, and will
     * also decide how to shuffle all the nunique columns (it will use the same
     * scheme to shuffle all the nunique columns since the decision is not
     * based on the characteristics on any particular column). There are two
     * strategies for shuffling:
     * a) Shuffle based on groupby keys. Shuffles nunique data to its final
     *    destination. If there are no other groupby operations other than
     *    nunique then this equals shuffle_before_update=true and we just
     *    need an update and eval step (no second shuffle and combine). But
     *    if there are other operations mixed in, for simplicity we will do
     *    update, shuffle and combine step for nunique columns even though
     *    the nunique data is already in the final destination.
     * b) Shuffle based on keys+value. This is done if the number of *global*
     *    groups is small compared to the number of ranks, since shuffling
     *    based on keys in this case can generate significant load imbalance.
     *    In this case the update step calculates number of unique values
     *    for (key, value) tuples, the second shuffle (after update) collects
     *    the nuniques for a given group on the same rank, and the combine sums
     *    them.
     * @param ftypes: list of groupby function types passed directly from
     * GroupbyPipeline constructor.
     * @param num_funcs: number of functions in ftypes
     * @param nunique_hashes_global: estimated number of global unique hashes
     * of groupby keys (gives an estimate of global number of unique groups)
     */
    void gb_nunique_preprocess(int* ftypes, int num_funcs,
                               size_t nunique_hashes_global) {
        tracing::Event ev("gb_nunique_preprocess", is_parallel);
        if (!is_parallel) {
            throw std::runtime_error(
                "gb_nunique_preprocess called for non-distributed data");
        }
        if (shuffle_before_update) {
            throw std::runtime_error(
                "gb_nunique_preprocess called with shuffle_before_update=true");
        }

        // If it's just nunique we set table_id_counter to 0 because we won't
        // add in_table to our list of tables. Otherwise, set to 1 as 0 is
        // reserved for in_table
        int table_id_counter = 1;
        if (nunique_only) {
            table_id_counter = 0;
        }

        static constexpr float threshold_of_fraction_of_unique_hash = 0.5;
        ev.add_attribute("g_threshold_of_fraction_of_unique_hash",
                         threshold_of_fraction_of_unique_hash);

        // If the number of global groups is small we need to shuffle
        // based on keys *and* values to maximize data distribution
        // and improve scaling. If we only spread based on keys, scaling
        // will be limited by the number of groups.
        int num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        size_t num_ranks_unsigned = size_t(num_ranks);
        // When number of groups starts to approximate the number of ranks
        // there will be a high chance that a single rank ends up with 2-3
        // times the load (number of groups) than others after shuffling
        // TODO investigate what is the best threshold:
        // https://bodo.atlassian.net/browse/BE-1308
        const bool shuffle_by_keys_and_value =
            (nunique_hashes_global <= num_ranks_unsigned * 3);
        ev.add_attribute("g_nunique_shuffle_by_keys_and_values",
                         shuffle_by_keys_and_value);

        for (int i = 0, col_idx = num_keys; i < num_funcs; i++, col_idx++) {
            if (ftypes[i] != Bodo_FTypes::nunique) {
                continue;
            }

            std::shared_ptr<table_info> tmp = std::make_shared<table_info>();
            tmp->columns.assign(in_table->columns.begin(),
                                in_table->columns.begin() + num_keys);
            int8_t num_input_cols_used = ncols_per_func[i];
            if (num_input_cols_used != 1) {
                throw std::runtime_error(
                    "nunique can only be used with a "
                    "single column as input");
            }
            tmp->columns.push_back(in_table->columns[col_idx]);

            // --------- drop local duplicates ---------
            // If we know that the |set(values)| / len(values)
            // is low on all ranks then it should be beneficial to
            // drop local duplicates before the shuffle.

            const size_t n_rows = static_cast<size_t>(in_table->nrows());
            // get hashes of keys+value
            std::unique_ptr<uint32_t[]> key_value_hashes =
                std::make_unique<uint32_t[]>(n_rows);
            // TODO: Restore the memcpy?
            for (size_t i = 0; i < n_rows; i++) {
                key_value_hashes[i] = hashes[i];
            }
            // TODO: do a hash combine that writes to an empty hash
            // array to avoid memcpy?
            hash_array_combine(key_value_hashes.get(), tmp->columns[num_keys],
                               n_rows, SEED_HASH_PARTITION,
                               /*global_dict_needed=*/true, is_parallel);

            std::shared_ptr<uint32_t[]> shared_key_value_hashes =
                std::move(key_value_hashes);
            // Compute the local fraction of unique hashes
            size_t nunique_keyval_hashes = get_nunique_hashes(
                shared_key_value_hashes, n_rows, is_parallel);
            float local_fraction_unique_hashes =
                static_cast<float>(nunique_keyval_hashes) /
                static_cast<float>(n_rows);
            float global_fraction_unique_hashes;
            if (ev.is_tracing()) {
                ev.add_attribute("nunique_" + std::to_string(i) +
                                     "_local_fraction_unique_hashes",
                                 local_fraction_unique_hashes);
            }
            CHECK_MPI(MPI_Allreduce(&local_fraction_unique_hashes,
                                    &global_fraction_unique_hashes, 1,
                                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD),
                      "GroupbyPipeline::gb_nunique_preprocess: MPI "
                      "error on MPI_Allreduce:");
            global_fraction_unique_hashes /= static_cast<float>(num_ranks);
            ev.add_attribute("g_nunique_" + std::to_string(i) +
                                 "_global_fraction_unique_hashes",
                             global_fraction_unique_hashes);
            const bool drop_duplicates = global_fraction_unique_hashes <
                                         threshold_of_fraction_of_unique_hash;
            ev.add_attribute(
                "g_nunique_" + std::to_string(i) + "_drop_duplicates",
                drop_duplicates);

            std::shared_ptr<table_info> tmp2 = nullptr;
            if (drop_duplicates) {
                // Set dropna to false because skipna is handled at
                // a later step. Setting dropna=True here removes NA
                // from the keys, which we do not want
                tmp2 = drop_duplicates_table_inner(
                    tmp, tmp->ncols(), 0, 1, is_parallel, false,
                    /*drop_duplicates_dict=*/true, shared_key_value_hashes);
                tmp = tmp2;
            }

            // --------- shuffle column ---------
            if (shuffle_by_keys_and_value) {
                uint64_t ncols = tmp->ncols();
                if (drop_duplicates) {
                    // Note that tmp here no longer contains the
                    // original input arrays
                    tmp2 = shuffle_table(std::move(tmp), ncols, is_parallel);
                } else {
                    // Since the arrays are unmodified we can reuse the hashes
                    tmp2 =
                        shuffle_table(std::move(tmp), ncols, is_parallel, false,
                                      std::move(shared_key_value_hashes));
                }
            } else {
                if (drop_duplicates) {
                    tmp2 = shuffle_table(std::move(tmp), num_keys, is_parallel);
                } else {
                    tmp2 = shuffle_table(std::move(tmp), num_keys, is_parallel,
                                         false, hashes);
                }
            }
            shared_key_value_hashes.reset();
            tmp2->id = table_id_counter++;
            nunique_tables[col_idx] = tmp2;
        }

        if (!shuffle_by_keys_and_value && nunique_only) {
            // We have shuffled the data to its final destination so this is
            // equivalent to shuffle_before_update=true and we don't need to
            // do a combine step
            shuffle_before_update = true;
        }

        if (nunique_only) {
            // in the case of nunique_only the hashes that we calculated in
            // GroupbyPipeline() are not valid, since we have shuffled all of
            // the input columns
            hashes.reset();
        }
    }

    std::shared_ptr<table_info>
        orig_in_table;  // original input table of groupby received from Python
    std::shared_ptr<table_info> in_table;  // input table of groupby
    int64_t num_keys;
    int8_t* ncols_per_func;           // number of input columns per aggregation
    int8_t* n_window_calls_per_func;  // number of window function calls per
                                      // aggregation
    std::shared_ptr<table_info>
        dispatch_table;  // input dispatching table of pivot_table
    std::shared_ptr<table_info>
        dispatch_info;  // input dispatching info of pivot_table
    bool is_parallel;
    bool return_key;
    bool return_index;
    bool key_dropna;
    std::vector<std::shared_ptr<BasicColSet>> col_sets;
    std::vector<std::shared_ptr<GeneralUdfColSet>> gen_udf_col_sets;
    std::shared_ptr<table_info> udf_table;
    int* udf_n_redvars;
    // total number of UDFs applied to input columns (includes regular and
    // general UDFs)
    int n_udf = 0;
    int udf_table_idx = 0;
    // shuffling before update requires more communication and is needed
    // when one of the groupby functions is
    // median/nunique/cumsum/cumprod/cummin/cummax/shift/transform
    bool shuffle_before_update = false;
    bool cumulative_op = false;
    bool shift_op = false;
    bool transform_op = false;
    bool nunique_op = false;
    bool head_op = false;
    bool ngroup_op = false;
    bool window_op = false;
    bool has_reverse_shuffle = false;
    int64_t head_n;
    bool req_extended_group_info = false;
    bool do_combine;
    bool maintain_input_size;
    int64_t n_shuffle_keys;
    bool use_sql_rules;

    // column position in in_table -> table that contains key columns + one
    // nunique column after [dropping local duplicates] + shuffling
    std::map<int, std::shared_ptr<table_info>> nunique_tables;
    bool nunique_only = false;  // there are only groupby nunique operations

    udfinfo_t udf_info;

    std::shared_ptr<table_info> update_table = nullptr;
    std::shared_ptr<table_info> combine_table = nullptr;
    std::shared_ptr<table_info> cur_table = nullptr;

    std::vector<grouping_info> grp_infos;
    size_t num_groups;
    // shuffling stuff
    std::shared_ptr<uint32_t[]> in_hashes =
        std::shared_ptr<uint32_t[]>(nullptr);
    std::shared_ptr<mpi_comm_info> comm_info_ptr = nullptr;
    std::shared_ptr<uint32_t[]> hashes = std::shared_ptr<uint32_t[]>(nullptr);
    size_t nunique_hashes = 0;
};

table_info* groupby_and_aggregate_py_entry(
    table_info* input_table, int64_t num_keys, int8_t* ncols_per_func,
    int8_t* n_window_calls_per_func, int64_t num_funcs, bool input_has_index,
    int* ftypes, int* func_offsets, int* udf_nredvars, bool is_parallel,
    bool skip_na_data, int64_t periods, int64_t* transform_funcs,
    int64_t head_n, bool return_key, bool return_index, bool key_dropna,
    void* update_cb, void* combine_cb, void* eval_cb, void* general_udfs_cb,
    table_info* in_udf_dummy_table, int64_t* n_out_rows, bool* window_ascending,
    bool* window_na_position, table_info* window_args_,
    int8_t* n_window_args_per_func, int* n_input_cols_per_func,
    bool maintain_input_size, int64_t n_shuffle_keys, bool use_sql_rules) {
    try {
        std::shared_ptr<table_info> in_table =
            std::shared_ptr<table_info>(input_table);

        std::shared_ptr<table_info> udf_dummy_table =
            std::shared_ptr<table_info>(in_udf_dummy_table);

        std::shared_ptr<table_info> window_args(window_args_);

        tracing::Event ev("groupby_and_aggregate", is_parallel);
        int strategy =
            determine_groupby_strategy(in_table, num_keys, ncols_per_func,
                                       num_funcs, ftypes, input_has_index);
        ev.add_attribute("g_strategy", strategy);

        auto implement_strategy0 = [&]() -> table_info* {
            std::shared_ptr<table_info> dispatch_info = nullptr;
            std::shared_ptr<table_info> dispatch_table = nullptr;
            GroupbyPipeline groupby(
                in_table, num_keys, ncols_per_func, n_window_calls_per_func,
                num_funcs, dispatch_table, dispatch_info, input_has_index,
                is_parallel, ftypes, func_offsets, udf_nredvars,
                udf_dummy_table, (udf_table_op_fn)update_cb,
                (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb,
                (udf_general_fn)general_udfs_cb, skip_na_data, periods,
                transform_funcs, head_n, return_key, return_index, key_dropna,
                window_ascending, window_na_position, window_args,
                n_window_args_per_func, n_input_cols_per_func,
                maintain_input_size, n_shuffle_keys, use_sql_rules);

            std::shared_ptr<table_info> ret_table = groupby.run(n_out_rows);
            return new table_info(*ret_table);
        };
        auto implement_categorical_exscan =
            [&](std::shared_ptr<array_info> cat_column) -> table_info* {
            std::shared_ptr<table_info> ret_table =
                mpi_exscan_computation(cat_column, in_table, num_keys, ftypes,
                                       func_offsets, is_parallel, skip_na_data,
                                       return_key, return_index, use_sql_rules);
            *n_out_rows = in_table->nrows();
            return new table_info(*ret_table);
        };
        if (strategy == 0) {
            return implement_strategy0();
        }
        if (strategy == 1) {
            std::shared_ptr<array_info> cat_column = in_table->columns[0];
            return implement_categorical_exscan(cat_column);
        }
        if (strategy == 2) {
            std::shared_ptr<array_info> cat_column = compute_categorical_index(
                in_table, num_keys, is_parallel, key_dropna);
            if (cat_column ==
                nullptr) {  // It turns out that there are too many
                            // different keys for exscan to be ok.
                return implement_strategy0();
            } else {
                return implement_categorical_exscan(cat_column);
            }
        }
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
