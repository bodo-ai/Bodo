#pragma once

#include <../../bodo/libs/_bodo_common.h>
#include <arrow/scalar.h>
#include "../gpu_utils.h"
#include "../groupby/_groupby_ftypes.h"
#include "_groupby.h"

#ifdef USE_CUDF
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

struct FinalMerge {
    std::vector<size_t> column_indices;  // indices into the input table
    std::unique_ptr<cudf::column> (*fn)(const std::vector<cudf::column_view> &,
                                        rmm::cuda_stream_view &);
};

// --------- Prototypes for converting possibly multiple cols into final result.
std::unique_ptr<cudf::column> mean_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> var_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> std_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> skew_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> distinct_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> nunique_final_merge(
    const std::vector<cudf::column_view> &input_cols,
    rmm::cuda_stream_view &output_stream);

// --------- Prototypes for things to do to columns before aggregation.
std::unique_ptr<cudf::column> square_col(const cudf::column_view &input_col,
                                         rmm::cuda_stream_view &output_stream);
std::unique_ptr<cudf::column> cubed_col(const cudf::column_view &input_col,
                                        rmm::cuda_stream_view &output_stream);

// --------- Prototypes for things to do to columns after aggregation.
std::unique_ptr<cudf::column> cast_int64(const cudf::column_view &input_col,
                                         rmm::cuda_stream_view &output_stream);

// --------- Prototypes for things to do to tables before aggregation.
std::unique_ptr<cudf::table> nunique_pre_agg(
    const cudf::table_view &tv, cudf::size_type col_index,
    rmm::cuda_stream_view &output_stream);

std::unique_ptr<cudf::table> apply_final_merges(
    cudf::table_view const &input, std::vector<FinalMerge> const &merges,
    rmm::cuda_stream_view &output_stream);

using col_to_col_fn = std::unique_ptr<cudf::column> (*)(
    const cudf::column_view &, rmm::cuda_stream_view &);

using col_to_col_fn_vec = std::vector<col_to_col_fn>;

using tbl_to_tbl_fn = std::unique_ptr<cudf::table> (*)(const cudf::table_view &,
                                                       cudf::size_type,
                                                       rmm::cuda_stream_view &);

using tbl_to_tbl_fn_vec = std::vector<tbl_to_tbl_fn>;

class CudaGroupbyState {
   private:
    bool all_local_done = false;

    std::shared_ptr<arrow::Schema> output_schema;

    std::vector<uint64_t> key_indices;
    std::vector<uint64_t> column_indices;
    std::vector<cudf::groupby::aggregation_request> aggregation_requests;
    col_to_col_fn_vec aggregation_fns;
    col_to_col_fn_vec post_agg_fns;
    tbl_to_tbl_fn_vec pre_agg_table_fns;

    std::vector<cudf::size_type> shuffle_key_indices;
    std::vector<uint64_t> merge_key_indices;
    std::vector<uint64_t> merge_column_indices;
    std::vector<cudf::groupby::aggregation_request> merge_aggregation_requests;
    col_to_col_fn_vec merge_aggregation_fns;
    col_to_col_fn_vec post_merge_agg_fns;
    tbl_to_tbl_fn_vec pre_merge_agg_table_fns;

    std::unique_ptr<cudf::table> accumulation;
    std::vector<FinalMerge> final_merges;

    static std::unique_ptr<cudf::table> do_groupby(
        cudf::table_view const &input, std::vector<uint64_t> &key_indices,
        std::vector<uint64_t> &column_indices,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        col_to_col_fn_vec &aggregation_fns, col_to_col_fn_vec &post_agg_fns,
        tbl_to_tbl_fn_vec &pre_agg_table_fns, rmm::cuda_stream_view &stream);

    void add_agg_entry(
        std::unique_ptr<cudf::groupby_aggregation> agg,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        col_to_col_fn_vec &aggregation_fns,
        col_to_col_fn_vec &post_aggregation_fns,
        tbl_to_tbl_fn_vec &pre_aggregation_table_fns,
        col_to_col_fn fn = nullptr, col_to_col_fn post_agg_fn = nullptr,
        tbl_to_tbl_fn pre_agg_table_fn = nullptr) {
        cudf::groupby::aggregation_request req;
        req.aggregations.push_back(std::move(agg));
        aggregation_requests.push_back(std::move(req));
        aggregation_fns.push_back(fn);
        post_aggregation_fns.push_back(post_agg_fn);
        pre_aggregation_table_fns.push_back(pre_agg_table_fn);
    }

    GpuShuffleManager merge_shuffler;

    void bodo_agg_to_cudf(
        uint64_t ftype,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        col_to_col_fn_vec &aggregation_fns, col_to_col_fn_vec &post_agg_fns,
        tbl_to_tbl_fn_vec &pre_aggregation_table_fns) {
        switch (ftype) {
            case Bodo_FTypes::sum:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::min:
                add_agg_entry(
                    cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::max:
                add_agg_entry(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::count:
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, cast_int64);
                break;
            case Bodo_FTypes::size:
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::INCLUDE),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, cast_int64);
                break;
            case Bodo_FTypes::mean:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, cast_int64);
                break;
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, square_col);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, cast_int64);
                break;
            case Bodo_FTypes::skew:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, square_col);
                // For sum of the cube of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, cubed_col);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, cast_int64);
                break;
            case Bodo_FTypes::nunique:
                add_agg_entry(
                    cudf::make_collect_set_aggregation<
                        cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE,
                                                   cudf::null_equality::UNEQUAL,
                                                   cudf::nan_equality::UNEQUAL),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            default:
                throw std::runtime_error(
                    "Cannot convert Bodo agg type to cudf in "
                    "bodo_agg_to_cudf " +
                    std::to_string(ftype));
        }
    }

    void bodo_agg_to_merge_cudf(
        uint64_t ftype,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        col_to_col_fn_vec &aggregation_fns, col_to_col_fn_vec &post_agg_fns,
        tbl_to_tbl_fn_vec &pre_aggregation_table_fns) {
        switch (ftype) {
            case Bodo_FTypes::sum:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::min:
                add_agg_entry(
                    cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::max:
                add_agg_entry(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::count:
            case Bodo_FTypes::size:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::mean:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // merging of counts is summation
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::skew:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                // For sum of the cube of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns);
                break;
            case Bodo_FTypes::nunique:
                add_agg_entry(
                    cudf::make_collect_set_aggregation<
                        cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE,
                                                   cudf::null_equality::UNEQUAL,
                                                   cudf::nan_equality::UNEQUAL),
                    aggregation_requests, aggregation_fns, post_agg_fns,
                    pre_aggregation_table_fns, nullptr, nullptr,
                    nunique_pre_agg);
                break;
            default:
                throw std::runtime_error(
                    "Cannot convert Bodo agg type to cudf in "
                    "bodo_agg_to_merge_cudf " +
                    std::to_string(ftype));
        }
    }

    void addFinalMerge(uint64_t ftype, size_t cur_col_size) {
        switch (ftype) {
            case Bodo_FTypes::sum:
            case Bodo_FTypes::min:
            case Bodo_FTypes::max:
            case Bodo_FTypes::count:
            case Bodo_FTypes::size:
                break;
            case Bodo_FTypes::mean:
                final_merges.push_back(
                    {{cur_col_size - 2, cur_col_size - 1}, mean_final_merge});
                break;
            case Bodo_FTypes::var:
                final_merges.push_back(
                    {{cur_col_size - 3, cur_col_size - 2, cur_col_size - 1},
                     var_final_merge});
                break;
            case Bodo_FTypes::std:
                final_merges.push_back(
                    {{cur_col_size - 3, cur_col_size - 2, cur_col_size - 1},
                     std_final_merge});
                break;
            case Bodo_FTypes::skew:
                final_merges.push_back({{cur_col_size - 4, cur_col_size - 3,
                                         cur_col_size - 2, cur_col_size - 1},
                                        skew_final_merge});
                break;
            case Bodo_FTypes::nunique:
                final_merges.push_back(
                    {{cur_col_size - 1}, nunique_final_merge});
                break;
            default:
                throw std::runtime_error(
                    "Cannot convert Bodo agg type to cudf in addFinalMerge" +
                    std::to_string(ftype));
        }
    }

   public:
    CudaGroupbyState(
        const std::vector<uint64_t> &_key_indices,
        const std::vector<std::pair<uint64_t, int32_t>> &column_agg_funcs,
        std::shared_ptr<arrow::Schema> _output_schema)
        : output_schema(_output_schema), key_indices(_key_indices) {
        unsigned num_keys = key_indices.size();

        if (column_agg_funcs.size() == 0) {
            // Used for distinct/drop_duplicates.
            bodo_agg_to_cudf(Bodo_FTypes::size, aggregation_requests,
                             aggregation_fns, post_agg_fns, pre_agg_table_fns);
            column_indices.push_back(0);
            bodo_agg_to_merge_cudf(Bodo_FTypes::size,
                                   merge_aggregation_requests,
                                   merge_aggregation_fns, post_merge_agg_fns,
                                   pre_merge_agg_table_fns);
            final_merges.push_back({{num_keys}, distinct_final_merge});
        } else {
            // Create as much as we can of the aggregation info.
            // Each batch overwrites the column with the given batch's column
            // view.
            for (auto &column_agg_func : column_agg_funcs) {
                size_t agg_req_start_size = aggregation_requests.size();
                bodo_agg_to_cudf(column_agg_func.second, aggregation_requests,
                                 aggregation_fns, post_agg_fns,
                                 pre_agg_table_fns);
                size_t new_agg_req =
                    aggregation_requests.size() - agg_req_start_size;

                for (size_t i = 0; i < new_agg_req; ++i) {
                    column_indices.push_back(column_agg_func.first);
                }
                bodo_agg_to_merge_cudf(
                    column_agg_func.second, merge_aggregation_requests,
                    merge_aggregation_fns, post_merge_agg_fns,
                    pre_merge_agg_table_fns);
                addFinalMerge(column_agg_func.second,
                              num_keys + column_indices.size());
            }
        }

        for (size_t i = 0; i < num_keys; ++i) {
            merge_key_indices.push_back(i);
            shuffle_key_indices.push_back(i);
        }

        for (size_t i = 0; i < column_indices.size(); ++i) {
            merge_column_indices.push_back(num_keys + i);
        }
    }

    void all_local_data_processed() {
        if (!all_local_done) {
            all_local_done = true;
            merge_shuffler.complete();
        }
    }

    bool all_complete() {
        return all_local_done && merge_shuffler.all_complete();
    }

    /**
     * @brief Logic to consume a build table batch.
     *
     * @param in_table build table batch
     * @param is_last is last batch (in this pipeline) locally
     * Union-Distinct case where this is called in multiple pipelines. For
     * regular groupby, this should always be true. We only call FinalizeBuild
     * in the last pipeline.
     */
    void build_consume_batch(std::shared_ptr<cudf::table> input_table,
                             bool is_last, rmm::cuda_stream_view &output_stream,
                             std::shared_ptr<StreamAndEvent> input_se);

    /**
     * @brief Function to produce an output table.
     *
     * @param[out] out_is_last is last batch
     * @param produce_output whether to produce output
     * @return cudf::table output table batch
     */
    std::unique_ptr<cudf::table> produce_output_batch(
        bool &out_is_last, bool produce_output,
        rmm::cuda_stream_view &output_stream);
};

#else   // USE_CUDF
struct CudaGroupbyState {};
#endif  // USE_CUDF
