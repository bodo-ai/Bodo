#pragma once

#include <../../bodo/libs/_bodo_common.h>
#include <arrow/scalar.h>
#include "../groupby/_groupby_ftypes.h"
#include "_groupby.h"

#ifdef USE_CUDF
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include "../gpu_utils.h"

struct FinalMerge {
    std::vector<size_t> column_indices;  // indices into the input table
    std::unique_ptr<cudf::column> (*fn)(const std::vector<cudf::column_view> &);
};

std::unique_ptr<cudf::column> mean_final_merge(
    const std::vector<cudf::column_view> &input_cols);
std::unique_ptr<cudf::column> var_final_merge(
    const std::vector<cudf::column_view> &input_cols);
std::unique_ptr<cudf::column> std_final_merge(
    const std::vector<cudf::column_view> &input_cols);
std::unique_ptr<cudf::column> skew_final_merge(
    const std::vector<cudf::column_view> &input_cols);
std::unique_ptr<cudf::column> distinct_final_merge(
    const std::vector<cudf::column_view> &input_cols);

std::unique_ptr<cudf::column> square_col(const cudf::column_view &input_col);
std::unique_ptr<cudf::column> cubed_col(const cudf::column_view &input_col);

std::unique_ptr<cudf::table> apply_final_merges(
    cudf::table_view const &input, std::vector<FinalMerge> const &merges);

class CudaGroupbyState {
   private:
    bool all_local_done = false;

    std::vector<uint64_t> key_indices;
    std::vector<uint64_t> column_indices;
    std::vector<cudf::groupby::aggregation_request> aggregation_requests;
    std::vector<std::unique_ptr<cudf::column> (*)(const cudf::column_view &)>
        aggregation_fns;

    std::vector<cudf::size_type> shuffle_key_indices;
    std::vector<uint64_t> merge_key_indices;
    std::vector<uint64_t> merge_column_indices;
    std::vector<cudf::groupby::aggregation_request> merge_aggregation_requests;
    std::vector<std::unique_ptr<cudf::column> (*)(const cudf::column_view &)>
        merge_aggregation_fns;

    std::unique_ptr<cudf::table> accumulation;
    std::vector<FinalMerge> final_merges;

    static std::unique_ptr<cudf::table> do_groupby(
        cudf::table_view const &input, std::vector<uint64_t> &key_indices,
        std::vector<uint64_t> &column_indices,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        std::vector<std::unique_ptr<cudf::column> (*)(
            const cudf::column_view &)> &aggregation_fns,
        rmm::cuda_stream_view &stream);

    void add_agg_entry(
        std::unique_ptr<cudf::groupby_aggregation> agg,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        std::vector<std::unique_ptr<cudf::column> (*)(
            const cudf::column_view &)> &aggregation_fns,
        std::unique_ptr<cudf::column> (*fn)(const cudf::column_view &) =
            nullptr) {
        cudf::groupby::aggregation_request req;
        req.aggregations.push_back(std::move(agg));
        aggregation_requests.push_back(std::move(req));
        aggregation_fns.push_back(fn);
    }

    GpuShuffleManager merge_shuffler;

    void bodo_agg_to_cudf(
        uint64_t ftype,
        std::vector<cudf::groupby::aggregation_request> &aggregation_requests,
        std::vector<std::unique_ptr<cudf::column> (*)(
            const cudf::column_view &)> &aggregation_fns) {
        switch (ftype) {
            case Bodo_FTypes::sum:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::min:
                add_agg_entry(
                    cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::max:
                add_agg_entry(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::count:
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::size:
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::INCLUDE),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::mean:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, square_col);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::skew:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, square_col);
                // For sum of the cube of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns, cubed_col);
                add_agg_entry(
                    cudf::make_count_aggregation<cudf::groupby_aggregation>(
                        cudf::null_policy::EXCLUDE),
                    aggregation_requests, aggregation_fns);
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
        std::vector<std::unique_ptr<cudf::column> (*)(
            const cudf::column_view &)> &aggregation_fns) {
        switch (ftype) {
            case Bodo_FTypes::sum:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::min:
                add_agg_entry(
                    cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::max:
                add_agg_entry(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::count:
            case Bodo_FTypes::size:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::mean:
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // merging of counts is summation
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                break;
            case Bodo_FTypes::skew:
                // For sum of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // For sum of the square of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                // For sum of the cube of the column.
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
                add_agg_entry(
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    aggregation_requests, aggregation_fns);
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
            default:
                throw std::runtime_error(
                    "Cannot convert Bodo agg type to cudf in addFinalMerge" +
                    std::to_string(ftype));
        }
    }

   public:
    CudaGroupbyState(
        const std::vector<uint64_t> &_key_indices,
        const std::vector<std::pair<uint64_t, int32_t>> &column_agg_funcs)
        : key_indices(_key_indices) {
        unsigned num_keys = key_indices.size();

        if (column_agg_funcs.size() == 0) {
            // Used for distinct/drop_duplicates.
            add_agg_entry(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(
                    cudf::null_policy::INCLUDE),
                aggregation_requests, aggregation_fns);

            // Find lowest number column that isn't a key.
            bool found_col = false;
            uint64_t col_to_use = 0;
            while (!found_col) {
                found_col = true;
                for (size_t i = 0; i < _key_indices.size(); ++i) {
                    if (_key_indices[i] == col_to_use) {
                        found_col = false;
                        break;
                    }
                }
                if (!found_col) {
                    col_to_use++;
                }
            }
            column_indices.push_back(col_to_use);

            add_agg_entry(
                cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                merge_aggregation_requests, merge_aggregation_fns);

            final_merges.push_back({{num_keys}, distinct_final_merge});
        } else {
            // Create as much as we can of the aggregation info.
            // Each batch overwrites the column with the given batch's column
            // view.
            for (auto &column_agg_func : column_agg_funcs) {
                bodo_agg_to_cudf(column_agg_func.second, aggregation_requests,
                                 aggregation_fns);
                for (size_t i = 0; i < aggregation_requests.size(); ++i) {
                    column_indices.push_back(column_agg_func.first);
                }
                bodo_agg_to_merge_cudf(column_agg_func.second,
                                       merge_aggregation_requests,
                                       merge_aggregation_fns);
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
                             bool is_last, rmm::cuda_stream_view &stream);

    /**
     * @brief Function to produce an output table.
     *
     * @param[out] out_is_last is last batch
     * @param produce_output whether to produce output
     * @return cudf::table output table batch
     */
    std::unique_ptr<cudf::table> produce_output_batch(bool &out_is_last,
                                                      bool produce_output);
};

#else   // USE_CUDF
struct CudaGroupbyState {};
#endif  // USE_CUDF
