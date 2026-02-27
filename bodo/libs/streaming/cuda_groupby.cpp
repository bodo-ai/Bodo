#include "cuda_groupby.h"
#include "_groupby.h"

#include <fmt/format.h>
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
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
#include "_util.h"
#include "arrow/util/bit_util.h"

#define MAX_SHUFFLE_TABLE_SIZE 50 * 1024 * 1024
#define MAX_SHUFFLE_HASHTABLE_SIZE 50 * 1024 * 1024

std::unique_ptr<cudf::table> CudaGroupbyState::do_groupby(
    cudf::table_view const& _input, std::vector<uint64_t>& key_indices,
    std::vector<uint64_t>& column_indices,
    std::vector<cudf::groupby::aggregation_request>& aggregation_requests,
    col_to_col_fn_vec& aggregation_fns, col_to_col_fn_vec& post_agg_fns,
    tbl_to_tbl_fn_vec& pre_agg_table_fns, rmm::cuda_stream_view& stream) {
    cudf::table_view input = _input;

    // Aggregations like nunique need to do whole table operations before
    // the code below can do the merge.  For example, nunique will explode
    // a column containing a list to multiple rows in the table.
    std::unique_ptr<cudf::table> hold_updated_table;
    for (size_t i = 0; i < pre_agg_table_fns.size(); ++i) {
        if (pre_agg_table_fns[i] != nullptr) {
            hold_updated_table =
                pre_agg_table_fns[i](input, key_indices.size() + i, stream);
            input = hold_updated_table->view();
        }
    }

    // Get the key columns from the input table.
    std::vector<cudf::column_view> key_columns;
    for (auto key_col : key_indices) {
        key_columns.push_back(input.column(key_col));
    }

    // Create a view with just the key columns.
    cudf::table_view keys{key_columns};
    // Build the groupby object
    cudf::groupby::groupby gb_obj(keys, cudf::null_policy::EXCLUDE);

    std::vector<std::unique_ptr<cudf::column>> fn_cols;
    // Put the column view for the aggregations into aggregation_requests.
    for (size_t i = 0; i < column_indices.size(); ++i) {
        cudf::column_view col_to_use = input.column(column_indices[i]);
        // Some aggregations like variance needs to things like square
        // the column before aggregating with it.
        if (aggregation_fns[i] != nullptr) {
            fn_cols.push_back(aggregation_fns[i](col_to_use, stream));
            col_to_use = fn_cols[fn_cols.size() - 1]->view();
        }
        aggregation_requests[i].values = col_to_use;
    }

    // Run the groupby
    std::pair<std::unique_ptr<cudf::table>,
              std::vector<cudf::groupby::aggregation_result>>
        result = gb_obj.aggregate(aggregation_requests, stream);

    // result.first  = grouped keys table
    // result.second = vector<aggregation_result>
    // Each aggregation_result is a vector<column>.

    // stitch keys + results together
    std::vector<std::unique_ptr<cudf::column>> cols;

    // Move grouped keys
    for (auto& c : result.first->release()) {
        cols.push_back(std::move(c));
    }

    size_t cur_col = 0;
    // Move aggregated values
    for (auto& agg_result : result.second) {
        for (auto& c : agg_result.results) {
            // Right now this is only used to do casting of column
            // types when the local groupby and merge groupby
            // produce different column types.
            if (post_agg_fns[cur_col] != nullptr) {
                c = post_agg_fns[cur_col](c->view(), stream);
            }
            cols.push_back(std::move(c));
            cur_col++;
        }
    }
    if (cur_col != post_agg_fns.size()) {
        throw std::runtime_error(
            "do_groupby agg columns added not same size as post_agg_fns.");
    }

    return std::make_unique<cudf::table>(std::move(cols));
}

void CudaGroupbyState::build_consume_batch(
    std::shared_ptr<cudf::table> input_table, bool is_last,
    rmm::cuda_stream_view& output_stream,
    std::shared_ptr<StreamAndEvent> input_se) {
    // During the phase where all_complete() is not true, we'll get
    // blank tables, on which we don't need to run the first groupby
    // pass.
    if (input_table->view().num_rows() != 0) {
        std::shared_ptr<StreamAndEvent> local_groupby_se =
            make_stream_and_event(g_use_async);
        input_se->event.wait(local_groupby_se->stream);
        std::shared_ptr<cudf::table> new_data = std::move(
            do_groupby(input_table->view(), key_indices, column_indices,
                       aggregation_requests, aggregation_fns, post_agg_fns,
                       pre_agg_table_fns, local_groupby_se->stream));
        local_groupby_se->event.record(local_groupby_se->stream);
        merge_shuffler.shuffle_table(new_data, shuffle_key_indices,
                                     local_groupby_se->event);
    }

    // Give shuffler a chance to receive chunks.
    std::vector<std::unique_ptr<cudf::table>> shuffled_merge_chunks =
        merge_shuffler.progress();

    std::vector<cudf::table_view> views;

    // If we have already accumulated on this node then add the result
    // of that to the new shuffled tables to merge.
    if (accumulation) {
        views.emplace_back(accumulation->view());
    }

    // Add all the shuffled chunks to the merge set.
    for (auto& merge_chunk : shuffled_merge_chunks) {
        views.emplace_back(merge_chunk->view());
    }

    if (views.size() == 0) {
        return;
    }

    // Make one table out of all the views.
    auto combined = cudf::concatenate(views, output_stream);
    // Do the groupby on the combined table.
    accumulation = std::move(
        do_groupby(combined->view(), merge_key_indices, merge_column_indices,
                   merge_aggregation_requests, merge_aggregation_fns,
                   post_merge_agg_fns, pre_merge_agg_table_fns, output_stream));
}

std::unique_ptr<cudf::table> CudaGroupbyState::produce_output_batch(
    bool& out_is_last, bool produce_output,
    rmm::cuda_stream_view& output_stream) {
    out_is_last = true;

    // Will happen on non-GPU ranks.
    if (accumulation == nullptr) {
        return empty_table_from_arrow_schema(output_schema);
    }

    // accumulation is guaranteed here to have all the merged data from
    // all the other nodes.  If there are any final merges to collapse
    // multiple columns back down to one then run them here.
    if (final_merges.size() > 0) {
        accumulation = apply_final_merges(accumulation->view(), final_merges,
                                          output_stream);
    }

    return std::move(accumulation);
}

std::unique_ptr<cudf::table> apply_final_merges(
    cudf::table_view const& input, std::vector<FinalMerge> const& merges,
    rmm::cuda_stream_view& output_stream) {
    int ncols = input.num_columns();

    // Build a quick lookup: for each column index, which merge (if any) owns
    // it?
    std::vector<int> merge_owner(ncols, -1);

    for (size_t m = 0; m < merges.size(); ++m) {
        for (size_t idx : merges[m].column_indices) {
            merge_owner[idx] = m;
        }
    }

    // Precompute merged columns
    std::vector<std::unique_ptr<cudf::column>> merged_results(merges.size());

    for (size_t m = 0; m < merges.size(); ++m) {
        auto const& fm = merges[m];

        std::vector<cudf::column_view> views;
        views.reserve(fm.column_indices.size());

        for (size_t idx : fm.column_indices) {
            views.push_back(input.column(idx));
        }

        merged_results[m] = fm.fn(views, output_stream);
    }

    // Build output columns
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(ncols);  // upper bound

    int i = 0;
    while (i < ncols) {
        int m = merge_owner[i];

        if (m == -1) {
            // Not part of any merge → copy column
            out_cols.push_back(std::make_unique<cudf::column>(input.column(i)));
            i += 1;
        } else {
            // distinct/drop_duplicates will return no data for the merged
            // result and in that case we just drop the column
            if (merged_results[m] != nullptr) {
                // Part of merge m → insert merged column at this position
                out_cols.push_back(std::move(merged_results[m]));
            }

            // Skip all columns consumed by this merge
            int last = merges[m].column_indices.back();
            i = last + 1;
        }
    }

    return std::make_unique<cudf::table>(std::move(out_cols));
}

std::unique_ptr<cudf::column> mean_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    if (input_cols.size() != 2) {
        throw std::runtime_error("mean_final_merge didn't get 2 columns.");
    }

    auto const& sum_col = input_cols[0];
    auto const& count_col = input_cols[1];

    // The output type is float64 unless you want to preserve the input type.
    // Using float64 is safest for mean.
    cudf::data_type out_type{cudf::type_id::FLOAT64};

    // Perform elementwise division: sum / count
    auto mean_col =
        cudf::binary_operation(sum_col, count_col, cudf::binary_operator::DIV,
                               out_type, output_stream);

    return mean_col;
}

std::unique_ptr<cudf::column> distinct_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    return nullptr;  // distinct just drops the fake count col
}

std::unique_ptr<cudf::column> var_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    if (input_cols.size() != 3) {
        throw std::runtime_error("var_final_merge didn't get 3 columns.");
    }

    auto const& sum_col = input_cols[0];
    auto const& sumsq_col = input_cols[1];
    auto const& count_col = input_cols[2];

    cudf::data_type out_type{cudf::type_id::FLOAT64};

    auto sum_sq =
        cudf::binary_operation(sum_col, sum_col, cudf::binary_operator::MUL,
                               sum_col.type(),  // preserve dtype of sum_col
                               output_stream);

    auto sum_sq_div_n = cudf::binary_operation(sum_sq->view(), count_col,
                                               cudf::binary_operator::DIV,
                                               out_type, output_stream);

    auto numerator = cudf::binary_operation(sumsq_col, sum_sq_div_n->view(),
                                            cudf::binary_operator::SUB,
                                            out_type, output_stream);

    auto one_scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(1);
    auto denom = cudf::binary_operation(count_col, *one_scalar,
                                        cudf::binary_operator::SUB, out_type,
                                        output_stream);

    auto variance = cudf::binary_operation(numerator->view(), denom->view(),
                                           cudf::binary_operator::DIV, out_type,
                                           output_stream);

    return variance;
}

std::unique_ptr<cudf::column> std_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    if (input_cols.size() != 3) {
        throw std::runtime_error("std_final_merge didn't get 3 columns.");
    }

    auto variance = var_final_merge(input_cols, output_stream);

    auto stddev = cudf::unary_operation(
        variance->view(), cudf::unary_operator::SQRT, output_stream);

    return stddev;
}

std::unique_ptr<cudf::column> skew_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    if (input_cols.size() != 4) {
        throw std::runtime_error("skew_final_merge didn't get 4 columns.");
    }

    auto const& sum_col = input_cols[0];      // S
    auto const& sumsq_col = input_cols[1];    // Q
    auto const& sumcube_col = input_cols[2];  // M
    auto const& count_col = input_cols[3];    // N

    cudf::data_type out_type{cudf::type_id::FLOAT64};

    // ------------ means -------------
    auto mean =
        cudf::binary_operation(sum_col, count_col, cudf::binary_operator::DIV,
                               out_type, output_stream);

    auto mean_squared = cudf::binary_operation(mean->view(), mean->view(),
                                               cudf::binary_operator::MUL,
                                               out_type, output_stream);

    auto mean_cubed = cudf::binary_operation(mean_squared->view(), mean->view(),
                                             cudf::binary_operator::MUL,
                                             out_type, output_stream);

    // ------------- population mean 2 --------------

    auto Q_over_N =
        cudf::binary_operation(sumsq_col, count_col, cudf::binary_operator::DIV,
                               out_type, output_stream);

    auto mu2 = cudf::binary_operation(Q_over_N->view(), mean_squared->view(),
                                      cudf::binary_operator::SUB, out_type,
                                      output_stream);

    // ------------- population mean 3 --------------

    auto one_scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(1);
    auto two_scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(2);
    auto three_scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(3);

    auto M_over_N = cudf::binary_operation(sumcube_col, count_col,
                                           cudf::binary_operator::DIV, out_type,
                                           output_stream);

    auto three_mu = cudf::binary_operation(mean->view(), *three_scalar,
                                           cudf::binary_operator::MUL, out_type,
                                           output_stream);

    auto three_mu_Q_over_N = cudf::binary_operation(
        three_mu->view(), Q_over_N->view(), cudf::binary_operator::MUL,
        out_type, output_stream);

    auto two_mu3 = cudf::binary_operation(mean_cubed->view(), *two_scalar,
                                          cudf::binary_operator::MUL, out_type,
                                          output_stream);

    auto mu3_tmp = cudf::binary_operation(
        M_over_N->view(), three_mu_Q_over_N->view(), cudf::binary_operator::SUB,
        out_type, output_stream);

    auto mu3 = cudf::binary_operation(mu3_tmp->view(), two_mu3->view(),
                                      cudf::binary_operator::ADD, out_type,
                                      output_stream);

    // ---------- convert population to sample ------------

    auto N_minus_1 = cudf::binary_operation(count_col, *one_scalar,
                                            cudf::binary_operator::SUB,
                                            count_col.type(), output_stream);

    auto N_minus_2 = cudf::binary_operation(count_col, *two_scalar,
                                            cudf::binary_operator::SUB,
                                            count_col.type(), output_stream);

    // m2 = (N / (N-1)) * μ2
    auto N_over_Nm1 = cudf::binary_operation(count_col, N_minus_1->view(),
                                             cudf::binary_operator::DIV,
                                             out_type, output_stream);

    auto m2 = cudf::binary_operation(mu2->view(), N_over_Nm1->view(),
                                     cudf::binary_operator::MUL, out_type,
                                     output_stream);

    // m3 = (N^2 / ((N-1)(N-2))) * μ3
    auto N_sq =
        cudf::binary_operation(count_col, count_col, cudf::binary_operator::MUL,
                               out_type, output_stream);

    auto Nm1_Nm2 = cudf::binary_operation(N_minus_1->view(), N_minus_2->view(),
                                          cudf::binary_operator::MUL, out_type,
                                          output_stream);

    auto Nsq_over = cudf::binary_operation(N_sq->view(), Nm1_Nm2->view(),
                                           cudf::binary_operator::DIV, out_type,
                                           output_stream);

    auto m3 = cudf::binary_operation(mu3->view(), Nsq_over->view(),
                                     cudf::binary_operator::MUL, out_type,
                                     output_stream);

    // -------------- skew -------------

    auto m2_sq = cudf::binary_operation(m2->view(), m2->view(),
                                        cudf::binary_operator::MUL, out_type,
                                        output_stream);

    auto m2_cu = cudf::binary_operation(m2_sq->view(), m2->view(),
                                        cudf::binary_operator::MUL, out_type,
                                        output_stream);

    auto denom = cudf::unary_operation(
        m2_cu->view(), cudf::unary_operator::SQRT, output_stream);

    auto skew = cudf::binary_operation(m3->view(), denom->view(),
                                       cudf::binary_operator::DIV, out_type,
                                       output_stream);

    return skew;
}

std::unique_ptr<cudf::column> nunique_final_merge(
    const std::vector<cudf::column_view>& input_cols,
    rmm::cuda_stream_view& output_stream) {
    if (input_cols.size() != 1) {
        throw std::runtime_error("nunique_final_merge didn't get 1 column.");
    }

    auto const& collect_set_col = input_cols[0];

    auto nunique_col =
        cudf::lists::count_elements(collect_set_col, output_stream);

    return nunique_col;
}

std::unique_ptr<cudf::column> square_col(const cudf::column_view& input_col,
                                         rmm::cuda_stream_view& output_stream) {
    return cudf::binary_operation(input_col, input_col,
                                  cudf::binary_operator::MUL, input_col.type(),
                                  output_stream);
}

std::unique_ptr<cudf::column> cubed_col(const cudf::column_view& input_col,
                                        rmm::cuda_stream_view& output_stream) {
    auto squared = square_col(input_col, output_stream);
    return cudf::binary_operation(input_col, squared->view(),
                                  cudf::binary_operator::MUL, input_col.type(),
                                  output_stream);
}

std::unique_ptr<cudf::column> cast_int64(const cudf::column_view& input_col,
                                         rmm::cuda_stream_view& output_stream) {
    return cudf::cast(input_col, cudf::data_type{cudf::type_id::INT64},
                      output_stream);
}

std::unique_ptr<cudf::table> nunique_pre_agg(
    const cudf::table_view& tv, cudf::size_type col_index,
    rmm::cuda_stream_view& output_stream) {
    cudf::size_type ncols = static_cast<cudf::size_type>(tv.num_columns());
    if (col_index < 0 || col_index >= ncols) {
        throw std::out_of_range("explode: column index out of range");
    }

    return cudf::explode(tv, col_index, output_stream);
}

#undef MAX_SHUFFLE_HASHTABLE_SIZE
#undef MAX_SHUFFLE_TABLE_SIZE
