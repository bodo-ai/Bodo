#include "../../libs/_distributed.h"
#include "../../libs/gpu_utils.h"
#include "../../libs/streaming/cuda_sort.h"
#include "../test.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>

/**
 * @brief Helper to create a simple integer cudf table from a vector of values.
 */
std::unique_ptr<cudf::table> create_int_table_to_sort(
    std::vector<int64_t> const& vals) {
    if (vals.empty()) {
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(
            cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
        return std::make_unique<cudf::table>(std::move(cols));
    }
    std::vector<std::unique_ptr<cudf::column>> cols;

    rmm::device_uvector<int64_t> d_vals(vals.size(),
                                        cudf::get_default_stream());
    CHECK_CUDA(cudaMemcpy(d_vals.data(), vals.data(),
                          vals.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    auto col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        static_cast<cudf::size_type>(vals.size()), d_vals.release(),
        rmm::device_buffer{}, 0);

    cols.push_back(std::move(col));
    return std::make_unique<cudf::table>(std::move(cols));
}

/**
 * @brief Helper to create an integer cudf table with nulls.
 */
std::unique_ptr<cudf::table> create_masked_int_table(
    std::vector<int64_t> const& vals,
    std::vector<uint8_t> const& null_mask_bytes, int null_count) {
    auto col =
        cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                  static_cast<cudf::size_type>(vals.size()),
                                  cudf::mask_state::UNINITIALIZED);

    CHECK_CUDA(cudaMemcpy(col->mutable_view().head<int64_t>(), vals.data(),
                          vals.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    if (null_count > 0) {
        CHECK_CUDA(cudaMemcpy(col->mutable_view().null_mask(),
                              null_mask_bytes.data(), null_mask_bytes.size(),
                              cudaMemcpyHostToDevice));
        col->set_null_count(null_count);
    } else {
        col->set_null_count(0);
    }

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    return std::make_unique<cudf::table>(std::move(cols));
}

/**
 * @brief Helper to create a simple INT64 schema with a given number of columns.
 */
std::shared_ptr<bodo::Schema> make_simple_int64_schema(int num_cols) {
    std::vector<std::unique_ptr<bodo::DataType>> column_types;
    std::vector<std::string> column_names;
    std::shared_ptr<bodo::TableMetadata> metadata =
        std::make_shared<bodo::TableMetadata>();
    for (int i = 0; i < num_cols; i++) {
        column_types.push_back(std::make_unique<bodo::DataType>(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT64));
        column_names.push_back("col" + std::to_string(i));
    }
    return std::make_shared<bodo::Schema>(
        std::move(column_types), std::move(column_names), std::move(metadata));
}

/**
 * @brief Common test runner for cuda_sort.
 */
void run_cuda_sort_test(
    std::shared_ptr<bodo::Schema> schema,
    std::vector<cudf::size_type> const& key_indices,
    std::vector<cudf::order> const& column_order,
    std::vector<cudf::null_order> const& null_precedence,
    std::function<std::unique_ptr<cudf::table>(int, int, size_t&)>
        create_input_fn,
    std::function<void(cudf::table_view, int, int, MPI_Comm)> extra_verify_fn =
        nullptr) {
    CudaSortState sort_state(schema, key_indices, column_order,
                             null_precedence);
    if (!is_gpu_rank()) {
        return;
    }
    MPI_Comm gpu_comm = sort_state.get_mpi_comm();

    if (gpu_comm == MPI_COMM_NULL) {
        return;
    }

    int gpu_rank, n_gpu_ranks;
    MPI_Comm_rank(gpu_comm, &gpu_rank);
    MPI_Comm_size(gpu_comm, &n_gpu_ranks);

    size_t local_input_rows_count = 0;
    auto input_table =
        create_input_fn(gpu_rank, n_gpu_ranks, local_input_rows_count);
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);

    sort_state.ConsumeBatch(std::move(input_table), se);

    bool global_is_last = false;
    while (!global_is_last) {
        global_is_last = sort_state.FinalizeAccumulation(true);
    }

    bool out_is_last = false;
    auto output =
        sort_state.GetOutputBatch(out_is_last, cudf::get_default_stream());

    bodo::tests::check(out_is_last == true);

    // Total rows should be preserved
    int local_rows = output->num_rows();
    int total_rows = 0;
    CHECK_MPI(
        MPI_Allreduce(&local_rows, &total_rows, 1, MPI_INT, MPI_SUM, gpu_comm),
        "MPI_Allreduce failed");

    int expected_total = 0;
    int local_input_rows = static_cast<int>(local_input_rows_count);
    CHECK_MPI(MPI_Allreduce(&local_input_rows, &expected_total, 1, MPI_INT,
                            MPI_SUM, gpu_comm),
              "MPI_Allreduce failed");

    bodo::tests::check(total_rows == expected_total);

    // Verify global order across ranks (on the first key column)
    auto const& col = output->get_column(key_indices[0]);
    std::vector<int64_t> h_output(local_rows);
    std::vector<cudf::bitmask_type> h_mask;
    if (local_rows > 0) {
        CHECK_CUDA(cudaMemcpy(h_output.data(), col.view().head<int64_t>(),
                              local_rows * sizeof(int64_t),
                              cudaMemcpyDeviceToHost));
        if (col.nullable()) {
            size_t mask_size = cudf::bitmask_allocation_size_bytes(local_rows);
            h_mask.resize(mask_size / sizeof(cudf::bitmask_type));
            CHECK_CUDA(cudaMemcpy(h_mask.data(), col.view().null_mask(),
                                  mask_size, cudaMemcpyDeviceToHost));
        }
    }

    auto is_null = [&](int i) {
        if (!col.nullable()) {
            return false;
        }
        return !cudf::bit_is_set(h_mask.data(), i);
    };

    auto get_val = [&](int i) {
        if (is_null(i)) {
            // For these tests we always use NULLS BEFORE
            // In ASCENDING: NULL is -inf
            // In DESCENDING: NULL is +inf
            return (column_order[0] == cudf::order::ASCENDING)
                       ? std::numeric_limits<int64_t>::min()
                       : std::numeric_limits<int64_t>::max();
        }
        return h_output[i];
    };

    bool ascending = column_order[0] == cudf::order::ASCENDING;

    // Check local sorting
    for (int i = 0; i < local_rows - 1; ++i) {
        if (ascending) {
            bodo::tests::check(get_val(i) <= get_val(i + 1));
        } else {
            bodo::tests::check(get_val(i) >= get_val(i + 1));
        }
    }

    // Check global ordering
    int64_t local_min, local_max;
    if (ascending) {
        local_min =
            local_rows > 0 ? get_val(0) : std::numeric_limits<int64_t>::max();
        local_max = local_rows > 0 ? get_val(local_rows - 1)
                                   : std::numeric_limits<int64_t>::min();
    } else {
        local_min = local_rows > 0 ? get_val(local_rows - 1)
                                   : std::numeric_limits<int64_t>::max();
        local_max =
            local_rows > 0 ? get_val(0) : std::numeric_limits<int64_t>::min();
    }

    std::vector<int64_t> all_mins(n_gpu_ranks);
    std::vector<int64_t> all_maxes(n_gpu_ranks);

    CHECK_MPI(MPI_Allgather(&local_min, 1, MPI_INT64_T, all_mins.data(), 1,
                            MPI_INT64_T, gpu_comm),
              "MPI_Allgather failed");
    CHECK_MPI(MPI_Allgather(&local_max, 1, MPI_INT64_T, all_maxes.data(), 1,
                            MPI_INT64_T, gpu_comm),
              "MPI_Allgather failed");

    int last_rank_with_data = -1;
    for (int r = 0; r < n_gpu_ranks; ++r) {
        if (ascending) {
            if (all_mins[r] == std::numeric_limits<int64_t>::max()) {
                continue;
            }
            if (last_rank_with_data != -1) {
                bodo::tests::check(all_mins[r] >=
                                   all_maxes[last_rank_with_data]);
            }
        } else {
            if (all_maxes[r] == std::numeric_limits<int64_t>::min()) {
                continue;
            }
            if (last_rank_with_data != -1) {
                bodo::tests::check(all_maxes[r] <=
                                   all_mins[last_rank_with_data]);
            }
        }
        last_rank_with_data = r;
    }

    if (extra_verify_fn) {
        extra_verify_fn(output->view(), gpu_rank, n_gpu_ranks, gpu_comm);
    }
}

static bodo::tests::suite cuda_sort_tests([] {
    bodo::tests::test("test_cuda_sort_integers_asc",
                      [] {
                          run_cuda_sort_test(
                              make_simple_int64_schema(1), {0},
                              {cudf::order::ASCENDING},
                              {cudf::null_order::BEFORE},
                              [](int rank, int n_ranks, size_t& rows_count) {
                                  std::vector<int64_t> vals;
                                  if (rank == 0) {
                                      vals = {50, 10, 80, 30};
                                  } else if (rank == 1 % n_ranks) {
                                      vals = {20, 70, 40, 60};
                                  }
                                  rows_count = vals.size();
                                  return create_int_table_to_sort(vals);
                              });
                      },
                      {"gpu_cpp"});

    bodo::tests::test(
        "test_cuda_sort_nulls_before",
        [] {
            run_cuda_sort_test(
                make_simple_int64_schema(1), {0}, {cudf::order::ASCENDING},
                {cudf::null_order::BEFORE},
                [](int rank, int n_ranks, size_t& rows_count) {
                    std::vector<int64_t> vals;
                    std::vector<uint8_t> mask_bytes = {0b00001010};
                    int null_count = 2;
                    if (rank == 0) {
                        vals = {0, 50, 0, 10};
                    } else if (rank == 1 % n_ranks) {
                        vals = {80, 0, 30, 0};
                    } else {
                        mask_bytes = {0b00000000};
                        null_count = 0;
                    }
                    rows_count = vals.size();
                    return create_masked_int_table(vals, mask_bytes,
                                                   null_count);
                });
        },
        {"gpu_cpp"});

    bodo::tests::test("test_cuda_sort_integers_desc",
                      [] {
                          run_cuda_sort_test(
                              make_simple_int64_schema(1), {0},
                              {cudf::order::DESCENDING},
                              {cudf::null_order::BEFORE},
                              [](int rank, int n_ranks, size_t& rows_count) {
                                  std::vector<int64_t> vals;
                                  if (rank == 0) {
                                      vals = {10, 50, 30, 80};
                                  } else if (rank == 1 % n_ranks) {
                                      vals = {60, 40, 70, 20};
                                  }
                                  rows_count = vals.size();
                                  return create_int_table_to_sort(vals);
                              });
                      },
                      {"gpu_cpp"});

    bodo::tests::test(
        "test_cuda_sort_key_not_first",
        [] {
            run_cuda_sort_test(
                make_simple_int64_schema(2), {1}, {cudf::order::ASCENDING},
                {cudf::null_order::BEFORE},
                [](int rank, int n_ranks, size_t& rows_count) {
                    std::vector<int64_t> keys;
                    std::vector<int64_t> data;
                    if (rank == 0) {
                        keys = {50, 10, 80, 30};
                        data = {1, 2, 3, 4};
                    } else if (rank == 1 % n_ranks) {
                        keys = {20, 70, 40, 60};
                        data = {5, 6, 7, 8};
                    }
                    rows_count = keys.size();

                    auto data_col =
                        std::move(create_int_table_to_sort(data)->release()[0]);
                    auto key_col =
                        std::move(create_int_table_to_sort(keys)->release()[0]);
                    std::vector<std::unique_ptr<cudf::column>> cols;
                    cols.push_back(std::move(data_col));
                    cols.push_back(std::move(key_col));
                    return std::make_unique<cudf::table>(std::move(cols));
                },
                [](cudf::table_view output, int rank, int n_ranks,
                   MPI_Comm comm) {
                    int num_rows = output.num_rows();
                    std::vector<int64_t> h_data(num_rows);
                    std::vector<int64_t> h_keys(num_rows);
                    if (num_rows > 0) {
                        CHECK_CUDA(cudaMemcpy(h_data.data(),
                                              output.column(0).head<int64_t>(),
                                              num_rows * sizeof(int64_t),
                                              cudaMemcpyDeviceToHost));
                        CHECK_CUDA(cudaMemcpy(h_keys.data(),
                                              output.column(1).head<int64_t>(),
                                              num_rows * sizeof(int64_t),
                                              cudaMemcpyDeviceToHost));
                    }

                    // Mapping based on input:
                    // 10 -> 2, 20 -> 5, 30 -> 4, 40 -> 7,
                    // 50 -> 1, 60 -> 8, 70 -> 6, 80 -> 3
                    for (int i = 0; i < num_rows; i++) {
                        int64_t k = h_keys[i];
                        int64_t d = h_data[i];
                        switch (k) {
                            case 10:
                                bodo::tests::check(d == 2);
                                break;
                            case 20:
                                bodo::tests::check(d == 5);
                                break;
                            case 30:
                                bodo::tests::check(d == 4);
                                break;
                            case 40:
                                bodo::tests::check(d == 7);
                                break;
                            case 50:
                                bodo::tests::check(d == 1);
                                break;
                            case 60:
                                bodo::tests::check(d == 8);
                                break;
                            case 70:
                                bodo::tests::check(d == 6);
                                break;
                            case 80:
                                bodo::tests::check(d == 3);
                                break;
                            default:
                                bodo::tests::check(false);
                                break;
                        }
                    }
                });
        },
        {"gpu_cpp"});
});
