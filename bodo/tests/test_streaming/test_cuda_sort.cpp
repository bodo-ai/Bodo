#include "../../libs/_distributed.h"
#include "../../libs/gpu_utils.h"
#include "../../libs/streaming/cuda_sort.h"
#include "../test.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
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

static bodo::tests::suite cuda_sort_tests([] {
    // Basic test: sort integers in ascending order
    bodo::tests::test(
        "test_cuda_sort_integers_asc",
        [] {
            int rank, n_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

            if (!is_gpu_rank()) {
                return;
            }

            // Schema: 1 INT64 column
            std::vector<std::unique_ptr<bodo::DataType>> column_types;
            std::vector<std::string> column_names;
            std::shared_ptr<bodo::TableMetadata> metadata =
                std::make_shared<bodo::TableMetadata>();
            column_types.push_back(std::make_unique<bodo::DataType>(
                bodo_array_type::NUMPY, Bodo_CTypes::INT64));
            column_names.emplace_back("col0");
            auto schema = std::make_shared<bodo::Schema>(
                std::move(column_types), std::move(column_names),
                std::move(metadata));

            std::vector<cudf::size_type> key_indices = {0};
            std::vector<cudf::order> column_order = {cudf::order::ASCENDING};
            std::vector<cudf::null_order> null_precedence = {
                cudf::null_order::BEFORE};

            CudaSortState sort_state(schema, key_indices, column_order,
                                     null_precedence);
            MPI_Comm gpu_comm = sort_state.get_mpi_comm();

            if (gpu_comm == MPI_COMM_NULL) {
                return;
            }

            int gpu_rank, n_gpu_ranks;
            MPI_Comm_rank(gpu_comm, &gpu_rank);
            MPI_Comm_size(gpu_comm, &n_gpu_ranks);

            // Data: unsorted
            std::vector<int64_t> vals;
            if (gpu_rank == 0) {
                vals = {50, 10, 80, 30};
            } else if (gpu_rank == 1 % n_gpu_ranks) {
                vals = {20, 70, 40, 60};
            }

            auto input_table = create_int_table_to_sort(vals);
            std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);

            sort_state.ConsumeBatch(std::move(input_table), se);

            bool global_is_last = false;
            while (!global_is_last) {
                global_is_last = sort_state.FinalizeAccumulation(true);
            }

            bool out_is_last = false;
            auto output = sort_state.GetOutputBatch(out_is_last,
                                                    cudf::get_default_stream());

            bodo::tests::check(out_is_last == true);

            // Total rows should be preserved
            int local_rows = output->num_rows();
            int total_rows = 0;
            CHECK_MPI(MPI_Allreduce(&local_rows, &total_rows, 1, MPI_INT,
                                    MPI_SUM, gpu_comm),
                      "MPI_Allreduce failed");

            int expected_total = 0;
            int local_input_rows = static_cast<int>(vals.size());
            CHECK_MPI(MPI_Allreduce(&local_input_rows, &expected_total, 1,
                                    MPI_INT, MPI_SUM, gpu_comm),
                      "MPI_Allreduce failed");

            bodo::tests::check(total_rows == expected_total);

            // Verify global order across ranks
            std::vector<int64_t> h_output(local_rows);
            if (local_rows > 0) {
                CHECK_CUDA(cudaMemcpy(
                    h_output.data(),
                    output->get_column(0).view().head<int64_t>(),
                    local_rows * sizeof(int64_t), cudaMemcpyDeviceToHost));
            }

            // 1. Check local sorting
            for (int i = 0; i < local_rows - 1; ++i) {
                bodo::tests::check(h_output[i] <= h_output[i + 1]);
            }

            // 2. Check global ordering (min of current rank >= max of previous
            // rank)
            int64_t local_min = local_rows > 0
                                    ? h_output[0]
                                    : std::numeric_limits<int64_t>::max();
            int64_t local_max = local_rows > 0
                                    ? h_output[local_rows - 1]
                                    : std::numeric_limits<int64_t>::min();

            std::vector<int64_t> all_mins(n_gpu_ranks);
            std::vector<int64_t> all_maxes(n_gpu_ranks);

            CHECK_MPI(MPI_Allgather(&local_min, 1, MPI_INT64_T, all_mins.data(),
                                    1, MPI_INT64_T, gpu_comm),
                      "MPI_Allgather failed");
            CHECK_MPI(MPI_Allgather(&local_max, 1, MPI_INT64_T,
                                    all_maxes.data(), 1, MPI_INT64_T, gpu_comm),
                      "MPI_Allgather failed");

            int last_rank_with_data = -1;
            for (int r = 0; r < n_gpu_ranks; ++r) {
                if (all_mins[r] == std::numeric_limits<int64_t>::max())
                    continue;

                if (last_rank_with_data != -1) {
                    bodo::tests::check(all_mins[r] >=
                                       all_maxes[last_rank_with_data]);
                }
                last_rank_with_data = r;
            }
        },
        {"gpu_cpp"});

    // Test with nulls: NULLS BEFORE
    bodo::tests::test(
        "test_cuda_sort_nulls_before",
        [] {
            int rank, n_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

            if (!is_gpu_rank()) {
                return;
            }

            std::vector<std::unique_ptr<bodo::DataType>> column_types;
            std::vector<std::string> column_names;
            std::shared_ptr<bodo::TableMetadata> metadata =
                std::make_shared<bodo::TableMetadata>();
            column_types.push_back(std::make_unique<bodo::DataType>(
                bodo_array_type::NUMPY, Bodo_CTypes::INT64));
            column_names.emplace_back("col0");

            auto schema = std::make_shared<bodo::Schema>(
                std::move(column_types), std::move(column_names),
                std::move(metadata));

            std::vector<cudf::size_type> key_indices = {0};
            std::vector<cudf::order> column_order = {cudf::order::ASCENDING};
            std::vector<cudf::null_order> null_precedence = {
                cudf::null_order::BEFORE};

            CudaSortState sort_state(schema, key_indices, column_order,
                                     null_precedence);
            MPI_Comm gpu_comm = sort_state.get_mpi_comm();

            if (gpu_comm == MPI_COMM_NULL) {
                return;
            }

            int gpu_rank, n_gpu_ranks;
            MPI_Comm_rank(gpu_comm, &gpu_rank);
            MPI_Comm_size(gpu_comm, &n_gpu_ranks);

            std::vector<int64_t> vals;
            std::vector<uint8_t> mask_bytes = {0b00001010};
            int null_count = 2;
            if (gpu_rank == 0) {
                vals = {0, 50, 0, 10};
            } else if (gpu_rank == 1 % n_gpu_ranks) {
                vals = {80, 0, 30, 0};
            } else {
                mask_bytes = {0b00000000};
                null_count = 0;
            }

            auto input_table =
                create_masked_int_table(vals, mask_bytes, null_count);
            std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);

            sort_state.ConsumeBatch(std::move(input_table), se);

            bool global_is_last = false;
            while (!global_is_last) {
                global_is_last = sort_state.FinalizeAccumulation(true);
            }

            bool out_is_last = false;
            auto output = sort_state.GetOutputBatch(out_is_last,
                                                    cudf::get_default_stream());

            bodo::tests::check(out_is_last == true);

            int local_rows = output->num_rows();
            int total_nulls = 0;
            int local_nulls = output->get_column(0).null_count();
            CHECK_MPI(MPI_Allreduce(&local_nulls, &total_nulls, 1, MPI_INT,
                                    MPI_SUM, gpu_comm),
                      "MPI_Allreduce failed");

            int expected_nulls =
                (n_gpu_ranks >= 2) ? 4 : (n_gpu_ranks == 1 ? 2 : 0);
            bodo::tests::check(total_nulls == expected_nulls);

            if (local_rows > 0) {
                bodo::tests::check(output->get_column(0).view().null_count(
                                       0, local_nulls) == local_nulls);
                bodo::tests::check(output->get_column(0).view().null_count(
                                       local_nulls, local_rows) == 0);
            }
        },
        {"gpu_cpp"});

    // Test descending
    bodo::tests::test(
        "test_cuda_sort_integers_desc",
        [] {
            int rank, n_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

            if (!is_gpu_rank())
                return;

            std::vector<std::unique_ptr<bodo::DataType>> column_types;
            std::vector<std::string> column_names;
            std::shared_ptr<bodo::TableMetadata> metadata =
                std::make_shared<bodo::TableMetadata>();
            column_types.push_back(std::make_unique<bodo::DataType>(
                bodo_array_type::NUMPY, Bodo_CTypes::INT64));
            column_names.emplace_back("col0");
            auto schema = std::make_shared<bodo::Schema>(
                std::move(column_types), std::move(column_names),
                std::move(metadata));

            std::vector<cudf::size_type> key_indices = {0};
            std::vector<cudf::order> column_order = {cudf::order::DESCENDING};
            std::vector<cudf::null_order> null_precedence = {
                cudf::null_order::BEFORE};

            CudaSortState sort_state(schema, key_indices, column_order,
                                     null_precedence);
            MPI_Comm gpu_comm = sort_state.get_mpi_comm();

            if (gpu_comm == MPI_COMM_NULL) {
                return;
            }

            int gpu_rank, n_gpu_ranks;
            MPI_Comm_rank(gpu_comm, &gpu_rank);
            MPI_Comm_size(gpu_comm, &n_gpu_ranks);

            std::vector<int64_t> vals;
            if (gpu_rank == 0) {
                vals = {10, 50, 30, 80};
            } else if (gpu_rank == 1 % n_gpu_ranks) {
                vals = {60, 40, 70, 20};
            }

            auto input_table = create_int_table_to_sort(vals);
            std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);
            sort_state.ConsumeBatch(std::move(input_table), se);

            bool global_is_last = false;
            while (!global_is_last) {
                global_is_last = sort_state.FinalizeAccumulation(true);
            }

            bool out_is_last = false;
            auto output = sort_state.GetOutputBatch(out_is_last,
                                                    cudf::get_default_stream());

            int local_rows = output->num_rows();
            std::vector<int64_t> h_output(local_rows);
            if (local_rows > 0) {
                CHECK_CUDA(cudaMemcpy(
                    h_output.data(),
                    output->get_column(0).view().head<int64_t>(),
                    local_rows * sizeof(int64_t), cudaMemcpyDeviceToHost));
            }

            for (int i = 0; i < local_rows - 1; ++i) {
                bodo::tests::check(h_output[i] >= h_output[i + 1]);
            }

            int64_t local_min = local_rows > 0
                                    ? h_output[local_rows - 1]
                                    : std::numeric_limits<int64_t>::max();
            int64_t local_max = local_rows > 0
                                    ? h_output[0]
                                    : std::numeric_limits<int64_t>::min();

            std::vector<int64_t> all_mins(n_gpu_ranks);
            std::vector<int64_t> all_maxes(n_gpu_ranks);
            CHECK_MPI(MPI_Allgather(&local_min, 1, MPI_INT64_T, all_mins.data(),
                                    1, MPI_INT64_T, gpu_comm),
                      "MPI_Allgather failed");
            CHECK_MPI(MPI_Allgather(&local_max, 1, MPI_INT64_T,
                                    all_maxes.data(), 1, MPI_INT64_T, gpu_comm),
                      "MPI_Allgather failed");

            int last_rank_with_data = -1;
            for (int r = 0; r < n_gpu_ranks; ++r) {
                if (all_maxes[r] == std::numeric_limits<int64_t>::min())
                    continue;
                if (last_rank_with_data != -1) {
                    bodo::tests::check(all_maxes[r] <=
                                       all_mins[last_rank_with_data]);
                }
                last_rank_with_data = r;
            }
        },
        {"gpu_cpp"});
});
