#include "../libs/_distributed.h"
#include "../libs/gpu_utils.h"
#include "../test.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

// Helper to create a simple integer cudf table for testing
std::unique_ptr<cudf::table> create_int_table(int num_rows, int start_val) {
    // Fill with a sequence for verification
    cudf::numeric_scalar<int64_t> scalar_start =
        cudf::numeric_scalar<int64_t>(start_val);
    cudf::numeric_scalar<int64_t> scalar_step =
        cudf::numeric_scalar<int64_t>(1);

    std::unique_ptr<cudf::column> col =
        cudf::sequence(num_rows, scalar_start, scalar_step);

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));

    return std::make_unique<cudf::table>(std::move(cols));
}

static bodo::tests::suite tests([] {
    // 1. Basic Lifecycle Test
    // Verifies we can instantiate the manager and it initializes NCCL without
    // crashing
    bodo::tests::test("test_gpu_shuffle_init", [] {
        // Ensure we have a GPU context for this rank
        // Note: In a real test runner, this might be handled by a fixture
        rmm::cuda_device_id device_id = get_gpu_id();
        if (device_id.value() >= 0) {
            cudaSetDevice(device_id.value());
        }

        try {
            GpuShuffleManager manager;

            if (device_id.value() < 0) {
                // If no GPU assigned, NCCL comm should be null
                bodo::tests::check(manager.get_nccl_comm() == nullptr);
                bodo::tests::check(manager.get_stream() == nullptr);
                bodo::tests::check(manager.get_mpi_comm() == MPI_COMM_NULL);
            } else {
                // Should be empty on init
                bodo::tests::check(manager.all_complete() == false);

                // Check communicators exist
                bodo::tests::check(manager.get_nccl_comm() != nullptr);
                bodo::tests::check(manager.get_stream() != nullptr);
                bodo::tests::check(manager.get_mpi_comm() != MPI_COMM_NULL);
            }

        } catch (const std::exception& e) {
            std::cout << "Init failed: " << e.what() << std::endl;
            bodo::tests::check(false);
        }
    });

    // 2. Hash Partition Utility Test
    // Verifies the partitioning logic works locally before sending data
    bodo::tests::test("test_hash_partition_logic", [] {
        rmm::cuda_device_id device_id = get_gpu_id();
        if (device_id.value() < 0) {
            // Skip test if no GPU assigned
            return;
        }
        int num_rows = 100;
        int num_partitions = 4;  // Arbitrary partition count
        auto table = create_int_table(num_rows, 0);

        std::vector<cudf::size_type> hash_cols = {
            0};  // Hash on the first column

        auto result =
            hash_partition_table(std::shared_ptr<cudf::table>(std::move(table)),
                                 hash_cols, num_partitions);

        auto partitioned_table = std::move(result.first);
        auto offsets = result.second;

        // Verify we got the table back
        bodo::tests::check(partitioned_table->num_rows() == num_rows);
        bodo::tests::check(partitioned_table->num_columns() == 1);

        // Verify offsets size
        bodo::tests::check(offsets.size() ==
                           static_cast<size_t>(num_partitions));

        // Verify offsets are monotonic and less than the total rows
        for (size_t i = 0; i < offsets.size() - 1; ++i) {
            bodo::tests::check(offsets[i] <= offsets[i + 1]);
            bodo::tests::check(offsets[i] <= num_rows);
        }
    });

    // 3. Simple Shuffle Test
    // Performs a shuffle and ensures data integrity (row counts)
    bodo::tests::test("test_shuffle_integers_end_to_end", [] {
        int rank, n_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

        rmm::cuda_device_id device_id = get_gpu_id();
        if (device_id.value() >= 0) {
            cudaSetDevice(device_id.value());
        }

        // Setup: Create 10 rows per rank with a gpu assigned
        // Rank 0 has [0..9], Rank 1 has [10..19], etc.
        int rows_per_device = 10;
        auto table =
            create_int_table(device_id.value() < 0 ? 0 : rows_per_device,
                             rank * rows_per_device);

        // Shared pointer required by API
        std::shared_ptr<cudf::table> input_ptr = std::move(table);

        GpuShuffleManager manager;

        // Shuffle based on column 0
        std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);
        manager.shuffle_table(input_ptr, {0}, se->event);
        manager.complete();

        std::vector<std::unique_ptr<cudf::table>> received_tables;

        // Pump the progress loop
        while (!manager.all_complete()) {
            auto out_batch = manager.progress();
            // Move received tables into our accumulator
            for (auto& t : out_batch) {
                if (t) {
                    received_tables.push_back(std::move(t));
                }
            }
        }

        // Verification
        // 1. Calculate total rows received on this rank
        int local_received_rows = 0;
        for (const auto& t : received_tables) {
            local_received_rows += t->num_rows();
        }

        // 2. Sum global rows received (requires MPI check)
        int global_received_rows = 0;
        CHECK_MPI(MPI_Allreduce(&local_received_rows, &global_received_rows, 1,
                                MPI_INT, MPI_SUM, MPI_COMM_WORLD),
                  "test_shuffle_integers_end_to_end: MPI_Allreduce failed:");

        // 3. Check conservation of data: Total rows in system must equal total
        // rows out Input was rows_per_rank * number of devices
        int expected_total_rows =
            rows_per_device *
            std::min(get_cluster_cuda_device_count(), n_ranks);

        bodo::tests::check(global_received_rows == expected_total_rows);

        // 4. Check schema preservation
        if (local_received_rows > 0 && !received_tables.empty()) {
            bodo::tests::check(received_tables[0]->num_columns() ==
                               input_ptr->num_columns());
            bodo::tests::check(received_tables[0]->get_column(0).type().id() ==
                               cudf::type_id::INT64);
        }
    });
    bodo::tests::test("test_shuffle_empty", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        rmm::cuda_device_id device_id = get_gpu_id();
        if (device_id.value() >= 0) {
            cudaSetDevice(device_id.value());
        }

        // Setup: Create 0 rows (empty table)
        // We use the same helper, just requesting 0 rows.
        auto table = create_int_table(0, 0);
        std::shared_ptr<cudf::table> input_ptr = std::move(table);

        GpuShuffleManager manager;

        // Shuffle based on column 0
        std::shared_ptr<StreamAndEvent> se = make_stream_and_event(false);
        manager.shuffle_table(input_ptr, {0}, se->event);
        manager.complete();

        std::vector<std::unique_ptr<cudf::table>> received_tables;

        // Pump the progress loop
        while (!manager.all_complete()) {
            auto out_batch = manager.progress();
            // Move received tables into our accumulator
            for (auto& t : out_batch) {
                if (t) {
                    received_tables.push_back(std::move(t));
                }
            }
        }

        // Verification
        // 1. Calculate total rows received on this rank
        int local_received_rows = 0;
        for (const auto& t : received_tables) {
            local_received_rows += t->num_rows();
        }

        // 2. Sum global rows received (requires MPI check)
        int global_received_rows = 0;
        CHECK_MPI(MPI_Allreduce(&local_received_rows, &global_received_rows, 1,
                                MPI_INT, MPI_SUM, MPI_COMM_WORLD),
                  "test_shuffle_empty: MPI_Allreduce failed:");

        // 3. Check conservation of data: Total rows in system must be 0
        bodo::tests::check(global_received_rows == 0);

        // 4. Check schema preservation
        // Even if the table is empty, we expect the schema (column types) to
        // remain INT64
        if (!received_tables.empty()) {
            bodo::tests::check(received_tables[0]->num_columns() ==
                               input_ptr->num_columns());
            bodo::tests::check(received_tables[0]->get_column(0).type().id() ==
                               cudf::type_id::INT64);
        }
    });
});
