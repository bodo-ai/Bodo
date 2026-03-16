#include "../libs/_distributed.h"
#include "../libs/gpu_bloom_filter.h"
#include "../test.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

// Helper to create a simple integer cudf table for testing
static std::unique_ptr<cudf::table> create_int_table(int num_rows,
                                                     int start_val) {
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
    // Bloom Filter Merge + Filter Test
    bodo::tests::test(
        "test_bloom_filter_merge_and_filter",
        [] {
            rmm::cuda_device_id device_id = get_gpu_id();
            if (device_id.value() < 0) {
                // Skip if no GPU
                return;
            }
            cudaSetDevice(device_id.value());
            rmm::cuda_stream_view stream = rmm::cuda_stream_default;

            // ------------------------------------------------------------
            // Build first bloom filter from keys {7, 13, 17}
            // ------------------------------------------------------------
            std::vector<int64_t> vals1 = {7, 13, 17};

            auto col1 = cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::INT64}, vals1.size());

            cudaMemcpyAsync(col1->mutable_view().head<int64_t>(), vals1.data(),
                            vals1.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream.value());

            std::vector<std::unique_ptr<cudf::column>> cols1;
            cols1.push_back(std::move(col1));
            cudf::table tbl1(std::move(cols1));

            // ------------------------------------------------------------
            // Build second bloom filter from keys {23, 29}
            // ------------------------------------------------------------
            std::vector<int64_t> vals2 = {23, 29};

            auto col2 = cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::INT64}, vals2.size());

            cudaMemcpyAsync(col2->mutable_view().head<int64_t>(), vals2.data(),
                            vals2.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream.value());

            std::vector<std::unique_ptr<cudf::column>> cols2;
            cols2.push_back(std::move(col2));
            cudf::table tbl2(std::move(cols2));

            size_t total_rows = tbl2.num_rows() + tbl2.num_rows();

            auto bf1 =
                build_bloom_filter_from_table(tbl1.view(), total_rows,
                                              0.01,  // false positive rate
                                              stream);

            auto bf2 =
                build_bloom_filter_from_table(tbl2.view(), total_rows,
                                              0.01,  // false positive rate
                                              stream);

            // ------------------------------------------------------------
            // Merge bf2 into bf1
            // ------------------------------------------------------------
            mergeBloomBitset(bf1->bitset, bf2->bitset, stream.value());

            // ------------------------------------------------------------
            // Build probe table with keys 0..99
            // ------------------------------------------------------------
            int probe_rows = 100;
            auto probe_tbl = create_int_table(probe_rows, 0);

            // ------------------------------------------------------------
            // Run bloom filter on probe table
            // ------------------------------------------------------------
            std::unique_ptr<cudf::column> mask;
            filter_table_with_bloom(probe_tbl->view(),
                                    {0},  // probe key indices
                                    *bf1, mask, stream);

            // mask is a BOOL8 column; count number of true values
            auto mask_view = mask->view();
            auto sum_red_agg =
                cudf::make_sum_aggregation<cudf::reduce_aggregation>();
            std::unique_ptr<cudf::scalar> mask_sum =
                cudf::reduce(mask_view, *sum_red_agg,
                             cudf::data_type{cudf::type_id::INT64}, stream);
            int64_t matched =
                static_cast<cudf::numeric_scalar<uint64_t>*>(mask_sum.get())
                    ->value(stream);

            // ------------------------------------------------------------
            // Verify: should match keys {7,13,17,23,29}
            // but bloom filter may include a few false positives.
            // We assert: 5 <= matched <= 10
            // ------------------------------------------------------------
            bodo::tests::check(matched >= 5);
            bodo::tests::check(matched <= 10);

            // ------------------------------------------------------------
            // Extract the actual matched keys to verify specific values
            // ------------------------------------------------------------
            auto filtered_tbl =
                cudf::apply_boolean_mask(probe_tbl->view(), mask_view, stream);

            // Extract column 0 as host vector
            auto col_view = filtered_tbl->get_column(0).view();
            std::vector<int64_t> host_vals(col_view.size());
            cudaMemcpyAsync(host_vals.data(), col_view.head<int64_t>(),
                            col_view.size() * sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream.value());
            cudaStreamSynchronize(stream.value());

            // Required keys
            std::vector<int64_t> required = vals1;
            vals1.insert(vals1.end(), vals2.begin(), vals2.end());

            // Check each required key is present
            for (auto key : required) {
                bool found = std::find(host_vals.begin(), host_vals.end(),
                                       key) != host_vals.end();
                bodo::tests::check(found);
            }
        },
        {"gpu_cpp"});
});
