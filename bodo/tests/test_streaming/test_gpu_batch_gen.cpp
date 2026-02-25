#include <arrow/util/bit_util.h>
#include <fmt/core.h>
#include <mpi.h>
#include <mpi_proto.h>
#include <cudf/concatenate.hpp>
#include <numeric>
#include <sstream>
#include "../../pandas/physical/operator.h"
#include "../libs/_table_builder_utils.h"
#include "../table_generator.hpp"
#include "../test.hpp"
#include "arrow/util/key_value_metadata.h"

static bodo::tests::suite tests([] {
    bodo::tests::test(
        "test_gpu_batch_gen",
        [] {
            int npes;
            MPI_Comm_size(MPI_COMM_WORLD, &npes);
            if (npes != 1) {
                std::cout << "Skipping test_gpu_batch_gen since it is only "
                             "designed to run with 1 process."
                          << std::endl;
                return;
            }

            auto bodo_table = bodo::tests::cppToBodo(
                {"A"}, {true}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});

            auto make_gpu_generator = [&bodo_table](bool use_async) {
                auto se_empty = make_stream_and_event(use_async);
                auto empty_data =
                    convertTableToGPU(alloc_table_like(bodo_table), se_empty);
                se_empty->event.record(se_empty->stream);
                return GPUBatchGenerator(empty_data, 8);
            };

            auto combine_output_batches = [](std::vector<GPU_DATA> gpu_datas,
                                             bool use_async) {
                std::vector<cudf::table_view> table_views;
                auto se_concat = make_stream_and_event(use_async);
                for (auto &gpu_data : gpu_datas) {
                    gpu_data.stream_event->event.wait(se_concat->stream);
                    table_views.emplace_back(gpu_data.table->view());
                }
                auto concatenated =
                    cudf::concatenate(table_views, se_concat->stream);
                auto schema_meta = std::make_shared<arrow::KeyValueMetadata>();
                auto arrow_schema = std::make_shared<arrow::Schema>(
                    std::vector<std::shared_ptr<arrow::Field>>{
                        arrow::field("A", arrow::int64(), /*nullable=*/true)},
                    schema_meta);
                return convertGPUToTable(
                    GPU_DATA(std::move(concatenated), arrow_schema, se_concat));
            };

            std::vector<bool> use_async_options = {false, true};
            for (bool use_async : use_async_options) {
                // Calling next when generator is empty:
                {
                    auto gpu_batch_generator = make_gpu_generator(use_async);
                    auto se = make_stream_and_event(use_async);
                    auto batch = gpu_batch_generator.next(se, false);
                    bodo::tests::check(batch.table->num_rows() == 0);
                }

                // Iterating with small batches:
                {
                    auto gpu_batch_generator = make_gpu_generator(use_async);
                    std::vector<GPU_DATA> gpu_datas;
                    for (int i = 0; i < 5; i++) {
                        auto se_batch = make_stream_and_event(use_async);
                        auto gpu_data = convertTableToGPU(bodo_table, se_batch);
                        gpu_batch_generator.append_batch(gpu_data);
                        auto batch = gpu_batch_generator.next(se_batch, false);
                        se_batch->event.record(se_batch->stream);
                        gpu_datas.push_back(batch);
                    }
                    auto se_last_batch = make_stream_and_event(use_async);
                    auto last_batch =
                        gpu_batch_generator.next(se_last_batch, true);
                    gpu_datas.push_back(last_batch);

                    std::shared_ptr<table_info> collected_table =
                        combine_output_batches(gpu_datas, use_async);

                    std::stringstream ss_result;
                    DEBUG_PrintTable(ss_result, collected_table);
                    auto expected_table =
                        concat_tables({bodo_table, bodo_table, bodo_table,
                                       bodo_table, bodo_table});
                    std::stringstream ss_expected;
                    DEBUG_PrintTable(ss_expected, expected_table);
                    bodo::tests::check(ss_result.str() == ss_expected.str());
                }

                // Append a large batch:
                {
                    std::vector<GPU_DATA> gpu_datas;
                    std::vector<int> big_vec(100);
                    std::iota(big_vec.begin(), big_vec.end(), 0);
                    auto big_table =
                        bodo::tests::cppToBodo({"A"}, {true}, {}, big_vec);
                    auto big_se = make_stream_and_event(use_async);
                    auto big_gpu_data = convertTableToGPU(big_table, big_se);
                    big_se->event.record(big_se->stream);

                    auto gpu_batch_generator = make_gpu_generator(use_async);
                    gpu_batch_generator.append_batch(big_gpu_data);

                    for (int i = 0; i < 12; i++) {
                        auto batch_se = make_stream_and_event(use_async);
                        auto batch = gpu_batch_generator.next(batch_se, false);
                        gpu_datas.push_back(batch);
                    }
                    auto last_se = make_stream_and_event(use_async);
                    auto last_batch = gpu_batch_generator.next(last_se, true);
                    gpu_datas.push_back(last_batch);

                    // Append another batch to make sure that leftover data is
                    // handled correctly.
                    auto other_se = make_stream_and_event(use_async);
                    auto other_gpu_data =
                        convertTableToGPU(big_table, other_se);
                    gpu_batch_generator.append_batch(other_gpu_data);
                    auto next_batch = gpu_batch_generator.next(other_se, false);
                    gpu_datas.push_back(next_batch);

                    std::shared_ptr<table_info> collected_table =
                        combine_output_batches(gpu_datas, use_async);

                    std::stringstream ss_result;
                    DEBUG_PrintTable(ss_result, collected_table);

                    auto last_batch_expected_data = std::vector<int>(8);
                    std::iota(last_batch_expected_data.begin(),
                              last_batch_expected_data.end(), 0);
                    auto expected_last_batch_table = bodo::tests::cppToBodo(
                        {"A"}, {true}, {}, last_batch_expected_data);
                    auto expected_table =
                        concat_tables({big_table, expected_last_batch_table});
                    std::stringstream ss_expected;
                    DEBUG_PrintTable(ss_expected, expected_table);
                    bodo::tests::check(ss_result.str() == ss_expected.str());
                }
            }
        },
        {"gpu_cpp"});
});
