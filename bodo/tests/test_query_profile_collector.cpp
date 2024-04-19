
#include "../libs/_bodo_common.h"
#include "../libs/_query_profile_collector.h"
#include "./test.hpp"

bodo::tests::suite query_profile_collector_tests([] {
    bodo::tests::test("test_record_timestamps", [] {
        QueryProfileCollector collector;
        collector.StartPipeline(0);
        // Sleep for 100ms
        size_t expected_0 = 100 * 1000;
        usleep(expected_0);
        collector.EndPipeline(0, 100);

        collector.StartPipeline(1);
        // Sleep for 500us
        size_t expected_1 = 500;
        usleep(expected_1);
        collector.EndPipeline(1, 200);

        auto timestamps = collector.GetPipelineStartEndTimestamps();
        auto elapsed_0 = timestamps[0].second - timestamps[0].first;
        auto elapsed_1 = timestamps[1].second - timestamps[1].first;

        // This assertion is only greater than to avoid random failures in CI.
        // This is because usleep only guarantees that it will sleep for at
        // least the given time, but it may sleep for longer.
        bodo::tests::check(elapsed_0 > expected_0);
        bodo::tests::check(elapsed_1 > expected_1);
    });

    bodo::tests::test("test_record_num_iters", [] {
        QueryProfileCollector collector;
        collector.StartPipeline(0);
        collector.EndPipeline(0, 100);

        collector.StartPipeline(1);
        collector.EndPipeline(1, 200);

        auto num_iters = collector.GetPipelineNumIterations();
        bodo::tests::check(num_iters[0] == 100);
        bodo::tests::check(num_iters[1] == 200);
    });

    bodo::tests::test("test_submit_op_stage_row_counts", [] {
        QueryProfileCollector collector;
        collector.Init();
        // Join Build
        collector.SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(4, 0), 3451, 0);
        // Join Probe
        collector.SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(4, 1), 2349012, 90123);
        // Groupby Build
        collector.SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(2, 0), 2830028, 0);
        // Groupby Produce Output
        collector.SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(2, 1), 0, 320948);
        // Reader
        collector.SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(0, 1), 0, 872194);

        const auto& input_row_counts =
            collector.GetOperatorStageInputRowCounts();

        bodo::tests::check(input_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(4, 0)));
        bodo::tests::check(
            input_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(4, 0))
                ->second == 3451);
        bodo::tests::check(input_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(4, 1)));
        bodo::tests::check(
            input_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(4, 1))
                ->second == 2349012);
        bodo::tests::check(input_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(2, 0)));
        bodo::tests::check(
            input_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(2, 0))
                ->second == 2830028);
        bodo::tests::check(input_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(2, 1)));
        bodo::tests::check(
            input_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(2, 1))
                ->second == 0);
        bodo::tests::check(input_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(0, 1)));
        bodo::tests::check(
            input_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(0, 1))
                ->second == 0);

        const auto& output_row_counts =
            collector.GetOperatorStageOutputRowCounts();
        bodo::tests::check(output_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(4, 0)));
        bodo::tests::check(
            output_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(4, 0))
                ->second == 0);
        bodo::tests::check(output_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(4, 1)));
        bodo::tests::check(
            output_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(4, 1))
                ->second == 90123);
        bodo::tests::check(output_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(2, 0)));
        bodo::tests::check(
            output_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(2, 0))
                ->second == 0);
        bodo::tests::check(output_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(2, 1)));
        bodo::tests::check(
            output_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(2, 1))
                ->second == 320948);
        bodo::tests::check(output_row_counts.contains(
            QueryProfileCollector::MakeOperatorStageID(0, 1)));
        bodo::tests::check(
            output_row_counts
                .find(QueryProfileCollector::MakeOperatorStageID(0, 1))
                ->second == 872194);
    });

    bodo::tests::test("test_get_operator_duration", [] {
        QueryProfileCollector collector;
        collector.SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(1, 0), 100);
        collector.SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(2, 0), 200);
        collector.SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(1, 1), 300);
        collector.SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(2, 3), 400);

        bodo::tests::check(collector.GetOperatorDuration(1) == 400);
        bodo::tests::check(collector.GetOperatorDuration(2) == 600);
    });
});
