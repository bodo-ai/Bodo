
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
});
