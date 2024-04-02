// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "./_query_profile_collector.h"
#include <chrono>
#include "_bodo_common.h"

void QueryProfileCollector::Init() {
    QueryProfileCollector new_query_profile_collector;
    *this = new_query_profile_collector;
    tracing_level = getTracingLevel();
}

static uint64_t us_since_epoch() {
    // Get the current time in microseconds
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void QueryProfileCollector::StartPipeline(pipeline_id_t pipeline_id) {
    pipeline_start_end_timestamps[pipeline_id] =
        std::make_pair(us_since_epoch(), 0);
}

void QueryProfileCollector::EndPipeline(pipeline_id_t pipeline_id,
                                        size_t num_iterations) {
    if (pipeline_start_end_timestamps.count(pipeline_id) == 0) {
        throw std::runtime_error("Pipeline ID not found in start timestamps");
    }
    pipeline_start_end_timestamps[pipeline_id].second = us_since_epoch();
    pipeline_num_iterations[pipeline_id] = num_iterations;
}

void QueryProfileCollector::SubmitOperatorStageRowCounts(
    operator_stage_t op_stage, uint64_t input_row_count,
    uint64_t output_row_count) {
    operator_stage_input_row_counts[op_stage] = input_row_count;
    operator_stage_output_row_counts[op_stage] = output_row_count;
}

void QueryProfileCollector::SubmitOperatorStageTime(operator_stage_t op_stage,
                                                    uint64_t time_us) {
    operator_stage_time[op_stage] = time_us;
}

void QueryProfileCollector::RegisterOperatorStageMetrics(
    operator_stage_t op_stage, std::vector<MetricBase> metrics) {
    if (operator_stage_metrics.count(op_stage) > 0) {
        auto& old_metrics = operator_stage_metrics[op_stage];
        old_metrics.insert(old_metrics.end(), metrics.begin(), metrics.end());
        return;
    }

    operator_stage_metrics[op_stage] = metrics;
}

void QueryProfileCollector::Finalize() {}

// Python Interface

static void init_query_profile_collector_py_entry() {
    QueryProfileCollector::Default().Init();
}

static void start_pipeline_query_profile_collector_py_entry(
    int64_t pipeline_id) {
    QueryProfileCollector::Default().StartPipeline(pipeline_id);
}

static void end_pipeline_query_profile_collector_py_entry(
    int64_t pipeline_id, int64_t num_iterations) {
    QueryProfileCollector::Default().EndPipeline(pipeline_id, num_iterations);
}

static void submit_operator_stage_row_counts_query_profile_collector_py_entry(
    int64_t operator_id, int64_t pipeline_id, int64_t input_row_count,
    int64_t output_row_count) {
    auto op_stage =
        QueryProfileCollector::MakeOperatorStageID(operator_id, pipeline_id);
    QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
        op_stage, input_row_count, output_row_count);
}

static void finalize_query_profile_collector_py_entry() {
    QueryProfileCollector::Default().Finalize();
}

PyMODINIT_FUNC PyInit_query_profile_collector_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "query_profile_collector", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m,
                             start_pipeline_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m, end_pipeline_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(
        m, submit_operator_stage_row_counts_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m, finalize_query_profile_collector_py_entry);
    return m;
}
