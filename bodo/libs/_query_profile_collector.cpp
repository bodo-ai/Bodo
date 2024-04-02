// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "./_query_profile_collector.h"
#include <mpi.h>
#include <boost/json/src.hpp>
#include <chrono>
#include <unordered_set>
#include "../io/_io.h"
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

template <MetricTypes::TypeEnum metric_type>
static boost::json::object metric_to_json_helper(Metric<metric_type>& metric) {
    boost::json::object info;
    info["start"] = metric.get();
    return info;
}

static std::optional<boost::json::object> metric_to_json(MetricBase& metric) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (metric.is_global && rank != 0) {
        return std::nullopt;
    }
    boost::json::object metric_json;
    metric_json["name"] = metric.name;
    metric_json["type"] = metric.type;

    boost::json::object info;
    switch (metric.type) {
        case MetricTypes::TIMER:
            info = metric_to_json_helper<MetricTypes::TIMER>(
                static_cast<TimerMetric&>(metric));
            break;
        case MetricTypes::STAT:
            info = metric_to_json_helper<MetricTypes::STAT>(
                static_cast<StatMetric&>(metric));
            break;
        case MetricTypes::BLOB:
            info = metric_to_json_helper<MetricTypes::BLOB>(
                static_cast<BlobMetric&>(metric));
    }
    return metric_json;
}

void QueryProfileCollector::Finalize() {
    if (tracing_level == 0) {
        return;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    boost::json::object profile;
    profile["rank"] = rank;
    profile["trace_level"] = tracing_level;

    boost::json::object pipelines;
    for (const auto& [pipeline_id, timestamps] :
         pipeline_start_end_timestamps) {
        boost::json::object pipeline;
        pipeline["start"] = timestamps.first;
        pipeline["end"] = timestamps.second;
        pipeline["duration"] = timestamps.second - timestamps.first;
        pipeline["num_iterations"] = pipeline_num_iterations[pipeline_id];
        pipelines[std::to_string(pipeline_id)] = pipeline;
    }
    profile["pipelines"] = pipelines;

    std::unordered_set<operator_stage_t> seen_operator_stages;
    for (const auto& [op_stage, _] : operator_stage_time) {
        seen_operator_stages.insert(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_metrics) {
        seen_operator_stages.insert(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_input_row_counts) {
        seen_operator_stages.insert(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_output_row_counts) {
        seen_operator_stages.insert(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_metrics) {
        seen_operator_stages.insert(op_stage);
    }

    boost::json::object operator_stages;
    for (const auto& op_stage : seen_operator_stages) {
        boost::json::object operator_stage;
        operator_stage["time"] = operator_stage_time[op_stage];
        if (operator_stage_input_row_counts.count(op_stage) > 0) {
            operator_stage["input_row_count"] =
                operator_stage_input_row_counts[op_stage];
        }
        if (operator_stage_output_row_counts.count(op_stage) > 0) {
            operator_stage["output_row_count"] =
                operator_stage_output_row_counts[op_stage];
        }

        if (operator_stage_metrics.count(op_stage) > 0) {
            boost::json::array metrics;
            for (auto& metric : operator_stage_metrics[op_stage]) {
                auto metric_json = metric_to_json(metric);
                if (metric_json.has_value()) {
                    metrics.push_back(metric_json.value());
                }
            }
            operator_stage["metrics"] = metrics;
        }
        operator_stages[std::to_string(op_stage)] = operator_stage;
    }
    profile["operator_stages"] = operator_stages;

    auto s = boost::json::serialize(profile);

    char filename[100];
    snprintf(filename, 100, "query_profile_%d.json", rank);

    // Note that this write is not parallel because every rank should write to
    // it's own location.
    file_write(filename, (void*)s.c_str(), s.size());
}

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
