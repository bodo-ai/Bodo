#include "./_query_profile_collector.h"
#include <fmt/core.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>  // _mkdir
#endif
#include <boost/json.hpp>
#include <chrono>
#include <iostream>

#include "../io/_io.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_memory.h"
#include "_memory_budget.h"
#include "_mpi.h"

#define DISABLE_IF_TRACING_DISABLED \
    if (tracing_level == 0) {       \
        return;                     \
    }

int makedir(std::string path, int mode) {
#ifdef _WIN32
    return _mkdir(path.data());
#else
    return mkdir(path.data(), mode);
#endif
}

void QueryProfileCollector::Init() {
    QueryProfileCollector new_query_profile_collector;
    *this = new_query_profile_collector;
    tracing_level = getTracingLevel();

    // Get the initial memory budget
    auto operator_comptroller = OperatorComptroller::Default();
    initial_operator_budget = operator_comptroller->GetOperatorBudgets();

    char* env_dir = std::getenv("BODO_TRACING_OUTPUT_DIR");
    // End of initialization when tracing is disabled or if there's no output
    // directory set
    if (tracing_level == 0 || env_dir == nullptr) {
        return;
    }
    std::string parent_dir = env_dir;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // We create the output directory only on rank 0 and assume the filesystem
    // is shared across all ranks.
    if (rank == 0) {
        // Check if the directory already exists
        struct stat info;
        bool exists = false;
        if (stat(parent_dir.c_str(), &info) == 0) {
            exists = true;
        }
        if (!exists) {
            int res = makedir(parent_dir.data(), 0700);
            if (res != 0) {
                // TODO XXX Needs error synchronization!
                throw std::runtime_error(
                    fmt::format("Failed to create output directory {}: {}",
                                output_dir, strerror(errno)));
            }
        }

        // create a subdirectory for the current run with the filename:
        // <parent_dir>/run_YYYYMMDD_HHMMSS
        auto timestamp = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(timestamp);
        auto tm = *gmtime(&tt);
        auto year = tm.tm_year + 1900;
        auto month = tm.tm_mon + 1;
        auto day = tm.tm_mday;

        auto seconds = timestamp.time_since_epoch();
        auto hour = std::chrono::duration_cast<std::chrono::hours>(seconds);
        auto minute =
            std::chrono::duration_cast<std::chrono::minutes>(seconds - hour);
        auto second = std::chrono::duration_cast<std::chrono::seconds>(
            seconds - hour - minute);
        output_dir = fmt::format("{}/run_{}{:02}{:02}_{:02}{:02}{:02}",
                                 parent_dir, year, month, day, hour.count(),
                                 minute.count(), second.count());
        int res = makedir(output_dir.data(), 0700);
        if (res != 0) {
            // TODO XXX Needs error synchronization!
            throw std::runtime_error(
                fmt::format("Failed to create output directory {}: {}",
                            output_dir, strerror(errno)));
        }
    }
    // Broadcast the output_dir length to all ranks
    int output_dir_len = output_dir.size();
    CHECK_MPI(MPI_Bcast(&output_dir_len, 1, MPI_INT, 0, MPI_COMM_WORLD),
              "QueryProfileCollector::Init: MPI error on MPI_Bcast:");

    // Reserve space for the output directory
    if (rank != 0) {
        output_dir.resize(output_dir_len);
    }

    // Broadcast the output directory to all ranks
    CHECK_MPI(MPI_Bcast((void*)output_dir.c_str(), output_dir.size(), MPI_CHAR,
                        0, MPI_COMM_WORLD),
              "QueryProfileCollector::Init: MPI error on MPI_Bcast:");
}

static uint64_t us_since_epoch() {
    // Get the current time in microseconds
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void QueryProfileCollector::StartPipeline(pipeline_id_t pipeline_id) {
    DISABLE_IF_TRACING_DISABLED;

    pipeline_start_end_timestamps[pipeline_id] =
        std::make_pair(us_since_epoch(), 0);
}

void QueryProfileCollector::EndPipeline(pipeline_id_t pipeline_id,
                                        size_t num_iterations) {
    DISABLE_IF_TRACING_DISABLED;

    if (pipeline_start_end_timestamps.count(pipeline_id) == 0) {
        throw std::runtime_error("Pipeline ID not found in start timestamps");
    }
    pipeline_start_end_timestamps[pipeline_id].second = us_since_epoch();
    pipeline_num_iterations[pipeline_id] = num_iterations;
}

void QueryProfileCollector::SubmitOperatorStageRowCounts(
    operator_stage_t op_stage, uint64_t output_row_count) {
    operator_stage_output_row_counts[op_stage] = output_row_count;
}

void QueryProfileCollector::SubmitOperatorStageTime(operator_stage_t op_stage,
                                                    int64_t time) {
    DISABLE_IF_TRACING_DISABLED;
    operator_stage_times[op_stage] = time;
}

void QueryProfileCollector::SubmitOperatorName(operator_id_t operator_id,
                                               const std::string& name) {
    DISABLE_IF_TRACING_DISABLED;
    operator_names[operator_id] = name;
}

int64_t QueryProfileCollector::GetOperatorDuration(operator_id_t operator_id) {
    int64_t total_time = 0;
    for (auto [op_stage, time] : operator_stage_times) {
        if (op_stage.operator_id == operator_id) {
            total_time += time;
        }
    }

    return total_time;
}

void QueryProfileCollector::RegisterOperatorStageMetrics(
    operator_stage_t op_stage, std::vector<MetricBase> metrics) {
    if (operator_stage_metrics.count(op_stage) == 0) {
        operator_stage_metrics[op_stage] = metrics;
    }
    // TODO(aneesh) error when metrics are resubmitted
}

template <MetricTypes::TypeEnum metric_type>
static void metric_to_json_helper(boost::json::object& metric_json,
                                  Metric<metric_type>& metric) {
    metric_json["stat"] = metric.get();
}

static std::optional<boost::json::object> metric_to_json(MetricBase& metric) {
    boost::json::object metric_json;
    metric_json["name"] = metric.name;
    metric_json["type"] = MetricTypes::ToString(metric.type);
    metric_json["global"] = metric.is_global;

    switch (metric.type) {
        case MetricTypes::TIMER:
            metric_to_json_helper<MetricTypes::TIMER>(
                metric_json, static_cast<TimerMetric&>(metric));
            break;
        case MetricTypes::STAT:
            metric_to_json_helper<MetricTypes::STAT>(
                metric_json, static_cast<StatMetric&>(metric));
            break;
        case MetricTypes::BLOB:
            metric_to_json_helper<MetricTypes::BLOB>(
                metric_json, static_cast<BlobMetric&>(metric));
    }
    return metric_json;
}

auto QueryProfileCollector::CollectSeenOperators()
    -> std::map<operator_id_t, stage_id_t> {
    // Map all seen operators to the max stage value seen
    std::map<operator_id_t, stage_id_t> seen_operator_stages;
    // Helper lambda to update the seen operator stages
    auto UpdateSeenOperatorStages = [&](operator_stage_t op_stage) {
        operator_id_t op_id = op_stage.operator_id;
        stage_id_t stage_id = op_stage.stage_id;
        if (seen_operator_stages.count(op_id) == 0) {
            seen_operator_stages[op_id] = stage_id;
        } else {
            seen_operator_stages[op_id] =
                std::max(seen_operator_stages[op_id], stage_id);
        }
    };
    for (const auto& [op_stage, _] : operator_stage_times) {
        UpdateSeenOperatorStages(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_metrics) {
        UpdateSeenOperatorStages(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_output_row_counts) {
        UpdateSeenOperatorStages(op_stage);
    }
    for (const auto& [op_stage, _] : operator_stage_metrics) {
        UpdateSeenOperatorStages(op_stage);
    }
    return seen_operator_stages;
}

boost::json::object QueryProfileCollector::PipelinesToJson() {
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
    return pipelines;
}

boost::json::object QueryProfileCollector::OperatorStageToJson(
    operator_stage_t op_stage) {
    boost::json::object stage_report;
    stage_report["time"] = operator_stage_times[op_stage];
    if (operator_stage_output_row_counts.count(op_stage) > 0) {
        stage_report["output_row_count"] =
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
        stage_report["metrics"] = metrics;
    }

    return stage_report;
}

boost::json::object QueryProfileCollector::OperatorToJson(
    operator_id_t op_id, stage_id_t max_stage) {
    boost::json::object op_output;
    auto name_iter = operator_names.find(op_id);
    std::cout << "OperatorToJson " << op_id << " "
              << (name_iter != operator_names.end()) << std::endl;
    if (name_iter != operator_names.end()) {
        op_output["name"] = name_iter->second;
    }
    for (stage_id_t stage_id = 0; stage_id <= max_stage; stage_id++) {
        boost::json::object stage_report =
            OperatorStageToJson(MakeOperatorStageID(op_id, stage_id));

        auto stage_str = fmt::format("stage_{}", stage_id);
        op_output[stage_str] = stage_report;
    }
    return op_output;
}

void QueryProfileCollector::Finalize(int64_t verbose_level) {
    DISABLE_IF_TRACING_DISABLED;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && verbose_level > 0) {
        std::cout << "Writing profiles to " << output_dir << "\n";
    }

    boost::json::object profile;
    profile["rank"] = rank;
    profile["trace_level"] = tracing_level;

    profile["pipelines"] = PipelinesToJson();

    boost::json::object initial_operator_budgets;
    for (const auto& [op_id, budget] : initial_operator_budget) {
        int64_t relnode_id = op_id / 1000;
        int64_t operator_id = op_id % 1000;
        std::string key = fmt::format("{}.{}", relnode_id, operator_id);
        initial_operator_budgets[key] = budget;
    }
    profile["initial_operator_budgets"] = initial_operator_budgets;

    auto seen_operator_stages = CollectSeenOperators();
    boost::json::object operator_reports;
    for (const auto& [op_id, max_stage] : seen_operator_stages) {
        operator_reports[std::to_string(op_id)] =
            OperatorToJson(op_id, max_stage);
    }
    profile["operator_reports"] = operator_reports;

    profile["buffer_pool_stats"] = bodo::BufferPool::Default()->get_stats();

    auto s = boost::json::serialize(profile);

    std::string filename =
        fmt::format("{}/query_profile_{}.json", output_dir, rank);
    // Note that this write is not parallel because every rank should write to
    // it's own location.
    file_write(filename.c_str(), (void*)s.c_str(), s.size());
}

// Python Interface

static void init_query_profile_collector_py_entry() {
    try {
        QueryProfileCollector::Default().Init();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

static void start_pipeline_query_profile_collector_py_entry(
    int64_t pipeline_id) {
    try {
        QueryProfileCollector::Default().StartPipeline(pipeline_id);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

static void end_pipeline_query_profile_collector_py_entry(
    int64_t pipeline_id, int64_t num_iterations) {
    try {
        QueryProfileCollector::Default().EndPipeline(pipeline_id,
                                                     num_iterations);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

static void submit_operator_stage_row_counts_query_profile_collector_py_entry(
    int64_t operator_id, int64_t stage_id, int64_t output_row_count) {
    try {
        auto op_stage =
            QueryProfileCollector::MakeOperatorStageID(operator_id, stage_id);
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            op_stage, output_row_count);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

static void submit_operator_stage_time_query_profile_collector_py_entry(
    int64_t operator_id, int64_t pipeline_id, double time) {
    try {
        auto op_stage = QueryProfileCollector::MakeOperatorStageID(operator_id,
                                                                   pipeline_id);
        // Note that python submits time as a double in seconds, but we must
        // convert to microseconds for the C++ side.
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            op_stage, static_cast<int64_t>(time * 1e6));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

static double get_operator_duration_query_profile_collector_py_entry(
    int64_t operator_id) {
    // Convert from microseconds to seconds when returning to Python.
    return static_cast<double>(
               QueryProfileCollector::Default().GetOperatorDuration(
                   operator_id)) /
           1e6;
}

static void finalize_query_profile_collector_py_entry(int64_t verbose_level) {
    try {
        QueryProfileCollector::Default().Finalize(verbose_level);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/// The following are only used for unit testing purposes:

static int64_t get_output_row_counts_for_op_stage_py_entry(int64_t operator_id,
                                                           int64_t stage_id) {
    try {
        auto op_stage =
            QueryProfileCollector::MakeOperatorStageID(operator_id, stage_id);
        const auto& output_row_counts =
            QueryProfileCollector::Default().GetOperatorStageOutputRowCounts();
        auto iter = output_row_counts.find(op_stage);
        if (iter == output_row_counts.end()) {
            throw std::runtime_error(
                fmt::format("get_output_row_counts_for_op_stage_py_entry: No "
                            "entry for operator id {} and stage id {}.",
                            operator_id, stage_id));
        } else {
            return iter->second;
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

/// PyMod

PyMODINIT_FUNC PyInit_query_profile_collector_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "query_profile_collector", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m,
                             start_pipeline_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m, end_pipeline_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(
        m, submit_operator_stage_row_counts_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(
        m, submit_operator_stage_time_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(
        m, get_operator_duration_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m, finalize_query_profile_collector_py_entry);
    SetAttrStringFromVoidPtr(m, get_output_row_counts_for_op_stage_py_entry);
    return m;
}
