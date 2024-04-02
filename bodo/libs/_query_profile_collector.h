// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/**
 * @brief Types of operator metrics that can be collected
 */
struct MetricTypes {
    enum TypeEnum { TIMER = 0, STAT = 1, BLOB = 2 };
};

/**
 * @brief Base class for metrics
 *
 * Note that this class is not meant to be instantiated directly, see Metric
 * below
 */
class MetricBase {
   public:
    bool is_global = false;
    std::string name;
    MetricTypes::TypeEnum type;
    using TimerValue = uint64_t;
    using StatValue = int64_t;
    using BlobValue = std::string;

    using VariantType = std::variant<TimerValue, StatValue, BlobValue>;

   protected:
    // Store any possible metric as a type safe variant
    VariantType value;
};

/**
 * @brief Metric class that wraps a specific type of metric and provides
 * type-safe access to the recorded value
 *
 * @tparam type_ The type of metric to wrap
 */
template <MetricTypes::TypeEnum type_>
class Metric : public MetricBase {
   public:
    MetricTypes::TypeEnum type = type_;
    // Use the enum as a key to get the type for the specific metric from the
    // underlying variant
    using ValueType = decltype(std::get<type_>(value));

    Metric() = delete;
    Metric(ValueType val) { set(val); }

    // getter and setter for the metric value
    ValueType get() { return std::get<type_>(value); }
    void set(ValueType val) { value = val; }
};

// Defined metric types
using TimerMetric = Metric<MetricTypes::TIMER>;
using StatMetric = Metric<MetricTypes::STAT>;
using BlobMetric = Metric<MetricTypes::BLOB>;

/**
 * @brief Class to collect query profile information
 */
class QueryProfileCollector {
   public:
    /**
     * @brief Get the globally available singleton instance
     */
    static QueryProfileCollector& Default() {
        static std::unique_ptr<QueryProfileCollector> collector_(
            new QueryProfileCollector());
        return *collector_;
    }

    using operator_id_t = int32_t;
    using operator_stage_t = uint64_t;
    using pipeline_id_t = uint32_t;

    /**
     * @brief Create an operator stage ID from an operator ID and pipeline ID
     *
     * This method packs two 32 bit ids into a single 64 bit id
     *
     * @param operator_id The operator ID
     * @param pipeline_id The pipeline ID
     * @return operator_stage_t The operator stage ID
     */
    static operator_stage_t MakeOperatorStageID(operator_id_t operator_id,
                                                pipeline_id_t pipeline_id) {
        return (static_cast<operator_stage_t>(operator_id) << 32) |
               static_cast<operator_stage_t>(pipeline_id);
    }

    void Init();
    void StartPipeline(pipeline_id_t pipeline_id);
    void EndPipeline(pipeline_id_t pipeline_id, size_t num_iterations);
    void SubmitOperatorStageRowCounts(operator_stage_t op_stage,
                                      uint64_t input_row_count,
                                      uint64_t output_row_count);
    void SubmitOperatorStageTime(operator_stage_t op_stage, uint64_t time_us);

    // This is only required by C++ at this point since
    // only operators with states in C++ will use this.
    void RegisterOperatorStageMetrics(operator_stage_t op_stage,
                                      std::vector<MetricBase> metrics);

    void Finalize();

#ifdef IS_TESTING
    // Getters for testing
    std::unordered_map<pipeline_id_t, std::pair<uint64_t, uint64_t>>&
    GetPipelineStartEndTimestamps() {
        return pipeline_start_end_timestamps;
    }

    std::unordered_map<pipeline_id_t, uint64_t>& GetPipelineNumIterations() {
        return pipeline_num_iterations;
    }

    std::unordered_map<operator_stage_t, uint64_t>& GetOperatorStageTime() {
        return operator_stage_time;
    }

    std::unordered_map<operator_stage_t, uint64_t>&
    GetOperatorStageInputRowCounts() {
        return operator_stage_input_row_counts;
    }

    std::unordered_map<operator_stage_t, uint64_t>&
    GetOperatorStageOutputRowCounts() {
        return operator_stage_output_row_counts;
    }

    std::unordered_map<operator_stage_t, std::vector<MetricBase>>&
    GetMetrics() {
        return operator_stage_metrics;
    }
#endif

   private:
    int tracing_level = 1;

    // Map the pipeline ID to its start and end timestamps
    std::unordered_map<pipeline_id_t, std::pair<uint64_t, uint64_t>>
        pipeline_start_end_timestamps;

    // Map the pipeline ID to the number of iterations
    std::unordered_map<pipeline_id_t, uint64_t> pipeline_num_iterations;

    // Timers (OperatorID, OperatorStageID (optional; default 0)) -> Time in us
    std::unordered_map<operator_stage_t, uint64_t> operator_stage_time;

    // Input Row Counts
    std::unordered_map<operator_stage_t, uint64_t>
        operator_stage_input_row_counts;

    // Output Row Counts
    std::unordered_map<operator_stage_t, uint64_t>
        operator_stage_output_row_counts;

    // Operator-Stage specific metrics
    std::unordered_map<operator_stage_t, std::vector<MetricBase>>
        operator_stage_metrics;

    int getTracingLevel() {
        char* tracing_level_env_ = std::getenv("BODO_TRACING_LEVEL");
        if (tracing_level_env_) {
            return atoi(tracing_level_env_);
        } else {
            // If env var is not set default to 0
            return 0;
        }
    }
};
