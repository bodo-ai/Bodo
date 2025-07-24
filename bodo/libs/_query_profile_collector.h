#pragma once
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// Forward declare boost::json::object to avoid including the entire header and
// increasing compile times
namespace boost::json {
class object;
}

/**
 * @brief Types of operator metrics that can be collected
 */
struct MetricTypes {
    enum TypeEnum { TIMER = 0, STAT = 1, BLOB = 2 };

    static std::string ToString(TypeEnum type) {
        switch (type) {
            case TIMER:
                return "TIMER";
            case STAT:
                return "STAT";
            case BLOB:
                return "BLOB";
            default:
                throw std::runtime_error(
                    "MetricTypes::ToString: Unsupported type!");
        }
    }
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
    // Use the enum as a key to get the type for the specific metric from the
    // underlying variant
    using ValueType = decltype(std::get<type_>(value));

    Metric() = delete;
    Metric(ValueType val) {
        type = type_;
        set(val);
    }
    Metric(std::string name, ValueType val) : Metric(val) { this->name = name; }
    Metric(std::string name, ValueType val, bool global) : Metric(name, val) {
        this->is_global = global;
    }

    // getter and setter for the metric value
    ValueType get() { return std::get<type_>(value); }
    void set(ValueType val) { value = val; }
};

// Defined metric types
using TimerMetric = Metric<MetricTypes::TIMER>;
using StatMetric = Metric<MetricTypes::STAT>;
using BlobMetric = Metric<MetricTypes::BLOB>;

// Short-hand for ease of use in operator code.
using time_pt = std::chrono::steady_clock::time_point;

/**
 * @brief Helper function for starting a timer.
 *
 * @return time_pt
 */
inline time_pt start_timer() noexcept {
    return std::chrono::steady_clock::now();
}

/**
 * @brief Helper function for ending a timer and getting the elapsed time (in
 * microseconds).
 * Example usage:
 *
 * ```
 * time_pt start = start_timer();
 * <DO WORK>
 * MetricBase::TimerValue elapsed_time = end_timer(start);
 * ```
 *
 * @param start_time_pt Time when timer was started.
 * @return MetricBase::TimerValue Time (in us) since the start of the timer.
 */
inline MetricBase::TimerValue end_timer(const time_pt& start_time_pt) noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start_time_pt)
        .count();
}

/**
 * @brief Scoped timer for cases where we want to measure time even when there
 * might be RuntimeErrors. This is useful in Join/Groupby where we may encounter
 * threshold-exceeded exceptions from the OperatorPool. In these cases, we don't
 * want to lose the timing information.
 * Note that this has a higher overhead than adding timers using 'start_timer()'
 * and 'end_timer(start)', so we should use these judiciously.
 *
 */
class ScopedTimer {
   public:
    /**
     * @brief Construct a new Scoped Timer object.
     *
     * @param to_update_ Reference to the timer value to update at the end of
     * the timer scope (or when finalize is called).
     */
    inline ScopedTimer(MetricBase::TimerValue& to_update_) noexcept
        : to_update(to_update_), start_time_pt(start_timer()) {}
    inline void finalize() noexcept {
        if (!this->finalized) {
            this->to_update += end_timer(this->start_time_pt);
            finalized = true;
        }
    }
    inline ~ScopedTimer() noexcept { finalize(); }

   private:
    MetricBase::TimerValue& to_update;
    const time_pt start_time_pt;
    bool finalized = false;
};

struct operator_stage {
    uint32_t operator_id;
    uint32_t stage_id;

    bool operator==(const operator_stage& other) const {
        return operator_id == other.operator_id && stage_id == other.stage_id;
    }
};

template <>
struct std::hash<operator_stage> {
    std::size_t operator()(const operator_stage& op_stage) const {
        return std::hash<uint32_t>()(op_stage.operator_id) ^
               std::hash<uint32_t>()(op_stage.stage_id);
    }
};

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

    using operator_id_t = uint32_t;
    using stage_id_t = uint32_t;
    using operator_stage_t = struct ::operator_stage;
    using pipeline_id_t = uint32_t;

    /**
     * @brief Create an operator stage ID from an operator ID and stage ID
     *
     * This method packs two 32 bit ids into a single 64 bit id
     *
     * @param operator_id The operator ID
     * @param stage_id The stage ID
     * @return operator_stage_t The operator stage ID
     */
    static operator_stage_t MakeOperatorStageID(operator_id_t operator_id,
                                                stage_id_t stage_id) {
        return operator_stage_t{operator_id, stage_id};
    }

    void Init();
    void StartPipeline(pipeline_id_t pipeline_id);
    void EndPipeline(pipeline_id_t pipeline_id, size_t num_iterations);
    void SubmitOperatorStageRowCounts(operator_stage_t op_stage,
                                      uint64_t output_row_count);
    void SubmitOperatorStageTime(operator_stage_t op_stage, int64_t time);
    int64_t GetOperatorDuration(operator_id_t operator_id);
    void SubmitOperatorName(operator_id_t operator_id, const std::string& name);

    /**
     * @brief This is only required by C++ at this point since
     * only operators with states in C++ will use this.
     *
     * @param op_stage id of the operator and stage to report metrics for
     * @param metrics metrics to report
     */
    void RegisterOperatorStageMetrics(operator_stage_t op_stage,
                                      std::vector<MetricBase> metrics);

    void Finalize(int64_t verbose_level = 0);

    // Getters for testing
    std::unordered_map<pipeline_id_t, std::pair<uint64_t, uint64_t>>&
    GetPipelineStartEndTimestamps() {
        return pipeline_start_end_timestamps;
    }

    std::unordered_map<pipeline_id_t, uint64_t>& GetPipelineNumIterations() {
        return pipeline_num_iterations;
    }

    std::unordered_map<operator_stage_t, uint64_t>&
    GetOperatorStageOutputRowCounts() {
        return operator_stage_output_row_counts;
    }

    std::unordered_map<operator_stage_t, std::vector<MetricBase>>&
    GetMetrics() {
        return operator_stage_metrics;
    }

   private:
    int tracing_level = 1;

    // Map the pipeline ID to its start and end timestamps
    std::unordered_map<pipeline_id_t, std::pair<uint64_t, uint64_t>>
        pipeline_start_end_timestamps;

    // Map the pipeline ID to the number of iterations
    std::unordered_map<pipeline_id_t, uint64_t> pipeline_num_iterations;

    // Map the operator stage ID to its start and end timestamps
    std::unordered_map<operator_stage_t, int64_t> operator_stage_times;

    // Map the operator ID to its name
    std::unordered_map<operator_id_t, std::string> operator_names;

    // Output Row Counts
    std::unordered_map<operator_stage_t, uint64_t>
        operator_stage_output_row_counts;

    // Operator-Stage specific metrics
    std::unordered_map<operator_stage_t, std::vector<MetricBase>>
        operator_stage_metrics;

    // Get a map from all seen operators ids to the largest observed stage
    std::unordered_map<operator_id_t, stage_id_t> CollectSeenOperators();

    // Generate report JSON for all pipelines
    boost::json::object PipelinesToJson();
    // Generate report JSON for a single operator stage
    boost::json::object OperatorStageToJson(operator_stage_t op_stage);
    // Generate report JSON for all stages of an operator
    boost::json::object OperatorToJson(operator_id_t op, stage_id_t max_stage);

    std::unordered_map<int64_t, int64_t> initial_operator_budget;

    // Location to write output profiles
    std::string output_dir;

    int getTracingLevel() {
        char* tracing_level_env_ = std::getenv("BODO_TRACING_LEVEL");
        if (tracing_level_env_) {
            return atoi(tracing_level_env_);
        } else {
            // If env var is not set default to 1. this is consistent with the
            // python equivalent - see the definition of
            // tracing_level in bodo/__init__.py
            return 1;
        }
    }
};
