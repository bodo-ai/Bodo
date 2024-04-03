// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "_memory.h"

// TODO(aneesh) explore better values for this constant
/// Fraction of the total available memory that the memory budgeting system will
/// use.
#define BODO_MEMORY_BUDGET_DEFAULT_USAGE_FRACTION 0.85

/// All supported streaming operator types. The order here must match the order
/// in _memory_budget.py's OperatorType.
enum class OperatorType {
    UNKNOWN = 0,
    SNOWFLAKE_WRITE,
    JOIN,
    GROUPBY,
    UNION,
    ACCUMULATE_TABLE,
};

/**
 * @brief Struct to keep track of operators' budget estimates and requests.
 *
 */
struct OperatorRequest {
    OperatorType operator_type;
    int64_t min_pipeline_id;
    int64_t max_pipeline_id;
    std::vector<int64_t> pipeline_ids;
    size_t original_estimate;
    size_t rem_estimate;

    OperatorRequest(OperatorType operator_type_, int64_t min_pipeline_id_,
                    int64_t max_pipeline_id_, size_t estimate_)
        : operator_type(operator_type_),
          min_pipeline_id(min_pipeline_id_),
          max_pipeline_id(max_pipeline_id_),
          original_estimate(estimate_),
          rem_estimate(estimate_) {
        if (this->operator_type == OperatorType::JOIN) {
            // All Join partitions are completely spillable/unpinnable.
            // Therefore, we should not block memory allocation in pipelines
            // other than their build and probe pipelines.
            // e.g. Say a join does its build phase in pipeline 0 and its
            // probe phase in pipeline 10. Other Joins or Groupby operators
            // that exist between pipelines 1 and 9 shouldn't be assigned a
            // lower budget and forced to re-partition to allow this join to
            // stay in memory. Pinning the join in pipeline 10 when it's
            // actually needed will lead to better utilization of memory in
            // general. If the other operators don't end up using the extra
            // memory, there's no side-effects since the join state will just
            // stay in memory.
            this->pipeline_ids.push_back(this->min_pipeline_id);
            this->pipeline_ids.push_back(this->max_pipeline_id);
        } else {
            this->pipeline_ids.reserve(this->max_pipeline_id -
                                       this->min_pipeline_id + 1);
            for (int64_t p_id = this->min_pipeline_id;
                 p_id <= this->max_pipeline_id; p_id++) {
                this->pipeline_ids.push_back(p_id);
            }
        }
    }

    // Default constructor for implicitly created operators.
    // We set the pipeline ids as -1 so that they're not
    // included in any pipelines during budget calculations.
    OperatorRequest()
        : operator_type(OperatorType::UNKNOWN),
          min_pipeline_id(-1),
          max_pipeline_id(-1),
          original_estimate(0),
          rem_estimate(0) {}

    /**
     * @brief Is the estimate a relative one.
     * Currently only Snowflake Write asks for an absolute budget.
     * We treat UNKNOWN as an absolute estimate for testing purposes.
     * The rest provide a relative estimate and need to be treated as such.
     *
     * @return true
     * @return false
     */
    inline bool estimate_is_relative() const {
        return (this->operator_type != OperatorType::SNOWFLAKE_WRITE) &&
               (this->operator_type != OperatorType::UNKNOWN);
    }

    /**
     * @brief Is this a single pipeline operator.
     *
     * @return true
     * @return false
     */
    inline bool is_single_pipeline_op() const {
        return this->min_pipeline_id == this->max_pipeline_id;
    }
};

/**
 * @brief
 * - Set of operator IDs with absolute estimates.
 * - Set of operator IDs with relative estimates.
 * - Sum of estimates of operators with absolute estimates.
 * - Sum of estimates of operators with relative estimates.
 */
typedef std::tuple</*abs_req_op_ids*/ std::set<int64_t>,
                   /*rel_req_op_ids*/ std::set<int64_t>,
                   /*abs_req_sum*/ size_t,
                   /*rel_req_sum*/ size_t>
    abs_rel_split_info;

/**
 * @brief Class that manages operator memory budget
 * This class will be available as a singleton through the Default() method.
 */
class OperatorComptroller {
   public:
    /**
     * @brief Get the globally available singleton instance
     */
    static std::shared_ptr<OperatorComptroller> Default() {
        static std::shared_ptr<OperatorComptroller> comptroller_(
            new OperatorComptroller());
        return comptroller_;
    }

    // Called by init_operator_comptroller
    void Initialize();

    /**
     * @brief Initialize and set the total memory budget for all pipeline.
     *
     * @param budget Total budget to set (in bytes)
     */
    void Initialize(size_t budget);

    /**
     * @brief Register an operator and associate it with all pipelines where it
     * will be present with the corresponding estimate.
     */
    void RegisterOperator(int64_t operator_id, OperatorType operator_type,
                          int64_t min_pipeline_id, int64_t max_pipeline_id,
                          size_t estimate);

    /**
     * @brief Get the budget for a given operator in the current pipeline
     * @param operator_id The operator owning the budget returned
     * If the operator_id is invalid (< 0), returns -1 to signify unlimited
     * budget
     */
    int64_t GetOperatorBudget(int64_t operator_id) const;

    /**
     * @brief Reduce the budget for an operator. Note that the new budget must
     * be <= the old budget.
     * @param operator_id The operator to modify
     * @param budget The new budget
     */
    void ReduceOperatorBudget(int64_t operator_id, size_t budget);

    /**
     * @brief Grant additional budget to operator if possible.
     * This API is called by the OperatorBufferPool (for HashJoin and Groupby)
     * to get additional budget at runtime that may be available
     * in their pipelines or that might've been given back by other
     * operators.
     *
     * @param operator_id The operator requesting additional budget.
     * @param addln_budget Maximum additional budget that the operator is
     * requesting. -1 implies it's asking for all possible additional budget
     * that can be allocated to it.
     * @return size_t Additional budget being allocated to this operator.
     */
    size_t RequestAdditionalBudget(int64_t operator_id,
                                   int64_t addln_budget = -1);

    /**
     * @brief Compute budgets for all known operators and pipelines. This must
     * be called only after all calls to RegisterOperator are completed.
     */
    void ComputeSatisfiableBudgets();

    /**
     * @brief Utility function to print the budget allocation state.
     *
     * @param os The stream to print to.
     */
    void PrintBudgetAllocations(std::ostream& os);

    size_t GetNumOperators() const { return num_operators; }

   private:
    /// @brief Total budget for each pipeline. Set during `Initialize`.
    size_t total_budget = 0;
    std::vector<size_t> pipeline_remaining_budget;
    std::vector<std::set<int64_t>> pipeline_to_remaining_operator_ids;
    std::vector<int64_t> operator_allocated_budget;

    std::vector<OperatorRequest> requests_per_operator;

    size_t num_pipelines = 0;
    size_t num_operators = 0;

    size_t debug_level = 0;
    double memory_usage_fraction = BODO_MEMORY_BUDGET_DEFAULT_USAGE_FRACTION;
    bool budgets_enabled = false;

    /// Helpers for ComputeSatisfiableBudgets:

    /**
     * @brief Assign an additional 'addln_budget' bytes of budget to
     * operator with operator-id 'op_id'.
     *
     * This will increase operator_allocated_budget[op_id],
     * decrease pipeline_remaining_budget for all pipelines that this
     * operator is in and update requests_per_operator[op_id].rem_estimate
     * if the estimate is absolute. For absolute requests, if the request is now
     * completely fulfilled, it will also remove op_id from
     * pipeline_to_remaining_operator_ids for all pipelines that this
     * operator is in.
     *
     * @param op_id Operator ID to assign additional budget to.
     * @param addln_budget Additional budget (in bytes) to assign to this
     * operator.
     */
    void assignAdditionalBudgetToOperator(int64_t op_id, size_t addln_budget);

    /**
     * @brief Reset and re-populate this->pipeline_to_remaining_operator_ids
     * based on the current request and allocation state.
     *
     */
    void refreshPipelineToRemainingOperatorIds();

    /**
     * @brief Helper function to split the operators in the pipeline with id
     * 'pipline_id' into those that have relative estimates and those that have
     * absolute estimates. Note that we only return operators whose requests
     * haven't been completely fulfilled yet.
     *
     * @param pipeline_id Pipeline ID of the pipeline to get this information
     * for.
     * @return abs_rel_split_info
     */
    abs_rel_split_info splitRemainingOperatorsIntoAbsoluteAndRelative(
        int64_t pipeline_id) const;

    /**
     * @brief Same as splitRemainingOperatorsIntoAbsoluteAndRelative, except
     * we only consider operators that only exist in pipeline with id
     * 'pipeline_id'.
     *
     * @param pipeline_id Pipeline ID of the pipeline to get this information
     * for.
     * @return abs_rel_split_info
     */
    abs_rel_split_info
    splitRemainingSinglePipelineOperatorsIntoAbsoluteAndRelative(
        int64_t pipeline_id) const;

    /**
     * @brief Assign 'budget' many bytes among the operators (corresponding to
     * 'op_ids') proportional to their remaining estimates.
     *
     * @param op_ids Operators to distribute the budget between.
     * @param ops_rem_est_sum Sum of the remaining estimates of the operators.
     * @param budget Budget to distribute between the operators.
     */
    inline void assignBudgetProportionalToRemEstimate(std::set<int64_t> op_ids,
                                                      size_t ops_rem_est_sum,
                                                      size_t budget);

    bool memoryBudgetsEnabledHelper() {
        char* use_mem_budget_env_ = std::getenv("BODO_USE_MEMORY_BUDGETS");
        if (use_mem_budget_env_) {
            // Use operator budgets based on the env var value if
            // the env var is set. This is primarily for testing
            // purposes.
            if (strcmp(use_mem_budget_env_, "1") == 0) {
                return true;
            } else if (strcmp(use_mem_budget_env_, "0") == 0) {
                return false;
            } else {
                throw std::runtime_error(
                    "BODO_USE_MEMORY_BUDGETS set to unsupported value: " +
                    std::string(use_mem_budget_env_));
            }
        } else {
            // If env var is not set (default case), turn on operator
            // budgets when spilling is enabled and turn them off
            // otherwise.
            return bodo::BufferPool::Default()->is_spilling_enabled();
        }
    }
};
