// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_memory_budget.h"

#include <iostream>

#include "mpi.h"

#include "./_utils.h"

// Max iterations of the step 1 of the compute satisfiable budgets algorithms.
#define SATISFIABLE_BUDGET_CALC_MAX_ITERS 100000

// Minimum budget to assign to every operator with relative estimates.
#define DEFAULT_MIN_BUDGET_REL_OPS 16L * 1024 * 1024  // 16MiB

/**
 * @brief Get string representation of the different Operator types.
 *
 * @param op_type
 * @return std::string
 */
std::string GetOperatorType_as_string(OperatorType const& op_type) {
    switch (op_type) {
        case OperatorType::UNKNOWN:
            return "UNKNOWN";
        case OperatorType::SNOWFLAKE_WRITE:
            return "SNOWFLAKE_WRITE";
        case OperatorType::JOIN:
            return "JOIN";
        case OperatorType::GROUPBY:
            return "GROUPBY";
        case OperatorType::UNION:
            return "UNION";
        case OperatorType::ACCUMULATE_TABLE:
            return "ACCUMULATE_TABLE";
        case OperatorType::WINDOW:
            return "WINDOW";
        default:
            throw std::runtime_error(
                "GetOperatorType_as_string: Uncovered Operator Type!");
    }
}

void OperatorComptroller::Initialize() {
    this->debug_level = 0;
    if (char* debug_level_env_ =
            std::getenv("BODO_MEMORY_BUDGETS_DEBUG_LEVEL")) {
        this->debug_level = static_cast<size_t>(std::stoi(debug_level_env_));
    }

    this->memory_usage_fraction = BODO_MEMORY_BUDGET_DEFAULT_USAGE_FRACTION;
    if (char* mem_percent_env_ =
            std::getenv("BODO_MEMORY_BUDGETS_USAGE_PERCENT")) {
        this->memory_usage_fraction =
            static_cast<double>(std::stoi(mem_percent_env_) / 100.0);
    }

    this->budgets_enabled = this->memoryBudgetsEnabledHelper();

    // If a budget is not explicitly specified (common/default case), initialize
    // it with the total memory available (with some factor applied to the total
    // size to allow for non-budgeted memory usage.
    this->total_budget = this->memory_usage_fraction *
                         bodo::BufferPool::Default()->get_memory_size_bytes();

    this->pipeline_remaining_budget.clear();
    this->pipeline_to_remaining_operator_ids.clear();
    this->operator_allocated_budget.clear();

    this->requests_per_operator.clear();

    this->num_pipelines = 0;
    this->num_operators = 0;
}

void OperatorComptroller::Initialize(size_t budget) {
    // Initialize the usual.
    this->Initialize();
    // Set the budget based on the provided value
    this->total_budget = budget;
}

void OperatorComptroller::RegisterOperator(int64_t operator_id,
                                           OperatorType operator_type,
                                           int64_t min_pipeline_id,
                                           int64_t max_pipeline_id,
                                           size_t estimate) {
    if (max_pipeline_id < min_pipeline_id) {
        // this should never happen - if it does, something is wrong on the
        // BodoSQL side.
        throw std::runtime_error(
            "OperatorComptroller::RegisterOperator: max_pipeline_id cannot be "
            "less than min_pipeline_id");
    }

    this->num_operators++;
    this->requests_per_operator[operator_id] = OperatorRequest(
        operator_type, min_pipeline_id, max_pipeline_id, estimate);
    this->operator_allocated_budget[operator_id] = 0;
    this->num_pipelines =
        std::max(this->num_pipelines, static_cast<size_t>(max_pipeline_id + 1));
}

int64_t OperatorComptroller::GetOperatorBudget(int64_t operator_id) const {
    if (!this->budgets_enabled || operator_id < 0) {
        return -1;
    }
    return this->operator_allocated_budget.at(operator_id);
}

void OperatorComptroller::ReduceOperatorBudget(int64_t operator_id,
                                               size_t budget) {
    if (!this->budgets_enabled || (operator_id < 0)) {
        return;
    }

    if (this->operator_allocated_budget[operator_id] <=
        static_cast<int64_t>(budget)) {
        std::cerr << "OperatorComptroller::ReduceOperatorBudget: New budget "
                     "for operator "
                  << operator_id << " is not strictly less than old budget"
                  << std::endl;
        return;
    }
    int64_t old_budget = this->operator_allocated_budget[operator_id];
    int64_t delta = old_budget - static_cast<int64_t>(budget);
    auto& req = this->requests_per_operator[operator_id];
    // Set remaining to 0 so that it's not considered in future
    // adjustments, regardless of whether it's a relative
    // request or an absolute one.
    req.rem_estimate = 0;

    // Reduce budget for this operator
    this->operator_allocated_budget[operator_id] = budget;
    // Add the freed budget back to pipeline_remaining_budget so that operators
    // that call RequestAdditionalBudget can take advantage of it.
    // Remove the operator from its pipelines remaining operator ids list.
    for (const int64_t pipeline_id : req.pipeline_ids) {
        this->pipeline_remaining_budget[pipeline_id] += delta;
        this->pipeline_to_remaining_operator_ids[pipeline_id].erase(
            operator_id);
    }

    // Log this budget update.
    if (this->debug_level >= 1) {
        std::cerr << "[DEBUG] OperatorComptroller::ReduceOperatorBudget: "
                     "Reduced budget for operator "
                  << operator_id << " from "
                  << BytesToHumanReadableString(old_budget) << " to "
                  << BytesToHumanReadableString(budget) << "." << std::endl;
    }
}

size_t OperatorComptroller::RequestAdditionalBudget(int64_t operator_id,
                                                    int64_t addln_budget) {
    if (!this->budgets_enabled || (operator_id < 0)) {
        return 0;
    }
    if (addln_budget < -1) {
        throw std::runtime_error(
            "OperatorComptroller::RequestAdditionalBudget called with "
            "addln_budget < -1!");
    }
    const auto& req = this->requests_per_operator[operator_id];
    size_t max_possible_update = std::numeric_limits<size_t>::max();
    // Determine the largest amount of budget we can allocate for this operator
    for (const int64_t pipeline_id : req.pipeline_ids) {
        max_possible_update = std::min(
            max_possible_update, this->pipeline_remaining_budget[pipeline_id]);
    }

    size_t addln_assigned_budget = 0;
    if (addln_budget != -1) {
        // If the operator asked for a specific amount, try to fulfill it
        // completely:
        addln_assigned_budget =
            std::min(max_possible_update, static_cast<size_t>(addln_budget));
    } else {
        // If the operator is asking for more memory in general, we will assign
        // it all of the remaining budget.
        // Currently, only Join-Build or Groupby-Build can request additional
        // budget. Only one of them can exist in one pipeline, so there is no
        // possibility of sharing the remaining budget at this point. We may
        // change this behavior in the future as more operators are supported
        // and more combinations are possible.
        addln_assigned_budget = max_possible_update;
    }

    // If we are able to increase the budget, then do so.
    if (addln_assigned_budget > 0) {
        const int64_t old_budget = this->operator_allocated_budget[operator_id];
        this->operator_allocated_budget[operator_id] += addln_assigned_budget;
        for (const int64_t pipeline_id : req.pipeline_ids) {
            this->pipeline_remaining_budget[pipeline_id] -=
                addln_assigned_budget;
        }
        // Log this budget update.
        if (this->debug_level >= 1) {
            std::cerr
                << "[DEBUG] OperatorComptroller::RequestAdditionalBudget: "
                   "Increased budget for operator "
                << operator_id << " by "
                << BytesToHumanReadableString(addln_assigned_budget)
                << " (from " << BytesToHumanReadableString(old_budget) << " to "
                << BytesToHumanReadableString(
                       this->operator_allocated_budget[operator_id])
                << ")." << std::endl;
        }
    }

    return addln_assigned_budget;
}

void OperatorComptroller::assignAdditionalBudgetToOperator(
    int64_t op_id, size_t addln_budget) {
    auto& req = this->requests_per_operator[op_id];
    /// Perform verification checks before making any changes to the state.

    // 1. Verify that all pipelines have required budget
    for (const int64_t pipeline_id : req.pipeline_ids) {
        if (this->pipeline_remaining_budget[pipeline_id] < addln_budget) {
            throw std::runtime_error(
                "OperatorComptroller::assignAdditionalBudgetToOperator: "
                "Pipeline " +
                std::to_string(pipeline_id) +
                " doesn't have the required remaining budget (required: " +
                std::to_string(addln_budget) + " vs available: " +
                std::to_string(this->pipeline_remaining_budget[pipeline_id]) +
                ").");
        }
    }

    // 2. Verify that addln_budget wouldn't take it over the original estimate
    // in the abs case.
    if (!req.estimate_is_relative() && (req.rem_estimate < addln_budget)) {
        throw std::runtime_error(
            "OperatorComptroller::assignAdditionalBudgetToOperator: Additional "
            "budget (" +
            std::to_string(addln_budget) +
            ") is greater than remaining estimate (" +
            std::to_string(req.rem_estimate) + ") for the operator.");
    }

    /// Update state:

    // Update the memory allocated for this operator
    this->operator_allocated_budget[op_id] += addln_budget;
    // Update the memory remaining in the request (only if it is an
    // absolute request)
    if (!req.estimate_is_relative()) {
        req.rem_estimate -= addln_budget;
    }

    for (const int64_t pipeline_id : req.pipeline_ids) {
        // Mark memory as used in all pipelines with this operator
        this->pipeline_remaining_budget[pipeline_id] -= addln_budget;
        // If this request is completely satisfied, remove it from
        // the remaining operators for this pipeline.
        // This only ever happens for operators with absolute
        // requests.
        if (req.rem_estimate == 0) {
            this->pipeline_to_remaining_operator_ids[pipeline_id].erase(op_id);
        }
    }
}

abs_rel_split_info splitOperatorsIntoAbsoluteAndRelativeHelper(
    const std::unordered_map<int64_t, /*const*/ OperatorRequest>&
        requests_per_operator,
    const std::set<int64_t>& pipeline_operator_ids) {
    std::set<int64_t> abs_req_op_ids;
    std::set<int64_t> rel_req_op_ids;
    size_t abs_req_sum = 0;
    size_t rel_req_sum = 0;
    for (const int64_t op_id : pipeline_operator_ids) {
        const auto& req = requests_per_operator.at(op_id);
        if (!req.estimate_is_relative()) {
            abs_req_op_ids.insert(op_id);
            abs_req_sum += req.rem_estimate;
        } else {
            rel_req_op_ids.insert(op_id);
            rel_req_sum += req.rem_estimate;
        }
    }
    return std::make_tuple(abs_req_op_ids, rel_req_op_ids, abs_req_sum,
                           rel_req_sum);
}

abs_rel_split_info
OperatorComptroller::splitRemainingOperatorsIntoAbsoluteAndRelative(
    int64_t pipeline_id) const {
    return splitOperatorsIntoAbsoluteAndRelativeHelper(
        this->requests_per_operator,
        this->pipeline_to_remaining_operator_ids[pipeline_id]);
}

abs_rel_split_info OperatorComptroller::
    splitRemainingSinglePipelineOperatorsIntoAbsoluteAndRelative(
        int64_t pipeline_id) const {
    std::set<int64_t> single_pipeline_op_ids;
    for (const int64_t op_id :
         this->pipeline_to_remaining_operator_ids[pipeline_id]) {
        const auto& req = requests_per_operator.at(op_id);
        if (req.is_single_pipeline_op()) {
            assert(req.min_pipeline_id == pipeline_id);
            single_pipeline_op_ids.insert(op_id);
        }
    }
    return splitOperatorsIntoAbsoluteAndRelativeHelper(
        this->requests_per_operator, single_pipeline_op_ids);
}

void OperatorComptroller::refreshPipelineToRemainingOperatorIds() {
    this->pipeline_to_remaining_operator_ids.clear();
    this->pipeline_to_remaining_operator_ids.resize(this->num_pipelines);
    for (auto& [op_id, req] : requests_per_operator) {
        if (req.rem_estimate > 0) {
            for (const int64_t pipeline_id : req.pipeline_ids) {
                this->pipeline_to_remaining_operator_ids[pipeline_id].insert(
                    op_id);
            }
        }
    }
}

inline void OperatorComptroller::assignBudgetProportionalToRemEstimate(
    std::set<int64_t> op_ids, size_t ops_rem_est_sum, size_t budget) {
    for (const int64_t op_id : op_ids) {
        const auto& req = this->requests_per_operator[op_id];
        size_t avail_budget = static_cast<size_t>(
            ((double)req.rem_estimate / (double)ops_rem_est_sum) * budget);
        this->assignAdditionalBudgetToOperator(op_id, avail_budget);
    }
}

void OperatorComptroller::ComputeSatisfiableBudgets() {
    if (!this->budgets_enabled) {
        // Until more work is done to refine the memory estimates, disable
        // memory budgeting by default so that performance doesn't degrade.
        for (auto& [op_id, req] : this->operator_allocated_budget) {
            req = -1;
        }
        return;
    }

    // Tests might have set their own max budgets.
    if (this->pipeline_remaining_budget.size() == 0) {
        // If the budgets for the pipelines has not been initialized yet
        // (common/default case), initialize it with this->total_budget (which
        // is set during Initialize)
        this->pipeline_remaining_budget.resize(this->num_pipelines,
                                               this->total_budget);
    } else {
        if (this->pipeline_remaining_budget.size() !=
            static_cast<size_t>(this->num_pipelines)) {
            throw std::runtime_error(
                "OperatorComptroller::ComputeSatisfiableBudgets: Either all "
                "pipeline budgets must be initialized manually or none of them "
                "should be!");
        }
    }

    // Reset pipeline_to_remaining_operator_ids:
    this->refreshPipelineToRemainingOperatorIds();

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // STEP 1: Do an iterative algorithm to distribute budgets.
    size_t num_iterations = 0;
    while (true) {
        bool changed = false;

        // Track available budgets for operators in different
        // pipelines:
        std::vector<
            std::unordered_map</*op_id*/ int64_t, /*avail_budget*/ size_t>>
            avail_budget_for_ops_in_pipelines(this->num_pipelines);

        // Populate avail_budget_for_ops_in_pipelines:
        for (size_t pipeline_id = 0; pipeline_id < this->num_pipelines;
             pipeline_id++) {
            // Split the operators in this pipeline into those that
            // provided absolute estimates and those that provided
            // relative estimates
            auto [abs_req_op_ids, rel_req_op_ids, abs_req_sum, rel_req_sum] =
                this->splitRemainingOperatorsIntoAbsoluteAndRelative(
                    pipeline_id);

            // Split the available budget between absolute and relative requests
            // based on the number of operators belonging to each kind.
            size_t max_abs_req_budget = static_cast<size_t>(
                ((double)abs_req_op_ids.size() /
                 (double)(this->pipeline_to_remaining_operator_ids[pipeline_id]
                              .size())) *
                this->pipeline_remaining_budget[pipeline_id]);
            size_t max_rel_req_budget =
                this->pipeline_remaining_budget[pipeline_id] -
                max_abs_req_budget;

            // Calculate available allocation for operators with
            // absolute estimates.
            // Since the request allocations will be proportional,
            // either all absolute requests can be completely fulfilled
            // or none of them can be.
            if (max_abs_req_budget >= abs_req_sum) {
                // All operator requests can be fulfilled
                for (const int64_t op_id : abs_req_op_ids) {
                    avail_budget_for_ops_in_pipelines[pipeline_id][op_id] =
                        this->requests_per_operator[op_id].rem_estimate;
                }
                // We have more budget to give to the operators with
                // relative estimates:
                max_rel_req_budget += (max_abs_req_budget - abs_req_sum);
            } else {
                // None of the requests can be completely fulfilled. Spread
                // available budget proportionally.
                for (const int64_t op_id : abs_req_op_ids) {
                    size_t avail_budget_for_op = static_cast<size_t>(
                        ((double)this->requests_per_operator[op_id]
                             .rem_estimate /
                         (double)abs_req_sum) *
                        max_abs_req_budget);
                    avail_budget_for_ops_in_pipelines[pipeline_id][op_id] =
                        avail_budget_for_op;
                }
            }

            // Initialize avail_budget_for_ops_in_pipelines[pipeline_id]
            // for relative est ops.
            for (const int64_t op_id : rel_req_op_ids) {
                avail_budget_for_ops_in_pipelines[pipeline_id][op_id] = 0;
            }

            // Calc min budget for the relative operators:
            // min(16MiB,
            //     min(0.01, 1/num_rel_ops_in_pipeline) *
            //     max_rel_req_budget
            //    )
            size_t min_rel_op_allocation = std::min(
                static_cast<size_t>(DEFAULT_MIN_BUDGET_REL_OPS),
                static_cast<size_t>(
                    std::min<double>(0.01, (1.0 / rel_req_op_ids.size())) *
                    max_rel_req_budget));

            // Allocate min_rel_op_allocation to each operator up-front (if it's
            // not already allocated)
            for (const int64_t op_id : rel_req_op_ids) {
                if (static_cast<size_t>(
                        this->operator_allocated_budget[op_id]) <
                    min_rel_op_allocation) {
                    size_t rem_alloc =
                        min_rel_op_allocation -
                        static_cast<size_t>(
                            this->operator_allocated_budget[op_id]);
                    avail_budget_for_ops_in_pipelines[pipeline_id][op_id] =
                        rem_alloc;
                    max_rel_req_budget -= rem_alloc;
                }
            }

            // Calculate available budgets for operators with relative
            // budget estimates. Distribute max_rel_req_budget proportionally
            // between them.
            for (const int64_t op_id : rel_req_op_ids) {
                size_t avail_budget_for_op = static_cast<size_t>(
                    ((double)this->requests_per_operator[op_id].rem_estimate /
                     (double)rel_req_sum) *
                    max_rel_req_budget);
                avail_budget_for_ops_in_pipelines[pipeline_id][op_id] +=
                    avail_budget_for_op;
            }
        }

        // Calculate and assign available allocations to the operators.
        for (auto& [op_id, req] : this->requests_per_operator) {
            // If already fulfilled, then skip updates.
            // (This can only happen for operators with absolute estimates).
            if (req.rem_estimate == 0) {
                continue;
            }

            // Calculate minimum available allocation for the operator across
            // all its pipelines:
            size_t available_allocation = std::numeric_limits<size_t>::max();
            for (const int64_t pipeline_id : req.pipeline_ids) {
                available_allocation = std::min(
                    avail_budget_for_ops_in_pipelines[pipeline_id][op_id],
                    available_allocation);
            }

            // If there's additional allocation available, allocate it:
            if (available_allocation > 0) {
                this->assignAdditionalBudgetToOperator(op_id,
                                                       available_allocation);
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
        num_iterations++;
        if (num_iterations > SATISFIABLE_BUDGET_CALC_MAX_ITERS) {
            if ((this->debug_level >= 1) && (myrank == 0)) {
                std::cerr << "[WARNING] "
                             "OperatorComptroller::ComputeSatisfiableBudgets: "
                             "Could not converge on a budget in "
                          << SATISFIABLE_BUDGET_CALC_MAX_ITERS
                          << " iterations! Breaking out early to avoid hangs."
                          << std::endl;
            }
            break;
        }
    }

    if ((this->debug_level >= 2) && (myrank == 0)) {
        std::cerr
            << "[DEBUG] OperatorComptroller::ComputeSatisfiableBudgets: After "
            << num_iterations << " iterations of step 1:" << std::endl;
        this->PrintBudgetAllocations(std::cerr);
    }

    // STEP 2: Give budget to single pipeline operators.
    // For each pipeline, if there's still some budget left,
    // give it to the operators that are only in that pipeline.
    // We only do it for the abs req ops (if any not fulfilled),
    // since the rel req ops can request additional budget dynamically.
    // We will mark the ops as done after this since there's no more budget
    // to assign to ops that only exist in this pipeline.
    for (size_t pipeline_id = 0; pipeline_id < this->num_pipelines;
         pipeline_id++) {
        // Split the operators in this pipeline into those that
        // provided absolute estimates and those that provided
        // relative estimates. Only select the ones that are only
        // in this pipeline.
        auto [abs_req_op_ids, _, abs_req_sum, __] =
            this->splitRemainingSinglePipelineOperatorsIntoAbsoluteAndRelative(
                pipeline_id);

        // Assign remaining budget to the absolute estimates first:
        if (this->pipeline_remaining_budget[pipeline_id] >= abs_req_sum) {
            // This means that all operator requests can be
            // fulfilled
            for (const int64_t op_id : abs_req_op_ids) {
                const auto& req = this->requests_per_operator[op_id];
                size_t avail_budget = req.rem_estimate;
                this->assignAdditionalBudgetToOperator(op_id, avail_budget);
            }
        } else {
            // None of the requests can be completely fulfilled:
            this->assignBudgetProportionalToRemEstimate(
                abs_req_op_ids, abs_req_sum,
                this->pipeline_remaining_budget[pipeline_id]);
        }
    }

    // Only print on rank-0.
    if ((this->debug_level >= 1) && (myrank == 0)) {
        std::cerr << "[DEBUG] OperatorComptroller::ComputeSatisfiableBudgets: "
                     "Final budget allocation:"
                  << std::endl;
        this->PrintBudgetAllocations(std::cerr);
    }
}

void OperatorComptroller::PrintBudgetAllocations(std::ostream& os) {
    os << "\n======================================= MEMORY BUDGET "
          "======================================="
       << std::endl;
    if (!this->budgets_enabled) {
        os << "NOT ENABLED." << std::endl;
        os << "================================================================"
              "============================="
           << std::endl;
        return;
    }
    os << "Number of Pipelines: " << this->num_pipelines << std::endl;
    os << "Number of Operators: " << this->num_operators << std::endl;
    os << "Total Budget: " << BytesToHumanReadableString(this->total_budget)
       << "\n\n";
    for (size_t pipeline_id = 0; pipeline_id < this->num_pipelines;
         pipeline_id++) {
        os << "Pipeline " << pipeline_id << " (Remaining Budget: "
           << BytesToHumanReadableString(
                  this->pipeline_remaining_budget[pipeline_id])
           << ")" << std::endl;
        for (auto& [op_id, avail_budget] : operator_allocated_budget) {
            const auto& req = requests_per_operator[op_id];
            if ((static_cast<int64_t>(pipeline_id) < req.min_pipeline_id) ||
                (req.max_pipeline_id < static_cast<int64_t>(pipeline_id))) {
                // If not a part of this pipeline, then skip.
                // This is wasteful, but fine for now.
                continue;
            }
            if ((req.operator_type == OperatorType::JOIN) &&
                (req.min_pipeline_id != static_cast<int64_t>(pipeline_id)) &&
                (req.max_pipeline_id != static_cast<int64_t>(pipeline_id))) {
                // Special case for join.
                continue;
            }
            // "- <OPERATOR_TYPE> (Operator ID: <OPERATOR_ID>, Original
            // Estimate: <ORIGINAL_ESTIMATE>, Allocated Budget:
            // <ALLOCATED_BUDGET>)"
            os << " - " << GetOperatorType_as_string(req.operator_type)
               << " (Operator ID: " << op_id << ", Original estimate: "
               << BytesToHumanReadableString(req.original_estimate)
               << (req.estimate_is_relative() ? " (relative)" : "")
               << ", Allocated budget: "
               << BytesToHumanReadableString(avail_budget) << ")" << std::endl;
            if (req.min_pipeline_id != req.max_pipeline_id) {
                if (req.operator_type == OperatorType::JOIN) {
                    os << "   - Build Pipeline ID: " << req.min_pipeline_id
                       << ", Probe Pipeline ID: " << req.max_pipeline_id << "."
                       << std::endl;
                } else {
                    os << "   - Min Pipeline ID: " << req.min_pipeline_id
                       << ", Max Pipeline ID: " << req.max_pipeline_id << "."
                       << std::endl;
                }
            }
        }
        os << std::endl;
    }
    os << "===================================================================="
          "========================="
       << std::endl;
}
