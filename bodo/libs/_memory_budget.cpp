// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_memory_budget.h"

#include <iostream>

#include "_bodo_common.h"

// Max iterations of the step 1 of the compute satisfiable budgets algorithms.
#define SATISFIABLE_BUDGET_CALC_MAX_ITERS 100000

/**
 * @brief Get string representation of the different Operator types.
 *
 * @param op_type
 * @return std::string
 */
std::string GetOperatorType_as_string(OperatorType const& op_type) {
    if (op_type == OperatorType::UNKNOWN)
        return "UNKNOWN";
    if (op_type == OperatorType::SNOWFLAKE_WRITE)
        return "SNOWFLAKE_WRITE";
    if (op_type == OperatorType::JOIN)
        return "JOIN";
    if (op_type == OperatorType::GROUPBY)
        return "GROUPBY";
    if (op_type == OperatorType::UNION)
        return "UNION";
    if (op_type == OperatorType::ACCUMULATE_TABLE)
        return "ACCUMULATE_TABLE";
    throw std::runtime_error(
        "GetOperatorType_as_string: Uncovered Operator Type!");
}

std::string BytesToHumanReadableString(size_t bytes) {
    std::string db_size;
    auto kibibytes = bytes / 1024;
    auto mebibyte = kibibytes / 1024;
    kibibytes -= mebibyte * 1024;
    auto gibibyte = mebibyte / 1024;
    mebibyte -= gibibyte * 1024;
    auto tebibyte = gibibyte / 1024;
    gibibyte -= tebibyte * 1024;
    auto pebibyte = tebibyte / 1024;
    tebibyte -= pebibyte * 1024;
    if (pebibyte > 0) {
        return std::to_string(pebibyte) + "." + std::to_string(tebibyte / 100) +
               "PiB";
    }
    if (tebibyte > 0) {
        return std::to_string(tebibyte) + "." + std::to_string(gibibyte / 100) +
               "TiB";
    } else if (gibibyte > 0) {
        return std::to_string(gibibyte) + "." + std::to_string(mebibyte / 100) +
               "GiB";
    } else if (mebibyte > 0) {
        return std::to_string(mebibyte) + "." +
               std::to_string(kibibytes / 100) + "MiB";
    } else if (kibibytes > 0) {
        return std::to_string(kibibytes) + "KiB";
    } else {
        return std::to_string(bytes) + (bytes == 1 ? " byte" : " bytes");
    }
}

void OperatorComptroller::Initialize() {
    this->current_pipeline_id = 0;

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

    this->pipeline_remaining_budget.clear();
    this->pipeline_to_remaining_operator_ids.clear();
    this->operator_allocated_budget.clear();

    this->requests_per_operator.clear();

    this->num_pipelines = 0;
    this->num_operators = 0;
}

void OperatorComptroller::SetPipelineMemoryBudget(int64_t pipeline_id,
                                                  size_t budget) {
    if (static_cast<int64_t>(this->pipeline_remaining_budget.size() - 1) <
        pipeline_id) {
        this->pipeline_remaining_budget.resize(pipeline_id + 1);
    }
    this->pipeline_remaining_budget[pipeline_id] = budget;
    this->num_pipelines =
        std::max(this->num_pipelines, static_cast<size_t>(pipeline_id + 1));
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

    if (this->num_operators < static_cast<size_t>(operator_id + 1)) {
        // We have to do this resize here, because we are currently writing to
        // the requests_per_operator map.
        this->operator_allocated_budget.resize(operator_id + 1);
        this->requests_per_operator.resize(operator_id + 1);
        this->num_operators = operator_id + 1;
    }

    this->requests_per_operator[operator_id] = OperatorRequest(
        operator_type, min_pipeline_id, max_pipeline_id, estimate);
    this->num_pipelines =
        std::max(this->num_pipelines, static_cast<size_t>(max_pipeline_id + 1));
}

void OperatorComptroller::IncrementPipelineID() {
    if (this->current_pipeline_id == UNINITIALIZED_PIPELINE_ID) {
        throw std::runtime_error(
            "Initialize() was not called before IncrementPipelineID()");
    }

    this->current_pipeline_id++;
}

int64_t OperatorComptroller::GetOperatorBudget(int64_t operator_id) const {
    if (operator_id < 0) {
        return -1;
    }
    return this->operator_allocated_budget[operator_id];
}

void OperatorComptroller::ReduceOperatorBudget(int64_t operator_id,
                                               size_t budget) {
    if (!OperatorComptroller::memoryBudgetsEnabled()) {
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
    auto old_budget = this->operator_allocated_budget[operator_id];
    auto delta = old_budget - budget;
    auto& req = this->requests_per_operator[operator_id];
    // Set remaining to 0 so that it's not considered in future
    // adjustments, regardless of whether it's a relative
    // request or an absolute one.
    req.rem_estimate = 0;

    // Reduce budget for this operator
    this->operator_allocated_budget[operator_id] = budget;
    // Add the freed budget back to pipeline_remaining_budget so that operators
    // that call IncreaseOperatorBudget can take advantage of it.
    // Remove the operator from its pipelines remaining operator ids list.
    for (const int64_t pipeline_id : req.pipeline_ids) {
        this->pipeline_remaining_budget[pipeline_id] += delta;
        this->pipeline_to_remaining_operator_ids[pipeline_id].erase(
            operator_id);
    }
}

void OperatorComptroller::IncreaseOperatorBudget(int64_t operator_id) {
    if (!OperatorComptroller::memoryBudgetsEnabled()) {
        return;
    }
    const auto& req = this->requests_per_operator[operator_id];
    size_t max_update = std::numeric_limits<size_t>::max();
    // Determine the largest amount of budget we can allocate for this operator
    for (const int64_t pipeline_id : req.pipeline_ids) {
        max_update =
            std::min(max_update, this->pipeline_remaining_budget[pipeline_id]);
    }

    // If we are able to increase the budget, then do so.
    if (max_update) {
        this->operator_allocated_budget[operator_id] += max_update;
        for (const int64_t pipeline_id : req.pipeline_ids) {
            this->pipeline_remaining_budget[pipeline_id] -= max_update;
        }
    }
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
    const std::vector</*const*/ OperatorRequest>& requests_per_operator,
    const std::set<int64_t>& pipeline_operator_ids) {
    std::set<int64_t> abs_req_op_ids;
    std::set<int64_t> rel_req_op_ids;
    size_t abs_req_sum = 0;
    size_t rel_req_sum = 0;
    for (const int64_t op_id : pipeline_operator_ids) {
        const auto& req = requests_per_operator[op_id];
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
        const auto& req = this->requests_per_operator[op_id];
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
    for (size_t op_id = 0; op_id < this->requests_per_operator.size();
         op_id++) {
        const auto& req = this->requests_per_operator[op_id];
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
    if (!OperatorComptroller::memoryBudgetsEnabled()) {
        // Until more work is done to refine the memory estimates, disable
        // memory budgeting by default so that performance doesn't degrade.
        for (int64_t op_id = 0;
             op_id <
             static_cast<int64_t>(this->operator_allocated_budget.size());
             op_id++) {
            this->operator_allocated_budget[op_id] = -1;
        }
        return;
    }

    // Tests might have set their own max budgets.
    if (this->pipeline_remaining_budget.size() == 0) {
        // If the budgets for each pipeline have not been initialized yet
        // (common/default case), initialize it with the total memory available
        // (with some factor applied to the total size to allow for non-budgeted
        // memory usage.
        size_t total_mem = this->memory_usage_fraction *
                           bodo::BufferPool::Default()->get_memory_size_bytes();
        this->pipeline_remaining_budget.resize(this->num_pipelines, total_mem);
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

            // Calculate available budgets for operators with relative
            // budget estimates. Distribute max_rel_req_budget proportionally
            // between them.
            for (const int64_t op_id : rel_req_op_ids) {
                size_t avail_budget_for_op = static_cast<size_t>(
                    ((double)this->requests_per_operator[op_id].rem_estimate /
                     (double)rel_req_sum) *
                    max_rel_req_budget);
                avail_budget_for_ops_in_pipelines[pipeline_id][op_id] =
                    avail_budget_for_op;
            }
        }

        // Calculate and assign available allocations to the operators.
        for (int64_t op_id = 0;
             op_id <
             static_cast<int64_t>(this->operator_allocated_budget.size());
             op_id++) {
            auto& req = this->requests_per_operator[op_id];

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
            if (this->debug_level >= 1) {
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

    if (this->debug_level >= 2) {
        std::cerr
            << "[DEBUG] OperatorComptroller::ComputeSatisfiableBudgets: After "
            << num_iterations << " iterations of step 1:" << std::endl;
        this->PrintBudgetAllocations(std::cerr);
    }

    // STEP 2: Give budget to single pipeline operators.
    // For each pipeline, if there's still some budget left,
    // give it to the operators that are only in that pipeline.
    // First to the abs req ops (if any not fulfilled), and then to the rel req
    // ops. Mark the ops as done after this since there's no more budget
    // to assign to ops that only exist in this pipeline.
    for (size_t pipeline_id = 0; pipeline_id < this->num_pipelines;
         pipeline_id++) {
        // Split the operators in this pipeline into those that
        // provided absolute estimates and those that provided
        // relative estimates. Only select the ones that are only
        // in this pipeline.
        auto [abs_req_op_ids, rel_req_op_ids, abs_req_sum, rel_req_sum] =
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

        // Now do it for relative estimate operators:
        this->assignBudgetProportionalToRemEstimate(
            rel_req_op_ids, rel_req_sum,
            this->pipeline_remaining_budget[pipeline_id]);
    }

    if (this->debug_level >= 2) {
        std::cerr << "[DEBUG] OperatorComptroller::ComputeSatisfiableBudgets: "
                     "After step 2:"
                  << std::endl;
        this->PrintBudgetAllocations(std::cerr);
    }

    // STEP 3: Give budget to multi-pipeline operators.
    // For every op, check if all its pipeline have some budget left. If they
    // do, take the min of the rem budgets from these pipelines
    // (that it belongs to) and assign it that budget.
    for (int64_t op_id = 0;
         op_id < static_cast<int64_t>(this->operator_allocated_budget.size());
         op_id++) {
        auto& req = this->requests_per_operator[op_id];
        if (req.rem_estimate == 0) {
            continue;
        }

        size_t available_allocation = std::numeric_limits<size_t>::max();
        for (const int64_t pipeline_id : req.pipeline_ids) {
            available_allocation =
                std::min(this->pipeline_remaining_budget[pipeline_id],
                         available_allocation);
        }
        this->assignAdditionalBudgetToOperator(op_id, available_allocation);
    }

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Only print on rank-0 if debug level is 1. If debug level is higher than
    // or equal to 2, print it on every rank.
    if (((this->debug_level == 1) && (myrank == 0)) ||
        (this->debug_level >= 2)) {
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
    os << "Number of Pipelines: " << this->num_pipelines << std::endl;
    os << "Number of Operators: " << this->num_operators << "\n\n";
    for (size_t pipeline_id = 0; pipeline_id < this->num_pipelines;
         pipeline_id++) {
        os << "Pipeline " << pipeline_id << " (Remaining Budget: "
           << BytesToHumanReadableString(
                  this->pipeline_remaining_budget[pipeline_id])
           << ")" << std::endl;
        for (int64_t op_id = 0;
             op_id <
             static_cast<int64_t>(this->operator_allocated_budget.size());
             op_id++) {
            const auto& req = this->requests_per_operator[op_id];
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
               << BytesToHumanReadableString(
                      this->operator_allocated_budget[op_id])
               << ")" << std::endl;
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

void init_operator_comptroller() {
    OperatorComptroller::Default()->Initialize();
}

void register_operator(int64_t operator_id, int64_t operator_type,
                       int64_t min_pipeline_id, int64_t max_pipeline_id,
                       int64_t estimate) {
    if (estimate < 0) {
        estimate = bodo::BufferPool::Default()->get_memory_size_bytes();
    }
    OperatorComptroller::Default()->RegisterOperator(
        operator_id, static_cast<OperatorType>(operator_type), min_pipeline_id,
        max_pipeline_id, static_cast<size_t>(estimate));
}

void reduce_operator_budget(int64_t operator_id, size_t new_estimate) {
    OperatorComptroller::Default()->ReduceOperatorBudget(operator_id,
                                                         new_estimate);
}

void increase_operator_budget(int64_t operator_id) {
    OperatorComptroller::Default()->IncreaseOperatorBudget(operator_id);
}

void compute_satisfiable_budgets() {
    OperatorComptroller::Default()->ComputeSatisfiableBudgets();
}

PyMODINIT_FUNC PyInit_memory_budget_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "memory_budget", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_operator_comptroller);
    SetAttrStringFromVoidPtr(m, register_operator);
    SetAttrStringFromVoidPtr(m, reduce_operator_budget);
    SetAttrStringFromVoidPtr(m, increase_operator_budget);
    SetAttrStringFromVoidPtr(m, compute_satisfiable_budgets);
    return m;
}
