// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_memory_budget.h"

#include <iostream>

#include "_bodo_common.h"

void OperatorComptroller::Initialize() { this->current_pipeline_id = 0; }

void OperatorComptroller::Reset() {
    this->current_pipeline_id = UNINITIALIZED_PIPELINE_ID;

    this->pipeline_remaining_budget.clear();
    this->pipeline_remaining_operators.clear();
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
    this->num_pipelines = std::max(this->num_pipelines, pipeline_id + 1);
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

    if (this->num_operators < (operator_id + 1)) {
        // We have to do this resize here, because we are currently writing to
        // the requests_per_operator map.
        this->operator_allocated_budget.resize(operator_id + 1);
        this->requests_per_operator.resize(operator_id + 1);
        this->num_operators = operator_id + 1;
    }

    this->requests_per_operator[operator_id] = OperatorRequest{
        operator_type, min_pipeline_id, max_pipeline_id, estimate};
    this->num_pipelines = std::max(this->num_pipelines, max_pipeline_id + 1);
}

void OperatorComptroller::IncrementPipelineID() {
    if (this->current_pipeline_id == UNINITIALIZED_PIPELINE_ID) {
        throw std::runtime_error(
            "Initialize() was not called before IncrementPipelineID()");
    }

    this->current_pipeline_id++;
}

int64_t OperatorComptroller::GetOperatorBudget(int64_t operator_id) {
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
    const auto& req = this->requests_per_operator[operator_id];

    // Reduce budget for this operator
    this->operator_allocated_budget[operator_id] = budget;
    // Add the freed budget back to pipeline_remaining_budget so that operators
    // that call IncreaseOperatorBudget can take advantage of it
    for (int64_t pipeline_id = req.min_pipeline_id;
         pipeline_id <= req.max_pipeline_id; pipeline_id++) {
        this->pipeline_remaining_budget[pipeline_id] += delta;
    }
}

void OperatorComptroller::IncreaseOperatorBudget(int64_t operator_id) {
    const auto& req = this->requests_per_operator[operator_id];
    size_t max_update = std::numeric_limits<size_t>::max();
    // Determine the largest amount of budget we can allocate for this operator
    for (int64_t pipeline_id = req.min_pipeline_id;
         pipeline_id <= req.max_pipeline_id; pipeline_id++) {
        max_update =
            std::min(max_update, this->pipeline_remaining_budget[pipeline_id]);
    }

    // If we are able to increase the budget, then do so.
    if (max_update) {
        this->operator_allocated_budget[operator_id] += max_update;
        for (int64_t pipeline_id = req.min_pipeline_id;
             pipeline_id <= req.max_pipeline_id; pipeline_id++) {
            this->pipeline_remaining_budget[pipeline_id] -= max_update;
        }
    }
}

void OperatorComptroller::ComputeSatisfiableBudgets() {
    // Tests might have set their own max budgets.
    if (this->pipeline_remaining_budget.size() == 0) {
        // If the budgets for each pipeline have not been initialized yet
        // (common/default case), initialize it with the total memory available
        // (with some factor applied to the total size to allow for non-budgeted
        // memory usage.
        size_t total_mem = BODO_MEMORY_BUDGET_USAGE_FRACTION *
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
    this->pipeline_remaining_operators.resize(this->num_pipelines);

    for (const auto& req : this->requests_per_operator) {
        for (int64_t pipeline_id = req.min_pipeline_id;
             pipeline_id <= req.max_pipeline_id; pipeline_id++) {
            this->pipeline_remaining_operators[pipeline_id]++;
        }
    }

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

    while (true) {
        bool changed = false;
        std::vector<size_t> updates_per_operator(
            this->operator_allocated_budget.size());
        // Discover new updates but don't apply them so that all operators can
        // receive the largest possible slice of memory in this iteration. Since
        // we only hand out memory in chunks of size:
        //   (total available)/(num unsatisfied operators)
        // it's safe to do this.
        for (int64_t op_id = 0;
             op_id <
             static_cast<int64_t>(this->operator_allocated_budget.size());
             op_id++) {
            // In this loop, for each operator, attempt to increase the allotted
            // memory by pipeline_remaining_budget / num_unsatisfied_operators
            // across all pipelines (take the min amount of memory across all
            // pipelines).
            const auto& req = this->requests_per_operator[op_id];
            if (req.estimate == 0) {
                continue;
            }

            size_t available_allocation = std::numeric_limits<size_t>::max();
            for (int64_t pipeline_id = req.min_pipeline_id;
                 pipeline_id <= req.max_pipeline_id; pipeline_id++) {
                size_t remaining_budget =
                    this->pipeline_remaining_budget[pipeline_id];
                size_t remaining_operators =
                    this->pipeline_remaining_operators[pipeline_id];
                size_t requesting_allocation = std::min(
                    req.estimate, remaining_budget / remaining_operators);
                available_allocation =
                    std::min(requesting_allocation, available_allocation);
            }
            updates_per_operator[op_id] = available_allocation;
        }

        // Apply updates
        for (int64_t op_id = 0;
             op_id < static_cast<int64_t>(this->requests_per_operator.size());
             op_id++) {
            auto& req = this->requests_per_operator[op_id];
            auto available_allocation = updates_per_operator[op_id];
            if (available_allocation > 0) {
                // Update the memory allocated for this operator
                this->operator_allocated_budget[op_id] += available_allocation;
                // Update the memory remaining in the request
                req.estimate -= available_allocation;

                for (int64_t pipeline_id = req.min_pipeline_id;
                     pipeline_id <= req.max_pipeline_id; pipeline_id++) {
                    // Mark memory as used in all pipelines with this operator
                    this->pipeline_remaining_budget[pipeline_id] -=
                        available_allocation;
                    // If this request is completely satisfied, remove it from
                    // the remaining operators for this pipeline
                    if (req.estimate == 0) {
                        this->pipeline_remaining_operators[pipeline_id] -= 1;
                    }
                }
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
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

void delete_operator_comptroller() { OperatorComptroller::Default()->Reset(); }

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
    SetAttrStringFromVoidPtr(m, delete_operator_comptroller);
    return m;
}
