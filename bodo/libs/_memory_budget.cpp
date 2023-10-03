// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_memory_budget.h"
#include "_bodo_common.h"

void OperatorComptroller::Initialize() { current_pipeline_id = 0; }

void OperatorComptroller::Reset() {
    current_pipeline_id = UNINITIALIZED_PIPELINE_ID;

    pipeline_remaining_budget.clear();
    pipeline_remaining_operators.clear();
    operator_allocated_budget.clear();

    requests_per_operator.clear();

    num_pipelines = 0;
    num_operators = 0;
}

void OperatorComptroller::SetMemoryBudget(int64_t pipeline_id, size_t budget) {
    if (static_cast<int64_t>(pipeline_remaining_budget.size() - 1) <
        pipeline_id) {
        pipeline_remaining_budget.resize(pipeline_id + 1);
    }
    pipeline_remaining_budget[pipeline_id] = budget;
}

void OperatorComptroller::RegisterOperator(int64_t operator_id,
                                           int64_t min_pipeline_id,
                                           int64_t max_pipeline_id,
                                           size_t estimate) {
    if (num_operators < (operator_id + 1)) {
        // We have to do this resize here, because we are currently writing to
        // the requests_per_operator map.
        operator_allocated_budget.resize(operator_id + 1);
        requests_per_operator.resize(operator_id + 1);
        num_operators = operator_id + 1;
    }
    requests_per_operator[operator_id] =
        OperatorRequest{min_pipeline_id, max_pipeline_id, estimate};
    this->num_pipelines = std::max(this->num_pipelines, max_pipeline_id + 1);
}

void OperatorComptroller::IncrementPipelineID() {
    if (current_pipeline_id == UNINITIALIZED_PIPELINE_ID) {
        throw std::runtime_error(
            "Initialize() was not called before IncrementPipelineID()");
    }

    current_pipeline_id++;
}

int64_t OperatorComptroller::GetOperatorBudget(int64_t operator_id) {
    if (operator_id < 0) {
        return -1;
    }
    return operator_allocated_budget[operator_id];
}

void OperatorComptroller::ReduceOperatorBudget(int64_t operator_id,
                                               size_t budget) {
    if (operator_allocated_budget[operator_id] <=
        static_cast<int64_t>(budget)) {
        throw std::runtime_error(
            "New budget is not strictly less than old budget");
    }
    auto old_budget = operator_allocated_budget[operator_id];
    auto delta = old_budget - budget;
    const auto& req = requests_per_operator[operator_id];

    // Reduce budget for this operator
    operator_allocated_budget[operator_id] = budget;
    // Add the freed budget back to pipeline_remaining_budget so that operators
    // that call IncreaseOperatorBudget can take advantage of it
    for (int64_t pipeline_id = req.min_pipeline_id;
         pipeline_id <= req.max_pipeline_id; pipeline_id++) {
        pipeline_remaining_budget[pipeline_id] += delta;
    }
}

void OperatorComptroller::IncreaseOperatorBudget(int64_t operator_id) {
    const auto& req = requests_per_operator[operator_id];
    size_t max_update = std::numeric_limits<size_t>::max();
    // Determine the largest amount of budget we can allocate for this operator
    for (int64_t pipeline_id = req.min_pipeline_id;
         pipeline_id <= req.max_pipeline_id; pipeline_id++) {
        max_update =
            std::min(max_update, pipeline_remaining_budget[pipeline_id]);
    }

    // If we are able to increase the budget, then do so.
    if (max_update) {
        operator_allocated_budget[operator_id] += max_update;
        for (int64_t pipeline_id = req.min_pipeline_id;
             pipeline_id <= req.max_pipeline_id; pipeline_id++) {
            pipeline_remaining_budget[pipeline_id] -= max_update;
        }
    }
}

void OperatorComptroller::ComputeSatisfiableBudgets() {
    // Tests might have set their own max budgets
    if (this->pipeline_remaining_budget.size() !=
        static_cast<size_t>(this->num_pipelines)) {
        size_t total_mem = bodo::BufferPool::Default()->get_memory_size_bytes();
        for (int64_t i = 0; i < this->num_pipelines; i++) {
            SetMemoryBudget(i, total_mem);
        }
    }
    pipeline_remaining_budget.resize(this->num_pipelines);
    pipeline_remaining_operators.resize(this->num_pipelines);

    for (const auto& req : requests_per_operator) {
        for (int64_t pipeline_id = req.min_pipeline_id;
             pipeline_id <= req.max_pipeline_id; pipeline_id++) {
            pipeline_remaining_operators[pipeline_id]++;
        }
    }

    char* use_mem_budget = std::getenv("BODO_USE_MEMORY_BUDGETS");
    if (!use_mem_budget || strcmp(use_mem_budget, "1") != 0) {
        // Until more work is done to refine the memory estimates, disable
        // memory budgeting by default so that performance doesn't degrade.
        for (int64_t op_id = 0;
             op_id < static_cast<int64_t>(operator_allocated_budget.size());
             op_id++) {
            operator_allocated_budget[op_id] = -1;
        }
        return;
    }

    while (true) {
        bool changed = false;
        std::vector<size_t> updates_per_operator(
            operator_allocated_budget.size());
        // Discover new updates but don't apply them so that all operators can
        // recieve the largest possible slice of memory in this iteration. Since
        // we only hand out memory in chunks of size:
        //   (total availible)/(num unsatisfied operators)
        // it's safe to do this.
        for (int64_t op_id = 0;
             op_id < static_cast<int64_t>(operator_allocated_budget.size());
             op_id++) {
            // In this loop, for each operator, attempt to increase the allotted
            // memory by pipeline_remaining_budget / num_unsatisfied_operators
            // across all pipelines (take the min amount of memory across all
            // pipelines).
            const auto& req = requests_per_operator[op_id];
            if (req.estimate == 0) {
                continue;
            }

            size_t available_allocation = std::numeric_limits<size_t>::max();
            for (int64_t pipeline_id = req.min_pipeline_id;
                 pipeline_id <= req.max_pipeline_id; pipeline_id++) {
                size_t remaining_budget =
                    pipeline_remaining_budget[pipeline_id];
                size_t remaining_operators =
                    pipeline_remaining_operators[pipeline_id];
                size_t requesting_allocation = std::min(
                    req.estimate, remaining_budget / remaining_operators);
                available_allocation =
                    std::min(requesting_allocation, available_allocation);
            }
            updates_per_operator[op_id] = available_allocation;
        }

        // Apply updates
        for (int64_t op_id = 0;
             op_id < static_cast<int64_t>(requests_per_operator.size());
             op_id++) {
            auto& req = requests_per_operator[op_id];
            auto available_allocation = updates_per_operator[op_id];
            if (available_allocation > 0) {
                // Update the memory allocated for this operator
                operator_allocated_budget[op_id] += available_allocation;
                // Update the memory remaining in the request
                req.estimate -= available_allocation;

                for (int64_t pipeline_id = req.min_pipeline_id;
                     pipeline_id <= req.max_pipeline_id; pipeline_id++) {
                    // Mark memory as used in all pipelines with this operator
                    pipeline_remaining_budget[pipeline_id] -=
                        available_allocation;
                    // If this request is completely satisfied, remove it from
                    // the remaining operators for this pipeline
                    if (req.estimate == 0) {
                        pipeline_remaining_operators[pipeline_id] -= 1;
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

void register_operator(int64_t operator_id, int64_t min_pipeline_id,
                       int64_t max_pipeline_id, int64_t estimate) {
    if (estimate < 0) {
        estimate = bodo::BufferPool::Default()->get_memory_size_bytes();
    }
    OperatorComptroller::Default()->RegisterOperator(
        operator_id, min_pipeline_id, max_pipeline_id,
        static_cast<size_t>(estimate));
}

void increment_pipeline_id() {
    OperatorComptroller::Default()->IncrementPipelineID();
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
    SetAttrStringFromVoidPtr(m, increment_pipeline_id);
    SetAttrStringFromVoidPtr(m, compute_satisfiable_budgets);
    SetAttrStringFromVoidPtr(m, delete_operator_comptroller);
    return m;
}
