// Copyright (C) 2023 Bodo Inc. All rights reserved.
// We define the python module in a separate file so that
// the BufferPool/OperatorBufferPool Cython extension
// doesn't need to include _bodo_common.h (and its dependencies).
#include "_bodo_common.h"
#include "_memory_budget.h"

void init_operator_comptroller() {
    OperatorComptroller::Default()->Initialize();
}

void init_operator_comptroller_with_budget(uint64_t budget) {
    OperatorComptroller::Default()->Initialize(budget);
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

void compute_satisfiable_budgets() {
    OperatorComptroller::Default()->ComputeSatisfiableBudgets();
}

// Only used for unit testing purposes
void reduce_operator_budget(int64_t operator_id, size_t new_estimate) {
    OperatorComptroller::Default()->ReduceOperatorBudget(operator_id,
                                                         new_estimate);
}

// Only used for unit testing purposes
void increase_operator_budget(int64_t operator_id) {
    OperatorComptroller::Default()->RequestAdditionalBudget(operator_id);
}

PyMODINIT_FUNC PyInit_memory_budget_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "memory_budget", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_operator_comptroller);
    SetAttrStringFromVoidPtr(m, init_operator_comptroller_with_budget);
    SetAttrStringFromVoidPtr(m, register_operator);
    SetAttrStringFromVoidPtr(m, compute_satisfiable_budgets);
    SetAttrStringFromVoidPtr(m, reduce_operator_budget);
    SetAttrStringFromVoidPtr(m, increase_operator_budget);
    return m;
}
