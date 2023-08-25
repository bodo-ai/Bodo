// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_memory_budget.h"
#include "_bodo_common.h"

void OperatorComptroller::Initialize() {}

void OperatorComptroller::Reset() {}

void OperatorComptroller::RegisterOperator(int64_t operator_id,
                                           int64_t min_pipeline_id,
                                           int64_t max_pipeline_id,
                                           size_t estimate) {}

void OperatorComptroller::IncrementPipelineID() {}

int64_t OperatorComptroller::GetOperatorBudget(int64_t operator_id) {
    return 0;
}

void OperatorComptroller::ReduceOperatorBudget(int64_t operator_id,
                                               size_t budget) {}

void OperatorComptroller::IncreaseOperatorBudget(int64_t operator_id) {}

void OperatorComptroller::ComputeSatisfiableBudgets() {}

// The print outs here are temporary and will go away after real implementations
// are added.
void init_operator_comptroller() {
    std::cout << "init OperatorComptroller" << std::endl;
}
void register_operator(int64_t operator_id, int64_t min_pipeline_id,
                       int64_t max_pipeline_id, int64_t estimate) {
    std::cout << "register_operator(" << operator_id << "," << min_pipeline_id
              << "," << max_pipeline_id << "," << estimate << ")" << std::endl;
}
void increment_pipeline_id() {
    std::cout << "increment_pipeline_id" << std::endl;
}
void delete_operator_comptroller() {
    std::cout << "delete OperatorComptroller" << std::endl;
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
    SetAttrStringFromVoidPtr(m, increment_pipeline_id);
    SetAttrStringFromVoidPtr(m, delete_operator_comptroller);
    return m;
}
