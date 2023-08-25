// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include <memory>

/**
 * @brief Class that manages operator memory budget
 * This class will be availible as a singleton through the Default() method.
 */
class OperatorComptroller {
    /**
     * @brief Get the globally availible singleton instance
     */
    static std::shared_ptr<OperatorComptroller> Default() {
        static std::shared_ptr<OperatorComptroller> comptroller_(
            new OperatorComptroller());
        return comptroller_;
    }

    // TODO: To be called by init_operator_comptroller
    void Initialize();
    // TODO: To be called by delete_operator_comptroller
    void Reset();

    /**
     * @brief Register an operator and associate it with all pipelines where it
     * will be present with the corresponding estimate.
     */
    void RegisterOperator(int64_t operator_id, int64_t min_pipeline_id,
                          int64_t max_pipeline_id, size_t estimate);

    /**
     * @brief Increment the id of the current pipeline. This represents the
     * program beginning execution of the next pipeline.
     */
    void IncrementPipelineID();

    /**
     * @brief Get the budget for a given operator in the current pipeline
     * @param operator_id The operator owning the budget returned
     */
    int64_t GetOperatorBudget(int64_t operator_id);

    /**
     * @brief Reduce the budget for an operator, note that the new budget must
     * be <= the old budget
     * @param operator_id The operator to modify
     * @param budget The new budget
     */
    void ReduceOperatorBudget(int64_t operator_id, size_t budget);

    /**
     * @brief Increase the budget for a given operator to the maximum possible
     * @param operator_id The operator to modify
     */
    void IncreaseOperatorBudget(int64_t operator_id);

    /**
     * @brief Compute budgets for all known operators and pipelines. This must
     * be called only after all calls to RegisterOperator are completed.
     */
    void ComputeSatisfiableBudgets();
};
