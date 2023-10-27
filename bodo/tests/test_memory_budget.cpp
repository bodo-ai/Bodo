#include "../libs/_bodo_common.h"
#include "../libs/_memory_budget.h"
#include "./test.hpp"

bodo::tests::suite memory_budget_tests([] {
    bodo::tests::before_each([] { setenv("BODO_USE_MEMORY_BUDGETS", "1", 1); });
    bodo::tests::after_each([] { unsetenv("BODO_USE_MEMORY_BUDGETS"); });

    bodo::tests::test("test_exception_if_increment_called_before_init", [] {
        auto comptroller = OperatorComptroller::Default();
        bodo::tests::check_exception(
            [&] { comptroller->IncrementPipelineID(); },
            "Initialize() was not called");
    });
    bodo::tests::test("test_exception_if_increment_called_after_reset", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->IncrementPipelineID();
        comptroller->Reset();
        bodo::tests::check_exception(
            [&] { comptroller->IncrementPipelineID(); },
            "Initialize() was not called");
    });
    bodo::tests::test("test_exception_if_invalid_start_end_pipeline_range", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        bodo::tests::check_exception(
            [&]() {
                comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 1, 0,
                                              100);
            },
            "OperatorComptroller::RegisterOperator: max_pipeline_id cannot be "
            "less than min_pipeline_id");
    });

    bodo::tests::test("test_1_op_satisfiable_estimate", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
        comptroller->Reset();
    });

    bodo::tests::test("test_1_op_unsatisfiable_estimate", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 200);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
        comptroller->Reset();
    });

    bodo::tests::test("test_2_ops_equal_unsatisfiable_estimates", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->RegisterOperator(1, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 50);
        comptroller->Reset();
    });
    bodo::tests::test("test_2_ops_unequal_satisfiable_estimates", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 75);
        comptroller->RegisterOperator(1, OperatorType::UNKNOWN, 0, 0, 25);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 75);
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 25);
        comptroller->Reset();
    });
    bodo::tests::test("test_1_ops_2_pipelines_satisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 0);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
        comptroller->Reset();
    });
    bodo::tests::test("test_1_ops_2_pipelines_unsatisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 50);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 1, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
        comptroller->Reset();
    });

    bodo::tests::test("test_1_ops_2_pipelines_unsatisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 1, 100);
        comptroller->RegisterOperator(1, OperatorType::UNKNOWN, 1, 1, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 50);
        comptroller->Reset();
    });

    bodo::tests::test("test_reduce_budget", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
        comptroller->ReduceOperatorBudget(0, 50);
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);

        // Check that increasing the budget via ReduceOperatorBudget is illegal
        comptroller->ReduceOperatorBudget(0, 75);
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);

        comptroller->Reset();
    });

    bodo::tests::test("test_increase_budget", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 50);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
        comptroller->IncreaseOperatorBudget(0);
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
        comptroller->Reset();
    });

    bodo::tests::test(
        "test_increase_budget_unsatisfiable_becomes_satisfiable", [] {
            auto comptroller = OperatorComptroller::Default();
            comptroller->Initialize();
            comptroller->SetPipelineMemoryBudget(0, 100);
            comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 75);
            comptroller->RegisterOperator(1, OperatorType::UNKNOWN, 0, 0, 75);
            comptroller->ComputeSatisfiableBudgets();
            bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
            bodo::tests::check(comptroller->GetOperatorBudget(1) == 50);
            comptroller->ReduceOperatorBudget(0, 25);
            comptroller->IncreaseOperatorBudget(1);
            bodo::tests::check(comptroller->GetOperatorBudget(0) == 25);
            bodo::tests::check(comptroller->GetOperatorBudget(1) == 75);
            comptroller->Reset();
        });
});
