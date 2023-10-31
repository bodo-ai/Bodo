#include "../libs/_bodo_common.h"
#include "../libs/_memory_budget.h"
#include "./test.hpp"

bodo::tests::suite memory_budget_tests([] {
    bodo::tests::before_each([] { setenv("BODO_USE_MEMORY_BUDGETS", "1", 1); });
    bodo::tests::after_each([] { unsetenv("BODO_USE_MEMORY_BUDGETS"); });

    bodo::tests::test("test_exception_if_increment_called_before_init", [] {
        OperatorComptroller comptroller;
        bodo::tests::check_exception([&] { comptroller.IncrementPipelineID(); },
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
    });

    bodo::tests::test("test_1_op_unsatisfiable_estimate", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 200);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
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
    });
    bodo::tests::test("test_1_ops_2_pipelines_satisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 0);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 0, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 100);
    });
    bodo::tests::test("test_1_ops_2_pipelines_unsatisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 50);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 1, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
    });

    bodo::tests::test("test_2_ops_2_pipelines_unsatisfiable", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        comptroller->SetPipelineMemoryBudget(0, 100);
        comptroller->SetPipelineMemoryBudget(1, 100);
        comptroller->RegisterOperator(0, OperatorType::UNKNOWN, 0, 1, 100);
        comptroller->RegisterOperator(1, OperatorType::UNKNOWN, 1, 1, 100);
        comptroller->ComputeSatisfiableBudgets();
        bodo::tests::check(comptroller->GetOperatorBudget(0) == 50);
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 50);
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
        });

    bodo::tests::test("test_join_groupby_write", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        for (int64_t p_id = 0; p_id < 5; p_id++) {
            comptroller->SetPipelineMemoryBudget(p_id, 2 * 1024 * 1024);
        }
        comptroller->RegisterOperator(0, OperatorType::JOIN, 2, 3, 3 * 1024);
        comptroller->RegisterOperator(1, OperatorType::JOIN, 0, 1, 5 * 1024);
        comptroller->RegisterOperator(7, OperatorType::GROUPBY, 1, 1, 1024);
        comptroller->RegisterOperator(11, OperatorType::GROUPBY, 3, 3,
                                      23 * 1024);
        comptroller->RegisterOperator(12, OperatorType::SNOWFLAKE_WRITE, 4, 4,
                                      400 * 1024);
        comptroller->ComputeSatisfiableBudgets();

        // Operator 0 shares its memory with Operator 11 (GROUPBY) in
        // pipeline 3. It will get 3/(23+3) of the total memory.
        bodo::tests::check(
            (comptroller->GetOperatorBudget(0) > 0.23 * 1024 * 1024) &&
            (comptroller->GetOperatorBudget(0) < 0.24 * 1024 * 1024));
        // Operator 11 will get the rest (23/(3+23))
        bodo::tests::check(
            (comptroller->GetOperatorBudget(11) > 1.76 * 1024 * 1024) &&
            (comptroller->GetOperatorBudget(11) < 1.77 * 1024 * 1024));
        // Operators 1 and 7 share their memory in pipeline 1, so the memory
        // will be split proportional to their estimates.
        bodo::tests::check(
            (comptroller->GetOperatorBudget(1) > 1.66 * 1024 * 1024) &&
            (comptroller->GetOperatorBudget(1) < 1.67 * 1024 * 1024));
        bodo::tests::check(
            (comptroller->GetOperatorBudget(7) > 0.33 * 1024 * 1024) &&
            (comptroller->GetOperatorBudget(7) < 0.34 * 1024 * 1024));
        // Operator 2 is a dummy operator that wasn't explicitly registered.
        // It's memory allocation should be 0.
        bodo::tests::check(comptroller->GetOperatorBudget(2) == 0);
        // Operator 12 (WRITE) doesn't share its memory with anything. But it's
        // an absolute estimate, so it should get what it asked for and no more.
        bodo::tests::check(comptroller->GetOperatorBudget(12) == 400 * 1024);
    });

    bodo::tests::test("test_abs_rel_same_pipeline_sufficient_abs", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        for (int64_t p_id = 0; p_id < 4; p_id++) {
            comptroller->SetPipelineMemoryBudget(p_id, 2 * 1024 * 1024);
        }
        comptroller->RegisterOperator(0, OperatorType::JOIN, 0, 3, 3 * 1024);
        comptroller->RegisterOperator(1, OperatorType::JOIN, 1, 3, 3 * 1024);
        comptroller->RegisterOperator(2, OperatorType::GROUPBY, 3, 3, 3 * 1024);
        // Absolute estimates need less than 2/5th
        comptroller->RegisterOperator(3, OperatorType::SNOWFLAKE_WRITE, 3, 3,
                                      100 * 1024);
        comptroller->RegisterOperator(4, OperatorType::UNKNOWN, 3, 3,
                                      100 * 1024);
        comptroller->ComputeSatisfiableBudgets();

        bodo::tests::check(comptroller->GetOperatorBudget(0) == 616 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 616 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(2) == 616 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(3) == 100 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(4) == 100 * 1024);
    });

    bodo::tests::test("test_abs_rel_same_pipeline_insufficient_abs", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        for (int64_t p_id = 0; p_id < 4; p_id++) {
            comptroller->SetPipelineMemoryBudget(p_id, 2 * 1024 * 1024);
        }
        comptroller->RegisterOperator(0, OperatorType::JOIN, 0, 3, 3 * 1024);
        comptroller->RegisterOperator(1, OperatorType::JOIN, 1, 3, 3 * 1024);
        comptroller->RegisterOperator(2, OperatorType::GROUPBY, 3, 3, 3 * 1024);
        // Absolute estimates need more than 2/5th
        comptroller->RegisterOperator(3, OperatorType::SNOWFLAKE_WRITE, 3, 3,
                                      500 * 1024);
        comptroller->RegisterOperator(4, OperatorType::UNKNOWN, 3, 3,
                                      500 * 1024);
        comptroller->ComputeSatisfiableBudgets();

        bodo::tests::check(comptroller->GetOperatorBudget(0) > 409.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(0) < 410.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(1) > 409.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(1) < 410.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(2) > 409.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(2) < 410.0 * 1024);
        // Abs ops don't get what they need, bound by 2/5th and other pipelines.
        bodo::tests::check(comptroller->GetOperatorBudget(3) > 409.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(3) < 410.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(4) > 409.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(4) < 410.0 * 1024);
    });

    bodo::tests::test("test_abs_rel_same_pipeline_insufficient_abs_2", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        for (int64_t p_id = 0; p_id < 4; p_id++) {
            comptroller->SetPipelineMemoryBudget(p_id, 2 * 1024 * 1024);
        }
        comptroller->RegisterOperator(0, OperatorType::JOIN, 0, 3, 3 * 1024);
        comptroller->RegisterOperator(1, OperatorType::JOIN, 1, 3, 3 * 1024);
        // Absolute estimates need more than 2/5th
        comptroller->RegisterOperator(2, OperatorType::SNOWFLAKE_WRITE, 3, 3,
                                      500 * 1024);
        comptroller->RegisterOperator(3, OperatorType::UNKNOWN, 3, 3,
                                      500 * 1024);
        comptroller->RegisterOperator(4, OperatorType::GROUPBY, 3, 3, 3 * 1024);
        // This will bound operators 0 and 1's available budget in pipeline 3
        comptroller->RegisterOperator(5, OperatorType::JOIN, 0, 1, 500 * 1024);

        comptroller->ComputeSatisfiableBudgets();

        bodo::tests::check(comptroller->GetOperatorBudget(0) > 12.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(0) < 13.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(1) > 12.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(1) < 13.0 * 1024);
        // Groupby gets larger than proportional share since the other rel
        // operators are bound by another pipeline.
        bodo::tests::check(comptroller->GetOperatorBudget(4) > 1023.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(4) < 1024.0 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(5) >
                           1.9 * 1024 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(5) <
                           2.0 * 1024 * 1024);
        // Abs ops get what they need, since join operators are bound by other
        // pipelines
        bodo::tests::check(comptroller->GetOperatorBudget(2) == 500 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(3) == 500 * 1024);
    });

    bodo::tests::test("test_join_only_used_in_2_pipelines", [] {
        auto comptroller = OperatorComptroller::Default();
        comptroller->Initialize();
        for (int64_t p_id = 0; p_id < 6; p_id++) {
            comptroller->SetPipelineMemoryBudget(p_id, 2 * 1024 * 1024);
        }
        comptroller->RegisterOperator(0, OperatorType::JOIN, 0, 5,
                                      1.5 * 1024 * 1024);
        comptroller->RegisterOperator(1, OperatorType::JOIN, 1, 2, 5 * 1024);
        comptroller->RegisterOperator(3, OperatorType::JOIN, 2, 3, 5 * 1024);
        comptroller->ComputeSatisfiableBudgets();

        // Operators 1 and 3 shouldn't be affected by op-0 since the pipelines
        // are separate. If we were to assume op-0 is "alive" in all 6 pipelines
        // (since min max is 0, 5), operators 1 and 3 would have a much lower
        // budget
        bodo::tests::check(comptroller->GetOperatorBudget(1) == 1024 * 1024);
        bodo::tests::check(comptroller->GetOperatorBudget(3) == 1024 * 1024);
        // Operator 0 should get the entire budget.
        bodo::tests::check(comptroller->GetOperatorBudget(0) ==
                           2 * 1024 * 1024);
    });
});
