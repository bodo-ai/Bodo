# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's memory budget python interface
"""

import bodo
import bodo.libs.table_builder


def test_memory_budget(memory_leak_check):
    @bodo.jit
    def test():
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(0, 0, 1, 1)
        bodo.libs.memory_budget.register_operator(0, 0, 1, 1)
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        bodo.libs.memory_budget.increment_pipeline_id()
        bodo.libs.memory_budget.delete_operator_comptroller()

    # This test doesn't assert anything, and just checks for compilation right
    # now.
    test()
