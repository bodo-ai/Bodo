# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's memory budget python interface
"""

import bodo
import bodo.libs.table_builder


def test_memory_budget(memory_leak_check):
    @bodo.jit
    def test():
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            0, bodo.libs.memory_budget.OperatorType.JOIN, 0, 1, -1
        )
        bodo.libs.memory_budget.register_operator(
            1, bodo.libs.memory_budget.OperatorType.JOIN, 0, 1, -1
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        bodo.libs.memory_budget.reduce_operator_budget(0, 0)
        bodo.libs.memory_budget.increase_operator_budget(1)

    # This test doesn't assert anything, and just checks for compilation right
    # now.
    test()
