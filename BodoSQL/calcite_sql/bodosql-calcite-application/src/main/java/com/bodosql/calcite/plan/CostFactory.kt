package com.bodosql.calcite.plan

import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptCostFactory

class CostFactory : RelOptCostFactory {
    override fun makeCost(
        rowCount: Double,
        cpu: Double,
        io: Double,
    ): RelOptCost {
        return Cost(rowCount, cpu, io, 0.0)
    }

    fun makeCost(
        rowCount: Double,
        cpu: Double,
        io: Double,
        mem: Double,
    ): RelOptCost {
        return Cost(rowCount, cpu, io, mem)
    }

    override fun makeHugeCost(): RelOptCost {
        return Cost(100.0, 100.0, 100.0, 100.0)
    }

    override fun makeInfiniteCost(): RelOptCost {
        return Cost.INFINITY
    }

    override fun makeTinyCost(): RelOptCost {
        return Cost(1.0, 1.0, 1.0, 1.0)
    }

    override fun makeZeroCost(): RelOptCost {
        return Cost(0.0, 0.0, 0.0, 0.0)
    }
}
