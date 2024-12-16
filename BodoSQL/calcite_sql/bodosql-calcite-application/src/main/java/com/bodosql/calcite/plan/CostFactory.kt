package com.bodosql.calcite.plan

import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptCostFactory

class CostFactory : RelOptCostFactory {
    override fun makeCost(
        rowCount: Double,
        cpu: Double,
        io: Double,
    ): RelOptCost = Cost(rowCount, cpu, io, 0.0)

    fun makeCost(
        rowCount: Double,
        cpu: Double,
        io: Double,
        mem: Double,
    ): RelOptCost = Cost(rowCount, cpu, io, mem)

    override fun makeHugeCost(): RelOptCost = Cost(100.0, 100.0, 100.0, 100.0)

    override fun makeInfiniteCost(): RelOptCost = Cost.INFINITY

    override fun makeTinyCost(): RelOptCost = Cost(1.0, 1.0, 1.0, 1.0)

    override fun makeZeroCost(): RelOptCost = Cost(0.0, 0.0, 0.0, 0.0)
}
