package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.plan.Cost
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall

/**
 * Class that handles the cost estimation for Aggregate Functions.
 * This currently outputs all aggregates as having the same cost except
 * for if a filter is present.
 */
class AggCostEstimator {

    companion object {
        fun estimateInputCost(input: RelNode): Cost {
            // Give a fixed cost to drop duplicates.
            val cpu = 0.7
            // All the input has to be consumed by the Aggregate State
            // Note: We must call getRowType
            val rowType = input.getRowType()
            val mem = PandasCostEstimator.averageTypeValueSize(rowType)
            return Cost(cpu = cpu, mem = mem)
        }

        fun estimateFunctionCost(aggCall: AggregateCall): Cost {
            val cpu = if (aggCall.hasFilter()) 2.0 else 1.0
            val mem = PandasCostEstimator.averageTypeValueSize(aggCall.type)
            return Cost(cpu = cpu, mem = mem)
        }
    }
}
