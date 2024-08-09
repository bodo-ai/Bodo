package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.adapter.bodo.BodoCostEstimator.Companion.averageTypeValueSize
import com.bodosql.calcite.plan.Cost
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexCorrelVariable
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexFieldAccess
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLambda
import org.apache.calcite.rex.RexLambdaRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexPatternFieldRef
import org.apache.calcite.rex.RexRangeRef
import org.apache.calcite.rex.RexSubQuery
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.rex.RexVisitor
import org.apache.calcite.sql.SqlOperator

object RexCostEstimator : RexVisitor<Cost> {
    // Input refs have 0 cost to either use or materialize.
    override fun visitInputRef(inputRef: RexInputRef): Cost = Cost()

    override fun visitLocalRef(localRef: RexLocalRef): Cost = throw UnsupportedOperationException()

    override fun visitLambda(var1: RexLambda): Cost = throw UnsupportedOperationException()

    override fun visitLambdaRef(var1: RexLambdaRef): Cost = throw UnsupportedOperationException()

    override fun visitLiteral(literal: RexLiteral): Cost {
        // Literals are a bit strange. They can produce a memory cost,
        // but only if they're materialized and not just part of another
        // operation.
        //
        // We're not really going to consider that case though at this stage
        // so just assign is a zero cost since that's probably what
        // it will be anyway.
        return Cost()
    }

    override fun visitCall(call: RexCall): Cost {
        // TODO(jsternberg): More complete usage but just doing something
        // very basic for now.
        // Base cost for this operation.
        var cost =
            Cost(
                cpu = if (call is RexOver) overFuncMultiplier(call.op) else 1.0,
                mem = averageTypeValueSize(call.type),
            )
        // If there are operands, include them in the cost.
        if (call.operands.isNotEmpty()) {
            cost =
                cost.plus(
                    call.operands.asSequence()
                        .map { op -> op.accept(this) }
                        .reduce { l, r -> l.plus(r) as Cost },
                ) as Cost
        }
        return cost
    }

    /**
     * Multiplier for windowed aggregate functions compared to a normal
     * call. These are rough ideas of how efficient a specific windowed
     * aggregate is.
     */
    private fun overFuncMultiplier(op: SqlOperator): Double =
        // MIN_ROW_NUMBER_FILTER is a special optimization for
        // row_number() = 1 that works inside filter and is much
        // more efficient than the corresponding operator.
        //
        // In order to coerce the planner to use this optimization
        // route, we set a lower cost for it so the planner is
        // more likely to choose it over the alternatives.
        //
        // In the future, we might want to revisit how
        // this optimization is represented in the plan
        // such as considering a special relational node
        // for this rather than using Filter/RexOver.
        if (op.name == "MIN_ROW_NUMBER_FILTER") {
            1.0
        } else {
            100.0
        }

    override fun visitOver(over: RexOver): Cost {
        // Same as call but use a base cost of 10 instead of 1.
        // The cost difference is represented in visitCall.
        val callCost = visitCall(over)

        // Include the cost of computing the window attributes.
        val partitionCosts = visitList(over.window.partitionKeys)
        val orderCosts = visitList(over.window.orderKeys.map { it.left })

        // Combine these together.
        return sequenceOf(partitionCosts, orderCosts)
            .flatten()
            .fold(callCost) { a, b -> a.plus(b) as Cost }
    }

    override fun visitCorrelVariable(correlVariable: RexCorrelVariable): Cost =
        // We don't support correlation variables.
        throw UnsupportedOperationException()

    override fun visitDynamicParam(param: RexDynamicParam): Cost = Cost(mem = averageTypeValueSize(param.type))

    override fun visitRangeRef(rangeRef: RexRangeRef): Cost = throw UnsupportedOperationException()

    override fun visitFieldAccess(fieldAccess: RexFieldAccess): Cost {
        // This is used by either correlation variables or accessing struct
        // members. Either of these should have 0 cost.
        return Cost()
    }

    override fun visitSubQuery(subQuery: RexSubQuery): Cost =
        // Sub queries should have been removed by now.
        // We don't support them within any Bodo operation.
        throw UnsupportedOperationException()

    override fun visitTableInputRef(tableInputRef: RexTableInputRef): Cost = visitInputRef(tableInputRef)

    override fun visitPatternFieldRef(patternFieldRef: RexPatternFieldRef): Cost = visitInputRef(patternFieldRef)
}
