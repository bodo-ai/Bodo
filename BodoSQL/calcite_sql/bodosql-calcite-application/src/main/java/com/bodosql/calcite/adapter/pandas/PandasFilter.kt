package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.ExprTypeVisitor.isScalar
import com.bodosql.calcite.ir.BodoSQLKernel
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver

class PandasFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode
) : Filter(cluster, traitSet, child, condition), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): PandasFilter {
        return PandasFilter(cluster, traitSet, input, condition)
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        val inputRows = mq.getRowCount(input)
        // Filter cost consists of two parts. The cost to compute the actual filter
        // and the cost to filter each column.
        val conditionCost = condition.accept(RexCostEstimator).multiplyBy(inputRows)
        // Compute the expected number of output rows
        val outputRows = mq.getRowCount(this)
        // We will assume the CPU cost to apply the boolean filter to each row is 0
        // and just consider the memory allocation of the new output.
        // val filterCost = Cost(mem = mq.getAverageRowSize(this) ?: 0.0).multiplyBy(outputRows)
        // TODO(jsternberg): Temporarily disable memory accounting costs for filters.
        // This sometimes causes some undesirable results with choosing not to use
        // a filter in a place where a filter might be useful. This is because we seem
        // to not have all of the correct planner rules included, specifically the
        // ProjectFilterTransposeRule. When we have transitioned to the VolcanoPlanner
        // as our primary development target and are making changes to that,
        // consider adding the filter's memory cost again.
        // If the memory cost is still unhelpful, just delete it entirely.
        val filterCost = planner.costFactory.makeZeroCost()
        val totalCost = (conditionCost + filterCost) as Cost
        return planner.makeCost(rows = outputRows, from = totalCost)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        return if (isStreaming()) {
            emitStreaming(implementor)
        } else {
            emitSingleBatch(implementor)
        }
    }

    fun emitSingleBatch(implementor: PandasRel.Implementor): Dataframe {
        val input = implementor.visitChild(input, 0)
        return implementor.build { ctx ->
            val translator = ctx.rexTranslator(input)
            val condition = this.condition.accept(translator).let { filter ->
                if (isScalarCondition()) {
                    // If the output of this filter is a scalar, we need to
                    // coerce it to an array value for the filter operation.
                    coerceScalar(input, filter)
                } else {
                    filter
                }
            }
            // Generate the filter df1[df2] operation and assign to the destination.
            ctx.returns(Expr.GetItem(input, condition))
        }
    }

    fun emitStreaming(implementor: PandasRel.Implementor): Dataframe {
        val input = implementor.visitChild(input, 0)
        // TODO: Move to a wrapper function to avoid the timerInfo calls.
        // This requires more information about the high level design of the streaming
        // operators since there are several parts (e.g. state, multiple loop sections, etc.)
        // At this time it seems like it would be too much work to have a clean interface.
        // There may be a need to pass in several lambdas, so other changes may be needed to avoid
        // constant rewriting.
        return implementor.buildStreaming { ctx ->
            val translator = ctx.rexTranslator(input)
            val condition = this.condition.accept(translator).let { filter ->
                if (isScalarCondition()) {
                    // If the output of this filter is a scalar, we need to
                    // coerce it to an array value for the filter operation.
                    coerceScalar(input, filter)
                } else {
                    filter
                }
            }
            // Generate the filter df1[df2] operation and assign to the destination.
            ctx.returns(Expr.GetItem(input, condition))
        }
    }

    /**
     * Returns true if the condition returns a scalar value.
     */
    private fun isScalarCondition(): Boolean = isScalar(this.condition, cluster.rexBuilder)

    /**
     * Coerces a scalar value to a boolean array.
     */
    private fun coerceScalar(input: Dataframe, filter: Expr): Expr =
        Expr.Call("bodo.utils.utils.full_type",
            Expr.Len(input),
            BodoSQLKernel("is_true", listOf(filter)),
            Expr.Attribute(
                Expr.Raw("bodo"),
                "boolean_array_type",
            )
        )

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, condition: RexNode): PandasFilter {
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replaceIf(BatchingPropertyTraitDef.INSTANCE) {
                    if (RexOver.containsOver(condition)) {
                        BatchingProperty.SINGLE_BATCH
                    } else {
                        BatchingProperty.STREAMING
                    }
                }
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return PandasFilter(cluster, traitSet, input, condition)
        }
    }
}
