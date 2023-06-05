package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.traits.BatchingProperty
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

    override fun emit(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        inputs: () -> List<Dataframe>
    ): Dataframe {
        TODO("Not yet implemented")
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

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, condition: RexNode): PandasFilter {
            val mq = cluster.metadataQuery
            val containsOver = RexOver.containsOver(condition)
            val batchProperty = if (containsOver) BatchingProperty.SINGLE_BATCH else BatchingProperty.STREAMING
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION).replace(batchProperty)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return PandasFilter(cluster, traitSet, input, condition)
        }
    }
}
