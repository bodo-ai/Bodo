package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.MinRowNumberFilterBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelDistributionTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMdDistribution
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.ImmutableBitSet

class BodoLogicalMinRowNumberFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode,
    inputsToKeep: ImmutableBitSet,
) : MinRowNumberFilterBase(cluster, traitSet, child, condition, inputsToKeep) {

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): BodoLogicalMinRowNumberFilter {
        return BodoLogicalMinRowNumberFilter(cluster, traitSet, input, condition, inputsToKeep)
    }

    companion object {
        @JvmStatic
        fun create(input: RelNode, condition: RexNode): BodoLogicalMinRowNumberFilter {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(Convention.NONE)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
                .replaceIf(RelDistributionTraitDef.INSTANCE) {
                    RelMdDistribution.filter(mq, input)
                }
            val inputsToKeep = ImmutableBitSet.range(input.rowType.fieldCount)
            return BodoLogicalMinRowNumberFilter(cluster, traitSet, input, condition, inputsToKeep)
        }
    }
}
