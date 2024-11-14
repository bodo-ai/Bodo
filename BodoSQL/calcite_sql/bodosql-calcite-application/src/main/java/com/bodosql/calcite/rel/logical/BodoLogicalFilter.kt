package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.FilterBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelDistributionTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMdDistribution
import org.apache.calcite.rex.RexNode

class BodoLogicalFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode,
) : FilterBase(cluster, traitSet, child, condition) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): BodoLogicalFilter = BodoLogicalFilter(cluster, traitSet, input, condition)

    companion object {
        fun create(
            input: RelNode,
            condition: RexNode,
        ): BodoLogicalFilter {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet =
                cluster
                    .traitSetOf(Convention.NONE)
                    .replaceIfs(RelCollationTraitDef.INSTANCE) {
                        RelMdCollation.filter(mq, input)
                    }.replaceIf(RelDistributionTraitDef.INSTANCE) {
                        RelMdDistribution.filter(mq, input)
                    }
            return BodoLogicalFilter(cluster, traitSet, input, condition)
        }
    }
}
