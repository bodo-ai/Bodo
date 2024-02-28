package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.SortBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rex.RexNode

class BodoLogicalSort(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : SortBase(cluster, traitSet, input, collation, offset, fetch) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): BodoLogicalSort {
        return BodoLogicalSort(cluster, traitSet, input, collation, offset, fetch)
    }

    companion object {
        @JvmStatic
        fun create(
            input: RelNode,
            collation: RelCollation,
            offset: RexNode?,
            fetch: RexNode?,
        ): BodoLogicalSort {
            val cluster = input.cluster
            val traitSet = cluster.traitSet().replace(Convention.NONE)
            return BodoLogicalSort(cluster, traitSet, input, collation, offset, fetch)
        }
    }
}
