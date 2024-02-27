package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.UnionBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode

class BodoLogicalUnion(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : UnionBase(cluster, traitSet, inputs, all) {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
        all: Boolean,
    ): BodoLogicalUnion {
        return BodoLogicalUnion(cluster, traitSet, inputs, all)
    }

    companion object {
        @JvmStatic
        fun create(
            inputs: List<RelNode>,
            all: Boolean,
        ): BodoLogicalUnion {
            val cluster = inputs[0].cluster
            val traitSet = cluster.traitSet().replace(Convention.NONE)
            return BodoLogicalUnion(cluster, traitSet, inputs, all)
        }
    }
}
