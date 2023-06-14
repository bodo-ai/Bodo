package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Intersect

class PandasIntersect(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : Intersect(cluster, traitSet, inputs, all), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>, all: Boolean): PandasIntersect {
        return PandasIntersect(cluster, traitSet, inputs, all)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        TODO("Not yet implemented")
    }
}
