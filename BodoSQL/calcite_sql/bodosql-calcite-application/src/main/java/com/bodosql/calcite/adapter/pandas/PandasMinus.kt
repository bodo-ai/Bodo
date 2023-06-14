package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Minus

class PandasMinus(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : Minus(cluster, traitSet, inputs, all), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>, all: Boolean): PandasMinus {
        return PandasMinus(cluster, traitSet, inputs, all)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        TODO("Not yet implemented")
    }
}
