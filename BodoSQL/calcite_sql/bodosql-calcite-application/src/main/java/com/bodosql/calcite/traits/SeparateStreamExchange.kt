package com.bodosql.calcite.traits

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.SingleRel

class SeparateStreamExchange(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) : SingleRel(cluster,  traits,  input), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
        assert(traits.contains(BatchingProperty.STREAMING))
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): RelNode? {
        return SeparateStreamExchange(cluster, traitSet, inputs[0])
    }

    override fun isEnforcer(): Boolean {
        return true
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        // This should never be called, we need to handle SeparateStreamExchange
        // in the visitor itself due to needing to mess with the visitor state slightly
        TODO("Not yet implemented")
    }

    override fun splitCount(numRanks: Int): Int {
        return 1
    }
}
