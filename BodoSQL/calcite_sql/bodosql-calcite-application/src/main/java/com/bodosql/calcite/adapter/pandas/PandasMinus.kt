package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Minus

class PandasMinus(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    inputs: List<RelNode>,
    all: Boolean,
) : Minus(cluster, traitSet.replace(PandasRel.CONVENTION), inputs, all), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
        all: Boolean,
    ): PandasMinus {
        return PandasMinus(cluster, traitSet, inputs, all)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }
}
