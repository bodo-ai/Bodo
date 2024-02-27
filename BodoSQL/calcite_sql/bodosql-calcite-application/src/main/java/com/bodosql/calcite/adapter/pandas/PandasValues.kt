package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexLiteral

class PandasValues(
    cluster: RelOptCluster,
    rowType: RelDataType,
    tuples: ImmutableList<ImmutableList<RexLiteral>>,
    traitSet: RelTraitSet,
) : Values(cluster, rowType, tuples, traitSet), PandasRel {
    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): PandasValues {
        return PandasValues(cluster, getRowType(), tuples, traitSet.replace(PandasRel.CONVENTION))
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

    companion object {
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            rowType: RelDataType,
            tuples: ImmutableList<ImmutableList<RexLiteral>>,
        ): PandasValues {
            val newTraitSet = traitSet.replace(PandasRel.CONVENTION)
            return PandasValues(cluster, rowType, tuples, newTraitSet)
        }
    }
}
