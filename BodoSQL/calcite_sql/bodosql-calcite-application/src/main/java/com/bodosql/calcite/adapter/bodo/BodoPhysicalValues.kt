package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexLiteral

class BodoPhysicalValues(
    cluster: RelOptCluster,
    rowType: RelDataType,
    tuples: ImmutableList<ImmutableList<RexLiteral>>,
    traitSet: RelTraitSet,
) : Values(cluster, rowType, tuples, traitSet.replace(BodoPhysicalRel.CONVENTION)),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalValues = BodoPhysicalValues(cluster, getRowType(), tuples, traitSet.replace(BodoPhysicalRel.CONVENTION))

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
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
        ): BodoPhysicalValues = BodoPhysicalValues(cluster, rowType, tuples, traitSet)
    }
}
