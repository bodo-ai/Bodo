package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.SortBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rex.RexNode

class BodoPhysicalSort(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : SortBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), input, collation, offset, fetch), BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): BodoPhysicalSort {
        return BodoPhysicalSort(cluster, traitSet, input, collation, offset, fetch)
    }

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
            child: RelNode,
            collation: RelCollation,
            offset: RexNode?,
            fetch: RexNode?,
        ): BodoPhysicalSort {
            val cluster = child.cluster
            val traitSet = cluster.traitSet().replace(collation)
            return BodoPhysicalSort(cluster, traitSet, child, collation, offset, fetch)
        }
    }
}
