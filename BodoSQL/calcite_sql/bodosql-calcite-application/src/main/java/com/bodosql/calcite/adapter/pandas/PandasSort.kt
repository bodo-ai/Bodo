package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.SortBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rex.RexNode

class PandasSort(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : SortBase(cluster, traitSet.replace(PandasRel.CONVENTION), input, collation, offset, fetch), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): PandasSort {
        return PandasSort(cluster, traitSet, input, collation, offset, fetch)
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
            child: RelNode,
            collation: RelCollation,
            offset: RexNode?,
            fetch: RexNode?,
        ): PandasSort {
            val cluster = child.cluster
            val traitSet = cluster.traitSet().replace(collation)
            return PandasSort(cluster, traitSet, child, collation, offset, fetch)
        }
    }
}
