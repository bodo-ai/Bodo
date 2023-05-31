package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rex.RexNode

class PandasSort(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : Sort(cluster, traitSet, input, collation, offset, fetch), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(
        traitSet: RelTraitSet,
        newInput: RelNode,
        newCollation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?
    ): Sort {
        return PandasSort(cluster, traitSet, newInput, newCollation, offset, fetch)
    }

    override fun emit(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        inputs: () -> List<Dataframe>
    ): Dataframe {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(child: RelNode, collation: RelCollation, offset: RexNode?, fetch: RexNode?): PandasSort {
            val cluster = child.cluster
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replace(BatchingProperty.SINGLE_BATCH)
                .replace(collation)
            return PandasSort(cluster, traitSet, child, collation, offset, fetch);
        }
    }
}
