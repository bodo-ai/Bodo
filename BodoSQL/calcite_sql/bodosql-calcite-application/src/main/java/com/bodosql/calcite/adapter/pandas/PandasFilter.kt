package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rex.RexNode

class PandasFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode
) : Filter(cluster, traitSet, child, condition), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): PandasFilter {
        return PandasFilter(cluster, traitSet, input, condition)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, condition: RexNode): PandasFilter {
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return PandasFilter(cluster, traitSet, input, condition)
        }
    }
}
