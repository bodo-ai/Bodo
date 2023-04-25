package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptRowSamplingParameters
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelInput
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.core.RowSample
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rex.RexNode

class PandasRowSample(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    val params: RelOptRowSamplingParameters,
) : SingleRel(cluster, traitSet, child), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun explainTerms(pw : RelWriter) : RelWriter {
        return super.explainTerms(pw)
            .item("mode", if (params.isBernoulli()) "bernoulli" else "system")
            .item("rows", params.getNumberOfRows())
            .item("repeatableSeed",
                if (params.isRepeatable()) params.getRepeatableSeed() else "-");
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasRowSample {
        return PandasRowSample(cluster, traitSet, sole(inputs), params)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, params: RelOptRowSamplingParameters): PandasRowSample {
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    mq.collations(input)
                }
            return PandasRowSample(cluster, traitSet, input, params)
        }
    }
}
