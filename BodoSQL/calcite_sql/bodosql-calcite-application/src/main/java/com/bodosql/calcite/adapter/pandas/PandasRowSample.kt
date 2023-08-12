package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.plan.RelOptRowSamplingParameters
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel

class PandasRowSample(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    val params: RelOptRowSamplingParameters,
) : SingleRel(cluster, traitSet, child), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        return super.explainTerms(pw)
            .item("mode", if (params.isBernoulli()) "bernoulli" else "system")
            .item("rows", params.getNumberOfRows())
            .item(
                "repeatableSeed",
                if (params.isRepeatable()) params.getRepeatableSeed() else "-",
            )
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasRowSample {
        return PandasRowSample(cluster, traitSet, sole(inputs), params)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, params: RelOptRowSamplingParameters): PandasRowSample {
            val mq = cluster.metadataQuery
            val batchingProperty = ExpectedBatchingProperty.alwaysSingleBatchProperty()
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION).replace(batchingProperty)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    mq.collations(input)
                }
            return PandasRowSample(cluster, traitSet, input, params)
        }
    }
}
