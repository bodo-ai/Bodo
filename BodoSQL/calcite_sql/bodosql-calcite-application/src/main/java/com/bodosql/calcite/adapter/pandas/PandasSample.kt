package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptSamplingParameters
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel

class PandasSample(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    val params: RelOptSamplingParameters,
) : SingleRel(cluster, traitSet, child), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        return super.explainTerms(pw)
            .item("mode", if (params.isBernoulli()) "bernoulli" else "system")
            .item("rate", params.getSamplingPercentage())
            .item(
                "repeatableSeed",
                if (params.isRepeatable()) params.getRepeatableSeed() else "-",
            )
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasSample {
        return PandasSample(cluster, traitSet, sole(inputs), params)
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
        fun create(cluster: RelOptCluster, input: RelNode, params: RelOptSamplingParameters): PandasSample {
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    mq.collations(input)
                }
            return PandasSample(cluster, traitSet, input, params)
        }
    }
}
