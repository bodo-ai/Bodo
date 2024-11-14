package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.plan.RelOptRowSamplingParameters
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel

class BodoPhysicalRowSample(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    val params: RelOptRowSamplingParameters,
) : SingleRel(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), child),
    BodoPhysicalRel {
    override fun explainTerms(pw: RelWriter): RelWriter =
        super
            .explainTerms(pw)
            .item("mode", if (params.isBernoulli) "bernoulli" else "system")
            .item("rows", params.numberOfRows)
            .item(
                "repeatableSeed",
                if (params.isRepeatable) params.repeatableSeed else "-",
            )

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalRowSample = BodoPhysicalRowSample(cluster, traitSet, sole(inputs), params)

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
            input: RelNode,
            params: RelOptRowSamplingParameters,
        ): BodoPhysicalRowSample {
            val mq = cluster.metadataQuery
            val traitSet =
                cluster
                    .traitSet()
                    .replaceIfs(RelCollationTraitDef.INSTANCE) {
                        mq.collations(input)
                    }
            return BodoPhysicalRowSample(cluster, traitSet, input, params)
        }
    }
}
