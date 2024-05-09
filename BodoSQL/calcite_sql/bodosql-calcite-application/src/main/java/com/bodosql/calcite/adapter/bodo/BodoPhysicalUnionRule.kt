package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Union

class BodoPhysicalUnionRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalUnion::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalUnionRule",
                )
                .withRuleFactory { config -> BodoPhysicalUnionRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val union = rel as Union
        val convention = BodoPhysicalRel.CONVENTION
        val traitSet = rel.cluster.traitSet().replace(convention)
        val inputs =
            union.inputs.map { input ->
                convert(input, input.traitSet.replace(convention))
            }
        return BodoPhysicalUnion(rel.cluster, traitSet, inputs, union.all)
    }
}
