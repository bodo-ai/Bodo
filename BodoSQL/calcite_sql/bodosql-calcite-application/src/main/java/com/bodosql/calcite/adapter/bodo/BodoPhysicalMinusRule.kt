package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Minus
import org.apache.calcite.rel.logical.LogicalMinus

class BodoPhysicalMinusRule private constructor(
    config: Config,
) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalMinus::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalMinusRule",
                ).withRuleFactory { config -> BodoPhysicalMinusRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val minus = rel as Minus
        val traitSet = rel.traitSet.replace(BodoPhysicalRel.CONVENTION)
        val inputs =
            minus.inputs.map { input ->
                convert(input, input.traitSet.replace(BodoPhysicalRel.CONVENTION))
            }
        return BodoPhysicalMinus(rel.cluster, traitSet, inputs, minus.all)
    }
}
