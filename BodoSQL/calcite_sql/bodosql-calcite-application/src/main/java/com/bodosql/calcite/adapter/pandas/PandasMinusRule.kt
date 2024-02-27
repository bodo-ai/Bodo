package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Minus
import org.apache.calcite.rel.logical.LogicalMinus

class PandasMinusRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalMinus::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasMinusRule",
                )
                .withRuleFactory { config -> PandasMinusRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val minus = rel as Minus
        val traitSet = rel.traitSet.replace(PandasRel.CONVENTION)
        val inputs =
            minus.inputs.map { input ->
                convert(input, input.traitSet.replace(PandasRel.CONVENTION))
            }
        return PandasMinus(rel.cluster, traitSet, inputs, minus.all)
    }
}
