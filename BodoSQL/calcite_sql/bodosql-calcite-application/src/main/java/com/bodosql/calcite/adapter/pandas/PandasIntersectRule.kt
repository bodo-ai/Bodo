package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Intersect
import org.apache.calcite.rel.logical.LogicalIntersect

class PandasIntersectRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalIntersect::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasIntersectRule",
                )
                .withRuleFactory { config -> PandasIntersectRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val intersect = rel as Intersect
        val convention = PandasRel.CONVENTION
        val traitSet = intersect.traitSet.replace(convention)
        val inputs =
            intersect.inputs.map { input ->
                convert(input, input.traitSet.replace(convention))
            }
        return PandasIntersect(rel.cluster, traitSet, inputs, intersect.all)
    }
}
