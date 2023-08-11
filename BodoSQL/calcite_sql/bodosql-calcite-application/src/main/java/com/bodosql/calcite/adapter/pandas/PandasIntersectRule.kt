package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Intersect
import org.apache.calcite.rel.logical.LogicalIntersect

class PandasIntersectRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalIntersect::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasIntersectRule")
            .withRuleFactory { config -> PandasIntersectRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val intersect = rel as Intersect
        val convention = PandasRel.CONVENTION
        val batchingProperty = ExpectedBatchingProperty.alwaysSingleBatchProperty()
        val traitSet = intersect.traitSet.replace(convention).replace(batchingProperty)
        val inputs = intersect.inputs.map { input ->
            convert(input, input.traitSet.replace(convention).replace(batchingProperty))
        }
        return PandasIntersect(rel.cluster, traitSet, inputs, intersect.all)
    }
}
