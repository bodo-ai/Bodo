package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Sample

class PandasSampleRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                Sample::class.java,
                Convention.NONE,
                PandasRel.CONVENTION,
                "PandasSampleRule",
            )
            .withRuleFactory { config -> PandasSampleRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sample = rel as Sample
        val batchingProperty = ExpectedBatchingProperty.alwaysSingleBatchProperty()
        return PandasSample.create(
            rel.cluster,
            convert(
                sample.input,
                sample.input.traitSet
                    .replace(PandasRel.CONVENTION).replace(batchingProperty),
            ),
            sample.getSamplingParameters(),
        )
    }
}
