package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Sample

class BodoPhysicalSampleRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    Sample::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalSampleRule",
                )
                .withRuleFactory { config -> BodoPhysicalSampleRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sample = rel as Sample
        return BodoPhysicalSample.create(
            rel.cluster,
            convert(
                sample.input,
                sample.input.traitSet
                    .replace(BodoPhysicalRel.CONVENTION),
            ),
            sample.getSamplingParameters(),
        )
    }
}
