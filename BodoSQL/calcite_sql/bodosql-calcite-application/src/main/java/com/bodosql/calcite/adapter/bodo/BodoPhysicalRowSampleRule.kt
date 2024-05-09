package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.core.RowSample
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class BodoPhysicalRowSampleRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RowSample::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalRowSampleRule",
                )
                .withRuleFactory { config -> BodoPhysicalRowSampleRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sample = rel as RowSample
        return BodoPhysicalRowSample.create(
            rel.cluster,
            convert(
                sample.input,
                sample.input.traitSet
                    .replace(BodoPhysicalRel.CONVENTION),
            ),
            sample.getRowSamplingParameters(),
        )
    }
}
