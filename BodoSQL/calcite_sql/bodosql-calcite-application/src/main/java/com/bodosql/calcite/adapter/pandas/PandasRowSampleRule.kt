package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import com.bodosql.calcite.rel.core.RowSample

class PandasRowSampleRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                RowSample::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasRowSampleRule")
            .withRuleFactory { config -> PandasRowSampleRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sample = rel as RowSample
        return PandasRowSample.create(rel.cluster,
            convert(sample.input,
                sample.input.traitSet
                    .replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)),
            sample.getRowSamplingParameters())
    }
}
