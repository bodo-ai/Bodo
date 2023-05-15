package com.bodosql.calcite.traits

import com.bodosql.calcite.adapter.pandas.PandasRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class CombineStreamRule private constructor(config: Config) : ConverterRule(config) {

    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                PandasRel::class.java, BatchingProperty.STREAMING,
                BatchingProperty.SINGLE_BATCH, "CombineStreamRule")
            .withRuleFactory { config -> CombineStreamRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outTrait)
        return CombineStreamsExchange(rel.cluster, newTraitSet, rel)
    }
}
