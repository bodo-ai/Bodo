package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.pandas.PandasRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class SnowflakeToPandasConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RelNode::class.java,
                    SnowflakeRel.CONVENTION,
                    PandasRel.CONVENTION,
                    "SnowflakeToPandasConverterRule",
                )
                .withRuleFactory { config -> SnowflakeToPandasConverterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outConvention)
        return SnowflakeToPandasConverter(rel.cluster, newTraitSet, rel)
    }
}
