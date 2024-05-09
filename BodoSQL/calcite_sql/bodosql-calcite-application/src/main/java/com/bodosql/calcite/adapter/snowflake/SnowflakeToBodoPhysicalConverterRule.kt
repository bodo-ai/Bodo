package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class SnowflakeToBodoPhysicalConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RelNode::class.java,
                    SnowflakeRel.CONVENTION,
                    BodoPhysicalRel.CONVENTION,
                    "SnowflakeToBodoPhysicalConverterRule",
                )
                .withRuleFactory { config -> SnowflakeToBodoPhysicalConverterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outConvention)
        return SnowflakeToBodoPhysicalConverter(rel.cluster, newTraitSet, rel)
    }
}
