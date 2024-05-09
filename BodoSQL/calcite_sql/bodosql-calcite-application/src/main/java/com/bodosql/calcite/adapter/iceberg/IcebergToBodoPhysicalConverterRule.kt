package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class IcebergToBodoPhysicalConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RelNode::class.java,
                    IcebergRel.CONVENTION,
                    BodoPhysicalRel.CONVENTION,
                    "IcebergToBodoPhysicalConverterRule",
                )
                .withRuleFactory { config -> IcebergToBodoPhysicalConverterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outConvention)
        return IcebergToBodoPhysicalConverter(rel.cluster, newTraitSet, rel)
    }
}
