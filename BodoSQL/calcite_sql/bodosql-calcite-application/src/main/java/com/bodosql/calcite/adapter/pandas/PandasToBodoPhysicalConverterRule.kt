package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class PandasToBodoPhysicalConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RelNode::class.java,
                    PandasRel.CONVENTION,
                    BodoPhysicalRel.CONVENTION,
                    "PandasToBodoPhysicalConverterRule",
                )
                .withRuleFactory { config -> PandasToBodoPhysicalConverterRule(config) }
    }

    override fun convert(rel: RelNode): PandasToBodoPhysicalConverter {
        val newTraitSet = rel.traitSet.replace(outConvention)
        return PandasToBodoPhysicalConverter(rel.cluster, newTraitSet, rel)
    }
}
