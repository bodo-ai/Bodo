package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.pandas.PandasRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class IcebergToPandasConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    RelNode::class.java,
                    IcebergRel.CONVENTION,
                    PandasRel.CONVENTION,
                    "IcebergToPandasConverterRule",
                )
                .withRuleFactory { config -> IcebergToPandasConverterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outConvention)
        return IcebergToPandasConverter(rel.cluster, newTraitSet, rel)
    }
}
