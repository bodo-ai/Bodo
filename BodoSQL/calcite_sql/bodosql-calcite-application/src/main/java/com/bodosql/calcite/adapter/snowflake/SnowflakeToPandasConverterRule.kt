package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.pandas.PandasProject
import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.SeparateStreamExchange
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rex.RexInputRef

class SnowflakeToPandasConverterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(RelNode::class.java, SnowflakeRel.CONVENTION,
                PandasRel.CONVENTION, "SnowflakeToPandasConverterRule")
            .withRuleFactory { config -> SnowflakeToPandasConverterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val newTraitSet = rel.traitSet.replace(outConvention).replace(BatchingProperty.SINGLE_BATCH)
        val converter = SnowflakeToPandasConverter(rel.cluster, newTraitSet, rel)
        // In addition to the converter, add a projection to return the type
        // to the original type of the input relation.
        val projects = rel.rowType.fieldList.mapIndexed { index, field ->
            RexInputRef(index, field.type)
        }
        var input: RelNode = converter
        // If the batchingProperty is not included in the required output traits (e.g. streaming is disabled),
        // then the `replace` above will ignore adding BatchingProperty.SINGLE_BATCH to the newTraitSet. As a result,
        // we can check for the presence of the batchingProperty to check if streaming is enabled.
        if (newTraitSet.contains(BatchingProperty.SINGLE_BATCH)) {
            // Check if we required batching. Replace will do nothing otherwise.
            input = SeparateStreamExchange(converter.cluster, newTraitSet.replace(BatchingProperty.STREAMING), converter)
        }
        return PandasProject.create(input, projects, rel.rowType)
    }
}
