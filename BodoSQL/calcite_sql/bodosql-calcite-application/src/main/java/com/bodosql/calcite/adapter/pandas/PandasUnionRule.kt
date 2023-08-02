package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Union

class PandasUnionRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                BodoLogicalUnion::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasUnionRule")
            .withRuleFactory { config -> PandasUnionRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val union = rel as Union
        val traitSet = rel.cluster.traitSet().replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)
        val inputs = union.inputs.map { input ->
            convert(input, input.traitSet.replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH))
        }
        return PandasUnion(rel.cluster, traitSet, inputs, union.all)
    }
}
