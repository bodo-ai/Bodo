package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.logical.LogicalAggregate

class PandasAggregateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalAggregate::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasAggregateRule")
            .withRuleFactory { config -> PandasAggregateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val agg = rel as Aggregate
        val traitSet = rel.cluster.traitSet().replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)
        return PandasAggregate(rel.cluster, traitSet, convert(agg.input, traitSet.replace(BatchingProperty.SINGLE_BATCH)),
            agg.groupSet, agg.groupSets, agg.aggCallList)
    }
}
