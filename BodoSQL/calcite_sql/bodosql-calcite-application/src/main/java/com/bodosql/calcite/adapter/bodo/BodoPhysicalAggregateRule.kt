package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Aggregate

class BodoPhysicalAggregateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalAggregate::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalAggregateRule",
                )
                .withRuleFactory { config -> BodoPhysicalAggregateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val agg = rel as Aggregate
        val traitSet = rel.cluster.traitSet().replace(BodoPhysicalRel.CONVENTION)
        return BodoPhysicalAggregate(
            rel.cluster,
            traitSet,
            convert(agg.input, traitSet),
            agg.groupSet,
            agg.groupSets,
            agg.aggCallList,
        )
    }
}
