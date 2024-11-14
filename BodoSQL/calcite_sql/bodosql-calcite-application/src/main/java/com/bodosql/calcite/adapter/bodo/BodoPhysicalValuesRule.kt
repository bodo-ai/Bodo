package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.logical.LogicalValues

class BodoPhysicalValuesRule private constructor(
    config: Config,
) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalValues::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalValuesRule",
                ).withRuleFactory { config -> BodoPhysicalValuesRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val values = rel as Values
        return BodoPhysicalValues.create(rel.cluster, rel.traitSet, values.rowType, values.tuples)
    }
}
