package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.logical.LogicalValues

class PandasValuesRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalValues::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasValuesRule",
                )
                .withRuleFactory { config -> PandasValuesRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val values = rel as Values
        return PandasValues.create(rel.cluster, rel.traitSet, values.getRowType(), values.tuples)
    }
}
