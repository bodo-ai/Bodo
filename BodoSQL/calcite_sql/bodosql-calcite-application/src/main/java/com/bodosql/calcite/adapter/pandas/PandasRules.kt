package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.RelOptRule

class PandasRules private constructor() {
    companion object {
        private val TO_BODO_PHYSICAL: RelOptRule = PandasToBodoPhysicalConverterRule.DEFAULT_CONFIG.toRule()

        private val PANDAS_RULES: List<RelOptRule> = listOf(TO_BODO_PHYSICAL)

        fun rules(): List<RelOptRule> = PANDAS_RULES
    }
}
