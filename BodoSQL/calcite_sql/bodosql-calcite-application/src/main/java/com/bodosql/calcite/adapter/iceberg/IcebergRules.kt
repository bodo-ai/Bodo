package com.bodosql.calcite.adapter.iceberg

import org.apache.calcite.plan.RelOptRule

class IcebergRules private constructor() {
    companion object {
        private val ICEBERG_LIMIT_RULE: RelOptRule =
            IcebergLimitRule.Config.DEFAULT_CONFIG.toRule()

        private val TO_BODO_PHYSICAL: RelOptRule = IcebergToBodoPhysicalConverterRule.DEFAULT_CONFIG.toRule()

        private val ICEBERG_FILTER_RULE: RelOptRule =
            IcebergFilterRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val ICEBERG_RULES: List<RelOptRule> =
            listOf(
                ICEBERG_LIMIT_RULE,
                ICEBERG_FILTER_RULE,
                TO_BODO_PHYSICAL,
            )

        fun rules(): List<RelOptRule> = ICEBERG_RULES
    }
}
