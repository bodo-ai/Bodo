package com.bodosql.calcite.adapter.iceberg

import org.apache.calcite.plan.RelOptRule

class IcebergRules private constructor() {
    companion object {
        private val TO_PANDAS: RelOptRule = IcebergToPandasConverterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val ICEBERG_RULES: List<RelOptRule> = listOf(TO_PANDAS)

        fun rules(): List<RelOptRule> = ICEBERG_RULES
    }
}
