package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.plan.RelOptRule

class SnowflakeRules private constructor() {
    companion object {
        private val SNOWFLAKE_AGGREGATE_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.DEFAULT_CONFIG.toRule()

        private val SNOWFLAKE_LIMIT_RULE: RelOptRule =
            SnowflakeLimitRule.Config.DEFAULT_CONFIG.toRule()

        private val SNOWFLAKE_FILTER_RULE: RelOptRule =
            SnowflakeFilterRule.Config.DEFAULT_CONFIG.toRule()

        private val SNOWFLAKE_PROJECT_RULE: RelOptRule = SnowflakeProjectRule.Config.DEFAULT_CONFIG.toRule()

        private val TO_PANDAS: RelOptRule = SnowflakeToPandasConverterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_RULES: List<RelOptRule> =
            listOf(
                SNOWFLAKE_AGGREGATE_RULE,
                SNOWFLAKE_LIMIT_RULE,
                SNOWFLAKE_FILTER_RULE,
                SNOWFLAKE_PROJECT_RULE,
                TO_PANDAS,
            )

        fun rules(): List<RelOptRule> = SNOWFLAKE_RULES
    }
}
