package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.common.LimitUtils.Companion.extractSortNodes
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeLimitRule protected constructor(
    config: Config,
) : RelRule<AbstractSnowflakeLimitRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall) {
        val (sort, rel) = extractSortNodes<SnowflakeRel>(call)
        val catalogTable = rel.getCatalogTable()

        val newNode =
            SnowflakeSort.create(
                sort.cluster,
                sort.traitSet,
                rel,
                sort.collation,
                sort.offset,
                sort.fetch,
                catalogTable,
            )
        call.transformTo(newNode)
        // New plan is absolutely better than old plan.
        call.planner.prune(sort)
    }

    interface Config : RelRule.Config
}
