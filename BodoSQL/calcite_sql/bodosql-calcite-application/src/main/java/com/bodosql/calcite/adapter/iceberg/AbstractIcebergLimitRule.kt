package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.common.LimitUtils.Companion.extractSortNodes
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractIcebergLimitRule protected constructor(
    config: Config,
) : RelRule<AbstractIcebergLimitRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall) {
        val (sort, rel) = extractSortNodes<IcebergRel>(call)
        val catalogTable = rel.getCatalogTable()

        val newNode =
            IcebergSort.create(
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
