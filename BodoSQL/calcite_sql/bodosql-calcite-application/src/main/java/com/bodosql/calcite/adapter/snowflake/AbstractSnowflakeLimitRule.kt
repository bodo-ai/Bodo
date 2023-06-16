package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.logical.LogicalSort
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeLimitRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeLimitRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }

        val (sort, filter, scan) = extractNodes(call)
        val catalogTable = scan.catalogTable

        val newNode = scan.let { input ->
            filter?.let { f ->
                // TODO(jsternberg): There should be another planner rule that's already
                // constructed the SnowflakeFilter and this rule should not have to do that.
                // That would simplify this rule some amount, but we don't presently have
                // proper code generation for the snowflake filter and it may conflict
                // with pushdowns in the bodosql code so we're just using this as a placeholder.
                SnowflakeFilter.create(
                    f.cluster,
                    f.traitSet,
                    input,
                    f.condition,
                    catalogTable,
                )
            } ?: input
        }.let { input ->
            SnowflakeSort.create(
                sort.cluster,
                sort.traitSet,
                input,
                sort.collation,
                sort.offset,
                sort.fetch,
                catalogTable,
            )
        }
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Triple<Sort, Filter?, SnowflakeTableScan> {
        return if (call.rels.size == 3) {
            Triple(call.rel(0), call.rel(1), call.rel(2))
        } else {
            Triple(call.rel(0), null, call.rel(1))
        }
    }

    companion object {
        @JvmStatic
        fun isOnlyLimit(sort: LogicalSort): Boolean {
            // We pushdown sorts that only contain fetch and/or offset
            return (sort.offset != null || sort.fetch != null) && sort.getCollation().fieldCollations.isEmpty()
        }
    }

    interface Config : RelRule.Config
}
