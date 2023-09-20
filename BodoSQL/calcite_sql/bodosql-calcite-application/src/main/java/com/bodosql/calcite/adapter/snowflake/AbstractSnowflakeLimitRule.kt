package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Sort
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeLimitRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeLimitRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }

        val (sort, rel) = extractNodes(call)
        val catalogTable = rel.getCatalogTable()

        val newNode = SnowflakeSort.create(
            sort.cluster,
            sort.traitSet,
            rel,
            sort.collation,
            sort.offset,
            sort.fetch,
            catalogTable,
        )
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Pair<Sort, SnowflakeRel> {
        // Inputs are:
        // Sort ->
        //   SnowflakeToPandasConverter ->
        //      SnowflakeRel
        return Pair(call.rel(0), call.rel(2))
    }

    companion object {
        @JvmStatic
        fun isOnlyLimit(sort: Sort): Boolean {
            // We pushdown sorts that only contain fetch and/or offset
            return (sort.offset != null || sort.fetch != null) && sort.getCollation().fieldCollations.isEmpty()
        }
    }

    interface Config : RelRule.Config
}
