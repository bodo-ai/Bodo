package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.Util.isDistinct
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeAggregateRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeAggregateRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }
        val (aggregate, rel) = extractNodes(call)
        val catalogTable = rel.getCatalogTable()

        val newNode = SnowflakeAggregate.create(
            aggregate.cluster,
            aggregate.traitSet,
            rel,
            aggregate.groupSet,
            aggregate.groupSets,
            aggregate.aggCallList.toList(),
            catalogTable,
        )
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Pair<Aggregate, SnowflakeRel> {
        // Inputs are:
        // Aggregate ->
        //   SnowflakeToPandasConverter ->
        //      SnowflakeRel
        return Pair(call.rel(0), call.rel(2))
    }

    companion object {
        // Being conservative in the ones I choose here.
        private val SUPPORTED_AGGREGATES = setOf(
            SqlKind.COUNT,
            SqlKind.COUNTIF,
            SqlKind.MIN,
            SqlKind.MAX,
            // We want to add these function, but until we add proper
            // decimal support this will be problematic for large table as
            // we won't be able to properly sample/infer the TYPEOF for the
            // output.
            // SqlKind.SUM,
            // SqlKind.SUM0,
            // SqlKind.AVG,
        )

        /**
         * Determine if an Aggregate matches a distinct operation that we
         * want to push to Snowflake. Right now we just check that it matches
         * a distinct but in the future we may want to check metadata for if
         * a distinct is advisable.
         */
        @JvmStatic
        private fun isPushableDistinct(aggregate: Aggregate): Boolean {
            return aggregate.aggCallList.isEmpty() && Aggregate.isSimple(aggregate)
        }

        @JvmStatic
        fun isPushableAggregate(aggregate: Aggregate): Boolean {
            // We only allow aggregates to be pushed if there's no grouping,
            // and they are one of our supported functions.
            return (
                aggregate.groupSet.isEmpty &&
                    aggregate.aggCallList.isNotEmpty() &&
                    aggregate.aggCallList.all { agg ->
                        SUPPORTED_AGGREGATES.contains(agg.aggregation.kind) &&
                            !agg.hasFilter() &&
                            !agg.isDistinct
                    }
                ) || isPushableDistinct(aggregate)
        }
    }

    interface Config : RelRule.Config
}
