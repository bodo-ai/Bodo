package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.sql.SqlKind
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
        return when (call.rels.size) {
            // Inputs are:
            // Aggregate ->
            //      CombineStreamsExchange ->
            //          SnowflakeToPandasConverter ->
            //             SnowflakeRel
            4 -> Pair(call.rel(0), call.rel(3))
            // Inputs are:
            // Aggregate ->
            //   SnowflakeToPandasConverter ->
            //      SnowflakeRel
            3 -> Pair(call.rel(0), call.rel(2))
            // Inputs are:
            // Aggregate ->
            //    SnowflakeRel
            else -> Pair(call.rel(0), call.rel(1))
        }
    }

    companion object {
        // Being conservative in the ones I choose here.
        private val SUPPORTED_AGGREGATES = setOf(
            SqlKind.COUNT,
            SqlKind.COUNTIF,
            SqlKind.SUM,
            SqlKind.SUM0,
            SqlKind.MIN,
            SqlKind.MAX,
            SqlKind.MEDIAN,
            SqlKind.AVG,
        )

        @JvmStatic
        fun isPushableAggregate(aggregate: Aggregate): Boolean {
            // We only allow aggregates to be pushed if there's no grouping
            // and they are considered simple. I'm honestly not sure what the
            // other types of aggregates are. It's likely fine to allow more
            // than just group count 0, but I am just leaving it at no grouping
            // for now.
            return Aggregate.isSimple(aggregate) &&
                aggregate.aggCallList.all { agg ->
                    SUPPORTED_AGGREGATES.contains(agg.aggregation.kind)
                }
        }
    }

    interface Config : RelRule.Config
}
