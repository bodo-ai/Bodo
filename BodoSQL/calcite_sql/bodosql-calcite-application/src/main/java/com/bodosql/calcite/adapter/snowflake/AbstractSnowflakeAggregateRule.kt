package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeFamily
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeAggregateRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeAggregateRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }

        val (aggregate, filter, scan) = extractNodes(call)
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
            SnowflakeAggregate.create(
                aggregate.cluster,
                aggregate.traitSet,
                input,
                aggregate.groupSet,
                aggregate.groupSets,
                aggregate.aggCallList.toList(),
                catalogTable,
            )
        }
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Triple<Aggregate, Filter?, SnowflakeTableScan> {
        return if (call.rels.size == 3) {
            Triple(call.rel(0), call.rel(1), call.rel(2))
        } else {
            Triple(call.rel(0), null, call.rel(1))
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
        fun isPushableAggregate(aggregate: LogicalAggregate): Boolean {
            // We only allow aggregates to be pushed if there's no grouping
            // and they are considered simple. I'm honestly not sure what the
            // other types of aggregates are. It's likely fine to allow more
            // than just group count 0, but I am just leaving it at no grouping
            // for now.
            return Aggregate.isSimple(aggregate)
                    && aggregate.aggCallList.all { agg ->
                SUPPORTED_AGGREGATES.contains(agg.aggregation.kind)
            }
        }

        private val SUPPORTED_CALLS = setOf(
            // Logical operators.
            SqlKind.AND,
            SqlKind.OR,
            SqlKind.NOT,
            // Comparison operators.
            SqlKind.EQUALS,
            SqlKind.NOT_EQUALS,
            SqlKind.NULL_EQUALS,
            SqlKind.LESS_THAN,
            SqlKind.LESS_THAN_OR_EQUAL,
            SqlKind.GREATER_THAN,
            SqlKind.GREATER_THAN_OR_EQUAL,
            // Logical identity operators.
            SqlKind.IS_FALSE,
            SqlKind.IS_NOT_FALSE,
            SqlKind.IS_TRUE,
            SqlKind.IS_NOT_TRUE,
            SqlKind.IS_NULL,
            SqlKind.IS_NOT_NULL,
            // Math operators.
            SqlKind.PLUS,
            SqlKind.MINUS,
            SqlKind.TIMES,
            SqlKind.DIVIDE,
            SqlKind.MINUS_PREFIX,
        )

        @JvmStatic
        fun isPushableFilter(filter: Filter): Boolean {
            // Not sure what things are ok to push, but we're going to be fairly conservative
            // and whitelist specific things rather than blacklist.
            return filter.condition.accept(object : RexVisitorImpl<Boolean?>(true) {
                override fun visitLiteral(literal: RexLiteral?): Boolean {
                    return when (literal?.typeName?.family) {
                        // We can't yet write the rex literals for intervals back to sql.
                        // See SnowflakeSqlDialect for the area where we need to implement this.
                        SqlTypeFamily.INTERVAL_DAY_TIME, SqlTypeFamily.INTERVAL_WEEK, SqlTypeFamily.INTERVAL_YEAR_MONTH -> false
                        else -> true
                    }
                }
                override fun visitInputRef(inputRef: RexInputRef?): Boolean = true

                override fun visitCall(call: RexCall?): Boolean {
                    // Allow select operators to be pushed down.
                    // This list is non-exhaustive, but are generally the functions we've seen
                    // and verified to work. Add more as appropriate.
                    return if (call != null && SUPPORTED_CALLS.contains(call.kind)) {
                        // Arguments also need to be pushable.
                        call.operands.all { op -> op.accept(this) ?: false }
                    } else false
                }
            }) ?: false
        }
    }

    interface Config : RelRule.Config
}
