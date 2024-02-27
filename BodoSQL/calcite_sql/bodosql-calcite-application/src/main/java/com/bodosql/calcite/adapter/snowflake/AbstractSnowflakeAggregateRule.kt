package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import com.bodosql.calcite.application.utils.Utils.literalAggPrunedAggList
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rex.RexNode
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
            // Do we need a transformation with a projection for any literal values.
            val needsProjection = aggregate.aggCallList.any { x -> x.aggregation.kind == SqlKind.LITERAL_AGG }
            // Filter out any literals from the Snowflake Pushdown.
            val usedCallList = literalAggPrunedAggList(aggregate.aggCallList)
            val newAggregate =
                SnowflakeAggregate.create(
                    aggregate.cluster,
                    aggregate.traitSet,
                    rel,
                    aggregate.groupSet,
                    aggregate.groupSets,
                    usedCallList,
                    catalogTable,
                )
            val newNode =
                if (needsProjection) {
                    // Generate a wrapping projection if necessary.
                    val builder = call.builder()
                    builder.push(newAggregate)
                    // Track the offset for any functions that were pushed down.
                    var offset = aggregate.groupCount
                    val rexNodes: ArrayList<RexNode> = ArrayList()
                    for (i in 0 until aggregate.getRowType().fieldCount) {
                        if (i < aggregate.groupCount) {
                            rexNodes.add(builder.field(i))
                        } else {
                            val callListOffset = i - aggregate.groupCount
                            if (aggregate.aggCallList[callListOffset].aggregation.kind == SqlKind.LITERAL_AGG) {
                                // Safety check for unsupported patterns.
                                if (aggregate.aggCallList[callListOffset].rexList.size != 1) {
                                    return
                                }
                                rexNodes.add(aggregate.aggCallList[callListOffset].rexList[0])
                            } else {
                                // We have encountered a regular call. Extract it from the result.
                                rexNodes.add(builder.field(offset))
                                offset += 1
                            }
                        }
                    }
                    builder.project(rexNodes, aggregate.getRowType().fieldNames).build()
                } else {
                    newAggregate
                }
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
            private val SUPPORTED_AGGREGATES =
                setOf(
                    SqlKind.COUNT,
                    SqlKind.COUNTIF,
                    SqlKind.MIN,
                    SqlKind.MAX,
                    // Note: We support pushing aggregates that contain SqlKind.LITERAL_AGG,
                    // but since SqlKind.LITERAL_AGG is meant to be a literal/scalar that
                    // can be evaluated without any filter we will transform the aggregate
                    // into an Aggregate that can be pushed into Snowflake without the literal
                    // and a projection adding the literal afterwards.
                    SqlKind.LITERAL_AGG,
                    // We want to add these function, but until we add proper
                    // decimal support this will be problematic for large table as
                    // we won't be able to properly sample/infer the TYPEOF for the
                    // output.
                    // SqlKind.SUM,
                    // SqlKind.SUM0,
                    // SqlKind.AVG,
                )

            /**
             * Determine if the aggregate call list will be equivalent to only
             * keys (e.g. distinct) after transformations.
             */
            @JvmStatic
            private fun transformsToDistinct(aggCallList: List<AggregateCall>): Boolean {
                return aggCallList.isEmpty() || aggCallList.all { x -> x.aggregation.kind == SqlKind.LITERAL_AGG }
            }

            /**
             * Determine if an Aggregate matches a distinct operation that we
             * want to push to Snowflake. Right now we just check that it matches
             * a distinct but in the future we may want to check metadata for if
             * a distinct is advisable.
             */
            @JvmStatic
            private fun isPushableDistinct(aggregate: Aggregate): Boolean {
                return transformsToDistinct(aggregate.aggCallList) && Aggregate.isSimple(aggregate)
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
