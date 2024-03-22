package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.common.FilterUtils
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import com.bodosql.calcite.application.utils.IsScalar
import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.BodoRexSimplify
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexExecutor
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexSimplify
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.util.Util
import org.immutables.value.Value

/**
 * The framework for building an Iceberg conversion rule. At this time,
 * since we don't convert using either the HEP or Volcano planner
 * this just contains the static objects with the core conversion logic,
 * but it isn't a proper rule.
 */

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractIcebergFilterRule protected constructor(config: Config) :
    RelRule<AbstractIcebergFilterRule.Config>(config) {
        override fun onMatch(call: RelOptRuleCall) {
            val builder = call.builder()
            val (filter, rel) = FilterUtils.extractFilterNodes<IcebergRel>(call)
            val catalogTable = rel.getCatalogTable()

            // TODO: Fix the config to enable partial function generation.
            // Right now we include false to avoid duplicating the code, but we
            // can't take the if path in general Iceberg because it can lead to
            // an infinite loop.
            val partialFunction =
                if (false) {
                    // Calculate the subset of the conjunction that is pushable versus the
                    // subset that is not.
                    val rexBuilder: RexBuilder = builder.rexBuilder
                    val executor: RexExecutor = Util.first(call.planner.executor, RexUtil.EXECUTOR)
                    val simplify: RexSimplify = BodoRexSimplify(rexBuilder, RelOptPredicateList.EMPTY, executor)
                    generatePartialDerivationFunction(rexBuilder, simplify)
                } else {
                    { x -> x }
                }

            // We have to check directly in the filter if it is pushable, because
            // we need a simplifier and rexBuilder.
            if (!isPartiallyPushableFilter(filter, partialFunction)) {
                return
            }

            val (icebergConditions, pandasConditions) =
                FilterUtils.extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    ::isPushableCondition,
                    partialFunction,
                )
            assert(icebergConditions != null)

            if (pandasConditions == null) {
                // If none of the conditions cannot be pushed, then the entire filter can
                // become a IcebergFilter
                val newNode =
                    IcebergFilter.create(
                        filter.cluster,
                        filter.traitSet,
                        rel,
                        filter.condition,
                        catalogTable,
                    )
                call.transformTo(newNode)
                // New plan is absolutely better than old plan.
                call.planner.prune(filter)
            } else {
                // If at least 1 condition cannot be pushed, split the filter into
                // the component that can be pushed and the component that can not be.

                // Create the IcebergFilter from the subset that is pushable.
                val childFilter =
                    IcebergFilter.create(
                        filter.cluster,
                        filter.traitSet,
                        rel,
                        icebergConditions!!,
                        catalogTable,
                    )
                builder.push(childFilter)
                // Create the PandasFilter from the subset that is not pushable.
                builder.filter(pandasConditions)
                call.transformTo(builder.build())
                // New plan is absolutely better than old plan.
                call.planner.prune(filter)
            }
        }

        interface Config : RelRule.Config

        companion object {
            /**
             * Supported builtin calls for columns are based on:
             * https://iceberg.apache.org/javadoc/1.4.3/?org/apache/iceberg/expressions/Expressions.html.
             * Note some functions are not 1:1 but can be remapped in code generation. In particular:
             *  - SEARCH
             *  - IS_FALSE
             *  - IS_NOT_FALSE
             *  - IS_TRUE
             *  - IS_NOT_TRUE
             *  - IS_DISTINCT_FROM
             *  - IS_NOT_DISTINCT_FROM
             */
            private val SUPPORTED_BUILTIN_CALLS =
                setOf(
                    // Logical operators.
                    SqlKind.AND,
                    SqlKind.OR,
                    SqlKind.NOT,
                    // Comparison operators.
                    SqlKind.EQUALS,
                    SqlKind.NOT_EQUALS,
                    SqlKind.LESS_THAN,
                    SqlKind.LESS_THAN_OR_EQUAL,
                    SqlKind.GREATER_THAN,
                    SqlKind.GREATER_THAN_OR_EQUAL,
                    SqlKind.SEARCH,
                    // Equivalent to A != B AND (A IS NOT NULL OR B IS NOT NULL)
                    SqlKind.IS_DISTINCT_FROM,
                    // Equivalent to A == B OR (A IS NULL AND B IS NULL)
                    SqlKind.IS_NOT_DISTINCT_FROM,
                    // Logical identity operators.
                    SqlKind.IS_FALSE,
                    // Equivalent to A IS NULL OR A == TRUE.
                    // This is not the same as A != FALSE.
                    // SqlKind.IS_NOT_FALSE,
                    SqlKind.IS_TRUE,
                    // Equivalent to A IS NULL OR A == FALSE.
                    // This is not the same as A != TRUE.
                    // SqlKind.IS_NOT_TRUE,
                    SqlKind.IS_NULL,
                    SqlKind.IS_NOT_NULL,
                    // Other functions
                    SqlKind.IN,
                )

            /**
             * Supported function calls without a builtin kind for columns are based on:
             * https://iceberg.apache.org/javadoc/1.4.3/?org/apache/iceberg/expressions/Expressions.html.
             */
            private val SUPPORTED_GENERIC_CALL_NAME =
                setOf(
                    StringOperatorTable.STARTSWITH.name,
                )

            @JvmStatic
            private fun isSupportedGenericCall(call: RexCall): Boolean {
                return (call.kind == SqlKind.OTHER_FUNCTION || call.kind == SqlKind.OTHER) &&
                    SUPPORTED_GENERIC_CALL_NAME.contains(
                        call.operator.name,
                    )
            }

            @JvmStatic
            fun isPushableCondition(condition: RexNode): Boolean {
                // Not sure what things are ok to push, but we're going to be fairly conservative
                // and whitelist specific things rather than blacklist.
                return condition.accept(
                    object : RexVisitorImpl<Boolean?>(true) {
                        // We can't handle filters that are just scalars yet,
                        // so we track a state variable to see when we can push
                        // scalars.
                        var canPushScalar = false

                        override fun visitLiteral(literal: RexLiteral): Boolean = true

                        override fun visitInputRef(inputRef: RexInputRef): Boolean = true

                        // We only support Search Args with a column input, no complex expressions.
                        // All other Search Arg checks are enforced by the SearchArgExpandProgram.
                        fun visitSearch(call: RexCall): Boolean {
                            val op0 = call.operands[0]
                            val op1 = call.operands[1]
                            return op0 is RexInputRef && op1 is RexLiteral
                        }

                        override fun visitCall(call: RexCall): Boolean {
                            return if (canPushScalar && IsScalar.isScalar(call)) {
                                true
                            } else if (SUPPORTED_BUILTIN_CALLS.contains(call.kind) || isSupportedGenericCall(call)) {
                                // Search Calls in Some Formats Not Supported Yet
                                if (call.kind == SqlKind.SEARCH && !visitSearch(call)) {
                                    return false
                                }

                                val oldCanPushScalar = canPushScalar
                                if (call.kind != SqlKind.AND && call.kind != SqlKind.OR) {
                                    // Call Operands on Multiple Columns Not Supported Yet
                                    val numCols = call.operands.map { !IsScalar.isScalar(it) }.count { it }
                                    if (numCols != 1) {
                                        return false
                                    }
                                    if (call.kind != SqlKind.NOT) {
                                        // Update the state variable to allow pushing scalars when checking the operands.
                                        canPushScalar = true
                                    }
                                }
                                // Arguments also need to be pushable.
                                val operandResults = call.operands.all { op -> op.accept(this) ?: false }
                                // Restore the state
                                canPushScalar = oldCanPushScalar
                                operandResults
                            } else {
                                false
                            }
                        }
                    },
                ) ?: false
            }

            /**
             * Determine if any part of a filter is pushable.
             */
            @JvmStatic
            fun isPartiallyPushableFilter(
                filter: Filter,
                partialFilterDerivationFunction: (RexNode) -> RexNode,
            ): Boolean {
                return FilterUtils.isPartiallyPushableFilter(
                    filter,
                    ::isPushableCondition,
                    partialFilterDerivationFunction,
                )
            }

            /**
             * Determine if all of a filter is pushable.
             */
            @JvmStatic
            fun isFullyPushableFilter(
                filter: Filter,
                rexBuilder: RexBuilder,
                rexSimplify: RexSimplify,
            ): Boolean {
                return FilterUtils.isFullyPushableFilter(
                    filter,
                    ::isPushableCondition,
                    generatePartialDerivationFunction(rexBuilder, rexSimplify),
                )
            }

            /**
             * Split a Filter into a pair of RexNodes that are pushed into Iceberg
             * and kept in Pandas.
             * @return The first value is the Iceberg section and the second value
             * is the section that cannot be pushed.
             */
            @JvmStatic
            fun splitFilterConditions(
                filter: Filter,
                rexBuilder: RexBuilder,
                rexSimplify: RexSimplify,
            ): Pair<RexNode?, RexNode?> {
                return FilterUtils.extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    ::isPushableCondition,
                    generatePartialDerivationFunction(rexBuilder, rexSimplify),
                )
            }

            @JvmStatic
            fun generatePartialDerivationFunction(
                rexBuilder: RexBuilder,
                rexSimplify: RexSimplify,
            ): (RexNode) -> RexNode {
                return { condition: RexNode ->
                    if (condition.type.isNullable) {
                        val updatedCondition = rexBuilder.makeCall(SqlStdOperatorTable.IS_NOT_NULL, condition)
                        rexSimplify.simplifyUnknownAsFalse(updatedCondition)
                    } else {
                        condition
                    }
                }
            }
        }
    }
