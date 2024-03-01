package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.common.FilterUtils
import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import com.bodosql.calcite.application.utils.IsScalar
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
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

            // Calculate the subset of the conjunction that is pushable versus the
            // subset that is not.
            val (icebergConditions, pandasConditions) =
                FilterUtils.extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    ::isPushableCondition,
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
             *  - NULL_EQUALS
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
                    SqlKind.NULL_EQUALS,
                    SqlKind.LESS_THAN,
                    SqlKind.LESS_THAN_OR_EQUAL,
                    SqlKind.GREATER_THAN,
                    SqlKind.GREATER_THAN_OR_EQUAL,
                    SqlKind.SEARCH,
                    SqlKind.IS_DISTINCT_FROM,
                    SqlKind.IS_NOT_DISTINCT_FROM,
                    // Logical identity operators.
                    SqlKind.IS_FALSE,
                    SqlKind.IS_NOT_FALSE,
                    SqlKind.IS_TRUE,
                    SqlKind.IS_NOT_TRUE,
                    SqlKind.IS_NULL,
                    SqlKind.IS_NOT_NULL,
                    SqlKind.GREATEST,
                    // Other functions
                    SqlKind.IN,
                )

            /**
             * Supported function calls without a builtin kind for columns are based on:
             * https://iceberg.apache.org/javadoc/1.4.3/?org/apache/iceberg/expressions/Expressions.html.
             * Note some functions are not 1:1 but can be remapped in code generation. In particular:
             *  - EQUAL_NULL
             */
            private val SUPPORTED_GENERIC_CALL_NAME =
                setOf(
                    StringOperatorTable.STARTSWITH.name,
                    CondOperatorTable.EQUAL_NULL.name,
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
                        override fun visitLiteral(literal: RexLiteral): Boolean = true

                        override fun visitInputRef(inputRef: RexInputRef): Boolean = true

                        override fun visitCall(call: RexCall): Boolean {
                            return if (IsScalar.isScalar(call)) {
                                // All scalars can always be computed.
                                true
                            } else if (SUPPORTED_BUILTIN_CALLS.contains(call.kind) || isSupportedGenericCall(call)) {
                                // Arguments also need to be pushable.
                                call.operands.all { op -> op.accept(this) ?: false }
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
            fun isPartiallyPushableFilter(filter: Filter): Boolean {
                return FilterUtils.isPartiallyPushableFilter(filter, ::isPushableCondition)
            }

            /**
             * Determine if all of a filter is pushable.
             */
            @JvmStatic
            fun isFullyPushableFilter(filter: Filter): Boolean {
                return FilterUtils.isFullyPushableFilter(filter, ::isPushableCondition)
            }

            /**
             * Split a Filter into a pair of RexNodes that are pushed into Iceberg
             * and kept in Pandas.
             * @return The first value is the Iceberg section and the second value
             * is the section that cannot be pushed.
             */
            @JvmStatic
            fun splitFilterConditions(filter: Filter): Pair<RexNode?, RexNode?> {
                return FilterUtils.extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    ::isPushableCondition,
                )
            }
        }
    }
