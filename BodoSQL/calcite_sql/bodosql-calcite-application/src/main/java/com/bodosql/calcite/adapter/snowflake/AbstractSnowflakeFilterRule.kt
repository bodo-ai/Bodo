package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.common.FilterUtils
import com.bodosql.calcite.adapter.common.FilterUtils.Companion.extractFilterNodes
import com.bodosql.calcite.adapter.common.FilterUtils.Companion.extractPushableConditions
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter.Companion.create
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.ObjectOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlLibraryOperators
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.VariantSqlType
import org.immutables.value.Value

@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeFilterRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeFilterRule.Config>(config) {
        override fun onMatch(call: RelOptRuleCall) {
            val builder = call.builder()
            val (filter, rel) = extractFilterNodes<SnowflakeRel>(call)
            val catalogTable = rel.getCatalogTable()

            // Calculate the subset of the conjunction that is pushable versus the
            // subset that is not.
            val (snowflakeConditions, pandasConditions) =
                extractPushableConditions(
                    filter.condition,
                    filter.cluster.rexBuilder,
                    ::isPushableCondition,
                )
            assert(snowflakeConditions != null)

            if (pandasConditions == null) {
                // If none of the conditions cannot be pushed, then the entire filter can
                // become a SnowflakeFilter
                val newNode =
                    create(
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

                // Create the SnowflakeFilter from the subset that is pushable.
                val childFilter =
                    create(
                        filter.cluster,
                        filter.traitSet,
                        rel,
                        snowflakeConditions!!,
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
            private val SUPPORTED_CALLS =
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
                    // String functions,
                    SqlKind.REVERSE,
                    SqlKind.LTRIM,
                    SqlKind.RTRIM,
                    SqlKind.TRIM,
                    SqlKind.CONTAINS,
                    SqlKind.COALESCE,
                    SqlKind.POSITION,
                    SqlKind.MOD,
                    SqlKind.CEIL,
                    SqlKind.FLOOR,
                    // Common compute steps for selecting rows for comparison.
                    SqlKind.COALESCE,
                    SqlKind.LEAST,
                    SqlKind.GREATEST,
                    // Other functions
                    SqlKind.IN,
                )

            // Note that several of these functions also have their SqlKind in
            // SUPPORTED_CALLS. For several functions, we have our own implementations
            // separate from Calcite which use SqlKind.Other_function.
            // This is generally due to differences with Calcite vs Snowflake Syntax.
            // There's no harm in including both for safety, so that is what I've chosen to do.
            private val SUPPORTED_GENERIC_CALL_NAMES =
                setOf(
                    // String manipulation functions.
                    SqlStdOperatorTable.LOWER.name,
                    SqlStdOperatorTable.UPPER.name,
                    StringOperatorTable.CONCAT.name,
                    StringOperatorTable.CONCAT_WS.name,
                    SqlLibraryOperators.LTRIM.name,
                    SqlLibraryOperators.RTRIM.name,
                    StringOperatorTable.CHARINDEX.name,
                    StringOperatorTable.EDITDISTANCE.name,
                    StringOperatorTable.INITCAP.name,
                    StringOperatorTable.STARTSWITH.name,
                    StringOperatorTable.ENDSWITH.name,
                    StringOperatorTable.REGEXP_LIKE.name,
                    StringOperatorTable.RLIKE.name,
                    StringOperatorTable.REGEXP_SUBSTR.name,
                    StringOperatorTable.INSTR.name,
                    StringOperatorTable.REGEXP_INSTR.name,
                    StringOperatorTable.REGEXP_REPLACE.name,
                    StringOperatorTable.REGEXP_COUNT.name,
                    StringOperatorTable.STRTOK.name,
                    StringOperatorTable.LENGTH.name,
                    StringOperatorTable.LPAD.name,
                    StringOperatorTable.RPAD.name,
                    SqlStdOperatorTable.TRUNCATE.name,
                    SqlStdOperatorTable.UPPER.name,
                    SqlStdOperatorTable.LOWER.name,
                    StringOperatorTable.REVERSE.name,
                    StringOperatorTable.CONTAINS.name,
                    CastingOperatorTable.TO_VARCHAR.name,
                    SqlStdOperatorTable.SUBSTRING.name,
                    StringOperatorTable.SUBSTR.name,
                    SqlStdOperatorTable.CHAR_LENGTH.name,
                    // This handles ||
                    SqlStdOperatorTable.CONCAT.name,
                    SqlStdOperatorTable.ABS.name,
                    SqlStdOperatorTable.SIGN.name,
                    SqlStdOperatorTable.ROUND.name,
                    SqlStdOperatorTable.REPLACE.name,
                    SqlLibraryOperators.TRANSLATE3.name,
                    StringOperatorTable.LEFT.name,
                    StringOperatorTable.RIGHT.name,
                    StringOperatorTable.REPEAT.name,
                    StringOperatorTable.SPLIT.name,
                    // Date/Time related functions
                    DatetimeOperatorTable.DATE_TRUNC.name,
                    DatetimeOperatorTable.NANOSECOND.name,
                    SqlStdOperatorTable.HOUR.name,
                    SqlStdOperatorTable.MINUTE.name,
                    SqlStdOperatorTable.SECOND.name,
                    SqlStdOperatorTable.YEAR.name,
                    DatetimeOperatorTable.YEAROFWEEK.name,
                    DatetimeOperatorTable.YEAROFWEEKISO.name,
                    DatetimeOperatorTable.DAY.name,
                    SqlStdOperatorTable.DAYOFMONTH.name,
                    SqlStdOperatorTable.DAYOFWEEK.name,
                    SqlStdOperatorTable.DAYOFYEAR.name,
                    SqlStdOperatorTable.WEEK.name,
                    DatetimeOperatorTable.WEEKOFYEAR.name,
                    DatetimeOperatorTable.DAYOFWEEKISO.name,
                    SqlStdOperatorTable.MONTH.name,
                    SqlStdOperatorTable.QUARTER.name,
                    SqlStdOperatorTable.CURRENT_DATE.name,
                    DatetimeOperatorTable.CURDATE.name,
                    DatetimeOperatorTable.GETDATE.name,
                    // Math functions
                    SqlStdOperatorTable.CEIL.name,
                    SqlStdOperatorTable.FLOOR.name,
                    // Other functions
                    ObjectOperatorTable.GET_PATH.name,
                )

            @JvmStatic
            private fun isSupportedOtherFunction(call: RexCall): Boolean {
                return (call.kind == SqlKind.OTHER_FUNCTION || call.kind == SqlKind.OTHER) &&
                    SUPPORTED_GENERIC_CALL_NAMES.contains(
                        call.operator.name,
                    )
            }

            /**
             * Casts that we want to push into Snowflake.
             * Currently, it's only a cast call to/from variant from/to any datatype
             * and casts to date.
             * i.e. VARIANT::DataType, Datatype::Variant, TIMESTAMP::DATE
             * @param call: Operator call
             * @return true/false based on whether it's a supported cast operation or not.
             */
            private fun isSupportedCast(call: RexCall): Boolean {
                return if (call.kind == SqlKind.CAST) {
                    // Cast to Variant or Cast input is a variant
                    // Support cast to TIMESTAMP_LTZ/TIMESTAMP_TZ. In the future we may want to make this more restrictive
                    // if we ever allow casting a type to TIMESTAMP_LTZ that Snowflake doesn't.
                    // Cast is to Date. In the future we may want to make this more restrictive
                    // if we ever allow casting a type to DATE that Snowflake doesn't.
                    // Cast is to a numeric type
                    (call.getType() is VariantSqlType) ||
                        (call.operands[0].type is VariantSqlType) ||
                        (call.getType().sqlTypeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) ||
                        (call.getType().sqlTypeName == SqlTypeName.TIMESTAMP_TZ) ||
                        (call.getType().sqlTypeName == SqlTypeName.DATE) ||
                        (SqlTypeFamily.NUMERIC.contains(call.getType()))
                } else {
                    false
                }
            }

            /**
             * Test for like variants we can push to Snowflake. Snowflake requires
             * escape to be a String literal.
             */
            private fun isSupportedLike(call: RexCall): Boolean {
                return call.kind == SqlKind.LIKE && (call.operands.size < 3 || (call.operands[2] is RexLiteral))
            }

            /**
             * Enable pushing DateAdd but only if there is the 3 argument version.
             */
            private fun isSupportedDateAdd(call: RexCall): Boolean {
                return call.kind == SqlKind.OTHER_FUNCTION &&
                    call.operator.name == DatetimeOperatorTable.DATEADD.name &&
                    call.operands.size == 3
            }

            /**
             * Important note. If you update this function or its supported functions you will likely change
             * 1 or more filters from being pushed to Snowflake by the Python compiler to instead be
             * pushed by code generation. This means you need to run test_snowflake_catalog_filter_pushdown.py
             * and check if any of the given tests need to have their expected logging messages updated.
             */
            @JvmStatic
            fun isPushableFilter(filter: Filter): Boolean {
                // Not sure what things are ok to push, but we're going to be fairly conservative
                // and whitelist specific things rather than blacklist.
                return isPushableCondition(filter.condition)
            }

            @JvmStatic
            fun isPushableCondition(condition: RexNode): Boolean {
                // Not sure what things are ok to push, but we're going to be fairly conservative
                // and whitelist specific things rather than blacklist.
                return condition.accept(
                    object : RexVisitorImpl<Boolean?>(true) {
                        override fun visitLiteral(literal: RexLiteral): Boolean = true

                        override fun visitInputRef(inputRef: RexInputRef): Boolean = true

                        override fun visitCall(call: RexCall?): Boolean {
                            // Allow select operators to be pushed down.
                            // This list is non-exhaustive, but are generally the functions we've seen
                            // and verified to work. Add more as appropriate.
                            return if (call != null && (
                                    SUPPORTED_CALLS.contains(call.kind) || isSupportedOtherFunction(call) ||
                                        isSupportedCast(
                                            call,
                                        ) || isSupportedLike(call) || isSupportedDateAdd(call)
                                )
                            ) {
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
        }
    }
