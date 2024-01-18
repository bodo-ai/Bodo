package com.bodosql.calcite.application.utils

import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeName

/**
 * Helper class of static functions to determine if a join contains a valid
 * condition that Bodo can support, possibly after some transformation(s).
 */
class BodoJoinConditionUtil {
    companion object {

        /**
         * SqlTypeNames for literals that can definitely be used in a join condition's
         * generated code.
         */
        @JvmStatic
        private val validLiteralTypes = setOf(
            SqlTypeName.TINYINT, SqlTypeName.SMALLINT, SqlTypeName.INTEGER, SqlTypeName.BIGINT,
            SqlTypeName.FLOAT, SqlTypeName.REAL, SqlTypeName.DOUBLE, SqlTypeName.DECIMAL, SqlTypeName.CHAR,
            SqlTypeName.VARCHAR, SqlTypeName.BOOLEAN,
        )

        /**
         * Determine if the literal is valid to use in a join condition's
         * generated code.
         */
        @JvmStatic
        private fun isValidLiteral(literal: RexLiteral): Boolean {
            return validLiteralTypes.contains(literal.type.sqlTypeName)
        }

        /**
         * SqlKinds of functions that are valid to use in a join condition's
         * generated code.
         */
        @JvmStatic
        private val validBuiltinFunctionKinds = setOf(
            SqlKind.EQUALS, SqlKind.NOT_EQUALS, SqlKind.GREATER_THAN,
            SqlKind.GREATER_THAN_OR_EQUAL, SqlKind.LESS_THAN,
            SqlKind.LESS_THAN_OR_EQUAL, SqlKind.AND, SqlKind.OR, SqlKind.PLUS, SqlKind.MINUS,
            SqlKind.TIMES, SqlKind.DIVIDE, SqlKind.NOT, SqlKind.IS_NOT_TRUE,
        )

        /**
         * Determine if a function with a kind that is not in otherFunctionKinds
         * (Calcite builtin functions) is valid to use in a join condition's
         * generated code.
         */
        @JvmStatic private fun isValidBuiltinFunction(call: RexCall): Boolean {
            return validBuiltinFunctionKinds.contains(call.kind)
        }

        /**
         * SqlKinds of functions that represent functions whose names need to be
         * checked for validity.
         */
        @JvmStatic
        private val otherFunctionKinds = setOf(SqlKind.OTHER, SqlKind.OTHER_FUNCTION)

        /**
         * Determine if a function with a kind that is in otherFunctionKinds
         * (generally functions we have added) is valid to use in a join condition's
         * generated code.
         */
        @JvmStatic private fun isValidOtherFunction(call: RexCall): Boolean {
            return otherFunctionKinds.contains(call.kind) && call.op.name == NumericOperatorTable.POW.name
        }

        /**
         * Determine if a generic RexNode is valid to use in a join condition's
         * generated code. Current we support a subset of literals, input refs,
         * and function calls.
         */
        @JvmStatic
        fun isValidNode(node: RexNode): Boolean {
            return when (node) {
                is RexLiteral -> isValidLiteral(node)
                is RexInputRef -> true
                is RexCall -> if (isValidBuiltinFunction(node) || isValidOtherFunction(node)) {
                    node.operands.all {
                        isValidNode(it)
                    }
                } else {
                    false
                }
                else -> false
            }
        }

        /**
         * Determine if the given node section is pushable. The section
         * is pushable if either it is an invalid node that uses at most one
         * side of the join or a valid node whose inputs are all valid nodes.
         */
        @JvmStatic
        fun isPushableFunction(node: RexNode, numLeftColumns: Int, totalColumns: Int): Boolean {
            return when (node) {
                // Even unsupported literals can always be pushed down as columns.
                is RexLiteral, is RexInputRef -> true
                is RexCall -> {
                    if (isValidBuiltinFunction(node) || isValidOtherFunction(node)) {
                        node.operands.all {
                            isPushableFunction(it, numLeftColumns, totalColumns)
                        }
                    } else {
                        // This is a node we must be able to push down.
                        val usedColumns = RelOptUtil.InputFinder.bits(node)
                        val usesLeft = !(usedColumns.get(0, numLeftColumns).isEmpty)
                        val usesRight = !(usedColumns.get(numLeftColumns, totalColumns).isEmpty)
                        // We can transform the data if at least one of the tables is unused.
                        return !usesLeft || !usesRight
                    }
                }
                // Fallback for safety.
                else -> false
            }
        }

        /**
         * Given a join whose condition returns False for
         * isValidNode, this function determines if it's possible for
         * Bodo to transform it to valid format.
         *
         * The function is used to either block features that are only valid/safe
         * if everything can be supported or to facilitate a faster failure with
         * a clean error message.
         */
        @JvmStatic
        fun isTransformableToValid(join: Join): Boolean {
            return when (join.joinType) {
                JoinRelType.INNER -> {
                    true
                }
                JoinRelType.ANTI, JoinRelType.SEMI -> {
                    // Note: We don't support SEMI or ANTI join, so this shouldn't be reachable.
                    // We add this to be defensive.
                    false
                }
                else -> {
                    isPushableFunction(join.condition, join.left.getRowType().fieldCount, join.getRowType().fieldCount)
                }
            }
        }
    }
}
