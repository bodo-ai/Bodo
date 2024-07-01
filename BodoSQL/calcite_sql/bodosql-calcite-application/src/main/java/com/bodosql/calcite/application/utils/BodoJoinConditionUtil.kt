package com.bodosql.calcite.application.utils

import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexPermuteInputsShuttle
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
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
        private val validLiteralTypes =
            setOf(
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
        private val validBuiltinFunctionKinds =
            setOf(
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
            return (call.op != SqlStdOperatorTable.DATETIME_PLUS) &&
                (call.op != SqlStdOperatorTable.MINUS_DATE) &&
                validBuiltinFunctionKinds.contains(call.kind)
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
                is RexCall ->
                    if (isValidBuiltinFunction(node) || isValidOtherFunction(node)) {
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
         * Returns a pair to describe if the left and right table are used.
         */
        @JvmStatic
        private fun determineUsedTables(
            node: RexNode,
            numLeftColumns: Int,
            totalColumns: Int,
        ): Pair<Boolean, Boolean> {
            val usedColumns = RelOptUtil.InputFinder.bits(node)
            val usesLeft = !(usedColumns.get(0, numLeftColumns).isEmpty)
            val usesRight = !(usedColumns.get(numLeftColumns, totalColumns).isEmpty)
            return Pair(usesLeft, usesRight)
        }

        /**
         * Determine if the given node section is pushable. The section
         * is pushable if either it is an invalid node that uses at most one
         * side of the join or a valid node whose inputs are all valid nodes.
         */
        @JvmStatic
        fun isPushableFunction(
            node: RexNode,
            numLeftColumns: Int,
            totalColumns: Int,
        ): Boolean {
            return when (node) {
                // Even unsupported literals can always be pushed down as columns.
                is RexLiteral, is RexInputRef -> true
                is RexCall -> {
                    if (isValidBuiltinFunction(node) || isValidOtherFunction(node)) {
                        node.operands.all {
                            isPushableFunction(it, numLeftColumns, totalColumns)
                        }
                    } else {
                        val (usesLeft, usesRight) = determineUsedTables(node, numLeftColumns, totalColumns)
                        // We can transform the data if at least one of the tables is unused.
                        return !usesLeft || !usesRight
                    }
                }
                // Fallback for safety.
                else -> false
            }
        }

        /**
         * Given a function that requires a transformation, extracts all pushable
         * components to the appropriate table side. This function should only be
         * called by extractPushableFunctions and recursively by itself. It works
         * by traversing over the node components and then updating leftNodes, rightNodes,
         * and nodesCache.
         * @param node The node that may need parts extracted.
         * @param numLeftColumns The number of columns on the left side.
         * @param totalColumns The number of total columns.
         * @param leftShuttle The RexVisitor to remap inputs refs desired for the left table.
         * @param rightShuttle The RexVisitor to remap inputs refs desired for the right table.
         * @param leftNodes OUTPUT: The list of expressions that should be inserted into the left child.
         * @param rightNodes OUTPUT: The list of expressions that should be inserted into the right child.
         * @param nodeCache OUTPUT: A mapping of expression to index number table side + index number. This is
         * used for both caching the update and later transformation stages.
         */
        private fun extractPushableFunctionsInternal(
            node: RexNode,
            numLeftColumns: Int,
            totalColumns: Int,
            leftShuttle: RexPermuteInputsShuttle,
            rightShuttle: RexPermuteInputsShuttle,
            leftNodes: MutableList<RexNode>,
            rightNodes: MutableList<RexNode>,
            nodeCache: MutableMap<RexNode, Pair<Boolean, Int>>,
        ) {
            when {
                // Any that already matches we can skip.
                nodeCache.contains(node) -> return
                // If a literal can be used directly in the join then we don't need
                // to convert it into a column (which will generally be more expensive).
                // However, if the literal is not supported then we need to push it into
                // one of the inputs as a column.
                node is RexLiteral && !isValidLiteral(node) -> {
                    // Literals can go to either side, so append to left since it should be larger.
                    nodeCache[node] = Pair(false, numLeftColumns + leftNodes.size)
                    leftNodes.add(node.accept(leftShuttle))
                }
                node is RexCall -> {
                    if (isValidBuiltinFunction(node) || isValidOtherFunction(node)) {
                        // Just visit every child. The function operands may need to be pushed
                        // down but the function doesn't. For example =(my_func1(...), my_func2(...))
                        // would push my_func1(...) and my_func2(...) but not the equals function.
                        node.operands.map {
                            extractPushableFunctionsInternal(
                                it,
                                numLeftColumns,
                                totalColumns,
                                leftShuttle,
                                rightShuttle,
                                leftNodes,
                                rightNodes,
                                nodeCache,
                            )
                        }
                    } else {
                        val (usesLeft, usesRight) = determineUsedTables(node, numLeftColumns, totalColumns)
                        // We can't push down this section if it uses both tables. No point in trying
                        // to push the children.
                        if (!usesLeft || !usesRight) {
                            if (usesRight) {
                                nodeCache[node] = Pair(true, (totalColumns - numLeftColumns) + rightNodes.size)
                                rightNodes.add(node.accept(rightShuttle))
                            } else {
                                // Note: If no column is used then this prioritizes the left side as that's the probe side.
                                nodeCache[node] = Pair(false, numLeftColumns + leftNodes.size)
                                leftNodes.add(node.accept(leftShuttle))
                            }
                        }
                    }
                }
                // All other case do nothing
                else -> return
            }
        }

        /**
         * Given a function that requires a transformation, extracts all pushable
         * components to the appropriate table side.
         * @param condition The whole condition that needs parts extracted.
         * @param numLeftColumns The number of columns on the left side.
         * @param totalColumns The number of total columns.
         * @param leftShuttle The RexVisitor to remap inputs refs desired for the left table.
         * @param rightShuttle The RexVisitor to remap inputs refs desired for the right table.
         * @return Triple consisting of:
         *      - A list of nodes added to the left child.
         *      - A list of nodes added to the right child
         *      - A mapping from the node to a Pair of value mapping to isRightTable and column number within
         *        the table.
         */
        @JvmStatic
        fun extractPushableFunctions(
            condition: RexNode,
            numLeftColumns: Int,
            totalColumns: Int,
            leftShuttle: RexPermuteInputsShuttle,
            rightShuttle: RexPermuteInputsShuttle,
        ): Triple<List<RexNode>, List<RexNode>, Map<RexNode, Pair<Boolean, Int>>> {
            val leftNodes = ArrayList<RexNode>()
            val rightNodes = ArrayList<RexNode>()
            val nodeMap = HashMap<RexNode, Pair<Boolean, Int>>()
            extractPushableFunctionsInternal(
                condition,
                numLeftColumns,
                totalColumns,
                leftShuttle,
                rightShuttle,
                leftNodes,
                rightNodes,
                nodeMap,
            )
            return Triple(leftNodes, rightNodes, nodeMap)
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
                    isPushableFunction(join.condition, join.left.rowType.fieldCount, join.rowType.fieldCount)
                }
            }
        }

        /**
         * Determine if a join can become valid through a transformation.
         * Returns false if either the join doesn't require a transformation
         * or no transformation is possible.
         */
        @JvmStatic
        fun requiresTransformationToValid(join: Join): Boolean {
            return !isValidNode(join.condition) && isTransformableToValid(join)
        }
    }
}
