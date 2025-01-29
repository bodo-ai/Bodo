package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.logical.LogicalFilter
import org.apache.calcite.rel.metadata.BuiltInMetadata.ColumnUniqueness
import org.apache.calcite.rel.metadata.MetadataDef
import org.apache.calcite.rel.metadata.MetadataHandler
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexSubQuery
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet
import org.apache.calcite.util.Pair
import org.apache.commons.lang3.mutable.MutableBoolean
import java.util.BitSet
import java.util.function.Consumer

class BodoRelMdColumnUniqueness : MetadataHandler<ColumnUniqueness> {
    override fun getDef(): MetadataDef<ColumnUniqueness> = ColumnUniqueness.DEF

    fun areColumnsUnique(
        rel: Join,
        mq: RelMetadataQuery,
        columns: ImmutableBitSet,
        ignoreNulls: Boolean,
    ): Boolean? {
        var columns = columns
        columns = decorateWithConstantColumnsFromPredicates(columns, rel, mq)
        val left = rel.left
        val right = rel.right

        // Semi or anti join should ignore uniqueness of the right input.
        if (!rel.joinType.projectsRight()) {
            return mq.areColumnsUnique(left, columns, ignoreNulls)
        }
        val leftColumnCount = rel.left.rowType.fieldCount
        // Divide up the input column mask into column masks for the left and
        // right sides of the join
        val leftAndRightColumns = splitLeftAndRightColumns(leftColumnCount, columns)
        var leftColumns = leftAndRightColumns.left
        var rightColumns = leftAndRightColumns.right

        // for FULL OUTER JOIN if columns contain column from both inputs it is not
        // guaranteed that the result will be unique
        if (!ignoreNulls && rel.joinType == JoinRelType.FULL && leftColumns.cardinality() > 0 && rightColumns.cardinality() > 0) {
            return false
        }
        val joinInfo = rel.analyzeCondition()

        // Joining with a singleton constrains the keys on the other table
        val rightMaxRowCount = mq.getMaxRowCount(right)
        if (rightMaxRowCount != null && rightMaxRowCount <= 1.0) {
            leftColumns = leftColumns.union(joinInfo.leftSet())
        }
        val leftMaxRowCount = mq.getMaxRowCount(left)
        if (leftMaxRowCount != null && leftMaxRowCount <= 1.0) {
            rightColumns = rightColumns.union(joinInfo.rightSet())
        }

        // If we have an inner join on either side, we may be able to determine
        // uniqueness from the condition.
        var equalities = findCommonEqualities(rel.condition)
        // Check each equality. If we are only checking uniqueness with a single
        // side we can include the other side.
        val isOuterLeft = rel.joinType.generatesNullsOnRight()
        val isOuterRight = rel.joinType.generatesNullsOnLeft()
        var combinedLeftBits = leftColumns
        var combinedRightBits = rightColumns
        var bitAdded = true
        while (bitAdded) {
            val addedLeftCols = BitSet()
            val addedRightCols = BitSet()
            bitAdded = false
            val unmatchedEqualities: HashSet<RexCall> = HashSet()
            for (equals in equalities) {
                var arg0 = (equals!!.getOperands()[0] as RexInputRef).index
                var arg1 = (equals.getOperands()[1] as RexInputRef).index
                if (arg0 < leftColumnCount && arg1 < leftColumnCount) {
                    // Both columns are in the left table.
                    if (!isOuterLeft) {
                        // An outer join may not have the columns equal
                        val foundArg0 = combinedLeftBits[arg0]
                        val foundArg1 = combinedLeftBits[arg1]
                        if (foundArg0 && !foundArg1 || !foundArg0 && foundArg1) {
                            addedLeftCols.set(arg0)
                            addedLeftCols.set(arg1)
                            bitAdded = true
                        } else {
                            unmatchedEqualities.add(equals)
                        }
                    }
                } else if (arg0 >= leftColumnCount && arg1 >= leftColumnCount) {
                    // Both columns are in the right table.
                    if (!isOuterRight) {
                        // An outer join may not have the columns equal
                        arg0 -= leftColumnCount
                        arg1 -= leftColumnCount
                        val foundArg0 = combinedRightBits[arg0]
                        val foundArg1 = combinedRightBits[arg1]
                        if (foundArg0 && !foundArg1 || !foundArg0 && foundArg1) {
                            addedRightCols.set(arg0)
                            addedRightCols.set(arg1)
                            bitAdded = true
                        } else {
                            unmatchedEqualities.add(equals)
                        }
                    }
                } else {
                    // Left and right are in different tables
                    var leftArg: Int
                    var rightArg: Int
                    if (arg0 < leftColumnCount) {
                        // arg0 -> Left, arg1 -> Right
                        leftArg = arg0
                        rightArg = arg1 - leftColumnCount
                    } else {
                        // arg0 -> Right, arg1 -> Left
                        leftArg = arg1
                        rightArg = arg0 - leftColumnCount
                    }
                    val foundLeft = combinedLeftBits[leftArg]
                    val foundRight = combinedRightBits[rightArg]
                    if (foundLeft && !foundRight) {
                        if (!isOuterLeft) {
                            addedRightCols.set(rightArg)
                            bitAdded = true
                        }
                    } else if (!foundLeft && foundRight) {
                        if (!isOuterRight) {
                            addedLeftCols.set(leftArg)
                            bitAdded = true
                        }
                    } else {
                        unmatchedEqualities.add(equals)
                    }
                }
            }
            equalities = unmatchedEqualities
            combinedLeftBits = combinedLeftBits.union(addedLeftCols)
            combinedRightBits = combinedRightBits.union(addedRightCols)
        }
        leftColumns = combinedLeftBits
        rightColumns = combinedRightBits

        // If the original column mask contains columns from both the left and
        // right hand side, then the columns are unique if and only if they're
        // unique for their respective join inputs
        val leftUnique = mq.areColumnsUnique(left, leftColumns, ignoreNulls)
        val rightUnique = mq.areColumnsUnique(right, rightColumns, ignoreNulls)
        if (leftColumns.cardinality() > 0 && rightColumns.cardinality() > 0) {
            return if (leftUnique == null || rightUnique == null) {
                null
            } else {
                leftUnique && rightUnique
            }
        }

        // If we're only trying to determine uniqueness for columns that
        // originate from one join input, then determine if the equijoin
        // columns from the other join input are unique.  If they are, then
        // the columns are unique for the entire join if they're unique for
        // the corresponding join input, provided that input is not null
        // generating.
        if (leftColumns.cardinality() > 0) {
            if (rel.joinType.generatesNullsOnLeft()) {
                return false
            }
            val rightJoinColsUnique = mq.areColumnsUnique(right, joinInfo.rightSet(), ignoreNulls)
            return if (rightJoinColsUnique == null || leftUnique == null) {
                null
            } else {
                rightJoinColsUnique && leftUnique
            }
        } else if (rightColumns.cardinality() > 0) {
            if (rel.joinType.generatesNullsOnRight()) {
                return false
            }
            val leftJoinColsUnique = mq.areColumnsUnique(left, joinInfo.leftSet(), ignoreNulls)
            return if (leftJoinColsUnique == null || rightUnique == null) {
                null
            } else {
                leftJoinColsUnique && rightUnique
            }
        }
        return false
    }

    companion object {
        /**
         * Takes a RexNode that represents a condition and finds
         * all equalities that are always True. This is used by Join
         * to help identify when the columns are unique.
         */
        @JvmStatic
        private fun findCommonEqualities(cond: RexNode): java.util.HashSet<RexCall> {
            // Note: Hash(RexNode) is equivalent to RexNode.toString().
            val equalityExprs = java.util.HashSet<RexCall>()
            if (cond.kind == SqlKind.EQUALS) {
                val equalNode = cond as RexCall
                if (equalNode.getOperands()[0] is RexInputRef &&
                    equalNode.getOperands()[1] is RexInputRef
                ) {
                    equalityExprs.add(equalNode)
                }
            } else if (cond.kind == SqlKind.AND) {
                val andCond = cond as RexCall
                // And can have many children, we want to merge on
                // all of them
                for (operandCond in andCond.operands) {
                    equalityExprs.addAll(findCommonEqualities(operandCond))
                }
            } else if (cond.kind == SqlKind.OR) {
                val orCond = cond as RexCall
                // Or can have many children, we only want to merge on nodes common to all of them
                equalityExprs.addAll(findCommonEqualities(orCond.operands[0]))
                for (i in 1 until orCond.operands.size) {
                    val otherExprs = findCommonEqualities(orCond.operands[i])
                    equalityExprs.retainAll(otherExprs)
                }
            }
            return equalityExprs
        }

        /** Splits a column set between left and right sets.
         * Note: Copied from Calcite.
         */
        @JvmStatic
        private fun splitLeftAndRightColumns(
            leftCount: Int,
            columns: ImmutableBitSet,
        ): Pair<ImmutableBitSet, ImmutableBitSet> {
            val leftBuilder = ImmutableBitSet.builder()
            val rightBuilder = ImmutableBitSet.builder()
            for (bit in columns) {
                if (bit < leftCount) {
                    leftBuilder.set(bit)
                } else {
                    rightBuilder.set(bit - leftCount)
                }
            }
            return Pair.of(leftBuilder.build(), rightBuilder.build())
        }

        /**
         * Deduce constant columns from predicates of rel and return the union
         * bitsets of checkingColumns and the constant columns.
         *
         * Note: Copied from Calcite
         */
        @JvmStatic
        private fun decorateWithConstantColumnsFromPredicates(
            checkingColumns: ImmutableBitSet,
            rel: RelNode,
            mq: RelMetadataQuery,
        ): ImmutableBitSet {
            val predicates = mq.getPulledUpPredicates(rel)
            if (!RelOptPredicateList.isEmpty(predicates)) {
                val constantIndexes = getConstantColumnSet(predicates)
                if (!constantIndexes.isEmpty) {
                    return checkingColumns.union(ImmutableBitSet.of(constantIndexes))
                }
            }
            // If no constant columns deduced, return the original "checkingColumns".
            return checkingColumns
        }

        /**
         * Returns the set of columns that are set to a constant literal or a scalar query (as
         * in a correlated subquery). Examples of constants are `x` in the following:
         * <pre>SELECT x FROM table WHERE x = 5</pre>
         * and
         * <pre>SELECT x, y FROM table WHERE x = (SELECT MAX(x) FROM table)</pre>
         *
         *
         * NOTE: Subqueries that reference correlating variables are not considered constant:
         * <pre>SELECT x, y FROM table A WHERE x = (SELECT MAX(x) FROM table B WHERE A.y = B.y)</pre>
         *
         * Note: Copied from Calcite.
         */
        @JvmStatic
        private fun getConstantColumnSet(relOptPredicateList: RelOptPredicateList): ImmutableBitSet {
            val builder = ImmutableBitSet.builder()
            relOptPredicateList.constantMap.keys
                .stream()
                .filter { obj: RexNode? -> RexInputRef::class.java.isInstance(obj) }
                .map { obj: RexNode? ->
                    RexInputRef::class.java.cast(
                        obj,
                    )
                }.map { obj: RexInputRef -> obj.index }
                .forEach { bit: Int? -> builder.set(bit!!) }
            relOptPredicateList.pulledUpPredicates.forEach(
                Consumer { rex: RexNode ->
                    if (rex.kind == SqlKind.EQUALS ||
                        rex.kind == SqlKind.IS_NOT_DISTINCT_FROM
                    ) {
                        val ops = (rex as RexCall).getOperands()
                        val op0 = ops[0]
                        val op1 = ops[1]
                        addInputRefIfOtherConstant(builder, op0, op1)
                        addInputRefIfOtherConstant(builder, op1, op0)
                    }
                },
            )
            return builder.build()
        }

        /**
         * Note: Copied from Calcite.
         */
        @JvmStatic
        private fun addInputRefIfOtherConstant(
            builder: ImmutableBitSet.Builder,
            inputRef: RexNode,
            other: RexNode,
        ) {
            if (inputRef is RexInputRef &&
                (other.kind == SqlKind.LITERAL || isConstantScalarQuery(other))
            ) {
                builder.set(inputRef.index)
            }
        }

        /**
         * Returns whether the supplied [RexNode] is a constant scalar subquery - one that does not
         * reference any correlating variables.
         *
         * Note: Copied from Calcite.
         */
        @JvmStatic
        private fun isConstantScalarQuery(rexNode: RexNode): Boolean {
            if (rexNode.kind == SqlKind.SCALAR_QUERY) {
                val hasCorrelatingVars = MutableBoolean(false)
                (rexNode as RexSubQuery).rel.accept(
                    object : RelShuttleImpl() {
                        override fun visit(filter: LogicalFilter): RelNode {
                            if (RexUtil.containsCorrelation(filter.condition)) {
                                hasCorrelatingVars.setTrue()
                                return filter
                            }
                            return super.visit(filter)
                        }
                    },
                )
                return hasCorrelatingVars.isFalse
            }
            return false
        }
    }
}
