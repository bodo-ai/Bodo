package com.bodosql.calcite.prepare

import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind

data class NonEqualityJoinFilterColumnInfo(
    val type: NonEqualityType,
    val columnIndex: Int,
    val originalJoinBuildIndex: Int,
) {
    override fun toString(): String {
        // Note: We don't include originalJoinBuildIndex because it may have been remapped.
        return "$$columnIndex $type buildCol"
    }

    companion object {
        /**
         * Convert a RexNode that is a supported non-equality type and divided evenly across the build
         * and probe side of a join into its probe, build, and enum type. Returns null if the node
         * cannot be split.
         * @param node The RexNode to split.
         * @param numLeftCols The number of columns on the left side of the join.
         * @return A triple containing the probe column index, build column index, and non-equality type. Null
         * if the node cannot be split.
         */
        fun splitRexNodeByType(
            node: RexNode,
            numLeftCols: Int,
        ): Triple<Int, Int, NonEqualityType>? {
            val supportedKinds =
                setOf(SqlKind.LESS_THAN, SqlKind.LESS_THAN_OR_EQUAL, SqlKind.GREATER_THAN, SqlKind.GREATER_THAN_OR_EQUAL)
            if (node !is RexCall || !supportedKinds.contains(node.op.kind) || node.getOperands().size != 2) {
                return null
            }
            val left = node.operands[0]
            val right = node.operands[1]
            if (left !is RexInputRef || right !is RexInputRef) {
                return null
            }
            val leftIndex = left.index
            val rightIndex = right.index
            if (leftIndex < numLeftCols && rightIndex >= numLeftCols) {
                // Probe is on the left, no need to swap the operator.
                return Triple(leftIndex, rightIndex, nonEqualityTypeFromKind(node.op.kind))
            } else if (rightIndex < numLeftCols && leftIndex >= numLeftCols) {
                // Probe is on the right, swap the operator.
                return Triple(rightIndex, leftIndex, nonEqualityTypeFromKind(node.op.kind).reverse())
            } else {
                // Operator is on 1 side of the join.
                return null
            }
        }
    }
}

enum class NonEqualityType {
    LESS_THAN {
        override fun toString(): String {
            return "<"
        }
    },
    LESS_THAN_OR_EQUAL {
        override fun toString(): String {
            return "<="
        }
    },
    GREATER_THAN {
        override fun toString(): String {
            return ">"
        }
    },
    GREATER_THAN_OR_EQUAL {
        override fun toString(): String {
            return ">="
        }
    }, ;

    override fun toString(): String {
        return when (this) {
            LESS_THAN -> "<"
            LESS_THAN_OR_EQUAL -> "<="
            GREATER_THAN -> ">"
            GREATER_THAN_OR_EQUAL -> ">="
        }
    }

    fun reverse(): NonEqualityType {
        return when (this) {
            LESS_THAN -> GREATER_THAN
            LESS_THAN_OR_EQUAL -> GREATER_THAN_OR_EQUAL
            GREATER_THAN -> LESS_THAN
            GREATER_THAN_OR_EQUAL -> LESS_THAN_OR_EQUAL
        }
    }
}

fun nonEqualityTypeFromKind(kind: SqlKind): NonEqualityType {
    return when (kind) {
        SqlKind.LESS_THAN -> NonEqualityType.LESS_THAN
        SqlKind.LESS_THAN_OR_EQUAL -> NonEqualityType.LESS_THAN_OR_EQUAL
        SqlKind.GREATER_THAN -> NonEqualityType.GREATER_THAN
        SqlKind.GREATER_THAN_OR_EQUAL -> NonEqualityType.GREATER_THAN_OR_EQUAL
        else -> throw IllegalArgumentException("Unsupported kind: $kind")
    }
}
