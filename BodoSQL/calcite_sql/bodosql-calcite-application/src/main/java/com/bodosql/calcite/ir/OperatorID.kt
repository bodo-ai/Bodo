package com.bodosql.calcite.ir

/**
 * Wrapper for operator IDs. In tests we represent operator IDs by "OPID" instead of the true value to avoid changes in operator ID assignment from causing large diffs when updated expected code
 */
class OperatorID(
    val id: Int,
    val hide: Boolean,
) : Comparable<OperatorID> {
    fun toExpr(): Expr {
        if (hide) {
            return Expr.Raw("OPID")
        }
        return Expr.IntegerLiteral(id)
    }

    override fun toString(): String {
        if (hide) {
            return "OPID"
        }
        return id.toString()
    }

    override fun hashCode(): Int = id.hashCode()

    override fun compareTo(other: OperatorID): Int = id.compareTo(other.id)

    override fun equals(other: Any?): Boolean {
        if (other is OperatorID) {
            return id == other.id
        }
        return false
    }
}
