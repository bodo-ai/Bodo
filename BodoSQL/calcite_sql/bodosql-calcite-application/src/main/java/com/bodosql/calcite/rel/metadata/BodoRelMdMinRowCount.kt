package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.metadata.RelMdMinRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery

class BodoRelMdMinRowCount : RelMdMinRowCount() {
    // Joins that produce a cartesian product
    // when the condition is always true
    private val productJoinTypes =
        setOf(
            JoinRelType.INNER,
            JoinRelType.LEFT,
            JoinRelType.RIGHT,
            JoinRelType.FULL,
        )

    /**
     * Provide a minimum row count for join. Calcite
     * just provides 0, which is a safe bound, but when
     * the condition is always true, we can provide a tighter
     * bound because we know the join will produce a cartesian
     * product.
     *
     * This result still isn't exact because we may not know
     * the exact size of the inputs, but when we do it can
     * be useful for optimizations that depend on knowing the
     * exact row count statically (e.g. SingleValue pruning).
     *
     */
    override fun getMinRowCount(
        rel: Join,
        mq: RelMetadataQuery,
    ): Double {
        return if (rel.condition.isAlwaysTrue && productJoinTypes.contains(rel.joinType)) {
            val left = mq.getMinRowCount(rel.left)
            val right = mq.getMinRowCount(rel.right)
            right?.let { left?.times(it) } ?: 0.0
        } else {
            0.0
        }
    }
}
