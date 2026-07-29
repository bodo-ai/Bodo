package com.bodosql.calcite.adapter.common

import com.bodosql.calcite.rel.core.CachedSubPlanBase
import org.apache.calcite.rel.RelNode

/**
 * Shared utilities for traversing [RelNode] trees that may contain
 * [CachedSubPlanBase] nodes inserted by covering expression caching.
 */
object RelUtils {
    /**
     * Depth-first search for the first descendant (or the node itself) of
     * type [T] in the given rel tree, traversing through [CachedSubPlanBase]
     * bodies. Returns the first match found in pre-order traversal of inputs,
     * or `null` if no node of type [T] is found.
     *
     * Structural assumption: each source-convention converter (Snowflake,
     * Iceberg) sits above a subtree containing exactly one source rel of that
     * convention.
     */
    inline fun <reified T : RelNode> findRelOfTypeOrNull(node: RelNode): T? = findRelOfTypeOrNull(node, T::class.java)

    /**
     * Depth-first search for the first descendant (or the node itself) of
     * type [T]. Throws [IllegalStateException] if none is found.
     *
     */
    inline fun <reified T : RelNode> findRelOfType(node: RelNode): T =
        findRelOfTypeOrNull<T>(node)
            ?: throw IllegalStateException("Cannot find ${T::class.java.simpleName} in rel tree")

    @PublishedApi
    internal fun <T : RelNode> findRelOfTypeOrNull(
        node: RelNode,
        type: Class<T>,
    ): T? {
        if (type.isInstance(node)) return type.cast(node)
        if (node is CachedSubPlanBase) {
            return findRelOfTypeOrNull(node.cachedPlan.plan, type)
        }
        for (input in node.inputs) {
            val result = findRelOfTypeOrNull(input, type)
            if (result != null) return result
        }
        return null
    }
}
