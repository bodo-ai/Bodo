package com.bodosql.calcite.rel.core.cachePlanContainers

import com.bodosql.calcite.rel.core.CachedSubPlanBase

/**
 * Cache visitor that processes a CachedSubPlanBase on its first encounter with the provided
 * function. Then any future encounters will use the cached result of the function. This ensures
 * that the CachedSubPlanBase is only modified by the first function but all consumers can
 * access the result.
 */
class CachedResultVisitor<E, T : CachedSubPlanBase>(private val resultFunction: (T) -> E) {
    private val cacheMap: MutableMap<Int, E> = mutableMapOf()

    /**
     * Get the result of the CachedSubPlanBase by executing the result
     * function. If the result has already been computed, then we just
     * fetch the result from the cache, which allows the operation
     * to be both efficient and idempotent.
     *
     * @param rel The CachedSubPlanBase to get the result of.
     * @return The result of the CachedSubPlanBase.
     */
    fun visit(rel: T): E {
        if (!cacheMap.containsKey(rel.cacheID)) {
            cacheMap[rel.cacheID] = resultFunction.invoke(rel)
        }
        return cacheMap[rel.cacheID]!!
    }
}
