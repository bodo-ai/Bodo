package com.bodosql.calcite.rel.core.cachePlanContainers

import com.bodosql.calcite.rel.core.CachedSubPlanBase
import java.lang.RuntimeException

/**
 * A data structure that holds CachedSubPlan information to ensure that
 * every cache node is traversed/visited exactly once. This gives no ordering
 * guarantees and treats all consumers as "equivalent".
 */
class CacheNodeSingleVisitHandler {
    private val seenCacheIDs = mutableSetOf<Int>()
    private val cacheQueue = ArrayDeque<CachedSubPlanBase>()

    /**
     * Adds a CachedSubPlan to the queue if it has not been seen before.
     *
     * @param rel The CachedSubPlan to add to the queue.
     */
    fun add(rel: CachedSubPlanBase) {
        if (!seenCacheIDs.contains(rel.cacheID)) {
            seenCacheIDs.add(rel.cacheID)
            cacheQueue.add(rel)
        }
    }

    /**
     * Remove the next CachedSubPlan from the queue to be
     * processed for the only time.
     */
    fun pop(): CachedSubPlanBase {
        if (cacheQueue.isEmpty()) {
            throw RuntimeException("No more cache nodes to visit.")
        }
        return cacheQueue.removeFirst()
    }

    /**
     * Return if there are no more CachedSubPlans to visit.
     */
    fun isEmpty(): Boolean = cacheQueue.isEmpty()

    /**
     * Return if there are more CachedSubPlans to visit.
     */
    fun isNotEmpty(): Boolean = !isEmpty()
}
