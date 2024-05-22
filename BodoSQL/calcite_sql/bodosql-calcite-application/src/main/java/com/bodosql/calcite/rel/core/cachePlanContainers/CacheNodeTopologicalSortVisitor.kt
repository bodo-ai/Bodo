package com.bodosql.calcite.rel.core.cachePlanContainers

import com.bodosql.calcite.rel.core.CachedSubPlanBase
import java.lang.RuntimeException

/**
 * Cache visitor node that ensures cache nodes are visited in a valid topological order
 * and then are returned exactly once. This is done by checking the number of consumers
 * associated with a cache node and only returning the cache node when all consumers have
 * been reached.
 *
 * In addition, since this use case typically means we want to do "something"
 * for every consumer, this data structure is generic to track additional information
 * about each consumer. For example, if we may want to inline a cache node then you can
 * specify E as a RelNode and provide the parent when inserting.
 */
class CacheNodeTopologicalSortVisitor<E> {
    private val consumedNodes: MutableMap<Int, Pair<CachedSubPlanBase, MutableList<E>>> = mutableMapOf()
    private val cacheQueue: ArrayDeque<Pair<CachedSubPlanBase, List<E>>> = ArrayDeque()

    /**
     * Adds a CachedSubPlan to the queue if it has not been seen before.
     *
     * @param rel The CachedSubPlan to add to the queue.
     */
    fun add(
        rel: CachedSubPlanBase,
        info: E,
    ) {
        if (!consumedNodes.containsKey(rel.cacheID)) {
            consumedNodes[rel.cacheID] = Pair(rel, mutableListOf())
        }
        val infoList = consumedNodes[rel.cacheID]!!.second
        infoList.add(info)
        // If we have seen every consumer we can add it to the queue
        if (rel.cachedPlan.getNumConsumers() == infoList.size) {
            cacheQueue.add(Pair(rel, infoList.toList()))
            consumedNodes.remove(rel.cacheID)
        }
    }

    /**
     * Remove the next CachedSubPlan from the queue to be
     * processed for the only time. This ensures a result
     * in topological order.
     */
    fun pop(): Pair<CachedSubPlanBase, List<E>> {
        if (cacheQueue.isEmpty()) {
            throw RuntimeException("No cache nodes are ready.")
        }
        return cacheQueue.removeFirst()
    }

    /**
     * Return if there are any plans that are "ready"
     * to visit. Returning false doesn't mean the visitor is "empty"
     * because some nodes may not have visited consumer yet.
     */
    fun hasReadyNode(): Boolean = cacheQueue.isNotEmpty()
}
