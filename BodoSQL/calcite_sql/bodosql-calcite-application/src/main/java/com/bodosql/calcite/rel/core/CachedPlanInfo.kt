package com.bodosql.calcite.rel.core

import org.apache.calcite.rel.RelNode

/**
 * Wrapper around a RelRoot to track information about caching in a way that is
 * mutable.
 */
class CachedPlanInfo private constructor(var plan: RelNode, private var numConsumers: Int) {
    fun removeConsumer() {
        numConsumers--
    }

    fun addConsumer() {
        numConsumers++
    }

    fun addConsumers(num: Int) {
        numConsumers += num
    }

    fun getNumConsumers(): Int {
        return numConsumers
    }

    companion object {
        fun create(
            plan: RelNode,
            numConsumers: Int,
        ): CachedPlanInfo {
            return CachedPlanInfo(plan, numConsumers)
        }
    }
}
