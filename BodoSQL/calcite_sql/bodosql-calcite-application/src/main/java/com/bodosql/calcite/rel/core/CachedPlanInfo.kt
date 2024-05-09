package com.bodosql.calcite.rel.core

import org.apache.calcite.rel.RelRoot

/**
 * Wrapper around a RelRoot to track information about caching in a way that is
 * mutable.
 */
class CachedPlanInfo private constructor(val plan: RelRoot, private var numConsumers: Int) {
    fun removeConsumer() {
        numConsumers--
    }

    fun addConsumer() {
        numConsumers++
    }

    fun getNumConsumers(): Int {
        return numConsumers
    }

    companion object {
        fun create(
            plan: RelRoot,
            numConsumers: Int,
        ): CachedPlanInfo {
            return CachedPlanInfo(plan, numConsumers)
        }
    }
}
