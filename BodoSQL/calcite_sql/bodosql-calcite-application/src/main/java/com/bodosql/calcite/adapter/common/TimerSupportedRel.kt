package com.bodosql.calcite.adapter.common

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import org.apache.calcite.rel.RelNode
import java.util.Arrays

/**
 * The set of require operations that can support generating timers.
 * This can be removed if BodoPhysicalRel becomes the only timeable
 * operation.
 */
interface TimerSupportedRel : RelNode {
    /**
     * Generates the SingleBatchRelNodeTimer for the appropriate operator.
     */
    fun getTimerType(): SingleBatchRelNodeTimer.OperationType = SingleBatchRelNodeTimer.OperationType.BATCH

    fun operationDescriptor() = "RelNode"

    fun loggingTitle() = "RELNODE_TIMING"

    fun nodeDetails() =
        Arrays.stream(
            nodeString()
                .split("\n".toRegex()).dropLastWhile { it.isEmpty() }
                .toTypedArray(),
        ).findFirst().get()

    /**
     * Generates a string representation of the node.
     */
    fun nodeString(): String

    /**
     * Allows a Timed Physical Rel to override the number of ranks it will utilize.
     * If unknown, return null. Defaults to utilizing all ranks.
     *
     * This can be later integrated to add "parallelism cost" to the cost model.
     */
    fun splitCount(numRanks: Int): Int? = numRanks
}
