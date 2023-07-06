package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType
import java.util.*

interface PandasRel : RelNode {
    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("PANDAS", PandasRel::class.java)
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    fun emit(implementor: Implementor): Dataframe

    /**
     * Returns true if this node can be cached with another call to the same node.
     */
    fun canUseNodeCache(): Boolean =
        // Do not use node caching with streaming until we identify how to properly do this.
        !traitSet.contains(BatchingProperty.STREAMING)

    /**
     * Allows a PandasRel to override the number of ranks it will utilize.
     * If unknown, return null. Defaults to utilizing all ranks.
     */
    fun splitCount(numRanks: Int): Int? = numRanks

    /**
     * Generates the SingleBatchRelNodeTimer for the appropriate operator.
     */
    fun getTimerType(): SingleBatchRelNodeTimer.OperationType = SingleBatchRelNodeTimer.OperationType.BATCH

    fun operationDescriptor() = "RelNode"
    fun loggingTitle() = "RELNODE_TIMING"

    fun nodeDetails() = Arrays.stream(
        RelOptUtil.toString(this)
            .split("\n".toRegex()).dropLastWhile { it.isEmpty() }
            .toTypedArray()).findFirst().get()

    /**
     * Determine if an operator is streaming.
     */
    fun isStreaming() = traitSet.contains(BatchingProperty.STREAMING)

    interface Implementor {
        fun visitChild(input: RelNode, ordinal: Int): Dataframe

        fun visitChildren(inputs: List<RelNode>): List<Dataframe> =
            inputs.mapIndexed { index, input -> visitChild(input, index) }

        fun build(fn: (BuildContext) -> Dataframe): Dataframe

        fun buildStreaming(fn: (BuildContext) -> Dataframe): Dataframe
    }

    interface BuildContext {
        /**
         * Returns the Module.Builder used to construct this operation in this context.
         */
        fun builder(): Module.Builder

        /**
         * Lowers an expression into a global variable that can be retrieved using
         * the returned [Variable].
         */
        fun lowerAsGlobal(expression: Expr): Variable

        /**
         * Returns a PandasToRexTranslator that works in this build context.
         */
        fun rexTranslator(input: Dataframe): RexToPandasTranslator

        /**
         * Returns a PandasToRexTranslator that works in this build context
         * and is initialized with the given local refs.
         */
        fun rexTranslator(input: Dataframe, localRefs: List<Expr>): RexToPandasTranslator

        /**
         * Creates an assignment to the destination dataframe with the
         * result expression and returns the destination dataframe to
         * return from [emit].
         */
        fun returns(result: Expr): Dataframe

        /**
         * Returns configuration used for streaming.
         */
        fun streamingOptions(): StreamingOptions

        /**
         * Initialize the streaming IO loop.
         *
         * TODO(jsternberg): I haven't investigated the streaming code well enough, but
         * I suspect this shouldn't be here and the streaming code loop should be handled
         * by the exchanges or some other common interaction rather than having so much
         * of the logic in the code emitters.
         */
        fun initStreamingIoLoop(expr: Expr, rowType: RelDataType): Dataframe
    }
}
