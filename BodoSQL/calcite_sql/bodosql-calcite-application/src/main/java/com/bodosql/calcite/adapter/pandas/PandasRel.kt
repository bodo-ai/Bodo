package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.common.TimerSupportedRel
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptUtil

interface PandasRel : TimerSupportedRel {
    /**
     * Emits the code necessary for implementing this relational operator.
     * This is needed because the current implementation just uses the
     * PandasToBodoPhysicalConverter as a barrier and generates all the
     * code in its inner nodes.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable

    /**
     * What is the expected batching property for the output data given the property
     * of the inputs. Most implementation will ignore the argument but some nodes may allow
     * matching it under some circumstances.
     */
    fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty =
        ExpectedBatchingProperty.alwaysSingleBatchProperty()

    /**
     * The expected batching property for the given input node's property.
     * Most implementation will ignore the argument but some nodes may allow
     * matching it under some circumstances.
     */
    fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty =
        expectedOutputBatchingProperty(inputBatchingProperty)

    /**
     * Get the batching property.
     */
    private fun batchingProperty(): BatchingProperty = traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) ?: BatchingProperty.NONE

    /**
     * Determine if an operator is streaming.
     */
    fun isStreaming() = batchingProperty() == BatchingProperty.STREAMING

    override fun nodeString(): String {
        return RelOptUtil.toString(this)
    }

    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("Pandas", PandasRel::class.java)
    }
}
