package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.adapter.common.TimerSupportedRel
import com.bodosql.calcite.application.utils.RelationalOperatorCache
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.OperatorID
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.type.BodoTZInfo

interface BodoPhysicalRel : TimerSupportedRel {
    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("BODO", BodoPhysicalRel::class.java)
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    fun emit(implementor: Implementor): BodoEngineTable

    /**
     * Determine if an operator is streaming.
     */
    fun isStreaming() = batchingProperty() == BatchingProperty.STREAMING

    /**
     * Get the batching property.
     */
    fun batchingProperty(): BatchingProperty = traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) ?: BatchingProperty.NONE

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    )

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

    override fun nodeString(): String {
        return RelOptUtil.toString(this)
    }

    interface Implementor {
        fun build(fn: (BuildContext) -> BodoEngineTable): BodoEngineTable

        fun buildStreaming(operatorEmission: OperatorEmission): BodoEngineTable?

        /**
         * getter for relationalOperatorCache
         */
        fun getRelationalOperatorCache(): RelationalOperatorCache
    }

    interface BuildContext {
        fun operatorID(): OperatorID

        /**
         * Visits a child relational expression and returns the result as a BodoEngineTable.
         */
        fun visitChild(
            input: RelNode,
            ordinal: Int,
        ): BodoEngineTable

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
         * Lowers an expression into a global MetaType variable that can be retrieved using
         * the returned [Variable].
         */
        fun lowerAsMetaType(expression: Expr): Variable

        /**
         * Returns a RexToBodoTranslator that works in this build context.
         */
        fun rexTranslator(input: BodoEngineTable): RexToBodoTranslator

        /**
         * Returns a RexToBodoTranslator that works in this build context
         * and is initialized with the given local refs.
         */
        fun rexTranslator(
            input: BodoEngineTable,
            localRefs: List<Expr>,
        ): RexToBodoTranslator

        /**
         * Returns a RexToBodoTranslator that works in this build context and
         * will wrap the final result of any scalars in arrays the same size as the input.
         */
        fun arrayRexTranslator(input: BodoEngineTable): ArrayRexToBodoTranslator

        /**
         * Returns a RexToBodoTranslator that works in this build context when
         * only operating on scalars.
         */
        fun scalarRexTranslator(): ScalarRexToBodoTranslator

        /**
         * Returns a RexToBodoTranslator that works in this a streaming context.
         */
        fun streamingRexTranslator(
            input: BodoEngineTable,
            localRefs: List<Expr>,
            stateVar: StateVariable,
        ): StreamingRexToBodoTranslator

        /**
         * Creates an assignment to the destination dataframe with the
         * result expression and returns the destination dataframe to
         * return from [emit].
         */
        fun returns(result: Expr): BodoEngineTable

        /**
         * Converts a DataFrame into a Table using an input rel node to infer
         * the number of columns in the DataFrame.
         * @param df The DataFrame that is being converted into a Table.
         * @param node The rel node whose output schema is used to infer
         * the number of columns in the DataFrame.
         * @return The BodoEngineTable that the DataFrame has been converted into.
         */
        fun convertDfToTable(
            df: Variable?,
            node: RelNode,
        ): BodoEngineTable

        /**
         * An overload of convertDfToTable that works the same but accepts
         * a row type instead of a node.
         */
        fun convertDfToTable(
            df: Variable?,
            rowType: RelDataType,
        ): BodoEngineTable

        /**
         * An overload of convertDfToTable that works the same but accepts
         * the number of fields instead of a node or type.
         */
        fun convertDfToTable(
            df: Variable?,
            numCols: Int,
        ): BodoEngineTable

        /**
         * Converts a Table into a DataFrame.
         * @param table The table that is being converted into a DataFrame.
         * @return The variable used to store the new DataFrame.
         */
        fun convertTableToDf(table: BodoEngineTable?): Variable

        /**
         * Returns configuration used for streaming.
         */
        fun streamingOptions(): StreamingOptions

        /**
         * Returns the default timezone
         */
        fun getDefaultTZ(): BodoTZInfo

        /**
         * Start a new active pipeline. This is used for streaming.
         */
        fun startPipeline()

        /**
         * End the active pipeline. This is used for streaming.
         */
        fun endPipeline()
    }
}
