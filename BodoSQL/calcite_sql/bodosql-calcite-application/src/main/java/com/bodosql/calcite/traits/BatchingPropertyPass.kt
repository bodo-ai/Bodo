package com.bodosql.calcite.traits

import com.bodosql.calcite.adapter.bodo.BodoPhysicalProject
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.RelNode
import org.apache.calcite.tools.RelBuilder

/**
 * Append Batching property information to every Physical Node. If the input property doesn't
 * meet expectations then a corresponding Exchange Node will be inserted.
 *
 * These physical traits are not added in the volcano model because considering the exchange
 * operators requires the duplication of rules and while the exchange operators can be used
 * to determine relative cost, it generally leads to regressions as we typically don't want
 * to make decisions to avoid additional streaming.
 *
 * Note: This isn't a RelShuttle because the BodoPhysicalRel is a simpler interface given
 * the implementation requirements.
 */
class BatchingPropertyPass(private val builder: RelBuilder) {
    /**
     * Generate the code to insert a SeparateStreamExchange or
     * CombineStreamsExchange. This includes a special case where
     * if the parent is a BodoProjection we may also insert an
     * additional projection to prune any columns before the
     * exchange operator.
     */
    private fun generateExchangeInput(
        input: RelNode,
        expectedInputProperty: BatchingProperty,
    ): RelNode {
        return if (expectedInputProperty == BatchingProperty.STREAMING) {
            SeparateStreamExchange(input.cluster, input.traitSet.replace(expectedInputProperty), input)
        } else {
            CombineStreamsExchange(input.cluster, input.traitSet.replace(expectedInputProperty), input)
        }
    }

    /**
     * BodoPhysicalProject is special because we may need to insert another Projection
     * before the exchange operator to prune columns.
     */
    fun visit(node: BodoPhysicalProject): RelNode {
        val visitedInput = visit(node.input)
        val origInputRowType = visitedInput.getRowType()
        val outputRowType = node.getRowType()
        val actualInputBatchingProperty = getOutputBatchingProperty(visitedInput)
        val expectedInputProperty = node.expectedInputBatchingProperty(actualInputBatchingProperty)
        val newTraitSet = node.traitSet.replace(node.expectedOutputBatchingProperty(actualInputBatchingProperty))
        return if (!actualInputBatchingProperty.satisfies(expectedInputProperty)) {
            // First we need to check if we need to prune columns
            val usedColumns = RelOptUtil.InputFinder.bits(node.projects, null)
            // Note we cannot prune all columns, so don't insert a projection for 0 used columns.
            val pruneColumns = usedColumns.cardinality() != 0 && usedColumns.cardinality() != origInputRowType.fieldCount
            val prunedInput =
                if (pruneColumns) {
                    builder.push(visitedInput)
                    // The builder doesn't extend the batching property yet. Instead
                    // we directly create a Bodo project to avoid updating the
                    // inputs(TODO: Remove).
                    val nodes = usedColumns.map { i -> builder.field(i) }
                    val fieldNames = usedColumns.map { i -> origInputRowType.fieldNames[i] }
                    val proj = BodoPhysicalProject.create(visitedInput, nodes, fieldNames)
                    // Copy to update the traitset
                    proj.copy(proj.traitSet.replace(actualInputBatchingProperty), proj.input, proj.projects, proj.getRowType())
                } else {
                    visitedInput
                }
            // Insert the exchange
            val exchangeInput = generateExchangeInput(prunedInput, expectedInputProperty)
            // Generate an updated projection to remap columns if we pruned columns
            if (pruneColumns) {
                // Create the adjustments.
                val adjustments = IntArray(origInputRowType.fieldCount) { _ -> -1 }
                var columnCount = 0
                for (idx in usedColumns) {
                    adjustments[idx] = columnCount - idx
                    columnCount++
                }
                // Create the inputRef update
                val visitor = RelOptUtil.RexInputConverter(builder.rexBuilder, origInputRowType.fieldList, adjustments)
                val newProjections = visitor.visitList(node.projects)
                node.copy(newTraitSet, exchangeInput, newProjections, outputRowType)
            } else {
                // Just copy and update the traitSet
                node.copy(newTraitSet, exchangeInput, node.projects, outputRowType)
            }
        } else {
            // Just copy and update the traitSet
            node.copy(newTraitSet, visitedInput, node.projects, outputRowType)
        }
    }

    fun visit(node: BodoPhysicalRel): RelNode {
        val newInputs: MutableList<RelNode> = ArrayList()
        for (input in node.inputs) {
            val visitedInput = visit(input)
            val actualInputBatchingProperty = getOutputBatchingProperty(visitedInput)
            // Note: We allow None to match to avoid special handling for other conventions
            // (e.g. Snowflake) where streaming doesn't exist.
            val expectedInputProperty = node.expectedInputBatchingProperty(actualInputBatchingProperty)
            val newInput =
                if (!actualInputBatchingProperty.satisfies(expectedInputProperty)) {
                    generateExchangeInput(visitedInput, expectedInputProperty)
                } else {
                    visitedInput
                }
            newInputs.add(newInput)
        }
        val inputProperty =
            if (newInputs.size > 0) {
                getOutputBatchingProperty(newInputs[0])
            } else {
                // If there are no inputs the input batching property shouldn't matter.
                BatchingProperty.NONE
            }
        val newTraitSet = node.traitSet.replace(node.expectedOutputBatchingProperty(inputProperty))
        return node.copy(newTraitSet, newInputs)
    }

    /**
     * Workaround for dispatching since we have a simple use case.
     */
    fun visit(node: RelNode): RelNode {
        return when (node) {
            is BodoPhysicalProject -> {
                visit(node)
            }
            is BodoPhysicalRel -> {
                return visit(node)
            }
            else -> {
                return node
            }
        }
    }

    companion object {
        @JvmStatic
        fun applyBatchingInfo(
            rel: RelNode,
            builder: RelBuilder,
        ): RelNode {
            return if (rel.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == null) {
                // Ignore for non-streaming
                rel
            } else {
                val requiredBatchingProperty = BatchingProperty.SINGLE_BATCH
                val visitor = BatchingPropertyPass(builder)
                val output = visitor.visit(rel)
                val finalBatchingProperty = getOutputBatchingProperty(output)
                // The final result must be non-streaming
                if (finalBatchingProperty == requiredBatchingProperty) {
                    output
                } else {
                    CombineStreamsExchange(output.cluster, output.traitSet.replace(requiredBatchingProperty), output)
                }
            }
        }

        @JvmStatic
        private fun getOutputBatchingProperty(node: RelNode): BatchingProperty {
            return node.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) ?: BatchingProperty.NONE
        }
    }
}
