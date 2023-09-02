package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.tools.RelBuilder
import org.immutables.value.Value

/**
 * Abstract rule that enables pushing projects into snowflake. Currently, only projects that simply
 * select a subset of columns are pushed into snowflake.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeProjectRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeProjectRule.Config>(config) {

    private fun isAllInputRefs(project: Project): Boolean {
        return project.projects.map { it is RexInputRef }.all { it }
    }

    /**
     * Take a projection that contains some operations from pure column accesses and generates equivalent
     * nodes that consist of two parts:
     * 1. A LogicalProjection equivalent to the original projection with only input Refs "re-indexed"
     * 2. A Snowflake Project contains only InputRefs that prune any unused columns
     *
     * This is done, so the LogicalProjection keeps any function calls / window functions
     * with the column selection is separated and pushed into Snowflake.
     */
    private fun splitComputeProjection(project: Project, snowflakeRel: SnowflakeRel, relBuilder: RelBuilder): RelNode {
        // First we will determine what columns are used and use that to build a SnowflakeProjection that just
        // prunes columns.
        val usedColumns = determineUsedInputColumns(project, snowflakeRel.getRowType().fieldCount)
        relBuilder.push(snowflakeRel)
        val fieldIndexRemap = mutableMapOf<Int, Int>()
        var nextIndex = 0
        // Remap the field indices in the current projection to the indices
        // after pruning columns.
        usedColumns.forEachIndexed {
                idx, keep ->
            if (keep) {
                fieldIndexRemap[idx] = nextIndex
                nextIndex++
            }
        }
        val fieldSelects = fieldIndexRemap.map { entry -> relBuilder.field(entry.key) }
        val fieldNames = fieldIndexRemap.map { entry -> snowflakeRel.getRowType().fieldNames[entry.key] }
        // Next we create the snowflake project from the logical project.
        val catalogTable = snowflakeRel.getCatalogTable()
        val snowflakeProject = SnowflakeProject.create(
            project.cluster,
            project.traitSet,
            snowflakeRel,
            fieldSelects,
            fieldNames,
            catalogTable,
        )
        relBuilder.push(snowflakeProject)
        // Now we need to update all the inputRef indices in the original project and generate
        // the final projection.
        val visitor = InputRefUpdate(fieldIndexRemap, relBuilder)
        val newProjects = project.projects.map { x -> x.accept(visitor) }
        val finalProject = BodoLogicalProject.create(
            snowflakeProject,
            project.hints,
            newProjects,
            // The type shouldn't change at all.
            project.getRowType(),
        )
        return finalProject
    }

    /**
     * RexShuttle implementation that only modifies a RexNode by updating
     * all RexInputRefs with the new column number. If a section of the RexNode
     * can avoid being changed it returns the original RexNode.
     *
     * All other changes are methods for ensuring the children are properly traversed
     * and all nodes are updated.
     */

    private class InputRefUpdate constructor(
        private val indexMap: Map<Int, Int>,
        private val builder: RelBuilder,
    ) :
        RexShuttle() {
        override fun visitInputRef(inputRef: RexInputRef): RexNode {
            val oldIndex = inputRef.index
            val newIndex = indexMap[oldIndex]!!
            if (oldIndex == newIndex) {
                // Avoid creating new nodes when possible
                return inputRef
            } else {
                return builder.field(newIndex)
            }
        }
    }

    /**
     * Creates a projection that consists of only pruning columns. This is basically trivial
     * to implement as we do not need to modify the projection at all.
     */
    private fun createOnlyPruningProjection(project: Project, snowflakeRel: SnowflakeRel): RelNode {
        val catalogTable = snowflakeRel.getCatalogTable()
        return SnowflakeProject.create(
            project.cluster,
            project.traitSet,
            snowflakeRel,
            project.projects,
            project.getRowType(),
            catalogTable,
        )
    }

    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }
        val (proj, snowflakeRel) = extractNodes(call)
        val newNode = if (isAllInputRefs(proj)) {
            createOnlyPruningProjection(proj, snowflakeRel)
        } else {
            splitComputeProjection(proj, snowflakeRel, call.builder())
        }
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Pair<Project, SnowflakeRel> {
        return when (call.rels.size) {
            // Inputs are:
            // Project ->
            //      CombineStreamsExchange ->
            //          SnowflakeToPandasConverter ->
            //             SnowflakeRel
            4 -> Pair(call.rel(0), call.rel(3))
            // Inputs are:
            // Project ->
            //   SnowflakeToPandasConverter ->
            //      SnowflakeRel
            else -> Pair(call.rel(0), call.rel(2))
        }
    }

    companion object {

        /**
         * Class to detect all columns that are used in a RexNode.
         */
        private class InputRefVisitor(private val mutableList: MutableList<Boolean>) : RexVisitorImpl<Unit>(true) {
            override fun visitInputRef(inputRef: RexInputRef) {
                mutableList[inputRef.index] = true
            }
        }

        /**
         * Determine the input columns that are directly referenced by a projection.
         */
        @JvmStatic
        private fun determineUsedInputColumns(project: Project, maxNumCols: Int): List<Boolean> {
            val mutableList = MutableList(maxNumCols) { false }
            val visitor = InputRefVisitor(mutableList)
            for (column in project.projects) {
                column.accept(visitor)
            }
            return mutableList.toList()
        }

        /**
         * A projection is pushable/has a pushable component if it doesn't use
         * every column from the input type but uses at least 1 column.
         *
         * TODO(njriasan): Is this check cheap enough? Should we be doing a cheaper check?
         */
        @JvmStatic
        fun isPushableProject(project: Project): Boolean {
            // Note: We use getRowType() because it may be initialized to null.
            val inputType = project.input.getRowType()
            val inputCols = inputType.fieldCount
            val usedColumns = determineUsedInputColumns(project, inputCols)
            return !usedColumns.all { x -> x } and usedColumns.any { x -> x }
        }
    }

    interface Config : RelRule.Config
}
