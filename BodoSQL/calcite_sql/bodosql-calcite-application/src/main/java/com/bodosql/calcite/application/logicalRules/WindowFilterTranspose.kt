package com.bodosql.calcite.application.logicalRules

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.util.ImmutableBitSet

/**
 * Set of utilities used to help with pushing non-window function filters past
 * operations containing window functions. Similar to the work in FilterAggregateTranspose,
 * it's safe to push filters that only operate on the columns of a partition by in a window
 * function as it will always eliminate an entire partition (thus preserving the output).
 *
 * However, unlike aggregate, window functions can be found in multiple locations and an operation
 * can contain multiple window functions, so additional utilities are needed to ensure an operation is safe
 * for all window functions.
 */
class WindowFilterTranspose(numCols: Int) : RexVisitorImpl<Unit>(true) {
    // Track which columns can be filtered. The visitor will modify this by computing
    // an intersection.
    private var filterableColumns = ImmutableBitSet.range(numCols)

    override fun visitOver(over: RexOver) {
        val window = over.window
        // Track if the window function is a form we cannot support.
        // These are situations where filtering could be possible, but
        // additional logic is required.
        val cannotFilter = window.partitionKeys.isEmpty() || window.partitionKeys.any { x -> x !is RexInputRef && x !is RexLiteral }
        if (cannotFilter) {
            filterableColumns = ImmutableBitSet.of()
            return
        }
        val windowUsedColumns = RelOptUtil.InputFinder.bits(window.partitionKeys, null)
        // The columns must be used by every window function, so compute the intersection
        filterableColumns = filterableColumns.intersect(windowUsedColumns)
    }

    fun build(): ImmutableBitSet {
        return filterableColumns
    }

    companion object {
        /**
         * Wrapper to call WindowFilterTranspose and determine which columns can be safely
         * filtered.
         */
        @JvmStatic
        fun getFilterableColumnIndices(
            nodes: List<RexNode>,
            numCols: Int,
        ): ImmutableBitSet {
            val visitor = WindowFilterTranspose(numCols)
            for (node in nodes) {
                node.accept(visitor)
            }
            return visitor.build()
        }

        /**
         * Given a Projection contains a RexOver determine which filters components can be safely pushed.
         * This implementation requires remapping the columns used by the filter, so it cannot reuse
         * the filter only implementation and cannot support RexOver in the filter.
         *
         * @return A pair of values. The left value is the pushable filters and the right value is the
         * non-pushable filters.
         */
        @JvmStatic
        fun findPushableFilterComponents(
            project: Project,
            filter: Filter,
        ): Pair<List<RexNode>, List<RexNode>> {
            if (!project.containsOver() || filter.containsOver()) {
                return Pair(listOf(), listOf(filter.condition))
            }
            // We will need to replace columns, so let's cache the results
            // in case the columns are complex. NULL means unevaluated.
            val canPushCache = arrayOfNulls<Boolean?>(project.projects.size)
            val overNodes = ArrayList<RexNode>()
            project.projects.forEachIndexed { idx, node ->
                val containsOver = RexOver.containsOver(node)
                if (containsOver) {
                    overNodes.add(node)
                    canPushCache[idx] = false
                }
            }
            // Generate which columns are legal within the projection.
            val pushableColumns: ImmutableBitSet = getFilterableColumnIndices(overNodes, project.getRowType().fieldCount)
            // Check each of the filters
            val conditionParts = RelOptUtil.conjunctions(filter.condition)
            val pushedFilters = ArrayList<RexNode>()
            val keptFilters = ArrayList<RexNode>()
            for (condition in conditionParts) {
                var canPush = true
                // Find the columns used in the condition
                val filterColumns = RelOptUtil.InputFinder.bits(condition)
                // Remap the bitset to the original projection.
                for (idx in filterColumns) {
                    // We haven't computed the result, check it.
                    if (canPushCache[idx] == null) {
                        val projectColumns = RelOptUtil.InputFinder.bits(project.projects[idx])
                        val newColumns = projectColumns.except(pushableColumns)
                        val columnCanPush = newColumns.isEmpty
                        // Update the cache.
                        canPushCache[idx] = columnCanPush
                        if (!columnCanPush) {
                            canPush = false
                            break
                        }
                    } else if (canPushCache[idx] == false) {
                        canPush = false
                        break
                    }
                }
                if (canPush) {
                    pushedFilters.add(condition)
                } else {
                    keptFilters.add(condition)
                }
            }
            return Pair(pushedFilters, keptFilters)
        }

        /**
         * Given a Filter contains a RexOver determine which filters components can be safely pushed.
         *
         * @return A pair of values. The left value is the pushable filters and the right value is the
         * non-pushable filters.
         */
        @JvmStatic
        fun findPushableFilterComponents(filter: Filter): Pair<List<RexNode>, List<RexNode>> {
            val conditionParts = RelOptUtil.conjunctions(filter.condition)
            val filterOverNodes = ArrayList<RexNode>()
            val pushedFilterCandidates = ArrayList<RexNode>()
            for (node in conditionParts) {
                val containsOver = RexOver.containsOver(node)
                if (containsOver) {
                    filterOverNodes.add(node)
                } else {
                    pushedFilterCandidates.add(node)
                }
            }
            // If there is no RexOver all nodes can be pushed. This is defensive programming as for
            // performance reasons the caller should verify there is at least 1 RexOver.
            if (filterOverNodes.isEmpty()) {
                return Pair(conditionParts, listOf())
            }
            val pushableColumns: ImmutableBitSet = getFilterableColumnIndices(filterOverNodes, filter.getRowType().fieldCount)
            // No pushing is possible
            if (pushableColumns.isEmpty) {
                return Pair(listOf(), conditionParts)
            }
            // Check each filter's used columns. A filter can only be pushed if it only uses columns in the partition by intersection.
            val pushedFilters = ArrayList<RexNode>()
            val keptFilters = ArrayList<RexNode>()
            keptFilters.addAll(filterOverNodes)
            for (condition in pushedFilterCandidates) {
                val filterColumns = RelOptUtil.InputFinder.bits(condition)
                val newColumns = filterColumns.except(pushableColumns)
                // Verify there are no columns remaining from an except
                if (newColumns.isEmpty) {
                    pushedFilters.add(condition)
                } else {
                    keptFilters.add(condition)
                }
            }
            return Pair(pushedFilters, keptFilters)
        }
    }
}
