package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalAggregate
import com.bodosql.calcite.adapter.bodo.BodoPhysicalFilter
import com.bodosql.calcite.adapter.bodo.BodoPhysicalFlatten
import com.bodosql.calcite.adapter.bodo.BodoPhysicalIntersect
import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinRowNumberFilter
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinus
import com.bodosql.calcite.adapter.bodo.BodoPhysicalProject
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRuntimeJoinFilter
import com.bodosql.calcite.adapter.bodo.BodoPhysicalSort
import com.bodosql.calcite.adapter.bodo.BodoPhysicalUnion
import com.bodosql.calcite.adapter.common.LimitUtils
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.logicalRules.WindowFilterTranspose
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.core.SetOp
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.tools.Program
import org.apache.calcite.util.ImmutableBitSet
import java.util.function.Predicate

object RuntimeJoinFilterProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        return if (RelationalAlgebraGenerator.enableRuntimeJoinFilters) {
            val shuttle = RuntimeJoinFilterShuttle()
            rel.accept(shuttle)
        } else {
            rel
        }
    }

    private class RuntimeJoinFilterShuttle() : RelShuttleImpl() {
        private var joinFilterID: Int = 0
        private var liveJoins: List<LiveJoinInfo> = ArrayList()

        override fun visit(rel: RelNode): RelNode {
            return when (rel) {
                is BodoPhysicalProject -> {
                    visit(rel)
                }

                is BodoPhysicalFilter -> {
                    visit(rel)
                }

                is BodoPhysicalMinRowNumberFilter -> {
                    visit(rel)
                }

                is BodoPhysicalSort -> {
                    // Sort can produce the filter only if
                    // we have an order by + limit.
                    visit(rel)
                }

                is BodoPhysicalAggregate -> {
                    visit(rel)
                }

                is BodoPhysicalJoin -> {
                    visit(rel)
                }

                is BodoPhysicalUnion, is BodoPhysicalIntersect, is BodoPhysicalMinus -> {
                    visit(rel as SetOp)
                }

                is BodoPhysicalFlatten -> {
                    visit(rel)
                }

                else -> {
                    // Unknown Node: We must pop any remaining filters.
                    val oldLiveJoins = liveJoins
                    liveJoins = mutableListOf()
                    applyFilters(super.visit(rel), oldLiveJoins)
                }
            }
        }

        /**
         * Split the filters currently found inside the liveJoins into two components.
         * The first component is the filters which can be pushed deeper into the plan
         * via the inputs of the current node. The second component is the filters which
         * must be applied at the current node.
         * @param canPushPredicate The boolean predicate which determines if a filter can be pushed.
         * @param columnTransformFunction The function which transforms a column to a new column.
         * @return A pair of lists. The first list contains the filters which can be pushed, the second
         * list contains the filters which must be applied at the current node.
         */
        private fun splitFilterSections(
            canPushPredicate: Predicate<Int>,
            columnTransformFunction: (Int) -> Int,
        ): Pair<List<LiveJoinInfo>, List<LiveJoinInfo>> {
            val pushLiveJoins: MutableList<LiveJoinInfo> = mutableListOf()
            val outputLiveJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
            for (joinInfo in liveJoins) {
                val pushKeys: MutableList<Int> = mutableListOf()
                val pushIsFirstLocation: MutableList<Boolean> = mutableListOf()
                val outputKeys: MutableList<Int> = mutableListOf()
                val outputIsFirstLocation: MutableList<Boolean> = mutableListOf()
                for (column in joinInfo.remainingColumns) {
                    outputKeys.add(column)
                    if (column == -1) {
                        // Any column with -1 must propagate -1. These represent
                        // columns that were evaluated higher in the plan.
                        pushKeys.add(-1)
                        pushIsFirstLocation.add(false)
                        outputIsFirstLocation.add(false)
                    } else if (canPushPredicate.test(column)) {
                        // This is a column that can be pushed into the input.
                        pushKeys.add(columnTransformFunction(column))
                        pushIsFirstLocation.add(true)
                        outputIsFirstLocation.add(false)
                    } else {
                        // This is a column that must be evaluated here.
                        pushKeys.add(-1)
                        pushIsFirstLocation.add(false)
                        outputIsFirstLocation.add(true)
                    }
                }
                if (pushIsFirstLocation.any { it }) {
                    pushLiveJoins.add(LiveJoinInfo(joinInfo.joinFilterKey, pushKeys, pushIsFirstLocation))
                }
                if (outputIsFirstLocation.any { it }) {
                    outputLiveJoinInfo.add(LiveJoinInfo(joinInfo.joinFilterKey, outputKeys, outputIsFirstLocation))
                }
            }

            return Pair(pushLiveJoins.toList(), outputLiveJoinInfo.toList())
        }

        /**
         * Perform the processing of a RelNode with a single input. This function
         * modifies the liveJoins, then recursively applies the visit to update the input,
         * and finally generates any filters.
         * @param rel The RelNode to process.
         * @param pushLiveJoinInfo The filters which can be pushed into the input.
         * @param outputLiveJoinInfo The filters which must be applied at the current node.
         * @return The new RelNode with the filters applied.
         */
        private fun processSingleRel(
            rel: RelNode,
            pushLiveJoinInfo: List<LiveJoinInfo>,
            outputLiveJoinInfo: List<LiveJoinInfo>,
        ): RelNode {
            liveJoins = pushLiveJoinInfo
            val newNode = super.visit(rel)
            return applyFilters(newNode, outputLiveJoinInfo)
        }

        private fun visit(project: Project): RelNode {
            // If the project contains an OVER clause we can only push filters
            // that are shared by all partition by columns.
            val numCols = project.getRowType().fieldCount
            var pushableColumns =
                if (project.containsOver()) {
                    WindowFilterTranspose.getFilterableColumnIndices(project.projects, numCols)
                } else {
                    ImmutableBitSet.range(numCols)
                }

            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    canPushPredicate = { pushableColumns.get(it) && (project.projects[it] is RexInputRef) },
                    columnTransformFunction = { (project.projects[it] as RexInputRef).index },
                )
            return processSingleRel(project, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(filter: BodoPhysicalFilter): RelNode {
            // If the filter contains an OVER clause we can only push filters
            // that are shared by all partition by columns.
            val numCols = filter.rowType.fieldCount
            var pushableColumns =
                if (filter.containsOver()) {
                    WindowFilterTranspose.getFilterableColumnIndices(listOf(filter.condition), numCols)
                } else {
                    ImmutableBitSet.range(numCols)
                }
            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    canPushPredicate = { pushableColumns.get(it) },
                    columnTransformFunction = { it },
                )
            return processSingleRel(filter, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(node: BodoPhysicalMinRowNumberFilter): RelNode {
            val keptInputs = node.inputsToKeep.toList()
            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    // Only push columns which are in the partition set.
                    canPushPredicate = { node.partitionColSet.contains(it) },
                    // Remap any columns in the filter based upon kept inputs.
                    columnTransformFunction = { keptInputs[it] },
                )
            return processSingleRel(node, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(sort: BodoPhysicalSort): RelNode {
            val canPush = !LimitUtils.isOrderedLimit(sort)
            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    canPushPredicate = { canPush },
                    columnTransformFunction = { it },
                )
            return processSingleRel(sort, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(aggregate: Aggregate): RelNode {
            // Split the join info into those which must be processed now
            // and those which can be processed later.
            val groupByKeys = aggregate.groupSet.toList()
            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    // Only push columns which are in the group by set.
                    canPushPredicate = { it < aggregate.groupCount },
                    columnTransformFunction = { groupByKeys[it] },
                )
            return processSingleRel(aggregate, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(join: Join): RelNode {
            val info = join.analyzeCondition()
            // Note: You must use getRowType() here due to a caching issue.
            val numLeftColumns = join.left.rowType.fieldCount
            // Split the join info into those which are processed on
            // the left input and those which are processed on the right input.
            // equi-join keys can be processed on both inputs.
            val leftJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
            val rightJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
            // If the filters are split across both sides and neither side
            // gets all the columns, we may need to generate a filter after
            // this join to ensure the bloom filter runs.
            val outputJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
            for (joinInfo in liveJoins) {
                val leftKeys: MutableList<Int> = mutableListOf()
                val leftIsFirstLocation: MutableList<Boolean> = mutableListOf()
                val rightKeys: MutableList<Int> = mutableListOf()
                val rightIsFirstLocation: MutableList<Boolean> = mutableListOf()
                for (column in joinInfo.remainingColumns) {
                    if (column == -1) {
                        // Any column with -1 must propagate -1. These represent
                        // columns that were evaluated higher in the plan.
                        leftKeys.add(-1)
                        leftIsFirstLocation.add(false)
                        rightKeys.add(-1)
                        rightIsFirstLocation.add(false)
                    } else if (column < numLeftColumns) {
                        // Column is on the left side.
                        leftKeys.add(column)
                        leftIsFirstLocation.add(true)
                        // If this is also a left key then we can push
                        // it into the right side as well.
                        val keyIndex = info.leftKeys.indexOf(column)
                        if (keyIndex != -1) {
                            // Note: info.rightKeys doesn't need remapping
                            // It is already in terms of the right side.
                            val rightKey = info.rightKeys[keyIndex]
                            rightKeys.add(rightKey)
                            rightIsFirstLocation.add(true)
                        } else {
                            rightKeys.add(-1)
                            rightIsFirstLocation.add(false)
                        }
                    } else {
                        // Column is on the right side.
                        rightKeys.add(column - numLeftColumns)
                        rightIsFirstLocation.add(true)
                        // If this is also a right key then we can push
                        // it into the left side as well.
                        val keyIndex = info.rightKeys.indexOf(column)
                        if (keyIndex != -1) {
                            val leftKey = info.leftKeys[keyIndex]
                            leftKeys.add(leftKey)
                            leftIsFirstLocation.add(true)
                        } else {
                            leftKeys.add(-1)
                            leftIsFirstLocation.add(false)
                        }
                    }
                }
                val keepLeft = leftIsFirstLocation.any { it }
                val keepRight = rightIsFirstLocation.any { it }
                if (keepLeft) {
                    leftJoinInfo.add(LiveJoinInfo(joinInfo.joinFilterKey, leftKeys, leftIsFirstLocation))
                }
                if (keepRight) {
                    rightJoinInfo.add(LiveJoinInfo(joinInfo.joinFilterKey, rightKeys, rightIsFirstLocation))
                }
                // Add the filter to be generated now if:
                // 1. The join is split across both sides.
                // 2. The join has all columns.
                // 3. Neither side has all columns.
                //
                // We don't need to generate the filter for the partial side
                // because all shared columns must be key columns, so we already
                // check equality.
                if (keepLeft && keepRight) {
                    val hasAllColumns = joinInfo.isFirstLocation.all { it }
                    if (hasAllColumns) {
                        val allLeft = leftIsFirstLocation.all { it }
                        val allRight = rightIsFirstLocation.all { it }
                        if (!allLeft && !allRight) {
                            outputJoinInfo.add(joinInfo)
                        }
                    }
                }
            }
            // If we have a RIGHT or Inner join we can generate
            // a runtime join filter.
            val filterKey =
                if (!join.joinType.generatesNullsOnRight()) {
                    val columns = info.leftKeys
                    if (columns.isEmpty()) {
                        -1
                    } else {
                        val filterKey = joinFilterID++
                        // Add a new join to the left side.
                        leftJoinInfo.add(LiveJoinInfo(filterKey, columns, List(columns.size) { true }))
                        filterKey
                    }
                } else {
                    -1
                }
            liveJoins = leftJoinInfo
            val leftInput = join.left.accept(this)
            liveJoins = rightJoinInfo
            val rightInput = join.right.accept(this)
            val newJoin = BodoPhysicalJoin.create(leftInput, rightInput, join.condition, join.joinType, joinFilterID = filterKey)
            return applyFilters(newJoin, outputJoinInfo)
        }

        /**
         * Visit a SetOp. Since a SetOp can't produce a new column,
         * and it won't produce an incorrect output, we can duplicate
         * the filters for each input.
         */
        private fun visit(node: SetOp): RelNode {
            val liveJoinsCopy = liveJoins
            val newInputs =
                node.inputs.map {
                    liveJoins = liveJoinsCopy
                    it.accept(this)
                }
            return node.copy(node.traitSet, newInputs, node.all)
        }

        private fun visit(flatten: BodoPhysicalFlatten): RelNode {
            val repeatColumns = flatten.repeatColumns.toList()
            val (pushLiveJoinInfo, outputLiveJoinInfo) =
                splitFilterSections(
                    // Only push columns which are copied from the input, not computed.
                    canPushPredicate = { it < repeatColumns.size },
                    columnTransformFunction = { repeatColumns[it] },
                )
            return processSingleRel(flatten, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun applyFilters(
            rel: RelNode,
            liveJoins: List<LiveJoinInfo>,
        ): RelNode {
            return if (liveJoins.isEmpty()) {
                rel
            } else {
                val filterKeys = liveJoins.map { it.joinFilterKey }
                val filterColumns = liveJoins.map { it.remainingColumns }
                val areFirstLocations = liveJoins.map { it.isFirstLocation }
                BodoPhysicalRuntimeJoinFilter.create(
                    rel,
                    filterKeys,
                    filterColumns,
                    areFirstLocations,
                )
            }
        }
    }

    private data class LiveJoinInfo(val joinFilterKey: Int, val remainingColumns: List<Int>, val isFirstLocation: List<Boolean>)
}
