package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.common.LimitUtils
import com.bodosql.calcite.adapter.pandas.PandasAggregate
import com.bodosql.calcite.adapter.pandas.PandasFilter
import com.bodosql.calcite.adapter.pandas.PandasFlatten
import com.bodosql.calcite.adapter.pandas.PandasIntersect
import com.bodosql.calcite.adapter.pandas.PandasJoin
import com.bodosql.calcite.adapter.pandas.PandasMinRowNumberFilter
import com.bodosql.calcite.adapter.pandas.PandasMinus
import com.bodosql.calcite.adapter.pandas.PandasProject
import com.bodosql.calcite.adapter.pandas.PandasRuntimeJoinFilter
import com.bodosql.calcite.adapter.pandas.PandasSort
import com.bodosql.calcite.adapter.pandas.PandasUnion
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
                is PandasProject -> {
                    visit(rel)
                }

                is PandasFilter -> {
                    visit(rel)
                }

                is PandasMinRowNumberFilter -> {
                    visit(rel)
                }

                is PandasSort -> {
                    // Sort can produce the filter only if
                    // we have an order by + limit.
                    visit(rel)
                }

                is PandasAggregate -> {
                    visit(rel)
                }

                is PandasJoin -> {
                    visit(rel)
                }

                is PandasUnion, is PandasIntersect, is PandasMinus -> {
                    visit(rel as SetOp)
                }

                is PandasFlatten -> {
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

        private fun visit(filter: PandasFilter): RelNode {
            // If the filter contains an OVER clause we can only push filters
            // that are shared by all partition by columns.
            val numCols = filter.getRowType().fieldCount
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

        private fun visit(node: PandasMinRowNumberFilter): RelNode {
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

        private fun visit(sort: PandasSort): RelNode {
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
            val numLeftColumns = join.left.getRowType().fieldCount
            // Split the join info into those which are processed on
            // the left input and those which are processed on the right input.
            // equi-join keys can be processed on both inputs.
            val leftJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
            val rightJoinInfo: MutableList<LiveJoinInfo> = mutableListOf()
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
                if (leftIsFirstLocation.any { it }) {
                    leftJoinInfo.add(LiveJoinInfo(joinInfo.joinFilterKey, leftKeys, leftIsFirstLocation))
                }
                if (rightIsFirstLocation.any { it }) {
                    rightJoinInfo.add(LiveJoinInfo(joinInfo.joinFilterKey, rightKeys, rightIsFirstLocation))
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
            return PandasJoin.create(leftInput, rightInput, join.condition, join.joinType, joinFilterID = filterKey)
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

        private fun visit(flatten: PandasFlatten): RelNode {
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
            var returnNode = rel
            for (liveJoin in liveJoins) {
                returnNode =
                    PandasRuntimeJoinFilter.create(
                        returnNode, liveJoin.joinFilterKey, liveJoin.remainingColumns, liveJoin.isFirstLocation,
                    )
            }
            return returnNode
        }
    }

    private data class LiveJoinInfo(val joinFilterKey: Int, val remainingColumns: List<Int>, val isFirstLocation: List<Boolean>)
}
