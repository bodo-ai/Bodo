package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalAggregate
import com.bodosql.calcite.adapter.bodo.BodoPhysicalCachedSubPlan
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
import com.bodosql.calcite.adapter.bodo.BodoPhysicalWindow
import com.bodosql.calcite.adapter.common.LimitUtils
import com.bodosql.calcite.adapter.iceberg.IcebergRel
import com.bodosql.calcite.adapter.iceberg.IcebergRuntimeJoinFilter
import com.bodosql.calcite.adapter.iceberg.IcebergToBodoPhysicalConverter
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeRuntimeJoinFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeToBodoPhysicalConverter
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.logicalRules.WindowFilterTranspose
import com.bodosql.calcite.rel.core.CachedPlanInfo
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.rel.core.cachePlanContainers.CacheNodeTopologicalSortVisitor
import com.bodosql.calcite.rel.core.cachePlanContainers.CachedResultVisitor
import com.google.common.annotations.VisibleForTesting
import org.apache.calcite.plan.BodoRelOptCluster
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.SetOp
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.tools.Program
import org.apache.calcite.util.ImmutableBitSet
import java.util.function.Predicate

/**
 * Program that generates runtime join filters for a plan.
 * The key concepts are explained in this document:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1734901763/Runtime+Join+Filter+Program
 */
object RuntimeJoinFilterProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        return if (RelationalAlgebraGenerator.enableRuntimeJoinFilters) {
            val cluster = rel.cluster
            if (cluster !is BodoRelOptCluster) {
                throw InternalError("Cluster must be a BodoRelOptCluster")
            }
            val filterShuttle = RuntimeJoinFilterShuttle(cluster, true)
            val result = rel.accept(filterShuttle)
            filterShuttle.visitCacheNodes()
            val cacheReplaceShuttle =
                JoinFilterCacheReplace(
                    filterShuttle.keptCacheConsumerUpdates,
                    filterShuttle.inlinedCacheConsumerUpdates,
                )
            result.accept(cacheReplaceShuttle)
        } else {
            rel
        }
    }

    /**
     * Visitor for generating runtime join filters. This code is reused by caching first pass without generating filters,
     * so we also add an emitFilters flag to control whether we generate filters or not. This doesn't impact correctness
     * but reduces the number of required copies.
     */
    @VisibleForTesting
    internal open class RuntimeJoinFilterShuttle(
        private val cluster: BodoRelOptCluster,
        private val emitFilters: Boolean,
        protected var liveJoins: JoinFilterProgramState = JoinFilterProgramState(),
    ) : RelShuttleImpl() {
        // Keep track of nodes which should be a source for a new cache node. This should
        // only be set/modified by the caching passes.
        private var createdCacheIDs = mutableMapOf<Int, BodoPhysicalCachedSubPlan?>()

        // Visitor for tracking cache nodes. Note we track the exact
        // RelNode ID of each consumer, so we can identify which node to
        // replace.
        private val cacheVisitor = CacheNodeTopologicalSortVisitor<Pair<Int, JoinFilterProgramState>>()

        // Keep maps for tracking the impact of each cache consumer
        internal val keptCacheConsumerUpdates = mutableMapOf<Int, RelNode>()
        internal val inlinedCacheConsumerUpdates = mutableMapOf<Int, RelNode>()

        fun visitCacheNodes() {
            while (cacheVisitor.hasReadyNode()) {
                val (plan, consumerInfo) = cacheVisitor.pop()
                val filters = consumerInfo.map { it.second }
                // Compute intersection of all join filters.
                val intersectFilter = filters.reduce { acc, joinFilters -> acc.intersection(joinFilters) }
                // We 0 wasted work if any consumer gets all its filters pushed.
                val keepCache = filters.any { it == intersectFilter }
                val (unionFilter, tryInline) =
                    if (!keepCache) {
                        try {
                            // Note: In some cases it's not possible to union together the results of RTJFs. For example,
                            // consider the following example case.
                            //                            UNION
                            //                          /       \
                            //              Project (0, 1)        Project (1, 0)
                            //                  |                       |
                            //                  C                       C
                            //
                            // Right now we don't have any way to expression that our join requires column 0 or 1
                            // for any key. As a result, when we encounter this case we will not inline the cache node.
                            val unionFilter = filters.reduce { acc, joinFilters -> acc.union(joinFilters) }
                            unionFilter to true
                        } catch (e: IllegalArgumentException) {
                            // If we can't union the filters, then we can't inline the cache node.
                            null to false
                        }
                    } else {
                        null to false
                    }
                if (!tryInline) {
                    // Just visit the cache node once.
                    liveJoins = intersectFilter
                    plan.cachedPlan.plan = visit(plan.cachedPlan.plan)
                    // Mark how to inline the nodes.
                    for ((consumerID, totalFilters) in consumerInfo) {
                        val additionalFilters = totalFilters.difference(intersectFilter)
                        val idBody = applyFilters(plan.copy(plan.traitSet, plan.inputs), additionalFilters)
                        keptCacheConsumerUpdates[consumerID] = idBody
                    }
                } else {
                    // There is some wasted work, so we opt to inline the cache nodes. However, this doesn't mean
                    // that we need to inline all the nodes. We only need to inline until all RTJFs would have
                    // been generated. If this happens, and we reach a node that is valid to cache, then we can cache
                    // subsections of the original cached plan.
                    //
                    // TODO: Update the number of consumers for the cache node when we check if any consumers have
                    // completely identical filters.
                    val inlineVisitor = InlineCacheVisitor(cluster, unionFilter!!, consumerInfo.size)
                    inlineVisitor.visit(plan.cachedPlan.plan)
                    val oldCacheIDs = createdCacheIDs
                    this.createdCacheIDs = inlineVisitor.getNewCacheNodes().associateWith { null }.toMutableMap()
                    // Visit the node once per inlined consumer.
                    for ((consumerID, totalFilters) in consumerInfo) {
                        val cacheRoot = plan.cachedPlan.plan
                        liveJoins = totalFilters
                        val idBody = visit(cacheRoot.copy(cacheRoot.traitSet, cacheRoot.inputs))
                        inlinedCacheConsumerUpdates[consumerID] = idBody
                    }
                    this.createdCacheIDs = oldCacheIDs
                }
            }
        }

        override fun visit(rel: RelNode): RelNode {
            // Check if we have already visited this node. This is necessary because
            // we must ensure the new cache node we generate has the same ID in all locations
            // in addition to ensuring we only visit the body once.
            if (createdCacheIDs.contains(rel.id) && createdCacheIDs[rel.id] != null) {
                val cacheNode = createdCacheIDs[rel.id]!!
                cacheNode.cachedPlan.addConsumer()
                // This step will generate any join filters that are needed. Since we know a cached
                // node step cannot generate any new filters (as otherwise we would not have reached
                // this point), we can safely call visit. This also ensures that the new cache component
                // that we generated will not be repeatedly visited.
                return visit(cacheNode.copy(cacheNode.traitSet, cacheNode.inputs))
            }
            val result =
                when (rel) {
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

                    is BodoPhysicalWindow -> {
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

                    is SnowflakeToBodoPhysicalConverter -> {
                        visit(rel)
                    }

                    is IcebergToBodoPhysicalConverter -> {
                        visit(rel)
                    }

                    is BodoPhysicalCachedSubPlan -> {
                        visit(rel)
                    }

                    else -> {
                        // Unknown Node: We must pop any remaining filters.
                        val oldLiveJoins = liveJoins
                        liveJoins = JoinFilterProgramState()
                        val parent = super.visit(rel)
                        if (emitFilters) {
                            applyFilters(parent, oldLiveJoins)
                        } else {
                            parent
                        }
                    }
                }
            // If this is our first cache node visit we need to store the result.
            return if (createdCacheIDs.contains(rel.id)) {
                // If we cache a node we may have omitted RTJFs right before the new cache node.
                val (newCacheNode, returnNode) =
                    if (result is BodoPhysicalRuntimeJoinFilter) {
                        val newCacheNode =
                            BodoPhysicalCachedSubPlan.create(
                                CachedPlanInfo.create(result.input, 1),
                                cluster.nextCacheId(),
                            )
                        val returnNode = result.copy(result.traitSet, listOf(newCacheNode))
                        Pair(newCacheNode, returnNode)
                    } else {
                        val newCacheNode =
                            BodoPhysicalCachedSubPlan.create(
                                CachedPlanInfo.create(result, 1),
                                cluster.nextCacheId(),
                            )
                        Pair(newCacheNode, newCacheNode)
                    }
                createdCacheIDs[rel.id] = newCacheNode
                returnNode
            } else {
                result
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
            liveJoins: JoinFilterProgramState,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            val pushLiveJoins = JoinFilterProgramState()
            val outputLiveJoinInfo = JoinFilterProgramState()
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
                    pushLiveJoins.add(joinInfo.joinFilterKey, pushKeys, pushIsFirstLocation)
                }
                if (outputIsFirstLocation.any { it }) {
                    outputLiveJoinInfo.add(joinInfo.joinFilterKey, outputKeys, outputIsFirstLocation)
                }
            }

            return Pair(pushLiveJoins, outputLiveJoinInfo)
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
            pushLiveJoinInfo: JoinFilterProgramState,
            outputLiveJoinInfo: JoinFilterProgramState,
        ): RelNode {
            liveJoins = pushLiveJoinInfo
            val newNode = super.visit(rel)
            return if (emitFilters) {
                applyFilters(newNode, outputLiveJoinInfo)
            } else {
                newNode
            }
        }

        private fun visit(project: BodoPhysicalProject): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processProject(project, liveJoins, true)
            return processSingleRel(project, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(filter: BodoPhysicalFilter): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processFilter(filter, liveJoins)
            return processSingleRel(filter, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(converter: SnowflakeToBodoPhysicalConverter): RelNode {
            return if (liveJoins.isEmpty() || !emitFilters) {
                converter
            } else {
                val (filterKeys, filterColumns, areFirstLocations) = liveJoins.flattenToLists()
                val snowflakeRtjf =
                    SnowflakeRuntimeJoinFilter.create(
                        converter.input,
                        filterKeys,
                        filterColumns,
                        areFirstLocations,
                        (converter.input as SnowflakeRel).getCatalogTable(),
                    )
                liveJoins = JoinFilterProgramState()
                converter.copy(converter.traitSet, listOf(snowflakeRtjf))
            }
        }

        private fun visit(converter: IcebergToBodoPhysicalConverter): RelNode {
            return if (liveJoins.isEmpty() || !emitFilters) {
                converter
            } else {
                val (filterKeys, filterColumns, areFirstLocations) = liveJoins.flattenToLists()
                val icebergRtjf =
                    IcebergRuntimeJoinFilter.create(
                        converter.input,
                        filterKeys,
                        filterColumns,
                        areFirstLocations,
                        (converter.input as IcebergRel).getCatalogTable(),
                    )
                liveJoins = JoinFilterProgramState()
                converter.copy(converter.traitSet, listOf(icebergRtjf))
            }
        }

        private fun visit(node: BodoPhysicalMinRowNumberFilter): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processMinRowNumberFilter(node, liveJoins, true)
            return processSingleRel(node, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(sort: BodoPhysicalSort): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processSort(sort, liveJoins)
            return processSingleRel(sort, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(window: BodoPhysicalWindow): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processWindow(window, liveJoins, true)
            return processSingleRel(window, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(aggregate: BodoPhysicalAggregate): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processAggregate(aggregate, liveJoins, true)
            return processSingleRel(aggregate, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(join: BodoPhysicalJoin): RelNode {
            val info = join.analyzeCondition()
            // Note: You must use getRowType() here due to a caching issue.
            val numLeftColumns = join.left.rowType.fieldCount
            // Split the join info into those which are processed on
            // the left input and those which are processed on the right input.
            // equi-join keys can be processed on both inputs.
            val leftJoinInfo = JoinFilterProgramState()
            val rightJoinInfo = JoinFilterProgramState()
            // If the filters are split across both sides and neither side
            // gets all the columns, we may need to generate a filter after
            // this join to ensure the bloom filter runs.
            val outputJoinInfo = JoinFilterProgramState()
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
                    leftJoinInfo.add(
                        joinInfo.joinFilterKey,
                        leftKeys,
                        leftIsFirstLocation,
                    )
                }
                if (keepRight) {
                    rightJoinInfo.add(
                        joinInfo.joinFilterKey,
                        rightKeys,
                        rightIsFirstLocation,
                    )
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
            val (filterKey, originalLocations) =
                if (!join.joinType.generatesNullsOnRight()) {
                    val columns = info.leftKeys
                    if (columns.isEmpty()) {
                        Pair(-1, listOf())
                    } else {
                        val filterKey = cluster.nextJoinId()
                        // Add a new join to the left side.
                        leftJoinInfo.add(
                            filterKey,
                            columns,
                            List(columns.size) { true },
                        )
                        Pair(filterKey, (0 until columns.size).toList())
                    }
                } else {
                    Pair(-1, listOf())
                }
            liveJoins = leftJoinInfo
            val leftInput = join.left.accept(this)
            liveJoins = rightJoinInfo
            val rightInput = join.right.accept(this)
            return if (emitFilters) {
                val newJoin =
                    BodoPhysicalJoin.create(
                        leftInput,
                        rightInput,
                        join.hints,
                        join.condition,
                        join.joinType,
                        rebalanceOutput = join.rebalanceOutput,
                        joinFilterID = filterKey,
                        originalJoinFilterKeyLocations = originalLocations,
                        broadcastBuildSide = join.broadcastBuildSide,
                    )
                return applyFilters(newJoin, outputJoinInfo)
            } else {
                join
            }
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
            return if (emitFilters) {
                node.copy(node.traitSet, newInputs, node.all)
            } else {
                node
            }
        }

        private fun visit(flatten: BodoPhysicalFlatten): RelNode {
            val (pushLiveJoinInfo, outputLiveJoinInfo) = processFlatten(flatten, liveJoins, true)
            return processSingleRel(flatten, pushLiveJoinInfo, outputLiveJoinInfo)
        }

        private fun visit(cachedSubPlan: BodoPhysicalCachedSubPlan): RelNode {
            if (!emitFilters) {
                throw IllegalStateException("Cache plan visiting requires filters to be emitted.")
            }
            val pushedFilters = getPushableJoinFilters(cachedSubPlan.cachedPlan.plan, liveJoins)
            val generatedFilters = liveJoins.difference(pushedFilters)
            // Clear the filters
            liveJoins = JoinFilterProgramState()
            // We must make a copy in case there is nested caching since we identify nodes by ID.
            val cachedCopy = cachedSubPlan.copy(cachedSubPlan.traitSet, cachedSubPlan.inputs)
            // Mark the cache node as visited.
            cacheVisitor.add(cachedCopy, Pair(cachedCopy.id, pushedFilters))
            return applyFilters(cachedCopy, generatedFilters)
        }

        /**
         * Process a project node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. If updateColumns is true, the columns will be updated
         * based on the project. If false we will just return the old columns indices, which is used
         * for cache processing.
         * @param project The project node to process.
         * @param liveJoins The current state of the live joins.
         * @param updateColumns If true, the columns will be updated based on the project. If false, then
         * the pushed join filters will use the original indices. This is useful for cache processing when we
         * just want to detect the pushable subset.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processProject(
            project: BodoPhysicalProject,
            liveJoins: JoinFilterProgramState,
            updateColumns: Boolean,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            // If the project contains an OVER clause we can only push filters
            // that are shared by all partition by columns. These are based
            // on the input column locations.
            val numCols = project.input.rowType.fieldCount
            var pushableColumns =
                if (project.containsOver()) {
                    WindowFilterTranspose.getFilterableColumnIndices(project.projects, numCols)
                } else {
                    ImmutableBitSet.range(numCols)
                }

            val columnTransformFunction =
                if (updateColumns) {
                    { idx: Int -> (project.projects[idx] as RexInputRef).index }
                } else {
                    { idx: Int -> idx }
                }

            return splitFilterSections(
                canPushPredicate = {
                    project.projects[it] is RexInputRef && pushableColumns.get((project.projects[it] as RexInputRef).index)
                },
                columnTransformFunction = columnTransformFunction,
                liveJoins,
            )
        }

        /**
         * Process a filter node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. Filters cannot reorder columns.
         * @param filter The Filter node to process.
         * @param liveJoins The current state of the live joins.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processFilter(
            filter: BodoPhysicalFilter,
            liveJoins: JoinFilterProgramState,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            // If the filter contains an OVER clause we can only push filters
            // that are shared by all partition by columns. The input column
            // and filter column locations must match exactly, so we can just
            // look at the filter.
            val numCols = filter.rowType.fieldCount
            var pushableColumns =
                if (filter.containsOver()) {
                    WindowFilterTranspose.getFilterableColumnIndices(listOf(filter.condition), numCols)
                } else {
                    ImmutableBitSet.range(numCols)
                }
            return splitFilterSections(canPushPredicate = { pushableColumns.get(it) }, columnTransformFunction = { it }, liveJoins)
        }

        /**
         * Process a min row number filter node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. If updateColumns is true, the columns will be updated
         * based on the kept columns. If false we will just return the old columns indices, which is used
         * for cache processing.
         * @param node The min row number node to process.
         * @param liveJoins The current state of the live joins.
         * @param updateColumns If true, the columns will be updated based on the kept inputs. If false, then
         * the pushed join filters will use the original indices. This is useful for cache processing when we
         * just want to detect the pushable subset.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processMinRowNumberFilter(
            node: BodoPhysicalMinRowNumberFilter,
            liveJoins: JoinFilterProgramState,
            updateColumns: Boolean,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            val keptInputs = node.inputsToKeep.toList()
            val columnTransformFunction =
                if (updateColumns) {
                    { idx: Int -> keptInputs[idx] }
                } else {
                    { idx: Int -> idx }
                }
            return splitFilterSections(
                // Only push columns which are in the partition set.
                canPushPredicate = { node.partitionColSet.contains(it) },
                columnTransformFunction = columnTransformFunction,
                liveJoins,
            )
        }

        /**
         * Process a sort node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. Sorts cannot reorder columns.
         * @param sort The Sort node to process.
         * @param liveJoins The current state of the live joins.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processSort(
            sort: BodoPhysicalSort,
            liveJoins: JoinFilterProgramState,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            val canPush = !LimitUtils.isOrderedLimit(sort)
            return splitFilterSections(
                canPushPredicate = { canPush },
                columnTransformFunction = { it },
                liveJoins,
            )
        }

        /**
         * Process a window node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. If updateColumns is true, the columns will be updated
         * based on the original key indices. If false we will just return the old columns indices, which is used
         * for cache processing.
         * @param window The window node to process.
         * @param liveJoins The current state of the live joins.
         * @param updateColumns If true, the columns will be updated based on the aggregate mapping. If false, then
         * the pushed join filters will use the original indices. This is useful for cache processing when we
         * just want to detect the pushable subset.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processWindow(
            window: BodoPhysicalWindow,
            liveJoins: JoinFilterProgramState,
            updateColumns: Boolean,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            // Split the join info into those which must be processed now
            // and those which can be processed later.
            assert(window.supportsStreamingWindow())
            val keptInputs = window.inputsToKeep.toList()
            val columnTransformFunction =
                if (updateColumns) {
                    { idx: Int -> keptInputs[idx] }
                } else {
                    { idx: Int -> idx }
                }
            // Only push columns which are partition keys in every group.
            val commonPartitionKeys = window.groups.map { it.keys }.reduce(ImmutableBitSet::intersect)
            return splitFilterSections(
                canPushPredicate = { colIdx -> commonPartitionKeys.get(colIdx) },
                columnTransformFunction = columnTransformFunction,
                liveJoins,
            )
        }

        /**
         * Process an aggregate node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. If updateColumns is true, the columns will be updated
         * based on the original key indices. If false we will just return the old columns indices, which is used
         * for cache processing.
         * @param aggregate The aggregate node to process.
         * @param liveJoins The current state of the live joins.
         * @param updateColumns If true, the columns will be updated based on the aggregate mapping. If false, then
         * the pushed join filters will use the original indices. This is useful for cache processing when we
         * just want to detect the pushable subset.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processAggregate(
            aggregate: BodoPhysicalAggregate,
            liveJoins: JoinFilterProgramState,
            updateColumns: Boolean,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            // Split the join info into those which must be processed now
            // and those which can be processed later.
            val groupByKeys = aggregate.groupSet.toList()
            val columnTransformFunction =
                if (updateColumns) {
                    { idx: Int -> groupByKeys[idx] }
                } else {
                    { idx: Int -> idx }
                }
            return splitFilterSections(
                // Only push columns which are in the group by set.
                canPushPredicate = { it < aggregate.groupCount },
                columnTransformFunction = columnTransformFunction,
                liveJoins,
            )
        }

        /**
         * Process a Flatten node by splitting into the remaining join filters for the current node and
         * join filters that can be pushed further. If updateColumns is true, the columns will be updated
         * based on the repeats columns. If false we will just return the old columns indices, which is used
         * for cache processing.
         * @param flatten The flatten node to process.
         * @param liveJoins The current state of the live joins.
         * @param updateColumns If true, the columns will be updated based on the repeats columns. If false, then
         * the pushed join filters will use the original indices. This is useful for cache processing when we
         * just want to detect the pushable subset.
         * @return The pair of join filters that can be pushed and the join filters that must be applied at the current node.
         */
        private fun processFlatten(
            flatten: BodoPhysicalFlatten,
            liveJoins: JoinFilterProgramState,
            updateColumns: Boolean,
        ): Pair<JoinFilterProgramState, JoinFilterProgramState> {
            val repeatColumns = flatten.repeatColumns.toList()
            val columnTransformFunction =
                if (updateColumns) {
                    { idx: Int -> repeatColumns[idx] }
                } else {
                    { idx: Int -> idx }
                }
            return splitFilterSections(
                // Only push columns which are copied from the input, not computed.
                canPushPredicate = { it < repeatColumns.size },
                columnTransformFunction = columnTransformFunction,
                liveJoins,
            )
        }

        /**
         * Apply the filters to the current node to generate a new RuntimeJoinFilter if
         * necessary.
         */
        private fun applyFilters(
            rel: RelNode,
            liveJoins: JoinFilterProgramState,
        ): RelNode {
            return if (liveJoins.isEmpty()) {
                rel
            } else {
                val (filterKeys, filterColumns, areFirstLocations) = liveJoins.flattenToLists()
                BodoPhysicalRuntimeJoinFilter.create(
                    rel,
                    filterKeys,
                    filterColumns,
                    areFirstLocations,
                )
            }
        }

        /**
         * Derive the filters that can be pushed past the current node. This is used for determining
         * the interaction between caching nodes and runtime join filters currently sitting atop them.
         * In most cases, this logic is identical to the logic for visiting a node, but without generating
         * any nodes. In cases where something is always pushable to multiple inputs, the results will differ
         * because we only care about what can be pushed to any input.
         *
         * @param rel The RelNode to process.
         * @param liveJoins The current state of the live joins.
         */
        @VisibleForTesting
        internal fun getPushableJoinFilters(
            rel: RelNode,
            liveJoins: JoinFilterProgramState,
        ): JoinFilterProgramState {
            return when (rel) {
                is BodoPhysicalProject -> {
                    val (pushed, _) = processProject(rel, liveJoins, false)
                    pushed
                }

                is BodoPhysicalFilter -> {
                    val (pushed, _) = processFilter(rel, liveJoins)
                    pushed
                }

                is BodoPhysicalMinRowNumberFilter -> {
                    val (pushed, _) = processMinRowNumberFilter(rel, liveJoins, false)
                    pushed
                }

                is BodoPhysicalSort -> {
                    val (pushed, _) = processSort(rel, liveJoins)
                    pushed
                }

                is BodoPhysicalWindow -> {
                    val (pushed, _) = processWindow(rel, liveJoins, false)
                    pushed
                }

                is BodoPhysicalAggregate -> {
                    val (pushed, _) = processAggregate(rel, liveJoins, false)
                    pushed
                }

                is BodoPhysicalFlatten -> {
                    val (pushed, _) = processFlatten(rel, liveJoins, false)
                    pushed
                }
                is BodoPhysicalJoin, is BodoPhysicalMinus, is BodoPhysicalIntersect, is BodoPhysicalUnion,
                is SnowflakeToBodoPhysicalConverter, is IcebergToBodoPhysicalConverter,
                -> {
                    // We can push everything past these nodes.
                    liveJoins
                }
                else -> {
                    // Fallback to pushing nothing.
                    JoinFilterProgramState()
                }
            }
        }
    }

    internal class JoinFilterCacheReplace(
        private val keptCacheUpdates: Map<Int, RelNode>,
        private val inlinedCacheUpdates: Map<Int, RelNode>,
    ) : RelShuttleImpl() {
        // Visitor for updating the body of cache nodes.
        private val cacheInlineApply =
            CachedResultVisitor<Unit, CachedSubPlanBase> {
                    plan ->
                val result = visit(plan.cachedPlan.plan)
                plan.cachedPlan.plan = result
            }

        override fun visit(rel: RelNode): RelNode {
            return when (rel) {
                is BodoPhysicalCachedSubPlan -> {
                    val id = rel.id
                    if (keptCacheUpdates.containsKey(id)) {
                        // Ensure the body of the cache node is updated.
                        cacheInlineApply.visit(rel)
                        keptCacheUpdates[id]!!
                    } else if (inlinedCacheUpdates.contains(rel.id)) {
                        // We must visit the node once per inlined location.
                        val result = visit(inlinedCacheUpdates[rel.id]!!)
                        result
                    } else {
                        // This is a newly generated cache node
                        cacheInlineApply.visit(rel)
                        rel
                    }
                }
                else -> super.visit(rel)
            }
        }
    }

    /**
     * Visitor for updating the state of how to process the body of a cache node
     * that needs to be inlined. This runs as the first pass, where we update the
     * filter state information without actually generating any nodes.
     *
     * This outputs a set of nodes that should still be cached by all consumers and
     * has a side effect of updating the number of expected consumers for each cache node
     * that increases its consumer count due to a parent cache node getting inlined.
     */
    internal class InlineCacheVisitor(
        cluster: BodoRelOptCluster,
        liveJoins: JoinFilterProgramState,
        private val numInlineConsumers: Int,
    ) : RuntimeJoinFilterShuttle(cluster, false, liveJoins) {
        private val cachedNodes = mutableSetOf<Int>()

        override fun visit(node: RelNode): RelNode {
            if (node is BodoPhysicalCachedSubPlan) {
                // If we inline this node, we need to update the number of consumers.
                // It moves from 1 -> C, so we add C - 1.
                node.cachedPlan.addConsumers(numInlineConsumers - 1)
            } else {
                if (getPushableJoinFilters(node, liveJoins).isEmpty()) {
                    // If no join filters can push past this node, then this is the new cache location.
                    if (CacheSubPlanProgram.canCacheNode(node)) {
                        cachedNodes.add(node.id)
                    }
                } else {
                    super.visit(node)
                }
            }
            // We never actually modify the nodes in this pass.
            return node
        }

        fun getNewCacheNodes(): Set<Int> {
            return cachedNodes
        }
    }
}
