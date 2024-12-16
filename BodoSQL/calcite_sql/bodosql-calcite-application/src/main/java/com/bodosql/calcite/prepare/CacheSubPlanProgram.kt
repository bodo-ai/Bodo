package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalCachedSubPlan
import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.common.TreeReverserDuplicateTracker
import com.bodosql.calcite.adapter.iceberg.IcebergToBodoPhysicalConverter
import com.bodosql.calcite.adapter.pandas.PandasToBodoPhysicalConverter
import com.bodosql.calcite.adapter.snowflake.SnowflakeToBodoPhysicalConverter
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.logicalRules.FilterRulesCommon
import com.bodosql.calcite.rel.core.BodoPhysicalRelFactories
import com.bodosql.calcite.rel.core.CachedPlanInfo
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.rel.core.MinRowNumberFilterBase
import com.bodosql.calcite.rel.core.cachePlanContainers.CachedResultVisitor
import org.apache.calcite.plan.BodoRelOptCluster
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelCollations
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.BodoRexSimplify
import org.apache.calcite.rex.RexExecutor
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexPermuteInputsShuttle
import org.apache.calcite.rex.RexSimplify
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.tools.Program
import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.tools.RelBuilder.AggCall
import org.apache.calcite.util.ImmutableBitSet
import org.apache.calcite.util.Util
import org.apache.calcite.util.mapping.MappingType
import org.apache.calcite.util.mapping.Mappings

/**
 * Find identical sections of the plan and cache them in explicit
 * cache nodes. Depending on the value of
 * RelationalAlgebraGenerator.coveringExpressionCaching this may
 * result in only caching exact matches, or it may apply to a
 * broader set of operations.
 */
class CacheSubPlanProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode =
        if (RelationalAlgebraGenerator.coveringExpressionCaching) {
            runCoveringExpressions(rel, planner.executor)
        } else {
            runExactMatch(rel)
        }

    /**
     * Implement caching across sections of the plan that can match
     * by providing covering expressions or exact matches. This largely
     * implemented the covering expression handling that is covered by
     * this design document:
     * https://bodo.atlassian.net/wiki/spaces/B/pages/1763803160/Intra-Query+Covering+Expression+Caching+POC.
     *
     * However, there are a couple key limitations. In particular, there are limitations around
     * "defining groups of matching expressions" and in many situations we can only cache
     * if we see "all aggregates" or "all joins". In addition, several places don't continue
     * caching on "groups" of inputs.
     *
     * Note: This is not fully deployed across our test suite yet, so any change to this code
     * should ensure we run all tests with RelationalAlgebraGenerator.coveringExpressionCaching
     * = True before merging.
     *
     * @param rel The original plan.
     * @return The new plan with covering expressions generated if
     * possible.
     */
    private fun runCoveringExpressions(
        rel: RelNode,
        executor: RexExecutor?,
    ): RelNode {
        val reverser = TreeReverserDuplicateTracker()
        reverser.visit(rel, 0, null)
        val nodeCounts = reverser.getNodeCounts()
        val heights = reverser.getHeights()
        val reversedTree = reverser.getReversedTree()
        // Remove any nodes that are only visited once as candidate starting point. Then we
        // sort by height in case any cache node is the descendant of another cache node (to
        // potentially allow for maximum fusion). Use the name to break ties for consistent testing.
        val cacheCandidates = nodeCounts.filter { it.value > 1 }.keys.sortedWith(compareBy({ heights[it] }, { it.toString() }))
        return if (cacheCandidates.isEmpty()) {
            rel
        } else {
            val cluster = rel.cluster
            if (cluster !is BodoRelOptCluster) {
                throw InternalError("Cluster must be a BodoRelOptCluster")
            }
            val relBuilder = BodoPhysicalRelFactories.BODO_PHYSICAL_BUILDER.create(cluster, null)
            val replacementMap = generateCoveringExpressionCacheNodes(cacheCandidates, reversedTree, cluster, relBuilder, executor)
            val replacer = CoveringExpressionCacheReplacement(replacementMap)
            replacer.visit(rel)
        }
    }

    /**
     * Process all caching candidates and generate a mapping from the original nodes to the
     * new cache nodes that should generate a replacement. A followup steps does the replacement
     * of any original RelNodes including within the body of cache nodes for nested caching.
     *
     * This is implement this for nested caching, results will either be "complete subsets", in which
     * case one of the original nodes will exist in the new generated cache node to be replaced, or
     * completely combined, in which case just a single top level cache replacement will be provided.
     *
     * @param cacheCandidates The list of cache candidates to consider.
     * @param reversedTree The reversed tree mapping from each node to its parents. This is used to combine
     * and compare various sections of the plan to enable larger caching.
     * @param cluster Used to generate cache nodes.
     * @param relBuilder Used to generate new RelNodes for caching.
     * @param executor Used for simplification.
     * @return A mapping from the original nodes to the new cache nodes that should replace them.
     */
    private fun generateCoveringExpressionCacheNodes(
        cacheCandidates: List<RelNode>,
        reversedTree: Map<RelNode, MutableList<RelNode>>,
        cluster: BodoRelOptCluster,
        relBuilder: RelBuilder,
        executor: RexExecutor?,
    ): MutableMap<Pair<RelNode, RelNode>, RelNode> {
        val map = HashMap<Pair<RelNode, RelNode>, RelNode>()
        // Track the nodes we have already seen in case we "combine" caching locations.
        val seenSet = mutableSetOf<RelNode>()
        val cacheGenerator =
            CoveringExpressionCacheGenerator(
                reversedTree,
                map,
                seenSet,
                cluster,
                relBuilder,
                executor,
            )
        for (candidate in cacheCandidates) {
            if (seenSet.contains(candidate)) {
                continue
            }
            // Build the initial stateful input.
            val keptColumns = (0 until candidate.rowType.fieldCount).toList()
            val filter = null
            val state = CoveringExpressionState(candidate, keptColumns, filter)
            cacheGenerator.processCaching(candidate, listOf(state))
        }
        cacheGenerator.finalizeInProgressCaching()
        return map
    }

    /**
     * Generate the cache expressions for a given list of states.
     * If the parents of multiple nodes can be combined into a valid
     * covering expressions, then this will simply continue to extend
     * the possible cache consumers. However, if this finds incompatible
     * nodes, then this will generate a new cache node by updating the
     * cacheMap. If afterwards there are further "sections" that can be
     * cached together, it will continue to recurse.
     * @param reversedTree The reversed tree mapping from each node to its parents.
     * @param cacheMap The mapping from the original nodes to the new cache nodes. This
     * will be updated if a new cache node is generated.
     * @param seenSet The set of nodes that have already been processed as part of caching.
     * By definition, we add the "states" not the parents to this set until the parents are
     * determined to be included in caching.
     * @param cluster Used to generate cache nodes.
     * @param relBuilder Used to generate new RelNodes for caching.
     * @param executor Used for simplification.
     */
    private class CoveringExpressionCacheGenerator(
        private val reversedTree: Map<RelNode, MutableList<RelNode>>,
        private val cacheMap: MutableMap<Pair<RelNode, RelNode>, RelNode>,
        private val seenSet: MutableSet<RelNode>,
        private val cluster: BodoRelOptCluster,
        private val relBuilder: RelBuilder,
        executor: RexExecutor?,
    ) {
        private val simplify: RexSimplify =
            BodoRexSimplify(relBuilder.rexBuilder, RelOptPredicateList.EMPTY, Util.first(executor, RexUtil.EXECUTOR))

        // All joins that only have 1 side processed. This maps the join to
        // the cache node in progress for the input, which is used as a key
        // for a larger caching group.
        // Note: This could be extended to any multi-input node, but currently
        // only joins are supported.
        private val pausedJoins: MutableMap<RelNode, RelNode> = HashMap()

        // Whole cache state for nodes that are paused by a join.
        private val inProgressNodes: MutableMap<RelNode, List<Pair<CoveringExpressionState, RelNode>>> = HashMap()

        /**
         * Generates the common cache nodes for a given set of states
         * and update the cacheMap with the new cache nodes.
         * @param cacheRoot The current root of the expression presently constructed
         * that should be converted into a cache node if it isn't already.
         * @param parents The states + parent RelNode that should be replaced with
         * the cache value materialization.
         * @param numConsumers The number of consumers that will be added to the cache node
         * if it already exists or the total number of consumeres for any new cache node.
         * @return A copy of the underlying cache node in cache this is only
         * a subset of the possible caching. If no caching is possible then
         * the original progress is returned.
         */
        private fun generateCacheNodes(
            cacheRoot: RelNode,
            parents: List<Pair<CoveringExpressionState, RelNode>>,
            numConsumers: Int,
        ): RelNode {
            val canCache = cacheRoot.convention == BodoPhysicalRel.CONVENTION
            return if (canCache) {
                // Don't create a new cache node if the current root is a cache node.
                val cacheNode =
                    if (cacheRoot is BodoPhysicalCachedSubPlan) {
                        // We are reusing the same cache node as earlier. Previously
                        // this "group" would have been mapped as a single consumer,
                        // so we need to add n - 1 more consumers to the cache node.
                        cacheRoot.cachedPlan.addConsumers(numConsumers - 1)
                        cacheRoot
                    } else {
                        BodoPhysicalCachedSubPlan.create(
                            CachedPlanInfo.create(cacheRoot, numConsumers),
                            cluster.nextCacheId(),
                        )
                    }
                parents.forEach { parent ->
                    var cacheReplacement: RelNode = cacheNode.copy(cacheNode.traitSet, cacheNode.inputs)
                    val baseNode = parent.first.baseNode
                    if (parent.first.filter != null) {
                        relBuilder.push(cacheReplacement)
                        relBuilder.filter(parent.first.filter)
                        cacheReplacement = relBuilder.build()
                    }
                    // Note: We assume the RelBuilder omits trivial projects.
                    relBuilder.push(cacheReplacement)
                    relBuilder.project(
                        parent.first.keptColumns.map { idx ->
                            if (idx < 0) {
                                // Dummy column we need to maintain locations
                                relBuilder.rexBuilder.makeLiteral(0, relBuilder.typeFactory.createSqlType(SqlTypeName.BIGINT))
                            } else {
                                relBuilder.field(idx)
                            }
                        },
                    )
                    cacheReplacement = relBuilder.build()
                    cacheMap[Pair(baseNode, parent.second)] = cacheReplacement
                }
                cacheNode
            } else {
                cacheRoot
            }
        }

        /**
         * Finalize any caching steps that were paused because we were waiting for another input.
         * This is currently only used for join but could be extended to any multi-input node.
         */
        fun finalizeInProgressCaching() {
            inProgressNodes.toSortedMap(compareBy { it.toString() }).forEach { (cacheRoot, parents) ->
                generateCacheNodes(cacheRoot, parents, parents.size)
            }
        }

        /**
         * Perform the actual processing to generate the cache nodes and update the cacheMap.
         * @param cacheRoot The current root of the expression presently constructed
         * for caching purposes.
         * @param states The list of states that are currently being considered for
         * caching.
         */
        fun processCaching(
            cacheRoot: RelNode,
            states: List<CoveringExpressionState>,
        ) {
            if (states.isEmpty()) {
                return
            }
            // Mark all the state node as seen and fetch the parents. We associate each parent with a state
            // in case we need to materialize the node(s).
            val parents = mutableListOf<Pair<CoveringExpressionState, RelNode>>()
            for (state in states) {
                seenSet.add(state.baseNode)
            }
            for (state in states) {
                val parentNodes = reversedTree[state.baseNode]!!
                for (parentNode in parentNodes) {
                    // Don't include parents we have already seen in case different
                    // caching is available for two parents of a node.
                    if (!seenSet.contains(parentNode)) {
                        parents.add(Pair(state, parentNode))
                    }
                }
            }
            if (parents.any { it.second is Filter && it.second !is MinRowNumberFilterBase }) {
                // Check for any filter nodes and if so we need to process them across all nodes.
                // Any node that isn't a filter should generate a "TRUE".
                val rexBuilder = relBuilder.rexBuilder
                val trueNode = rexBuilder.makeLiteral(true)
                // Compute the union of all filters.
                val filtersList =
                    parents.map {
                        val columnIndices = it.first.keptColumns
                        val parent = it.second
                        if (parent is Filter && parent !is MinRowNumberFilterBase) {
                            val baseCondition = parent.condition
                            // Remap any columns in the condition.
                            val mapping = Mappings.source(columnIndices, cacheRoot.rowType.fieldCount)
                            baseCondition.accept(RexPermuteInputsShuttle(mapping, cacheRoot))
                        } else {
                            trueNode
                        }
                    }
                val (newRoot, newStates) = applyCommonFiltersUpdateState(cacheRoot, parents, filtersList, true, true)
                processCaching(newRoot, newStates)
            } else if (parents.any { it.second is Project }) {
                // Combine all project components into a map to avoid repeat compute.
                val computeMap = HashMap<RexNode, Int>()
                relBuilder.push(cacheRoot)
                val newIndices =
                    parents.map {
                        val columnIndices = it.first.keptColumns
                        val parent = it.second
                        if (parent is Project) {
                            val projects = parent.projects
                            projects.map { node ->
                                // Remap any columns in the condition.
                                val mapping = Mappings.source(columnIndices, cacheRoot.rowType.fieldCount)
                                val updatedNode = node.accept(RexPermuteInputsShuttle(mapping, cacheRoot))
                                if (computeMap.contains(updatedNode)) {
                                    computeMap[updatedNode]!!
                                } else {
                                    val newIdx = computeMap.size
                                    computeMap[updatedNode] = newIdx
                                    newIdx
                                }
                            }
                        } else {
                            // None projections just need to maintain every column they have already seen.
                            columnIndices.map { idx ->
                                val node = relBuilder.field(idx)
                                if (computeMap.contains(node)) {
                                    computeMap[node]!!
                                } else {
                                    val newIdx = computeMap.size
                                    computeMap[node] = newIdx
                                    newIdx
                                }
                            }
                        }
                    }
                // Generate the new shared projection by flattening the map to a list.
                val newProjects = computeMap.toList().sortedBy { it.second }.map { it.first }
                val newRoot = relBuilder.project(newProjects).build()
                // Update the filters and check for any filter columns no longer existing. To simplify the logic,
                // we will just generate a cache node from the previous cache root and then rerun the result.
                relBuilder.push(cacheRoot)
                val filterInfo =
                    parents.map {
                        val filter = it.first.filter
                        if (filter == null) {
                            Pair(null, true)
                        } else {
                            // Check every column used by the filter is found in the new projection map and use
                            // the new index if it is.
                            val usedColumns = RelOptUtil.InputFinder.bits(filter)
                            val indices = MutableList(cacheRoot.rowType.fieldCount) { -1 }
                            usedColumns.forEach { idx ->
                                val node = relBuilder.field(idx)
                                if (computeMap.contains(node)) {
                                    indices[idx] = computeMap[node]!!
                                } else {
                                    // We pruned the filter column, so we will need to materialize this
                                    // node. In the future we may want to explore keeping the column around.
                                    return@map Pair(null, false)
                                }
                            }
                            val mapping = Mappings.source(indices, newRoot.rowType.fieldCount)
                            val updatedFilter = filter.accept(RexPermuteInputsShuttle(mapping, newRoot))
                            Pair(updatedFilter, true)
                        }
                    }
                val materializedCaching =
                    filterInfo.withIndex().filter { !it.value.second }.map { Pair(it.value.first, it.index) }
                if (materializedCaching.isNotEmpty()) {
                    // TODO: If we don't have to materialize at least two nodes we could continue caching.
                    generateCacheNodes(cacheRoot, parents, parents.size)
                } else {
                    val seenStates = mutableSetOf<CoveringExpressionState>()
                    val newStates = mutableListOf<CoveringExpressionState>()
                    parents.withIndex().forEach { (idx, parent) ->
                        val newBaseCandidate = parent.second
                        val baseNode = parent.first.baseNode
                        val newBaseNode =
                            if (newBaseCandidate is Project) {
                                newBaseCandidate
                            } else if (!seenStates.contains(parent.first)) {
                                // Avoid duplicates since only some node progress.
                                seenStates.add(parent.first)
                                baseNode
                            } else {
                                null
                            }
                        if (newBaseNode != null) {
                            val keptColumns = newIndices[idx]
                            val newFilter = filterInfo[idx].first
                            newStates.add(CoveringExpressionState(newBaseNode, keptColumns, newFilter))
                        }
                    }
                    processCaching(newRoot, newStates)
                }
            } else if (parents.all {
                    it.second is Aggregate &&
                        (it.second as Aggregate).groupSets.size == 1 &&
                        (it.second as Aggregate).aggCallList.all { agg ->
                            !agg.hasFilter() && agg.distinctKeys == null
                        }
                }
            ) {
                // Fuse aggregates if all the keys match exactly, there is exactly one group set, and there are no filters.
                // This function also contains logic for splitting the aggregate into multiple caching groups if the
                // set of keys is different, but this should be generalized beyond requiring every node to contain
                // aggregates.
                processAggregate(cacheRoot, parents)
            } else if (parents.all { it.second is Join && (it.second as Join).joinType == JoinRelType.INNER }) {
                // Joins require special handling because to ensure a join can be cached we need to ensure
                // both side can consistently be cached. As a simpler check we require all joins be inner joins.
                // We also need the conditions to be the same, but this requires having the cache information about
                // both sides of the join to be able to normalize.
                processJoin(cacheRoot, parents)
            } else if (parents.all {
                    it.second is SnowflakeToBodoPhysicalConverter ||
                        it.second is PandasToBodoPhysicalConverter ||
                        it.second is IcebergToBodoPhysicalConverter
                }
            ) {
                // Check for converters which should always match
                // Note: We don't need to check these separately by construction because they couldn't match to this point
                // if they had different conventions.
                val newRoot = parents[0].second.copy(parents[0].second.traitSet, listOf(cacheRoot))
                processCaching(
                    newRoot,
                    parents.map { CoveringExpressionState(it.second, it.first.keptColumns, it.first.filter) },
                )
            } else {
                generateCacheNodes(cacheRoot, parents, parents.size)
            }
        }

        /**
         * Process an Aggregate that is known to be a candidate for caching. If all aggregates
         * have the same set of keys these can be cached/fused into the same aggregation. Otherwise,
         * we cache the input and group any set of two or more locations with the same aggregate
         * into additional caching. If not all keys match but any aggregate matches the union of
         * all keys then we also compute that "larger" aggregate as a cache for all nodes as
         * a "partial aggregation" since the majority of cost from an aggregate is computing the
         * distinct groups (so little additional cost is incurred).
         * @param cacheRoot The current caching root.
         * @param parents The list of parents all of which have aggregates as the RelNode above
         * the current caching state.
         */
        private fun processAggregate(
            cacheRoot: RelNode,
            parents: List<Pair<CoveringExpressionState, RelNode>>,
        ) {
            val groupSets =
                parents.map {
                    val agg = it.second as Aggregate
                    val indices = it.first.keptColumns
                    val newGroupsBuilder = ImmutableBitSet.builder()
                    agg.groupSet.forEach { idx ->
                        newGroupsBuilder.set(indices[idx])
                    }
                    newGroupsBuilder.build()
                }
            val unionGroupSet = groupSets.reduce { acc, set -> acc.union(set) }
            val intersectGroupSet = groupSets.reduce { acc, set -> acc.intersect(set) }
            val allKeysMatch = unionGroupSet == intersectGroupSet
            if (allKeysMatch) {
                val (newRoot, newIndices, filterInfo) = createNewAggregate(unionGroupSet, groupSets, cacheRoot, parents)
                // If any filter cannot be remapped, we will need to materialize.
                val materializedCaching =
                    filterInfo.withIndex().filter { !it.value.second }.map { Pair(it.value.first, it.index) }
                if (materializedCaching.isNotEmpty()) {
                    // TODO: If we don't have to materialize at least two nodes we could continue caching.
                    generateCacheNodes(cacheRoot, parents, parents.size)
                } else {
                    val newStates =
                        parents.withIndex().map { (idx, parent) ->
                            val newBaseNode = parent.second
                            val keptColumns = newIndices[idx]
                            val newFilter = filterInfo[idx].first
                            CoveringExpressionState(newBaseNode, keptColumns, newFilter)
                        }
                    processCaching(newRoot, newStates)
                }
            } else {
                // All keys don't match, but there may be a subset that we can continue
                // to cache. TODO: Generalize this to allow for partial caching.
                val groupMap = HashMap<ImmutableBitSet, MutableList<Int>>()
                groupSets.withIndex().forEach {
                    val (idx, groupSet) = it
                    if (!groupMap.contains(groupSet)) {
                        groupMap[groupSet] = mutableListOf()
                    }
                    groupMap[groupSet]!!.add(idx)
                }
                val numConsumers = groupMap.size
                val (canDoPartialAgg, partialAggInfo) =
                    if (groupMap.contains(unionGroupSet)) {
                        // If any node contains every key, then we should use that as
                        // a "partial" aggregation for all other aggregations.
                        try {
                            val (combinedAggregate, updatedIndices, filterInfo) =
                                createNewAggregate(
                                    unionGroupSet,
                                    groupSets,
                                    cacheRoot,
                                    parents,
                                )
                            // If any filter cannot be remapped, we will need to materialize.
                            val materializedCaching =
                                filterInfo
                                    .withIndex()
                                    .filter { !it.value.second }
                                    .map { Pair(it.value.first, it.index) }
                            Pair(
                                materializedCaching.isEmpty(),
                                Triple(combinedAggregate, updatedIndices, filterInfo.map { it.first }),
                            )
                        } catch (e: Exception) {
                            Pair(false, Triple(cacheRoot, listOf(), listOf()))
                        }
                    } else {
                        Pair(false, Triple(cacheRoot, listOf(), listOf()))
                    }
                val (newCacheRoot, continuedGroups) =
                    if (canDoPartialAgg) {
                        val (combinedAggregate, updatedIndices, updatedFilters) = partialAggInfo
                        // Since we are doing a partial aggregation the caching may no longer match the "base nodes."
                        // This logic is based on the groupMap and we only need to update the baseNode for the parent(s)
                        // that match groupKey. Since the parent(s) change this may expand the size of this group and will
                        // need to be processed again, but not as an aggregate. All other nodes just need updated indices
                        // and filters.
                        val matchingIndices = groupMap.remove(unionGroupSet)!!
                        val matchingStates =
                            matchingIndices.map {
                                CoveringExpressionState(
                                    parents[it].second,
                                    updatedIndices[it],
                                    updatedFilters[it],
                                )
                            }
                        val materializedParents = mutableListOf<Pair<CoveringExpressionState, RelNode>>()
                        val continuedGroups = mutableListOf<List<Pair<CoveringExpressionState, RelNode>>>()
                        // Note: We never add materializedStates to continuedGroups because those require a separate code path.
                        if (matchingIndices.size == 1) {
                            val state = matchingStates[0]
                            val parentNodes = reversedTree[state.baseNode]!!
                            for (parentNode in parentNodes) {
                                // Don't include parents we have already seen in case different
                                // caching is available for two parents of a node.
                                if (!seenSet.contains(parentNode)) {
                                    materializedParents.add(Pair(state, parentNode))
                                }
                            }
                        }
                        groupMap.forEach { (groupKey, groupIndices) ->
                            val groupStates =
                                groupIndices.map {
                                    Pair(
                                        CoveringExpressionState(
                                            parents[it].first.baseNode,
                                            updatedIndices[it],
                                            updatedFilters[it],
                                        ),
                                        parents[it].second,
                                    )
                                }
                            if (groupStates.size == 1) {
                                materializedParents.add(groupStates[0])
                            } else {
                                continuedGroups.add(groupStates)
                            }
                        }
                        val newCacheRoot = generateCacheNodes(combinedAggregate, materializedParents, numConsumers)
                        if (matchingStates.size > 1) {
                            // The states with matching keys should run the full caching approach,
                            // not the aggregate approach.
                            processCaching(newCacheRoot, matchingStates)
                        }
                        Pair(newCacheRoot, continuedGroups)
                    } else {
                        // Any groups of size 1 must be fully materialized.
                        val materializedParents =
                            groupMap
                                .filter { it.value.size == 1 }
                                .values
                                .flatten()
                                .map { parents[it] }
                        val newCacheRoot =
                            generateCacheNodes(cacheRoot, materializedParents, numConsumers)
                        val continuedGroups =
                            groupMap.filter { it.value.size > 1 }.values.map { it.map { idx -> parents[idx] } }
                        Pair(newCacheRoot, continuedGroups)
                    }
                continuedGroups.forEach { groupParents ->
                    val rexBuilder = relBuilder.rexBuilder
                    val trueNode = rexBuilder.makeLiteral(true)
                    val groupFilters =
                        groupParents.map {
                            if (it.first.filter == null) {
                                trueNode
                            } else {
                                it.first.filter!!
                            }
                        }
                    val (groupRoot, groupStates) =
                        applyCommonFiltersUpdateState(
                            newCacheRoot,
                            groupParents,
                            groupFilters,
                            false,
                            false,
                        )
                    processAggregate(
                        groupRoot,
                        groupParents.withIndex().map { Pair(groupStates[it.index], it.value.second) },
                    )
                }
            }
        }

        /**
         * Create a new aggregation node that combines all the aggregations
         * with the given keys. This simplifies the process of generating the
         * "state" associated with building various types of new aggregates.
         * @param groupKeys The set of the columns in the input that are keys.
         * @param groupSets The group sets after being remapped for each parent. This is
         * used for partial aggregation to determine if columns and filters should be
         * relative to a newly generated aggregate or relative to an upcoming aggregate.
         * @param input The input node that is used as the base for the new aggregate.
         * @param parents The list of parents that contain the aggregates to be combined.
         * @return A triple containing the new aggregate, the indices for the new
         * states with each parent and the filter information for the new state with
         * each parent.
         *
         */
        private fun createNewAggregate(
            groupKeys: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>,
            input: RelNode,
            parents: List<Pair<CoveringExpressionState, RelNode>>,
        ): Triple<RelNode, List<List<Int>>, List<Pair<RexNode?, Boolean>>> {
            relBuilder.push(input)
            // Note: You can't hash RelBuilder.AggCall consistently, so we assume the string representations are unique.
            val aggCallsMap = HashMap<String, Pair<AggCall, Int>>()
            parents.forEach {
                val agg = it.second as Aggregate
                val indices = it.first.keptColumns
                agg.aggCallList.forEach { aggCall ->
                    val newArgs = aggCall.argList.map { idx -> relBuilder.field(indices[idx]) }
                    val newCollation =
                        RelCollations.of(
                            aggCall.collation.fieldCollations.map { fieldCollation ->
                                fieldCollation.withFieldIndex(indices[fieldCollation.fieldIndex])
                            },
                        )
                    val aggCall = buildEquivalentAggCall(aggCall, newArgs, newCollation)
                    val aggCallString = aggCall.toString()
                    if (!aggCallsMap.contains(aggCallString)) {
                        val newIdx = aggCallsMap.size
                        aggCallsMap[aggCallString] = Pair(aggCall, newIdx)
                    }
                }
            }
            val newAggCalls = aggCallsMap.values.sortedBy { it.second }.map { it.first }
            relBuilder.aggregate(relBuilder.groupKey(groupKeys), newAggCalls)
            val newAggregate = relBuilder.build()
            val newKeysIndices =
                groupKeys.withIndex().associate {
                    Pair(it.value, it.index)
                }
            val newIndices =
                parents.withIndex().map {
                    val (idx, parent) = it
                    val agg = parent.second as Aggregate
                    val baseNode = parent.first.baseNode
                    val indices = parent.first.keptColumns
                    val finishedAgg = groupKeys == groupSets[idx]
                    val numIndices =
                        if (finishedAgg) {
                            agg.rowType.fieldCount
                        } else {
                            baseNode.rowType.fieldCount
                        }
                    val newIndices = MutableList(numIndices) { -1 }
                    if (finishedAgg) {
                        // Here we have fully finished processing the current aggregate. As a result,
                        // all column indices are relative to the output of the aggregate.
                        agg.groupSet.withIndex().forEach { groupInfo ->
                            val (colIdx, keyIdx) = groupInfo
                            val newKeyIdx = indices[keyIdx]
                            newIndices[colIdx] = newKeysIndices[newKeyIdx]!!
                        }
                        // Push the input again for field generation
                        relBuilder.push(input)
                        agg.aggCallList.withIndex().forEach { callInfo ->
                            val (colIdx, aggCall) = callInfo
                            val newArgs = aggCall.argList.map { idx -> relBuilder.field(indices[idx]) }
                            val newCollation =
                                RelCollations.of(
                                    aggCall.collation.fieldCollations.map { fieldCollation ->
                                        fieldCollation.withFieldIndex(indices[fieldCollation.fieldIndex])
                                    },
                                )
                            val newAggCall = buildEquivalentAggCall(aggCall, newArgs, newCollation)
                            val newAggIdx = aggCallsMap[newAggCall.toString()]!!.second
                            newIndices[colIdx + agg.groupSet.cardinality()] =
                                groupKeys.cardinality() + newAggIdx
                        }
                    } else {
                        // Here we are taking the "partial aggregation" path. That means we still need to
                        // execute the aggregate. As a result, columns need to be place relative to their
                        // position in the input to the aggregate.
                        //
                        // At this point its possible some columns have already been pruned. However, since these
                        // weren't used in either the aggregate or the keys they are safe to remove, and we will
                        // replace them with a dummy -1. If there is any function that cannot be used for partial
                        // aggregation and this assumption becomes invalid, a later step ensures correctness by checking
                        // the aggregate node.
                        agg.groupSet.forEach { keyIdx ->
                            val oldIndex = indices[keyIdx]
                            newIndices[keyIdx] = newKeysIndices[oldIndex]!!
                        }
                        // Push the input again for field generation
                        relBuilder.push(input)
                        // For the aggregate call(s) we currently require that every function can compute
                        // the partial aggregation directly from its result without any necessary remapping.
                        // For example, max, min, sum. As a result, we need to remap every function location
                        // to the original location of its argument. This fundamentally requires both that the
                        // function doesn't require rewriting and that no aggregate produces multiple functions
                        // with the same input.
                        val requiredIndices = mutableSetOf<Int>()
                        // All of these functions can do partial aggregation without any remapping
                        // of the second function.
                        val supportedAggCalls =
                            setOf(
                                SqlStdOperatorTable.SUM,
                                SqlStdOperatorTable.SUM0,
                                SqlStdOperatorTable.MIN,
                                SqlStdOperatorTable.MAX,
                            )
                        agg.aggCallList.forEach { aggCall ->
                            if (!supportedAggCalls.contains(aggCall.aggregation)) {
                                throw IllegalStateException("Internal Error: Unsupported aggregate function for partial aggregation")
                            }
                            val newArgs = aggCall.argList.map { idx -> relBuilder.field(indices[idx]) }
                            val newCollation =
                                RelCollations.of(
                                    aggCall.collation.fieldCollations.map { fieldCollation ->
                                        fieldCollation.withFieldIndex(indices[fieldCollation.fieldIndex])
                                    },
                                )
                            val newAggCall = buildEquivalentAggCall(aggCall, newArgs, newCollation)
                            val newAggIdx = aggCallsMap[newAggCall.toString()]!!.second
                            // Must be exactly 1 argument to allow partial aggregation
                            val arg = aggCall.argList[0]
                            val oldIndex = indices[arg]
                            // Verify that we haven't already seen this argument. If the aggregate requires both
                            // MIN($0) and MAX($0) for example then we can't map both of these functions back to
                            // the original location.
                            if (requiredIndices.contains(oldIndex)) {
                                throw IllegalStateException(
                                    "Internal Error: Cannot partially aggregate with multiple functions on the same column",
                                )
                            }
                            requiredIndices.add(oldIndex)
                            newIndices[oldIndex] = newAggIdx + groupKeys.cardinality()
                        }
                    }
                    newIndices
                }
            val filterInfo =
                parents.map {
                    val filter = it.first.filter
                    if (filter == null) {
                        Pair(null, true)
                    } else {
                        // Check every column used by the filter is found in the new projection map and use
                        // the new index if it is.
                        val usedColumns = RelOptUtil.InputFinder.bits(filter)
                        val indices = MutableList(input.rowType.fieldCount) { -1 }
                        usedColumns.forEach { idx ->
                            if (groupKeys.get(idx)) {
                                indices[idx] = newKeysIndices[idx]!!
                            } else {
                                // We pruned the filter column, so we will need to materialize this
                                // node. In the future we may want to explore keeping the column around.
                                return@map Pair(null, false)
                            }
                        }
                        val mapping = Mappings.source(indices, newAggregate.rowType.fieldCount)
                        val updatedFilter = filter.accept(RexPermuteInputsShuttle(mapping, newAggregate))
                        Pair(updatedFilter, true)
                    }
                }
            return Triple(newAggregate, newIndices, filterInfo)
        }

        /**
         * Builds an aggCall that matches the one passed in but with new arguments.
         * @param aggCall The aggCall to match.
         */
        private fun buildEquivalentAggCall(
            aggCall: AggregateCall,
            newArgs: List<RexNode>,
            newCollation: RelCollation,
        ): AggCall =
            relBuilder
                .aggregateCall(aggCall.aggregation, newArgs)
                .distinct(aggCall.isDistinct)
                .approximate(aggCall.isApproximate)
                .ignoreNulls(aggCall.ignoreNulls())
                .sort(newCollation)

        /**
         * Process a join that is known to be a candidate for caching. This also includes
         * code for handling diverging joins as separate caching groups (and caching the
         * current input). This is currently only implemented for inner joins due to
         * filter limitations, which is checked outside this function.
         * @param cacheRoot The current caching root.
         * @param parents The list of parents all of which have joins as the RelNode above
         */
        private fun processJoin(
            cacheRoot: RelNode,
            parents: List<Pair<CoveringExpressionState, RelNode>>,
        ) {
            // Check that all inputs are on the same side of a join. Strictly speaking this is not
            // 100% required, but it simplifies our work significantly.
            val isLeft = parents.map { (it.second as Join).left == it.first.baseNode }
            val isRight = parents.map { (it.second as Join).right == it.first.baseNode }
            val allLeft = isLeft.all { it }
            val allRight = isRight.all { it }
            // Need to also track if a node is on both sides to protect against corner cases
            val anyLeft = isLeft.any { it }
            val anyRight = isRight.any { it }
            val canCacheAll = (allLeft && !anyRight) || (allRight && !anyLeft)
            if (canCacheAll) {
                val alreadyVisited = parents.map { pausedJoins.contains(it.second) }
                val allVisited = alreadyVisited.all { it }
                val anyVisited = alreadyVisited.any { it }
                if (allVisited) {
                    // We have already visited the other side of this join, so we can process it.
                    val otherSideRoots = parents.map { pausedJoins[it.second]!! }.toSet()
                    if (otherSideRoots.size != 1) {
                        // Cannot cache. There are different caching steps we are aiming to fuse.
                        generateCacheNodes(cacheRoot, parents, parents.size)
                    } else {
                        val otherCacheRoot = otherSideRoots.first()
                        val otherParents = inProgressNodes[otherCacheRoot]!!
                        if (otherParents.size != parents.size) {
                            // Only a subset of the other side joins match. We cannot cache.
                            generateCacheNodes(cacheRoot, parents, parents.size)
                        } else {
                            // We may be able to cache. Now we need to verify that the conditions match.
                            // As a cleanup step we will unify the left and right side for each parent join
                            // as well.
                            val parentMap =
                                parents.withIndex().associate {
                                    Pair(it.value.second, it.index)
                                }
                            val currStates = parents.map { it.first }
                            // Initialize other states with a dummy value
                            val otherStates: MutableList<CoveringExpressionState> =
                                MutableList(parents.size) { otherParents.first().first }
                            otherParents.forEach {
                                val idx = parentMap[it.second]!!
                                otherStates[idx] = it.first
                            }
                            val (leftStates, rightStates) =
                                if (allLeft) {
                                    relBuilder.push(cacheRoot)
                                    relBuilder.push(otherCacheRoot)
                                    Pair(currStates, otherStates)
                                } else {
                                    relBuilder.push(otherCacheRoot)
                                    relBuilder.push(cacheRoot)
                                    Pair(otherStates, currStates)
                                }
                            // Generate the new join conditions to ensure they are all the same.
                            val conditionSet = mutableSetOf<RexNode>()
                            val (rightSourceSize, leftSourceSize) =
                                if (allLeft) {
                                    otherCacheRoot.rowType.fieldCount to cacheRoot.rowType.fieldCount
                                } else {
                                    cacheRoot.rowType.fieldCount to otherCacheRoot.rowType.fieldCount
                                }
                            val newIndices =
                                leftStates.withIndex().map {
                                    val (idx, leftState) = it
                                    val rightState = rightStates[idx]
                                    val join = parents[idx].second as Join
                                    val columnIndices =
                                        leftState.keptColumns +
                                            rightState.keptColumns.map { idx ->
                                                idx + leftSourceSize
                                            }
                                    val targetSize = leftSourceSize + rightSourceSize
                                    val mapping = Mappings.source(columnIndices, targetSize)
                                    val condition =
                                        join.condition.accept(
                                            RexPermuteInputsShuttle(
                                                mapping,
                                                cacheRoot,
                                                otherCacheRoot,
                                            ),
                                        )
                                    conditionSet.add(condition)
                                    columnIndices
                                }
                            if (conditionSet.size != 1) {
                                // Not all conditions match so we can't cache.
                                generateCacheNodes(cacheRoot, parents, parents.size)
                            } else {
                                // Conditions match so we can cache.
                                // Remove the other side from the paused caching.
                                inProgressNodes.remove(otherCacheRoot)
                                val condition = conditionSet.first()
                                val initialRoot = relBuilder.join((parents[0].second as Join).joinType, condition).build()
                                val rebalanceOutput = parents.any { (it.second as BodoPhysicalJoin).rebalanceOutput }
                                val broadcastBuild = parents.any { (it.second as BodoPhysicalJoin).broadcastBuildSide }
                                val newRoot =
                                    if (initialRoot is BodoPhysicalJoin) {
                                        initialRoot.withRebalanceOutput(rebalanceOutput).withBroadcastBuildSide(broadcastBuild)
                                    } else {
                                        initialRoot
                                    }
                                relBuilder.push(newRoot)
                                val filters =
                                    leftStates.withIndex().map {
                                        val (idx, leftState) = it
                                        val rightState = rightStates[idx]
                                        if (rightState.filter == null) {
                                            leftState.filter
                                        } else {
                                            // Remap the right filter.
                                            // TODO: Is there a shorthand for this mapping?
                                            val mapping =
                                                Mappings.create(
                                                    MappingType.PARTIAL_FUNCTION,
                                                    rightSourceSize,
                                                    leftSourceSize + rightSourceSize,
                                                )
                                            for (i in 0 until rightSourceSize) {
                                                mapping.set(i, leftSourceSize + i)
                                            }
                                            val newRightFilter =
                                                rightState.filter.accept(
                                                    RexPermuteInputsShuttle(
                                                        mapping,
                                                        cacheRoot,
                                                        otherCacheRoot,
                                                    ),
                                                )
                                            if (leftState.filter == null) {
                                                newRightFilter
                                            } else {
                                                simplify.simplifyUnknownAsFalse(
                                                    relBuilder.call(
                                                        SqlStdOperatorTable.AND,
                                                        leftState.filter,
                                                        newRightFilter,
                                                    ),
                                                )
                                            }
                                        }
                                    }
                                val newStates =
                                    parents.withIndex().map { (idx, parent) ->
                                        val newBaseNode = parent.second
                                        val keptColumns = newIndices[idx]
                                        val newFilter = filters[idx]
                                        CoveringExpressionState(newBaseNode, keptColumns, newFilter)
                                    }
                                processCaching(newRoot, newStates)
                            }
                        }
                    }
                } else if (anyVisited) {
                    // Some of our joins have been visited, so we can't cache. We will let the
                    // regular cleanup process generate the cache nodes for the other side.
                    generateCacheNodes(cacheRoot, parents, parents.size)
                } else {
                    // We are the first side of the join to be processed, so we need to pause and wait for the other side.
                    parents.forEach {
                        pausedJoins[it.second] = cacheRoot
                    }
                    // Map the cacheRoot to the state.
                    inProgressNodes[cacheRoot] = parents
                }
            } else {
                // Some nodes diverge. As a result, we must generate a cache node here.
                // However, it may still be possible to cache the joins separately after
                // generating the cache node.
                val leftParents = parents.withIndex().filter { isLeft[it.index] }.map { it.value }
                val rightParents = parents.withIndex().filter { isRight[it.index] }.map { it.value }
                if (leftParents.size == 1 && rightParents.size == 1) {
                    // We can't do any caching as there is no group of at least size 2.
                    generateCacheNodes(cacheRoot, parents, parents.size)
                } else {
                    // Due to implementation restrictions, we can't cache unless there is no overlap between the
                    // left and right side. This is because we don't match directly on ordinal position, but on
                    // just parent and child.
                    val (setParents, iterateParents) =
                        if (leftParents.size < rightParents.size) {
                            leftParents to rightParents
                        } else {
                            rightParents to leftParents
                        }
                    val baseNodeSet = setParents.map { it.first.baseNode }.toSet()
                    val hasOverlap =
                        iterateParents.any {
                            baseNodeSet.contains(it.first.baseNode)
                        }
                    if (hasOverlap) {
                        generateCacheNodes(cacheRoot, parents, parents.size)
                    } else {
                        // We can process the left and right sides separately.
                        if (leftParents.size == 1) {
                            val newCacheRoot = generateCacheNodes(cacheRoot, leftParents, 2)
                            // Generate any common filter.
                            // TODO: Also prune columns.
                            val rexBuilder = relBuilder.rexBuilder
                            val trueNode = rexBuilder.makeLiteral(true)
                            val filters =
                                rightParents.map {
                                    if (it.first.filter == null) {
                                        trueNode
                                    } else {
                                        it.first.filter!!
                                    }
                                }
                            val (newRoot, newStates) = applyCommonFiltersUpdateState(newCacheRoot, rightParents, filters, false, false)
                            processJoin(newRoot, rightParents.withIndex().map { Pair(newStates[it.index], it.value.second) })
                        } else if (rightParents.size == 1) {
                            val newCacheRoot = generateCacheNodes(cacheRoot, rightParents, 2)
                            val rexBuilder = relBuilder.rexBuilder
                            val trueNode = rexBuilder.makeLiteral(true)
                            val filters =
                                leftParents.map {
                                    if (it.first.filter == null) {
                                        trueNode
                                    } else {
                                        it.first.filter!!
                                    }
                                }
                            val (newRoot, newStates) = applyCommonFiltersUpdateState(newCacheRoot, leftParents, filters, false, false)
                            processJoin(newRoot, leftParents.withIndex().map { Pair(newStates[it.index], it.value.second) })
                        } else {
                            val newCacheRoot = generateCacheNodes(cacheRoot, listOf(), 2)
                            val rexBuilder = relBuilder.rexBuilder
                            val trueNode = rexBuilder.makeLiteral(true)
                            val leftFilters =
                                leftParents.map {
                                    if (it.first.filter == null) {
                                        trueNode
                                    } else {
                                        it.first.filter!!
                                    }
                                }
                            val (newLeftRoot, newLeftStates) =
                                applyCommonFiltersUpdateState(
                                    newCacheRoot,
                                    leftParents,
                                    leftFilters,
                                    false,
                                    false,
                                )
                            processJoin(newLeftRoot, leftParents.withIndex().map { Pair(newLeftStates[it.index], it.value.second) })
                            val rightFilters =
                                rightParents.map {
                                    if (it.first.filter == null) {
                                        trueNode
                                    } else {
                                        it.first.filter!!
                                    }
                                }
                            val (newRightRoot, newRightStates) =
                                applyCommonFiltersUpdateState(
                                    newCacheRoot,
                                    rightParents,
                                    rightFilters,
                                    false,
                                    false,
                                )
                            processJoin(newRightRoot, rightParents.withIndex().map { Pair(newRightStates[it.index], it.value.second) })
                        }
                    }
                }
            }
        }

        /**
         * Applies the changes necessary to generate a new cache node and list of
         * states by materializing any common filters.
         * @param cacheRoot The current caching root.
         * @param parents The list of parents that are being considered for caching.
         * @param filtersList The list of filters to apply to each parent.
         * @param updateBaseNode Whether to update the base node of the parent. This maps
         * to if we have newly encountered a filter or are simply materializing our filters
         * because we have divided caching into smaller subsets.
         * @param pruneDuplicateUnchangedStates Whether to prune states that have not changed
         * and are repeated multiple times.
         * @return A pair containing the new root and the new states for each parent.
         */
        private fun applyCommonFiltersUpdateState(
            cacheRoot: RelNode,
            parents: List<Pair<CoveringExpressionState, RelNode>>,
            filtersList: List<RexNode>,
            updateBaseNode: Boolean,
            pruneDuplicateUnchangedStates: Boolean,
        ): Pair<RelNode, List<CoveringExpressionState>> {
            val rexBuilder = relBuilder.rexBuilder
            val baseFilter = rexBuilder.makeCall(SqlStdOperatorTable.OR, filtersList)
            val (reorderedFilter, _) =
                FilterRulesCommon.updateConditionsExtractCommon(
                    relBuilder,
                    baseFilter,
                    HashSet(),
                )
            val combinedFilter = simplify.simplifyUnknownAsFalse(reorderedFilter)
            // Generate the new root.
            val changedRoot = !combinedFilter.isAlwaysTrue
            val newRoot =
                if (changedRoot) {
                    relBuilder.push(cacheRoot)
                    relBuilder.filter(combinedFilter)
                    relBuilder.build()
                } else {
                    cacheRoot
                }

            val filterParts = RelOptUtil.conjunctions(combinedFilter)
            val filterPartsAsPredicates = RelOptPredicateList.of(rexBuilder, filterParts)
            val filterSimplifier = simplify.withPredicates(filterPartsAsPredicates)

            // Generate the new state for each node.
            val newStates = mutableListOf<CoveringExpressionState>()
            val seenStates = mutableSetOf<CoveringExpressionState>()
            parents.withIndex().forEach { (idx, parent) ->
                val parentNode = parent.second
                val baseNode =
                    if (updateBaseNode && parentNode is Filter && parentNode !is MinRowNumberFilterBase) {
                        parentNode
                    } else {
                        if (!pruneDuplicateUnchangedStates || !seenStates.contains(parent.first)) {
                            seenStates.add(parent.first)
                            parent.first.baseNode
                        } else {
                            null
                        }
                    }
                if (baseNode != null) {
                    val keptColumns = parent.first.keptColumns
                    // Update state if necessary.
                    val reorderedNodeFilter = filtersList[idx]
                    val newBaseFilter =
                        if (reorderedFilter.isAlwaysTrue) {
                            parent.first.filter
                        } else if (parent.first.filter == null) {
                            reorderedNodeFilter
                        } else {
                            val mergedFilter =
                                rexBuilder.makeCall(SqlStdOperatorTable.AND, parent.first.filter, reorderedNodeFilter)
                            val (reorderedFilter, _) =
                                FilterRulesCommon.updateConditionsExtractCommon(
                                    relBuilder,
                                    mergedFilter,
                                    HashSet(),
                                )
                            filterSimplifier.simplifyUnknownAsFalse(reorderedFilter)
                        }
                    val newFilter =
                        if (newBaseFilter == null || newBaseFilter.isAlwaysTrue) {
                            null
                        } else {
                            filterSimplifier.simplifyUnknownAsFalse(newBaseFilter)
                        }
                    newStates.add(CoveringExpressionState(baseNode, keptColumns, newFilter))
                }
            }
            return Pair(newRoot, newStates)
        }
    }

    /**
     * Data class that represents the state of information to represent a cache
     * target that may need to be materialized.
     */
    private data class CoveringExpressionState(
        val baseNode: RelNode,
        val keptColumns: List<Int>,
        val filter: RexNode?,
    )

    /**
     * Class that will update the plan to insert the caching information from the replacement
     * map.
     * @param replacementMap Maps a node and its parent to the new tree to materialize as a cached
     * result.
     */
    private class CoveringExpressionCacheReplacement(
        private val replacementMap: MutableMap<Pair<RelNode, RelNode>, RelNode>,
    ) : RelShuttleImpl() {
        // Visitor for processing cached node.
        private val cacheVisitor = CachedResultVisitor<RelNode, CachedSubPlanBase> { base -> this.visit(base.cachedPlan.plan) }

        // Store state for using parents as the key.
        private var parent: RelNode? = null

        override fun visit(node: RelNode): RelNode {
            val oldParent = parent
            // Update the parent for recursion
            parent = node
            if (oldParent != null && replacementMap.contains(Pair(node, oldParent))) {
                val replacement = replacementMap[Pair(node, oldParent)]!!
                parent = null
                val result = replacement.accept(this)
                parent = oldParent
                return result.copy(result.traitSet, result.inputs)
            }
            if (node is CachedSubPlanBase) {
                node.cachedPlan.plan = cacheVisitor.visit(node)
            }
            val result = super.visit(node)
            parent = oldParent
            return result
        }
    }

    /**
     * Implement caching across only identical sections of the plan.
     * @param rel The original plan.
     * @return The new plan with identical sections cached.
     */
    private fun runExactMatch(rel: RelNode): RelNode {
        val cluster = rel.cluster
        val finder = CacheCandidateFinder()
        finder.go(rel)
        val cacheNodes = finder.cacheNodes
        return if (cacheNodes.isEmpty()) {
            rel
        } else {
            if (cluster !is BodoRelOptCluster) {
                throw InternalError("Cluster must be a BodoRelOptCluster")
            }
            val visitor = CacheReplacement(cluster, cacheNodes)
            rel.accept(visitor)
        }
    }

    private class CacheCandidateFinder : RelVisitor() {
        // Set of seen IDs
        private val seenNodes = mutableSetOf<Int>()
        val cacheNodes = mutableSetOf<Int>()

        // ~ Methods ----------------------------------------------------------------
        override fun visit(
            node: RelNode,
            ordinal: Int,
            parent: RelNode?,
        ) {
            if (seenNodes.contains(node.id)) {
                // We have seen this node before, so we should cache it.
                // We can only cache Pandas sections as they represent whole operations.
                if (node is BodoPhysicalRel) {
                    // TODO: Add a check for non-deterministic nodes? Right now
                    // we don't have any that we don't allow caching.
                    cacheNodes.add(node.id)
                }
            } else {
                seenNodes.add(node.id)
                node.childrenAccept(this)
            }
        }
    }

    private class CacheReplacement(
        private val cluster: BodoRelOptCluster,
        private val cacheNodes: Set<Int>,
    ) : RelShuttleImpl() {
        // Ensure we only compute each cache node once.
        private val cacheNodeMap = mutableMapOf<Int, CachedSubPlanBase>()

        override fun visit(rel: RelNode): RelNode {
            val id = rel.id
            return if (cacheNodeMap.contains(id)) {
                val result = cacheNodeMap[id]!!
                result.cachedPlan.addConsumer()
                // Create a copy to get separate operator IDs for timing
                result.copy(result.traitSet, listOf())
            } else {
                val children = rel.inputs.map { it.accept(this) }
                val node = rel.copy(rel.traitSet, children)
                if (cacheNodes.contains(id)) {
                    val plan = CachedPlanInfo.create(node, 1)
                    val cachedSubPlan = BodoPhysicalCachedSubPlan.create(plan, cluster.nextCacheId())
                    cacheNodeMap[id] = cachedSubPlan
                    cachedSubPlan
                } else {
                    node
                }
            }
        }
    }

    companion object {
        @JvmStatic
        fun canCacheNode(rel: RelNode): Boolean = rel is BodoPhysicalRel && rel !is BodoPhysicalCachedSubPlan
    }
}
