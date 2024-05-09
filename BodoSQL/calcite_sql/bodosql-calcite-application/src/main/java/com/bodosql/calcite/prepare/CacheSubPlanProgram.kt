package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalCachedSubPlan
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.rel.core.CachedPlanInfo
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.Program

/**
 * Find identical sections of the plan and cache them in explicit
 * cache nodes.
 */
class CacheSubPlanProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        val finder = CacheCandidateFinder()
        finder.go(rel)
        val cacheNodes = finder.cacheNodes
        return if (cacheNodes.isEmpty()) {
            rel
        } else {
            val visitor = CacheReplacement(cacheNodes)
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

    private class CacheReplacement(val cacheNodes: Set<Int>) : RelShuttleImpl() {
        // Ensure we only compute each cache node once.
        private val cacheNodeMap = mutableMapOf<Int, CachedSubPlanBase>()
        private var cacheID = 0

        override fun visit(rel: RelNode): RelNode {
            val id = rel.id
            return if (cacheNodeMap.contains(id)) {
                val result = cacheNodeMap[id]!!
                result.cachedPlan.addConsumer()
                result
            } else {
                val children = rel.inputs.map { it.accept(this) }
                val node = rel.copy(rel.traitSet, children)
                if (cacheNodes.contains(id)) {
                    val root = RelRoot.of(node, SqlKind.OTHER)
                    val plan = CachedPlanInfo.create(root, 1)
                    val cachedSubPlan = BodoPhysicalCachedSubPlan.create(plan, cacheID++)
                    cacheNodeMap[id] = cachedSubPlan
                    cachedSubPlan
                } else {
                    node
                }
            }
        }
    }
}
