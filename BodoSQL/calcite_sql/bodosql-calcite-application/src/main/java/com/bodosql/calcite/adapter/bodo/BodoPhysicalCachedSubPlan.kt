package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.rel.core.CachedPlanInfo
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode

class BodoPhysicalCachedSubPlan private constructor(
    cachedPlan: CachedPlanInfo,
    cacheID: Int,
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
) :
    CachedSubPlanBase(
            cachedPlan,
            cacheID,
            cluster,
            traitSet.replace(BodoPhysicalRel.CONVENTION).replace(cachedPlan.plan.rel.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)),
        ),
        BodoPhysicalRel {
        override fun copy(
            traitSet: RelTraitSet,
            inputs: List<RelNode>,
        ): BodoPhysicalCachedSubPlan {
            assert(inputs.isEmpty()) { "BodoPhysicalCachedSubPlan should not have any inputs" }
            return BodoPhysicalCachedSubPlan(cachedPlan, cacheID, cluster, traitSet)
        }

        /**
         * Emits the code necessary for implementing this relational operator.
         *
         * @param implementor implementation handler.
         * @return the variable that represents this relational expression.
         */
        override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
            // Note: This depends on streaming and non-streaming having the same implementation.
            return implementor.build {
                val relationalOperatorCache = it.builder().getRelationalOperatorCache()
                if (relationalOperatorCache.isNodeCached(this, cacheID)) {
                    relationalOperatorCache.getCachedTable(this, cacheID)
                } else {
                    val table = implementor.visitChild(cachedPlan.plan.rel, 0)
                    relationalOperatorCache.tryCacheNode(this, cacheID, table)
                    table
                }
            }
        }

        /**
         * Function to create the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
            return UnusedStateVariable
        }

        /**
         * Function to delete the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun deleteStateVariable(
            ctx: BodoPhysicalRel.BuildContext,
            stateVar: StateVariable,
        ) {
            // Do Nothing
        }

        companion object {
            @JvmStatic
            fun create(
                cachedPlan: CachedPlanInfo,
                cacheID: Int,
            ): BodoPhysicalCachedSubPlan {
                val rootNode = cachedPlan.plan.rel
                return BodoPhysicalCachedSubPlan(cachedPlan, cacheID, rootNode.cluster, rootNode.traitSet)
            }
        }
    }
