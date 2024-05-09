package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot

class PandasCachedSubPlan private constructor(cachedPlan: RelRoot, cacheID: Int, cluster: RelOptCluster, traitSet: RelTraitSet) :
    CachedSubPlanBase(
        cachedPlan,
        cacheID,
        cluster,
        traitSet.replace(PandasRel.CONVENTION).replace(cachedPlan.rel.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)),
    ),
    PandasRel {
        override fun copy(
            traitSet: RelTraitSet,
            inputs: List<RelNode>,
        ): PandasCachedSubPlan {
            assert(inputs.isEmpty()) { "PandasCachedSubPlan should not have any inputs" }
            return PandasCachedSubPlan(cachedPlan, cacheID, cluster, traitSet)
        }

        /**
         * Emits the code necessary for implementing this relational operator.
         *
         * @param implementor implementation handler.
         * @return the variable that represents this relational expression.
         */
        override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
            // Note: This depends on streaming and non-streaming having the same implementation.
            return implementor.build {
                val relationalOperatorCache = it.builder().getRelationalOperatorCache()
                if (relationalOperatorCache.isNodeCached(this, cacheID)) {
                    relationalOperatorCache.getCachedTable(this, cacheID)
                } else {
                    val table = implementor.visitChild(cachedPlan.rel, 0)
                    relationalOperatorCache.tryCacheNode(this, cacheID, table)
                    table
                }
            }
        }

        /**
         * Function to create the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
            return UnusedStateVariable
        }

        /**
         * Function to delete the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun deleteStateVariable(
            ctx: PandasRel.BuildContext,
            stateVar: StateVariable,
        ) {
            // Do Nothing
        }

        companion object {
            @JvmStatic
            fun create(
                cachedPlan: RelRoot,
                cacheID: Int,
            ): PandasCachedSubPlan {
                val rootNode = cachedPlan.rel
                return PandasCachedSubPlan(cachedPlan, cacheID, rootNode.cluster, rootNode.traitSet)
            }
        }
    }
