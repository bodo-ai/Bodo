package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.traits.BatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver

class PandasProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType
) : Project(cluster, traitSet, ImmutableList.of(), input, projects, rowType), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, projects: List<RexNode>, rowType: RelDataType): Project {
        return PandasProject(cluster, traitSet, input, projects, rowType)
    }

    override fun emit(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        inputs: () -> List<Dataframe>
    ): Dataframe {
        TODO("Not yet implemented")
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        val rows = mq.getRowCount(this)
        val cost = projects.map { project -> project.accept(RexCostEstimator) }
            .reduce { l, r -> l.plus(r) as Cost }
        return planner.makeCost(from = cost).multiplyBy(rows)
    }

    companion object {
        fun create(input: RelNode, projects: List<RexNode>, rowType: RelDataType): PandasProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val containsOver = RexOver.containsOver(projects, null)
            val batchProperty = if (containsOver) BatchingProperty.SINGLE_BATCH else BatchingProperty.STREAMING
            val traitSet = cluster.traitSet().replace(PandasRel.CONVENTION).replace(batchProperty)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.project(mq, input, projects)
                }
            return PandasProject(cluster, traitSet, input, projects, rowType)
        }
    }
}
