package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.pandas.RexCostEstimator
import com.bodosql.calcite.application.utils.RexNormalizer
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode

open class ProjectBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
) : Project(cluster, traitSet, hints, input, projects.map { RexNormalizer.normalize(cluster.rexBuilder, it, true) }, rowType, setOf()) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): Project {
        return ProjectBase(cluster, traitSet, hints, input, projects, rowType)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        val cost =
            projects.map { project -> project.accept(RexCostEstimator) }
                .reduce { l, r -> l.plus(r) as Cost }
        return planner.makeCost(from = cost).multiplyBy(rows)
    }
}
