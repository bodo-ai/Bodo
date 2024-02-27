package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.ProjectBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.validate.SqlValidatorUtil

class BodoLogicalProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
) : ProjectBase(cluster, traitSet, hints, input, projects, rowType) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): Project {
        return BodoLogicalProject(cluster, traitSet, hints, input, projects, rowType)
    }

    companion object {
        @JvmStatic
        fun create(
            input: RelNode,
            hints: List<RelHint>,
            projects: List<RexNode>,
            rowType: RelDataType,
        ): BodoLogicalProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet =
                cluster.traitSet().replace(Convention.NONE)
                    .replaceIfs(RelCollationTraitDef.INSTANCE) {
                        RelMdCollation.project(mq, input, projects)
                    }
            return BodoLogicalProject(cluster, traitSet, hints, input, projects, rowType)
        }

        @JvmStatic
        fun create(
            input: RelNode,
            hints: List<RelHint>,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
        ): BodoLogicalProject {
            val cluster = input.cluster
            val rowType =
                RexUtil.createStructType(
                    cluster.typeFactory,
                    projects,
                    fieldNames,
                    SqlValidatorUtil.F_SUGGESTER,
                )
            return create(input, hints, projects, rowType)
        }
    }
}
