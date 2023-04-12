package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode

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

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(input: RelNode, projects: List<RexNode>, rowType: RelDataType): PandasProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSet().replace(PandasRel.CONVENTION)
                .replaceIfs(
                    RelCollationTraitDef.INSTANCE,
                    {RelMdCollation.project(mq, input, projects)})
            return PandasProject(cluster, traitSet, input, projects, rowType)
        }
    }
}
