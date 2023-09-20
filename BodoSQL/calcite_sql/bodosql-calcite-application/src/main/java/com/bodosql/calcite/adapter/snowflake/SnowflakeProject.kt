package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.ProjectBase
import com.bodosql.calcite.table.CatalogTableImpl
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.validate.SqlValidatorUtil

/**
 * RelNode that represents a projection that occurs within Snowflake. See SnowflakeRel for more information.
 */
class SnowflakeProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
    private val catalogTable: CatalogTableImpl,
) : ProjectBase(cluster, traitSet, ImmutableList.of(), input, projects, rowType), SnowflakeRel {

    init {
        // SnowflakeProject should always have the Snowflake convention.
        assert(convention == SnowflakeRel.CONVENTION)
    }

    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): Project {
        return SnowflakeProject(cluster, traitSet, input, projects, rowType, catalogTable)
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost(cpu = 0.0, mem = 0.0, rows = rows)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            rowType: RelDataType,
            catalogTable: CatalogTableImpl,
        ): SnowflakeProject {
            val newTraitSet = traitSet.replace(SnowflakeRel.CONVENTION)
            return SnowflakeProject(cluster, newTraitSet, input, projects, rowType, catalogTable)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
            catalogTable: CatalogTableImpl,
        ): SnowflakeProject {
            val rowType = RexUtil.createStructType(
                cluster.typeFactory,
                projects,
                fieldNames,
                SqlValidatorUtil.F_SUGGESTER,
            )
            return create(cluster, traitSet, input, projects, rowType, catalogTable)
        }
    }

    override fun getCatalogTable(): CatalogTableImpl = catalogTable
}
