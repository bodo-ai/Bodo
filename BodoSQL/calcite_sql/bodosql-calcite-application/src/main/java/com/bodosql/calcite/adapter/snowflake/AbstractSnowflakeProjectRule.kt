package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexInputRef
import org.immutables.value.Value

/**
 * Abstract rule that enables pushing projects into snowflake. Currently, only projects that simply
 * select a subset of columns are pushed into snowflake.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeProjectRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeProjectRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }
        val (proj, SnowflakeRel) = extractNodes(call)
        val catalogTable = SnowflakeRel.getCatalogTable()

        val newNode = SnowflakeProject.create(
            proj.cluster,
            proj.traitSet,
            SnowflakeRel,
            proj.getProjects(),
            proj.getRowType(),
            catalogTable,
        )
        call.transformTo(newNode)
    }

    private fun extractNodes(call: RelOptRuleCall): Pair<Project, SnowflakeRel> {
        return Pair(call.rel(0), call.rel(2))
    }

    companion object {

        @JvmStatic
        fun isPushableProject(project: Project): Boolean {
            // Only allow pushing projects that only consist of input refs
            val onlyContainsInputRefs = project.projects.map { it is RexInputRef }.all { it }

            return onlyContainsInputRefs
        }
    }

    interface Config : RelRule.Config
}
