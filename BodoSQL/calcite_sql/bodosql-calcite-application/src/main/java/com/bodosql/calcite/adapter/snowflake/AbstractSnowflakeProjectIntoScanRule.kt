package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelOptUtil.InputFinder
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.rules.ProjectRemoveRule
import org.apache.calcite.rex.RexInputRef
import org.immutables.value.Value

/**
 * Take a project that just prunes columns sitting on top of a SnowflakeTableScan
 * and fuses the column pruning directly into the table scan. This can be run either
 * before or after adding SnowflakeProjects for maximum coverage, so it accepts all
 * projects.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeProjectIntoScanRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeProjectIntoScanRule.Config>(config) {
        override fun onMatch(call: RelOptRuleCall?) {
            if (call == null) {
                return
            }
            val (proj, tableScan) = extractNodes(call)
            // Update the used columns in the table scan.
            val usedColumns = InputFinder.bits(proj.projects, null)
            val newTableScan = tableScan.cloneWithProject(usedColumns)
            val relBuilder = call.builder()
            relBuilder.push(newTableScan)
            // Compute the mapping from old index to new index
            val oldUsedColumns = tableScan.keptColumns
            val projMap: MutableMap<Int, Int> = HashMap()
            var newIndex = 0
            for (indexVal in oldUsedColumns.iterator().withIndex()) {
                val oldIndex = indexVal.index
                if (usedColumns.contains(oldIndex)) {
                    projMap[oldIndex] = newIndex
                    newIndex++
                }
            }
            // Generate a new projection. This is needed in case the columns are reordered.
            val newProjects =
                proj.projects.map { x ->
                    val oldIndex = (x as RexInputRef).index
                    val updatedIndex = projMap[oldIndex]!!
                    relBuilder.field(updatedIndex)
                }
            relBuilder.project(newProjects)
            val newProject = relBuilder.build()
            if (newProject is Project &&
                ProjectRemoveRule.isTrivial(newProject) &&
                newTableScan.getRowType().fullTypeString == newProject.getRowType().fullTypeString
            ) {
                // If we generate a trivial projection without renaming/reordering then we can just
                // push the table.
                call.transformTo(newTableScan)
            } else {
                call.transformTo(newProject)
            }
        }

        private fun extractNodes(call: RelOptRuleCall): Pair<Project, SnowflakeTableScan> {
            // Inputs are:
            // Project ->
            //      SnowflakeTableScan
            return Pair(call.rel(0), call.rel(1))
        }

        companion object {
            /**
             * Only apply this rule if the projection is entirely
             * inputRefs and doesn't use all the columns.
             * We may update this in the future.
             */
            @JvmStatic
            fun isApplicable(proj: Project): Boolean {
                // Skip trivial projections
                if (ProjectRemoveRule.isTrivial(proj)) {
                    return false
                }
                val isInputRef = proj.projects.map { x -> x is RexInputRef }
                val containsAllInputRef = isInputRef.all { x -> x }
                // All nodes must be input refs
                if (!containsAllInputRef) {
                    return false
                }
                // Now check that we don't use every column
                val usedColumns = InputFinder.bits(proj.projects, null)
                return usedColumns.cardinality() != proj.input.getRowType().fieldCount
            }
        }

        interface Config : RelRule.Config
    }
