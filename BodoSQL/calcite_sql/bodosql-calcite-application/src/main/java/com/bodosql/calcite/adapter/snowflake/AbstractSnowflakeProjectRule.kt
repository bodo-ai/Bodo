package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.VariantSqlType
import org.apache.calcite.tools.RelBuilder
import org.immutables.value.Value

/**
 * Abstract rule that enables pushing projects into snowflake. Currently, only projects that simply
 * select a subset of columns are pushed into snowflake.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
abstract class AbstractSnowflakeProjectRule protected constructor(config: Config) :
    RelRule<AbstractSnowflakeProjectRule.Config>(config) {
        /**
         * Take a projection that contains some operations from pure column accesses and generates equivalent
         * nodes that consist of two parts:
         * 1. A LogicalProjection equivalent to the original projection with any pushed nodes replaced with
         *    "re-indexed" references to the next part.
         * 2. A Snowflake Project contains only RexNodes that are deemed push-able into Snowflake. See
         *    PushableDetectionVisitor for a description of what types of RexNodes can be pushed.
         *
         * This is done, so the LogicalProjection keeps any function calls / window functions
         * with the column selection is separated and pushed into Snowflake.
         */
        private fun splitComputeProjection(
            project: Project,
            snowflakeRel: SnowflakeRel,
            relBuilder: RelBuilder,
            pushableNodes: Set<RexNode>,
        ): RelNode {
            // After we determine what columns are used, we use that to build a SnowflakeProjection that just
            // prunes columns and any other nodes that can be pushed. The set of pushable nodes is sorted so that
            // input refs come first, and they are in the same order that they were originally. This sort order needs
            // to be a deterministic total ordering in order to avoid issues during testing.
            val usedColumns =
                pushableNodes.toList().sortedWith(
                    compareBy<RexNode> {
                        if (it is RexInputRef) {
                            0
                        } else {
                            1
                        }
                    }.thenBy {
                        if (it is RexInputRef) {
                            it.index
                        } else {
                            it.toString()
                        }
                    },
                )
            relBuilder.push(snowflakeRel)
            val nodeRemap = mutableMapOf<RexNode, Int>()
            var nextIndex = 0
            // Remap the field indices in the current projection to the indices
            // after pruning columns.
            usedColumns.forEach {
                    node ->
                nodeRemap[node] = nextIndex
                nextIndex++
            }
            val fieldSelects = nodeRemap.map { entry -> entry.key }
            val fieldNames =
                nodeRemap.map { entry ->
                    if (entry.key is RexInputRef) {
                        snowflakeRel.rowType.fieldNames[(entry.key as RexInputRef).index]
                    } else {
                        "\$EXPR${entry.value}"
                    }
                }
            // Next we create the snowflake project from the logical project.
            val catalogTable = snowflakeRel.getCatalogTable()
            val snowflakeProject =
                SnowflakeProject.create(
                    project.cluster,
                    project.traitSet,
                    snowflakeRel,
                    fieldSelects,
                    fieldNames,
                    catalogTable,
                )
            relBuilder.push(snowflakeProject)
            // Now we need to generate the final projection by replacing all the
            // RexNodes that were pushable with references to the snowflake project.
            val visitor = SplitUpdate(nodeRemap, relBuilder)
            val newProjects = project.projects.map { x -> x.accept(visitor) }
            val finalProject =
                BodoLogicalProject.create(
                    snowflakeProject,
                    project.hints,
                    newProjects,
                    // The type shouldn't change at all.
                    project.getRowType(),
                )
            return finalProject
        }

        /**
         * RexShuttle implementation that replaces RexNodes with references to a column
         * from the newly created SnowflakeProject node. If a section of the RexNode
         * can avoid being changed it returns the original RexNode.
         *
         * All other changes are methods for ensuring the children are properly traversed
         * and all nodes are updated.
         *
         * Currently, only supports replacing input refs and calls.
         */

        private class SplitUpdate constructor(
            private val rexMap: Map<RexNode, Int>,
            private val builder: RelBuilder,
        ) :
            RexShuttle() {
                override fun visitInputRef(inputRef: RexInputRef): RexNode {
                    if (rexMap.containsKey(inputRef)) {
                        return builder.field(rexMap[inputRef]!!)
                    }
                    // Avoid creating new nodes when possible
                    return inputRef
                }

                override fun visitCall(call: RexCall): RexNode {
                    if (rexMap.containsKey(call)) {
                        return builder.field(rexMap[call]!!)
                    }
                    return super.visitCall(call)
                }
            }

        /**
         * Transforms a projection that exists entirely of nodes that can be pushed to Snowflake
         * into a SnowflakeProject. This is basically trivial to implement as we do not need to
         * modify the projection at all.
         */
        private fun createSnowflakeProjection(
            project: Project,
            snowflakeRel: SnowflakeRel,
        ): RelNode {
            val catalogTable = snowflakeRel.getCatalogTable()
            return SnowflakeProject.create(
                project.cluster,
                project.traitSet,
                snowflakeRel,
                project.projects,
                project.getRowType(),
                catalogTable,
            )
        }

        override fun onMatch(call: RelOptRuleCall) {
            val (proj, snowflakeRel) = extractNodes(call)
            val pushableNodes = getPushableNodes(proj)
            // If every node in the project is pushable, just turn the BodoPhysicalProject
            // into a SnowflakeProject, otherwise split it off into two.
            val newNode =
                if (pushableNodes.containsAll(proj.projects)) {
                    createSnowflakeProjection(proj, snowflakeRel)
                } else {
                    splitComputeProjection(proj, snowflakeRel, call.builder(), pushableNodes)
                }
            call.transformTo(newNode)
            // New plan is absolutely better than old plan.
            call.planner.prune(proj)
        }

        private fun extractNodes(call: RelOptRuleCall): Pair<Project, SnowflakeRel> {
            return when (call.rels.size) {
                // Inputs are:
                // Project ->
                //     SnowflakeToBodoPhysicalConverter ->
                //         SnowflakeRel
                3 -> Pair(call.rel(0), call.rel(2))
                // Inputs are:
                // Project ->
                //     SnowflakeRel
                else -> Pair(call.rel(0), call.rel(1))
            }
        }

        companion object {
            private val SUPPORTED_CALLS =
                setOf(
                    "CAST",
                    "GET_PATH",
                )

            // Returns whether a rex node can be pushed into Snowflake.
            fun isPushableNode(node: RexNode): Boolean {
                if (node is RexInputRef) return true
                if (node is RexCall) {
                    return isSupportedCall(node)
                }
                return false
            }

            /** Calls that can be projected by Snowflake.
             * Currently only
             *  CAST(Variant) (from variant to any datatype)
             * @param call
             * @return true/false based on whether it's a supported operation or not.
             */
            private fun isSupportedCall(call: RexCall): Boolean {
                return SUPPORTED_CALLS.contains(call.operator.name) && checkOperandsPushable(call)
            }

            /** For a given rexCall, checks the operands that are needed to check if the call can be pushed
             * @param call: Operator call
             * @return Boolean true/false if the operands allow for the call to be pushed
             */
            private fun checkOperandsPushable(call: RexCall): Boolean {
                if (call.kind == SqlKind.CAST) {
                    return isVariantPushableNode(call.operands[0])
                } else if (call.operator.name == "GET_PATH") {
                    return isPushableNode(call.operands[0])
                } else {
                    return call.operands.all { node -> isPushableNode(node) }
                }
            }

            /** Check whether node can be pushed or not
             * Node must be variant and is itself a pushable node
             * @param call: Operator call
             * @return true/false
             */
            private fun isVariantPushableNode(node: RexNode): Boolean {
                return (node.type is VariantSqlType && isPushableNode(node))
            }

            /**
             * Class to detect all nodes that can be pushed into a SnowflakeProject.
             *
             * Currently only allows the following to be considered pushable nodes:
             *
             * - Input refs
             * - Calls to GET_PATH (where the first operand is also a pushable node)
             * - Variant CAST calls
             */
            private class PushableDetectionVisitor(private val pushableNodes: MutableSet<RexNode>) : RexVisitorImpl<Unit>(true) {
                override fun visitInputRef(inputRef: RexInputRef) {
                    pushableNodes.add(inputRef)
                }

                override fun visitCall(call: RexCall) {
                    if (isPushableNode(call)) {
                        pushableNodes.add(call)
                    } else {
                        call.operands.forEach { it.accept(this) }
                    }
                }
            }

            /**
             * Determine the input columns that are directly referenced by a projection.
             */
            @JvmStatic
            fun getPushableNodes(project: Project): Set<RexNode> {
                return getPushableNodes(project.projects)
            }

            /**
             * Determine if any of the nodes passed in are pushable
             */
            @JvmStatic
            fun getPushableNodes(nodes: Iterable<RexNode>): Set<RexNode> {
                val mutableSet = HashSet<RexNode>()
                val visitor = PushableDetectionVisitor(mutableSet)
                for (node in nodes) {
                    node.accept(visitor)
                }
                return mutableSet
            }

            /**
             * A projection is pushable/has a pushable component if it doesn't use
             * every column from the input type but uses at least 1 column.
             *
             * TODO(njriasan): Is this check cheap enough? Should we be doing a cheaper check?
             */
            @JvmStatic
            fun isPushableProject(project: Project): Boolean {
                val pushableNodes = getPushableNodes(project)
                return pushableNodes.isNotEmpty()
            }

            @JvmStatic
            fun canPushNonTrivialSnowflakeProjectAndProjectNotAlreadySnowflake(project: Project): Boolean {
                if (project is SnowflakeProject) return false
                val pushableNodes = getPushableNodes(project)
                return pushableNodes.filter { node -> node !is RexInputRef }.isNotEmpty()
            }
        }

        interface Config : RelRule.Config
    }
