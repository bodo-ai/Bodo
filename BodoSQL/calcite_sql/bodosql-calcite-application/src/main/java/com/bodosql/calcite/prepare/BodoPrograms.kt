package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRules
import com.bodosql.calcite.adapter.bodo.BodoPhysicalWindow
import com.bodosql.calcite.application.logicalRules.JoinExtractOverRule
import com.bodosql.calcite.application.logicalRules.ListAggOptionalReplaceRule
import com.bodosql.calcite.application.logicalRules.SubQueryRemoveRule.verifyNoSubQueryRemaining
import com.bodosql.calcite.prepare.BodoRules.FieldPushdownRules
import com.bodosql.calcite.prepare.BodoRules.JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE
import com.bodosql.calcite.prepare.BodoRules.MULTI_JOIN_CONSTRUCTION_RULES
import com.bodosql.calcite.prepare.BodoRules.PROJECTION_PULL_UP_RULES
import com.bodosql.calcite.prepare.BodoRules.ProjectionPushdownRules
import com.bodosql.calcite.prepare.BodoRules.SINGLE_VALUE_REMOVE_RULE
import com.bodosql.calcite.prepare.BodoRules.SNOWFLAKE_CLEANUP_RULES
import com.bodosql.calcite.prepare.BodoRules.SNOWFLAKE_PROJECT_CONVERTER_LOCK_RULE
import com.bodosql.calcite.prepare.BodoRules.SUB_QUERY_REMOVAL_RULES
import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import com.bodosql.calcite.rel.logical.BodoLogicalSort
import com.bodosql.calcite.rel.logical.BodoLogicalTableFunctionScan
import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import com.bodosql.calcite.rel.metadata.BodoMetadataRestrictionScan
import com.bodosql.calcite.rel.metadata.BodoRelMetadataProvider
import com.bodosql.calcite.sql2rel.BodoRelDecorrelator
import com.bodosql.calcite.traits.BatchingPropertyPass
import com.google.common.collect.Iterables
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.plan.hep.HepMatchOrder
import org.apache.calcite.plan.hep.HepPlanner
import org.apache.calcite.plan.hep.HepProgramBuilder
import org.apache.calcite.rel.RelHomogeneousShuttle
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.rel.logical.LogicalFilter
import org.apache.calcite.rel.logical.LogicalJoin
import org.apache.calcite.rel.logical.LogicalProject
import org.apache.calcite.rel.logical.LogicalSort
import org.apache.calcite.rel.logical.LogicalTableFunctionScan
import org.apache.calcite.rel.logical.LogicalUnion
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexExecutorImpl
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.sql.SqlExplainFormat
import org.apache.calcite.sql.SqlExplainLevel
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.tools.Program
import org.apache.calcite.tools.Programs

/**
 * Holds a collection of programs for Calcite to produce plans.
 */
object BodoPrograms {
    /**
     * Standard program utilizing the volcano planner to perform optimization and conversion.
     */
    fun standard(optimize: Boolean = true): Program =
        Programs.sequence(
            TrimFieldsProgram(false),
            SnowflakeColumnPruning(),
            // CSE step. Several sections of the parsing calcite integration support may
            // involve directly copying compute when aliases need to be inserted. Depending
            // on the context different sections could be simplified differently, so we want to
            // run our CSE changes up front until we can develop more robust CSE support. After
            // simplification new similarities may be uncovered, so we also include those rules
            // in the simplification step.
            if (optimize) {
                HepOptimizerProgram(BodoRules.CSE_RULES)
            } else {
                NoopProgram
            },
            // Simplification & filter push down step.
            if (optimize) {
                FilterPushdownPass()
            } else {
                NoopProgram
            },
            // We eliminate common subexpressions in filters only after all filters have been pushed down.
            if (optimize) {
                HepOptimizerProgram(listOf(BodoRules.CSE_IN_FILTERS_RULE))
            } else {
                NoopProgram
            },
            // Field pushdown step
            // This includes generic projection pushdown code,
            // and some specialized rules to handle semi-structure field accesses
            if (optimize) {
                FieldPushdownPass()
            } else {
                NoopProgram
            },
            // Projection pull up pass
            if (optimize) {
                ProjectionPullUpPass()
            } else {
                NoopProgram
            },
            // Even when set to 0 bloat, several rules in the pushdown/pull up pass sometimes break CSE incorrectly,
            // the reason for this is not immediately clear.
            // Therefore, we perform a second CSE pass after the pushdown/pull up rules run in order to
            // re-introduce CSE wherever possible
            if (optimize) {
                HepOptimizerProgram(listOf(BodoRules.CSE_IN_FILTERS_RULE))
            } else {
                NoopProgram
            },
            // Rewrite step. The Filter Case changes risk keeping a filter from passing through a join by inserting
            // a projection, so we run it after filter pushdown.
            if (optimize) {
                HepOptimizerProgram(BodoRules.REWRITE_RULES)
            } else {
                NoopProgram
            },
            // Multi Join building step.
            if (optimize) {
                Programs.of(
                    HepProgramBuilder()
                        // Note: You must build the multi-join BOTTOM_UP
                        .addMatchOrder(HepMatchOrder.BOTTOM_UP)
                        .addRuleCollection(MULTI_JOIN_CONSTRUCTION_RULES)
                        .build(),
                    false,
                    BodoRelMetadataProvider(),
                )
            } else {
                NoopProgram
            },
            AnalysisSuite.multiJoinAnalyzer,
            WindowConversionProgram(),
            MetadataPreprocessProgram(),
            RuleSetProgram(
                Iterables.concat(
                    BodoRules.VOLCANO_MINIMAL_RULE_SET,
                    ifTrue(optimize, BodoRules.VOLCANO_OPTIMIZE_RULE_SET),
                ),
            ),
            // This analysis pass has to come after VOLCANO_MINIMAL_RULE_SET which
            // contains the filterPushdown step.
            AnalysisSuite.filterPushdownAnalysis,
            // Convert Window nodes back to Project nodes
            WindowProjectTransformProgram,
            // Cleanup Snowflake Nodes in preparation of Iceberg Conversion
            SnowflakeCleanupProgram,
            // Update Iceberg Nodes
            IcebergConvertProgram,
            TrimFieldsProgram(true),
            // TODO(jsternberg): This can likely be adapted and integrated directly with
            // the VolcanoPlanner, but that hasn't been done so leave this here.
            DecorateAttributesProgram(),
            MergeRelProgram(),
            CacheSubPlanProgram(),
            // Note: No program after this should call builder.join() without
            // special handling for propagating the join keys.
            RuntimeJoinFilterProgram,
            // RTJFs may disrupt column pruning
            TrimFieldsProgram(true),
            BatchingPropertyProgram(),
            // Remove Unused Sarg nodes. This must run after all other programs
            // so no simplification step undoes this conversion.
            SearchArgExpandProgram(),
        )

    /**
     * Preprocessor program to remove subqueries, correlations, and perform other
     * necessary transformations on the plan before the optimization pass.
     */
    fun preprocessor(): Program = PreprocessorProgram()

    /**
     * General optimization program.
     *
     * This will run the set of planner rules that we use to optimize
     * the relational expression before we perform code generation.
     */
    private open class HepOptimizerProgram(ruleSet: Iterable<RelOptRule>, limitMap: Map<RelOptRule, Int> = mapOf()) : Program {
        private val program =
            HepProgramBuilder().also { b ->
                for (rule in ruleSet) {
                    if (limitMap.containsKey(rule)) {
                        // addMatchLimit will apply to all subsequently added rule instances, so we add the limit,
                        // add the rule instance that we want to limit, and then reset the match limit to the default.
                        b.addMatchLimit(
                            limitMap[rule]!!,
                        ).addRuleInstance(rule).addMatchLimit(org.apache.calcite.plan.hep.HepProgram.MATCH_UNTIL_FIXPOINT)
                    } else {
                        b.addRuleInstance(rule)
                    }
                }
            }.build()

        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: List<RelOptMaterialization>?,
            lattices: List<RelOptLattice>?,
        ): RelNode {
            // This is needed because the hep planner will only run rules
            // repeatedly until they don't modify the node anymore, but it
            // will only do this per node rather than as a group.
            //
            // Calcite has functionality through a subprogram to do this
            // repeatedly on a group of rules, but that functionality is broken
            // and a fix is not present at the current moment.
            // See https://issues.apache.org/jira/browse/CALCITE-5561 for details.
            //
            // Another option is to switch to using the VolcanoPlanner which
            // does this automatically. That's probably a better idea than
            // fixing the HepPlanner, but just leaving this as-is until we're
            // able to test the VolcanoPlanner.
            var lastOptimizedPlan = rel
            for (i in 1..25) {
                val curOptimizedPlan = run(lastOptimizedPlan, planner?.context)
                if (curOptimizedPlan.deepEquals(lastOptimizedPlan)) {
                    return lastOptimizedPlan
                }
                lastOptimizedPlan = curOptimizedPlan
            }
            return lastOptimizedPlan
        }

        private fun run(
            rel: RelNode,
            context: Context?,
        ): RelNode {
            val hepPlanner = HepPlanner(program, context)
            rel.cluster.planner.executor = RexExecutorImpl(null)
            hepPlanner.root = rel
            return hepPlanner.findBestExp()
        }
    }

    private class RuleSetProgram(ruleSet: Iterable<RelOptRule>) : Program {
        private val program = Programs.ofRules(ruleSet)

        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: List<RelOptMaterialization>,
            lattices: List<RelOptLattice>,
        ): RelNode {
            return program.run(planner, rel, requiredOutputTraits, materializations, lattices)
        }
    }

    /**
     * This program removes subqueries from the query and then
     * removes the correlation nodes that it produces.
     */
    private class PreprocessorProgram : Program by Programs.sequence(
        // Remove subqueries and correlation nodes from the query.
        SubQueryRemoveProgram(),
        DecorrelateProgram(),
        RewriteProgram(),
        FlattenCaseExpressionsProgram,
        // Convert calcite logical nodes to bodo logical nodes
        // when necessary.
        LogicalConverterProgram,
    )

    /**
     * This program removes subqueries from the query.
     */
    private class SubQueryRemoveProgram : HepOptimizerProgram(SUB_QUERY_REMOVAL_RULES) {
        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: List<RelOptMaterialization>?,
            lattices: List<RelOptLattice>?,
        ): RelNode {
            val result = super.run(planner, rel, requiredOutputTraits, materializations, lattices)
            verifyNoSubQueryRemaining(result)
            return result
        }
    }

    private class RewriteProgram : Program by HepOptimizerProgram(
        // TODO: Move to the normal HEP step and operate on our logical
        // nodes instead of the default.
        listOf(
            JoinExtractOverRule.Config.DEFAULT.toRule(),
            ListAggOptionalReplaceRule.Config.DEFAULT.toRule(),
            SINGLE_VALUE_REMOVE_RULE,
        ),
    )

    /**
     * Simplify Snowflake sections in a way to creates a simpler
     * plan for Iceberg Conversion.
     */
    private object SnowflakeCleanupProgram : Program by HepOptimizerProgram(SNOWFLAKE_CLEANUP_RULES)

    /**
     * The decorrelate program will convert Correlate nodes to
     * equivalent relational expressions usually involving joins.
     *
     * This prevents the use of correlate within the program.
     */
    private class DecorrelateProgram : Program {
        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: List<RelOptMaterialization>?,
            lattices: List<RelOptLattice>?,
        ): RelNode {
            val relBuilder = RelFactories.LOGICAL_BUILDER.create(rel.cluster, null)
            val result = BodoRelDecorrelator.decorrelateQuery(rel, relBuilder)
            BodoRelDecorrelator.verifyNoCorrelationsRemaining(result)
            return result
        }
    }

    private class MetadataPreprocessProgram() : Program {
        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: List<RelOptMaterialization>,
            lattices: List<RelOptLattice>,
        ): RelNode {
            BodoMetadataRestrictionScan.scanForRequestableMetadata(rel)
            return rel
        }
    }

    object LogicalConverterProgram : Program by ShuttleProgram(Visitor) {
        private object Visitor : RelShuttleImpl() {
            override fun visit(project: LogicalProject): RelNode =
                BodoLogicalProject.create(
                    project.input.accept(this),
                    project.hints,
                    project.projects,
                    project.rowType,
                )

            override fun visit(filter: LogicalFilter): RelNode =
                BodoLogicalFilter.create(
                    filter.input.accept(this),
                    filter.condition,
                )

            override fun visit(join: LogicalJoin): RelNode =
                BodoLogicalJoin.create(
                    join.left.accept(this),
                    join.right.accept(this),
                    join.hints,
                    join.condition,
                    join.joinType,
                )

            override fun visit(union: LogicalUnion): RelNode {
                val updatedUnion = this.visitChildren(union)
                return BodoLogicalUnion.create(
                    updatedUnion.inputs,
                    union.all,
                )
            }

            override fun visit(agg: LogicalAggregate): RelNode =
                BodoLogicalAggregate.create(
                    agg.input.accept(this),
                    agg.hints,
                    agg.groupSet,
                    agg.groupSets,
                    agg.aggCallList,
                )

            override fun visit(sort: LogicalSort): RelNode =
                BodoLogicalSort.create(
                    sort.input.accept(this),
                    sort.collation,
                    sort.offset,
                    sort.fetch,
                )

            override fun visit(other: RelNode): RelNode {
                if (other is LogicalTableFunctionScan) {
                    return BodoLogicalTableFunctionScan.create(
                        other.cluster,
                        other.inputs,
                        other.call as RexCall,
                        other.rowType,
                    )
                }
                return super.visit(other)
            }
        }
    }

    object FlattenCaseExpressionsProgram : Program by ShuttleProgram(Visitor) {
        private object Visitor : RelShuttleImpl() {
            override fun visit(filter: LogicalFilter): RelNode {
                val inputRel = filter.input.accept(this)
                val rexBuilder = filter.cluster.rexBuilder
                val visitor = CaseExpressionUnwrapper(rexBuilder)
                val condition = filter.condition.accept(visitor)
                return filter.copy(filter.traitSet, inputRel, condition)
            }

            override fun visit(project: LogicalProject): RelNode {
                val inputRel = project.input.accept(this)
                val rexBuilder = project.cluster.rexBuilder
                val projects = ArrayList<RexNode>()
                val visitor = CaseExpressionUnwrapper(rexBuilder)
                for (rexNode in project.projects) {
                    projects.add(rexNode.accept(visitor))
                }
                return project.copy(project.traitSet, inputRel, projects, project.getRowType())
            }

            override fun visit(join: LogicalJoin): RelNode {
                val left = join.left.accept(this)
                val right = join.right.accept(this)
                val rexBuilder = join.cluster.rexBuilder
                val visitor = CaseExpressionUnwrapper(rexBuilder)
                val condition = join.condition.accept(visitor)
                return join.copy(
                    join.traitSet,
                    condition,
                    left,
                    right,
                    join.joinType,
                    join.isSemiJoinDone,
                )
            }

            class CaseExpressionUnwrapper(private val rexBuilder: RexBuilder) : RexShuttle() {
                override fun visitCall(call: RexCall): RexNode {
                    var foundNestedCase = false
                    if (call.operator === SqlStdOperatorTable.CASE) {
                        /*
                         * Flatten the children to see if we have nested case expressions
                         * Case operands are always 2n + 1, and they are like:
                         * (RexNode -> When expression
                         * RexNode -> Then expression) repeats n times
                         * RexNode -> Else expression
                         */
                        val operands: MutableList<RexNode> = ArrayList(call.getOperands())

                        /*
                         * Flatten all ELSE expressions. Anything nested under ELSE expression can be
                         * pulled up to the parent case. e.g.
                         *
                         * CASE WHEN col1 = 'abc' THEN 0
                         *      WHEN col1 = 'def' THEN 1
                         *      ELSE (CASE WHEN col2 = 'ghi' THEN -1
                         *                 ELSE (CASE WHEN col3 = 'jkl' THEN -2
                         *                            ELSE -3))
                         *
                         * can be rewritten as:
                         * CASE WHEN col1 = 'abc' THEN 0
                         *      WHEN col1 = 'def' THEN 1
                         *      WHEN col2 = 'ghi' THEN -1
                         *      WHEN col3 = 'jkl' THEN -2
                         *      ELSE -3
                         */
                        var unwrapped = true
                        while (unwrapped) { // Recursively unwrap the ELSE expression
                            val elseOperators: MutableList<RexNode> = ArrayList()
                            val elseExpr = operands[operands.size - 1]
                            if (elseExpr is RexCall) {
                                if (elseExpr.operator === SqlStdOperatorTable.CASE) {
                                    foundNestedCase = true
                                    elseOperators.addAll(elseExpr.getOperands())
                                }
                            }
                            if (elseOperators.isEmpty()) {
                                unwrapped = false
                            } else {
                                operands.removeAt(operands.size - 1) // Remove the ELSE expression and replace with the unwrapped one
                                operands.addAll(elseOperators)
                            }
                        }
                        if (foundNestedCase) {
                            return rexBuilder.makeCall(SqlStdOperatorTable.CASE, operands)
                        }
                    }
                    return super.visitCall(call)
                }
            }
        }
    }

    /**
     * Decorate Attributes Program.
     *
     * This will run the set of planner rules that won't change any of
     * the orderings for the final plan but may decorate the nodes with additional
     * information based on the contents of that node or the surrounding nodes.
     *
     * This information can then be utilized by the runtime stage to enable specific
     * runtime checks.
     */
    private class DecorateAttributesProgram : Program by Programs.hep(
        listOf(BodoPhysicalRules.BODO_PHYSICAL_JOIN_REBALANCE_OUTPUT_RULE),
        false,
        DefaultRelMetadataProvider.INSTANCE,
    )

    /**
     * Merges relational nodes that have identical digests.
     *
     * The calcite planners will generally try to merge nodes that are identical using
     * the digest, but it sometimes doesn't succeed. This program iterates through the
     * plan and finds nodes with identical digests and replaces duplicates with each other.
     *
     * This is generally performed as a final step before code generation to allow
     * for the caching code to work properly.
     */
    private class MergeRelProgram : Program by ShuttleProgram(Visitor()) {
        private class Visitor : RelHomogeneousShuttle() {
            val mapDigestToRel: HashMap<String, RelNode> = hashMapOf()

            override fun visit(other: RelNode): RelNode {
                val node = visitChildren(other)
                return mapDigestToRel.computeIfAbsent(node.digest) { node }
            }
        }
    }

    // Converts RelNodes containing RexOver calls to a sequence of
    // Window RelNodes, possibly interleaved with Project nodes.
    private class WindowConversionProgram : Program by HepOptimizerProgram(
        BodoRules.WINDOW_CONVERSION_RULES,
    )

    // Converts BodoPhysicalWindow rel nodes back to BodoPhysicalProject for codegen purposes
    object WindowProjectTransformProgram : Program by ShuttleProgram(Visitor) {
        private object Visitor : RelShuttleImpl() {
            override fun visit(other: RelNode): RelNode =
                if (other is BodoPhysicalWindow) {
                    other.convertToProject().accept(this)
                } else {
                    super.visit(other)
                }
        }
    }

    private class ProjectionPullUpPass : Program by HepOptimizerProgram(
        PROJECTION_PULL_UP_RULES,
    )

    private class FilterPushdownPass : Program by HepOptimizerProgram(
        Iterables.concat(BodoRules.FILTER_PUSH_DOWN_RULES, BodoRules.SIMPLIFICATION_RULES),
        mapOf(
            Pair(com.bodosql.calcite.prepare.BodoRules.JOIN_PUSH_TRANSITIVE_PREDICATES, 100),
            Pair(
                JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE,
                100,
            ),
        ),
    )

    private class FieldPushdownPass : Program by HepOptimizerProgram(
        ProjectionPushdownRules.plus(FieldPushdownRules),
    )

    private class SnowflakeColumnPruning : Program by Programs.hep(
        listOf(SNOWFLAKE_PROJECT_CONVERTER_LOCK_RULE),
        true,
        DefaultRelMetadataProvider.INSTANCE,
    )

    private class TrimFieldsProgram(private val isPhysical: Boolean) : Program {
        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: List<RelOptMaterialization>,
            lattices: List<RelOptLattice>,
        ): RelNode {
            val relBuilder =
                if (isPhysical) {
                    val physicalBuilder =
                        com.bodosql.calcite.rel.core.BodoPhysicalRelFactories.BODO_PHYSICAL_BUILDER.create(
                            rel.cluster,
                            null,
                        )
                    physicalBuilder.transform { t -> t.withBloat(-1) }
                } else {
                    com.bodosql.calcite.rel.core.BodoLogicalRelFactories.BODO_LOGICAL_BUILDER.create(rel.cluster, null)
                }
            return BodoRelFieldTrimmer(null, relBuilder, isPhysical).trim(rel)
        }
    }

    private class BatchingPropertyProgram() : Program {
        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: MutableList<RelOptMaterialization>,
            lattices: MutableList<RelOptLattice>,
        ): RelNode {
            val physicalBuilder = com.bodosql.calcite.rel.core.BodoPhysicalRelFactories.BODO_PHYSICAL_BUILDER.create(rel.cluster, null)
            val builder = physicalBuilder.transform { t -> t.withBloat(-1) }
            return BatchingPropertyPass.applyBatchingInfo(rel, builder)
        }
    }

    private object NoopProgram : Program {
        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: List<RelOptMaterialization>,
            lattices: List<RelOptLattice>,
        ): RelNode = rel
    }

    /**
     * Simple program that does nothing but dump the output to stdout.
     * Should only be used for debugging
     */
    class PrintDebugProgram(private val prefixMessage: String = "") : Program {
        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: MutableList<RelOptMaterialization>?,
            lattices: MutableList<RelOptLattice>?,
        ): RelNode {
            println(RelOptUtil.dumpPlan(prefixMessage, rel, SqlExplainFormat.TEXT, SqlExplainLevel.NON_COST_ATTRIBUTES))
            return rel
        }
    }
}

/**
 * Return the passed in iterable on true otherwise return an empty iterable.
 */
private fun <T> ifTrue(
    condition: Boolean,
    onTrue: Iterable<T>,
): Iterable<T> =
    if (condition) {
        onTrue
    } else {
        listOf()
    }
