package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.pandas.PandasRules
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.application.logicalRules.JoinExtractOverRule
import com.bodosql.calcite.application.logicalRules.ListAggOptionalReplaceRule
import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import com.bodosql.calcite.rel.logical.BodoLogicalSort
import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import com.bodosql.calcite.rel.metadata.BodoRelMetadataProvider
import com.google.common.collect.Iterables
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.plan.hep.HepMatchOrder
import org.apache.calcite.plan.hep.HepPlanner
import org.apache.calcite.plan.hep.HepProgram
import org.apache.calcite.plan.hep.HepProgramBuilder
import org.apache.calcite.rel.RelHomogeneousShuttle
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttle
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.rel.logical.LogicalFilter
import org.apache.calcite.rel.logical.LogicalJoin
import org.apache.calcite.rel.logical.LogicalProject
import org.apache.calcite.rel.logical.LogicalSort
import org.apache.calcite.rel.logical.LogicalUnion
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider
import org.apache.calcite.rel.rules.SubQueryRemoveRule
import org.apache.calcite.rex.RexExecutorImpl
import org.apache.calcite.sql2rel.RelDecorrelator
import org.apache.calcite.tools.Program
import org.apache.calcite.tools.Programs

/**
 * Holds a collection of programs for Calcite to produce plans.
 */
object BodoPrograms {
    /**
     * Uses the heuristic planner to perform optimizations using the default
     * rule set and then utilizes the volcano planner to assign traits
     * and convert logical nodes to physical nodes.
     */
    fun hepStandard(optimize: Boolean = true): Program = Programs.sequence(
        if (optimize) {
            HepOptimizerProgram(BodoRules.HEURISTIC_RULE_SET)
        } else {
            NoopProgram
        },
        SnowflakeTraitAdder(),
        // Includes minimal set of rules to produce a valid plan.
        // This is a subset of the heuristic rule set.
        RuleSetProgram(BodoRules.VOLCANO_MINIMAL_RULE_SET),
        DecorateAttributesProgram(),
        MergeRelProgram(),
    )

    /**
     * Standard program utilizing the volcano planner to perform optimization and conversion.
     */
    fun standard(optimize: Boolean = true): Program = Programs.sequence(
        // When the HepStandardProgram is removed entirely, we would add the
        // convention when the SnowflakeTableScan is created instead of here.
        SnowflakeTraitAdder(),
        if (optimize) {
            Programs.of(
                HepProgramBuilder()
                    .addRuleInstance(BodoRules.FILTER_INTO_JOIN_RULE)
                    .addMatchOrder(HepMatchOrder.BOTTOM_UP)
                    .addRuleInstance(BodoRules.JOIN_TO_MULTI_JOIN)
                    .build(),
                false,
                BodoRelMetadataProvider(),
            )
        } else {
            NoopProgram
        },
        RuleSetProgram(
            Iterables.concat(
                BodoRules.VOLCANO_MINIMAL_RULE_SET,
                ifTrue(optimize, BodoRules.VOLCANO_OPTIMIZE_RULE_SET),
            ),
        ),
        // TODO(jsternberg): This can likely be adapted and integrated directly with
        // the VolcanoPlanner, but that hasn't been done so leave this here.
        DecorateAttributesProgram(),
        MergeRelProgram(),
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
    private class HepOptimizerProgram(ruleSet: Iterable<RelOptRule>) : Program {
        private val program = HepProgramBuilder().also { b ->
            for (rule in ruleSet) {
                b.addRuleInstance(rule)
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

        private fun run(rel: RelNode, context: Context?): RelNode {
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
            val metadataProvider = BodoRelMetadataProvider()
            rel.cluster.invalidateMetadataQuery()
            rel.cluster.metadataProvider = metadataProvider
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
        // Convert calcite logical nodes to bodo logical nodes
        // when necessary.
        LogicalConverterProgram,
    )

    /**
     * This program removes subqueries from the query.
     */
    private class SubQueryRemoveProgram : Program by Programs.sequence(
        // Remove subqueries and convert to correlations when necessary.
        Programs.of(
            HepProgram.builder()
                .addRuleCollection(
                    listOf(
                        SubQueryRemoveRule.Config.FILTER.toRule(),
                        SubQueryRemoveRule.Config.PROJECT.toRule(),
                        SubQueryRemoveRule.Config.JOIN.toRule(),
                        JoinExtractOverRule.Config.DEFAULT.toRule(),
                        ListAggOptionalReplaceRule.Config.DEFAULT.toRule(),
                    ),
                )
                .build(),
            true,
            BodoRelMetadataProvider(),
        ),
        // Remove correlations.
        DecorrelateProgram(),
    )

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
            return RelDecorrelator.decorrelateQuery(rel, relBuilder)
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
        listOf(PandasRules.PANDAS_JOIN_STREAMING_REBALANCE_OUTPUT_RULE, PandasRules.PANDAS_JOIN_BATCH_REBALANCE_OUTPUT_RULE),
        true,
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

    /**
     * Adds SnowflakeRel.CONVENTION to any SnowflakeTableScan nodes.
     * See the comment in SnowflakeTableScan about why this is needed.
     */
    class SnowflakeTraitAdder : Program by ShuttleProgram(Visitor) {
        private object Visitor : RelShuttleImpl() {
            override fun visit(scan: TableScan): RelNode {
                return when (scan) {
                    is SnowflakeTableScan -> scan.copy(scan.traitSet.replace(SnowflakeRel.CONVENTION), scan.inputs)
                    else -> super.visit(scan)
                }
            }
        }
    }

    private class ShuttleProgram(private val shuttle: RelShuttle) : Program {
        override fun run(
            planner: RelOptPlanner,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet,
            materializations: MutableList<RelOptMaterialization>,
            lattices: MutableList<RelOptLattice>,
        ): RelNode = rel.accept(shuttle)
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
}

/**
 * Return the passed in iterable on true otherwise return an empty iterable.
 */
private fun <T> ifTrue(condition: Boolean, onTrue: Iterable<T>): Iterable<T> =
    if (condition) {
        onTrue
    } else {
        listOf()
    }
