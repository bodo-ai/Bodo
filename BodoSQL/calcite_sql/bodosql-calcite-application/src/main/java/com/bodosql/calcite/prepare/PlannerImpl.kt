/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is a derivative work of the PlannerImpl in the core calcite
 * project located here: https://github.com/apache/calcite/blob/main/core/src/main/java/org/apache/calcite/prepare/PlannerImpl.java.
 *
 * It has been modified for Bodo purposes. As this is a derivative work,
 * the license has been retained above.
 */
package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregateRule
import com.bodosql.calcite.application.BodoSQLOperatorTables.*
import com.bodosql.calcite.application.bodo_sql_rules.*
import com.bodosql.calcite.sql.parser.SqlBodoParserImpl
import com.google.common.collect.ImmutableList
import org.apache.calcite.avatica.util.Casing
import org.apache.calcite.avatica.util.Quoting
import org.apache.calcite.config.NullCollation
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.plan.*
import org.apache.calcite.plan.hep.HepPlanner
import org.apache.calcite.plan.hep.HepProgramBuilder
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider
import org.apache.calcite.rel.rules.*
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexExecutorImpl
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.sql.util.SqlOperatorTables
import org.apache.calcite.sql.validate.SqlConformanceEnum
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.sql2rel.RelDecorrelator
import org.apache.calcite.sql2rel.SqlToRelConverter
import org.apache.calcite.sql2rel.StandardConvertletTable
import org.apache.calcite.sql2rel.StandardConvertletTableConfig
import org.apache.calcite.tools.*

class PlannerImpl(config: Config) : AbstractPlannerImpl(frameworkConfig(config)) {
    private val defaultSchemas = config.defaultSchemas
    private val namedParamTableName = config.namedParamTableName

    companion object {
        private fun frameworkConfig(config: Config): FrameworkConfig {
            return Frameworks.newConfigBuilder()
                .operatorTable(
                    SqlOperatorTables.chain(
                        SqlStdOperatorTable.instance(),
                        DatetimeOperatorTable.instance(),
                        NumericOperatorTable.instance(),
                        StringOperatorTable.instance(),
                        JsonOperatorTable.instance(),
                        CondOperatorTable.instance(),
                        SinceEpochFnTable.instance(),
                        ThreeOperatorStringTable.instance(),
                        CastingOperatorTable.instance(),
                    )
                )
                .typeSystem(config.typeSystem)
                .sqlToRelConverterConfig(
                    SqlToRelConverter.config()
                        .withExpand(false)
                        .withInSubQueryThreshold(Integer.MAX_VALUE))
                .parserConfig(
                    SqlParser.Config.DEFAULT
                        .withCaseSensitive(false)
                        .withQuoting(Quoting.BACK_TICK)
                        .withQuotedCasing(Casing.UNCHANGED)
                        .withUnquotedCasing(Casing.UNCHANGED)
                        .withConformance(SqlConformanceEnum.LENIENT)
                        .withParserFactory(SqlBodoParserImpl.FACTORY))
                .convertletTable(
                    StandardConvertletTable(StandardConvertletTableConfig(false, false)))
                .sqlValidatorConfig(
                    SqlValidator.Config.DEFAULT
                        .withNamedParamTableName(config.namedParamTableName)
                        .withDefaultNullCollation(NullCollation.LOW)
                        .withCallRewrite(false))
                .programs(
                    // We create a new program each time we construct a new planner.
                    // This is because calcite 1.30.0's hep program is not threadsafe
                    // so we're just going to be careful and not use a singleton.
                    //
                    // Version 1.32.0 and beyond has fixed this and it's no longer
                    // necessary there.
                    SubQueryRemoveProgram(),
                    OptimizerProgram(),
                )
                .build()
        }

        private fun rootSchema(schema: SchemaPlus): SchemaPlus {
            var currSchema = schema
            while (true) {
                val parentSchema = currSchema.parentSchema ?: return currSchema
                currSchema = parentSchema
            }
        }
    }

    override fun createCatalogReader(): CalciteCatalogReader {
        val rootSchema = rootSchema(defaultSchemas[0])
        val defaultSchemaPaths: ImmutableList.Builder<List<String>> = ImmutableList.builder()
        for (schema in defaultSchemas) {
            defaultSchemaPaths.add(CalciteSchema.from(schema).path(null))
        }
        defaultSchemaPaths.add(listOf())
        return BodoCatalogReader(
            CalciteSchema.from(rootSchema),
            defaultSchemaPaths.build(),
            typeFactory, connectionConfig,
        )
    }

    class Config(
        val defaultSchemas: List<SchemaPlus>,
        val typeSystem: RelDataTypeSystem,
        val namedParamTableName: String,
    )

    /**
     * This program removes subqueries from the query and then
     * removes the correlation nodes that it produces.
     */
    private class SubQueryRemoveProgram : Program {
        private val program = Programs.sequence(
            Programs.hep(
                listOf(
                    SubQueryRemoveRule.Config.FILTER.toRule(),
                    SubQueryRemoveRule.Config.PROJECT.toRule(),
                    SubQueryRemoveRule.Config.JOIN.toRule(),
                    JoinExtractOverRule.Config.DEFAULT.toRule(),
                ),
                false, DefaultRelMetadataProvider.INSTANCE
            ),
            DecorrelateProgram(),
        )

        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: List<RelOptMaterialization>?,
            lattices: List<RelOptLattice>?
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
                val curOptimizedPlan = program.run(planner, lastOptimizedPlan, requiredOutputTraits, materializations, lattices)
                if (curOptimizedPlan.deepEquals(lastOptimizedPlan)) {
                    return lastOptimizedPlan
                }
                lastOptimizedPlan = curOptimizedPlan
            }
            return lastOptimizedPlan
        }
    }

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
            lattices: List<RelOptLattice>?
        ): RelNode {
            val relBuilder = RelFactories.LOGICAL_BUILDER.create(rel.cluster, null)
            return RelDecorrelator.decorrelateQuery(rel, relBuilder)
        }

    }

    /**
     * General optimization program.
     *
     * This will run the set of planner rules that we use to optimize
     * the relational expression before we perform code generation.
     */
    private class OptimizerProgram : Program {
        private val program = HepProgramBuilder()
            /*
            Planner rule that, given a Project node that merely returns its input, converts the node into its child.
            */
            .addRuleInstance(ProjectUnaliasedRemoveRule.Config.DEFAULT.toRule())
            /*
            Planner rule that combines two LogicalFilters.
            */
            .addRuleInstance(FilterMergeRuleNoWindow.Config.DEFAULT.toRule())
            /*
               Planner rule that merges a Project into another Project,
               provided the projects aren't projecting identical sets of input references
               and don't have any dependencies.
            */
            .addRuleInstance(DependencyCheckingProjectMergeRule.Config.DEFAULT.toRule())
            /*
            Planner rule that pushes a Filter past a Aggregate.
            */
            .addRuleInstance(FilterAggregateTransposeRuleNoWindow.Config.DEFAULT.toRule())
            /*
             * Planner rule that matches an {@link org.apache.calcite.rel.core.Aggregate}
             * on a {@link org.apache.calcite.rel.core.Join} and removes the join
             * provided that the join is a left join or right join and it computes no
             * aggregate functions or all the aggregate calls have distinct.
             *
             * <p>For instance,</p>
             *
             * <blockquote>
             * <pre>select distinct s.product_id from
             * sales as s
             * left join product as p
             * on s.product_id = p.product_id</pre></blockquote>
             *
             * <p>becomes
             *
             * <blockquote>
             * <pre>select distinct s.product_id from sales as s</pre></blockquote>
             */
            .addRuleInstance(AggregateJoinRemoveRule.Config.DEFAULT.toRule())
            /*
            Planner rule that pushes an Aggregate past a join
            */
            .addRuleInstance(AggregateJoinTransposeRule.Config.EXTENDED.toRule())
            /*
            Rule that tries to push filter expressions into a join condition and into the inputs of the join.
            */
            .addRuleInstance(FilterJoinRuleNoWindow.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT.toRule())
            /*
            Rule that applies moves any filters that depend on a single table before the join in
            which they occur.
            */
            .addRuleInstance(FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT.toRule())
            /*
            Filters tables for unused columns before join.
            */
            .addRuleInstance(AliasPreservingProjectJoinTransposeRule.Config.DEFAULT.toRule())
            /*
            This reduces expressions inside of the conditions of filter statements.
            Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE)
            TODO(Ritwika: figure out SEARCH handling later. SARG attributes do not have public access methods.
            */
            .addRuleInstance(BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT.toRule())
            // Simplify constant expressions inside a Projection. Ex condition($0 = 1 and $0 = 2)
            // ==> condition(FALSE)
            .addRuleInstance(BodoSQLReduceExpressionsRule.ProjectReduceExpressionsRule.ProjectReduceExpressionsRuleConfig.DEFAULT.toRule())
            /*
            Pushes predicates that are used on one side of equality in a join to
            the other side of the join as well, enabling further filter pushdown
            and reduce the amount of data joined.

            For example, consider the query:

            select t1.a, t2.b from table1 t1, table2 t2 where t1.a = 1 AND t1.a = t2.b

            This produces a plan like

            LogicalProject(a=[$0], b=[$1])
              LogicalJoin(condition=[=($0, $1)], joinType=[inner])
                LogicalProject(A=[$0])
                  LogicalFilter(condition=[=($0, 1)])
                    LogicalTableScan(table=[[main, table1]])
                LogicalProject(B=[$1])
                    LogicalFilter(condition=[=($1, 1)])
                      LogicalTableScan(table=[[main, table2]])

             So both table1 and table2 filter on col = 1.
             */
            .addRuleInstance(JoinPushTransitivePredicatesRule.Config.DEFAULT.toRule())
            /*
             * Planner rule that removes
             * a {@link org.apache.calcite.rel.core.Aggregate}
             * if it computes no aggregate functions
             * (that is, it is implementing {@code SELECT DISTINCT}),
             * or all the aggregate functions are splittable,
             * and the underlying relational expression is already distinct.
             *
             */
            .addRuleInstance(AggregateRemoveRule.Config.DEFAULT.toRule())
            /*
             * Planner rule that matches an {@link org.apache.calcite.rel.core.Aggregate}
             * on a {@link org.apache.calcite.rel.core.Join} and removes the left input
             * of the join provided that the left input is also a left join if possible.
             *
             * <p>For instance,
             *
             * <blockquote>
             * <pre>select distinct s.product_id, pc.product_id
             * from sales as s
             * left join product as p
             *   on s.product_id = p.product_id
             * left join product_class pc
             *   on s.product_id = pc.product_id</pre></blockquote>
             *
             * <p>becomes
             *
             * <blockquote>
             * <pre>select distinct s.product_id, pc.product_id
             * from sales as s
             * left join product_class pc
             *   on s.product_id = pc.product_id</pre></blockquote>
             *
             * @see CoreRules#AGGREGATE_JOIN_JOIN_REMOVE
             */
            .addRuleInstance(AggregateJoinJoinRemoveRule.Config.DEFAULT.toRule())
            /*
             * Planner rule that merges an Aggregate into a projection when possible,
             * maintaining any aliases.
             */
            .addRuleInstance(AliasPreservingAggregateProjectMergeRule.Config.DEFAULT.toRule())
            /*
             * Planner rule that merges a Projection into an Aggregate when possible,
             * maintaining any aliases.
             */
            .addRuleInstance(ProjectAggregateMergeRule.Config.DEFAULT.toRule())
            /*
             * Planner rule that ensures filter is always pushed into join. This is needed
             * for complex queries.
             */
            // Ensure filters always occur before projections. Here we set a limit
            // so extremely complex filters aren't pushed.
            .addRuleInstance(FilterProjectTransposeNoCaseRule.Config.DEFAULT.toRule())
            // Prune trivial cross-joins
            .addRuleInstance(InnerJoinRemoveRule.Config.DEFAULT.toRule())
            // Rewrite filters in either Filter or Join to convert OR with shared subexpression
            // into
            // an AND and then OR. For example
            // OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
            // Another rule pushes filters into join and we do not know if the LogicalFilter
            // optimization will get to run before its pushed into the join. As a result,
            // we write a duplicate rule that operates directly on the condition of the join.
            .addRuleInstance(JoinReorderConditionRule.Config.DEFAULT.toRule())
            .addRuleInstance(LogicalFilterReorderConditionRule.Config.DEFAULT.toRule())
            // Push a limit before a project (e.g. select col as alias from table limit 10)
            .addRuleInstance(LimitProjectTransposeRule.Config.DEFAULT.toRule())
            // If a column has been repeated or rewritten as a part of another column, possibly
            // due to aliasing, then replace a projection with multiple projections.
            // For example convert:
            // LogicalProject(x=[$0], x2=[+($0, 10)], x3=[/(+($0, 10), 2)], x4=[*(/(+($0, 10), 2),
            // 3)])
            // to
            // LogicalProject(x=[$0], x2=[$1], x3=[/(+($1, 10), 2)], x4=[*(/(+($1, 10), 2), 3)])
            //  LogicalProject(x=[$0], x2=[+($0, 10)])
            .addRuleInstance(ProjectionSubcolumnEliminationRule.Config.DEFAULT.toRule())
            // Remove any case expressions from filters because we cannot use them in filter
            // pushdown.
            .addRuleInstance(FilterExtractCaseRule.Config.DEFAULT.toRule())
            // For two projections separated by a filter, determine if any computation in
            // the uppermost filter can be removed by referencing a column in the innermost
            // projection. See the rule docstring for more detail.
            .addRuleInstance(ProjectFilterProjectColumnEliminationRule.Config.DEFAULT.toRule())
            .addRuleInstance(MinRowNumberFilterRule.Config.DEFAULT.toRule())
            // Push down aggregates on snowflake tables directly to snowflake.
            .addRuleInstance(SnowflakeAggregateRule.Config.DEFAULT.toRule())
            .addRuleInstance(SnowflakeAggregateRule.Config.WITH_FILTER.toRule())
            .build()

        override fun run(
            planner: RelOptPlanner?,
            rel: RelNode,
            requiredOutputTraits: RelTraitSet?,
            materializations: List<RelOptMaterialization>?,
            lattices: List<RelOptLattice>?
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
}
