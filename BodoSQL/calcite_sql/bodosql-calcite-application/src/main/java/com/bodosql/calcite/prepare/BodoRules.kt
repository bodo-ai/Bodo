package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRules
import com.bodosql.calcite.adapter.iceberg.IcebergFilterLockRule
import com.bodosql.calcite.adapter.iceberg.IcebergLimitLockRule
import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregate
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilterLockRule
import com.bodosql.calcite.adapter.snowflake.SnowflakeLimitLockRule
import com.bodosql.calcite.adapter.snowflake.SnowflakeProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeProjectIntoScanRule
import com.bodosql.calcite.adapter.snowflake.SnowflakeProjectLockRule
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeSort
import com.bodosql.calcite.application.logicalRules.AggregateExpandDistinctAggregatesRule
import com.bodosql.calcite.application.logicalRules.AggregateFilterToCaseRule
import com.bodosql.calcite.application.logicalRules.AggregateMergeRule
import com.bodosql.calcite.application.logicalRules.AggregateReduceFunctionsRule
import com.bodosql.calcite.application.logicalRules.BodoAggregateJoinTransposeRule
import com.bodosql.calcite.application.logicalRules.BodoCommonSubexpressionRule
import com.bodosql.calcite.application.logicalRules.BodoJoinDeriveIsNotNullFilterRule
import com.bodosql.calcite.application.logicalRules.BodoJoinProjectTransposeNoCSEUndoRule
import com.bodosql.calcite.application.logicalRules.BodoJoinPushTransitivePredicatesRule
import com.bodosql.calcite.application.logicalRules.BodoProjectToWindowRule
import com.bodosql.calcite.application.logicalRules.BodoProjectWindowTransposeRule
import com.bodosql.calcite.application.logicalRules.BodoSQLReduceExpressionsRule
import com.bodosql.calcite.application.logicalRules.ExtractPushableExpressionsJoin
import com.bodosql.calcite.application.logicalRules.FilterAggregateTransposeRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterExtractCaseRule
import com.bodosql.calcite.application.logicalRules.FilterFlattenTranspose
import com.bodosql.calcite.application.logicalRules.FilterJoinRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterMergeRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterProjectTransposeNoCaseRule
import com.bodosql.calcite.application.logicalRules.FilterSetOpTransposeRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterWindowEjectRule
import com.bodosql.calcite.application.logicalRules.FilterWindowMrnfRule
import com.bodosql.calcite.application.logicalRules.FilterWindowSplitRule
import com.bodosql.calcite.application.logicalRules.GroupingSetsToUnionAllRule
import com.bodosql.calcite.application.logicalRules.JoinConditionToFilterRule
import com.bodosql.calcite.application.logicalRules.JoinReorderConditionRule
import com.bodosql.calcite.application.logicalRules.LimitProjectTransposeRule
import com.bodosql.calcite.application.logicalRules.LogicalFilterReorderConditionRule
import com.bodosql.calcite.application.logicalRules.MinRowNumberFilterRule
import com.bodosql.calcite.application.logicalRules.PartialJoinConditionIntoChildrenRule
import com.bodosql.calcite.application.logicalRules.ProjectFilterProjectColumnEliminationRule
import com.bodosql.calcite.application.logicalRules.ProjectionSubcolumnEliminationRule
import com.bodosql.calcite.application.logicalRules.SingleValuePruneRule
import com.bodosql.calcite.application.logicalRules.ValuesReduceRule
import com.bodosql.calcite.application.logicalRules.WindowDecomposeRule
import com.bodosql.calcite.application.utils.BodoJoinConditionUtil
import com.bodosql.calcite.prepare.MultiJoinRules.FILTER_MULTI_JOIN_MERGE
import com.bodosql.calcite.prepare.MultiJoinRules.JOIN_TO_MULTI_JOIN
import com.bodosql.calcite.prepare.MultiJoinRules.MULTI_JOIN_BOTH_PROJECT
import com.bodosql.calcite.prepare.MultiJoinRules.MULTI_JOIN_LEFT_PROJECT
import com.bodosql.calcite.prepare.MultiJoinRules.MULTI_JOIN_RIGHT_PROJECT
import com.bodosql.calcite.prepare.MultiJoinRules.PROJECT_MULTI_JOIN_MERGE
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories.BODO_LOGICAL_BUILDER
import com.bodosql.calcite.rel.core.BodoPhysicalRelFactories.BODO_PHYSICAL_BUILDER
import com.bodosql.calcite.rel.core.MinRowNumberFilterBase
import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import com.bodosql.calcite.rel.logical.BodoLogicalSort
import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import com.bodosql.calcite.rel.logical.BodoLogicalWindow
import com.google.common.collect.Iterables
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.plan.RelRule.OperandBuilder
import org.apache.calcite.plan.RelRule.OperandTransform
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.SetOp
import org.apache.calcite.rel.rules.AggregateJoinJoinRemoveRule
import org.apache.calcite.rel.rules.AggregateJoinRemoveRule
import org.apache.calcite.rel.rules.AggregateProjectMergeRule
import org.apache.calcite.rel.rules.AggregateProjectPullUpConstantsRule
import org.apache.calcite.rel.rules.CoreRules
import org.apache.calcite.rel.rules.FilterJoinRule
import org.apache.calcite.rel.rules.FilterWindowTransposeRule
import org.apache.calcite.rel.rules.JoinCommuteRule
import org.apache.calcite.rel.rules.LoptOptimizeJoinRule
import org.apache.calcite.rel.rules.ProjectAggregateMergeRule
import org.apache.calcite.rel.rules.ProjectFilterTransposeRule
import org.apache.calcite.rel.rules.ProjectJoinTransposeRule
import org.apache.calcite.rel.rules.ProjectMergeRule
import org.apache.calcite.rel.rules.ProjectRemoveRule
import org.apache.calcite.rel.rules.PruneEmptyRules
import org.apache.calcite.rel.rules.SortMergeRule
import org.apache.calcite.rel.rules.SortProjectTransposeRule
import org.apache.calcite.rel.rules.UnionMergeRule
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver

object BodoRules {
    // INDIVIDUAL RULES

    /**
     * Planner rule that, given a Project node that merely returns its input,
     * converts the node into its child.
     */
    @JvmField
    val PROJECT_REMOVE_RULE: RelOptRule =
        ProjectRemoveRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that combines two LogicalFilters.
     */
    @JvmField
    val FILTER_MERGE_RULE: RelOptRule =
        FilterMergeRuleNoWindow.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that merges a Project into another Project,
     * provided the projects aren't projecting identical sets of input references.
     */
    @JvmField
    val PROJECT_MERGE_RULE: RelOptRule =
        ProjectMergeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that merges a Project into another Project,
     * provided the projects aren't projecting identical sets of input references.
     */
    @JvmField
    val PROJECT_MERGE_ALLOW_NO_BLOAT_RULE: RelOptRule =
        ProjectMergeRule.Config.DEFAULT
            .withBloat(0)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalProject::class.java).oneInput(
                    OperandTransform { b1: OperandBuilder -> b1.operand(BodoLogicalProject::class.java).anyInputs() },
                )
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes a Filter past a Aggregate.
     */
    @JvmField
    val FILTER_AGGREGATE_TRANSPOSE_RULE: RelOptRule =
        FilterAggregateTransposeRuleNoWindow.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val FILTER_SETOP_TRANSPOSE_RULE: RelOptRule =
        FilterSetOpTransposeRuleNoWindow.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val WINDOW_DECOMPOSE_RULE: RelOptRule =
        WindowDecomposeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
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
    @JvmField
    val AGGREGATE_JOIN_REMOVE_RULE: RelOptRule =
        AggregateJoinRemoveRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, BodoLogicalJoin::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Fuse together two aggregates directly on top of each other.
     * For example SUM(SUM()) -> SUM().
     */
    @JvmField
    val AGGREGATE_MERGE_RULE: RelOptRule =
        AggregateMergeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Decompose aggregates when rewrites are applicable.
     * For example AVG(X) -> SUM(X) / COUNT(X).
     */
    @JvmField
    val AGGREGATE_REDUCE_FUNCTIONS_RULE: RelOptRule =
        AggregateReduceFunctionsRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Convert COUNT (DISTINCT x) to COUNT() via an extra group by.
     */
    @JvmField
    val EXPAND_COUNT_DISTINCT_RULE: RelOptRule =
        AggregateExpandDistinctAggregatesRule.Config.DEFAULT
            .withOperandSupplier { b: OperandBuilder ->
                b.operand(BodoLogicalAggregate::class.java).anyInputs()
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts an aggregate with a filter expression into an aggregate
     * whose input has converted all invalid values to NULL.
     */
    @JvmField
    val AGGREGATE_FILTER_TO_CASE_RULE: RelOptRule =
        AggregateFilterToCaseRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes an Aggregate past a join
     */
    @JvmField
    val AGGREGATE_JOIN_TRANSPOSE_RULE: RelOptRule =
        BodoAggregateJoinTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, BodoLogicalJoin::class.java, true)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push filter expressions into a join condition and into the inputs of the join.
     */
    @JvmField
    val FILTER_INTO_JOIN_RULE: RelOptRule =
        FilterJoinRuleNoWindow.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that applies moves any filters that depend on a single table before the join in
     * which they occur.
     */
    @JvmField
    val FILTER_JOIN_RULE: RelOptRule =
        FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Pushes full projects past joins when possible.
     */
    @JvmField
    val FULL_PROJECT_JOIN_TRANSPOSE_RULE: RelOptRule =
        ProjectJoinTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java, BodoLogicalJoin::class.java)
            .withPreserveExprCondition { expr: RexNode -> !RexOver.containsOver(expr) }
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Push only field references past a filter.
     */
    @JvmField
    val TRIVIAL_PROJECT_FILTER_TRANSPOSE: RelOptRule =
        ProjectFilterTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java, BodoLogicalFilter::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * This reduces expressions inside of the conditions of filter statements.
     * Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE)
     */
    @JvmField
    val FILTER_REDUCE_EXPRESSIONS_RULE: RelOptRule =
        BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT
            .withOperandFor(BodoLogicalFilter::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Simplify constant expressions inside a Projection. Ex condition($0 = 1 and $0 = 2)
     * ==> condition(FALSE)
     */
    @JvmField
    val PROJECT_REDUCE_EXPRESSIONS_RULE: RelOptRule =
        BodoSQLReduceExpressionsRule.ProjectReduceExpressionsRule.ProjectReduceExpressionsRuleConfig.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Pushes predicates that are used on one side of equality in a join to
     * the other side of the join as well, enabling further filter pushdown
     * and reduce the amount of data joined.
     *
     * For example, consider the query:
     *
     * select t1.a, t2.b from table1 t1, table2 t2 where t1.a = 1 AND t1.a = t2.b
     *
     * This produces a plan like
     *
     * ```
     * LogicalProject(a=[$0], b=[$1])
     *   LogicalJoin(condition=[=($0, $1)], joinType=[inner])
     *     LogicalProject(A=[$0])
     *       LogicalFilter(condition=[=($0, 1)])
     *         LogicalTableScan(table=[[main, table1]])
     *   LogicalProject(B=[$1])
     *     LogicalFilter(condition=[=($1, 1)])
     *       LogicalTableScan(table=[[main, table2]])
     * ```
     *
     * So both table1 and table2 filter on col = 1.
     */
    @JvmField
    val JOIN_PUSH_TRANSITIVE_PREDICATES: RelOptRule =
        BodoJoinPushTransitivePredicatesRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that removes a [org.apache.calcite.rel.core.Aggregate]
     * if it computes no aggregate functions (that is, it is implementing
     * {@code SELECT DISTINCT}), or all the aggregate functions are splittable,
     * and the underlying relational expression is already distinct.
     */
    @JvmField
    val AGGREGATE_REMOVE_RULE: RelOptRule =
        AggregateJoinRemoveRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, BodoLogicalJoin::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that matches an [org.apache.calcite.rel.core.Aggregate]
     * on a [org.apache.calcite.rel.core.Join] and removes the left input
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
    @JvmField
    val AGGREGATE_JOIN_JOIN_REMOVE_RULE: RelOptRule =
        AggregateJoinJoinRemoveRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, BodoLogicalJoin::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that merges an Aggregate into a projection when possible.
     */
    @JvmField
    val AGGREGATE_PROJECT_MERGE_RULE: RelOptRule =
        AggregateProjectMergeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that merges a Projection into an Aggregate when possible.
     */
    @JvmField
    val PROJECT_AGGREGATE_MERGE_RULE: RelOptRule =
        ProjectAggregateMergeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that ensures filter is always pushed into project. This avoids
     * case statements to ensure we never create a case statement inside a filter.
     */
    @JvmField
    val FILTER_PROJECT_TRANSPOSE_RULE: RelOptRule =
        FilterProjectTransposeNoCaseRule.Config.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(BodoLogicalFilter::class.java)
                    .oneInput { b1: OperandBuilder ->
                        b1.operand(BodoLogicalProject::class.java).anyInputs()
                    }
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that ensures filter is always pushed past flatten if possible.
     * If only part of a filter can be pushed before the flatten then this node
     * will split the filter into parts and split the component that can be pushed
     * down.
     */
    @JvmField
    val FILTER_FLATTEN_TRANSPOSE_RULE: RelOptRule =
        FilterFlattenTranspose.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val LEFT_JOIN_REMOVE_RULE: RelOptRule =
        PruneEmptyRules.JoinLeftEmptyRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val RIGHT_JOIN_REMOVE_RULE: RelOptRule =
        PruneEmptyRules.JoinLeftEmptyRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PRUNE_EMPTY_PROJECT_RULE: RelOptRule =
        PruneEmptyRules.RemoveEmptySingleRule.RemoveEmptySingleRuleConfig.PROJECT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PRUNE_EMPTY_FILTER_RULE: RelOptRule =
        PruneEmptyRules.RemoveEmptySingleRule.RemoveEmptySingleRuleConfig.FILTER
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PRUNE_EMPTY_AGGREGATE_RULE: RelOptRule =
        PruneEmptyRules.RemoveEmptySingleRule.RemoveEmptySingleRuleConfig.AGGREGATE
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PRUNE_EMPTY_SORT_RULE: RelOptRule =
        PruneEmptyRules.RemoveEmptySingleRule.RemoveEmptySingleRuleConfig.SORT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PRUNE_EMPTY_UNION_RULE: RelOptRule =
        PruneEmptyRules.UnionEmptyPruneRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val PROJECT_VALUES_REDUCE_RULE: RelOptRule =
        ValuesReduceRule.Config.PROJECT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Rewrite filters in either Filter or Join to convert OR with shared subexpression
    // into
    // an AND and then OR. For example
    // OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
    // Another rule pushes filters into join and we do not know if the LogicalFilter
    // optimization will get to run before its pushed into the join. As a result,
    // we write a duplicate rule that operates directly on the condition of the join.
    @JvmField
    val JOIN_REORDER_CONDITION_RULE: RelOptRule =
        JoinReorderConditionRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val FILTER_REORDER_CONDITION_RULE: RelOptRule =
        LogicalFilterReorderConditionRule.Config.DEFAULT
            .withOperandFor(BodoLogicalFilter::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Push a limit before a project (e.g. select col as alias from table limit 10)
    @JvmField
    val LIMIT_PROJECT_TRANSPOSE_RULE: RelOptRule =
        LimitProjectTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalSort::class.java, BodoLogicalProject::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Pushes a sort before a project
    @JvmField
    val SORT_PROJECT_TRANSPOSE_RULE: RelOptRule =
        SortProjectTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalSort::class.java, BodoLogicalProject::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Rules that handle Pulling projects above joins when possible
    // Because of the rule configuration limitations, I need 2 separate rules to match
    // All possible cases

    // JOIN
    //   Project
    //   Any (Including another project, which will NOT be handled by this rule)
    //
    @JvmField
    val JOIN_PROJECT_LEFT_TRANSPOSE_INCLUDE_OUTER: RelOptRule =
        BodoJoinProjectTransposeNoCSEUndoRule.Config.DEFAULT
            .withIncludeOuter(true)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalJoin::class.java).inputs(
                    OperandTransform { b1: OperandBuilder -> b1.operand(BodoLogicalProject::class.java).anyInputs() },
                )
            }.withDescription("BodoJoinProjectTransposeLeftIncludeOuterRule")
            .toRule()

    // JOIN
    //   Any (Including another project, which will be pulled above the join by this rule if possible)
    //   Project
    //
    @JvmField
    val JOIN_PROJECT_RIGHT_TRANSPOSE_INCLUDE_OUTER: RelOptRule =
        BodoJoinProjectTransposeNoCSEUndoRule.Config.DEFAULT
            .withIncludeOuter(true)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalJoin::class.java).inputs(
                    OperandTransform { b1: OperandBuilder ->
                        b1
                            .operand(RelNode::class.java)
                            // Can't allow snowflakeRels here. If we do, it may allow snowflake project that we've locked
                            // in to be pulled up.
                            .predicate { node -> node !is SnowflakeRel }
                            .anyInputs()
                    },
                    OperandTransform { b2: OperandBuilder -> b2.operand(BodoLogicalProject::class.java).anyInputs() },
                )
            }.withDescription("BodoJoinProjectTransposeRightIncludeOuterRule")
            .toRule()

    // If a column has been repeated or rewritten as a part of another column, possibly
    // due to aliasing, then replace a projection with multiple projections.
    // For example convert:
    // LogicalProject(x=[$0], x2=[+($0, 10)], x3=[/(+($0, 10), 2)], x4=[*(/(+($0, 10), 2),
    // 3)])
    // to
    // LogicalProject(x=[$0], x2=[$1], x3=[/(+($1, 10), 2)], x4=[*(/(+($1, 10), 2), 3)])
    //  LogicalProject(x=[$0], x2=[+($0, 10)])
    @JvmField
    val PROJECTION_SUBCOLUMN_ELIMINATION_RULE: RelOptRule =
        ProjectionSubcolumnEliminationRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val BODO_COMMON_SUBEXPRESSION_RULE: RelOptRule =
        BodoCommonSubexpressionRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Remove any case expressions from filters because we cannot use them in filter
    // pushdown.
    @JvmField
    val FILTER_EXTRACT_CASE_RULE: RelOptRule =
        FilterExtractCaseRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    // Remove any SINGLE_VALUE calls that are unnecessary.
    @JvmField
    val SINGLE_VALUE_REMOVE_RULE: RelOptRule =
        SingleValuePruneRule.Config.DEFAULT.toRule()

    // For two projections separated by a filter, determine if any computation in
    // the uppermost filter can be removed by referencing a column in the innermost
    // projection. See the rule docstring for more detail.
    @JvmField
    val PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE: RelOptRule =
        ProjectFilterProjectColumnEliminationRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val MIN_ROW_NUMBER_FILTER_RULE: RelOptRule =
        MinRowNumberFilterRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Extract the join condition to the filter rule to allow invalid join
     * conditions to be utilized.
     */
    @JvmField
    val JOIN_CONDITION_TO_FILTER_RULE: RelOptRule =
        JoinConditionToFilterRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Split a FILTER containing a window function and other functions if possible,
     * effectively pushing some filters past the others.
     */
    @JvmField
    val FILTER_WINDOW_SPLIT_RULE: RelOptRule =
        FilterWindowSplitRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push parts of a filter below a window node if the filter components only
     * apply to the partition keys.
     */
    @JvmField
    val FILTER_WINDOW_TRANSPOSE_RULE: RelOptRule =
        FilterWindowTransposeRule.Config.DEFAULT
            .withOperandSupplier { b0 ->
                b0.operand(BodoLogicalFilter::class.java).oneInput { b1 ->
                    b1.operand(BodoLogicalWindow::class.java).anyInputs()
                }
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push parts of a project below a window node.
     */
    @JvmField
    val PROJECT_WINDOW_TRANSPOSE_RULE: RelOptRule =
        BodoProjectWindowTransposeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push filter expressions into a join condition and into the inputs of the join.
     */
    @JvmField
    val BODO_PHYSICAL_FILTER_INTO_JOIN_RULE: RelOptRule =
        FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
            .withPredicate { join, _, exp ->
                when (join) {
                    is BodoPhysicalJoin -> BodoJoinConditionUtil.isValidNode(exp)
                    else -> true
                }
            }.withOperandSupplier { b0: OperandBuilder ->
                b0.operand(Filter::class.java).predicate { f -> f !is MinRowNumberFilterBase }.oneInput { b1: OperandBuilder ->
                    b1
                        .operand(
                            Join::class.java,
                        ).anyInputs()
                }
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Expand Grouping sets into a UNION ALL of Aggregates
     */
    @JvmField
    val BODO_PHYSICAL_GROUPING_SETS_TO_UNION_ALL_RULE: RelOptRule =
        GroupingSetsToUnionAllRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Allow join inputs to be swapped. This is kept to reorder outer joins
     * until we are certain it will be handled by multi-join, but then it should
     * be removed.
     */
    @JvmField
    val JOIN_COMMUTE_RULE: RelOptRule =
        JoinCommuteRule.Config.DEFAULT
            .withSwapOuter(true)
            // Disable join commute rules for joins that contain hints.
            // The only hints we support define the join ordering.
            .withOperandSupplier { b: OperandBuilder ->
                b
                    .operand(BodoLogicalJoin::class.java)
                    // FIXME Enable this rule for joins with system fields
                    .predicate { j: Join ->
                        (
                            j.left.id != j.right.id &&
                                j.systemFieldList.isEmpty() &&
                                j.hints.isEmpty()
                        )
                    }.anyInputs()
            }.withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Determine an optimal ordering of the join ordering from
     * the MultiJoin.
     *
     * Converts a MultiJoin into a tree of Join nodes.
     */
    @JvmField
    val LOPT_OPTIMIZE_JOIN_RULE: RelOptRule =
        LoptOptimizeJoinRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Pull up Constants used in Aggregates.
     */
    @JvmField
    val AGGREGATE_CONSTANT_PULL_UP_RULE: RelOptRule =
        AggregateProjectPullUpConstantsRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, RelNode::class.java)
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that derives IS NOT NULL predicates from a inner {@link Join} and creates
     * {@link Filter}s with those predicates as new inputs of the {@link Join}.
     */
    @JvmField
    val JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE: RelOptRule =
        BodoJoinDeriveIsNotNullFilterRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts a BodoPhysicalFilter to a SnowflakeFilter if it is located directly on top of a
     * SnowflakeRel.
     */
    @JvmField
    val SNOWFLAKE_FILTER_LOCK_RULE: RelOptRule =
        SnowflakeFilterLockRule.Config.DEFAULT_CONFIG
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts a BodoPhysicalFilter to an IcebergFilter if it is located directly on top of a
     * IcebergRel. Should only be used in HEP pass to avoid infinite loop.
     */
    @JvmField
    val ICEBERG_FILTER_LOCK_RULE: RelOptRule =
        IcebergFilterLockRule.Config.HEP_DEFAULT_CONFIG
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts a BodoLogicalProject to a SnowflakeProject if it is located directly on top of a
     * SnowflakeRel.
     */
    @JvmField
    val SNOWFLAKE_PROJECT_LOCK_RULE: RelOptRule =
        SnowflakeProjectLockRule.Config.DEFAULT_CONFIG
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts a BodoLogicalSort with only limit to a SnowflakeLimit if it is located directly on top of a
     * SnowflakeRel.
     */
    @JvmField
    val SNOWFLAKE_LIMIT_LOCK_RULE: RelOptRule =
        SnowflakeLimitLockRule.Config.DEFAULT_CONFIG
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Converts a BodoLogicalSort with only limit to a IcebergLimit if it is located directly on top of a
     * SnowflakeRel.
     */
    @JvmField
    val ICEBERG_LIMIT_LOCK_RULE: RelOptRule =
        IcebergLimitLockRule.Config.DEFAULT_CONFIG
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    /**
     * Merge two UNION operators together into a single UNION operator.
     * The rule will check that the ALL values are compatible.
     *
     * TODO: Investigate how to improve this rule. The current rule's requirement for binary inputs potentially
     * prevent full flattening.
     */
    @JvmField
    val UNION_MERGE_RULE: RelOptRule =
        UnionMergeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalUnion::class.java).inputs(
                    OperandTransform { b1: OperandBuilder -> b1.operand(RelNode::class.java).anyInputs() },
                    OperandTransform { b2: OperandBuilder -> b2.operand(RelNode::class.java).anyInputs() },
                )
            }.withDescription("BodoUnionMergeRule")
            .toRule()

    @JvmField
    val CSE_IN_FILTERS_RULE: RelOptRule =
        InClauseCommonSubexpressionEliminationRule.Config.DEFAULT
            .withRelBuilderFactory(
                BODO_LOGICAL_BUILDER,
            ).toRule()

    @JvmField
    val EXTRACT_PUSHABLE_JOIN_CONDITIONS: RelOptRule =
        ExtractPushableExpressionsJoin.Config.DEFAULT
            .withRelBuilderFactory(
                BODO_LOGICAL_BUILDER,
            ).withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalJoin::class.java).anyInputs()
            }.toRule()

    @JvmField
    val PROJECT_FILTER_TRANSPOSE_PUSHABLE_EXPRESSIONS: RelOptRule =
        com.bodosql.calcite.application.logicalRules.ProjectFilterTransposePushableExpressionsRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalProject::class.java).oneInput { b1: OperandBuilder ->
                    b1
                        .operand(
                            BodoLogicalFilter::class.java,
                        ).anyInputs()
                }
            }.toRule()

    @JvmField
    val PROJECT_SET_OP_TRANSPOSE: RelOptRule =
        com.bodosql.calcite.application.logicalRules.BodoProjectSetOpTransposeRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0.operand(BodoLogicalProject::class.java).oneInput { b1: OperandBuilder ->
                    b1
                        .operand(
                            SetOp::class.java,
                        ).anyInputs()
                }
            }.toRule()

    /**
     * Lock in SnowflakeProjects a top SnowflakeToBodoPhysicalConverter
     */
    @JvmStatic
    val SNOWFLAKE_PROJECT_CONVERTER_LOCK_RULE: RelOptRule = SnowflakeProjectIntoScanRule.Config.BODO_LOGICAL_CONFIG.toRule()

    @JvmField
    val PROJECT_TO_WINDOW_RULE: RelOptRule =
        BodoProjectToWindowRule.BodoProjectToLogicalProjectAndWindowRule.BodoProjectToLogicalProjectAndWindowRuleConfig.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val FILTER_WINDOW_EJECT: RelOptRule =
        FilterWindowEjectRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val FILTER_WINDOW_MRNF_RULE: RelOptRule =
        FilterWindowMrnfRule.Config.DEFAULT
            .withRelBuilderFactory(BODO_LOGICAL_BUILDER)
            .toRule()

    @JvmStatic
    val PARTIAL_JOIN_CONDITION_INTO_CHILDREN_VOLCANO_RULE: RelOptRule =
        PartialJoinConditionIntoChildrenRule.Config.DEFAULT
            .withRelBuilderFactory(
                BODO_LOGICAL_BUILDER,
            ).toRule()

    /**
     * Planner rule that combines two SnowflakeFilters.
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_FILTER_MERGE_RULE: RelOptRule =
        FilterMergeRuleNoWindow.Config.DEFAULT
            .withOperandSupplier { b0 ->
                b0
                    .operand(SnowflakeFilter::class.java)
                    .predicate { f: SnowflakeFilter -> f.getCatalogTable().isIcebergTable() }
                    .oneInput { b1: OperandBuilder ->
                        b1
                            .operand(SnowflakeFilter::class.java)
                            .predicate { f: SnowflakeFilter -> f.getCatalogTable().isIcebergTable() }
                            .anyInputs()
                    }
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes a SnowflakeFilter past a SnowflakeProject.
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_FILTER_PROJECT_TRANSPOSE_RULE: RelOptRule =
        FilterProjectTransposeNoCaseRule.Config.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeFilter::class.java)
                    .predicate { f: SnowflakeFilter -> f.getCatalogTable().isIcebergTable() }
                    .oneInput { b1: OperandBuilder ->
                        b1
                            .operand(SnowflakeProject::class.java)
                            .predicate { p: SnowflakeProject -> p.getCatalogTable().isIcebergTable() }
                            .anyInputs()
                    }
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * This reduces expressions inside of the conditions of Snowflake filter statements
     * Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE).
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_FILTER_REDUCE_EXPRESSIONS_RULE: RelOptRule =
        BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeFilter::class.java)
                    .predicate { f: SnowflakeFilter -> f.getCatalogTable().isIcebergTable() }
                    .anyInputs()
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes a SnowflakeFilter past a SnowflakeAggregate.
     * This rule should only fire for pushing past distinct clauses.
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_FILTER_AGGREGATE_TRANSPOSE_RULE: RelOptRule =
        FilterAggregateTransposeRuleNoWindow.Config.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeFilter::class.java)
                    .predicate { f: SnowflakeFilter -> f.getCatalogTable().isIcebergTable() }
                    .oneInput { b1: OperandBuilder ->
                        b1
                            .operand(SnowflakeAggregate::class.java)
                            .predicate { a: SnowflakeAggregate -> a.getCatalogTable().isIcebergTable() }
                            .anyInputs()
                    }
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that merges a SnowflakeProject into another SnowflakeProject,
     * provided the projects aren't projecting identical sets of input references.
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_PROJECT_MERGE_RULE: RelOptRule =
        ProjectMergeRule.Config.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeProject::class.java)
                    .predicate { p: SnowflakeProject -> p.getCatalogTable().isIcebergTable() }
                    .oneInput { b1: OperandBuilder ->
                        b1
                            .operand(SnowflakeProject::class.java)
                            .predicate { p: SnowflakeProject -> p.getCatalogTable().isIcebergTable() }
                            .anyInputs()
                    }
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that, given a Physical Project node that merely returns its input,
     * converts the node into its child.
     * This rule only fires on Snowflake nodes atop Iceberg tables.
     */
    @JvmField
    val SNOWFLAKE_PROJECT_REMOVE_RULE: RelOptRule =
        ProjectRemoveRule.Config.DEFAULT
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeProject::class.java)
                    .predicate { p: SnowflakeProject -> ProjectRemoveRule.isTrivial(p) && p.getCatalogTable().isIcebergTable() }
                    .anyInputs()
            }.withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that merges two Snowflake Limit rules together.
     * In general, we can/should move this to the Volcano planner,
     * but we need to refactor how this occurs.
     *
     * Note: We don't need the previous limitations in the LIMIT_MERGE
     * configuration because these are already invariants of SnowflakeSorts.
     */
    @JvmField
    val SNOWFLAKE_LIMIT_MERGE_RULE: RelOptRule =
        SortMergeRule.Config.LIMIT_MERGE
            .withRelBuilderFactory(BODO_PHYSICAL_BUILDER)
            .withOperandSupplier { b0: OperandBuilder ->
                b0
                    .operand(SnowflakeSort::class.java)
                    .predicate { p: SnowflakeSort -> p.getCatalogTable().isIcebergTable() }
                    .oneInput { b1: OperandBuilder ->
                        b1
                            .operand(SnowflakeSort::class.java)
                            .predicate { p: SnowflakeSort -> p.getCatalogTable().isIcebergTable() }
                            .anyInputs()
                    }
            }.toRule()

    // RULE GROUPINGS

    /**
     * These are rules to remove sub-queries.
     */
    val SUB_QUERY_REMOVAL_RULES: List<RelOptRule> =
        listOf(
            com.bodosql.calcite.application.logicalRules.SubQueryRemoveRule.Config.FILTER
                .toRule(),
            com.bodosql.calcite.application.logicalRules.SubQueryRemoveRule.Config.PROJECT
                .toRule(),
            com.bodosql.calcite.application.logicalRules.SubQueryRemoveRule.Config.JOIN
                .toRule(),
        )

    /**
     * Rules that handle generic projection pushdown. Currently, this just consists
     * of the available rules in Calcite. All of these rules run during FieldPushdown,
     * and some run in other locations.
     */
    val ProjectionPushdownRules: List<RelOptRule> =
        listOf(
            PROJECT_SET_OP_TRANSPOSE,
            FULL_PROJECT_JOIN_TRANSPOSE_RULE,
            AGGREGATE_PROJECT_MERGE_RULE,
            PROJECT_MERGE_RULE,
            FILTER_MERGE_RULE,
            PROJECT_REMOVE_RULE,
        )

    /**
     * Specialized rules that specifically handles field pushdown.
     */
    val FieldPushdownRules: List<RelOptRule> =
        listOf(
            EXTRACT_PUSHABLE_JOIN_CONDITIONS,
            PROJECT_FILTER_TRANSPOSE_PUSHABLE_EXPRESSIONS,
            // rules necessary to lock in pushed projects
            SNOWFLAKE_PROJECT_LOCK_RULE,
            SNOWFLAKE_PROJECT_CONVERTER_LOCK_RULE,
        )

    /**
     * These are all the rules that allow for pulling a project closer to the root
     * of the RelTree.
     *
     * All of these run during ProjectionPullUpPass. Some of these also run during other volcano passes
     */
    val PROJECTION_PULL_UP_RULES: List<RelOptRule> =
        listOf(
            FILTER_PROJECT_TRANSPOSE_RULE,
            LIMIT_PROJECT_TRANSPOSE_RULE,
            SORT_PROJECT_TRANSPOSE_RULE,
            JOIN_PROJECT_LEFT_TRANSPOSE_INCLUDE_OUTER,
            JOIN_PROJECT_RIGHT_TRANSPOSE_INCLUDE_OUTER,
            PROJECT_MERGE_ALLOW_NO_BLOAT_RULE,
            PROJECT_REMOVE_RULE,
        )

    /**
     * These are rules that essentially "rewrite" an unsupported or non-ideal plan construction
     * into another. The assumption is all these are irreversible and are not generated by
     * any of our optimizations.
     */
    val REWRITE_RULES: List<RelOptRule> =
        listOf(
            WINDOW_DECOMPOSE_RULE,
            FILTER_EXTRACT_CASE_RULE,
            MIN_ROW_NUMBER_FILTER_RULE,
            AGGREGATE_FILTER_TO_CASE_RULE,
            AGGREGATE_REDUCE_FUNCTIONS_RULE,
            EXPAND_COUNT_DISTINCT_RULE,
            AGGREGATE_CONSTANT_PULL_UP_RULE,
            BODO_COMMON_SUBEXPRESSION_RULE,
        )

    /**
     * These are rules that push filters deeper into the plan and attempt to force filters
     * into source nodes.
     */
    val FILTER_PUSH_DOWN_RULES: List<RelOptRule> =
        listOf(
            // Rules for pushing filters
            FILTER_PROJECT_TRANSPOSE_RULE,
            FILTER_AGGREGATE_TRANSPOSE_RULE,
            FILTER_SETOP_TRANSPOSE_RULE,
            FILTER_INTO_JOIN_RULE,
            FILTER_JOIN_RULE,
            FILTER_FLATTEN_TRANSPOSE_RULE,
            // Combining filters can simplify filters for pushing
            FILTER_MERGE_RULE,
            // Push some filters past filters in window functions.
            FILTER_WINDOW_SPLIT_RULE,
            // Reordering conditions can lead to greater filter pushing.
            FILTER_REORDER_CONDITION_RULE,
            JOIN_REORDER_CONDITION_RULE,
            // Process for inserting new filters to push
            JOIN_PUSH_TRANSITIVE_PREDICATES,
            JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE,
            // Locking in any filters
            SNOWFLAKE_FILTER_LOCK_RULE,
            ICEBERG_FILTER_LOCK_RULE,
            // Locking in any limits
            SNOWFLAKE_LIMIT_LOCK_RULE,
            ICEBERG_LIMIT_LOCK_RULE,
        )

    /**
     * These are rules related to pushing limits into source tables.
     */
    val LIMIT_PUSH_DOWN_RULES: List<RelOptRule> =
        listOf(
            LIMIT_PROJECT_TRANSPOSE_RULE,
        )

    /**
     * These are rules that only operate on physical nodes because
     * the general structure of the plan should be maintained.
     */
    val PHYSICAL_CONVERSION_RULES: List<RelOptRule> =
        listOf(
            BODO_PHYSICAL_GROUPING_SETS_TO_UNION_ALL_RULE,
        )

    /**
     * These are rules used for simplification of expressions.
     *
     * Note we don't include FILTER_REDUCE_EXPRESSIONS_RULE because it creates a cycle
     * in q_succulent_archway. This didn't alter any other plans, so it should be
     * fine to remove, but we may want to revisit in the future with a more concrete
     * reason or re-enabling this rule.
     */
    val SIMPLIFICATION_RULES: List<RelOptRule> =
        listOf(
            PROJECT_REDUCE_EXPRESSIONS_RULE,
            // simplifies constants after aggregates
            // This is needed to take advantage of constant prop done by the above rules
            AGGREGATE_CONSTANT_PULL_UP_RULE,
        )

    /**
     * These are rules that work with some form of Common Subexpression Elimination
     * (CSE).
     *
     * Note: These rules likely need to be written to enable tracking "existing expressions",
     * ideally through metadata.
     */
    val CSE_RULES: List<RelOptRule> =
        listOf(
            // NOTE: TRIVIAL_PROJECT_FILTER_TRANSPOSE can't be run with filter pushdown without Volcano.
            // If we ever want to run CSE_RULES with filter pushdown, this needs to be removed.
            TRIVIAL_PROJECT_FILTER_TRANSPOSE,
            PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE,
            PROJECTION_SUBCOLUMN_ELIMINATION_RULE,
        )

    /**
     * These are the rules to construct a MultiJoin
     */
    val MULTI_JOIN_CONSTRUCTION_RULES: List<RelOptRule> =
        listOf(
            MULTI_JOIN_BOTH_PROJECT,
            MULTI_JOIN_LEFT_PROJECT,
            MULTI_JOIN_RIGHT_PROJECT,
            JOIN_TO_MULTI_JOIN,
            PROJECT_MULTI_JOIN_MERGE,
            FILTER_MULTI_JOIN_MERGE,
            // Need to merge filters/projects to prevent blocking multi-joins.
            PROJECT_MERGE_RULE,
            FILTER_MERGE_RULE,
        )

    /**
     * These are rules that control join ordering.
     */
    val JOIN_ORDERING_RULES: List<RelOptRule> =
        listOf(
            JOIN_COMMUTE_RULE,
            LOPT_OPTIMIZE_JOIN_RULE,
        )

    /**
     * Rules that use an Aggregate to remove another operator.
     */
    private val AGGREGATE_OPERATOR_REMOVAL_RULES: List<RelOptRule> =
        listOf(
            AGGREGATE_PROJECT_MERGE_RULE,
            AGGREGATE_JOIN_JOIN_REMOVE_RULE,
            AGGREGATE_REMOVE_RULE,
            AGGREGATE_JOIN_REMOVE_RULE,
            AGGREGATE_MERGE_RULE,
        )

    /**
     * Rules that use a Project to remove another operator.
     */
    val PROJECT_OPERATOR_REMOVAL_RULES: List<RelOptRule> =
        listOf(
            PROJECT_MERGE_ALLOW_NO_BLOAT_RULE,
            PROJECT_AGGREGATE_MERGE_RULE,
            PROJECT_REMOVE_RULE,
        )

    /**
     * Rules that use a UNION to remove another operator.
     */
    private val UNION_OPERATOR_REMOVAL_RULES: List<RelOptRule> =
        listOf(
            UNION_MERGE_RULE,
        )

    /**
     * All rules that involve removing 1 or more operators.
     */
    val OPERATOR_REMOVAL_RULES: List<RelOptRule> =
        Iterables
            .concat(
                AGGREGATE_OPERATOR_REMOVAL_RULES,
                PROJECT_OPERATOR_REMOVAL_RULES,
                UNION_OPERATOR_REMOVAL_RULES,
            ).toList()

    /**
     * Rules that enable a cost based decision for reordering operators.
     */
    val OPERATOR_REORDERING_RULES: List<RelOptRule> =
        listOf(
            AGGREGATE_JOIN_TRANSPOSE_RULE,
        )

    /**
     * These are rules that we will want to remove
     * because there is some issue with their implementation.
     */
    val CANDIDATE_REMOVAL_RULES: List<RelOptRule> =
        listOf(
            // This may conflict with other optimizations because it doesn't allow
            // pushing filters that can enter 1 side of a join.
            BODO_PHYSICAL_FILTER_INTO_JOIN_RULE,
            // This rule needs to be rewritten/modified to only occur once a join is
            // finalized. Ideally this should only run after every join is pushed
            // as far as possible since this should be a correctness constraint.
            JOIN_CONDITION_TO_FILTER_RULE,
        )

    /**
     * Rules that are used in the window conversion pass.
     */
    @JvmField
    val WINDOW_CONVERSION_RULES =
        listOf(PROJECT_TO_WINDOW_RULE, FILTER_WINDOW_EJECT, FILTER_WINDOW_MRNF_RULE, PROJECT_REMOVE_RULE, FILTER_PROJECT_TRANSPOSE_RULE)

    /**
     * Rules that are used to clean up a Snowflake section into a standard form
     * for conversion into Iceberg. All of these rules only fire on Iceberg tables
     * to avoid possible regressions.
     */
    @JvmField
    val SNOWFLAKE_CLEANUP_RULES =
        listOf(
            SNOWFLAKE_FILTER_MERGE_RULE,
            SNOWFLAKE_FILTER_PROJECT_TRANSPOSE_RULE,
            SNOWFLAKE_FILTER_REDUCE_EXPRESSIONS_RULE,
            SNOWFLAKE_FILTER_AGGREGATE_TRANSPOSE_RULE,
            SNOWFLAKE_PROJECT_MERGE_RULE,
            SNOWFLAKE_PROJECT_REMOVE_RULE,
            SNOWFLAKE_LIMIT_MERGE_RULE,
        )

    // OPTIMIZER GROUPS
    @JvmField
    val VOLCANO_MINIMAL_RULE_SET: List<RelOptRule> =
        Iterables
            .concat(
                BodoPhysicalRules.rules(),
                listOf(
                    JOIN_CONDITION_TO_FILTER_RULE,
                    BODO_PHYSICAL_FILTER_INTO_JOIN_RULE,
                    FILTER_JOIN_RULE,
                    PARTIAL_JOIN_CONDITION_INTO_CHILDREN_VOLCANO_RULE,
                ),
            ).toList()

    @JvmField
    val VOLCANO_OPTIMIZE_RULE_SET: List<RelOptRule> =
        listOf(
            PROJECT_REMOVE_RULE,
            FILTER_MERGE_RULE,
            FILTER_WINDOW_SPLIT_RULE,
            PROJECT_MERGE_RULE,
            FILTER_AGGREGATE_TRANSPOSE_RULE,
            AGGREGATE_JOIN_REMOVE_RULE,
            AGGREGATE_MERGE_RULE,
            AGGREGATE_JOIN_TRANSPOSE_RULE,
            FILTER_REDUCE_EXPRESSIONS_RULE,
            PROJECT_REDUCE_EXPRESSIONS_RULE,
            JOIN_PUSH_TRANSITIVE_PREDICATES,
            AGGREGATE_REMOVE_RULE,
            AGGREGATE_JOIN_JOIN_REMOVE_RULE,
            AGGREGATE_PROJECT_MERGE_RULE,
            PROJECT_AGGREGATE_MERGE_RULE,
            FILTER_PROJECT_TRANSPOSE_RULE,
            LEFT_JOIN_REMOVE_RULE,
            RIGHT_JOIN_REMOVE_RULE,
            PRUNE_EMPTY_PROJECT_RULE,
            PRUNE_EMPTY_FILTER_RULE,
            PRUNE_EMPTY_AGGREGATE_RULE,
            PRUNE_EMPTY_SORT_RULE,
            PRUNE_EMPTY_UNION_RULE,
            PROJECT_VALUES_REDUCE_RULE,
            JOIN_REORDER_CONDITION_RULE,
            FILTER_REORDER_CONDITION_RULE,
            LIMIT_PROJECT_TRANSPOSE_RULE,
            FILTER_EXTRACT_CASE_RULE,
            PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE,
            JOIN_COMMUTE_RULE,
            LOPT_OPTIMIZE_JOIN_RULE,
            UNION_MERGE_RULE,
            FILTER_WINDOW_TRANSPOSE_RULE,
            PROJECT_WINDOW_TRANSPOSE_RULE,
            FILTER_WINDOW_MRNF_RULE,
            // Make cost based decisions for Project push down
            PROJECT_SET_OP_TRANSPOSE,
            FULL_PROJECT_JOIN_TRANSPOSE_RULE,
        )
}
