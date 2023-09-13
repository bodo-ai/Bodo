package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.pandas.PandasJoin
import com.bodosql.calcite.adapter.pandas.PandasJoinRule
import com.bodosql.calcite.adapter.pandas.PandasRules
import com.bodosql.calcite.application.logicalRules.BodoSQLReduceExpressionsRule
import com.bodosql.calcite.application.logicalRules.FilterAggregateTransposeRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterExtractCaseRule
import com.bodosql.calcite.application.logicalRules.FilterJoinRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterMergeRuleNoWindow
import com.bodosql.calcite.application.logicalRules.FilterProjectTransposeNoCaseRule
import com.bodosql.calcite.application.logicalRules.InnerJoinRemoveRule
import com.bodosql.calcite.application.logicalRules.JoinConditionToFilterRule
import com.bodosql.calcite.application.logicalRules.JoinReorderConditionRule
import com.bodosql.calcite.application.logicalRules.LimitProjectTransposeRule
import com.bodosql.calcite.application.logicalRules.LogicalFilterReorderConditionRule
import com.bodosql.calcite.application.logicalRules.MinRowNumberFilterRule
import com.bodosql.calcite.application.logicalRules.ProjectFilterProjectColumnEliminationRule
import com.bodosql.calcite.application.logicalRules.ProjectionSubcolumnEliminationRule
import com.bodosql.calcite.application.logicalRules.RexSimplificationRule
import com.bodosql.calcite.application.logicalRules.TrivialProjectJoinTransposeRule
import com.bodosql.calcite.application.logicalRules.VolcanoAcceptingAggregateProjectPullUpConstantsRule
import com.bodosql.calcite.rel.core.RelFactories
import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import com.bodosql.calcite.rel.logical.BodoLogicalSort
import com.google.common.collect.Iterables
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.rules.AggregateJoinJoinRemoveRule
import org.apache.calcite.rel.rules.AggregateJoinRemoveRule
import org.apache.calcite.rel.rules.AggregateJoinTransposeRule
import org.apache.calcite.rel.rules.AggregateProjectMergeRule
import org.apache.calcite.rel.rules.FilterJoinRule
import org.apache.calcite.rel.rules.JoinCommuteRule
import org.apache.calcite.rel.rules.JoinPushTransitivePredicatesRule
import org.apache.calcite.rel.rules.JoinToMultiJoinRule
import org.apache.calcite.rel.rules.LoptOptimizeJoinRule
import org.apache.calcite.rel.rules.ProjectAggregateMergeRule
import org.apache.calcite.rel.rules.ProjectFilterTransposeRule
import org.apache.calcite.rel.rules.ProjectMergeRule
import org.apache.calcite.rel.rules.ProjectRemoveRule

object BodoRules {
    /**
     * Planner rule that, given a Project node that merely returns its input,
     * converts the node into its child.
     */
    @JvmField
    val PROJECT_REMOVE_RULE: RelOptRule =
        ProjectRemoveRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that combines two LogicalFilters.
     */
    @JvmField
    val FILTER_MERGE_RULE: RelOptRule =
        FilterMergeRuleNoWindow.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that merges a Project into another Project,
     * provided the projects aren't projecting identical sets of input references.
     */
    @JvmField
    val PROJECT_MERGE_RULE: RelOptRule =
        ProjectMergeRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes a Filter past a Aggregate.
     */
    @JvmField
    val FILTER_AGGREGATE_TRANSPOSE_RULE: RelOptRule =
        FilterAggregateTransposeRuleNoWindow.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Planner rule that pushes an Aggregate past a join
     */
    @JvmField
    val AGGREGATE_JOIN_TRANSPOSE_RULE: RelOptRule =
        AggregateJoinTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalAggregate::class.java, BodoLogicalJoin::class.java, true)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push filter expressions into a join condition and into the inputs of the join.
     */
    @JvmField
    val FILTER_INTO_JOIN_RULE: RelOptRule =
        FilterJoinRuleNoWindow.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that applies moves any filters that depend on a single table before the join in
     * which they occur.
     */
    @JvmField
    val FILTER_JOIN_RULE: RelOptRule =
        FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Filters tables for unused columns before join.
     */
    @JvmField
    val PROJECT_JOIN_TRANSPOSE_RULE: RelOptRule =
        TrivialProjectJoinTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java, BodoLogicalJoin::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Push only field references past a filter.
     */
    @JvmField
    val TRIVIAL_PROJECT_FILTER_TRANSPOSE: RelOptRule =
        ProjectFilterTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java, BodoLogicalFilter::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * This reduces expressions inside of the conditions of filter statements.
     * Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE)
     * TODO(Ritwika: figure out SEARCH handling later. SARG attributes do not have public access methods.
     */
    @JvmField
    val FILTER_REDUCE_EXPRESSIONS_RULE: RelOptRule =
        BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT
            .withOperandFor(BodoLogicalFilter::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Simplify constant expressions inside a Projection. Ex condition($0 = 1 and $0 = 2)
     * ==> condition(FALSE)
     */
    @JvmField
    val PROJECT_REDUCE_EXPRESSIONS_RULE: RelOptRule =
        BodoSQLReduceExpressionsRule.ProjectReduceExpressionsRule.ProjectReduceExpressionsRuleConfig.DEFAULT
            .withOperandFor(BodoLogicalProject::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
        JoinPushTransitivePredicatesRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that merges an Aggregate into a projection when possible,
     * maintaining any aliases.
     */
    @JvmField
    val AGGREGATE_PROJECT_MERGE_RULE: RelOptRule =
        AggregateProjectMergeRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that merges a Projection into an Aggregate when possible,
     * maintaining any aliases.
     */
    @JvmField
    val PROJECT_AGGREGATE_MERGE_RULE: RelOptRule =
        ProjectAggregateMergeRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /*
     * Planner rule that ensures filter is always pushed into join. This is needed
     * for complex queries.
     */
    // Ensure filters always occur before projections. Here we set a limit
    // so extremely complex filters aren't pushed.
    @JvmField
    val FILTER_PROJECT_TRANSPOSE_RULE: RelOptRule =
        FilterProjectTransposeNoCaseRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    // Prune trivial cross-joins
    @JvmField
    val INNER_JOIN_REMOVE_RULE: RelOptRule =
        InnerJoinRemoveRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val FILTER_REORDER_CONDITION_RULE: RelOptRule =
        LogicalFilterReorderConditionRule.Config.DEFAULT
            .withOperandFor(BodoLogicalFilter::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    // Push a limit before a project (e.g. select col as alias from table limit 10)
    @JvmField
    val LIMIT_PROJECT_TRANSPOSE_RULE: RelOptRule =
        LimitProjectTransposeRule.Config.DEFAULT
            .withOperandFor(BodoLogicalSort::class.java, BodoLogicalProject::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    // Remove any case expressions from filters because we cannot use them in filter
    // pushdown.
    @JvmField
    val FILTER_EXTRACT_CASE_RULE: RelOptRule =
        FilterExtractCaseRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    // For two projections separated by a filter, determine if any computation in
    // the uppermost filter can be removed by referencing a column in the innermost
    // projection. See the rule docstring for more detail.
    @JvmField
    val PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE: RelOptRule =
        ProjectFilterProjectColumnEliminationRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val MIN_ROW_NUMBER_FILTER_RULE: RelOptRule =
        MinRowNumberFilterRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    @JvmField
    val REX_SIMPLIFICATION_RULE: RelOptRule =
        RexSimplificationRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Extract the join condition to the filter rule to allow invalid join
     * conditions to be utilized.
     */
    @JvmField
    val JOIN_CONDITION_TO_FILTER_RULE: RelOptRule =
        JoinConditionToFilterRule.Config.DEFAULT
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Rule that tries to push filter expressions into a join condition and into the inputs of the join.
     */
    @JvmField
    val PANDAS_FILTER_INTO_JOIN_RULE: RelOptRule =
        FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
            .withPredicate { join, _, exp ->
                when (join) {
                    is PandasJoin -> PandasJoinRule.isValidNode(exp)
                    else -> true
                }
            }
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Gather join nodes into a single multi join for optimization.
     */
    @JvmField
    val JOIN_TO_MULTI_JOIN: RelOptRule =
        JoinToMultiJoinRule.Config.DEFAULT
            .withOperandFor(BodoLogicalJoin::class.java)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Allow join inputs to be swapped. This is kept to reorder outer joins
     * until we are certain it will be handled by multi-join, but then it should
     * be removed.
     */
    @JvmField
    val JOIN_COMMUTE_RULE: RelOptRule =
        JoinCommuteRule.Config.DEFAULT
            .withOperandFor(BodoLogicalJoin::class.java)
            .withSwapOuter(true)
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
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
            .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
            .toRule()

    /**
     * Pull up Constants used in Aggregates.
     */
    @JvmField
    val AGGREGATE_CONSTANT_PULL_UP_RULE: RelOptRule = VolcanoAcceptingAggregateProjectPullUpConstantsRule.Config.DEFAULT
        .withOperandFor(BodoLogicalAggregate::class.java, RelNode::class.java)
        .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
        .toRule()

    @JvmField
    val HEURISTIC_RULE_SET: List<RelOptRule> = listOf(
        PROJECT_REMOVE_RULE,
        FILTER_MERGE_RULE,
        PROJECT_MERGE_RULE,
        FILTER_AGGREGATE_TRANSPOSE_RULE,
        AGGREGATE_JOIN_REMOVE_RULE,
        AGGREGATE_JOIN_TRANSPOSE_RULE,
        FILTER_INTO_JOIN_RULE,
        FILTER_JOIN_RULE,
        PROJECT_JOIN_TRANSPOSE_RULE,
        FILTER_REDUCE_EXPRESSIONS_RULE,
        PROJECT_REDUCE_EXPRESSIONS_RULE,
        JOIN_PUSH_TRANSITIVE_PREDICATES,
        AGGREGATE_REMOVE_RULE,
        AGGREGATE_JOIN_JOIN_REMOVE_RULE,
        AGGREGATE_PROJECT_MERGE_RULE,
        PROJECT_AGGREGATE_MERGE_RULE,
        FILTER_PROJECT_TRANSPOSE_RULE,
        INNER_JOIN_REMOVE_RULE,
        JOIN_REORDER_CONDITION_RULE,
        FILTER_REORDER_CONDITION_RULE,
        LIMIT_PROJECT_TRANSPOSE_RULE,
        PROJECTION_SUBCOLUMN_ELIMINATION_RULE,
        FILTER_EXTRACT_CASE_RULE,
        PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE,
        MIN_ROW_NUMBER_FILTER_RULE,
        REX_SIMPLIFICATION_RULE,
        AGGREGATE_CONSTANT_PULL_UP_RULE,
    )

    @JvmField
    val VOLCANO_MINIMAL_RULE_SET: List<RelOptRule> = Iterables.concat(
        PandasRules.rules(),
        listOf(
            JOIN_CONDITION_TO_FILTER_RULE,
            PANDAS_FILTER_INTO_JOIN_RULE,
            FILTER_JOIN_RULE,
        ),
    ).toList()

    @JvmField
    val VOLCANO_OPTIMIZE_RULE_SET: List<RelOptRule> = listOf(
        PROJECT_REMOVE_RULE,
        FILTER_MERGE_RULE,
        PROJECT_MERGE_RULE,
        FILTER_AGGREGATE_TRANSPOSE_RULE,
        AGGREGATE_JOIN_REMOVE_RULE,
        AGGREGATE_JOIN_TRANSPOSE_RULE,
        PROJECT_JOIN_TRANSPOSE_RULE,
        FILTER_REDUCE_EXPRESSIONS_RULE,
        PROJECT_REDUCE_EXPRESSIONS_RULE,
        JOIN_PUSH_TRANSITIVE_PREDICATES,
        AGGREGATE_REMOVE_RULE,
        AGGREGATE_JOIN_JOIN_REMOVE_RULE,
        AGGREGATE_PROJECT_MERGE_RULE,
        PROJECT_AGGREGATE_MERGE_RULE,
        FILTER_PROJECT_TRANSPOSE_RULE,
        INNER_JOIN_REMOVE_RULE,
        JOIN_REORDER_CONDITION_RULE,
        FILTER_REORDER_CONDITION_RULE,
        LIMIT_PROJECT_TRANSPOSE_RULE,
        PROJECTION_SUBCOLUMN_ELIMINATION_RULE,
        FILTER_EXTRACT_CASE_RULE,
        PROJECT_FILTER_PROJECT_COLUMN_ELIMINATION_RULE,
        MIN_ROW_NUMBER_FILTER_RULE,
        REX_SIMPLIFICATION_RULE,
        JOIN_COMMUTE_RULE,
        LOPT_OPTIMIZE_JOIN_RULE,
        AGGREGATE_CONSTANT_PULL_UP_RULE,
        TRIVIAL_PROJECT_FILTER_TRANSPOSE,
    )
}
