package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.pandas.PandasJoin
import com.bodosql.calcite.adapter.pandas.PandasJoinRule
import com.bodosql.calcite.adapter.pandas.PandasRules
import com.bodosql.calcite.application.bodo_sql_rules.*
import com.google.common.collect.Iterables
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.rel.rules.*

object BodoRules {
    @JvmField
    val HEURISTIC_RULE_SET: List<RelOptRule> = listOf(
        /*
        Planner rule that, given a Project node that merely returns its input, converts the node into its child.
        */
        ProjectUnaliasedRemoveRule.Config.DEFAULT.toRule(),
        /*
        Planner rule that combines two LogicalFilters.
        */
        FilterMergeRuleNoWindow.Config.DEFAULT.toRule(),
        /*
           Planner rule that merges a Project into another Project,
           provided the projects aren't projecting identical sets of input references
           and don't have any dependencies.
        */
        DependencyCheckingProjectMergeRule.Config.DEFAULT.toRule(),
        /*
        Planner rule that pushes a Filter past a Aggregate.
        */
        FilterAggregateTransposeRuleNoWindow.Config.DEFAULT.toRule(),
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
        AggregateJoinRemoveRule.Config.DEFAULT.toRule(),
        /*
        Planner rule that pushes an Aggregate past a join
        */
        AggregateJoinTransposeRule.Config.EXTENDED.toRule(),
        /*
        Rule that tries to push filter expressions into a join condition and into the inputs of the join.
        */
        FilterJoinRuleNoWindow.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT.toRule(),
        /*
        Rule that applies moves any filters that depend on a single table before the join in
        which they occur.
        */
        FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT.toRule(),
        /*
        Filters tables for unused columns before join.
        */
        AliasPreservingProjectJoinTransposeRule.Config.DEFAULT.toRule(),
        /*
        This reduces expressions inside of the conditions of filter statements.
        Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE)
        TODO(Ritwika: figure out SEARCH handling later. SARG attributes do not have public access methods.
        */
        BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT.toRule(),
        // Simplify constant expressions inside a Projection. Ex condition($0 = 1 and $0 = 2)
        // ==> condition(FALSE)
        BodoSQLReduceExpressionsRule.ProjectReduceExpressionsRule.ProjectReduceExpressionsRuleConfig.DEFAULT.toRule(),
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
        JoinPushTransitivePredicatesRule.Config.DEFAULT.toRule(),
        /*
         * Planner rule that removes
         * a {@link org.apache.calcite.rel.core.Aggregate}
         * if it computes no aggregate functions
         * (that is, it is implementing {@code SELECT DISTINCT}),
         * or all the aggregate functions are splittable,
         * and the underlying relational expression is already distinct.
         *
         */
        AggregateRemoveRule.Config.DEFAULT.toRule(),
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
        AggregateJoinJoinRemoveRule.Config.DEFAULT.toRule(),
        /*
         * Planner rule that merges an Aggregate into a projection when possible,
         * maintaining any aliases.
         */
        AliasPreservingAggregateProjectMergeRule.Config.DEFAULT.toRule(),
        /*
         * Planner rule that merges a Projection into an Aggregate when possible,
         * maintaining any aliases.
         */
        ProjectAggregateMergeRule.Config.DEFAULT.toRule(),
        /*
         * Planner rule that ensures filter is always pushed into join. This is needed
         * for complex queries.
         */
        // Ensure filters always occur before projections. Here we set a limit
        // so extremely complex filters aren't pushed.
        FilterProjectTransposeNoCaseRule.Config.DEFAULT.toRule(),
        // Prune trivial cross-joins
        InnerJoinRemoveRule.Config.DEFAULT.toRule(),
        // Rewrite filters in either Filter or Join to convert OR with shared subexpression
        // into
        // an AND and then OR. For example
        // OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
        // Another rule pushes filters into join and we do not know if the LogicalFilter
        // optimization will get to run before its pushed into the join. As a result,
        // we write a duplicate rule that operates directly on the condition of the join.
        JoinReorderConditionRule.Config.DEFAULT.toRule(),
        LogicalFilterReorderConditionRule.Config.DEFAULT.toRule(),
        // Push a limit before a project (e.g. select col as alias from table limit 10)
        LimitProjectTransposeRule.Config.DEFAULT.toRule(),
        // If a column has been repeated or rewritten as a part of another column, possibly
        // due to aliasing, then replace a projection with multiple projections.
        // For example convert:
        // LogicalProject(x=[$0], x2=[+($0, 10)], x3=[/(+($0, 10), 2)], x4=[*(/(+($0, 10), 2),
        // 3)])
        // to
        // LogicalProject(x=[$0], x2=[$1], x3=[/(+($1, 10), 2)], x4=[*(/(+($1, 10), 2), 3)])
        //  LogicalProject(x=[$0], x2=[+($0, 10)])
        ProjectionSubcolumnEliminationRule.Config.DEFAULT.toRule(),
        // Remove any case expressions from filters because we cannot use them in filter
        // pushdown.
        FilterExtractCaseRule.Config.DEFAULT.toRule(),
        // For two projections separated by a filter, determine if any computation in
        // the uppermost filter can be removed by referencing a column in the innermost
        // projection. See the rule docstring for more detail.
        ProjectFilterProjectColumnEliminationRule.Config.DEFAULT.toRule(),
        MinRowNumberFilterRule.Config.DEFAULT.toRule(),
        RexSimplificationRule.Config.DEFAULT.toRule(),
        ListAggOptionalReplaceRule.Config.DEFAULT.toRule(),
    )

    private val FILTER_INTO_JOIN_RULE: RelOptRule =
        FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
            .withPredicate { join, _, exp ->
                when (join) {
                    is PandasJoin -> PandasJoinRule.isValidNode(exp)
                    else -> true
                }
            }
            .toRule()

    @JvmField
    val MINIMAL_VOLCANO_RULE_SET: List<RelOptRule> = Iterables.concat(
        PandasRules.rules(),
        listOf(
            // Extract the join condition to the filter rule to allow invalid join
            // conditions to be utilized.
            JoinConditionToFilterRule.Config.DEFAULT.toRule(),
            /*
            Rule that tries to push filter expressions into a join condition and into the inputs of the join.
            */
            FILTER_INTO_JOIN_RULE,
            /*
            Rule that applies moves any filters that depend on a single table before the join in
            which they occur.
            */
            FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT.toRule(),
        )
    ).toList()

    @JvmField
    val VOLCANO_RULE_SET: List<RelOptRule> = Iterables.concat(
        MINIMAL_VOLCANO_RULE_SET,
        // When testing the new volcano planner, place rules from the heuristic planner
        // here in order to validate that it produces a better plan.
        // Previously, we were just including all heuristic planner rules. But, that
        // was causing problems since sometimes rules need to be included to get volcano
        // planner to work correctly and sometimes they're for optimization and the volcano
        // planner itself doesn't really differentiate these.
        // For that reason, this list is now separate so we can have rules that run
        // with the volcano planner with the heuristic planner and we can have rules
        // that run with only the complete set.
        listOf(
            ProjectUnaliasedRemoveRule.Config.DEFAULT.toRule(),
            FilterMergeRuleNoWindow.Config.DEFAULT.toRule(),
            DependencyCheckingProjectMergeRule.Config.DEFAULT.toRule(),
            FilterAggregateTransposeRuleNoWindow.Config.DEFAULT.toRule(),
            AggregateJoinRemoveRule.Config.DEFAULT.toRule(),
            AggregateJoinTransposeRule.Config.EXTENDED.toRule(),
            AliasPreservingProjectJoinTransposeRule.Config.DEFAULT.toRule(),
            BodoSQLReduceExpressionsRule.FilterReduceExpressionsRule.FilterReduceExpressionsRuleConfig.DEFAULT.toRule(),
            BodoSQLReduceExpressionsRule.ProjectReduceExpressionsRule.ProjectReduceExpressionsRuleConfig.DEFAULT.toRule(),
            JoinPushTransitivePredicatesRule.Config.DEFAULT.toRule(),
            AggregateRemoveRule.Config.DEFAULT.toRule(),
            AggregateJoinJoinRemoveRule.Config.DEFAULT.toRule(),
            AliasPreservingAggregateProjectMergeRule.Config.DEFAULT.toRule(),
            ProjectAggregateMergeRule.Config.DEFAULT.toRule(),
            FilterProjectTransposeNoCaseRule.Config.DEFAULT.toRule(),
            InnerJoinRemoveRule.Config.DEFAULT.toRule(),
            JoinReorderConditionRule.Config.DEFAULT.toRule(),
            LogicalFilterReorderConditionRule.Config.DEFAULT.toRule(),
            LimitProjectTransposeRule.Config.DEFAULT.toRule(),
            ProjectionSubcolumnEliminationRule.Config.DEFAULT.toRule(),
            FilterExtractCaseRule.Config.DEFAULT.toRule(),
            ProjectFilterProjectColumnEliminationRule.Config.DEFAULT.toRule(),
            MinRowNumberFilterRule.Config.DEFAULT.toRule(),
            RexSimplificationRule.Config.DEFAULT.toRule(),
            ListAggOptionalReplaceRule.Config.DEFAULT.toRule(),
            // Allow planner to swap inputs between the build/probe join
            // side to reduce total in-use memory cost.
            CoreRules.JOIN_COMMUTE_OUTER,
            // TODO(Nick): Explore JOIN_ASSOCIATE in a future PR to allow
            // for join reordering.
        )
    ).toList()
}
