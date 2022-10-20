package com.bodosql.calcite.application.bodo_sql_rules;

import static com.bodosql.calcite.application.bodo_sql_rules.FilterRulesCommon.filterContainsOr;
import static com.bodosql.calcite.application.bodo_sql_rules.FilterRulesCommon.updateConditionsExtractCommon;

import java.util.HashSet;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Pair;
import org.immutables.value.Value;

/**
 * Planner rule that recognizes a {@link org.apache.calcite.rel.core.LogicalFilter} that contains a
 * condition with an OR and checks for any shared subexpressions that can be converted into an AND.
 * For example: OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
 *
 * <p>This optimization ensures we avoid some unnecessary duplicate subexpressions and could improve
 * the performance of filters/filter push down in the future.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class LogicalFilterReorderConditionRule
    extends RelRule<LogicalFilterReorderConditionRule.Config> implements SubstitutionRule {

  /** Creates a LogicalFilterReorderConditionRule. */
  protected LogicalFilterReorderConditionRule(LogicalFilterReorderConditionRule.Config config) {
    super(config);
  }

  /**
   * Determines if the given LogicalFilter contains an OR as part of its condition. If so it may be
   * possible to rewrite the filter to extract a common expression.
   *
   * @param filter The LogicalFilter node that may be rewritten.
   * @return If the LogicalFilter contains an OR.
   */
  public static boolean containsOrFilter(LogicalFilter filter) {
    RexNode cond = filter.getCondition();
    return filterContainsOr(cond);
  }

  /** This method is called when the rule finds a RelNode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // Now that we know the RelNode contains an OR it may be possible to extract
    // a common condition.
    LogicalFilter filter = call.rel(0);
    RelBuilder builder = call.builder();
    HashSet<RexNode> commonExprs = new HashSet<RexNode>();
    Pair<RexNode, Boolean> updatedOR =
        updateConditionsExtractCommon(builder, filter.getCondition(), commonExprs);
    boolean changed = updatedOR.getValue();
    if (changed) {
      // If there are common filters we can extra from OR rewrite the RexNode
      LogicalFilter newFilter =
          filter.copy(filter.getTraitSet(), filter.getInput(), updatedOR.getKey());
      call.transformTo(newFilter);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is LogicalFilter and the condition contains
    // an OR.
    LogicalFilterReorderConditionRule.Config DEFAULT =
        ImmutableLogicalFilterReorderConditionRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(LogicalFilter.class)
                        .predicate(LogicalFilterReorderConditionRule::containsOrFilter)
                        .anyInputs())
            .as(LogicalFilterReorderConditionRule.Config.class);

    @Override
    default LogicalFilterReorderConditionRule toRule() {
      return new LogicalFilterReorderConditionRule(this);
    }
  }
}
