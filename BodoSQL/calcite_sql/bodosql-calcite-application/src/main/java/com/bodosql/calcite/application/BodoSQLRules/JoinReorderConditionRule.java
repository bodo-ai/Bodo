package com.bodosql.calcite.application.BodoSQLRules;

import static com.bodosql.calcite.application.BodoSQLRules.FilterRulesCommon.filterContainsOr;
import static com.bodosql.calcite.application.BodoSQLRules.FilterRulesCommon.updateConditionsExtractCommon;

import java.util.HashSet;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Pair;
import org.immutables.value.Value;

/**
 * Planner rule that recognizes a {@link org.apache.calcite.rel.core.Join} that contains a condition
 * with an OR and checks for any shared subexpressions that can be converted into an AND. For
 * example: OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
 *
 * <p>This is an important optimization because extracting an equality condition could be the
 * difference between an equijoin and a cross join at the planning stage. In addition, only AND
 * expressions can be pushed past a join.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class JoinReorderConditionRule extends RelRule<JoinReorderConditionRule.Config>
    implements SubstitutionRule {

  /** Creates a JoinReorderConditionRule. */
  protected JoinReorderConditionRule(JoinReorderConditionRule.Config config) {
    super(config);
  }

  /**
   * Determines if the given join contains an OR as part of its condition. If so it may be possible
   * to rewrite the filter to extract a common expression.
   *
   * @param join The join node that may be rewritten.
   * @return If the join contains an OR.
   */
  public static boolean containsOrFilter(Join join) {
    RexNode cond = join.getCondition();
    return filterContainsOr(cond);
  }

  /** This method is called when the rule finds a RelNode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // Now that we know the RelNode contains an OR it may be possible to extract
    // a common condition. For this we check the string representation.
    Join join = call.rel(0);
    RelBuilder builder = call.builder();
    HashSet<RexNode> commonExprs = new HashSet<RexNode>();
    Pair<RexNode, Boolean> updatedOR =
        updateConditionsExtractCommon(builder, join.getCondition(), commonExprs);
    boolean changed = updatedOR.getValue();
    if (changed) {
      // If there are common filters we can extra from OR rewrite the RexNode
      Join newJoin =
          join.copy(
              join.getTraitSet(),
              updatedOR.getKey(),
              join.getLeft(),
              join.getRight(),
              join.getJoinType(),
              join.isSemiJoinDone());
      call.transformTo(newJoin);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is Join, and it contains an OR in the
    // condition.
    JoinReorderConditionRule.Config DEFAULT =
        ImmutableJoinReorderConditionRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Join.class)
                        .predicate(JoinReorderConditionRule::containsOrFilter)
                        .anyInputs())
            .as(JoinReorderConditionRule.Config.class);

    @Override
    default JoinReorderConditionRule toRule() {
      return new JoinReorderConditionRule(this);
    }
  }
}
