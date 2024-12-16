package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoJoinConditionUtil;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

@BodoSQLStyleImmutable
@Value.Enclosing
public class JoinConditionToFilterRule extends RelRule<JoinConditionToFilterRule.Config>
    implements TransformationRule {

  /** Creates a JoinConditionToFilterRule. */
  protected JoinConditionToFilterRule(JoinConditionToFilterRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    RelBuilder builder = call.builder();
    RexBuilder rexBuilder = builder.getRexBuilder();
    Join join = call.rel(0);
    RexNode existingCondition = join.getCondition();
    List<RexNode> candidatesFilters = RelOptUtil.conjunctions(existingCondition);
    List<RexNode> keptFilters = new ArrayList();
    List<RexNode> movedFilters = new ArrayList();
    int totalColumns = join.getRowType().getFieldCount();
    int leftColumns = join.getLeft().getRowType().getFieldCount();

    for (RexNode filter : candidatesFilters) {
      if (!BodoJoinConditionUtil.isValidNode(filter)
          && !BodoJoinConditionUtil.isPushableFunction(filter, leftColumns, totalColumns)) {
        movedFilters.add(filter);
      } else {
        keptFilters.add(filter);
      }
    }

    // Set the join condition to True and
    // copy over all other attributes.
    Join newJoin =
        join.copy(
            join.getTraitSet(),
            RexUtil.composeConjunction(rexBuilder, keptFilters),
            join.getLeft(),
            join.getRight(),
            join.getJoinType(),
            join.isSemiJoinDone());

    builder.push(newJoin);

    // Create a filter using the join's condition
    builder.filter(RexUtil.composeConjunction(rexBuilder, movedFilters));

    call.transformTo(builder.build());
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is inner Join, the join condition is not always
    // true, and the join condition is not a valid join condition for
    // generating code/cannot be converted to a valid join by pushing
    // condition components into the input(s).
    //
    // Note: This doesn't work for OUTER joins because of nulls.
    //
    JoinConditionToFilterRule.Config DEFAULT =
        ImmutableJoinConditionToFilterRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Join.class)
                        .predicate(
                            join ->
                                join.getJoinType() == JoinRelType.INNER
                                    && !join.getCondition().isAlwaysTrue()
                                    && !BodoJoinConditionUtil.isValidNode(join.getCondition()))
                        .anyInputs())
            .as(JoinConditionToFilterRule.Config.class);

    @Override
    default JoinConditionToFilterRule toRule() {
      return new JoinConditionToFilterRule(this);
    }
  }
}
