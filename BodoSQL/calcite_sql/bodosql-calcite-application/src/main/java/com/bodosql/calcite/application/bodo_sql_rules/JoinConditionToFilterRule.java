package com.bodosql.calcite.application.bodo_sql_rules;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.rules.TransformationRule;
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
    Join join = call.rel(0);

    RelBuilder builder = call.builder();

    // Set the join condition to True and
    // copy over all other attributes.
    Join newJoin =
        join.copy(
            join.getTraitSet(),
            builder.literal(true),
            join.getLeft(),
            join.getRight(),
            join.getJoinType(),
            join.isSemiJoinDone());

    builder.push(newJoin);

    // Create a filter using the join's condition
    builder.filter(join.getCondition());

    call.transformTo(builder.build());
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is Join, and it contains an OR in the
    // condition.
    //
    // Only matches join conditions that are inner joins. This is because
    // the pattern of pulling the join condition into a filter only works
    // with inner joins since there's no way for filter to add a column
    // with null values for the right side of the join after the join
    // has been performed.
    JoinConditionToFilterRule.Config DEFAULT =
        ImmutableJoinConditionToFilterRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Join.class)
                        .predicate(join -> join.getJoinType() == JoinRelType.INNER)
                        .anyInputs())
            .as(JoinConditionToFilterRule.Config.class);

    @Override
    default JoinConditionToFilterRule toRule() {
      return new JoinConditionToFilterRule(this);
    }
  }
}
