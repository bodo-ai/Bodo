package com.bodosql.calcite.application.logicalRules;

import static com.bodosql.calcite.application.logicalRules.SnowflakeProjectPushdownHelpers.replaceValuesJoinCondition;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.rules.TransformationRule;
import org.immutables.value.Value;

/**
 * Planner rule that extracts pushable expressions from a Join condition, and pushes it into a
 * separate project below the join.
 *
 * <p>For an example of this, see the join_condition_field_pushdown.sql test file.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class ExtractPushableExpressionsJoin extends RelRule<ExtractPushableExpressionsJoin.Config>
    implements TransformationRule {

  /** Creates an ExtractPushableExpressionsJoin. */
  protected ExtractPushableExpressionsJoin(Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join oldJoin = call.rel(0);
    final RelNode newJoin = replaceValuesJoinCondition(call.builder(), oldJoin);
    if (newJoin != null) {
      call.transformTo(newJoin);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    Config DEFAULT =
        ImmutableExtractPushableExpressionsJoin.Config.of().withOperandFor(LogicalJoin.class);

    @Override
    default ExtractPushableExpressionsJoin toRule() {
      return new ExtractPushableExpressionsJoin(this);
    }

    default Config withOperandFor(Class<? extends Join> joinClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(joinClass)
                      .predicate(
                          join ->
                              SnowflakeProjectPushdownHelpers.Companion
                                  .containsNonTrivialPushableExprs(join))
                      .anyInputs())
          .as(Config.class);
    }
  }
}
