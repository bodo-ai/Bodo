package com.bodosql.calcite.application.bodo_sql_rules;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.immutables.value.Value;

/**
 * Planner rule that modifies rexNodes wherever they appear for a given plan. This rule essentially
 * calls RexSimplifyShuttle on each relnode in the plan, see RexSimplifyShuttle for the full list of
 * RexNode optimizations performed.
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class RexSimplificationRule extends RelRule<RexSimplificationRule.Config>
    implements SubstitutionRule {

  /** Creates a RexSimplificationRule. */
  protected RexSimplificationRule(Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  /** This method is called when the rule finds a Relnode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // Number of rel elements in call is determined by the number of values
    // in the withOperandSupplier below.
    RelNode node = call.rel(0);
    RelNode newNode = node.accept(new RexSimplifyShuttle(call.builder().getRexBuilder()));
    if (node != newNode) {
      call.transformTo(newNode);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any relnode, b
    Config DEFAULT =
        ImmutableRexSimplificationRule.Config.of()
            .withOperandSupplier(b -> b.operand(RelNode.class).anyInputs())
            .as(Config.class);

    @Override
    default RexSimplificationRule toRule() {
      return new RexSimplificationRule(this);
    }
  }
}
