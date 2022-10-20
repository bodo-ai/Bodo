package com.bodosql.calcite.application.bodo_sql_rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.immutables.value.Value;

/**
 * Planner rule that takes cross-join a {@link org.apache.calcite.rel.core.Join} with at least 1
 * input with 0 columns and removes the join from the plan.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class InnerJoinRemoveRule extends RelRule<InnerJoinRemoveRule.Config>
    implements SubstitutionRule {

  /** Creates a InnerJoinRemoveRule. */
  protected InnerJoinRemoveRule(InnerJoinRemoveRule.Config config) {
    super(config);
  }

  /**
   * Returns a child of join if it is a cross-join and only of the inputs has 0 columns. Its unclear
   * exactly how calcite is using this method, if at all, but the other classes support it should we
   * opt to match it.
   */
  public static RelNode strip(Join join) {
    boolean emptyLeft = join.getLeft().getRowType().getFieldNames().size() == 0;
    boolean emptyRight = join.getRight().getRowType().getFieldNames().size() == 0;
    if ((emptyLeft || emptyRight) && join.getCondition().isAlwaysTrue()) {
      RelNode stripped = emptyRight ? join.getLeft() : join.getRight();
      return stripped;
    }
    return join;
  }

  /**
   * This method notes that a join with 0 columns is defined to have exactly 1 row. Therefore, if we
   * have a cross join and either side has 0 columns, we know that we can remove the join and just
   * return the input.
   *
   * @param join The join mode that may be possible to optimize out.
   * @return If we can remove the join.
   */
  public static boolean isEmptyJoin(Join join) {
    boolean emptyLeft = join.getLeft().getRowType().getFieldNames().size() == 0;
    boolean emptyRight = join.getRight().getRowType().getFieldNames().size() == 0;
    return ((emptyLeft || emptyRight) && join.getCondition().isAlwaysTrue());
  }

  /** This method is called when the rule finds a Relnode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // Number of rel elements in call is determined by the number of values
    // in the withOperandSupplier below.
    Join join = call.rel(0);
    boolean emptyLeft = join.getLeft().getRowType().getFieldNames().size() == 0;
    boolean emptyRight = join.getRight().getRowType().getFieldNames().size() == 0;
    if ((emptyLeft || emptyRight) && join.getCondition().isAlwaysTrue()) {
      RelNode stripped = emptyRight ? join.getLeft() : join.getRight();
      stripped = convert(stripped, join.getConvention());
      call.transformTo(stripped);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is Join, the "how" is inner, and either left
    // or right outputs 0 columns.
    InnerJoinRemoveRule.Config DEFAULT =
        ImmutableInnerJoinRemoveRule.Config.of()
            .withOperandSupplier(
                b -> b.operand(Join.class).predicate(InnerJoinRemoveRule::isEmptyJoin).anyInputs())
            .as(InnerJoinRemoveRule.Config.class);

    @Override
    default InnerJoinRemoveRule toRule() {
      return new InnerJoinRemoveRule(this);
    }
  }
}
