package com.bodosql.calcite.application.logicalRules;

import static com.bodosql.calcite.application.logicalRules.WindowFilterTranspose.findPushableFilterComponents;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.List;
import kotlin.Pair;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that takes a filter containing at least 1 window function and attempts to push part
 * of the non-window function component in front of the window function. This is only possible if
 * the other filter only operates on the partition by section of window parts.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterWindowSplitRule extends RelRule<FilterWindowSplitRule.Config>
    implements TransformationRule {

  /** Creates a FilterWindowSplitRule. */
  protected FilterWindowSplitRule(FilterWindowSplitRule.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    Filter filterRel = call.rel(0);
    RelBuilder builder = call.builder();
    Pair<List<RexNode>, List<RexNode>> splits = findPushableFilterComponents(filterRel);
    List<RexNode> pushParts = splits.getFirst();
    List<RexNode> remainingParts = splits.getSecond();
    // Note the RexOver ensures remainingParts can't be empty.
    if (splits.getFirst().isEmpty()) {
      return;
    }
    builder.push(filterRel.getInput());
    builder.filter(pushParts);
    RelNode outerNode = builder.filter(remainingParts).build();
    call.transformTo(outerNode);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterWindowSplitRule.Config DEFAULT =
        ImmutableFilterWindowSplitRule.Config.of()
            .withOperandSupplier(
                b0 -> b0.operand(Filter.class).predicate(f -> f.containsOver()).anyInputs());

    @Override
    default FilterWindowSplitRule toRule() {
      return new FilterWindowSplitRule(this);
    }
  }
}
