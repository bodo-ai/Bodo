package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that prunes a SINGLE_VALUE introduced due to the inlining of a sub query if the
 * input can be statically proven to always have length 1. While this rule on its own likely isn't
 * the most significant, this can greatly simplify joins, which may enable stronger filters.
 *
 * <p>For example this would take this code segment:
 *
 * <blockquote>
 *
 * <pre>
 *  LogicalAggregate(group=[{}], agg#0=[SINGLE_VALUE($0)])
 *    LogicalValues(tuples=[[{ 'fee77b66-93e0-4518-abaa-6b81cd9f8acf' }]])
 * </pre>
 *
 * </blockquote>
 *
 * and convert it into
 *
 * <blockquote>
 *
 * <pre>
 * Project(NAME=$0)
 *     LogicalValues(tuples=[[{ 'fee77b66-93e0-4518-abaa-6b81cd9f8acf' }]])
 * </pre>
 *
 * We introduce the projection for type stability.
 *
 * </blockquote>
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class SingleValuePruneRule extends RelRule<SingleValuePruneRule.Config>
    implements TransformationRule {

  /** Creates a SingleValuePruneRule. */
  protected SingleValuePruneRule(SingleValuePruneRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final RelBuilder builder = call.builder();
    final RexBuilder rexBuilder = builder.getRexBuilder();
    final Aggregate aggregate = call.rel(0);
    final RelNode input = aggregate.getInput();
    final RelMetadataQuery mq = call.getMetadataQuery();
    // getMaxRowCount gives the maximum value that can be statically
    // determined. getMinRowCount does the same but for the minimum.
    //
    // If either of these cannot be determined it returns null,
    // which includes table scans returning null.
    Double maxInputRowCount = mq.getMaxRowCount(input);
    Double minInputRowCount = mq.getMinRowCount(input);
    // If the input is size one we can prune the aggregate and replace
    // it with an equivalent Projection.
    if (minInputRowCount.equals(maxInputRowCount)
        && maxInputRowCount != null
        && maxInputRowCount == 1D) {
      builder.push(input);
      List<RexNode> inputs = new ArrayList();
      // Note: we only call this on all SINGLE_VALUE calls
      for (int i = 0; i < aggregate.getAggCallList().size(); i++) {
        AggregateCall aggCall = aggregate.getAggCallList().get(i);
        assert aggCall.getArgList().size() == 1;
        int arg = aggCall.getArgList().get(0);
        // Note: We just directly use the type from the aggregate to avoid a type change.
        // This can happen if our value is more specific.
        inputs.add(
            rexBuilder.makeCast(
                aggregate.getRowType().getFieldList().get(i).getType(),
                builder.field(arg),
                true,
                false));
      }
      // Build the projection.
      builder.project(inputs, aggregate.getRowType().getFieldNames(), true);
      call.transformTo(builder.build());
    }
  }

  @Value.Immutable
  public interface Config extends RelRule.Config {
    SingleValuePruneRule.Config DEFAULT =
        ImmutableSingleValuePruneRule.Config.of().withOperandFor(Aggregate.class);

    @Override
    default SingleValuePruneRule toRule() {
      return new SingleValuePruneRule(this);
    }

    /** Defines an operand tree for the given 3 classes. */
    default SingleValuePruneRule.Config withOperandFor(Class<? extends Aggregate> aggregateClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(aggregateClass)
                      .predicate(SingleValuePruneRule::isSingleValuesNoGroups)
                      .anyInputs())
          .as(SingleValuePruneRule.Config.class);
    }
  }

  /**
   * Determine if an aggregate contains only SINGLE_VALUE calls without a group by.
   *
   * @param aggregate The aggregate to check.
   * @return Does the aggregate match the pattern.
   */
  private static boolean isSingleValuesNoGroups(Aggregate aggregate) {
    if (aggregate.getGroupCount() == 0 && !aggregate.getAggCallList().isEmpty()) {
      return aggregate.getAggCallList().stream()
          .allMatch(x -> x.getAggregation() == SqlStdOperatorTable.SINGLE_VALUE);
    } else {
      return false;
    }
  }
}
