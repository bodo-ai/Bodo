package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.adapter.bodo.BodoPhysicalAggregate;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.traits.ExpectedBatchingProperty;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.immutables.value.Value;

/**
 * Planner rule that converts a group by containing grouping sets into a union all of the group by
 * inputs. This rule depends on caching to be effective as otherwise this can lead to a significant
 * performance regression. As a result, this rule is only designed to run on the Physical plan after
 * other optimizations have finished.
 *
 * <p>For example, this converts an Aggregate like: <code>
 * Aggregate(group=[{0, 1}] groupingSets=[{0}, {1}, {}], agg#0=[SUM($2)], agg#1=[SUM($3)])
 * </code>
 *
 * <p>Into a UnionAll of the following Aggregates: <code>
 *   UNION(all=[true])
 *      Project($0, NULL, $1, $2)
 *          Aggregate(group=[{0}], agg#0=[SUM($2)], agg#1=[SUM($3)])
 *              INPUT
 *      Project(NULL, $0, $1, $2)
 *          Aggregate(group=[{1}], agg#0=[SUM($2)], agg#1=[SUM($3)])
 *              INPUT
 *      Project(NULL, NULL, $0, $1)
 *          Aggregate(group=[{}], agg#0=[SUM($2)], agg#1=[SUM($3)])
 *              INPUT
 *
 * </code>
 *
 * <p>Since this runs after other optimizations, we expect INPUT to be converted to a cache node. In
 * some situations where one aggregate is a super set of all others and every aggregate function is
 * splittable, it's possible to generate partial aggregations instead. However, this is not yet
 * implemented.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class GroupingSetsToUnionAllRule extends RelRule<GroupingSetsToUnionAllRule.Config>
    implements TransformationRule {

  /** Creates a GroupingSetsToUnionAllRule. */
  protected GroupingSetsToUnionAllRule(GroupingSetsToUnionAllRule.Config config) {
    super(config);
  }

  /**
   * Builds the aggregation for a given grouping set. This may also apply a projection to ensure
   * type stability for the UNION ALL and to implement GROUPING.
   *
   * @param input The input to the aggregation.
   * @param builder The RelBuilder to use.
   * @param baseGroupSet The superSet of all possible group by keys.
   * @param groupingSet The set of keys for the current aggregation.
   * @param aggCallList The list of aggregate calls to apply. If a GROUPING is found it will be
   *     skipped in the aggregation and replaced a constant.
   * @return A new aggregation with a projection if necessary for any constant to ensure type
   *     stability with the original aggregation.
   */
  private RelNode buildAggregation(
      RelBuilder builder,
      RelNode input,
      ImmutableBitSet baseGroupSet,
      ImmutableBitSet groupingSet,
      List<AggregateCall> aggCallList) {
    builder.push(input);
    RexBuilder rexBuilder = builder.getRexBuilder();
    // Select a subset of the aggregate calls without group.
    List<AggregateCall> filteredAggCallList =
        aggCallList.stream()
            .filter(aggCall -> aggCall.getAggregation().kind != SqlKind.GROUPING)
            .collect(Collectors.toList());
    builder.aggregate(builder.groupKey(groupingSet), filteredAggCallList);
    // Build the projection on top for type stability.
    int currentIndex = 0;
    List<RexNode> fields = new ArrayList<>();
    for (int key : baseGroupSet) {
      if (groupingSet.get(key)) {
        fields.add(builder.field(currentIndex++));
      } else {
        fields.add(
            rexBuilder.makeNullLiteral(input.getRowType().getFieldList().get(key).getType()));
      }
    }
    for (AggregateCall aggCall : aggCallList) {
      if (aggCall.getAggregation().kind == SqlKind.GROUPING) {
        long literalGroupingValue = getGroupingValue(baseGroupSet, groupingSet);
        fields.add(rexBuilder.makeLiteral(literalGroupingValue, aggCall.getType()));
      } else {
        fields.add(builder.field(currentIndex++));
      }
    }
    RelNode output = builder.project(fields).build();
    return output;
  }

  /**
   * Returns the grouping value for a given grouping set. Here we depend on group by always
   * enforcing the full group set to be stored in ascending order.
   *
   * @param fullGroupSet The full group set.
   * @param groupSet The grouping set.
   * @return The grouping value.
   */
  static long getGroupingValue(ImmutableBitSet fullGroupSet, ImmutableBitSet groupSet) {
    long v = 0;
    long x = 1L << (fullGroupSet.cardinality() - 1);
    for (int i : fullGroupSet) {
      if (!groupSet.get(i)) {
        v |= x;
      }
      x >>= 1;
    }
    return v;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final RelBuilder builder = call.builder();
    final List<ImmutableBitSet> groupSets = aggregate.getGroupSets();
    // Create the cache node.
    final RelNode input = aggregate.getInput();
    // Create each aggregation.
    List<RelNode> inputs =
        groupSets.stream()
            .map(
                groupingSet ->
                    buildAggregation(
                        builder,
                        input,
                        aggregate.getGroupSet(),
                        groupingSet,
                        aggregate.getAggCallList()))
            .collect(Collectors.toList());
    // Create the union all.
    builder.pushAll(inputs);
    builder.union(true, inputs.size());
    RelNode output = builder.build();
    call.transformTo(output);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    GroupingSetsToUnionAllRule.Config DEFAULT =
        ImmutableGroupingSetsToUnionAllRule.Config.of().withOperandFor(BodoPhysicalAggregate.class);

    @Override
    default GroupingSetsToUnionAllRule toRule() {
      return new GroupingSetsToUnionAllRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default GroupingSetsToUnionAllRule.Config withOperandFor(
        Class<? extends BodoPhysicalAggregate> aggregateClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(aggregateClass)
                      // Simple aggregates have exactly 1 grouping set.
                      // All others have multiple. There is a typing bug in
                      // Calcite that incorrectly thinks that empty grouping
                      // sets can have a non-nullable output, so we skip this
                      // rule in that case.
                      .predicate(
                          a ->
                              !Aggregate.isSimple(a)
                                  && a.getGroupSets().stream().allMatch(s -> !s.isEmpty())
                                  && a.getAggCallList().stream()
                                      .anyMatch(
                                          agg ->
                                              !ExpectedBatchingProperty
                                                  .streamingSupportedWithoutAccumulateAggFunction(
                                                      agg)))
                      .anyInputs())
          .as(GroupingSetsToUnionAllRule.Config.class);
    }
  }
}
