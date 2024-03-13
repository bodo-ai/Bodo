package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalJoin;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.plan.Strong;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.immutables.value.Value;

/**
 * Planner rule that derives IS NOT NULL predicates from a inner {@link
 * org.apache.calcite.rel.core.Join} and creates {@link org.apache.calcite.rel.core.Filter}s with
 * those predicates as new inputs of the join.
 *
 * <p>Since the Null value can never match in the inner join, and it can lead to skewness due to too
 * many Null values, a not-null filter can be created and pushed down into the input of join.
 *
 * <p>Similar to {@link CoreRules#FILTER_INTO_JOIN}, it would try to create filters and push them
 * into the inputs of the join to filter data as much as possible before join.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class BodoJoinDeriveIsNotNullFilterRule
    extends RelRule<BodoJoinDeriveIsNotNullFilterRule.Config> implements TransformationRule {

  public BodoJoinDeriveIsNotNullFilterRule(BodoJoinDeriveIsNotNullFilterRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    final RelBuilder relBuilder = call.builder();
    final RelMetadataQuery mq = call.getMetadataQuery();

    final ImmutableBitSet.Builder notNullableKeys = ImmutableBitSet.builder();
    RelOptUtil.InputFinder.bits(join.getCondition())
        .forEach(
            bit -> {
              if (Strong.isNotTrue(join.getCondition(), ImmutableBitSet.of(bit))) {
                notNullableKeys.set(bit);
              }
            });
    final List<Integer> leftKeys = new ArrayList<>();
    final List<Integer> rightKeys = new ArrayList<>();

    final int offset = join.getLeft().getRowType().getFieldCount();
    notNullableKeys
        .build()
        .asList()
        .forEach(
            i -> {
              if (i < offset) {
                leftKeys.add(i);
              } else {
                rightKeys.add(i - offset);
              }
            });

    // Nulls on the left means we have a RIGHT join, so we can only push
    // the filter on the left input.
    final boolean leftNulls = join.getJoinType().generatesNullsOnLeft();
    // Nulls on the right means we have a LEFT join, so we can only push
    // the filter on the right input.
    final boolean rightNulls = join.getJoinType().generatesNullsOnRight();

    relBuilder
        .push(join.getLeft())
        .withPredicates(
            mq,
            r ->
                r.filter(
                    leftKeys.stream()
                        .map(r::field)
                        .map(r::isNotNull)
                        .collect(Collectors.toList())));
    final RelNode newLeftCandidate = relBuilder.build();
    final RelNode newLeft;
    if (rightNulls) {
      newLeft = join.getLeft();
    } else {
      newLeft = newLeftCandidate;
    }

    relBuilder
        .push(join.getRight())
        .withPredicates(
            mq,
            r ->
                r.filter(
                    rightKeys.stream()
                        .map(r::field)
                        .map(r::isNotNull)
                        .collect(Collectors.toList())));
    final RelNode newRightCandidate = relBuilder.build();
    final RelNode newRight;
    if (leftNulls) {
      newRight = join.getRight();
    } else {
      newRight = newRightCandidate;
    }

    if (newLeft != join.getLeft() || newRight != join.getRight()) {
      final RelNode newJoin = join.copy(join.getTraitSet(), ImmutableList.of(newLeft, newRight));
      call.transformTo(newJoin);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    BodoJoinDeriveIsNotNullFilterRule.Config DEFAULT =
        ImmutableBodoJoinDeriveIsNotNullFilterRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(BodoLogicalJoin.class)
                        .predicate(
                            join ->
                                join.getJoinType() != JoinRelType.FULL
                                    && !join.getCondition().isAlwaysTrue())
                        .anyInputs());

    @Override
    default BodoJoinDeriveIsNotNullFilterRule toRule() {
      return new BodoJoinDeriveIsNotNullFilterRule(this);
    }
  }
}
