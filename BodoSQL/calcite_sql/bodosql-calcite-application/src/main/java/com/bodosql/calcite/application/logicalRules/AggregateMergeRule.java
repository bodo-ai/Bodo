package com.bodosql.calcite.application.logicalRules;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Aggregate.Group;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlSplittableAggFunction;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.util.ImmutableBitSet;
import org.immutables.value.Value;

/**
 * Planner rule that matches an {@link Aggregate} on a {@link Aggregate} and the top aggregate's
 * group key is a subset of the lower aggregate's group key, and the aggregates are expansions of
 * rollups, then it would convert into a single aggregate.
 *
 * <p>For example, SUM of SUM becomes SUM; SUM of COUNT becomes COUNT; MAX of MAX becomes MAX; MIN
 * of MIN becomes MIN. AVG of AVG would not match, nor would COUNT of COUNT.
 *
 * <p>Bodo extends this rule to handle the special case where the top and bottom aggregates have the
 * exact same group keys. In this case, some restrictions can be relaxed.
 *
 * @see CoreRules#AGGREGATE_MERGE
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class AggregateMergeRule extends RelRule<AggregateMergeRule.Config>
    implements TransformationRule {

  /** Creates an AggregateMergeRule. */
  protected AggregateMergeRule(AggregateMergeRule.Config config) {
    super(config);
  }

  private static boolean isAggregateSupported(AggregateCall aggCall) {
    if (aggCall.isDistinct()
        || aggCall.hasFilter()
        || aggCall.isApproximate()
        || aggCall.getArgList().size() > 1) {
      return false;
    }
    return aggCall.getAggregation().maybeUnwrap(SqlSplittableAggFunction.class).isPresent();
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate topAgg = call.rel(0);
    final Aggregate bottomAgg = call.rel(1);
    if (topAgg.getGroupCount() > bottomAgg.getGroupCount()) {
      return;
    }

    final ImmutableBitSet bottomGroupSet = bottomAgg.getGroupSet();
    final Map<Integer, Integer> map = new HashMap<>();
    bottomGroupSet.forEachInt(v -> map.put(map.size(), v));
    for (int k : topAgg.getGroupSet()) {
      if (!map.containsKey(k)) {
        return;
      }
    }

    // top aggregate keys must be subset of lower aggregate keys
    final ImmutableBitSet topGroupSet = topAgg.getGroupSet().permute(map);
    if (!bottomGroupSet.contains(topGroupSet)) {
      return;
    }

    boolean hasEmptyGroup = topAgg.getGroupSets().stream().anyMatch(ImmutableBitSet::isEmpty);

    final List<AggregateCall> finalCalls = new ArrayList<>();
    for (AggregateCall topCall : topAgg.getAggCallList()) {
      if (!isAggregateSupported(topCall) || topCall.getArgList().size() == 0) {
        return;
      }
      // Make sure top aggregate argument refers to one of the aggregate
      int bottomIndex = topCall.getArgList().get(0) - bottomGroupSet.cardinality();
      if (bottomIndex >= bottomAgg.getAggCallList().size() || bottomIndex < 0) {
        return;
      }
      AggregateCall bottomCall = bottomAgg.getAggCallList().get(bottomIndex);
      // Bodo Change: When the exact same keys are present, check if uppermost
      // aggregate is one of SUM, MIN, or MAX. If so, we can just reuse the
      // lower aggregate as is. Note: We can't include SUM0 because we can't
      // convert NULL to 0.
      boolean matchingKeys = bottomGroupSet.equals(topGroupSet);
      Set<SqlKind> matchingKinds = Set.of(SqlKind.SUM, SqlKind.MIN, SqlKind.MAX);
      if (matchingKeys && matchingKinds.contains(topCall.getAggregation().kind)) {
        finalCalls.add(bottomCall);
      } else {
        // Should not merge if top agg with empty group keys and the lower agg
        // function is COUNT, because in case of empty input for lower agg,
        // the result is empty, if we merge them, we end up with 1 result with
        // 0, which is wrong.
        if (!isAggregateSupported(bottomCall)
            || (bottomCall.getAggregation() == SqlStdOperatorTable.COUNT
                && topCall.getAggregation().getKind() != SqlKind.SUM0
                && hasEmptyGroup)) {
          return;
        }
        SqlSplittableAggFunction splitter =
            bottomCall.getAggregation().unwrapOrThrow(SqlSplittableAggFunction.class);
        AggregateCall finalCall = splitter.merge(topCall, bottomCall);
        // fail to merge the aggregate call, bail out
        if (finalCall == null) {
          return;
        }
        finalCalls.add(finalCall);
      }
    }

    // re-map grouping sets
    ImmutableList<ImmutableBitSet> newGroupingSets = null;
    if (topAgg.getGroupType() != Group.SIMPLE) {
      newGroupingSets =
          ImmutableBitSet.ORDERING.immutableSortedCopy(
              ImmutableBitSet.permute(topAgg.getGroupSets(), map));
    }

    final Aggregate finalAgg =
        topAgg.copy(
            topAgg.getTraitSet(), bottomAgg.getInput(), topGroupSet, newGroupingSets, finalCalls);
    call.transformTo(finalAgg);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    AggregateMergeRule.Config DEFAULT =
        ImmutableAggregateMergeRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Aggregate.class)
                        .oneInput(
                            b1 ->
                                b1.operand(Aggregate.class)
                                    .predicate(Aggregate::isSimple)
                                    .anyInputs()))
            .as(AggregateMergeRule.Config.class);

    @Override
    default AggregateMergeRule toRule() {
      return new AggregateMergeRule(this);
    }
  }
}
