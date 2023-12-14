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

import com.bodosql.calcite.application.operatorTables.CondOperatorTable;
import com.bodosql.calcite.application.operatorTables.NumericOperatorTable;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Aggregate.Group;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Filter} past a {@link
 * org.apache.calcite.rel.core.Aggregate}. We add the additional restriction that the filter can't
 * contain any Window functions, which is only possible due to our other rules.
 *
 * @see org.apache.calcite.rel.rules.AggregateFilterTransposeRule
 * @see CoreRules#FILTER_AGGREGATE_TRANSPOSE
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterAggregateTransposeRuleNoWindow
    extends RelRule<FilterAggregateTransposeRuleNoWindow.Config> implements TransformationRule {

  /** Creates a FilterAggregateTransposeRuleNoWindow. */
  protected FilterAggregateTransposeRuleNoWindow(
      FilterAggregateTransposeRuleNoWindow.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Filter filterRel = call.rel(0);
    final Aggregate aggRel = call.rel(1);
    final int groupCount = aggRel.getGroupCount();
    if (groupCount == 0) {
      // We can not push the Filter pass the Aggregate without group keys. The whole
      // input dataset would be the only group if there is no GROUP BY. Think about the case:
      // 'select count(*) from T1 having false', the result is expected to be an empty set,
      // but it would return zero if we push the Filter pass the Aggregate.
      return;
    }

    final List<RexNode> conditions = RelOptUtil.conjunctions(filterRel.getCondition());
    final RexBuilder rexBuilder = filterRel.getCluster().getRexBuilder();
    final List<RelDataTypeField> origFields = aggRel.getRowType().getFieldList();
    final int[] adjustments = new int[origFields.size()];
    int j = 0;
    for (int i : aggRel.getGroupSet()) {
      adjustments[j] = i - j;
      j++;
    }
    final List<RexNode> pushedConditions = new ArrayList<>();
    final List<RexNode> remainingConditions = new ArrayList<>();

    // Bodo Change: IS NOT NULL is a special filter because it can be pushed past
    // certain aggregations. This is only safe if the IS NOT NULL is shared for
    // every aggregate call and the function can only produce null when the entire
    // group is null.
    //
    // Right now we require there be exactly one aggregate, but this is also safe so long
    // as for every aggregate we are pushing is not null and they share an input column.

    List<AggregateCall> aggregateCalls = aggRel.getAggCallList();
    boolean pushIsNotNull =
        aggregateCalls.size() == 1
            && !aggregateCalls.get(0).hasFilter()
            && NULL_IF_ALL_NULL.contains(aggregateCalls.get(0).getAggregation().getName());

    for (RexNode condition : conditions) {
      ImmutableBitSet rCols = RelOptUtil.InputFinder.bits(condition);
      if (canPush(aggRel, rCols)) {
        pushedConditions.add(
            condition.accept(
                new RelOptUtil.RexInputConverter(
                    rexBuilder,
                    origFields,
                    aggRel.getInput(0).getRowType().getFieldList(),
                    adjustments)));

      } else if (pushIsNotNull
          && condition.getKind() == SqlKind.IS_NOT_NULL
          && ((RexCall) condition).getOperands().get(0) instanceof RexInputRef) {
        // Remap to the original input ref. Note all these functions accept 1 argument.
        Integer input = aggregateCalls.get(0).getArgList().get(0);
        // Note: Since we assume this only happens at most once we don't need to extract it from the
        // loop.
        final int[] notNullAdjustments = new int[origFields.size()];
        for (int i = groupCount; i < aggRel.getRowType().getFieldCount(); i++) {
          notNullAdjustments[j] = input - i;
        }

        pushedConditions.add(
            condition.accept(
                new RelOptUtil.RexInputConverter(
                    rexBuilder,
                    origFields,
                    aggRel.getInput(0).getRowType().getFieldList(),
                    notNullAdjustments)));
      } else {
        remainingConditions.add(condition);
      }
    }

    final RelBuilder builder = call.builder();
    RelNode rel = builder.push(aggRel.getInput()).filter(pushedConditions).build();
    if (rel == aggRel.getInput(0)) {
      return;
    }
    rel = aggRel.copy(aggRel.getTraitSet(), ImmutableList.of(rel));
    rel = builder.push(rel).filter(remainingConditions).build();
    call.transformTo(rel);
  }

  private static boolean canPush(Aggregate aggregate, ImmutableBitSet rCols) {
    // If the filter references columns not in the group key, we cannot push
    final ImmutableBitSet groupKeys =
        ImmutableBitSet.range(0, aggregate.getGroupSet().cardinality());
    if (!groupKeys.contains(rCols)) {
      return false;
    }

    if (aggregate.getGroupType() != Group.SIMPLE) {
      // If grouping sets are used, the filter can be pushed if
      // the columns referenced in the predicate are present in
      // all the grouping sets.
      for (ImmutableBitSet groupingSet : aggregate.getGroupSets()) {
        if (!groupingSet.contains(rCols)) {
          return false;
        }
      }
    }
    return true;
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterAggregateTransposeRuleNoWindow.Config DEFAULT =
        ImmutableFilterAggregateTransposeRuleNoWindow.Config.of()
            .withOperandFor(Filter.class, Aggregate.class);

    @Override
    default FilterAggregateTransposeRuleNoWindow toRule() {
      return new FilterAggregateTransposeRuleNoWindow(this);
    }

    /** Defines an operand tree for the given 2 classes. */
    default FilterAggregateTransposeRuleNoWindow.Config withOperandFor(
        Class<? extends Filter> filterClass, Class<? extends Aggregate> aggregateClass) {
      // Bodo Change: Add a requirement that there are no window functions in the filter.
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(f -> !f.containsOver())
                      .oneInput(b1 -> b1.operand(aggregateClass).anyInputs()))
          .as(FilterAggregateTransposeRuleNoWindow.Config.class);
    }

    /** Defines an operand tree for the given 3 classes. */
    default FilterAggregateTransposeRuleNoWindow.Config withOperandFor(
        Class<? extends Filter> filterClass,
        Class<? extends Aggregate> aggregateClass,
        Class<? extends RelNode> relClass) {
      // Bodo Change: Add a requirement that there are no window functions in the filter.
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(f -> !f.containsOver())
                      .oneInput(
                          b1 ->
                              b1.operand(aggregateClass)
                                  .oneInput(b2 -> b2.operand(relClass).anyInputs())))
          .as(FilterAggregateTransposeRuleNoWindow.Config.class);
    }
  }

  // Bodo Change: Define functions that can only be null if the whole group is null.
  private static Set<String> NULL_IF_ALL_NULL =
      Set.of(
          SqlStdOperatorTable.MAX.getName(),
          SqlStdOperatorTable.MIN.getName(),
          SqlStdOperatorTable.SUM.getName(),
          SqlStdOperatorTable.AVG.getName(),
          NumericOperatorTable.MEDIAN.getName(),
          NumericOperatorTable.BITAND_AGG.getName(),
          NumericOperatorTable.BITOR_AGG.getName(),
          NumericOperatorTable.BITXOR_AGG.getName(),
          CondOperatorTable.BOOLAND_AGG.getName(),
          CondOperatorTable.BOOLOR_AGG.getName(),
          CondOperatorTable.BOOLXOR_AGG.getName());
}
