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
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.SetOp;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Filter} past a {@link
 * org.apache.calcite.rel.core.SetOp}.
 *
 * <p>We have updated this rule to require the filter not contain any window functions.
 *
 * @see CoreRules#FILTER_SET_OP_TRANSPOSE
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterSetOpTransposeRuleNoWindow
    extends RelRule<FilterSetOpTransposeRuleNoWindow.Config> implements TransformationRule {

  /** Creates a FilterSetOpTransposeRule. */
  protected FilterSetOpTransposeRuleNoWindow(FilterSetOpTransposeRuleNoWindow.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    Filter filterRel = call.rel(0);
    SetOp setOp = call.rel(1);

    RexNode condition = filterRel.getCondition();

    // create filters on top of each setop child, modifying the filter
    // condition to reference each setop child
    RexBuilder rexBuilder = filterRel.getCluster().getRexBuilder();
    final RelBuilder relBuilder = call.builder();
    List<RelDataTypeField> origFields = setOp.getRowType().getFieldList();
    int[] adjustments = new int[origFields.size()];
    final List<RelNode> newSetOpInputs = new ArrayList<>();
    for (RelNode input : setOp.getInputs()) {
      RexNode newCondition =
          condition.accept(
              new RelOptUtil.RexInputConverter(
                  rexBuilder, origFields, input.getRowType().getFieldList(), adjustments));
      newSetOpInputs.add(relBuilder.push(input).filter(newCondition).build());
    }

    // create a new setop whose children are the filters created above
    SetOp newSetOp = setOp.copy(setOp.getTraitSet(), newSetOpInputs);

    call.transformTo(newSetOp);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterSetOpTransposeRuleNoWindow.Config DEFAULT =
        ImmutableFilterSetOpTransposeRuleNoWindow.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Filter.class)
                        .predicate(f -> !f.containsOver())
                        .oneInput(b1 -> b1.operand(SetOp.class).anyInputs()));

    @Override
    default FilterSetOpTransposeRuleNoWindow toRule() {
      return new FilterSetOpTransposeRuleNoWindow(this);
    }
  }
}
