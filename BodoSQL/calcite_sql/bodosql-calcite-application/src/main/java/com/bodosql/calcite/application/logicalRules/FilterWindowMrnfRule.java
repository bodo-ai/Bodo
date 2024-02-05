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
package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalMinRowNumberFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalWindow;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Window;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that takes a filter on top of a window node where the only windowed computation is a
 * MinRowNumberFilter and where the filter refers to the result of the windowed comptuation then
 * transforms into a MinRowNumberFilter RelNode. For example:
 *
 * <pre>
 * <code>
 *    Filter(AND(=($0, $1), $3))
 *        Window(groups=[[partition=[$1], order=[$2], calls=[MIN_ROW_NUMBER_FILTER()]]], inputsToKeep=[0, 1, 2])
 * </code>
 * </pre>
 *
 * Becomes the following:
 *
 * <pre>
 * <code>
 *    Project(A=[$0], B=[$1], C=[$2], DUMMY=[true])
 *        Filter=($0, $1)
 *            MinRowNumberFilter(partition=[$1], order=[$2], inputsToKeep=[0, 1, 2])
 * </code>
 * </pre>
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterWindowMrnfRule extends RelRule<FilterWindowMrnfRule.Config>
    implements TransformationRule {

  /** Creates a ProjectSetOpTransposeRule. */
  protected FilterWindowMrnfRule(Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    Filter origFilter = call.rel(0);
    BodoLogicalWindow window = call.rel(1);
    RelBuilder builder = call.builder();
    RelNode input = window.getInput();
    List<RexNode> newPredicates = new ArrayList<>();
    List<RexNode> trimExprs = new ArrayList<>();
    int mrnfIdx = window.getRowType().getFieldCount() - 1;
    boolean foundMatch = false;

    for (RexNode pred : RelOptUtil.conjunctions(origFilter.getCondition())) {
      if ((pred instanceof RexInputRef) && (((RexInputRef) pred).getIndex() == mrnfIdx)) {
        foundMatch = true;
      } else {
        // If another condition references the MRNF column, abort
        if (RelOptUtil.InputFinder.bits(List.of(pred), null).get(mrnfIdx)) {
          return;
        }
        newPredicates.add(pred);
      }
    }
    // If none of the conditions reference the MRNF column, abort
    if (!foundMatch) return;
    for (int i = 0; i < mrnfIdx; i++) {
      RelDataTypeField typ = window.getRowType().getFieldList().get(i);
      trimExprs.add(new RexInputRef(i, typ.getType()));
    }
    RexNode asOver = window.convertToProjExprs().get(mrnfIdx);
    builder.push(BodoLogicalMinRowNumberFilter.create(input, asOver));
    if (newPredicates.size() > 0) {
      builder.filter(newPredicates);
    }
    // Add the dummy boolean to replace the filter column
    trimExprs.add(builder.getRexBuilder().makeLiteral(true));
    builder.project(trimExprs);
    RelNode result = builder.build();
    call.transformTo(result);
  }

  private static boolean isSingletonMrnfWindow(BodoLogicalWindow window) {
    if (window.groups.size() != 1) return false;
    Window.Group group = window.groups.get(0);
    if (group.aggCalls.size() != 1) return false;
    RexCall call = group.aggCalls.get(0);
    return call.getKind() == SqlKind.MIN_ROW_NUMBER_FILTER;
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    Config DEFAULT =
        ImmutableFilterWindowMrnfRule.Config.builder()
            .build()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalFilter.class)
                        .oneInput(
                            b1 ->
                                b1.operand(BodoLogicalWindow.class)
                                    .predicate(FilterWindowMrnfRule::isSingletonMrnfWindow)
                                    .anyInputs()));

    @Override
    default FilterWindowMrnfRule toRule() {
      return new FilterWindowMrnfRule(this);
    }
  }
}
