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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that ejects RexOver calls from a Filter node. For example:
 *
 * <pre>
 * <code>
 *     Filter(AND(=($0, $1), MIN_ROW_NUMBER_FILTER() OVER (PARTITION BY $1 ORDER BY $2)))
 *        Project(A=[...], B=[...], C=[...])
 * </code>
 * </pre>
 *
 * Becomes the following:
 *
 * <pre>
 * <code>
 *     Project(A=$0, B=$1, C=$2)
 *        Filter(AND(=($0, $1), $3))
 *           Project(A=$0, B=$1, C=$2, D=MIN_ROW_NUMBER_FILTER() OVER (PARTITION BY $1 ORDER BY $2))
 *              Project(A=[...], B=[...], C=[...])
 * </code>
 * </pre>
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterWindowEjectRule extends RelRule<FilterWindowEjectRule.Config>
    implements TransformationRule {

  /** Creates a FilterWindowEjectRule. */
  protected FilterWindowEjectRule(Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    Filter origFilter = call.rel(0);
    RexNode origCond = origFilter.getCondition();
    RelBuilder builder = call.builder();
    List<RexNode> projExprs = new ArrayList<>();
    RelDataType inRowType = origFilter.getInput().getRowType();
    for (int i = 0; i < inRowType.getFieldCount(); i++) {
      projExprs.add(new RexInputRef(i, inRowType.getFieldList().get(i).getType()));
    }
    List<RexNode> origExprs = new ArrayList<>();
    origExprs.addAll(projExprs);
    RexShuttle overReplacer = new OverReplacer(projExprs);
    RexNode newCond = origCond.accept(overReplacer);
    builder.push(origFilter.getInput());
    builder.project(projExprs);
    builder.filter(newCond);
    builder.project(origExprs, origFilter.getRowType().getFieldNames());
    call.transformTo(builder.build());
  }

  /** Finds the highest level used by any of the inputs of a given expression. */
  private static class OverReplacer extends RexShuttle {
    final List<RexNode> projList;
    HashMap<RexNode, RexNode> replacementMap;

    public OverReplacer(List<RexNode> projList) {
      this.projList = projList;
      this.replacementMap = new HashMap<>();
    }

    @Override
    public RexNode visitOver(RexOver over) {
      if (!replacementMap.containsKey(over)) {
        RexInputRef newRef = new RexInputRef(projList.size(), over.getType());
        projList.add(over);
        replacementMap.put(over, newRef);
      }
      return replacementMap.get(over);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    Config DEFAULT =
        ImmutableFilterWindowEjectRule.Config.builder()
            .build()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalFilter.class)
                        .predicate(f -> f.containsOver())
                        .anyInputs());

    @Override
    default FilterWindowEjectRule toRule() {
      return new FilterWindowEjectRule(this);
    }
  }
}
