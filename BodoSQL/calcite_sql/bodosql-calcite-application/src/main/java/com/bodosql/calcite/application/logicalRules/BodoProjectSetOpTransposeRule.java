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
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.SetOp;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.PushProjector;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexOver;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.logical.LogicalProject} past a {@link
 * org.apache.calcite.rel.core.SetOp}.
 *
 * <p>The children of the {@code SetOp} will project only the {@link RexInputRef}s referenced in the
 * original {@code LogicalProject}.
 *
 * <p>Bodo Change: The Bodo version is an exact copy of the calcite rule, but it doesn't require the
 * input to be exclusively a logical project.
 *
 * @see CoreRules#PROJECT_SET_OP_TRANSPOSE
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class BodoProjectSetOpTransposeRule extends RelRule<BodoProjectSetOpTransposeRule.Config>
    implements TransformationRule {

  /** Creates a ProjectSetOpTransposeRule. */
  protected BodoProjectSetOpTransposeRule(Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    // BODO CHANGE: changed from LogicalProject to Project
    final Project origProject = call.rel(0);
    final SetOp setOp = call.rel(1);

    // cannot push project past a distinct
    if (!setOp.all) {
      return;
    }

    // locate all fields referenced in the projection
    final PushProjector pushProjector =
        new PushProjector(origProject, null, setOp, config.preserveExprCondition(), call.builder());
    pushProjector.locateAllRefs();

    final List<RelNode> newSetOpInputs = new ArrayList<>();
    final int[] adjustments = pushProjector.getAdjustments();

    final RelNode node;
    if (origProject.containsOver()) {
      // should not push over past set-op but can push its operand down.
      for (RelNode input : setOp.getInputs()) {
        Project p = pushProjector.createProjectRefsAndExprs(input, true, false);
        // make sure that it is not a trivial project to avoid infinite loop.
        if (p.getRowType().equals(input.getRowType())) {
          return;
        }
        newSetOpInputs.add(p);
      }
      final SetOp newSetOp = setOp.copy(setOp.getTraitSet(), newSetOpInputs);
      node = pushProjector.createNewProject(newSetOp, adjustments);
    } else {
      // push some expressions below the set-op; this
      // is different from pushing below a join, where we decompose
      // to try to keep expensive expressions above the join,
      // because UNION ALL does not have any filtering effect,
      // and it is the only operator this rule currently acts on
      setOp
          .getInputs()
          .forEach(
              input ->
                  newSetOpInputs.add(
                      pushProjector.createNewProject(
                          pushProjector.createProjectRefsAndExprs(input, true, false),
                          adjustments)));
      node = setOp.copy(setOp.getTraitSet(), newSetOpInputs);
    }

    call.transformTo(node);
  }

  /** Rule configuration. */
  @Value.Immutable(singleton = false)
  public interface Config extends RelRule.Config {
    Config DEFAULT =
        ImmutableBodoProjectSetOpTransposeRule.Config.builder()
            .withPreserveExprCondition(expr -> !RexOver.containsOver(expr))
            .build()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Project.class).oneInput(b1 -> b1.operand(SetOp.class).anyInputs()));

    @Override
    default BodoProjectSetOpTransposeRule toRule() {
      return new BodoProjectSetOpTransposeRule(this);
    }

    /** Defines when an expression should not be pushed. */
    PushProjector.ExprCondition preserveExprCondition();

    /** Sets {@link #preserveExprCondition()}. */
    Config withPreserveExprCondition(PushProjector.ExprCondition condition);
  }
}
