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
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Project} past a {@link
 * org.apache.calcite.rel.core.Filter}.
 *
 * <p>Heavily based on org.apache.calcite.rel.rules.ProjectFilterTransposeRule.
 *
 * <p>However, unlike the Calcite version of this rule, this rule restricts pushdown to expressions
 * that can eventually be pushed into Snowflake.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class ProjectFilterTransposePushableExpressionsRule
    extends RelRule<ProjectFilterTransposePushableExpressionsRule.Config>
    implements TransformationRule {

  /** Creates a ProjectFilterTransposeRule. */
  protected ProjectFilterTransposePushableExpressionsRule(Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project origProject;
    final Filter filter;
    if (call.rels.length >= 2) {
      origProject = call.rel(0);
      filter = call.rel(1);
    } else {
      origProject = null;
      filter = call.rel(0);
    }

    if (origProject != null && origProject.containsOver()) {
      // Cannot push project through filter if project contains a windowed
      // aggregate -- it will affect row counts. Abort this rule
      // invocation; pushdown will be considered after the windowed
      // aggregate has been implemented. It's OK if the filter contains a
      // windowed aggregate.
      return;
    }

    if ((origProject != null)
        && origProject.getRowType().isStruct()
        && origProject.getRowType().getFieldList().stream()
            .anyMatch(RelDataTypeField::isDynamicStar)) {
      // The PushProjector would change the plan:
      //
      //    prj(**=[$0])
      //    : - filter
      //        : - scan
      //
      // to form like:
      //
      //    prj(**=[$0])                    (1)
      //    : - filter                      (2)
      //        : - prj(**=[$0], ITEM= ...) (3)
      //            :  - scan
      // This new plan has more cost that the old one, because of the new
      // redundant project (3), if we also have FilterProjectTransposeRule in
      // the rule set, it will also trigger infinite match of the ProjectMergeRule
      // for project (1) and (3).
      return;
    }

    // BODO CHANGE: Everything below this point was removed and replaced with new logic
    RelNode newProject =
        SnowflakeProjectPushdownHelpers.Companion.replaceValuesProjectFilter(
            origProject, filter, call.builder());
    if (newProject != null) {
      call.transformTo(newProject);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    Config DEFAULT = ImmutableProjectFilterTransposePushableExpressionsRule.Config.of();

    @Override
    default ProjectFilterTransposePushableExpressionsRule toRule() {
      return new ProjectFilterTransposePushableExpressionsRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default Config withOperandFor(
        Class<? extends Project> projectClass, Class<? extends Filter> filterClass) {
      return withOperandSupplier(
              b0 -> b0.operand(projectClass).oneInput(b1 -> b1.operand(filterClass).anyInputs()))
          .as(Config.class);
    }

    @Override
    @Value.Default
    default OperandTransform operandSupplier() {
      return b0 ->
          b0.operand(LogicalProject.class)
              .oneInput(b1 -> b1.operand(LogicalFilter.class).anyInputs());
    }

    /** Defines an operand tree for the given 3 classes. */
    default Config withOperandFor(
        Class<? extends Project> projectClass,
        Class<? extends Filter> filterClass,
        Class<? extends RelNode> inputClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(projectClass)
                      .oneInput(
                          b1 ->
                              b1.operand(filterClass)
                                  .oneInput(b2 -> b2.operand(inputClass).anyInputs())))
          .as(Config.class);
    }
  }
}
