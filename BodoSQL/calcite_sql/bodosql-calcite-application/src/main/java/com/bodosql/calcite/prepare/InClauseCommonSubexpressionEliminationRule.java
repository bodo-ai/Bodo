/*
 * This file is modified from Dremio's implementation.
 *
 * Copyright (C) 2017-2019 Dremio Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.bodosql.calcite.prepare;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.immutables.value.Value;

/**
 * For a query like:
 *
 * <p>SELECT * FROM people WHERE FOO(name) IN ('son', 'john', 'bob')
 *
 * <p>It can get translated to:
 *
 * <p>SELECT * FROM people WHERE (FOO(name) = 'son') OR (FOO(name) = 'john') OR (FOO(name) = 'bob')
 *
 * <p>Notice that FOO(name) gets evaluated 3 times and FOO(x) might be an expensive expression to
 * evaluate.
 *
 * <p>We can apply common subexpression elimination (CSE) to mitigate this by rewriting the query
 * like so:
 *
 * <p>let temp = FOO(name); SELECT * FROM people WHERE temp IN ('son', 'john', 'bob')
 *
 * <p>Basically store the result of the common subexpression in a temp variable (or projection rel
 * node) and reference that.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public final class InClauseCommonSubexpressionEliminationRule
    extends RelRule<InClauseCommonSubexpressionEliminationRule.Config> {

  private InClauseCommonSubexpressionEliminationRule(
      InClauseCommonSubexpressionEliminationRule.Config config) {
    super(config);
  }

  // TODO(aneesh) this could probably be done with a predicate call in the Config instead.
  @Override
  public boolean matches(RelOptRuleCall relOptRuleCall) {
    final Filter filterRel = relOptRuleCall.rel(0);
    return RelOptUtil.conjunctions(filterRel.getCondition()).stream()
        .map(RexNode::getKind)
        .anyMatch(sqlKind -> sqlKind == SqlKind.OR);
  }

  public static List<RexNode> identityProjects(RelDataType type) {
    List<RexNode> projects = new ArrayList<>();
    List<RelDataTypeField> fieldList = type.getFieldList();
    for (int i = 0; i < type.getFieldCount(); i++) {
      RelDataTypeField field = fieldList.get(i);
      projects.add(new RexInputRef(i, field.getType()));
    }
    return projects;
  }

  // On matched filters, replace the filter condition with a rewritten condition that aims to
  // eliminate common subexpressions within.
  @Override
  public void onMatch(RelOptRuleCall relOptRuleCall) {
    final Filter filterRel = relOptRuleCall.rel(0);
    // This is safe because we know we only matched ORs, so the condition is definitely a RexCall.
    final RexCall condition = (RexCall) filterRel.getCondition();

    Set<RexNode> complexCommonSubexpressions = getCommonComplexSubexpressions(condition);
    if (complexCommonSubexpressions.isEmpty()) {
      return;
    }

    final List<RexNode> projectNodes = identityProjects(filterRel.getInput().getRowType());
    final Map<RexNode, Integer> commonSubExpressionToProjectionIndex = new HashMap<>();
    for (RexNode commonSubexpression : complexCommonSubexpressions) {
      int index = projectNodes.size();
      projectNodes.add(commonSubexpression);
      commonSubExpressionToProjectionIndex.put(commonSubexpression, index);
    }

    RexNode rewrittenCondition =
        getRewrittenFilter(filterRel, commonSubExpressionToProjectionIndex);

    List<RexNode> originalColumns = identityProjects(filterRel.getInput().getRowType());

    final RelNode rewrittenRelNode =
        relOptRuleCall
            .builder()
            .push(filterRel.getInput())
            // Eliminate the complex common sub-expressions.
            // We do this by pushing the common sub-expression to the projection and referencing
            // that.
            .project(projectNodes)
            .filter(rewrittenCondition)
            // Drop the extra columns we introduced earlier
            .project(originalColumns)
            .build();

    relOptRuleCall.transformTo(rewrittenRelNode);
  }

  private static RexNode tryGetComplexChildExpression(RexNode node) {
    // Check two see if we have an expression in the form "x = a" or "a = x"
    // where 'x' is a complex expression
    // and 'a' is any expression.
    if (!(node instanceof RexCall)) {
      return null;
    }

    RexCall rexCall = (RexCall) node;
    if (rexCall.op.kind != SqlKind.EQUALS) {
      return null;
    }

    RexNode lhs = rexCall.operands.get(0);
    if (lhs instanceof RexCall) {
      return lhs;
    }

    RexNode rhs = rexCall.operands.get(1);
    if (rhs instanceof RexCall) {
      return rhs;
    }

    return null;
  }

  private static Set<RexNode> getCommonComplexSubexpressions(RexCall condition) {
    // Find all the duplicate complex common sub-expressions.
    // Only check for the following type of pattern:
    // (x = a) OR (x = b) OR (x = c) ... (x = n)
    // where x is non-trivial expression.
    // We don't attempt to look for nested sub expressions
    final ImmutableList<RexNode> operands = condition.operands;
    Set<RexNode> seenRexNodes = new HashSet<>();
    Set<RexNode> duplicateRexNodes = new HashSet<>();
    for (RexNode operand : operands) {
      RexNode complexChildExpression = tryGetComplexChildExpression(operand);
      if (complexChildExpression == null) {
        continue;
      }

      if (!seenRexNodes.add(complexChildExpression)) {
        duplicateRexNodes.add(complexChildExpression);
      }
    }

    return duplicateRexNodes;
  }

  private static RexNode getRewrittenFilter(
      Filter filterRel, Map<RexNode, Integer> commonSubexpressionToProjectionIndex) {
    // At this point all the common subexpressions are pushed to the projection
    // We need to rewrite the filter to reference these common subexpression from the projection.
    final RexCall condition = (RexCall) filterRel.getCondition();
    final ImmutableList<RexNode> operands = condition.operands;
    final RexBuilder rexBuilder = filterRel.getCluster().getRexBuilder();

    final List<RexNode> rewrittenOperands = new ArrayList<>();

    for (RexNode operand : operands) {
      RexNode complexChildExpression = tryGetComplexChildExpression(operand);
      RexNode rewrittenOperand;
      if (complexChildExpression == null) {
        rewrittenOperand = operand;
      } else {
        Integer projectionIndex = commonSubexpressionToProjectionIndex.get(complexChildExpression);
        if (projectionIndex == null || !(operand instanceof RexCall)) {
          rewrittenOperand = operand;
        } else {
          RexCall binaryExpression = (RexCall) operand;
          RexNode binaryOperand0 = binaryExpression.operands.get(0);
          RexNode binaryOperand1 = binaryExpression.operands.get(1);

          RexNode nonComplexSubexpression =
              complexChildExpression == binaryOperand0 ? binaryOperand1 : binaryOperand0;

          RexInputRef rexInputRef =
              new RexInputRef(projectionIndex, complexChildExpression.getType());
          List<RexNode> rewrittenBinaryOperands = new ArrayList<>();
          rewrittenBinaryOperands.add(rexInputRef);
          rewrittenBinaryOperands.add(nonComplexSubexpression);

          rewrittenOperand =
              rexBuilder.makeCall(
                  binaryExpression.type, binaryExpression.op, rewrittenBinaryOperands);
        }
      }

      rewrittenOperands.add(rewrittenOperand);
    }

    return rexBuilder.makeCall(condition.type, condition.op, rewrittenOperands);
  }

  @Value.Immutable
  public interface Config extends RelRule.Config {
    InClauseCommonSubexpressionEliminationRule.Config DEFAULT =
        ImmutableInClauseCommonSubexpressionEliminationRule.Config.of()
            .withOperandSupplier(b -> b.operand(Filter.class).anyInputs())
            .as(InClauseCommonSubexpressionEliminationRule.Config.class);

    @Override
    default InClauseCommonSubexpressionEliminationRule toRule() {
      return new InClauseCommonSubexpressionEliminationRule(this);
    }
  }
}
