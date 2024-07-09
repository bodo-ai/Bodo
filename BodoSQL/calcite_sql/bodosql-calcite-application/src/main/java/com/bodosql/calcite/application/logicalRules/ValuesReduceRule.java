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

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptPredicateList;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.PruneEmptyRules;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.trace.CalciteTrace;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.immutables.value.Value;
import org.slf4j.Logger;

/**
 * Planner rule that folds projections and filters into an underlying {@link
 * org.apache.calcite.rel.core.Values}.
 *
 * <p>Returns a simplified {@code Values}, perhaps containing zero tuples if all rows are filtered
 * away.
 *
 * <p>For example,
 *
 * <blockquote>
 *
 * <code>select a - b from (values (1, 2), (3, 5), (7, 11)) as t (a,
 * b) where a + b &gt; 4</code>
 *
 * </blockquote>
 *
 * <p>becomes
 *
 * <blockquote>
 *
 * <code>select x from (values (-2), (-4))</code>
 *
 * </blockquote>
 *
 * <p>Ignores an empty {@code Values}; this is better dealt with by {@link PruneEmptyRules}.
 *
 * @see CoreRules#FILTER_VALUES_MERGE
 * @see CoreRules#PROJECT_VALUES_MERGE
 * @see CoreRules#PROJECT_FILTER_VALUES_MERGE
 *     <p>We extend this rule to patch the type restrictions + wrap
 *     ReduceExpressionsRule.reduceExpressions in a try-catch block to prevent the rule from failing
 *     when the expression is not reducible. We only copy over the components for the rule we use.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class ValuesReduceRule extends RelRule<ValuesReduceRule.Config>
    implements TransformationRule {

  private static final Logger LOGGER = CalciteTrace.getPlannerTracer();

  /** Creates a ValuesReduceRule. */
  protected ValuesReduceRule(ValuesReduceRule.Config config) {
    super(config);
    Util.discard(LOGGER);
  }

  private static void matchProject(ValuesReduceRule rule, RelOptRuleCall call) {
    Project project = call.rel(0);
    Values values = call.rel(1);
    rule.apply(call, project, null, values);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    config.matchHandler().accept(this, call);
  }

  /**
   * Does the work.
   *
   * @param call Rule call
   * @param project Project, may be null
   * @param filter Filter, may be null
   * @param values Values rel to be reduced
   */
  protected void apply(
      RelOptRuleCall call, @Nullable Project project, @Nullable Filter filter, Values values) {
    assert values != null;
    assert filter != null || project != null;
    final RexNode conditionExpr = (filter == null) ? null : filter.getCondition();
    final List<RexNode> projectExprs = (project == null) ? null : project.getProjects();
    RexBuilder rexBuilder = values.getCluster().getRexBuilder();

    // Find reducible expressions.
    final List<RexNode> reducibleExps = new ArrayList<>();
    final ValuesReduceRule.MyRexShuttle shuttle = new ValuesReduceRule.MyRexShuttle();
    for (final List<RexLiteral> literalList : values.getTuples()) {
      shuttle.literalList = literalList;
      if (conditionExpr != null) {
        RexNode c = conditionExpr.accept(shuttle);
        reducibleExps.add(c);
      }
      if (projectExprs != null) {
        requireNonNull(project, "project");
        int k = -1;
        for (RexNode projectExpr : projectExprs) {
          ++k;
          RexNode e = projectExpr.accept(shuttle);
          if (RexLiteral.isNullLiteral(e)) {
            RelDataType type = project.getRowType().getFieldList().get(k).getType();
            e = rexBuilder.makeAbstractCast(type, e, false);
          }
          reducibleExps.add(e);
        }
      }
    }
    int fieldsPerRow =
        ((conditionExpr == null) ? 0 : 1) + ((projectExprs == null) ? 0 : projectExprs.size());
    assert fieldsPerRow > 0;
    assert reducibleExps.size() == (values.getTuples().size() * fieldsPerRow);

    // Compute the values they reduce to.
    final RelOptPredicateList predicates = RelOptPredicateList.EMPTY;
    // Bodo Change: Wrap in a try catch to handle gaps in the shuttle.
    try {
      BodoSQLReduceExpressionsRule.reduceExpressions(
          values, reducibleExps, predicates, false, true, false);
    } catch (Exception e) {
      // Just exit if we can't reduce the expressions.
      return;
    }

    int changeCount = 0;
    final ImmutableList.Builder<ImmutableList<RexLiteral>> tuplesBuilder = ImmutableList.builder();
    for (int row = 0; row < values.getTuples().size(); ++row) {
      int i = 0;
      if (conditionExpr != null) {
        final RexNode reducedValue = reducibleExps.get((row * fieldsPerRow) + i);
        ++i;
        if (!reducedValue.isAlwaysTrue()) {
          ++changeCount;
          continue;
        }
      }

      final ImmutableList<RexLiteral> valuesList;
      if (projectExprs != null) {
        ++changeCount;
        final ImmutableList.Builder<RexLiteral> tupleBuilder = ImmutableList.builder();
        for (; i < fieldsPerRow; ++i) {
          final RexNode reducedValue = reducibleExps.get((row * fieldsPerRow) + i);
          if (reducedValue instanceof RexLiteral) {
            tupleBuilder.add((RexLiteral) reducedValue);
          } else if (RexUtil.isNullLiteral(reducedValue, true)) {
            tupleBuilder.add(rexBuilder.makeNullLiteral(reducedValue.getType()));
          } else {
            return;
          }
        }
        valuesList = tupleBuilder.build();
      } else {
        valuesList = values.getTuples().get(row);
      }
      tuplesBuilder.add(valuesList);
    }

    if (changeCount > 0) {
      final RelDataType rowType;
      if (projectExprs != null) {
        rowType = requireNonNull(project, "project").getRowType();
      } else {
        rowType = values.getRowType();
      }
      // Bodo Change: Use the RelBuilder to create the new RelNode.
      final RelNode newRel = call.builder().values(tuplesBuilder.build(), rowType).build();
      call.transformTo(newRel);
    } else {
      // Filter had no effect, so we can say that Filter(Values) ==
      // Values.
      call.transformTo(values);
    }

    // New plan is absolutely better than old plan. (Moreover, if
    // changeCount == 0, we've proved that the filter was trivial, and that
    // can send the volcano planner into a loop; see dtbug 2070.)
    if (filter != null) {
      call.getPlanner().prune(filter);
    }
  }

  // ~ Inner Classes ----------------------------------------------------------

  /** Shuttle that converts inputs to literals. */
  private static class MyRexShuttle extends RexShuttle {
    private @Nullable List<RexLiteral> literalList;

    @Override
    public RexNode visitInputRef(RexInputRef inputRef) {
      requireNonNull(literalList, "literalList");
      return literalList.get(inputRef.getIndex());
    }
  }

  /** Rule configuration. */
  @Value.Immutable(singleton = false)
  public interface Config extends RelRule.Config {
    ValuesReduceRule.Config PROJECT =
        ImmutableValuesReduceRule.Config.builder()
            .withDescription("ValuesReduceRule(Project)")
            .withOperandSupplier(
                b0 ->
                    b0.operand(Project.class)
                        .oneInput(
                            b1 ->
                                b1.operand(Values.class).predicate(Values::isNotEmpty).noInputs()))
            .withMatchHandler(ValuesReduceRule::matchProject)
            .build();

    @Override
    default ValuesReduceRule toRule() {
      return new ValuesReduceRule(this);
    }

    /** Forwards a call to {@link #onMatch(RelOptRuleCall)}. */
    @Value.Parameter
    MatchHandler<ValuesReduceRule> matchHandler();

    /** Sets {@link #matchHandler()}. */
    ValuesReduceRule.Config withMatchHandler(MatchHandler<ValuesReduceRule> matchHandler);

    /** Defines an operand tree for the given classes. */
    default ValuesReduceRule.Config withOperandFor(Class<? extends RelNode> relClass) {
      return withOperandSupplier(b -> b.operand(relClass).anyInputs())
          .as(ValuesReduceRule.Config.class);
    }
  }
}
