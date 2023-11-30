package com.bodosql.calcite.adapter.pandas;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexNode;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.immutables.value.Value;

/**
 * Planner rule that recognizes 1-2 {@link org.apache.calcite.rel.core.Project} on top of a {@link
 * org.apache.calcite.rel.core.Join} and determines if the Projection requires significant
 * computation. If the projection is "expensive" enough then the Join is updated with a hint
 * indicating that the output data should be rebalanced if there is significant skew.
 *
 * <p>In the future this rule should be updated with actual cost based decisions, ideally by reusing
 * cost metrics used for ordering nodes. However, since we don't have those available yet, we
 * instead look at the node count to estimate based on number of iterations over the data.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class PandasJoinRebalanceOutputRule<C extends PandasJoinRebalanceOutputRule.Config>
    extends RelRule<C> implements TransformationRule {

  // The threshold for the estimated number of non-trivial nodes encountered.
  static final double nodeCountThreshold = 4.5;

  protected PandasJoinRebalanceOutputRule(C config) {
    super(config);
  }

  @Nullable
  public static RelNode generateNewProject(PandasProject project, PandasJoin join) {
    // Determine the node count for the projection.
    int totalNodeCount = 0;
    for (RexNode projCol : project.getProjects()) {
      // Append the node count. We subtract one to remove
      // trivial nodes. In generally this may be inaccurate
      // for nested expressions because we care about the number
      // of passes over the data, but it should be a good enough
      // approximations. Trivial projections and all scalars will
      // have a weight of 0.
      totalNodeCount += (projCol.nodeCount() - 1);
    }
    // Check if we have seen enough compute to consider a rebalance.
    if (totalNodeCount > nodeCountThreshold) {
      RelNode newJoin = join.withRebalanceOutput(true);
      RelNode input = newJoin;
      RelNode newProject = project.copy(project.getTraitSet(), ImmutableList.of(input));
      return newProject;
    }
    return null;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    PandasProject project = call.rel(0);
    PandasJoin join = call.rel(1);
    RelNode output = generateNewProject(project, join);
    if (output != null) {
      call.transformTo(output);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    Config DEFAULT_CONFIG =
        ImmutablePandasJoinRebalanceOutputRule.Config.of()
            .withOperandFor(PandasProject.class, PandasJoin.class);

    @Override
    default PandasJoinRebalanceOutputRule toRule() {
      return new PandasJoinRebalanceOutputRule(this);
    }

    /** Defines an operand tree with two classes. */
    default PandasJoinRebalanceOutputRule.Config withOperandFor(
        Class<? extends PandasProject> projectClass, Class<? extends PandasJoin> joinClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(projectClass)
                      .oneInput(
                          b1 ->
                              b1.operand(joinClass)
                                  .predicate(f -> !f.getRebalanceOutput())
                                  .anyInputs()))
          .as(PandasJoinRebalanceOutputRule.Config.class);
    }
  }
}
