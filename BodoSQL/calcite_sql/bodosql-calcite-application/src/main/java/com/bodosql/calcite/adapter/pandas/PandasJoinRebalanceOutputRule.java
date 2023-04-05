package com.bodosql.calcite.adapter.pandas;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import com.google.common.collect.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rex.*;
import org.immutables.value.*;

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
public class PandasJoinRebalanceOutputRule extends RelRule<PandasJoinRebalanceOutputRule.Config>
    implements TransformationRule {

  // The threshold for the estimated number of non-trivial nodes encountered.
  final double nodeCountThreshold = 4.5;

  /** Creates an PandasJoinRebalanceOutputRule. */
  protected PandasJoinRebalanceOutputRule(PandasJoinRebalanceOutputRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    PandasProject project = call.rel(0);
    PandasJoin join = call.rel(1);

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
      RelNode newProject = project.copy(project.getTraitSet(), ImmutableList.of(newJoin));
      call.transformTo(newProject);
    }
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    PandasJoinRebalanceOutputRule.Config DEFAULT =
        ImmutablePandasJoinRebalanceOutputRule.Config.of()
            .withOperandFor(PandasProject.class, PandasJoin.class);

    @Override
    default PandasJoinRebalanceOutputRule toRule() {
      return new PandasJoinRebalanceOutputRule(this);
    }

    /** Defines an operand tree with two classes. */
    // Note we require a logical join because a generic Join doesn't support hints
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
