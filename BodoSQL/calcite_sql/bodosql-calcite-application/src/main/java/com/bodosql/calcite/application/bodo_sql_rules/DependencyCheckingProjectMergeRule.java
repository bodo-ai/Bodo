package com.bodosql.calcite.application.bodo_sql_rules;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import java.util.*;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Permutation;
import org.immutables.value.Value;

/**
 * ProjectMergeRule merges a {@link org.apache.calcite.rel.core.Project} into another {@link
 * org.apache.calcite.rel.core.Project}, provided the projects aren't projecting identical sets of
 * input references and that the outermost Projection doesn't depend on the inner projection.
 *
 * <p>Most of this is copied from the calcite ProjectMergeRule.
 *
 * @see CoreRules#PROJECT_MERGE
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class DependencyCheckingProjectMergeRule
    extends RelRule<DependencyCheckingProjectMergeRule.Config> implements TransformationRule {
  /**
   * Default amount by which complexity is allowed to increase.
   *
   * @see org.apache.calcite.rel.rules.ProjectMergeRule.Config#bloat()
   */
  public static final int DEFAULT_BLOAT = 100;

  /** Creates a ProjectMergeRule. */
  protected DependencyCheckingProjectMergeRule(DependencyCheckingProjectMergeRule.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public boolean matches(RelOptRuleCall call) {
    final Project topProject = call.rel(0);
    final Project bottomProject = call.rel(1);
    return topProject.getConvention() == bottomProject.getConvention();
  }

  /**
   * Determine if the given node contains any inputRefs that are not simple references to inputRefs
   * in the bottom project.
   *
   * @param node RexNode to check.
   * @param bottomProject The Project an inputRef may refer to. We allow an inputRef if it's also an
   *     inputRef in the bottomProject.
   * @return Does the node contain any inputRefs?
   */
  public static boolean nodeContainsInputRef(RexNode node, Project bottomProject) {
    if (node instanceof RexInputRef) {
      int index = ((RexInputRef) node).getIndex();
      // If we have found an input ref it's okay if the column is just an input
      // ref in the input.
      return !(bottomProject.getProjects().get(index) instanceof RexInputRef);
    } else if (node instanceof RexOver) {
      // For window functions check both the function and partitions/order by
      RexOver overNode = ((RexOver) node);
      RexWindow window = overNode.getWindow();
      for (RexNode child : window.partitionKeys) {
        if (nodeContainsInputRef(child, bottomProject)) {
          return true;
        }
      }
      for (RexFieldCollation childCollation : window.orderKeys) {
        RexNode child = childCollation.getKey();
        if (nodeContainsInputRef(child, bottomProject)) {
          return true;
        }
      }
      for (RexNode oldOperand : overNode.getOperands()) {
        if (nodeContainsInputRef(oldOperand, bottomProject)) {
          return true;
        }
      }
    } else if (node instanceof RexCall) {
      RexCall callNode = (RexCall) node;
      for (RexNode operand : callNode.getOperands()) {
        if (nodeContainsInputRef(operand, bottomProject)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Does the given projection have a dependency on its input projection. The projection is
   * dependent if it contains a column that performs computation with any of the input columns.
   * Simply referencing input columns should be fine because you can discard the outer column.
   *
   * @param project The projection to check
   * @return Does any column utilize its input in compute?
   */
  public static boolean hasDependency(Project topProject, Project bottomProject) {
    for (RexNode col : topProject.getProjects()) {
      if (!(col instanceof RexInputRef) && nodeContainsInputRef(col, bottomProject)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project topProject = call.rel(0);
    final Project bottomProject = call.rel(1);
    final RelBuilder relBuilder = call.builder();

    // Bodo Change: Verify that the top project doesn't
    // depend on the bottom project.
    if (hasDependency(topProject, bottomProject)) {
      return;
    }

    // If one or both projects are permutations, short-circuit the complex logic
    // of building a RexProgram.
    final Permutation topPermutation = topProject.getPermutation();
    if (topPermutation != null) {
      if (topPermutation.isIdentity()) {
        // Let ProjectRemoveRule handle this.
        return;
      }
      final Permutation bottomPermutation = bottomProject.getPermutation();
      if (bottomPermutation != null) {
        if (bottomPermutation.isIdentity()) {
          // Let ProjectRemoveRule handle this.
          return;
        }
        final Permutation product = topPermutation.product(bottomPermutation);
        relBuilder.push(bottomProject.getInput());
        relBuilder.project(relBuilder.fields(product), topProject.getRowType().getFieldNames());
        call.transformTo(relBuilder.build());
        return;
      }
    }

    // If we're not in force mode and the two projects reference identical
    // inputs, then return and let ProjectRemoveRule replace the projects.
    if (!config.force()) {
      if (RexUtil.isIdentity(topProject.getProjects(), topProject.getInput().getRowType())) {
        return;
      }
    }

    final List<RexNode> newProjects =
        RelOptUtil.pushPastProjectUnlessBloat(
            topProject.getProjects(), bottomProject, config.bloat());
    if (newProjects == null) {
      // Merged projects are significantly more complex. Do not merge.
      return;
    }
    final RelNode input = bottomProject.getInput();
    if (RexUtil.isIdentity(newProjects, input.getRowType())) {
      if (config.force()
          || input.getRowType().getFieldNames().equals(topProject.getRowType().getFieldNames())) {
        call.transformTo(input);
        return;
      }
    }

    // replace the two projects with a combined projection
    relBuilder.push(bottomProject.getInput());
    relBuilder.project(newProjects, topProject.getRowType().getFieldNames());
    call.transformTo(relBuilder.build());
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    DependencyCheckingProjectMergeRule.Config DEFAULT =
        ImmutableDependencyCheckingProjectMergeRule.Config.of().withOperandFor(Project.class);

    @Override
    default DependencyCheckingProjectMergeRule toRule() {
      return new DependencyCheckingProjectMergeRule(this);
    }

    /**
     * Limit how much complexity can increase during merging. Default is {@link #DEFAULT_BLOAT}
     * (100).
     */
    @Value.Default
    default int bloat() {
      return DependencyCheckingProjectMergeRule.DEFAULT_BLOAT;
    }

    /** Sets {@link #bloat()}. */
    DependencyCheckingProjectMergeRule.Config withBloat(int bloat);

    /** Whether to always merge projects, default true. */
    @Value.Default
    default boolean force() {
      return true;
    }

    /** Sets {@link #force()}. */
    DependencyCheckingProjectMergeRule.Config withForce(boolean force);

    /** Defines an operand tree for the given classes. */
    default DependencyCheckingProjectMergeRule.Config withOperandFor(
        Class<? extends Project> projectClass) {
      return withOperandSupplier(
              b0 -> b0.operand(projectClass).oneInput(b1 -> b1.operand(projectClass).anyInputs()))
          .as(DependencyCheckingProjectMergeRule.Config.class);
    }
  }
}
