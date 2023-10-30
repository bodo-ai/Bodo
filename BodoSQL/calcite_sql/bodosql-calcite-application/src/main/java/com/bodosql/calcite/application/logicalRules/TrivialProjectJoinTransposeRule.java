package com.bodosql.calcite.application.logicalRules;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalJoin;
import com.bodosql.calcite.rel.logical.BodoLogicalProject;
import java.util.ArrayList;
import java.util.List;
import java.util.NavigableSet;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.PushProjector;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.sql.SqlKind;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Project} past a {@link
 * org.apache.calcite.rel.core.Join} by splitting the projection into a projection on top of each
 * child of the join.
 *
 * @see org.apache.calcite.rel.rules.CoreRules#PROJECT_JOIN_TRANSPOSE
 *     <p>This code is a modified version of the default ProjectJoinTransposeRule found at:
 *     https://github.com/apache/calcite/blob/fd6ffc901ef28bf8408cca28b57eba9f8a204749/core/src/main/java/org/apache/calcite/rel/rules/ProjectJoinTransposeRule.java
 *     This code ensures that we only push trivial projections
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class TrivialProjectJoinTransposeRule extends RelRule<TrivialProjectJoinTransposeRule.Config>
    implements TransformationRule {

  /** Creates a ProjectJoinTransposeRule. */
  protected TrivialProjectJoinTransposeRule(Config config) {
    super(config);
  }

  /**
   * Determine if a projection performs any actual computation or is just selecting a subset of
   * columns (possibly with renaming). To determine if a projection is a subset we verify that all
   * nodes are InputRefs.
   *
   * @param node Projection in question
   * @return Does the node contain something other than an input ref.
   */
  private static boolean projectionContainsCompute(Project node) {
    for (RexNode r : node.getProjects()) {
      if (!(r instanceof RexInputRef)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Convert a Project from an implementation with non-InputRef expressions into a Projection with
   * only inputRef expressions that cover every column used by the original node.
   *
   * @param node Projection to be converted to just input refs
   * @return A projection that contains all of the used columns but only inputRefs.
   */
  private static Project convertProjectToInputRefs(Project node, RelDataType originalType) {
    // Find all of the inputRefs that are used
    RelOptUtil.InputReferencedVisitor finder = new RelOptUtil.InputReferencedVisitor();
    finder.visitEach(node.getProjects());
    NavigableSet<Integer> inputsReferenced = finder.inputPosReferenced;
    // Convert the indices into InputRefs
    List<String> totalFieldNames = originalType.getFieldNames();
    List<String> keptFieldNames = new ArrayList<>();
    List<RexNode> inputRefs = new ArrayList<>();
    List<RelDataTypeField> origFieldList = originalType.getFieldList();
    for (int pos : inputsReferenced) {
      keptFieldNames.add(totalFieldNames.get(pos));
      inputRefs.add(new RexInputRef(pos, origFieldList.get(pos).getType()));
    }
    // Create the new projects. Here we reuse the hints.
    return BodoLogicalProject.create(node.getInput(), node.getHints(), inputRefs, keptFieldNames);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project origProject = call.rel(0);
    final Join join = call.rel(1);

    // Normalize the join condition so we don't end up misidentified expanded
    // form of IS NOT DISTINCT FROM as PushProject also visit the filter condition
    // and push down expressions.
    RexNode joinFilter =
        join.getCondition()
            .accept(
                new RexShuttle() {
                  @Override
                  public RexNode visitCall(RexCall rexCall) {
                    final RexNode node = super.visitCall(rexCall);
                    if (!(node instanceof RexCall)) {
                      return node;
                    }
                    return RelOptUtil.collapseExpandedIsNotDistinctFromExpr(
                        (RexCall) node, call.builder().getRexBuilder());
                  }
                });
    // Update the projection to only contain subsets as opposed to actual pushdown.
    // Note: This is necessary because we require any compute in the join filter
    // that is a complex operation to be pushed down.
    boolean projectContainsCompute = projectionContainsCompute(origProject);
    final Project pushedProject;
    // Bodo Change: The decision to push compute into a join isn't trivial and needs to be
    // determined by a cost optimizer. However, selecting just a subset of columns is trivial.
    // Therefore, we determine if a projection contains compute we reduce the projection to
    // just a projection on the used columns, except for any compute in the join filter, which
    // must be pushed down.
    if (projectContainsCompute) {
      pushedProject = convertProjectToInputRefs(origProject, join.getRowType());
    } else {
      pushedProject = origProject;
    }

    // locate all fields referenced in the projection and join condition;
    // determine which inputs are referenced in the projection and
    // join condition; if all fields are being referenced and there are no
    // special expressions, no point in proceeding any further
    PushProjector pushProjector =
        new PushProjector(
            pushedProject, joinFilter, join, config.preserveExprCondition(), call.builder());
    if (pushProjector.locateAllRefs()) {
      return;
    }

    // create left and right projections, projecting only those
    // fields referenced on each side
    RelNode leftProject = pushProjector.createProjectRefsAndExprs(join.getLeft(), true, false);
    RelNode rightProject = pushProjector.createProjectRefsAndExprs(join.getRight(), true, true);
    // If both projections are now just the identity no optimization is possible
    boolean isLeftUnchanged = leftProject.getRowType().equals(join.getLeft().getRowType());
    boolean isRightUnchange = rightProject.getRowType().equals(join.getRight().getRowType());
    // If neither the left nor the right has changes we can exit.
    if (isLeftUnchanged && isRightUnchange) {
      return;
    }
    int[] adjustments = pushProjector.getAdjustments();
    RexNode newJoinFilter = null;
    if (joinFilter != null) {
      List<RelDataTypeField> projectJoinFieldList = new ArrayList<>();
      projectJoinFieldList.addAll(join.getSystemFieldList());
      projectJoinFieldList.addAll(leftProject.getRowType().getFieldList());
      projectJoinFieldList.addAll(rightProject.getRowType().getFieldList());
      newJoinFilter =
          pushProjector.convertRefsAndExprs(joinFilter, projectJoinFieldList, adjustments);
    }
    if (projectContainsCompute) {
      // If we have modified the projections we cannot reuse the old pushProjector. Instead, we
      // reset the PushProjector, so the original projection uses the new left and right. In
      // addition
      // we pass the new join filter which should also be updated.
      pushProjector =
          new PushProjector(
              origProject, newJoinFilter, join, config.preserveExprCondition(), call.builder());
    }

    // create a new join with the projected children
    final Join newJoin =
        join.copy(
            join.getTraitSet(),
            requireNonNull(newJoinFilter, "newJoinFilter must not be null"),
            leftProject,
            rightProject,
            join.getJoinType(),
            join.isSemiJoinDone());

    // put the original project on top of the join, converting it to
    // reference the modified projection list
    RelNode topProject = pushProjector.createNewProject(newJoin, adjustments);
    call.transformTo(topProject);
  }

  /** Rule configuration. */
  @Value.Immutable(singleton = false)
  public interface Config extends RelRule.Config {
    Config DEFAULT =
        ImmutableTrivialProjectJoinTransposeRule.Config.builder()
            .withPreserveExprCondition(
                expr -> {
                  // Do not push down over's expression by default
                  if (expr instanceof RexOver) {
                    return false;
                  }
                  if (SqlKind.CAST == expr.getKind()) {
                    final RelDataType relType = expr.getType();
                    final RexCall castCall = (RexCall) expr;
                    final RelDataType operand0Type = castCall.getOperands().get(0).getType();
                    if (relType.getSqlTypeName() == operand0Type.getSqlTypeName()
                        && operand0Type.isNullable()
                        && !relType.isNullable()) {
                      // Do not push down not nullable cast's expression with the same type by
                      // default
                      // eg: CAST($1):VARCHAR(10) NOT NULL, and type of $1 is nullable VARCHAR(10)
                      return false;
                    }
                  }
                  return true;
                })
            .build()
            .withOperandFor(BodoLogicalProject.class, BodoLogicalJoin.class);

    @Override
    default TrivialProjectJoinTransposeRule toRule() {
      return new TrivialProjectJoinTransposeRule(this);
    }

    /** Defines when an expression should not be pushed. */
    PushProjector.ExprCondition preserveExprCondition();

    /** Sets {@link #preserveExprCondition()}. */
    Config withPreserveExprCondition(PushProjector.ExprCondition condition);

    /** Defines an operand tree for the given classes. */
    default Config withOperandFor(
        Class<? extends Project> projectClass, Class<? extends Join> joinClass) {
      return withOperandSupplier(
              b0 -> b0.operand(projectClass).oneInput(b1 -> b1.operand(joinClass).anyInputs()))
          .as(Config.class);
    }
  }
}
