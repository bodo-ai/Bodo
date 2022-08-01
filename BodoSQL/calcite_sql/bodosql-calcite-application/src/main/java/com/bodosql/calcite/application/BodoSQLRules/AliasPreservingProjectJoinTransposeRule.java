package com.bodosql.calcite.application.BodoSQLRules;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.calcite.plan.*;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.rules.PushProjector;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.tools.RelBuilderFactory;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Project} past a {@link
 * org.apache.calcite.rel.core.Join} by splitting the projection into a projection on top of each
 * child of the join.
 *
 * @see CoreRules#PROJECT_JOIN_TRANSPOSE
 *     <p>This code is a modified version of the default ProjectJoinTransposeRule found at:
 *     https://github.com/apache/calcite/blob/fd6ffc901ef28bf8408cca28b57eba9f8a204749/core/src/main/java/org/apache/calcite/rel/rules/ProjectJoinTransposeRule.java
 *     However, the original rule does not preserve aliases in some conditions
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class AliasPreservingProjectJoinTransposeRule
    extends RelRule<AliasPreservingProjectJoinTransposeRule.Config> implements TransformationRule {

  /** Creates a ProjectJoinTransposeRule. */
  protected AliasPreservingProjectJoinTransposeRule(Config config) {
    super(config);
  }

  @Deprecated // to be removed before 2.0
  public AliasPreservingProjectJoinTransposeRule(
      Class<? extends Project> projectClass,
      Class<? extends Join> joinClass,
      PushProjector.ExprCondition preserveExprCondition,
      RelBuilderFactory relBuilderFactory) {
    this(
        Config.DEFAULT
            .withRelBuilderFactory(relBuilderFactory)
            .as(Config.class)
            .withOperandFor(projectClass, joinClass)
            .withPreserveExprCondition(preserveExprCondition));
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

    // locate all fields referenced in the projection and join condition;
    // determine which inputs are referenced in the projection and
    // join condition; if all fields are being referenced and there are no
    // special expressions, no point in proceeding any further
    final PushProjector pushProjector =
        new PushProjector(
            origProject, joinFilter, join, config.preserveExprCondition(), call.builder());
    if (pushProjector.locateAllRefs()) {
      return;
    }

    // create left and right projections, projecting only those
    // fields referenced on each side
    final RelNode leftProject =
        pushProjector.createProjectRefsAndExprs(join.getLeft(), true, false);
    final RelNode rightProject =
        pushProjector.createProjectRefsAndExprs(join.getRight(), true, true);

    // convert the join condition to reference the projected columns
    RexNode newJoinFilter = null;
    int[] adjustments = pushProjector.getAdjustments();
    if (joinFilter != null) {
      List<RelDataTypeField> projectJoinFieldList = new ArrayList<>();
      projectJoinFieldList.addAll(join.getSystemFieldList());
      projectJoinFieldList.addAll(leftProject.getRowType().getFieldList());
      projectJoinFieldList.addAll(rightProject.getRowType().getFieldList());
      newJoinFilter =
          pushProjector.convertRefsAndExprs(joinFilter, projectJoinFieldList, adjustments);
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

    // There is currently a bug wherein the alias information is lost
    // In order to correct this, we check if top project is a join node
    // If it is, check that the fieldnames of topProject == the original projection
    // If not, add a new projection Node that projects the values to the correct names
    if (topProject instanceof Join) {
      if (!topProject.getRowType().getFieldList().equals(origProject.getRowType().getFieldList())) {
        // RelHints are used to pass around some metadata that might allow for faster execution of
        // the logical plan
        // for our purposes, this can be left blank
        List<RelHint> newRelHints = Collections.emptyList();

        // A cluster is an object containing the environmental context for a particular query
        // As such, the new project should share the same cluster as the original project
        RelOptCluster newCluster = origProject.getCluster();

        // The relTraitSet is used to pass around reltraits, which indicate if a particular
        // operation has a property like reflexive, anti-symmetric, transitive... etc
        // It might assist some optimization rules to add any relevant traits, but for now, I'm
        // simply leaving it blank.
        RelTraitSet newRelTraitSet = RelTraitSet.createEmpty();

        // Create a list of projections from input 0 -> fieldname 0, input 1 -> fieldname 1... and
        // so on
        List<RexInputRef> projections = new ArrayList<>();
        List<RelDataTypeField> fieldList = origProject.getRowType().getFieldList();
        for (int i = 0; i < fieldList.size(); i++) {
          RexInputRef cur_input = new RexInputRef(i, fieldList.get(i).getType());
          projections.add(cur_input);
        }

        topProject =
            new LogicalProject(
                newCluster,
                newRelTraitSet,
                newRelHints,
                topProject,
                projections,
                origProject.getRowType());
      }
    }
    call.transformTo(topProject);
  }

  /** Rule configuration. */
  @Value.Immutable(singleton = false)
  public interface Config extends RelRule.Config {
    Config DEFAULT = ImmutableAliasPreservingProjectJoinTransposeRule.Config.builder()
            .withPreserveExprCondition(expr -> !(expr instanceof RexOver))
            .build()
            .withOperandFor(LogicalProject.class, LogicalJoin.class);

    @Override
    default AliasPreservingProjectJoinTransposeRule toRule() {
      return new AliasPreservingProjectJoinTransposeRule(this);
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
