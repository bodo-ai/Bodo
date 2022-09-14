package com.bodosql.calcite.application.BodoSQLRules;

import com.google.common.collect.ImmutableList;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.rules.TransformationRule;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Sort} past a {@link
 * org.apache.calcite.rel.core.Project} if there is only a limit and no sort.
 *
 * <p>This is based on CoreRules#SORT_PROJECT_TRANSPOSE but only applies when there is no sort
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class LimitProjectTransposeRule extends RelRule<LimitProjectTransposeRule.Config>
    implements TransformationRule {

  /** Creates a SortProjectTransposeRule. */
  protected LimitProjectTransposeRule(LimitProjectTransposeRule.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Sort sort = call.rel(0);
    final Project project = call.rel(1);

    if (sort.getConvention() != project.getConvention()) {
      return;
    }

    // Bodo Change: Only perform the optimization if we aren't doing a sort.
    // As a result we can also avoid remapping the collation
    if (sort.collation.getFieldCollations().size() > 0) {
      return;
    }

    final Sort newSort =
        sort.copy(
            sort.getTraitSet(), project.getInput(), sort.getCollation(), sort.offset, sort.fetch);
    RelNode newProject = project.copy(sort.getTraitSet(), ImmutableList.of(newSort));
    call.transformTo(newProject);
  }
  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    LimitProjectTransposeRule.Config DEFAULT =
        ImmutableLimitProjectTransposeRule.Config.of()
            .withOperandFor(Sort.class, LogicalProject.class);

    @Override
    default RelOptRule toRule() {
      return new LimitProjectTransposeRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default LimitProjectTransposeRule.Config withOperandFor(
        Class<? extends Sort> sortClass, Class<? extends Project> projectClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(sortClass)
                      .oneInput(
                          b1 ->
                              b1.operand(projectClass)
                                  .predicate(p -> !p.containsOver())
                                  .anyInputs()))
          .as(LimitProjectTransposeRule.Config.class);
    }

    /** Defines an operand tree for the given classes. */
    default LimitProjectTransposeRule.Config withOperandFor(
        Class<? extends Sort> sortClass,
        Class<? extends Project> projectClass,
        Class<? extends RelNode> inputClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(sortClass)
                      .oneInput(
                          b1 ->
                              b1.operand(projectClass)
                                  .predicate(p -> !p.containsOver())
                                  .oneInput(b2 -> b2.operand(inputClass).anyInputs())))
          .as(LimitProjectTransposeRule.Config.class);
    }
  }
}
