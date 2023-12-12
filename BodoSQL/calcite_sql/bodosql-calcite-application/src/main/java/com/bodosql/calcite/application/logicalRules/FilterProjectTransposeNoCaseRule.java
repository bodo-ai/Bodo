package com.bodosql.calcite.application.logicalRules;

import static com.bodosql.calcite.application.logicalRules.FilterRulesCommon.rexNodeContainsCase;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.Collections;
import java.util.function.Predicate;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelCollationTraitDef;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Filter} past a {@link
 * org.apache.calcite.rel.core.Project} but won't push down a filter if the output filter contains a
 * case statement.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterProjectTransposeNoCaseRule
    extends RelRule<FilterProjectTransposeNoCaseRule.Config> implements TransformationRule {

  /** Creates a FilterProjectTransposeNoCaseRule. */
  protected FilterProjectTransposeNoCaseRule(FilterProjectTransposeNoCaseRule.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Filter filter = call.rel(0);
    final Project project = call.rel(1);

    if (project.containsOver()) {
      // In general a filter cannot be pushed below a windowing calculation.
      // Applying the filter before the aggregation function changes
      // the results of the windowing invocation.
      //
      // When the filter is on the PARTITION BY expression of the OVER clause
      // it can be pushed down. For now, we don't support this.
      return;
    }
    // convert the filter to one that references the child of the project
    RexNode newCondition = RelOptUtil.pushPastProject(filter.getCondition(), project);
    // Bodo Change: Don't push the filter if it now contains a case statement.
    // If the previous filter contained a case statement already a separate rule will
    // extract it from the filter.
    if (rexNodeContainsCase(newCondition)) {
      return;
    }

    final RelBuilder relBuilder = call.builder();
    RelNode newFilterRel;
    if (config.isCopyFilter()) {
      final RelNode input = project.getInput();
      final RelTraitSet traitSet =
          filter
              .getTraitSet()
              .replaceIfs(
                  RelCollationTraitDef.INSTANCE,
                  () ->
                      Collections.singletonList(
                          input.getTraitSet().getTrait(RelCollationTraitDef.INSTANCE)))
              .replaceIfs(
                  RelDistributionTraitDef.INSTANCE,
                  () ->
                      Collections.singletonList(
                          input.getTraitSet().getTrait(RelDistributionTraitDef.INSTANCE)));
      newCondition = RexUtil.removeNullabilityCast(relBuilder.getTypeFactory(), newCondition);
      newFilterRel = filter.copy(traitSet, input, newCondition);
    } else {
      newFilterRel = relBuilder.push(project.getInput()).filter(newCondition).build();
    }

    RelNode newProject =
        config.isCopyProject()
            ? project.copy(
                project.getTraitSet(), newFilterRel, project.getProjects(), project.getRowType())
            : relBuilder
                .push(newFilterRel)
                .project(
                    project.getProjects(),
                    project.getRowType().getFieldNames(),
                    false,
                    project.getVariablesSet())
                .build();

    call.transformTo(newProject);
  }

  /**
   * Rule configuration.
   *
   * <p>If {@code copyFilter} is true, creates the same kind of Filter as matched in the rule,
   * otherwise it creates a Filter using the RelBuilder obtained by the {@code relBuilderFactory}.
   * Similarly for {@code copyProject}.
   *
   * <p>Defining predicates for the Filter (using {@code filterPredicate}) and/or the Project (using
   * {@code projectPredicate} allows making the rule more restrictive.
   */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterProjectTransposeNoCaseRule.Config DEFAULT =
        ImmutableFilterProjectTransposeNoCaseRule.Config.of()
            .withOperandFor(
                Filter.class,
                f -> !RexUtil.containsCorrelation(f.getCondition()),
                Project.class,
                p -> true)
            .withCopyFilter(true)
            .withCopyProject(true);

    @Override
    default FilterProjectTransposeNoCaseRule toRule() {
      return new FilterProjectTransposeNoCaseRule(this);
    }

    /** Whether to create a {@link Filter} of the same convention as the matched Filter. */
    @Value.Default
    default boolean isCopyFilter() {
      return true;
    }

    /** Sets {@link #isCopyFilter()}. */
    FilterProjectTransposeNoCaseRule.Config withCopyFilter(boolean copyFilter);

    /** Whether to create a {@link Project} of the same convention as the matched Project. */
    @Value.Default
    default boolean isCopyProject() {
      return true;
    }

    /** Sets {@link #isCopyProject()}. */
    FilterProjectTransposeNoCaseRule.Config withCopyProject(boolean copyProject);

    /** Defines an operand tree for the given 2 classes. */
    default FilterProjectTransposeNoCaseRule.Config withOperandFor(
        Class<? extends Filter> filterClass,
        Predicate<Filter> filterPredicate,
        Class<? extends Project> projectClass,
        Predicate<Project> projectPredicate) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(filterPredicate)
                      .oneInput(
                          b1 -> b1.operand(projectClass).predicate(projectPredicate).anyInputs()))
          .as(FilterProjectTransposeNoCaseRule.Config.class);
    }

    /** Defines an operand tree for the given 3 classes. */
    default FilterProjectTransposeNoCaseRule.Config withOperandFor(
        Class<? extends Filter> filterClass,
        Class<? extends Project> projectClass,
        Class<? extends RelNode> relClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .oneInput(
                          b1 ->
                              b1.operand(projectClass)
                                  .oneInput(b2 -> b2.operand(relClass).anyInputs())))
          .as(FilterProjectTransposeNoCaseRule.Config.class);
    }
  }
}
