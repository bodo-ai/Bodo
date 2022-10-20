package com.bodosql.calcite.application.bodo_sql_rules;

import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.apache.calcite.rex.RexUtil;
import org.immutables.value.Value;

/**
 * Planner rule that, given a {@link org.apache.calcite.rel.core.Project} node that merely returns
 * its input and does not contain any aliases, converts the node into its child.
 *
 * <p>This is directly based on the code in ProjectRemoveRule
 * https://github.com/apache/calcite/blob/fd6ffc901ef28bf8408cca28b57eba9f8a204749/core/src/main/java/org/apache/calcite/rel/rules/ProjectRemoveRule.java#L38
 * but that code doesn't check aliases.
 *
 * <p>You should use this class as an example of how to extend a Rule.
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class ProjectUnaliasedRemoveRule extends RelRule<ProjectUnaliasedRemoveRule.Config>
    implements SubstitutionRule {

  /** Creates a ProjectUnaliasedRemoveRule. */
  protected ProjectUnaliasedRemoveRule(Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------
  public static boolean isTrivial(Project project) {
    /*
     * Method to determine if any Project is trivial.
     * We consider a project to be trivial if it selects all of the
     * columns and doesn't alias or reorder the columns.
     */
    return RexUtil.isIdentity(project.getProjects(), project.getInput().getRowType())
        && !isAliased(project, project.getInput());
  }

  // ~ Methods ----------------------------------------------------------------
  public static boolean isAliased(Project project, RelNode child) {
    /*
     * Method to determine if a project aliases any member of its child node.
     * This is done by checking the row type and confirming the fields match.
     * Between both row types.
     */
    List<String> projectFields = project.getRowType().getFieldNames();
    List<String> childFields = child.getRowType().getFieldNames();
    if (projectFields.size() != childFields.size()) {
      // This rows should have the same number of fields. If they don't
      // we shouldn't apply this optimization
      return true;
    }
    for (int i = 0; i < childFields.size(); i++) {
      // If any fields don't match, we must have an alias (or a reordering).
      // As a result, we don't want to apply the rule.
      if (!projectFields.get(i).equals(childFields.get(i))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the child of a project if the project is trivial. Its unclear exactly how calcite is
   * using this method, but the other classes support it should we do to match it.
   */
  public static RelNode strip(Project project) {
    return isTrivial(project) ? project.getInput() : project;
  }

  /** This method is called when the rule finds a Relnode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // Number of rel elements in call is determined by the number of values
    // in the withOperandSupplier below.
    Project project = call.rel(0);
    // Double check the input is trivial. This is probably
    // necessary but can detect other application failures.
    assert isTrivial(project);
    RelNode stripped = project.getInput();
    stripped = convert(stripped, project.getConvention());
    call.transformTo(stripped);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is Project, and the predicate is true.
    Config DEFAULT =
        ImmutableProjectUnaliasedRemoveRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Project.class)
                        .predicate(ProjectUnaliasedRemoveRule::isTrivial)
                        .anyInputs())
            .as(Config.class);

    @Override
    default ProjectUnaliasedRemoveRule toRule() {
      return new ProjectUnaliasedRemoveRule(this);
    }
  }
}
