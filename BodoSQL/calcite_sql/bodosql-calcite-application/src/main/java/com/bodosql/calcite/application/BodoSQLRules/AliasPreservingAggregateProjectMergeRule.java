package com.bodosql.calcite.application.BodoSQLRules;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import java.util.*;
import javax.annotation.Nullable;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.mapping.Mappings;
import org.immutables.value.Value;

/**
 * Planner rule that recognizes a {@link org.apache.calcite.rel.core.Aggregate} on top of a {@link
 * org.apache.calcite.rel.core.Project} and if possible aggregate through the project or removes the
 * project.
 *
 * <p>This is only possible when the grouping expressions and argument to the aggregate functions
 * are field references (i.e. not expressions).
 *
 * <p>In some cases, this rule has the effect of trimming: the aggregate will use fewer columns than
 * the project did.
 *
 * @see CoreRules#AGGREGATE_PROJECT_MERGE
 *     <p>This code is a modified version of the default AggregateProjectMergeRule found at:
 *     https://github.com/apache/calcite/blob/7e38304093d803fba678817c07012f4d201ad8a0/core/src/main/java/org/apache/calcite/rel/rules/AggregateProjectMergeRule.java#L62
 *     However, the original rule does not preserve aliases in some conditions
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class AliasPreservingAggregateProjectMergeRule
    extends RelRule<AliasPreservingAggregateProjectMergeRule.Config> implements TransformationRule {

  /** Creates an AliasPreservingAggregateProjectMergeRule. */
  protected AliasPreservingAggregateProjectMergeRule(Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final Project project = call.rel(1);
    RelNode x = apply(call, aggregate, project);
    if (x != null) {
      call.transformTo(x);
    }
  }

  public static @Nullable RelNode apply(RelOptRuleCall call, Aggregate aggregate, Project project) {
    // Final all fields which we need to be straightforward field projects.
    final Set<Integer> interestingFields = RelOptUtil.getAllFields(aggregate);

    // Build the map from old to new; abort if any entry is not a
    // straightforward field projection.
    final Map<Integer, Integer> map = new HashMap<>();
    for (int source : interestingFields) {
      final RexNode rex = project.getProjects().get(source);
      if (!(rex instanceof RexInputRef)) {
        return null;
      }
      map.put(source, ((RexInputRef) rex).getIndex());
    }

    final ImmutableBitSet newGroupSet = aggregate.getGroupSet().permute(map);
    ImmutableList<ImmutableBitSet> newGroupingSets = null;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE) {
      newGroupingSets =
          ImmutableBitSet.ORDERING.immutableSortedCopy(
              ImmutableBitSet.permute(aggregate.getGroupSets(), map));
    }

    final ImmutableList.Builder<AggregateCall> aggCalls = ImmutableList.builder();
    final int sourceCount = aggregate.getInput().getRowType().getFieldCount();
    final int targetCount = project.getInput().getRowType().getFieldCount();
    final Mappings.TargetMapping targetMapping = Mappings.target(map, sourceCount, targetCount);
    for (AggregateCall aggregateCall : aggregate.getAggCallList()) {
      aggCalls.add(aggregateCall.transform(targetMapping));
    }

    final Aggregate newAggregate =
        aggregate.copy(
            aggregate.getTraitSet(),
            project.getInput(),
            newGroupSet,
            newGroupingSets,
            aggCalls.build());

    // Add a project if the group set is not in the same order or
    // contains duplicates.
    final RelBuilder relBuilder = call.builder();
    relBuilder.push(newAggregate);
    final List<Integer> newKeys =
        Util.transform(
            aggregate.getGroupSet().asList(),
            key ->
                requireNonNull(map.get(key), () -> "no value found for key " + key + " in " + map));

    // Rule change. In addition to checking if the column number has been
    // changed, we need to check that the names are not modified (to keep
    // any aliases). If there are aliases we will generate a projection
    // that just renames columns.
    List<String> newKeyNames = new ArrayList<>();
    List<String> newFieldNames = newAggregate.getRowType().getFieldNames();
    List<String> oldKeyNames = new ArrayList<>();
    List<String> oldFieldNames = aggregate.getRowType().getFieldNames();
    for (int i = 0; i < aggregate.getGroupSet().asList().size(); i++) {
      oldKeyNames.add(oldFieldNames.get(i));
      newKeyNames.add(newFieldNames.get(i));
    }
    if (!newKeys.equals(newGroupSet.asList()) || !oldKeyNames.equals(newKeyNames)) {
      final List<Integer> posList = new ArrayList<>();
      String[] keptFieldNames = new String[newAggregate.getRowType().getFieldCount()];
      // Rule change. Here we generate the keptFieldNames so the
      // future projection contains the original aliases
      for (int newKey : newKeys) {
        int idx = newGroupSet.indexOf(newKey);
        posList.add(idx);
        keptFieldNames[idx] = oldFieldNames.get(idx);
      }
      for (int i = newAggregate.getGroupCount();
          i < newAggregate.getRowType().getFieldCount();
          i++) {
        posList.add(i);
        keptFieldNames[i] = newFieldNames.get(i);
      }

      // Rule change. We have to force the projection to be generated if there
      // is only aliasing because the identity projection will not be generated.
      // The rules we apply do not remove this.
      relBuilder.project(
          relBuilder.fields(posList),
          Arrays.asList(keptFieldNames),
          !oldKeyNames.equals(newKeyNames));
    }
    return relBuilder.build();
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    AliasPreservingAggregateProjectMergeRule.Config DEFAULT =
        ImmutableAliasPreservingAggregateProjectMergeRule.Config.of()
            .withOperandFor(Aggregate.class, Project.class);

    @Override
    default AliasPreservingAggregateProjectMergeRule toRule() {
      return new AliasPreservingAggregateProjectMergeRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default AliasPreservingAggregateProjectMergeRule.Config withOperandFor(
        Class<? extends Aggregate> aggregateClass, Class<? extends Project> projectClass) {
      return withOperandSupplier(
              b0 -> b0.operand(aggregateClass).oneInput(b1 -> b1.operand(projectClass).anyInputs()))
          .as(AliasPreservingAggregateProjectMergeRule.Config.class);
    }
  }
}
