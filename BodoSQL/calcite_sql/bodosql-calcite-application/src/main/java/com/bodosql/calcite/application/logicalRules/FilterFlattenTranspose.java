package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.Flatten;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalFlatten;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.Mappings;
import org.immutables.value.Value;

/**
 * Planner rule that pushes a {@link org.apache.calcite.rel.core.Filter} past a {@link
 * org.apache.calcite.rel.core.Flatten}.
 *
 * <p>This rule enables pushing filters for any repeated columns past flatten, but at this time
 * doesn't support pushing filters to the output of the function. In the future we may expand this
 * when we can make strong guarantees about a relationship between the argument and the return type
 * (e.g. can we duplicate IS NOT NULL or replace with and equivalent complete filter).
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterFlattenTranspose extends RelRule<FilterFlattenTranspose.Config>
    implements TransformationRule {

  /** Creates a FilterFlattenTranspose. */
  protected FilterFlattenTranspose(FilterFlattenTranspose.Config config) {
    super(config);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final RelBuilder builder = call.builder();
    final Filter filter = call.rel(0);
    final Flatten flatten = call.rel(1);
    // Push the input in case we need to modify the inputs.
    builder.push(flatten.getInput());
    List<RexNode> keptConditions = new ArrayList<>();
    List<RexNode> pushedConditions = new ArrayList<>();
    // Create the information for checking and updating columns.
    final int numFunctionCols = flatten.getUsedColOutputs().cardinality();
    final int numRepeatCols = flatten.getRepeatColumns().cardinality();
    ImmutableBitSet usedFunctionBits = ImmutableBitSet.range(numFunctionCols);
    final Mappings.TargetMapping mapping =
        numFunctionCols == 0
            ? Mappings.createIdentity(numRepeatCols)
            : Mappings.createShiftMapping(
                flatten.getRowType().getFieldCount(), 0, numFunctionCols, numRepeatCols);

    // Iterate over the condition. Any condition that only impacts the
    // repeated columns can be pushed. All others should be kept.
    for (RexNode cond : RelOptUtil.conjunctions(filter.getCondition())) {
      RelOptUtil.InputFinder inputFinder = RelOptUtil.InputFinder.analyze(cond);
      ImmutableBitSet usedColumns = inputFinder.build();
      // Check if any
      if (usedColumns.intersects(usedFunctionBits)) {
        // At least 1 function output is used in the filter.
        // We cannot push this filter.
        keptConditions.add(cond);
      } else {
        // Remap the input refs.
        pushedConditions.add(RexUtil.apply(mapping, cond));
      }
    }

    // If no conditions were selected to push just exit.
    if (pushedConditions.isEmpty()) {
      return;
    }
    // Build the pushed conditions.
    builder.filter(pushedConditions);
    // Push Flatten
    Flatten newFlatten =
        flatten.copy(
            flatten.getTraitSet(),
            builder.build(),
            flatten.getCall(),
            flatten.getCallType(),
            flatten.getUsedColOutputs(),
            flatten.getRepeatColumns());
    builder.push(newFlatten);
    // Build the kept conditions.
    if (!keptConditions.isEmpty()) {
      builder.filter(keptConditions);
    }
    // Generate the new node.
    call.transformTo(builder.build());
  }

  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterFlattenTranspose.Config DEFAULT =
        ImmutableFilterFlattenTranspose.Config.of()
            .withOperandFor(BodoLogicalFilter.class, BodoLogicalFlatten.class);

    @Override
    default FilterFlattenTranspose toRule() {
      return new FilterFlattenTranspose(this);
    }

    /** Defines an operand tree for the given 3 classes. */
    default FilterFlattenTranspose.Config withOperandFor(
        Class<? extends Filter> filterClass, Class<? extends Flatten> flattenClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(f -> !f.containsOver())
                      .oneInput(b1 -> b1.operand(flattenClass).anyInputs()))
          .as(FilterFlattenTranspose.Config.class);
    }
  }
}
