package com.bodosql.calcite.prepare;

import com.bodosql.calcite.adapter.pandas.PandasMinRowNumberFilter;
import com.bodosql.calcite.adapter.pandas.PandasRowSample;
import com.bodosql.calcite.adapter.pandas.PandasSample;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.bodosql.calcite.rel.core.Flatten;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableCreate;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.CorrelationReferenceFinder;
import org.apache.calcite.sql2rel.RelFieldTrimmer;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.mapping.IntPair;
import org.apache.calcite.util.mapping.Mapping;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Extension of RelFieldTrimmer that covers Bodo's added RelNodes. */
public class BodoRelFieldTrimmer extends RelFieldTrimmer {

  // Store a copy of the RelBuilder
  private final RelBuilder relBuilder;
  // Should we insert projections to prune everything. We set this to false
  // when it may interfere with other optimizations.
  private final boolean pruneEverything;

  public BodoRelFieldTrimmer(
      @Nullable SqlValidator validator, RelBuilder relBuilder, boolean pruneEverything) {
    super(validator, relBuilder);
    this.relBuilder = relBuilder;
    this.pruneEverything = pruneEverything;
  }

  @Override
  protected TrimResult trimChild(
      RelNode rel,
      RelNode input,
      final ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    final ImmutableBitSet.Builder fieldsUsedBuilder = fieldsUsed.rebuild();

    // Bodo Change: We remove the section that requires avoiding pruning collation
    // columns. This should be safe because none of our implementations rely on this
    // collation information. In the future we could explore integrating this into
    // individual nodes we care about (or only the physical step).

    // Correlating variables are a means for other relational expressions to use
    // fields.
    for (final CorrelationId correlation : rel.getVariablesSet()) {
      rel.accept(
          new CorrelationReferenceFinder() {
            @Override
            protected RexNode handle(RexFieldAccess fieldAccess) {
              final RexCorrelVariable v = (RexCorrelVariable) fieldAccess.getReferenceExpr();
              if (v.id.equals(correlation)) {
                fieldsUsedBuilder.set(fieldAccess.getField().getIndex());
              }
              return fieldAccess;
            }
          });
    }

    return dispatchTrimFields(input, fieldsUsedBuilder.build(), extraFields);
  }

  /**
   * Compute a new TrimResult for an input by inserting a projection and creating a new Mapping if
   * there are unused columns even after the input has finished its pruning.
   */
  private TrimResult insertPruningProjection(
      TrimResult childResult, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    int finalFieldCount = fieldsUsed.cardinality();
    // Only completely prune excess columns in the Physical plan step to avoid
    // interfering with other optimizations like Filter Pushdown.
    if (pruneEverything
        && extraFields.size() == 0
        && finalFieldCount < childResult.left.getRowType().getFieldCount()) {
      // If we move to 0 columns make sure we maintain a dummy column
      if (fieldsUsed.cardinality() == 0) {
        return dummyProject(childResult.right.getSourceCount(), childResult.left);
      }
      // Push the input to the builder to generate fields.
      relBuilder.push(childResult.left);
      List<RexNode> projects = new ArrayList<>(finalFieldCount);
      List<String> fieldNames = new ArrayList<>(finalFieldCount);
      // Update the mapping
      Mapping inputMapping =
          Mappings.create(
              MappingType.INVERSE_SURJECTION, childResult.right.getSourceCount(), finalFieldCount);

      Iterator<Integer> setBitIterator = fieldsUsed.iterator();
      for (int i = 0; i < finalFieldCount; i++) {
        int idx = setBitIterator.next();
        int newIndex = childResult.right.getTarget(idx);
        projects.add(relBuilder.field(newIndex));
        fieldNames.add(childResult.left.getRowType().getFieldNames().get(newIndex));
        inputMapping.set(idx, i);
      }
      return result(relBuilder.project(projects, fieldNames).build(), inputMapping);
    } else {
      return childResult;
    }
  }

  /** Common implementation for SingleRel nodes that don't use any columns. */
  private TrimResult trimFieldsNoUsedColumns(
      SingleRel node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    TrimResult childResult = dispatchTrimFields(node.getInput(), fieldsUsed, extraFields);
    // Prune if there are still unused columns.
    TrimResult updateResult = insertPruningProjection(childResult, fieldsUsed, extraFields);
    // Copy the node and return
    RelNode newNode = node.copy(node.getTraitSet(), List.of(updateResult.left));
    return result(newNode, updateResult.right, node);
  }

  public TrimResult trimFields(
      TableCreate node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {

    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      SnowflakeToPandasConverter node,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      PandasSample node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      PandasRowSample node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  /**
   * Variant of {@link #trimFields(RelNode, ImmutableBitSet, Set)} for {@link
   * org.apache.calcite.rel.core.Sort}.
   */
  public TrimResult trimFields(
      Sort sort, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final RelDataType rowType = sort.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RelCollation collation = sort.getCollation();
    final RelNode input = sort.getInput();

    // We use the fields used by the consumer, plus any fields used as sort
    // keys.
    final ImmutableBitSet.Builder inputFieldsUsed = fieldsUsed.rebuild();
    for (RelFieldCollation field : collation.getFieldCollations()) {
      inputFieldsUsed.set(field.getFieldIndex());
    }

    // Create input with trimmed columns.
    final Set<RelDataTypeField> inputExtraFields = Collections.emptySet();
    final ImmutableBitSet inputFieldsUsedBitSet = inputFieldsUsed.build();
    TrimResult trimResult = trimChild(sort, input, inputFieldsUsedBitSet, inputExtraFields);
    // BODO CHANGE:
    // Make sure we're always completely trimming the input
    TrimResult completeTrimResult =
        insertPruningProjection(trimResult, inputFieldsUsedBitSet, inputExtraFields);

    RelNode newInput = completeTrimResult.left;
    final Mapping inputMapping = completeTrimResult.right;

    // If the input is unchanged, and we need to project all columns,
    // there's nothing we can do.
    if (newInput == input && inputMapping.isIdentity() && fieldsUsed.cardinality() == fieldCount) {
      return result(sort, Mappings.createIdentity(fieldCount));
    }

    relBuilder.push(newInput);
    final ImmutableList<RexNode> fields = relBuilder.fields(RexUtil.apply(inputMapping, collation));
    relBuilder.sortLimit(sort.offset, sort.fetch, fields);

    // The result has the same mapping as the input gave us. Sometimes we
    // return fields that the consumer didn't ask for, because the filter
    // needs them for its condition.
    return result(relBuilder.build(), inputMapping, sort);
  }

  /**
   * Variant of {@link #trimFields(RelNode, ImmutableBitSet, Set)} for {@link
   * org.apache.calcite.rel.core.Join}.
   */
  public TrimResult trimFields(
      Join join, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final int fieldCount =
        join.getSystemFieldList().size()
            + join.getLeft().getRowType().getFieldCount()
            + join.getRight().getRowType().getFieldCount();
    final RexNode conditionExpr = join.getCondition();
    final int systemFieldCount = join.getSystemFieldList().size();

    // Add in fields used in the condition.
    final Set<RelDataTypeField> combinedInputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder =
        new RelOptUtil.InputFinder(combinedInputExtraFields, fieldsUsed);
    conditionExpr.accept(inputFinder);
    final ImmutableBitSet fieldsUsedPlus = inputFinder.build();

    // If no system fields are used, we can remove them.
    int systemFieldUsedCount = 0;
    for (int i = 0; i < systemFieldCount; ++i) {
      if (fieldsUsed.get(i)) {
        ++systemFieldUsedCount;
      }
    }
    final int newSystemFieldCount;
    if (systemFieldUsedCount == 0) {
      newSystemFieldCount = 0;
    } else {
      newSystemFieldCount = systemFieldCount;
    }

    int offset = systemFieldCount;
    int changeCount = 0;
    int newFieldCount = newSystemFieldCount;
    final List<RelNode> newInputs = new ArrayList<>(2);
    final List<Mapping> inputMappings = new ArrayList<>();
    final List<Integer> inputExtraFieldCounts = new ArrayList<>();
    for (RelNode input : join.getInputs()) {
      final RelDataType inputRowType = input.getRowType();
      final int inputFieldCount = inputRowType.getFieldCount();

      // Compute required mapping.
      ImmutableBitSet.Builder inputFieldsUsed = ImmutableBitSet.builder();
      for (int bit : fieldsUsedPlus) {
        if (bit >= offset && bit < offset + inputFieldCount) {
          inputFieldsUsed.set(bit - offset);
        }
      }

      // If there are system fields, we automatically use the
      // corresponding field in each input.
      inputFieldsUsed.set(0, newSystemFieldCount);

      // FIXME: We ought to collect extra fields for each input
      // individually. For now, we assume that just one input has
      // on-demand fields.
      Set<RelDataTypeField> inputExtraFields =
          RelDataTypeImpl.extra(inputRowType) == null
              ? Collections.emptySet()
              : combinedInputExtraFields;
      inputExtraFieldCounts.add(inputExtraFields.size());
      ImmutableBitSet inputFieldsUsedBitSet = inputFieldsUsed.build();
      TrimResult trimResult = trimChild(join, input, inputFieldsUsedBitSet, inputExtraFields);
      // Bodo Change: Check if there are unused columns between the two nodes and if so insert a
      // projection.
      // Note: We shouldn't have any examples where inputExtraFields is not empty, so we can't
      // test this. To be conservative we only change this if the set is empty.
      TrimResult newTrimResult =
          insertPruningProjection(trimResult, inputFieldsUsedBitSet, inputExtraFields);
      newInputs.add(newTrimResult.left);
      if (newTrimResult.left != input) {
        ++changeCount;
      }

      final Mapping inputMapping = newTrimResult.right;
      inputMappings.add(inputMapping);

      // Move offset to point to start of next input.
      offset += inputFieldCount;
      newFieldCount += inputMapping.getTargetCount() + inputExtraFields.size();
    }

    Mapping mapping = Mappings.create(MappingType.INVERSE_SURJECTION, fieldCount, newFieldCount);
    for (int i = 0; i < newSystemFieldCount; ++i) {
      mapping.set(i, i);
    }
    offset = systemFieldCount;
    int newOffset = newSystemFieldCount;
    for (int i = 0; i < inputMappings.size(); i++) {
      Mapping inputMapping = inputMappings.get(i);
      for (IntPair pair : inputMapping) {
        mapping.set(pair.source + offset, pair.target + newOffset);
      }
      offset += inputMapping.getSourceCount();
      newOffset += inputMapping.getTargetCount() + inputExtraFieldCounts.get(i);
    }

    if (changeCount == 0 && mapping.isIdentity()) {
      return result(join, Mappings.createIdentity(join.getRowType().getFieldCount()));
    }

    // Build new join.
    final RexVisitor<RexNode> shuttle =
        new RexPermuteInputsShuttle(mapping, newInputs.get(0), newInputs.get(1));
    RexNode newConditionExpr = conditionExpr.accept(shuttle);

    relBuilder.push(newInputs.get(0));
    relBuilder.push(newInputs.get(1));

    switch (join.getJoinType()) {
      case SEMI:
      case ANTI:
        // For SemiJoins and AntiJoins only map fields from the left-side
        if (join.getJoinType() == JoinRelType.SEMI) {
          relBuilder.semiJoin(newConditionExpr);
        } else {
          relBuilder.antiJoin(newConditionExpr);
        }
        Mapping inputMapping = inputMappings.get(0);
        mapping =
            Mappings.create(
                MappingType.INVERSE_SURJECTION,
                join.getRowType().getFieldCount(),
                newSystemFieldCount + inputMapping.getTargetCount());
        for (int i = 0; i < newSystemFieldCount; ++i) {
          mapping.set(i, i);
        }
        offset = systemFieldCount;
        newOffset = newSystemFieldCount;
        for (IntPair pair : inputMapping) {
          mapping.set(pair.source + offset, pair.target + newOffset);
        }
        break;
      default:
        relBuilder.join(join.getJoinType(), newConditionExpr);
    }
    relBuilder.hints(join.getHints());
    return result(relBuilder.build(), mapping, join);
  }

  public TrimResult trimFields(
      TableModify tableModify, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {

    // Ignore passed in fieldsUsed and extraFields
    // and just recursively call trim on the input
    RelNode trimmedInput = this.trim(tableModify.getInput());
    List<RelNode> tempList = new ArrayList<>();
    tempList.add(trimmedInput);
    final RelNode newTableModify = tableModify.copy(tableModify.getTraitSet(), tempList);
    return result(
        newTableModify,
        Mappings.createIdentity(tableModify.getRowType().getFieldCount()),
        tableModify);
  }

  /**
   * Variant of {@link #trimFields(RelNode, ImmutableBitSet, Set)} for {@link
   * org.apache.calcite.rel.logical.LogicalAggregate}.
   */
  @Override
  public TrimResult trimFields(
      Aggregate aggregate, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    // Fields:
    //
    // | sys fields | group fields | indicator fields | agg functions |
    //
    // Two kinds of trimming:
    //
    // 1. If agg rel has system fields but none of these are used, create an
    // agg rel with no system fields.
    //
    // 2. If aggregate functions are not used, remove them.
    //
    // But group and indicator fields stay, even if they are not used.

    final RelDataType rowType = aggregate.getRowType();

    // Compute which input fields are used.
    // 1. group fields are always used
    final ImmutableBitSet.Builder inputFieldsUsed = aggregate.getGroupSet().rebuild();
    // 2. agg functions
    for (AggregateCall aggCall : aggregate.getAggCallList()) {
      inputFieldsUsed.addAll(aggCall.getArgList());
      if (aggCall.filterArg >= 0) {
        inputFieldsUsed.set(aggCall.filterArg);
      }
      if (aggCall.distinctKeys != null) {
        inputFieldsUsed.addAll(aggCall.distinctKeys);
      }
      inputFieldsUsed.addAll(RelCollations.ordinals(aggCall.collation));
    }

    // Create input with trimmed columns.
    final RelNode input = aggregate.getInput();
    final Set<RelDataTypeField> inputExtraFields = Collections.emptySet();
    final ImmutableBitSet usedInputFields = inputFieldsUsed.build();
    final TrimResult incompleteTrimResult =
        trimChild(aggregate, input, usedInputFields, inputExtraFields);

    // Bodo change, always prune all unused input columns
    final TrimResult trimResult =
        insertPruningProjection(incompleteTrimResult, usedInputFields, inputExtraFields);

    final RelNode newInput = trimResult.left;
    final Mapping inputMapping = trimResult.right;

    // We have to return group keys and (if present) indicators.
    // So, pretend that the consumer asked for them.
    final int groupCount = aggregate.getGroupSet().cardinality();
    fieldsUsed = fieldsUsed.union(ImmutableBitSet.range(groupCount));

    // If the input is unchanged, and we need to project all columns,
    // there's nothing to do.
    if (input == newInput && fieldsUsed.equals(ImmutableBitSet.range(rowType.getFieldCount()))) {
      return result(aggregate, Mappings.createIdentity(rowType.getFieldCount()));
    }

    // Which agg calls are used by our consumer?
    int j = groupCount;
    int usedAggCallCount = 0;
    for (int i = 0; i < aggregate.getAggCallList().size(); i++) {
      if (fieldsUsed.get(j++)) {
        ++usedAggCallCount;
      }
    }

    // Offset due to the number of system fields having changed.
    Mapping mapping =
        Mappings.create(
            MappingType.INVERSE_SURJECTION, rowType.getFieldCount(), groupCount + usedAggCallCount);

    final ImmutableBitSet newGroupSet = Mappings.apply(inputMapping, aggregate.getGroupSet());

    final ImmutableList<ImmutableBitSet> newGroupSets =
        ImmutableList.copyOf(
            Util.transform(
                aggregate.getGroupSets(), input1 -> Mappings.apply(inputMapping, input1)));

    // Populate mapping of where to find the fields. System, group key and
    // indicator fields first.
    for (j = 0; j < groupCount; j++) {
      mapping.set(j, j);
    }

    // Now create new agg calls, and populate mapping for them.
    relBuilder.push(newInput);
    final List<RelBuilder.AggCall> newAggCallList = new ArrayList<>();
    j = groupCount;
    for (AggregateCall aggCall : aggregate.getAggCallList()) {
      if (fieldsUsed.get(j)) {
        mapping.set(j, groupCount + newAggCallList.size());
        newAggCallList.add(relBuilder.aggregateCall(aggCall, inputMapping));
      }
      ++j;
    }

    if (newAggCallList.isEmpty() && newGroupSet.isEmpty()) {
      // Add a dummy call if all the column fields have been trimmed
      mapping = Mappings.create(MappingType.INVERSE_SURJECTION, mapping.getSourceCount(), 1);
      newAggCallList.add(relBuilder.count(false, "DUMMY"));
    }

    final RelBuilder.GroupKey groupKey = relBuilder.groupKey(newGroupSet, newGroupSets);
    relBuilder.aggregate(groupKey, newAggCallList);

    final RelNode newAggregate = RelOptUtil.propagateRelHints(aggregate, relBuilder.build());
    return result(newAggregate, mapping, aggregate);
  }

  @Override
  public TrimResult trimFields(
      Filter filter, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final RelDataType rowType = filter.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RexNode conditionExpr = filter.getCondition();
    final RelNode input = filter.getInput();

    // We use the fields used by the consumer, plus any fields used in the
    // filter.
    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder(inputExtraFields, fieldsUsed);
    conditionExpr.accept(inputFinder);
    final ImmutableBitSet inputFieldsUsed = inputFinder.build();

    // Create input with trimmed columns.
    TrimResult incompleteTrimResult = trimChild(filter, input, inputFieldsUsed, inputExtraFields);

    // Bodo change, always prune all unused input columns
    final TrimResult trimResult =
        insertPruningProjection(incompleteTrimResult, inputFieldsUsed, inputExtraFields);

    RelNode newInput = trimResult.left;
    final Mapping inputMapping = trimResult.right;

    // If the input is unchanged, and we need to project all columns,
    // there's nothing we can do.
    if (newInput == input && fieldsUsed.cardinality() == fieldCount) {
      return result(filter, Mappings.createIdentity(fieldCount));
    }

    // Build new project expressions, and populate the mapping.
    final RexVisitor<RexNode> shuttle = new RexPermuteInputsShuttle(inputMapping, newInput);
    RexNode newConditionExpr = conditionExpr.accept(shuttle);

    // Build new filter with trimmed input and condition.
    relBuilder.push(newInput).filter(filter.getVariablesSet(), newConditionExpr);

    // The result has the same mapping as the input gave us. Sometimes we
    // return fields that the consumer didn't ask for, because the filter
    // needs them for its condition.
    return result(relBuilder.build(), inputMapping, filter);
  }

  public TrimResult trimFields(
      PandasMinRowNumberFilter filter,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    // Same idea as the Filter implementation, but uses the MRNF node
    // and allows usage of the inputsToKeep field
    final RelDataType rowType = filter.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RexNode conditionExpr = filter.getCondition();
    final RelNode input = filter.getInput();

    // inputsToKeep should include every field before this point
    if (filter.getInputsToKeep().cardinality() != fieldCount) {
      throw new RuntimeException(
          "PandasMinRowNumberFilter node does not support pruning via inputsToKeep before rel"
              + " trimming");
    }

    // We use the fields used by the consumer, plus any fields used in the
    // filter.
    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder(inputExtraFields, fieldsUsed);
    conditionExpr.accept(inputFinder);
    final ImmutableBitSet inputFieldsUsed = inputFinder.build();

    // Create input with trimmed columns.
    TrimResult incompleteTrimResult = trimChild(filter, input, inputFieldsUsed, inputExtraFields);

    // Bodo change, always prune all unused input columns
    final TrimResult trimResult =
        insertPruningProjection(incompleteTrimResult, inputFieldsUsed, inputExtraFields);

    RelNode newInput = trimResult.left;
    final Mapping inputMapping = trimResult.right;

    // If the input is unchanged, and we need to project all columns,
    // there's nothing we can do.
    if (newInput == input && fieldsUsed.cardinality() == fieldCount) {
      return result(filter, Mappings.createIdentity(fieldCount));
    }

    // Build new project expressions, and populate the mapping.
    final RexVisitor<RexNode> shuttle = new RexPermuteInputsShuttle(inputMapping, newInput);
    RexNode newConditionExpr = conditionExpr.accept(shuttle);

    // Build another mapping to account for the fact that not every field needs to be kept
    // afterward, and build which of the child columns we have to keep.
    ImmutableBitSet.Builder newInputsToKeep = ImmutableBitSet.builder();
    Mapping mapping =
        Mappings.create(
            MappingType.INVERSE_SURJECTION,
            inputMapping.getSourceCount(),
            fieldsUsed.cardinality());
    int keptIdx = 0;
    for (int i : fieldsUsed) {
      newInputsToKeep.set(inputMapping.getTarget(i));
      mapping.set(i, keptIdx);
      keptIdx++;
    }

    // Make a new filter with trimmed input and condition.
    PandasMinRowNumberFilter newFilter =
        PandasMinRowNumberFilter.Companion.create(
            filter.getCluster(), newInput, newConditionExpr, newInputsToKeep.build());

    // The result has the same mapping as the input gave us. Sometimes we
    // return fields that the consumer didn't ask for, because the filter
    // needs them for its condition.
    return result(newFilter, mapping, filter);
  }

  public TrimResult trimFields(
      Flatten flatten, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final RelDataType rowType = flatten.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RexNode flattenExpr = flatten.getCall();
    final RelNode input = flatten.getInput();

    // We use the fields used by the consumer, plus any fields used in the
    // flatten expression.
    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>(extraFields);

    List<Integer> remappedInputIndices = new ArrayList<Integer>();
    List<Integer> usedColOutputs = flatten.getUsedColOutputs().toList();
    int nOutColumns = usedColOutputs.size();
    for (int i : fieldsUsed.toList()) {
      if (i >= nOutColumns) {
        remappedInputIndices.add(i - nOutColumns);
      }
    }

    RelOptUtil.InputFinder inputFinder =
        new RelOptUtil.InputFinder(inputExtraFields, ImmutableBitSet.of(remappedInputIndices));
    flattenExpr.accept(inputFinder);
    final ImmutableBitSet inputFieldsUsed = inputFinder.build();

    // Create input with trimmed columns.
    TrimResult incompleteTrimResult = trimChild(flatten, input, inputFieldsUsed, inputExtraFields);

    // Prune all unused input columns
    final TrimResult trimResult =
        insertPruningProjection(incompleteTrimResult, inputFieldsUsed, inputExtraFields);

    RelNode newInput = trimResult.left;
    final Mapping inputMapping = trimResult.right;

    final RexVisitor<RexNode> shuttle = new RexPermuteInputsShuttle(inputMapping, newInput);
    RexNode newFlattenExpr = flattenExpr.accept(shuttle);

    List<Integer> newUsedColOutputs = new ArrayList<Integer>();
    List<Integer> newRepeatCols = new ArrayList<Integer>();
    List<RelDataTypeField> newRowType = new ArrayList<RelDataTypeField>();
    final Mapping newInputMapping =
        Mappings.create(MappingType.INVERSE_SURJECTION, fieldCount, fieldsUsed.cardinality());
    for (int i = 0; i < usedColOutputs.size(); i++) {
      if (fieldsUsed.get(i)) {
        Integer outIdx = usedColOutputs.get(i);
        newUsedColOutputs.add(outIdx);
        newInputMapping.set(i, newRowType.size());
        newRowType.add(rowType.getFieldList().get(i));
      }
    }
    for (int i : flatten.getRepeatColumns()) {
      int outIdx = i + nOutColumns;
      if (fieldsUsed.get(outIdx)) {
        newRepeatCols.add(inputMapping.getTarget(i));
        newInputMapping.set(outIdx, newRowType.size());
        newRowType.add(rowType.getFieldList().get(outIdx));
      }
    }
    Flatten newFlatten =
        flatten.copy(
            flatten.getTraitSet(),
            newInput,
            (RexCall) newFlattenExpr,
            flatten.getCallType(),
            ImmutableBitSet.of(newUsedColOutputs),
            ImmutableBitSet.of(newRepeatCols));

    return result(newFlatten, newInputMapping, flatten);
  }
}
