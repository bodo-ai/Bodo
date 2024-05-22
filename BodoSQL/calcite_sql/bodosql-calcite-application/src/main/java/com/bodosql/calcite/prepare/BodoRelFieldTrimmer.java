package com.bodosql.calcite.prepare;

import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinRowNumberFilter;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRowSample;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalSample;
import com.bodosql.calcite.adapter.iceberg.IcebergProject;
import com.bodosql.calcite.adapter.iceberg.IcebergRel;
import com.bodosql.calcite.adapter.iceberg.IcebergTableScan;
import com.bodosql.calcite.adapter.iceberg.IcebergToBodoPhysicalConverter;
import com.bodosql.calcite.adapter.pandas.PandasToBodoPhysicalConverter;
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToBodoPhysicalConverter;
import com.bodosql.calcite.application.utils.RexNormalizer;
import com.bodosql.calcite.rel.core.CachedSubPlanBase;
import com.bodosql.calcite.rel.core.Flatten;
import com.bodosql.calcite.rel.core.RuntimeJoinFilterBase;
import com.bodosql.calcite.rel.core.cachePlanContainers.CacheNodeSingleVisitHandler;
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
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableCreate;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.sql.validate.SqlValidator;
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

  // Store information about cache nodes, so we can prune columns inside the cache node in a fashion
  // that avoids redundant pruning.
  private final CacheNodeSingleVisitHandler cacheHandler;

  public BodoRelFieldTrimmer(
      @Nullable SqlValidator validator, RelBuilder relBuilder, boolean pruneEverything) {
    super(validator, relBuilder);
    this.relBuilder = relBuilder;
    this.pruneEverything = pruneEverything;
    this.cacheHandler = new CacheNodeSingleVisitHandler();
  }

  @Override
  public RelNode trim(RelNode root) {
    RelNode result = super.trim(root);
    // Process all cache nodes.
    while (cacheHandler.isNotEmpty()) {
      CachedSubPlanBase cacheNode = cacheHandler.pop();
      RelNode cacheRoot = cacheNode.getCachedPlan().getPlan();
      // Just call on the parent implementation so only the top
      // level iterates through cache nodes.
      cacheNode.getCachedPlan().setPlan(super.trim(cacheRoot));
    }
    return result;
  }

  /**
   * Compute a new TrimResult for an input by inserting a projection and creating a new Mapping if
   * there are unused columns even after the input has finished its pruning.
   */
  private TrimResult insertPruningProjection(
      TrimResult childResult, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    int finalFieldCount = fieldsUsed.cardinality();
    // Only completely prune excess columns in the Physical plan step to avoid
    // interfering with other optimizations like Filter push down.
    if (pruneEverything
        && extraFields.size() == 0
        && finalFieldCount < childResult.left.getRowType().getFieldCount()) {
      // Iceberg doesn't support the dummy column project yet, so always keep at least 1 column.
      if (finalFieldCount == 0 && childResult.left instanceof IcebergRel) {
        fieldsUsed = ImmutableBitSet.of(0);
        finalFieldCount = 1;
      }

      // If we move to 0 columns make sure we maintain a dummy column
      if (finalFieldCount == 0) {
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
      PandasToBodoPhysicalConverter node,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      SnowflakeToBodoPhysicalConverter node,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      SnowflakeTableScan node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final Set<RelDataTypeField> combinedInputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder =
        new RelOptUtil.InputFinder(combinedInputExtraFields, fieldsUsed);
    ImmutableBitSet totalFieldsUsed = inputFinder.build();
    int oldFieldCount = node.getRowType().getFieldCount();
    int newFieldCount = totalFieldsUsed.cardinality();
    if (oldFieldCount != 0 && newFieldCount == 0) {
      // We cannot prune all columns without breaking other assumptions.
      totalFieldsUsed = ImmutableBitSet.of(0);
      newFieldCount = 1;
    }
    if (oldFieldCount == newFieldCount) {
      // No change. Just return the original node.
      return result(node, Mappings.createIdentity(oldFieldCount));
    }
    SnowflakeTableScan newNode = node.cloneWithProject(totalFieldsUsed);
    // Update the mapping
    Mapping inputMapping =
        Mappings.create(MappingType.INVERSE_SURJECTION, oldFieldCount, newFieldCount);

    Iterator<Integer> setBitIterator = totalFieldsUsed.iterator();
    for (int i = 0; i < newFieldCount; i++) {
      int idx = setBitIterator.next();
      inputMapping.set(idx, i);
    }
    return result(newNode, inputMapping);
  }

  public TrimResult trimFields(
      IcebergToBodoPhysicalConverter node,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  /**
   * Special handling for IcebergProject to ensure we never prune all columns.
   *
   * @param node The IcebergProject node.
   * @param fieldsUsed The fields used by the consumer.
   * @param extraFields The extra fields used by the consumer.
   * @return The new node and mapping.
   */
  public TrimResult trimFields(
      IcebergProject node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    if (fieldsUsed.cardinality() == 0) {
      // Iceberg doesn't support the dummy projection.
      fieldsUsed = ImmutableBitSet.of(0);
    }
    return trimFields((Project) node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      IcebergTableScan node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final Set<RelDataTypeField> combinedInputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder =
        new RelOptUtil.InputFinder(combinedInputExtraFields, fieldsUsed);
    ImmutableBitSet totalFieldsUsed = inputFinder.build();
    int oldFieldCount = node.getRowType().getFieldCount();
    int newFieldCount = totalFieldsUsed.cardinality();
    if (oldFieldCount != 0 && newFieldCount == 0) {
      // We cannot prune all columns without breaking other assumptions.
      totalFieldsUsed = ImmutableBitSet.of(0);
      newFieldCount = 1;
    }
    if (oldFieldCount == newFieldCount) {
      // No change. Just return the original node.
      return result(node, Mappings.createIdentity(oldFieldCount));
    }
    IcebergTableScan newNode = node.cloneWithProject(totalFieldsUsed);
    // Update the mapping
    Mapping inputMapping =
        Mappings.create(MappingType.INVERSE_SURJECTION, oldFieldCount, newFieldCount);

    Iterator<Integer> setBitIterator = totalFieldsUsed.iterator();
    for (int i = 0; i < newFieldCount; i++) {
      int idx = setBitIterator.next();
      inputMapping.set(idx, i);
    }
    return result(newNode, inputMapping);
  }

  public TrimResult trimFields(
      BodoPhysicalSample node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    return trimFieldsNoUsedColumns(node, fieldsUsed, extraFields);
  }

  public TrimResult trimFields(
      BodoPhysicalRowSample node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
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

    switch (join.getJoinType()) {
      case SEMI:
      case ANTI:
        // For SemiJoins and AntiJoins only map fields from the left-side
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
    }
    RelNode newJoin =
        buildNewJoin(join, newInputs.get(0), newInputs.get(1), newConditionExpr, mapping);
    return result(newJoin, mapping, join);
  }

  /**
   * Wrapper to build a new join that has special handling for the Bodo Physical nodes because we
   * need to propagate extra values.
   *
   * @param oldJoin The old join to convert.
   * @param newLeft The new left child.
   * @param newRight The new right child.
   * @param newConditionExpr The condition.
   * @return The new join.
   */
  private RelNode buildNewJoin(
      Join oldJoin, RelNode newLeft, RelNode newRight, RexNode newConditionExpr, Mapping mapping) {
    if (oldJoin instanceof BodoPhysicalJoin) {
      // We need to remap the key locations because the condition may have updated.
      BodoPhysicalJoin oldBodoJoin = (BodoPhysicalJoin) oldJoin;
      int oldFilterID = oldBodoJoin.getJoinFilterID();
      // Now that we have pruned columns, we may have changed the order of the keys in
      // the join. As a result, we need to track this information for any RuntimeJoinFilters
      // to ensure the proper columns are updated.
      final List<Integer> keyLocationMapping;
      if (oldFilterID != -1) {
        JoinInfo oldInfo = oldBodoJoin.analyzeCondition();
        // Normalize to ensure we get the accurate result. Our JoinBase does this.
        newConditionExpr =
            RexNormalizer.normalize(this.relBuilder.getRexBuilder(), newConditionExpr);
        JoinInfo calculatedInfo = JoinInfo.of(newLeft, newRight, newConditionExpr);
        keyLocationMapping = new ArrayList();
        for (int oldLocation : oldBodoJoin.getOriginalJoinFilterKeyLocations()) {
          int oldKey = oldInfo.leftKeys.get(oldLocation);
          int newKey = mapping.getTarget(oldKey);
          int newLocation = calculatedInfo.leftKeys.indexOf(newKey);
          keyLocationMapping.add(newLocation);
        }
      } else {
        keyLocationMapping = List.of();
      }
      return BodoPhysicalJoin.Companion.create(
          oldBodoJoin.getCluster(),
          oldBodoJoin.getTraitSet(),
          newLeft,
          newRight,
          newConditionExpr,
          oldBodoJoin.getJoinType(),
          oldBodoJoin.getRebalanceOutput(),
          oldBodoJoin.getJoinFilterID(),
          keyLocationMapping);
    } else {
      relBuilder.push(newLeft);
      relBuilder.push(newRight);
      switch (oldJoin.getJoinType()) {
        case SEMI:
        case ANTI:
          // For SemiJoins and AntiJoins only map fields from the left-side
          if (oldJoin.getJoinType() == JoinRelType.SEMI) {
            relBuilder.semiJoin(newConditionExpr);
          } else {
            relBuilder.antiJoin(newConditionExpr);
          }
          break;
        default:
          relBuilder.join(oldJoin.getJoinType(), newConditionExpr);
      }
      relBuilder.hints(oldJoin.getHints());
      return relBuilder.build();
    }
  }

  public TrimResult trimFields(
      RuntimeJoinFilterBase joinFilter,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    // Join Filters don't "use" any columns. Everything that is a join key can't be pruned.
    TrimResult childResult = dispatchTrimFields(joinFilter.getInput(), fieldsUsed, extraFields);
    // Prune if there are still unused columns.
    TrimResult updateResult = insertPruningProjection(childResult, fieldsUsed, extraFields);
    // Remap the used columns.
    List<List<Integer>> newColumnsList = new ArrayList();
    for (List<Integer> columns : joinFilter.getFilterColumns()) {
      List<Integer> newColumns = new ArrayList();
      for (int column : columns) {
        final int newColumn;
        if (column == -1) {
          newColumn = -1;
        } else {
          newColumn = updateResult.right.getTarget(column);
        }
        newColumns.add(newColumn);
      }
      newColumnsList.add(newColumns);
    }
    RuntimeJoinFilterBase newFilter =
        joinFilter.copy(joinFilter.getTraitSet(), updateResult.left, newColumnsList);
    return result(newFilter, updateResult.right, joinFilter);
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
    // Bodo Change: Only add the aggCall if it won't be pruned.
    // Otherwise we don't need to count it as a used field.
    int startIdx = aggregate.getGroupSet().cardinality();
    for (int i = 0; i < aggregate.getAggCallList().size(); i++) {
      AggregateCall aggCall = aggregate.getAggCallList().get(i);
      if (fieldsUsed.get(startIdx + i)) {
        inputFieldsUsed.addAll(aggCall.getArgList());
        if (aggCall.filterArg >= 0) {
          inputFieldsUsed.set(aggCall.filterArg);
        }
        if (aggCall.distinctKeys != null) {
          inputFieldsUsed.addAll(aggCall.distinctKeys);
        }
        inputFieldsUsed.addAll(RelCollations.ordinals(aggCall.collation));
      }
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
      BodoPhysicalMinRowNumberFilter filter,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {
    // Same idea as the Filter implementation, but uses the MRNF node
    // and allows usage of the inputsToKeep field
    final RelDataType rowType = filter.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RexNode conditionExpr = filter.getCondition();
    final RelNode input = filter.getInput();

    // Update the fields based on the inputsToKeep field
    List<Integer> oldInputsToKeep = filter.getInputsToKeep().toList();
    ImmutableBitSet.Builder updatedFieldsUsedBuilder = ImmutableBitSet.builder();
    for (int i : fieldsUsed) {
      updatedFieldsUsedBuilder.set(oldInputsToKeep.get(i));
    }
    ImmutableBitSet updatedFieldsUsed = updatedFieldsUsedBuilder.build();

    // We use the fields used by the consumer, plus any fields used in the
    // filter.
    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>(extraFields);
    RelOptUtil.InputFinder inputFinder =
        new RelOptUtil.InputFinder(inputExtraFields, updatedFieldsUsed);
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
    if (newInput == input && updatedFieldsUsed.cardinality() == fieldCount) {
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
            filter.getRowType().getFieldCount(),
            updatedFieldsUsed.cardinality());
    int keptIdx = 0;
    // Generate the mapping based on the original fields used but the
    // newInputsToKeep needs the updated mapping.
    for (int i : fieldsUsed) {
      newInputsToKeep.set(inputMapping.getTarget(oldInputsToKeep.get(i)));
      mapping.set(i, keptIdx);
      keptIdx++;
    }

    // Make a new filter with trimmed input and condition.
    BodoPhysicalMinRowNumberFilter newFilter =
        BodoPhysicalMinRowNumberFilter.Companion.create(
            filter.getCluster(),
            filter.getTraitSet(),
            newInput,
            newConditionExpr,
            newInputsToKeep.build());

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

  /**
   * Trim result implementation for cache nodes, which just queues the internals for pruning without
   * pruning any fields at the cache's top level.
   *
   * @param node The cache node to trim.
   * @param fieldsUsed The fields used by the parents. This is ignored for now.
   * @param extraFields The extra fields used by the parents. This is ignored for now.
   * @return The original cache node but with a side effect of updating the queue of cache nodes to
   *     process.
   */
  public TrimResult trimFields(
      CachedSubPlanBase node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    // Cache nodes can't be trimmed right now, but we need to queue the body
    // for pruning.
    cacheHandler.add(node);
    return result(node, Mappings.createIdentity(node.getRowType().getFieldCount()));
  }
}
