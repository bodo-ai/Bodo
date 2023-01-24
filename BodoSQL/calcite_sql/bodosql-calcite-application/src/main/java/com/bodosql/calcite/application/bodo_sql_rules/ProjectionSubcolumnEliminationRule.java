package com.bodosql.calcite.application.bodo_sql_rules;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.Map;
import javax.annotation.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.tools.*;
import org.immutables.value.*;

/**
 * Planner rule that recognizes a {@link org.apache.calcite.rel.core.Project} that contains 2 or
 * more subexpressions and checks for any repeated columns. If a column subcomponent is found to
 * exactly match an existing column, then this projection is transformed into two projects, so the
 * second column can reference the original projection without duplicating computation.
 *
 * <p>For more information see this confluence document:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1248952337/Common+SubColumn+Elimination+Rule
 */

// This style ensures that the get/set methods match the naming conventions of those in the calcite
// application
// In calcite, the style is applied globally. I'm uncertain of how they did this, so for now, I'm
// going to manually
// add this annotation to ensure that the style is applied
@BodoSQLStyleImmutable
@Value.Enclosing
public class ProjectionSubcolumnEliminationRule
    extends RelRule<ProjectionSubcolumnEliminationRule.Config> implements TransformationRule {

  // Computing a check for if a Projection can replace a column with a reference to another
  // column is potentially expensive. As a result, we keep a cache of nodes that we know
  // cannot satisfy this rule to skip the check.
  static HashSet<Integer> seenNodes = new HashSet<>();

  /** Creates an AliasPreservingAggregateProjectMergeRule. */
  protected ProjectionSubcolumnEliminationRule(ProjectionSubcolumnEliminationRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project project = call.rel(0);
    RelNode x = apply(call, project);
    if (x != null) {
      call.transformTo(x);
    }
  }

  public static @Nullable RelNode apply(RelOptRuleCall call, Project project) {
    if (seenNodes.contains(project.getId())) {
      // Exit if we have cached this node cannot match.
      return null;
    }
    // Ensure we run on this node only once.
    seenNodes.add(project.getId());
    RelBuilder builder = call.builder();
    // Push the projection's current child to ensure creating input refs
    RelNode input = project.getInput();
    builder.push(input);
    // Create a new projection that combines input refs from each of the older
    // columns with our current projection. This is to handle the edge
    // case where the column that is modified references another column
    // in its unchanged portion. If any of these columns are unused they
    // will be pruned at the end of this function.
    List<RexNode> totalProjects = new ArrayList<>();
    List<String> totalFieldNames = new ArrayList<>();
    // Add the current expressions.
    List<RexNode> columns = project.getProjects();
    List<String> fieldNames = project.getRowType().getFieldNames();
    totalProjects.addAll(columns);
    totalFieldNames.addAll(fieldNames);
    // Create inputRefs to every column of the input
    for (int i = 0; i < input.getRowType().getFieldNames().size(); i++) {
      totalProjects.add(builder.field(i));
      // Generate new names to prevent conflicts
      totalFieldNames.add(null);
    }

    // Create a fused projection so columns in the outer projection can reference
    // columns from the input if necessary. This will be pruned later.
    LogicalProject mergedProject =
        LogicalProject.create(
            project.getInput(), project.getHints(), totalProjects, totalFieldNames);

    // Push the merged projection to enable creating input Refs
    builder.push(mergedProject);

    // Keep track of which columns will be kept
    Set<Integer> keptIndices = new HashSet<>();

    // Create the arrays of nodes for the inner and outer nodes. These will be updated
    // to lists later, but for now we will maintain order.
    int numColumns = project.getProjects().size();
    List<RexNode> outerProjectionNodes = Arrays.asList(new RexNode[numColumns]);

    // Sorting by NodeCount step:
    // Create mapping from NodeCount -> index. This will be used for sorting
    Map<Integer, List<Integer>> countMap = new TreeMap<>();
    for (int i = 0; i < numColumns; i++) {
      RexNode col = columns.get(i);
      int count = col.nodeCount();
      if (!countMap.containsKey(count)) {
        countMap.put(count, new ArrayList<>());
      }
      countMap.get(count).add(i);
    }

    // Checking for Matching Columns step

    // Keep a Map of columns that can replace
    Map<Integer, List<Integer>> replaceMap = new HashMap<>();

    int minCount = -1;
    boolean updated = false;
    for (Map.Entry<Integer, List<Integer>> countInfo : countMap.entrySet()) {
      int currCount = countInfo.getKey();
      List<Integer> columnIndices = countInfo.getValue();
      for (int idx : columnIndices) {
        RexNode currentCol = columns.get(idx);
        boolean replaceColumn = false;
        // Skip the search if we can't replace a column yet.
        if (!replaceMap.isEmpty()) {
          replaceColumn = shouldReplaceColumn(currentCol, minCount, replaceMap, columns);
        }
        if (replaceColumn) {
          // We only want to modify the inputRefs if we will replace the column, so we only do the
          // update
          // in two passes and only modify the column once we know it must be replaced.
          RexNode newCol = replaceSubColumns(builder, currentCol, replaceMap, columns, keptIndices);
          updated = true;
          outerProjectionNodes.set(idx, newCol);
        } else {
          // Mark this column as kept in the original projection.
          keptIndices.add(idx);
          // Create a new inputRef
          outerProjectionNodes.set(idx, builder.field(idx));
          if (currCount > 1) {
            // If the column is unchanged it can be used to replace
            // other columns. Here we require node count > 1 to skip
            // inputRefs and literals.
            if (!replaceMap.containsKey(currCount)) {
              replaceMap.put(currCount, new ArrayList<>());
            }
            replaceMap.get(currCount).add(idx);
            // Update the minimum count
            if (minCount == -1) {
              minCount = currCount;
            }
          }
        }
      }
    }
    if (updated) {
      // Prune Modified Columns Step
      int totalNumColumns = mergedProject.getProjects().size();
      // Determine which column should be removed from the original projection
      int[] adjustments = new int[totalNumColumns];
      int numRemoved = 0;
      List<RexNode> keptCols = new ArrayList<>();
      List<String> keptFieldNames = new ArrayList<>();
      for (int i = 0; i < totalNumColumns; i++) {
        if (!keptIndices.contains(i)) {
          numRemoved++;
        } else {
          keptCols.add(totalProjects.get(i));
          keptFieldNames.add(totalFieldNames.get(i));
        }
        // We decrease the project to account for removed columns.
        adjustments[i] = -numRemoved;
      }
      LogicalProject prunedProject =
          LogicalProject.create(input, project.getHints(), keptCols, keptFieldNames);
      // Add the pruned projection to the cache
      seenNodes.add(prunedProject.getId());
      // Set the correct types for any updated input refs
      builder.push(prunedProject);
      // Update the outerProjectionNodes with the adjustments.
      for (int i = 0; i < outerProjectionNodes.size(); i++) {
        outerProjectionNodes.set(
            i, adjustInputRefs(builder, outerProjectionNodes.get(i), adjustments));
      }
      return LogicalProject.create(
          prunedProject, project.getHints(), outerProjectionNodes, fieldNames);
    } else {
      // If no column needed to be updated just exit.
      return null;
    }
  }

  /**
   * Function used to determine if a column section should be partially replaced with a reference to
   * another column.
   *
   * @param node RexNode section that may need itself or a subcomponent replaced.
   * @param minCount The minimum NodeCount required for a node to be a candidate for replacement.
   * @param replaceMap The map of replacement candidate from nodeCount -> List[indices]
   * @param oldColumns The original list of columns to check for equality.
   * @return Should this RexNode or any subcomponent be replaced.
   */
  public static boolean shouldReplaceColumn(
      RexNode node,
      int minCount,
      Map<Integer, List<Integer>> replaceMap,
      List<RexNode> oldColumns) {
    int nodeCount = node.nodeCount();
    if (nodeCount < minCount) {
      // We cannot have a match in this node or any children
      // if it's smaller than the smallest column to check.
      return false;
    }
    // Check if the nodeCount matches as a cheaper check for possible
    // replacement.
    if (replaceMap.containsKey(nodeCount)) {
      List<Integer> possibleMatches = replaceMap.get(nodeCount);
      for (int idx : possibleMatches) {
        RexNode oldCol = oldColumns.get(idx);
        if (node.equals(oldCol)) {
          // We have a match and will need to update this column
          return true;
        }
      }
    }
    // We don't do anything if we have a RexOver because we cannot
    // replace a RexOver's subexpression.
    if (!(node instanceof RexOver) && node instanceof RexCall) {
      // Call expressions need to have their children traversed. No other RexNodes
      // should have children.
      RexCall callNode = ((RexCall) node);
      List<RexNode> oldOperands = callNode.getOperands();
      for (RexNode oldOperand : oldOperands) {
        if (shouldReplaceColumn(oldOperand, minCount, replaceMap, oldColumns)) {
          return true;
        }
      }
    }
    // The node is unchanged.
    return false;
  }

  /**
   * Takes a RexNode and potentially modifies it or a subcompoent with an inputRef to another
   * column. This function should only be called after shouldReplaceColumn has determined that a
   * column should be modified.
   *
   * @param builder Relbuild for construction new RexNodes.
   * @param node The RexNode that may be updated.
   * @param replaceMap The map of replacement candidate from nodeCount -> List[indices]
   * @param oldColumns The original list of columns to check for equality.
   * @param usedIndices A set used for tracking which columns must be kept alive for pruning. This
   *     is mostly required for tracking any inputRefs used in the modified projection containing
   *     all columns.
   * @return
   */
  public static RexNode replaceSubColumns(
      RelBuilder builder,
      RexNode node,
      Map<Integer, List<Integer>> replaceMap,
      List<RexNode> oldColumns,
      Set<Integer> usedIndices) {
    int nodeCount = node.nodeCount();
    if (node instanceof RexInputRef) {
      // If we encounter a node like *($5, +($4, 1)) and only replace +($4, 1)
      // then we need to make sure we can still find $5 in the outermost projection.
      // To do this we update the InputRef knowing reference to all of original columns were placed
      // after the oldColumns.
      int newIndex = ((RexInputRef) node).getIndex() + oldColumns.size();
      // Mark the index as used
      usedIndices.add(newIndex);
      // Create the new input ref
      return builder.field(newIndex);
    }
    if (replaceMap.containsKey(nodeCount)) {
      List<Integer> possibleMatches = replaceMap.get(nodeCount);
      for (int idx : possibleMatches) {
        RexNode oldCol = oldColumns.get(idx);
        if (node.equals(oldCol)) {
          // This column should already be marked as used, but just
          // in case we mark it here.
          usedIndices.add(idx);
          // We have a match and replace it with an InputRef
          return builder.field(idx);
        }
      }
    }
    if (node instanceof RexOver) {
      // If we have a RexOver we need to replace any Input refs in the partition keys,
      // orderKeys and function call.
      RexOver overNode = ((RexOver) node);
      RexWindow window = overNode.getWindow();
      List<RexNode> newPartitionKeys = new ArrayList<>();
      List<RexNode> newOrderKeys = new ArrayList<>();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode child : window.partitionKeys) {
        RexNode newChild = replaceSubColumns(builder, child, replaceMap, oldColumns, usedIndices);
        newPartitionKeys.add(newChild);
        replaceNode = replaceNode || !newChild.equals(child);
      }
      for (RexFieldCollation childCollation : window.orderKeys) {
        RexNode child = childCollation.getKey();
        Set<SqlKind> kinds = childCollation.getValue();
        RexNode newChild = replaceSubColumns(builder, child, replaceMap, oldColumns, usedIndices);
        // The new Window creates the orderBy from rexnodes that directly contain the nulls first,
        // nulls last,
        // or DESC operation performed on them.
        // https://calcite.apache.org/javadocAggregate/org/apache/calcite/tools/RelBuilder.OverCall.html#orderBy(java.lang.Iterable)
        if (kinds.contains(SqlKind.NULLS_FIRST)) {
          newChild = builder.nullsFirst(newChild);
        }
        if (kinds.contains(SqlKind.NULLS_LAST)) {
          newChild = builder.nullsLast(newChild);
        }
        if (kinds.contains(SqlKind.DESCENDING)) {
          newChild = builder.desc(newChild);
        }
        newOrderKeys.add(newChild);
        replaceNode = replaceNode || !newChild.equals(child);
      }
      for (RexNode oldOperand : overNode.getOperands()) {
        RexNode newOperand =
            replaceSubColumns(builder, oldOperand, replaceMap, oldColumns, usedIndices);
        newOperands.add(newOperand);
        replaceNode = replaceNode || !newOperand.equals(oldOperand);
      }
      if (replaceNode) {
        builder
            .aggregateCall(overNode.getAggOperator(), newOperands)
            .distinct(overNode.isDistinct())
            .ignoreNulls(overNode.ignoreNulls())
            .over()
            .partitionBy(newPartitionKeys)
            .orderBy(newOrderKeys)
            .rangeBetween(window.getLowerBound(), window.getUpperBound());
      }
    } else if (node instanceof RexCall) {
      // Call expressions need to have their children traversed. No other RexNodes
      // should have children.
      RexCall callNode = ((RexCall) node);
      List<RexNode> oldOperands = callNode.getOperands();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode oldOperand : oldOperands) {
        RexNode newOperand =
            replaceSubColumns(builder, oldOperand, replaceMap, oldColumns, usedIndices);
        newOperands.add(newOperand);
        replaceNode = replaceNode || !newOperand.equals(oldOperand);
      }
      if (replaceNode) {
        return builder.call(callNode.getOperator(), newOperands);
      }
    }
    // This portion is unchanged.
    return node;
  }

  public static RexNode adjustInputRefs(RelBuilder builder, RexNode node, int[] adjustments) {
    if (node instanceof RexInputRef) {
      int oldIndex = ((RexInputRef) node).getIndex();
      int adjustment = adjustments[oldIndex];
      if (adjustment != 0) {
        return builder.field(oldIndex + adjustment);
      }
    } else if (node instanceof RexOver) {
      // If we have a RexOver we need to replace any Input refs in the partition keys,
      // orderKeys and function call.
      RexOver overNode = ((RexOver) node);
      RexWindow window = overNode.getWindow();
      List<RexNode> newPartitionKeys = new ArrayList<>();
      List<RexNode> newOrderKeys = new ArrayList<>();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode child : window.partitionKeys) {
        RexNode newChild = adjustInputRefs(builder, child, adjustments);
        newPartitionKeys.add(newChild);
        replaceNode = replaceNode || !newChild.equals(child);
      }
      for (RexFieldCollation childCollation : window.orderKeys) {
        RexNode child = childCollation.getKey();
        Set<SqlKind> kinds = childCollation.getValue();
        RexNode newChild = adjustInputRefs(builder, child, adjustments);
        // The new Window creates the orderBy from rexnodes that directly contain the nulls first,
        // nulls last,
        // or DESC operation performed on them.
        // https://calcite.apache.org/javadocAggregate/org/apache/calcite/tools/RelBuilder.OverCall.html#orderBy(java.lang.Iterable)
        if (kinds.contains(SqlKind.NULLS_FIRST)) {
          newChild = builder.nullsFirst(newChild);
        }
        if (kinds.contains(SqlKind.NULLS_LAST)) {
          newChild = builder.nullsLast(newChild);
        }
        if (kinds.contains(SqlKind.DESCENDING)) {
          newChild = builder.desc(newChild);
        }
        newOrderKeys.add(newChild);
        replaceNode = replaceNode || !newChild.equals(child);
      }
      for (RexNode oldOperand : overNode.getOperands()) {
        RexNode newOperand = adjustInputRefs(builder, oldOperand, adjustments);
        newOperands.add(newOperand);
        replaceNode = replaceNode || !newOperand.equals(oldOperand);
      }
      if (replaceNode) {
        builder
            .aggregateCall(overNode.getAggOperator(), newOperands)
            .distinct(overNode.isDistinct())
            .ignoreNulls(overNode.ignoreNulls())
            .over()
            .partitionBy(newPartitionKeys)
            .orderBy(newOrderKeys)
            .rangeBetween(window.getLowerBound(), window.getUpperBound());
      }
    } else if (node instanceof RexCall) {
      // Call expressions need to have their children traversed. No other RexNodes
      // should have children.
      RexCall callNode = ((RexCall) node);
      List<RexNode> oldOperands = callNode.getOperands();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode oldOperand : oldOperands) {
        RexNode newOperand = adjustInputRefs(builder, oldOperand, adjustments);
        newOperands.add(newOperand);
        replaceNode = replaceNode || !newOperand.equals(oldOperand);
      }
      if (replaceNode) {
        return builder.call(callNode.getOperator(), newOperands);
      }
    }
    // The node is unchanged.
    return node;
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    ProjectionSubcolumnEliminationRule.Config DEFAULT =
        ImmutableProjectionSubcolumnEliminationRule.Config.of().withOperandFor(Project.class);

    @Override
    default ProjectionSubcolumnEliminationRule toRule() {
      return new ProjectionSubcolumnEliminationRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default ProjectionSubcolumnEliminationRule.Config withOperandFor(
        Class<? extends Project> projectClass) {
      return withOperandSupplier(b0 -> b0.operand(projectClass).anyInputs())
          .as(ProjectionSubcolumnEliminationRule.Config.class);
    }
  }
}
