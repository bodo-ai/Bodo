package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.operatorTables.AggOperatorTable;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalProject;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.*;
import org.immutables.value.*;
import org.jetbrains.annotations.NotNull;

/**
 * Planner rule that recognizes a {@link BodoLogicalFilter} on top of a {@link BodoLogicalProject}
 * where the filter condition is row_number() = 1 (which is a common case). This verifies that the
 * filter can safely be placed before the projection, pushes a new filter into the project that
 * computes a boolean array which is true where row_number() = 1 and false elsewhere, but enables
 * skipping the sorts. The row_number() is then also replaced with a constant 1.
 *
 * <p>This optimization ensures we avoid some unnecessary duplicate subexpressions and could improve
 * the performance of filters/filter push down in the future.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class MinRowNumberFilterRule extends RelRule<MinRowNumberFilterRule.Config>
    implements TransformationRule {

  protected MinRowNumberFilterRule(MinRowNumberFilterRule.Config config) {
    super(config);
  }

  /**
   * Determines if the given Filter is exactly EXPR = 1 or 1 = EXPR. If we pass checkForInputRef
   * then EXPR must be an input ref
   *
   * @param cond The filter condition to check.
   * @param checkForInputRef Is the expression required to be an inputRef
   * @return Pair of if the filter matches and the non-literal RexNode (possibly null).
   */
  public static Pair<Boolean, RexNode> equalsOneFilter(RexNode cond, boolean checkForInputRef) {
    if (cond.getKind() == SqlKind.EQUALS) {
      RexCall equalsNode = (RexCall) cond;
      List<RexNode> operands = equalsNode.getOperands();
      if (operands.size() == 2) {
        if (operands.get(0) instanceof RexLiteral) {
          boolean isValid = !checkForInputRef || (operands.get(1) instanceof RexInputRef);
          RexLiteral literal = (RexLiteral) operands.get(0);
          return new Pair<>(isValid && literal.getValue().toString().equals("1"), operands.get(1));
        } else if (operands.get(1) instanceof RexLiteral) {
          boolean isValid = !checkForInputRef || (operands.get(0) instanceof RexInputRef);
          RexLiteral literal = (RexLiteral) operands.get(1);
          return new Pair<>(isValid && literal.getValue().toString().equals("1"), operands.get(0));
        }
      }
    }
    return new Pair<>(false, null);
  }

  /**
   * Determines if the filter could match this optimization. Currently, we match if any filter
   * contains (EXPR = 1, 1 = EXPR, or InputRef). These will be checked for exact support after the
   * match.
   *
   * @param filter The filter to check.
   * @return Should this rule consider this filter?
   */
  public static boolean candidateRowNumberFilter(Filter filter) {
    // "Flatten" the conditions to remove any AND conditions.
    // We don't consider OR because we cannot guarantee pruning the ROW_NUMBER(),
    // so we may actually do more work.
    List<RexNode> conditions = RelOptUtil.conjunctions(filter.getCondition());
    for (RexNode condition : conditions) {
      Pair<Boolean, RexNode> filterInfo = equalsOneFilter(condition, true);
      if (filterInfo.getKey() || condition instanceof RexInputRef) {
        return true;
      }
    }
    return false;
  }

  /**
   * Determine the indices of the input refs that are candidates for replacement.
   *
   * @param filterNodes List of RexNodes to check for the inputRefs.
   * @return A set of information about columns (column number, format, original location in the
   *     Filter's condition) that may satisfy our window function requirement.
   */
  public static Set<WindowColumnInfo> getWindowCandidateColumnIndices(
      final List<RexNode> filterNodes) {
    Set<WindowColumnInfo> windowColumnInfo = new HashSet<>();
    for (RexNode filterNode : filterNodes) {
      if (filterNode instanceof RexInputRef) {
        windowColumnInfo.add(
            new WindowColumnInfo(((RexInputRef) filterNode).getIndex(), true, filterNode));
      } else {
        Pair<Boolean, RexNode> filterInfo = equalsOneFilter(filterNode, true);
        // Verify this is a filter that could match.
        if (filterInfo.getKey()) {
          RexCall condCall = (RexCall) filterNode;
          List<RexNode> equalsOperands = condCall.getOperands();
          final RexInputRef inputCol;
          if (equalsOperands.get(0) instanceof RexInputRef) {
            inputCol = ((RexInputRef) equalsOperands.get(0));
          } else {
            inputCol = ((RexInputRef) equalsOperands.get(1));
          }
          windowColumnInfo.add(new WindowColumnInfo(inputCol.getIndex(), false, filterNode));
        }
      }
    }
    return windowColumnInfo;
  }

  public static Set<WindowColumnInfo> findValidRowNumberIndices(
      final Project project, Set<WindowColumnInfo> candidateColumns) {
    Set<WindowColumnInfo> keptColumns = new HashSet<>();
    List<RexNode> colProjects = project.getProjects();
    for (WindowColumnInfo windowColumnInfo : candidateColumns) {

      RexNode rowNumberCandidate = colProjects.get(windowColumnInfo.index);
      if (windowColumnInfo.expectsEquality) {
        // If the projection was just an input ref we need to do the filter check
        // and update our candidate.
        Pair<Boolean, RexNode> filterInfo = equalsOneFilter(rowNumberCandidate, false);
        if (!filterInfo.getKey()) {
          continue;
        }
        rowNumberCandidate = filterInfo.getValue();
      }
      if (!(rowNumberCandidate instanceof RexOver)) {
        // If we don't have a RexOver skip to the next candidate.
        continue;
      }
      RexOver overFunction = (RexOver) rowNumberCandidate;
      if (!overFunction.getOperator().getKind().equals(SqlKind.ROW_NUMBER)) {
        // If we don't have ROW_NUMBER, then go to the next candidate.
        continue;
      }
      RexWindow window = overFunction.getWindow();
      // Verify the Window can be processed. We currently require
      // at one partition by column.
      if (window.partitionKeys.size() != 0) {
        keptColumns.add(windowColumnInfo);
      }
    }
    return keptColumns;
  }

  public static boolean projectContainsOtherWindows(
      final Project project, final WindowColumnInfo windowColumn) {
    List<RexNode> colProjects = project.getProjects();
    for (int i = 0; i < colProjects.size(); i++) {
      // Check the all other columns do not contain any window functions
      if (i != windowColumn.index) {
        try {
          colProjects
              .get(i)
              .accept(
                  new RexVisitorImpl<Void>(true) {
                    @Override
                    public Void visitCall(RexCall call) {
                      if (call instanceof RexOver) {
                        throw Util.FoundOne.NULL;
                      }
                      return super.visitCall(call);
                    }
                  });
        } catch (Util.FoundOne e) {
          // If we found a RexOver we failed.
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Builder the min_row_number_filter from the existing filter.
   *
   * @param builder The Relbuilder
   * @param origFilter The original filter condition. This is used to find the
   *     min_row_number_filter.
   * @param origProject The original projection. This is used to build the min_row_number_filter.
   * @param windowColumn The information about the window column.
   */
  public void buildMinRowNumberFilter(
      RelBuilder builder,
      Filter origFilter,
      final Project origProject,
      final WindowColumnInfo windowColumn) {
    // Find the part of the original condition to replace
    List<RexNode> oldConditions = RelOptUtil.conjunctions(origFilter.getCondition());
    for (RexNode cond : oldConditions) {
      if (cond.equals(windowColumn.sourceRexNode)) {
        // Fetch the original Window column
        List<RexNode> colProjects = origProject.getProjects();
        RexNode column = colProjects.get(windowColumn.index);
        RexOver overNode;
        if (windowColumn.expectsEquality) {
          // This must be row_number() = 1 or 1 = row_number()
          RexCall equalsNode = (RexCall) column;
          List<RexNode> operands = equalsNode.getOperands();
          if (operands.get(0) instanceof RexOver) {
            overNode = (RexOver) operands.get(0);
          } else {
            overNode = (RexOver) operands.get(1);
          }
        } else {
          overNode = (RexOver) column;
        }
        RexWindow window = overNode.getWindow();
        // Create a new window but update the window function.
        // Note the API won't just let us pass the window keys
        // so we need to copy them directly
        List<RexNode> newOrderKeys = new ArrayList<>();
        for (RexFieldCollation childCollation : window.orderKeys) {
          RexNode child = childCollation.getKey();
          Set<SqlKind> kinds = childCollation.getValue();
          if (kinds.contains(SqlKind.NULLS_FIRST)) {
            child = builder.nullsFirst(child);
          }
          if (kinds.contains(SqlKind.NULLS_LAST)) {
            child = builder.nullsLast(child);
          }
          if (kinds.contains(SqlKind.DESCENDING)) {
            child = builder.desc(child);
          }
          newOrderKeys.add(child);
        }
        // Create the new window function. Here we replace the function call
        // with the min_row_number_filter internal function.
        RelBuilder.OverCall baseCall =
            builder
                .aggregateCall(AggOperatorTable.MIN_ROW_NUMBER_FILTER, overNode.getOperands())
                .distinct(overNode.isDistinct())
                .ignoreNulls(overNode.ignoreNulls())
                .over()
                .partitionBy(window.partitionKeys)
                .orderBy(newOrderKeys);
        final RexNode newNode;
        if (window.isRows()) {
          newNode = baseCall.rowsBetween(window.getLowerBound(), window.getUpperBound()).toRex();
        } else {
          newNode = baseCall.rangeBetween(window.getLowerBound(), window.getUpperBound()).toRex();
        }
        builder.filter(newNode);
        return;
      }
    }
  }

  /**
   * Build the filter for all remaining filter components that don't map to the original
   * min_row_number_filter condition.
   *
   * @param builder The Relbuilder for creating the filter.
   * @param origFilter The original filter conditions.
   * @param windowColumn The information about the window function column.
   */
  public void buildRemainingFilters(
      RelBuilder builder, Filter origFilter, final WindowColumnInfo windowColumn) {
    // Find the parts of the original condition to replace
    List<RexNode> oldConditions = RelOptUtil.conjunctions(origFilter.getCondition());
    // Collect all parts of the condition
    List<RexNode> newConditions = new ArrayList<>();
    for (RexNode cond : oldConditions) {
      if (!cond.equals(windowColumn.sourceRexNode)) {
        newConditions.add(cond);
      }
    }
    if (!newConditions.isEmpty()) {
      builder.filter(newConditions);
    }
  }

  public void buildNewProject(
      RelBuilder builder, final Project origProject, final WindowColumnInfo windowColumn) {
    List<RexNode> oldProjects = origProject.getProjects();
    List<RexNode> newProjects = new ArrayList<>(oldProjects.size());
    List<String> oldFieldNames = origProject.getRowType().getFieldNames();
    List<String> newFieldNames = new ArrayList<>(oldProjects.size());
    for (int i = 0; i < oldProjects.size(); i++) {
      if (i == windowColumn.index) {
        // Replace the OverNode with a RexLiteral 1 or True, the output
        // of the filter. We need to do a cast to ensure the types match.
        RexNode literalValue;
        if (windowColumn.expectsEquality) {
          literalValue = builder.literal(true);
        } else {
          literalValue = builder.cast(builder.literal(1), SqlTypeName.BIGINT);
        }
        newProjects.add(i, literalValue);
      } else {
        // Maintain all other original nodes
        newProjects.add(i, oldProjects.get(i));
      }
      newFieldNames.add(i, oldFieldNames.get(i));
    }
    builder.project(newProjects, newFieldNames, false);
  }

  /** This method is called when the rule finds a RelNode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    // We have exactly matched InputRef = 1. Now we need to continue
    // validating this input. In particular at this stage we want to
    // check that:
    // 1. The input ref column is an over column that computes row_number
    // with exactly 1 order by node.
    // 2. There are no additional Window functions in the same projection. If
    // there was another window function we cannot push the filter in front.
    final Filter origFilter = call.rel(0);
    final Project origProject = call.rel(1);
    // Get the nodes in the condition.
    List<RexNode> filterNodes = RelOptUtil.conjunctions(origFilter.getCondition());
    // Get the column(s) to check
    final Set<WindowColumnInfo> windowColumnIndicesInfo =
        getWindowCandidateColumnIndices(filterNodes);
    // Find the columns that actually match
    final Set<WindowColumnInfo> matchingWindowInfo =
        findValidRowNumberIndices(origProject, windowColumnIndicesInfo);
    // We need at least 1 match to proceed. Since there may be gaps with multiple ROW_NUMBER_FILTER
    // calls in
    // the same filter we require exactly 1 match for now.
    if (matchingWindowInfo.size() != 1) {
      return;
    }
    WindowColumnInfo windowInfo = matchingWindowInfo.stream().collect(Collectors.toList()).get(0);

    // Verify that no other projection node contains a window function
    if (projectContainsOtherWindows(origProject, windowInfo)) {
      return;
    }

    // Now perform the actual update.
    RelBuilder builder = call.builder();
    builder.push(origProject.getInput());
    // Build the min row number filter
    buildMinRowNumberFilter(builder, origFilter, origProject, windowInfo);
    // Create the projection
    buildNewProject(builder, origProject, windowInfo);
    // Build any remaining filters
    buildRemainingFilters(builder, origFilter, windowInfo);
    // Grab the output and transform the call.
    RelNode output = builder.build();
    call.transformTo(output);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Descriptions of classes to match. The rule matches any node, b,
    // so long as its type is LogicalFilter and the condition contains
    // an OR.
    MinRowNumberFilterRule.Config DEFAULT =
        ImmutableMinRowNumberFilterRule.Config.builder()
            .build()
            .withOperandFor(BodoLogicalFilter.class, BodoLogicalProject.class);

    @Override
    default MinRowNumberFilterRule toRule() {
      return new MinRowNumberFilterRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default MinRowNumberFilterRule.Config withOperandFor(
        Class<? extends Filter> filterClass, Class<? extends Project> projectClass) {
      // Run this rule whenever we see a filter atop a project
      // and the filter is exactly 1 = InputRef or InputRef = 1
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(MinRowNumberFilterRule::candidateRowNumberFilter)
                      .oneInput(b1 -> b1.operand(projectClass).anyInputs()))
          .as(MinRowNumberFilterRule.Config.class);
    }
  }

  /**
   * Class to represent the information needed to check if a column in a projection matches our
   * required format for ROW_NUMBER()
   */
  private static class WindowColumnInfo {
    // What is the column number in the prior projection.
    final int index;
    // Should the prior column be an equality (TRUE) or
    // an inputRef (FALSE).
    final boolean expectsEquality;
    // The source RexNode in the filter. This is used to simplify
    // updating the filter and enforcing exactly 1 match.
    final @NotNull RexNode sourceRexNode;

    WindowColumnInfo(int index, boolean expectsEquality, @NotNull RexNode sourceRexNode) {
      this.index = index;
      this.expectsEquality = expectsEquality;
      this.sourceRexNode = sourceRexNode;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof WindowColumnInfo) {
        WindowColumnInfo info = (WindowColumnInfo) obj;
        return index == info.index
            && expectsEquality == info.expectsEquality
            && sourceRexNode.equals(info.sourceRexNode);
      }
      return false;
    }

    @Override
    public int hashCode() {
      // Based on the Pair implementation
      return Integer.hashCode(index) ^ Boolean.hashCode(expectsEquality) ^ sourceRexNode.hashCode();
    }
  }
}
