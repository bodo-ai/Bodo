package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.BodoSQLOperatorTables.*;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalProject;
import java.util.*;
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
   * Determines if the given Filter is just an inputRef.
   *
   * @param filter The filter to check.
   * @return Is the filter's condition an InputRef.
   */
  public static boolean inputRefFilter(Filter filter) {
    RexNode cond = filter.getCondition();
    return cond instanceof RexInputRef;
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
   * Determines if the filter could match this optimization. Currently, we only support EXPR = 1, 1
   * = EXPR, and InputRef. These will be checked for exact support after the match.
   *
   * @param filter THe filter to check.
   * @return Should this rule consider this filter?
   */
  public static boolean candidateRowNumberFilter(Filter filter) {
    Pair<Boolean, RexNode> filterInfo = equalsOneFilter(filter.getCondition(), true);
    return filterInfo.getKey() || inputRefFilter(filter);
  }

  public static int getWindowColNum(final Filter filter, final boolean equalsInProjection) {
    final RexInputRef inputCol;
    if (equalsInProjection) {
      inputCol = (RexInputRef) filter.getCondition();
    } else {
      RexCall condCall = (RexCall) filter.getCondition();
      List<RexNode> equalsOperands = condCall.getOperands();
      if (equalsOperands.get(0) instanceof RexInputRef) {
        inputCol = ((RexInputRef) equalsOperands.get(0));
      } else {
        inputCol = ((RexInputRef) equalsOperands.get(1));
      }
    }
    return inputCol.getIndex();
  }

  public static boolean isColumnValidRowNumber(
      final Project project, final int windowColNum, final boolean equalsInProjection) {
    List<RexNode> colProjects = project.getProjects();
    RexNode rowNumberCandidate = colProjects.get(windowColNum);
    if (equalsInProjection) {
      // If the projection was just an input ref we need to do the filter check
      // and update our candidate.
      Pair<Boolean, RexNode> filterInfo = equalsOneFilter(rowNumberCandidate, false);
      if (!filterInfo.getKey()) {
        return false;
      }
      rowNumberCandidate = filterInfo.getValue();
    }
    if (!(rowNumberCandidate instanceof RexOver)) {
      // If we don't have a RexOver we cannot proceed.
      return false;
    }
    RexOver overFunction = (RexOver) rowNumberCandidate;
    if (!overFunction.getOperator().getKind().equals(SqlKind.ROW_NUMBER)) {
      // If we don't have ROW_NUMBER then we can't proceed.
      return false;
    }
    RexWindow window = overFunction.getWindow();
    // Verify the Window can be processed. We currently require
    // at one partition by column.
    if (window.partitionKeys.size() == 0) {
      // We cannot handle this structure.
      return false;
    }
    return true;
  }

  public static boolean projectContainsOtherWindows(final Project project, final int windowColNum) {
    List<RexNode> colProjects = project.getProjects();
    for (int i = 0; i < colProjects.size(); i++) {
      // Check the all other columns do not contain any window functions
      if (i != windowColNum) {
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

  public void buildNewFilter(
      RelBuilder builder,
      final Project origProject,
      final int windowColNum,
      final boolean equalsInProjection) {
    // Fetch the original Window column
    List<RexNode> colProjects = origProject.getProjects();
    RexNode column = colProjects.get(windowColNum);
    RexOver overNode;
    if (equalsInProjection) {
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
    RexNode newNode =
        builder
            .aggregateCall(CondOperatorTable.MIN_ROW_NUMBER_FILTER, overNode.getOperands())
            .distinct(overNode.isDistinct())
            .ignoreNulls(overNode.ignoreNulls())
            .over()
            .partitionBy(window.partitionKeys)
            .orderBy(newOrderKeys)
            .rangeBetween(window.getLowerBound(), window.getUpperBound())
            .toRex();
    builder.filter(newNode);
  }

  public void buildNewProject(
      RelBuilder builder,
      final Project origProject,
      final int windowColNum,
      final boolean equalsInProjection) {
    List<RexNode> oldProjects = origProject.getProjects();
    List<RexNode> newProjects = new ArrayList<>(oldProjects.size());
    List<String> oldFieldNames = origProject.getRowType().getFieldNames();
    List<String> newFieldNames = new ArrayList<>(oldProjects.size());
    for (int i = 0; i < oldProjects.size(); i++) {
      if (i == windowColNum) {
        // Replace the OverNode with a RexLiteral 1 or True, the output
        // of the filter. We need to do a cast to ensure the types match.
        RexNode literalValue;
        if (equalsInProjection) {
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
    boolean equalsInProjection = inputRefFilter(origFilter);
    // Get the column to check
    final int windowColNum = getWindowColNum(origFilter, equalsInProjection);
    // Verify that we have a matching row number
    if (!isColumnValidRowNumber(origProject, windowColNum, equalsInProjection)) {
      return;
    }
    // Verify that no other projection node contains a window function
    if (projectContainsOtherWindows(origProject, windowColNum)) {
      return;
    }
    // Now perform the actual update.
    RelBuilder builder = call.builder();
    builder.push(origProject.getInput());
    buildNewFilter(builder, origProject, windowColNum, equalsInProjection);
    buildNewProject(builder, origProject, windowColNum, equalsInProjection);
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
}
