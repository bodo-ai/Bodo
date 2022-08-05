package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.HashMap;
import java.util.List;

/**
 * Class that returns the generated code for Set operations after all the inputs have been visited.
 */
public class SetOpCodeGen {

  /**
   * Function that return the necessary generated code for a Union expression.
   *
   * @param outVar The output variable.
   * @param childExprs The child expressions to be Unioned together. The expressions must all be
   *     dataframes
   * @param childExprColumns the columns of each of the child expressions. Must be the same length
   *     as childExprs
   * @param isAll Is the union a UnionAll Expression.
   * @param outputColumnNames a list containing the expected output column names
   * @return The code generated for the Union expression.
   */
  public static String generateUnionCode(
      String outVar,
      List<String> outputColumnNames,
      List<String> childExprs,
      List<List<String>> childExprColumns,
      boolean isAll) {
    StringBuilder unionBuilder = new StringBuilder();

    // check that the number of child column name lists passed in is equal to the number of child
    // expressions
    assert childExprColumns.size() == childExprs.size();

    // check that the number of columns in each input table is equal to the number of columns we
    // expect to see in the output
    assert (childExprColumns.size() == 0
        || childExprColumns.get(0).size() == outputColumnNames.size());

    final String indent = getBodoIndent();

    // Panda's Concat function can concatenate along the row index such that
    // the new table will have all of the rows of both the tables, with nulls filled in
    // where one table doesn't have a value for a particular column
    // IE  A  C CONCAT  B  C  --->      A    B   C
    //     1  2         2  3            1   NA   2
    //                                  NA   2   3
    //
    // Therefore, in order to get the concatenation function to behave like a union,
    // before I concat the two tables, I need to set the column names as being equal

    // perform the concat, renaming each table to have the output column names
    unionBuilder.append(indent).append(outVar).append(" = ").append("pd.concat((");
    for (int i = 0; i < childExprs.size(); i++) {
      String expr = childExprs.get(i);
      HashMap<String, String> renameMap = new HashMap<>();
      List<String> colNames = childExprColumns.get(i);
      for (int j = 0; j < colNames.size(); j++) {
        if (!colNames.get(j).equals(outputColumnNames.get(j))) {
          renameMap.put(colNames.get(j), outputColumnNames.get(j));
        }
      }
      unionBuilder.append(expr);
      if (!renameMap.isEmpty()) {
        unionBuilder
            .append(".rename(columns=")
            .append(renameColumns(renameMap))
            .append(", copy=False)");
      }
      /* Union All includes duplicates, Union doesn't. preemptively dropping duplicate here to reduce size of the concat */
      if (!isAll) {
        unionBuilder.append(".drop_duplicates()");
      }
      unionBuilder.append(",");
    }
    // We set ignore_index=True for faster runtime performance, as we don't care about the index in
    // BodoSQL
    unionBuilder.append("), ignore_index=True)");

    /* Need to perform a final drop to account for values in both tables */
    if (!isAll) {
      unionBuilder.append(".drop_duplicates()");
    }
    unionBuilder.append("\n");

    return unionBuilder.toString();
  }

  /**
   * Function that return the necessary generated code for a Intersect expression.
   *
   * @param outVar The output variable.
   * @param lhsExpr The expression of the left hand table
   * @param rhsExpr The expression of the left right hand table
   * @param rhsColNames The names of columns of the left hand table
   * @param lhsColNames The names of columns of the left hand table
   * @param columnNames a list containing the expected output column names
   * @return The code generated for the Intersect expression.
   */
  public static String generateIntersectCode(
      String outVar,
      String lhsExpr,
      List<String> lhsColNames,
      String rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames) {
    // we need there to be at least one column, in the right/left table, so we can perform the merge
    // This may be incorrect if Calcite does not optimize out empty intersects
    assert lhsColNames.size() == rhsColNames.size()
        && lhsColNames.size() == columnNames.size()
        && lhsColNames.size() > 0;

    StringBuilder intersectBuilder = new StringBuilder();

    final String indent = getBodoIndent();
    // For this, we rename all the columns to be the same as the expected output columns,
    // and perform an inner merge on each of the columns.
    HashMap<String, String> lhsRenameMap = new HashMap<>();
    HashMap<String, String> rhsRenameMap = new HashMap<>();
    for (int i = 0; i < lhsColNames.size(); i++) {
      lhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
      rhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
    }

    intersectBuilder
        .append(indent)
        .append(outVar)
        .append(" = ")
        .append(lhsExpr)
        .append(".rename(columns=")
        .append(renameColumns(lhsRenameMap))
        .append(", copy=False)")
        .append(".merge(")
        .append(rhsExpr)
        .append(".rename(columns=")
        .append(renameColumns(rhsRenameMap))
        .append(", copy=False), on = [");

    for (int i = 0; i < lhsColNames.size(); i++) {
      intersectBuilder.append(makeQuoted(columnNames.get(i))).append(", ");
    }
    // Intersect removes duplicate entries
    intersectBuilder.append("]).drop_duplicates()\n");

    return intersectBuilder.toString();
  }

  /**
   * Function that return the necessary generated code for a Except expression.
   *
   * @param outVar The output variable.
   * @param throwAwayVar a non conflicting variable name that can be used for intermediate steps in
   *     the Except code
   * @param lhsExpr The expression of the left hand table
   * @param rhsExpr The expression of the left right hand table
   * @param rhsColNames The names of columns of the left hand table
   * @param lhsColNames The names of columns of the left hand table
   * @param columnNames an empty list into which the column names of the output of the Except will
   *     be stored
   * @return The code generated for the Intersect expression.
   */
  public static String generateExceptCode(
      String outVar,
      String throwAwayVar,
      String lhsExpr,
      List<String> lhsColNames,
      String rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames) {
    StringBuilder exceptBuilder = new StringBuilder();
    final String indent = getBodoIndent();
    assert lhsColNames.size() == rhsColNames.size()
        && lhsColNames.size() > 0
        && columnNames.size() == 0;

    throw new BodoSQLCodegenException("Error, except not yet supported");
  }
}
