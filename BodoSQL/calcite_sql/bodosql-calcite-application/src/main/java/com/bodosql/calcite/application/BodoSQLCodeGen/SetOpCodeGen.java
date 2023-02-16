package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.Utils.Utils.renameColumns;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
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
   * @param outputColumnNames a list containing the expected output column names
   * @param childExprs The child expressions to be Unioned together. The expressions must all be
   *     dataframes
   * @param isAll Is the union a UnionAll Expression.
   * @param pdVisitorClass Pandas Visitor used to create global variables.
   * @return The code generated for the Union expression.
   */
  public static String generateUnionCode(
      String outVar,
      List<String> outputColumnNames,
      List<String> childExprs,
      boolean isAll,
      PandasCodeGenVisitor pdVisitorClass) {
    StringBuilder unionBuilder = new StringBuilder();
    unionBuilder
        .append(getBodoIndent())
        .append(outVar)
        .append(" = ")
        .append("bodo.hiframes.pd_dataframe_ext.union_dataframes((");
    for (int i = 0; i < childExprs.size(); i++) {
      unionBuilder.append(childExprs.get(i));
      unionBuilder.append(", ");
    }
    unionBuilder.append("), ");
    if (isAll) {
      unionBuilder.append("False");
    } else {
      unionBuilder.append("True");
    }
    // generate output column names for ColNamesMetaType
    StringBuilder colNameTupleString = new StringBuilder("(");
    for (String colName : outputColumnNames) {
      colNameTupleString.append(makeQuoted(colName)).append(", ");
    }
    colNameTupleString.append(")");
    String globalVarName = pdVisitorClass.lowerAsColNamesMetaType(colNameTupleString.toString());
    unionBuilder.append(", ").append(globalVarName).append(")\n");
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
   * @param isAll Is the intersect an IntersectAll Expression.
   * @return The code generated for the Intersect expression.
   */
  public static String generateIntersectCode(
      String outVar,
      String lhsExpr,
      List<String> lhsColNames,
      String rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames,
      boolean isAll) {
    if (isAll) {
      throw new BodoSQLCodegenException("Intersect All is not supported yet");
    }
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

    intersectBuilder.append(indent).append(outVar).append(" = ").append(lhsExpr);
    if (!isAll) {
      intersectBuilder.append(".drop_duplicates()");
    }
    intersectBuilder
        .append(".rename(columns=")
        .append(renameColumns(lhsRenameMap))
        .append(", copy=False)")
        .append(".merge(")
        .append(rhsExpr);
    /* IntersectAll includes duplicates, Intersect doesn't. preemptively dropping duplicate here to reduce size of the concat */
    if (!isAll) {
      intersectBuilder.append(".drop_duplicates()");
    }
    intersectBuilder
        .append(".rename(columns=")
        .append(renameColumns(rhsRenameMap))
        .append(", copy=False), on = [");

    for (int i = 0; i < lhsColNames.size(); i++) {
      intersectBuilder.append(makeQuoted(columnNames.get(i))).append(", ");
    }
    intersectBuilder.append("])");
    /* Need to perform a final drop to account for values in both tables */
    if (!isAll) {
      intersectBuilder.append(".drop_duplicates()");
    }
    intersectBuilder.append("\n");

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
