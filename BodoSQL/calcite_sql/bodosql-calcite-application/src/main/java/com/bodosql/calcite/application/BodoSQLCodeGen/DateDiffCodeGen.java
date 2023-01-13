package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.*;

public class DateDiffCodeGen {

  /**
   * Function that return the necessary generated code for a DateDiff Function Call.
   *
   * @param arg0 The first arg expr.
   * @param arg1 The second arg expr.
   * @return The code generated that matches the DateDiff expression.
   */
  public static String generateDateDiffCode(String arg0, String arg1) {
    // TODO: needs null checking, as null timestamps can be None
    StringBuilder diffExpr = new StringBuilder();
    // Create dummy visitors to reuse date trunc code.
    RexNodeVisitorInfo dayVisitor = new RexNodeVisitorInfo("", makeQuoted("day"));
    diffExpr.append("bodo.libs.bodosql_array_kernels.date_sub_date(");
    diffExpr.append(
        generateDateTruncCode(dayVisitor, new RexNodeVisitorInfo("", arg0)).getExprCode());
    diffExpr.append(", ");
    diffExpr.append(
        generateDateTruncCode(dayVisitor, new RexNodeVisitorInfo("", arg1)).getExprCode());
    diffExpr.append(")");
    return diffExpr.toString();
  }

  /**
   * Function that returns the generated name for a DateDiff Function Call.
   *
   * @param arg0Name The first arg's name.
   * @param arg1Name The second arg's name.
   * @return The name generated that matches the DateDiff expression.
   */
  public static String generateDateDiffName(String arg0Name, String arg1Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("DATEDIFF(").append(arg0Name).append(", ").append(arg1Name).append(")");
    return nameBuilder.toString();
  }
}
