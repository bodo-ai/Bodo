package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;

import com.bodosql.calcite.application.*;

public class DateDiffCodeGen {
  public static String generateDateDiffCode(String arg0, String arg1, String arg2) {
    StringBuilder diffExpr = new StringBuilder();
    // Create dummy visitors to reuse date trunc code.
    RexNodeVisitorInfo dayVisitor = new RexNodeVisitorInfo("", arg0);
    diffExpr.append("bodo.libs.bodosql_array_kernels.date_sub_date_unit(");
    diffExpr.append(arg0).append(",");
    diffExpr.append(
        generateDateTruncCode(dayVisitor, new RexNodeVisitorInfo("", arg1)).getExprCode());
    diffExpr.append(", ");
    diffExpr.append(
        generateDateTruncCode(dayVisitor, new RexNodeVisitorInfo("", arg2)).getExprCode());
    diffExpr.append(")");
    return diffExpr.toString();
  }

  public static String generateDateDiffName(String arg0Name, String arg1Name, String arg2Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder
        .append("DATEDIFF(")
        .append(arg0Name)
        .append(", ")
        .append(arg1Name)
        .append(", ")
        .append(arg2Name)
        .append(")");
    return nameBuilder.toString();
  }
}
