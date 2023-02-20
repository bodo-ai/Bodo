package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;

import com.bodosql.calcite.application.*;

public class DateDiffCodeGen {
  public static String generateDateDiffCode(String arg0, String arg1, String arg2) {
    StringBuilder diffExpr = new StringBuilder();
    // Create dummy visitors to reuse date trunc code.
    String unit = standardizeTimeUnit("DATEDIFF", arg0, false);
    diffExpr.append("bodo.libs.bodosql_array_kernels.date_sub_date_unit(\"");
    diffExpr.append(arg0).append("\", ");
    diffExpr.append(
        generateDateTruncCode(unit, new RexNodeVisitorInfo(arg1)).getExprCode());

    diffExpr.append(", ");
    diffExpr.append(
        generateDateTruncCode(unit, new RexNodeVisitorInfo(arg2)).getExprCode());
    diffExpr.append(")");
    return diffExpr.toString();
  }


  public static RexNodeVisitorInfo generateDateDiffFnInfo(
      String unit, RexNodeVisitorInfo arg1Info, RexNodeVisitorInfo arg2Info) {
    String diffExpr =
        generateDateDiffCode(unit, arg1Info.getExprCode(), arg2Info.getExprCode());
    return new RexNodeVisitorInfo(diffExpr);
  }
}
