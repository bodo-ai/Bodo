package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;

import com.bodosql.calcite.application.*;
import com.bodosql.calcite.ir.Expr;

public class DateDiffCodeGen {
  public static String generateDateDiffCode(String arg0, String arg1, String arg2) {
    StringBuilder diffExpr = new StringBuilder();
    // Create dummy visitors to reuse date trunc code.
    String unit = standardizeTimeUnit("DATEDIFF", arg0, DatetimeFnCodeGen.DateTimeType.TIMESTAMP);
    diffExpr.append("bodo.libs.bodosql_array_kernels.date_sub_date_unit(\"");
    diffExpr.append(arg0).append("\", ");
    diffExpr.append(generateDateTruncCode(unit, new Expr.Raw(arg1)).emit());

    diffExpr.append(", ");
    diffExpr.append(generateDateTruncCode(unit, new Expr.Raw(arg2)).emit());
    diffExpr.append(")");
    return diffExpr.toString();
  }

  public static Expr generateDateDiffFnInfo(String unit, Expr arg1Info, Expr arg2Info) {
    String diffExpr = generateDateDiffCode(unit, arg1Info.emit(), arg2Info.emit());
    return new Expr.Raw(diffExpr);
  }
}
