package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;

public class DateDiffCodeGen {
  public static Expr generateDateDiffFnInfo(String unit, Expr arg1Info, Expr arg2Info) {
    StringBuilder diffExpr = new StringBuilder();
    diffExpr
        .append("bodo.libs.bodosql_array_kernels.diff_")
        .append(unit)
        .append("(")
        .append(arg1Info.emit())
        .append(", ")
        .append(arg2Info.emit())
        .append(")");
    return new Expr.Raw(diffExpr.toString());
  }
}
