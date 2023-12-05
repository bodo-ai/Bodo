package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.List;

public class DateDiffCodeGen {
  public static Expr generateDateDiffFnInfo(String unit, Expr arg1Info, Expr arg2Info) {
    return ExprKt.BodoSQLKernel("diff_" + unit, List.of(arg1Info, arg2Info), List.of());
  }
}
