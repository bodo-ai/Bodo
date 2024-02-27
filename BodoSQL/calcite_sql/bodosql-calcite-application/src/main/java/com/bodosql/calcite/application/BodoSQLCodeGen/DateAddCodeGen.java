package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.List;

/**
 * Class that returns the generated code for a DateAdd expression after all inputs have been
 * visited.
 */
public class DateAddCodeGen {

  /**
   * Function that return the necessary generated code for a Snowflake DATEADD function call, which
   * adds an integer amount to a datetime of a certain unit.
   *
   * @param operands the list of arguments (UNIT, AMOUNT, START_DATETIME)
   * @return The code generated that matches the DATEADD expression.
   */
  public static Expr generateSnowflakeDateAddCode(List<Expr> operands, String unit) {
    // input check for time unit is moved to standardizeTimeUnit() function,
    // which is called in PandasCodeGenVisitor.java
    return ExprKt.bodoSQLKernel("add_interval_" + unit + "s", operands, List.of());
  }

  /**
   * Function that return the necessary generated code for a MySQL DATEADD function call, which
   * differs from Snowflake DATEADD as follows:
   *
   * <p>Both of the following add 42 days to column A: MySQL: DATEADD(A, 42) Snowflake:
   * DATEADD('day', 42, A)
   *
   * @param arg0 The first starting datetime (or string).
   * @param arg1 The amount of days to add to the starting datetime.
   * @param adding_delta Is the second argument a timedelta?
   * @param fnName The name of the function
   * @return The code generated that matches the DateAdd expression.
   */
  public static Expr generateMySQLDateAddCode(
      Expr arg0, Expr arg1, boolean adding_delta, String fnName) {
    if (fnName.equals("SUBDATE") || fnName.equals("DATE_SUB")) {
      arg1 = ExprKt.bodoSQLKernel("negate", List.of(arg1), List.of());
    }

    if (adding_delta) {
      return ExprKt.bodoSQLKernel("add_interval", List.of(arg0, arg1), List.of());
    } else {
      return ExprKt.bodoSQLKernel("add_interval_days", List.of(arg1, arg0), List.of());
    }
  }
}
