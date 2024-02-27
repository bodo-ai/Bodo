package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.List;

public class SinceEpochFnCodeGen {

  /**
   * Helper function that handles the codegen for mySQL's TO_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToDaysCode(Expr arg1Info) {
    return ExprKt.bodoSQLKernel("to_days", List.of(arg1Info), List.of());
  }

  /**
   * Helper function that handles the codegen for mySQL's TO_SECONDS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToSecondsCode(Expr arg1Info) {
    return ExprKt.bodoSQLKernel("to_seconds", List.of(arg1Info), List.of());
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateFromDaysCode(Expr arg1Info) {
    return ExprKt.bodoSQLKernel("from_days", List.of(arg1Info), List.of());
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_UNIXTIME
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateFromUnixTimeCode(Expr arg1Info) {
    return ExprKt.bodoSQLKernel("second_timestamp", List.of(arg1Info), List.of());
  }

  /**
   * Helper function that handles the codegen for mySQL's UNIX_TIMESTAMP() .
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateUnixTimestamp() {
    String output = "(pd.Timestamp.now().value // 1000000000)";
    return new Expr.Raw(output);
  }
}
