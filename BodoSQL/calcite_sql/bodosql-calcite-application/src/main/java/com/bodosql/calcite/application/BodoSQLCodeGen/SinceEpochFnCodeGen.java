package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;

public class SinceEpochFnCodeGen {

  /**
   * Helper function that handles the codegen for mySQL's TO_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToDaysCode(Expr arg1Info) {
    String outputExpr = "bodo.libs.bodosql_array_kernels.to_days(" + arg1Info.emit() + ")";
    return new Expr.Raw(outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's TO_SECONDS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToSecondsCode(Expr arg1Info) {
    String outputExpr = "bodo.libs.bodosql_array_kernels.to_seconds(" + arg1Info.emit() + ")";
    return new Expr.Raw(outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateFromDaysCode(Expr arg1Info) {
    String outputExpr = "bodo.libs.bodosql_array_kernels.from_days(" + arg1Info.emit() + ")";
    return new Expr.Raw(outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_UNIXTIME
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateFromUnixTimeCode(Expr arg1Info) {
    String outputExpression =
        String.format("bodo.libs.bodosql_array_kernels.second_timestamp(%s)", arg1Info.emit());
    return new Expr.Raw(outputExpression);
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
