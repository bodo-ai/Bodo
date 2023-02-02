package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.RexNodeVisitorInfo;

public class SinceEpochFnCodeGen {

  /**
   * Helper function that handles the codegen for mySQL's TO_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateToDaysCode(RexNodeVisitorInfo arg1Info) {
    String name = "TO_DAYS(" + arg1Info.getName() + ")";
    String outputExpr = "bodo.libs.bodosql_array_kernels.to_days(" + arg1Info.getExprCode() + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's TO_SECONDS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateToSecondsCode(RexNodeVisitorInfo arg1Info) {
    String name = "TO_SECONDS(" + arg1Info.getName() + ")";
    String outputExpr =
        "bodo.libs.bodosql_array_kernels.to_seconds(" + arg1Info.getExprCode() + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateFromDaysCode(RexNodeVisitorInfo arg1Info) {
    String name = "FROM_DAYS(" + arg1Info.getName() + ")";
    String outputExpr = "bodo.libs.bodosql_array_kernels.from_days(" + arg1Info.getExprCode() + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_UNIXTIME
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param isScalar should the input value be treated as a scalar
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateFromUnixTimeCode(RexNodeVisitorInfo arg1Info) {
    String name = "FROM_UNIXTIME(" + arg1Info.getName() + ")";
    String outputExpression =
        String.format(
            "bodo.libs.bodosql_array_kernels.second_timestamp(%s)", arg1Info.getExprCode());
    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles the codegen for mySQL's UNIX_TIMESTAMP() .
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateUnixTimestamp() {
    String name = "UNIX_TIMESTAMP()";
    String output = "(pd.Timestamp.now().value // 1000000000)";
    return new RexNodeVisitorInfo(name, output);
  }
}
