package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.application.Utils.BodoCtx;
import java.util.List;

public class TimestampDiffCodeGen {

  /**
   * Function that return the necessary generated code for a TimestampDiff Function Call.
   *
   * @param operandsInfo RexVisitorinfo's of the arguments to the function call
   * @param unit Time unit to calculate the difference between to timestamps/bodo.Time objects
   * @return
   */
  public static RexNodeVisitorInfo generateTimestampDiffInfo(
      List<RexNodeVisitorInfo> operandsInfo,
      String unit) {

    String arg0Expr = operandsInfo.get(1).getExprCode();
    String arg1Expr = operandsInfo.get(2).getExprCode();
    StringBuilder output = new StringBuilder();
    // pintaoz2: TODO: merge TIMEDIFF, DATEDIFF and TIMESTAMPDIFF to use the same kernel
    output
        .append("bodo.libs.bodosql_array_kernels.diff_")
        .append(unit)
        .append("(")
        .append(arg0Expr)
        .append(", ")
        .append(arg1Expr)
        .append(")");

    return new RexNodeVisitorInfo(output.toString());
  }
}
