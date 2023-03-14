package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;
import java.util.List;

public class TimestampDiffCodeGen {

  /**
   * Function that return the necessary generated code for a TimestampDiff Function Call.
   *
   * @param operandsInfo RexVisitorinfo's of the arguments to the function call
   * @param unit Time unit to calculate the difference between to timestamps/bodo.Time objects
   * @return
   */
  public static String generateTimestampDiffInfo(List<Expr> operandsInfo, String unit) {

    String arg0Expr = operandsInfo.get(1).emit();
    String arg1Expr = operandsInfo.get(2).emit();
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

    return output.toString();
  }
}
