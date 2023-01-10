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
   * @param inputVar Input Dataframe variable name
   * @param exprTypes ExpressionTypes of the arguments to the function call
   * @param operandsInfo RexVisitorinfo's of the arguments to the function call
   * @return
   */
  public static RexNodeVisitorInfo generateTimestampDiffInfo(
      String inputVar,
      List<BodoSQLExprType.ExprType> exprTypes,
      List<RexNodeVisitorInfo> operandsInfo,
      BodoCtx ctx) {

    assert exprTypes.size() == 3 && operandsInfo.size() == 3;

    String outputName =
        "TIMESTAMPDIFF("
            + operandsInfo.get(0).getName()
            + ", "
            + operandsInfo.get(1).getName()
            + ", "
            + operandsInfo.get(2).getName()
            + ", "
            + ")";

    String flagName = operandsInfo.get(0).getExprCode();
    switch (flagName) {
      case "YEAR":
      case "QUARTER":
      case "MONTH":
      case "WEEK":
      case "DAY":
      case "HOUR":
      case "MINUTE":
      case "SECOND":
      case "MILLISECOND":
      case "MICROSECOND":
      case "NANOSECOND":
        break;
      default:
        throw new BodoSQLCodegenException("Unsupported unit for TIMESTAMPDIFF: " + flagName);
    }

    String arg0Expr = operandsInfo.get(1).getExprCode();
    String arg1Expr = operandsInfo.get(2).getExprCode();
    StringBuilder output = new StringBuilder();
    output
        .append("bodo.libs.bodosql_array_kernels.diff_")
        .append(flagName.toLowerCase())
        .append("(")
        .append(arg0Expr)
        .append(", ")
        .append(arg1Expr)
        .append(")");

    return new RexNodeVisitorInfo(outputName, output.toString());
  }
}
