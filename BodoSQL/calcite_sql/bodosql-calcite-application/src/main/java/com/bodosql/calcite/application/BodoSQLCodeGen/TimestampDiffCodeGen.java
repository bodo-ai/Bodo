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
   * @param isSingleRow if the function takes place within an apply
   * @return
   */
  public static RexNodeVisitorInfo generateTimestampDiffInfo(
      String inputVar,
      List<BodoSQLExprType.ExprType> exprTypes,
      List<RexNodeVisitorInfo> operandsInfo,
      boolean isSingleRow,
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

    String arg0Expr = operandsInfo.get(1).getExprCode();
    String arg1Expr = operandsInfo.get(2).getExprCode();
    BodoSQLExprType.ExprType arg0ExprType = exprTypes.get(1);
    BodoSQLExprType.ExprType arg1ExprType = exprTypes.get(2);
    boolean arg0Scalar = isSingleRow || arg0ExprType == BodoSQLExprType.ExprType.SCALAR;
    boolean arg1Scalar = isSingleRow || arg1ExprType == BodoSQLExprType.ExprType.SCALAR;

    // Largely copied from DATE_SUB, basically just does null checking subtraction of the two
    // timestamp
    // values
    StringBuilder diffExpr = new StringBuilder();
    boolean allArgsScalar = arg0Scalar && arg1Scalar;
    if (allArgsScalar) {
      diffExpr.append("bodosql.libs.generated_lib.sql_null_checking_subtraction(");
      diffExpr.append(arg1Expr);
      diffExpr.append(", ");
      diffExpr.append(arg0Expr);
      diffExpr.append(")");
    } else {
      diffExpr.append("(");
      diffExpr.append(arg1Expr);
      diffExpr.append(" - ");
      diffExpr.append(arg0Expr);
      diffExpr.append(")");
    }

    String output;
    String flagName = operandsInfo.get(0).getExprCode();
    // TODO: If/When we stop using the null set, the scalar calls below will need to be replaced
    // with their
    // null checking equivalents.
    switch (flagName) {
        // For these cases, it makes more sense to do subtraction / division
        // We do float division and then cast to Int64, since the cast rounds towards zero,
        // while python's // rounds towards -inf

        // For some weird reason, I need to always use.value in the float case or else very small
        // time differences
        // are not accounted for
        /*
        In [9]: x = pd.Timestamp("2025-01-01")

        In [10]: x_p_1 = x + pd.Timedelta(1, unit="minute")

        In [11]: np.int64(((x - x_p_1) + pd.Timedelta(1, unit="ns")).total_seconds() / 60)
        Out[11]: -1

        In [12]: (((pd.Series(x) - pd.Series(x_p_1)) + pd.Timedelta(1, unit="ns")).dt.total_seconds() / 60).astype('int64')
        Out[12]:
        0    0
        dtype: int64

        In [13]: ((x - x_p_1) + pd.Timedelta(1, unit="ns")).total_seconds() / 60
        Out[13]: -1.0

        In [14]: ((pd.Series(x) - pd.Series(x_p_1)) + pd.Timedelta(1, unit="ns")).dt.total_seconds() / 60
        Out[14]:
        0   -1.0
        dtype: float64
         */

      case "NANOSECOND":
        if (allArgsScalar) {
          output = diffExpr + ".value";
        } else {
          output = diffExpr + ".astype(\"Int64\")";
        }
        break;
      case "MICROSECOND":
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 1000" + ")";
        } else {
          output = "(" + diffExpr + ".astype(\"Int64\") / 1000" + ").astype(\"Int64\")";
        }
        break;
      case "SECOND":
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 1000000000" + ")";
        } else {
          output = diffExpr + ".dt.total_seconds().astype(\"Int64\")";
        }
        break;
      case "MINUTE":
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 60000000000)";
        } else {
          output = "(" + diffExpr + ".dt.total_seconds() / 60).astype(\"Int64\")";
        }
        break;
      case "HOUR":
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 3600000000000)";
        } else {
          output = "(" + diffExpr + ".dt.total_seconds() / 3600).astype(\"Int64\")";
        }
        break;
      case "DAY":
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 86400000000000)";
        } else {
          output = "(" + diffExpr + ".dt.total_seconds() / 86400).astype(\"Int64\")";
        }
        break;
      case "WEEK":
        // Thankfully, for week, it seems like a difference in weeks is actually just a difference
        // of 7 days.
        // SELECT TIMESTAMPDIFF(WEEK, '2021-07-20', '2021-07-27') == 1
        // SELECT TIMESTAMPDIFF(WEEK, '2021-07-21', '2021-07-27') == 0
        // SELECT TIMESTAMPDIFF(WEEK, '2021-07-21', '2021-07-28') == 1
        if (allArgsScalar) {
          output = "np.int64(" + diffExpr + ".value / 604800000000000)";
        } else {
          output = "(" + diffExpr + ".dt.total_seconds() / 604800).astype(\"Int64\")";
        }
        break;
        // For these cases, we need to use a custom helper function, since months do not all
        // have the same number of days
      case "MONTH":
      case "QUARTER":
      case "YEAR":
        // Quarter is months divided by 3
        // and a year is just 4 quarters
        output = "bodo.libs.bodosql_array_kernels.month_diff(" + arg0Expr + ", " + arg1Expr + ")";

        // for quarter and month, we just do a division, and cast to int64 to round towards zero
        if (flagName.equals("QUARTER")) {
          if (allArgsScalar) {
            output = "np.int64(" + output + " / 3)";
          } else {
            output = "(" + output + " / 3).astype(\"Int64\")";
          }
        } else if (flagName.equals("YEAR")) {
          if (allArgsScalar) {
            output = "np.int64(" + output + " / 12)";
          } else {
            output = "(" + output + " / 12).astype(\"Int64\")";
          }
        }

        break;
      default:
        throw new BodoSQLCodegenException(
            "Unsupported flag passed into TimestampDiff: " + operandsInfo.get(0).getExprCode());
    }

    return new RexNodeVisitorInfo(outputName, output);
  }
}
