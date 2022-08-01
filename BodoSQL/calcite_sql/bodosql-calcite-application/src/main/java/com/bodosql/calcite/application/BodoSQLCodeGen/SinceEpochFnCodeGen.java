package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.generateBinOpCode;

import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlKind;

public class SinceEpochFnCodeGen {

  // I'm only creating these operator to pass to generateBinopCode
  static SqlBinaryOperator addBinop =
      new SqlBinaryOperator("PLUS", SqlKind.PLUS, 0, true, null, null, null);
  static SqlBinaryOperator subBinop =
      new SqlBinaryOperator("MINUS", SqlKind.MINUS, 0, true, null, null, null);
  // the difference in days between the unix epoch and the start of year 0
  static String dayDeltaUnixY0 = "np.int64(719528)";
  // the difference in days seconds between the unix epoch and the start of year 0
  static String secondDeltaUnixY0 = "np.int64(62167219200)";

  /**
   * Helper function that handles the codegen for mySQL's TO_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param arg1ExprType the expression type of the first argument.
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateToDaysCode(
      RexNodeVisitorInfo arg1Info, BodoSQLExprType.ExprType arg1ExprType, boolean isSingleRow) {
    String name = "TO_DAYS(" + arg1Info.getName() + ")";

    // In order to avoid needing to update this in the future, the subtraction and addition will be
    // handled by
    // generate binop, which will perform the correct null handling

    List<BodoSQLExprType.ExprType> exprTypes =
        Arrays.asList(arg1ExprType, BodoSQLExprType.ExprType.SCALAR);
    List<String> args =
        Arrays.asList(arg1Info.getExprCode(), "pd.Timestamp(year=1970, month=1, day=1)");
    String TimeDeltaFromUnixEpoch = generateBinOpCode(args, exprTypes, subBinop, isSingleRow);

    String daysSinceUnixEpoch;
    if (arg1ExprType == BodoSQLExprType.ExprType.SCALAR || isSingleRow) {
      daysSinceUnixEpoch =
          "bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days("
              + TimeDeltaFromUnixEpoch
              + ")";
    } else {
      daysSinceUnixEpoch = TimeDeltaFromUnixEpoch + ".dt.days";
    }

    List<String> args2 = Arrays.asList(daysSinceUnixEpoch, dayDeltaUnixY0);
    String outputExpr = generateBinOpCode(args2, exprTypes, addBinop, isSingleRow);

    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's TO_SECONDS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param arg1ExprType the expression type of the first argument.
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateToSecondsCode(
      RexNodeVisitorInfo arg1Info, BodoSQLExprType.ExprType arg1ExprType, boolean isSingleRow) {
    String name = "TO_SECONDS(" + arg1Info.getName() + ")";

    // In order to avoid needing to update this in the future, the subtraction and addition will be
    // handled by
    // generate binop, which will perform the correct null handling

    List<BodoSQLExprType.ExprType> exprTypes =
        Arrays.asList(arg1ExprType, BodoSQLExprType.ExprType.SCALAR);
    List<String> args =
        Arrays.asList(arg1Info.getExprCode(), "pd.Timestamp(year=1970, month=1, day=1)");
    String TimeDeltaFromUnixEpoch = generateBinOpCode(args, exprTypes, subBinop, isSingleRow);

    String secondsSinceUnixEpoch;
    if (arg1ExprType == BodoSQLExprType.ExprType.SCALAR || isSingleRow) {
      // TODO: null checking this
      secondsSinceUnixEpoch =
          "bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds("
              + TimeDeltaFromUnixEpoch
              + ")";
    } else {
      secondsSinceUnixEpoch = TimeDeltaFromUnixEpoch + ".dt.total_seconds()";
    }

    List<String> args2 = Arrays.asList(secondsSinceUnixEpoch, secondDeltaUnixY0);
    String outputExpr = generateBinOpCode(args2, exprTypes, addBinop, isSingleRow);

    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_DAYS
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateFromDaysCode(
      RexNodeVisitorInfo arg1Info, BodoSQLExprType.ExprType arg1ExprType, boolean isSingleRow) {
    String name = "FROM_DAYS(" + arg1Info.getName() + ")";
    String outputExpression;

    // In order to avoid needing to update several locations in the future, I'm going to do the
    // addition by going
    // through generate binop, which will perform the correct null handling
    List<BodoSQLExprType.ExprType> exprTypes =
        Arrays.asList(arg1ExprType, BodoSQLExprType.ExprType.SCALAR);
    boolean useScalarCode = isSingleRow || arg1ExprType == BodoSQLExprType.ExprType.SCALAR;

    List<String> args = Arrays.asList(arg1Info.getExprCode(), dayDeltaUnixY0);
    if (useScalarCode) {
      outputExpression =
          "bodo.utils.conversion.box_if_dt64(bodo.libs.bodosql_array_kernels.day_timestamp("
              + generateBinOpCode(args, exprTypes, subBinop, false)
              + "))";
    } else {
      outputExpression =
          "bodo.libs.bodosql_array_kernels.day_timestamp("
              + generateBinOpCode(args, exprTypes, subBinop, false)
              + ")";
    }
    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles the codegen for mySQL's FROM_UNIXTIME
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param isScalar should the input value be treated as a scalar
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateFromUnixTimeCode(
      RexNodeVisitorInfo arg1Info, boolean isScalar) {
    String name = "FROM_UNIXTIME(" + arg1Info.getName() + ")";
    String outputExpression;
    if (isScalar) {
      outputExpression =
          "bodo.utils.conversion.box_if_dt64(bodo.libs.bodosql_array_kernels.second_timestamp("
              + arg1Info.getExprCode()
              + "))";
    } else {
      outputExpression =
          "bodo.libs.bodosql_array_kernels.second_timestamp(" + arg1Info.getExprCode() + ")";
    }

    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles the codegen for mySQL's UNIX_TIMESTAMP() .
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateUnixTimestamp() {
    String name = "UNIX_TIMESTAMP()";
    String output =
        "(pd.Timestamp.now() - pd.Timestamp(year=1970, month=1, day=1)).total_seconds()";
    return new RexNodeVisitorInfo(name, output);
  }
}
