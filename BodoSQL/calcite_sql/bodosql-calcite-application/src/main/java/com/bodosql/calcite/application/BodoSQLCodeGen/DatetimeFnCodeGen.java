package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.DateTimeHelpers.*;
import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.application.Utils.BodoCtx;
import java.util.*;
import org.apache.calcite.sql.type.SqlTypeName;

public class DatetimeFnCodeGen {
  static List<String> fnList =
      Arrays.asList("DAYNAME", "LAST_DAY", "MONTHNAME", "PREVIOUS_DAY", "WEEKDAY", "YEAROFWEEKISO");

  // HashMap of all trig functions which maps to array kernels
  // which handle all combinations of scalars/arrays/nulls.
  static HashMap<String, String> equivalentFnMap = new HashMap<>();

  static {
    for (String fn : fnList) {
      equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels." + fn.toLowerCase());
    }
  }

  /**
   * Helper function that handles codegen for Single argument datetime functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgDatetimeFnInfo(
      String fnName, String inputVar, String arg1Expr, String arg1Name, boolean isSingleRow) {
    StringBuilder name = new StringBuilder();
    name.append(fnName).append("(").append(arg1Name).append(")");
    StringBuilder expr_code = new StringBuilder();

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMap.containsKey(fnName)) {
      if (isSingleRow) {
        expr_code
            .append("bodo.utils.conversion.box_if_dt64(")
            .append(equivalentFnMap.get(fnName))
            .append("(")
            .append("bodo.utils.conversion.unbox_if_timestamp(")
            .append(arg1Expr)
            .append(")))");
      } else {
        expr_code.append(equivalentFnMap.get(fnName)).append("(").append(arg1Expr).append(")");
      }
      return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for Single argument datetime functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getDoubleArgDatetimeFnInfo(
      String fnName,
      String inputVar,
      String arg1Expr,
      String arg1Name,
      String arg2Expr,
      String arg2Name,
      boolean isSingleRow) {
    StringBuilder name = new StringBuilder();
    name.append(fnName).append("(").append(arg1Name).append(")");
    StringBuilder expr_code = new StringBuilder();

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMap.containsKey(fnName)) {
      if (isSingleRow) {
        expr_code
            .append("bodo.utils.conversion.box_if_dt64(")
            .append(equivalentFnMap.get(fnName))
            .append("(")
            .append("bodo.utils.conversion.unbox_if_timestamp(")
            .append(arg1Expr)
            .append("),bodo.utils.conversion.unbox_if_timestamp(")
            .append(arg2Expr)
            .append(")))");
      } else {
        expr_code
            .append(equivalentFnMap.get(fnName))
            .append("(")
            .append(arg1Expr)
            .append(",")
            .append(arg2Expr)
            .append(")");
      }
      return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for makedate
   *
   * @param inputVar Name of dataframe which Columns expressions reference
   * @param arg1Info The VisitorInfo for the first argument
   * @param arg2Info The VisitorInfo for the second argument
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateMakeDateInfo(
      String inputVar,
      RexNodeVisitorInfo arg1Info,
      RexNodeVisitorInfo arg2Info,
      boolean isSingleRow,
      BodoCtx ctx) {
    String name = "MAKEDATE(" + arg1Info.getName() + ", " + arg2Info.getName() + ")";

    String outputExpr =
        "bodo.libs.bodosql_array_kernels.makedate("
            + arg1Info.getExprCode()
            + ", "
            + arg2Info.getExprCode()
            + ")";
    if (isSingleRow) {
      outputExpr = "bodo.utils.conversion.box_if_dt64(" + outputExpr + ")";
    }
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  public static RexNodeVisitorInfo generateCurtimeCode(String opName) {
    String fnName = opName + "()";
    String fnExpression = "pd.Timestamp.now()";
    return new RexNodeVisitorInfo(fnName, fnExpression);
  }

  public static RexNodeVisitorInfo generateUTCTimestampCode() {
    String fnName = "UTC_TIMESTAMP()";
    // use utcnow if/when we decide to support timezones
    String fnExpression = "pd.Timestamp.now()";
    return new RexNodeVisitorInfo(fnName, fnExpression);
  }

  public static RexNodeVisitorInfo generateUTCDateCode() {
    String fnName = "UTC_Date()";
    // use utcnow if/when we decide to support timezones
    String fnExpression = "pd.Timestamp.now().floor(freq=\"D\")";
    return new RexNodeVisitorInfo(fnName, fnExpression);
  }

  /**
   * Helper function handles the codegen for Date_Trunc
   *
   * @param arg1Info The VisitorInfo for the first argument. Currently, this is required to be a *
   *     constant string literal.
   * @param arg2Info The VisitorInfo for the second argument.
   * @param arg2ExprType Is arg2 a column or scalar?
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return the rexNodeVisitorInfo for the result.
   */
  public static RexNodeVisitorInfo generateDateTruncCode(
      RexNodeVisitorInfo arg1Info,
      RexNodeVisitorInfo arg2Info,
      BodoSQLExprType.ExprType arg2ExprType,
      boolean isSingleRow) {
    if (!isStringLiteral(arg1Info.getExprCode())) {
      throw new BodoSQLCodegenException(
          "DATE_TRUNC(): Argument 0 must be a constant literal String");
    }

    String name = "DATE_TRUNC(" + arg1Info.getName() + ", " + arg2Info.getName() + ")";
    // Extract the literal and ensure its not case sensitive
    String truncVal = getStringLiteralValue(arg1Info.getExprCode()).toUpperCase();
    // Valid DATE_TRUNC values:
    // https://docs.snowflake.com/en/sql-reference/functions-date-time.html#supported-date-and-time-parts
    String outputExpression;

    if (arg2ExprType == BodoSQLExprType.ExprType.SCALAR || isSingleRow) {
      // TODO [BS-638]: Support Scalar Values
      throw new BodoSQLCodegenException("DATE_TRUNC(): Not supported on scalar values.");
    } else {
      switch (truncVal) {
          // For offset values we need to use DateOffset.
        case "YEAR":
          // TODO [BE-2304]: Support YearBegin in the Engine
          throw new BodoSQLCodegenException(
              "DATE_TRUNC(): Specifying 'YEAR' for <date_or_time_part> not supported.");
        case "MONTH":
          // Month rounds down to the start of the Month.
          // We add 1 Day to avoid boundaries
          outputExpression =
              "((pd.Series("
                  + arg2Info.getExprCode()
                  + ") + pd.Timedelta(days=1)) - pd.tseries.offsets.MonthBegin(n=1,"
                  + " normalize=True)).values";
          break;
        case "WEEK":
          // Week rounds down to the Monday of that week.
          // We add 1 Day to avoid boundaries
          outputExpression =
              "((pd.Series("
                  + arg2Info.getExprCode()
                  + ") + pd.Timedelta(days=1)) - pd.tseries.offsets.Week(n=1, weekday=0,"
                  + " normalize=True)).values";
          break;
        case "QUARTER":
          // TODO [BE-2305]: Support QuarterBegin in the Engine
          throw new BodoSQLCodegenException(
              "DATE_TRUNC(): Specifying 'Quarter' for <date_or_time_part> not supported.");
        case "DAY":
          // For all timedelta valid values we can use .dt.floor
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"D\").values";
          break;
        case "HOUR":
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"H\").values";
          break;
        case "MINUTE":
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"min\").values";
          break;
        case "SECOND":
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"S\").values";
          break;
        case "MILLISECOND":
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"ms\").values";
          break;
        case "MICROSECOND":
          outputExpression = "pd.Series(" + arg2Info.getExprCode() + ").dt.floor(\"us\").values";
          break;
        case "NANOSECOND":
          // Timestamps have nanosecond precision so we don't need to round.
          outputExpression = arg2Info.getExprCode();
          break;
        default:
          throw new BodoSQLCodegenException(
              String.format("DATE_TRUNC(): Invalid <date_or_time_part> '%s'", truncVal));
      }
    }

    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles the codegen for Date format
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param arg1ExprType Is arg1 a column or scalar?
   * @param arg2Info The VisitorInfo for the second argument. Currently, this is required to be a
   *     constant string literal.
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply
   * @return the rexNodeVisitorInfo for the result.
   */
  public static RexNodeVisitorInfo generateDateFormatCode(
      RexNodeVisitorInfo arg1Info,
      BodoSQLExprType.ExprType arg1ExprType,
      RexNodeVisitorInfo arg2Info,
      boolean isSingleRow) {
    String name = "DATE_FORMAT(" + arg1Info.getName() + ", " + arg2Info.getName() + ")";

    assert isStringLiteral(arg2Info.getExprCode());
    String pythonFormatString = convertMySQLFormatStringToPython(arg2Info.getExprCode());
    String outputExpression;

    if (arg1ExprType == BodoSQLExprType.ExprType.SCALAR || isSingleRow) {
      outputExpression =
          "bodosql.libs.generated_lib.sql_null_checking_strftime("
              + arg1Info.getExprCode()
              + ", "
              + pythonFormatString
              + ")";
    } else {
      outputExpression =
          "pd.Series("
              + arg1Info.getExprCode()
              + ").dt.strftime("
              + pythonFormatString
              + ").values";
    }

    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles codegen for CURDATE and CURRENTDATE
   *
   * @param opName The name of the function
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateCurdateCode(String opName) {
    String fnName = opName + "()";
    String fnExpression = "pd.Timestamp.now().floor(freq=\"D\")";
    return new RexNodeVisitorInfo(fnName, fnExpression);
  }

  public static RexNodeVisitorInfo getYearWeekFnInfo(
      RexNodeVisitorInfo arg0Info, boolean isScalar) {

    String outputExpr;
    String arg0Expr = arg0Info.getExprCode();

    // performs yearNum * 100 + week num
    if (isScalar) {
      // TODO: Null check this
      outputExpr =
          "(bodosql.libs.generated_lib.sql_null_checking_year("
              + arg0Expr
              + ") * 100 + bodosql.libs.generated_lib.sql_null_checking_weekofyear("
              + arg0Expr
              + "))";
    } else {
      outputExpr =
          "(pd.Series("
              + arg0Expr
              + ").dt.year * 100 + pd.Series("
              + arg0Expr
              + ").dt.isocalendar().week).values";
    }

    String name = "YEARWEEK(" + arg0Info.getName() + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  public static String intExprToIntervalDays(String expr, boolean useScalar) {
    String arg1Expr;
    if (useScalar) {
      arg1Expr =
          "bodo.utils.conversion.box_if_dt64(bodo.libs.bodosql_array_kernels.int_to_days("
              + expr
              + "))";
    } else {
      arg1Expr = "bodo.libs.bodosql_array_kernels.int_to_days(" + expr + ")";
    }
    return arg1Expr;
  }

  /**
   * Helper function that handles the codegen for snowflake SQL's TIME and TO_TIME
   *
   * @param arg1Type The type of the first argument.
   * @param arg1Info The VisitorInfo for the first argument.
   * @param opName should be either "TIME" or "TO_TIME"
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateToTimeCode(
      SqlTypeName arg1Type, RexNodeVisitorInfo arg1Info, String opName) {
    String name = opName + "(" + arg1Info.getName() + ")";
    String outputExpression =
        "bodo.libs.bodosql_array_kernels."
            + opName.toLowerCase()
            + "_util("
            + arg1Info.getExprCode()
            + ")";
    return new RexNodeVisitorInfo(name, outputExpression);
  }

  /**
   * Helper function that handles the codegen for snowflake SQL's TIME_FROM_PARTS
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateTimeFromPartsCode(
      RexNodeVisitorInfo arg1Info,
      RexNodeVisitorInfo arg2Info,
      RexNodeVisitorInfo arg3Info,
      RexNodeVisitorInfo arg4Info) {
    String name;
    String outputExpression;

    if (arg4Info == null) {
      name =
          "TIME_FROM_PARTS("
              + arg1Info.getName()
              + ", "
              + arg2Info.getName()
              + ", "
              + arg3Info.getName()
              + ")";
      outputExpression =
          "bodo.libs.bodosql_array_kernels.time_from_parts_util("
              + arg1Info.getExprCode()
              + ", "
              + arg2Info.getExprCode()
              + ", "
              + arg3Info.getExprCode()
              + ", 0)";
    } else {
      name =
          "TIME_FROM_PARTS("
              + arg1Info.getName()
              + ", "
              + arg2Info.getName()
              + ", "
              + arg3Info.getName()
              + ", "
              + arg4Info.getName()
              + ")";
      outputExpression =
          "bodo.libs.bodosql_array_kernels.time_from_parts_util("
              + arg1Info.getExprCode()
              + ", "
              + arg2Info.getExprCode()
              + ", "
              + arg3Info.getExprCode()
              + ", "
              + arg4Info.getExprCode()
              + ")";
    }

    return new RexNodeVisitorInfo(name, outputExpression);
  }
}
