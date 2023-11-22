package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.DateTimeHelpers.convertMySQLFormatStringToPython;
import static com.bodosql.calcite.application.utils.DateTimeHelpers.isStringLiteral;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import kotlin.Pair;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.type.BodoTZInfo;

public class DatetimeFnCodeGen {
  static List<String> fnList =
      Arrays.asList(
          "DAYNAME",
          "MONTHNAME",
          "MONTH_NAME",
          "NEXT_DAY",
          "PREVIOUS_DAY",
          "WEEKDAY",
          "TIME_SLICE",
          "YEAROFWEEK",
          "YEAROFWEEKISO");

  // HashMap of all datetime functions which maps to array kernels
  // which handle all combinations of scalars/arrays/nulls.
  static HashMap<String, String> equivalentFnMap = new HashMap<>();

  static {
    for (String fn : fnList) {
      if (fn.equals("YEAROFWEEK")) {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels.get_year");
      } else if (fn.equals("MONTH_NAME")) {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels.monthname");
      } else {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels." + fn.toLowerCase());
      }
    }
  }

  /**
   * Helper function that handles codegen for Single argument datetime functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @return The Expr corresponding to the function call
   */
  public static Expr getSingleArgDatetimeFnInfo(String fnName, String arg1Expr) {
    StringBuilder expr_code = new StringBuilder();

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMap.containsKey(fnName)) {
      expr_code.append(equivalentFnMap.get(fnName)).append("(").append(arg1Expr).append(")");
      return new Expr.Raw(expr_code.toString());
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for Single argument datetime functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg2Expr The string expression of arg2
   * @return The Expr corresponding to the function call
   */
  public static Expr getDoubleArgDatetimeFnInfo(String fnName, String arg1Expr, String arg2Expr) {
    StringBuilder expr_code = new StringBuilder();

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMap.containsKey(fnName)) {
      expr_code
          .append(equivalentFnMap.get(fnName))
          .append("(")
          .append(arg1Expr)
          .append(",")
          .append(arg2Expr)
          .append(")");
      return new Expr.Raw(expr_code.toString());
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for makedate
   *
   * @param arg1Info The VisitorInfo for the first argument
   * @param arg2Info The VisitorInfo for the second argument
   * @return The Expr corresponding to the function call
   */
  public static Expr generateMakeDateInfo(Expr arg1Info, Expr arg2Info) {
    String outputExpr =
        "bodo.libs.bodosql_array_kernels.makedate("
            + arg1Info.emit()
            + ", "
            + arg2Info.emit()
            + ")";
    return new Expr.Raw(outputExpr);
  }

  /**
   * Generate code for computing a timestamp for the current time in the default timezone.
   *
   * @param tzInfo The Timezone information with which to create the Timestamp.
   * @param makeConsistent Should a function call be generated that ensures the data is consistent
   *     on all ranks. This is False if used within a case statement.
   * @return The generated code.
   */
  public static Expr generateCurrTimestampCode(BodoTZInfo tzInfo, boolean makeConsistent) {
    String fnName;
    if (makeConsistent) {
      fnName = "bodo.hiframes.pd_timestamp_ext.now_impl_consistent";
    } else {
      fnName = "pd.Timestamp.now";
    }
    return new Expr.Call(fnName, tzInfo.getZoneExpr());
  }

  /**
   * Generate code for computing a time value for the current time in the default timezone.
   *
   * @param tzInfo The Timezone information with which to create the Time.
   * @param makeConsistent Should a function call be generated that ensures the data is consistent
   *     on all ranks. This is False if used within a case statement.
   * @return The generated code.
   */
  public static Expr generateCurrTimeCode(BodoTZInfo tzInfo, boolean makeConsistent) {
    Expr tzArg = tzInfo == null ? Expr.None.INSTANCE : tzInfo.getZoneExpr();
    String fnName;
    if (makeConsistent) {
      fnName = "bodo.hiframes.pd_timestamp_ext.now_impl_consistent";
    } else {
      fnName = "pd.Timestamp.now";
    }
    Expr nowCall = new Expr.Call(fnName, tzArg);
    List<Pair<String, Expr>> namedArgs =
        List.of(
            new Pair<>("format_str", Expr.None.INSTANCE),
            new Pair<>("_try", Expr.BooleanLiteral.True.INSTANCE));
    return new Expr.Call("bodo.libs.bodosql_array_kernels.to_time", List.of(nowCall), namedArgs);
  }

  /**
   * Generate the code for the Timestamp in UTC.
   *
   * @param makeConsistent Should a function call be generated that ensures the data is consistent
   *     on all ranks. This is False if used within a case statement.
   * @return The generated code.
   */
  public static Expr generateUTCTimestampCode(boolean makeConsistent) {
    String fnName;
    if (makeConsistent) {
      fnName = "bodo.hiframes.pd_timestamp_ext.now_impl_consistent";
    } else {
      fnName = "pd.Timestamp.now";
    }
    return new Expr.Call(fnName);
  }

  /**
   * Generate the code for the current date in UTC.
   *
   * @param makeConsistent Should a function call be generated that ensures the data is consistent
   *     on all ranks. This is False if used within a case statement.
   * @return The generated code.
   */
  public static Expr generateUTCDateCode(boolean makeConsistent) {
    String fnName;
    if (makeConsistent) {
      fnName = "bodo.hiframes.datetime_date_ext.today_rank_consistent";
    } else {
      fnName = "datetime.date.today";
    }
    return new Expr.Call(fnName);
  }

  /**
   * Helper function handles the codegen for Date_Trunc
   *
   * @param unit A constant string literal, the time unit to truncate.
   * @param arg2Info The VisitorInfo for the second argument. = * @return the rexNodeVisitorInfo for
   *     the result.
   */
  public static Expr generateDateTruncCode(String unit, Expr arg2Info) {
    String codeGen =
        String.format(
            "bodo.libs.bodosql_array_kernels.date_trunc(\"%s\", %s)", unit, arg2Info.emit());
    return new Expr.Raw(codeGen);
  }

  /**
   * Helper function that handles the codegen for Date format
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @param arg2Info The VisitorInfo for the second argument. Currently, this is required to be a
   *     constant string literal.
   * @return the rexNodeVisitorInfo for the result.
   */
  public static Expr generateDateFormatCode(Expr arg1Info, Expr arg2Info) {
    assert isStringLiteral(arg2Info.emit());
    String pythonFormatString = convertMySQLFormatStringToPython(arg2Info.emit());
    String outputExpression =
        "bodo.libs.bodosql_array_kernels.date_format("
            + arg1Info.emit()
            + ", "
            + pythonFormatString
            + ")";

    return new Expr.Raw(outputExpression);
  }

  /**
   * Helper function that handles codegen for CONVERT_TIMEZONE
   *
   * @return The Expr corresponding to the function call
   */
  public static Expr generateConvertTimezoneCode(List<Expr> operands, BodoTZInfo tzTimeInfo) {
    Expr defaultTz = tzTimeInfo.getZoneExpr();
    List<Expr> args = new ArrayList<>();
    if (operands.size() == 2) {
      args.add(defaultTz);
      args.addAll(operands);
      args.add(new Expr.BooleanLiteral(true));
    } else if (operands.size() == 3) {
      args.addAll(operands);
      args.add(new Expr.BooleanLiteral(false));
    }
    return ExprKt.BodoSQLKernel("convert_timezone", args, List.of());
  }

  /**
   * Helper function that handles codegen for CURDATE and CURRENTDATE. CURRENT_DATE actually returns
   * the date from the Timestamp in the system's local timezone.
   * https://docs.snowflake.com/en/sql-reference/functions/current_date
   *
   * <p>As a result, if the timezone isn't in UTC we need to pass that information.
   *
   * @param defaultTZInfo The timezone to use for the date information.
   * @param makeConsistent Should a function call be generated that ensures the data is consistent
   *     on all ranks. This is False if used within a case statement.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateCurrentDateCode(BodoTZInfo defaultTZInfo, boolean makeConsistent) {
    String fnName;
    if (makeConsistent) {
      fnName = "bodo.hiframes.datetime_date_ext.now_date_wrapper_consistent";
    } else {
      fnName = "bodo.hiframes.datetime_date_ext.now_date_wrapper";
    }
    return new Expr.Call(fnName, defaultTZInfo.getZoneExpr());
  }

  /**
   * Helper function that handles codegen for YearWeek
   *
   * @param arg0Info The name and codegen for the argument.
   * @return The Expr corresponding to the function call
   */
  public static Expr getYearWeekFnInfo(Expr arg0Info) {
    String arg0Expr = arg0Info.emit();

    // performs yearNum * 100 + week num
    // TODO: Add proper null checking on scalars by converting * and +
    // to an array kernel
    String outputExpr =
        String.format(
            "bodo.libs.bodosql_array_kernels.add_numeric(bodo.libs.bodosql_array_kernels.multiply_numeric(bodo.libs.bodosql_array_kernels.get_year(%s),"
                + " 100), bodo.libs.bodosql_array_kernels.get_weekofyear(%s))",
            arg0Expr, arg0Expr);
    return new Expr.Raw(outputExpr);
  }

  /**
   * Helper function that handles the codegen for snowflake SQL's TIME, TO_TIME, and TRY_TO_TIME
   *
   * @param operands The function arguments.
   * @param opName should be either "TIME", "TO_TIME", or "TRY_TO_TIME"
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToTimeCode(
      List<Expr> operands, String opName, List<Pair<String, Expr>> streamingNamedArgs) {
    assert operands.size() == 1;
    List<Expr> args = new ArrayList<>();
    args.addAll(operands);
    if (args.size() == 1) {
      // Add the format string
      args.add(Expr.None.INSTANCE);
    }
    // Add the try arg.
    args.add(new Expr.BooleanLiteral(opName.contains("TRY")));
    return ExprKt.BodoSQLKernel("to_time", args, streamingNamedArgs);
  }

  /**
   * Helper function that handles the codegen for snowflake SQL's LAST_DAY
   *
   * @param arg0 A date or timestamp expression.
   * @param unit The time unit for calculating the last day.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateLastDayCode(String arg0, String unit) {
    String outputExpression = "bodo.libs.bodosql_array_kernels.last_day_" + unit + "(" + arg0 + ")";
    return new Expr.Raw(outputExpression);
  }

  public static Expr generateTimeSliceFnCode(List<Expr> operandsInfo, Integer weekStart) {
    assert (operandsInfo.size() == 3 || operandsInfo.size() == 4);

    List<Expr> args = new ArrayList<>(operandsInfo);
    if (operandsInfo.size() == 3) {
      args.add(new Expr.StringLiteral("START"));
    }
    args.add(new Expr.IntegerLiteral(weekStart));
    return ExprKt.BodoSQLKernel("time_slice", args, List.of());
  }

  /**
   * Helper function that handles the codegen for DATE_FROM_PARTS, TIME_FROM_PARTS,
   * TIMESTAMP_FROM_PARTS and all of their variants/aliases
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateDateTimeTypeFromPartsCode(
      final String fnName, List<Expr> operandsInfo, final Expr tzExpr) {
    boolean time_mode = false;
    boolean date_mode = false;
    boolean timestamp_mode = false;
    boolean timestamp_from_date_time_mode = false;

    int numArgs = operandsInfo.size();

    switch (fnName) {
      case "TIME_FROM_PARTS":
      case "TIMEFROMPARTS":
        time_mode = true;
        break;
      case "DATE_FROM_PARTS":
      case "DATEFROMPARTS":
        date_mode = true;
        break;
      case "TIMESTAMP_LTZ_FROM_PARTS":
      case "TIMESTAMPLTZFROMPARTS":
        // the LTZ versions cannot use the two argument constructor which always
        // produces NTZ values.
        timestamp_mode = true;
        break;
      default:
        timestamp_mode = numArgs >= 6;
        timestamp_from_date_time_mode = !timestamp_mode;
    }
    String generateFnName;
    if (time_mode) {
      generateFnName = "time_from_parts";
    } else if (date_mode) {
      generateFnName = "date_from_parts";
    } else if (timestamp_from_date_time_mode) {
      generateFnName = "timestamp_from_date_and_time";
    } else { // timestamp_mode
      generateFnName = "construct_timestamp";
    }

    // Copy the arguments because we will need to append.
    List<Expr> args = new ArrayList<>(operandsInfo);

    // For time, add the nanosecond argument if necessary
    if (time_mode && numArgs == 3) {
      args.add(Expr.Companion.getZero());
    }
    // For timestamp, fill in the nanosecond argument if necessary
    if (timestamp_mode && numArgs < 7) {
      args.add(Expr.Companion.getZero());
    }
    // For timestamp, fill in the time_zone argument if necessary
    if (timestamp_mode && numArgs < 8) {
      args.add(tzExpr);
    }

    return ExprKt.BodoSQLKernel(generateFnName, args, List.of());
  }

  public static ArrayList<String> TIME_PART_UNITS =
      new ArrayList<String>(
          Arrays.asList("hour", "minute", "second", "millisecond", "microsecond", "nanosecond"));

  public enum DateTimeType {
    TIMESTAMP,
    TIME,
    DATE,
  }

  /**
   * Helper function that verifies and determines the type of date or time expression
   *
   * @param rexNode RexNode of the expression
   * @return The expression is a timestamp, time or date object
   */
  public static DateTimeType getDateTimeDataType(RexNode rexNode) {
    if (rexNode.getType().getSqlTypeName().toString().equals("TIME")) {
      return DateTimeType.TIME;
    }
    if (rexNode.getType().getSqlTypeName().toString().equals("DATE")) {
      return DateTimeType.DATE;
    }
    return DateTimeType.TIMESTAMP;
  }

  /**
   * Helper function that verifies and standardizes the time unit input
   *
   * @param fnName the function which takes this time unit as input
   * @param inputTimeStr the input time unit string
   * @param dateTimeDataType if the time expression is Bodo.Time object, the time unit should be
   *     smaller or equal to hour if the time expression is date object, the time unit should be
   *     larger or equal to day
   * @return the standardized time unit string
   */
  public static String standardizeTimeUnit(
      String fnName, String inputTimeStr, DateTimeType dateTimeDataType) {
    String unit;
    switch (inputTimeStr.toLowerCase()) {
      case "\"year\"":
      case "\"y\"":
      case "\"yy\"":
      case "\"yyy\"":
      case "\"yyyy\"":
      case "\"yr\"":
      case "\"years\"":
      case "\"yrs\"":
      case "year":
      case "y":
      case "yy":
      case "yyy":
      case "yyyy":
      case "yr":
      case "years":
      case "yrs":
        if (dateTimeDataType == DateTimeType.TIME)
          throw new BodoSQLCodegenException(
              "Unsupported unit for " + fnName + " with TIME input: " + inputTimeStr);
        unit = "year";
        break;

      case "\"month\"":
      case "\"mm\"":
      case "\"mon\"":
      case "\"mons\"":
      case "\"months\"":
      case "month":
      case "mm":
      case "mon":
      case "mons":
      case "months":
        if (dateTimeDataType == DateTimeType.TIME)
          throw new BodoSQLCodegenException(
              "Unsupported unit for " + fnName + " with TIME input: " + inputTimeStr);
        unit = "month";
        break;

      case "\"day\"":
      case "\"d\"":
      case "\"dd\"":
      case "\"days\"":
      case "\"dayofmonth\"":
      case "day":
      case "d":
      case "dd":
      case "days":
      case "dayofmonth":
        if (dateTimeDataType == DateTimeType.TIME)
          throw new BodoSQLCodegenException(
              "Unsupported unit for " + fnName + " with TIME input: " + inputTimeStr);
        unit = "day";
        break;

      case "\"week\"":
      case "\"w\"":
      case "\"wk\"":
      case "\"weekofyear\"":
      case "\"woy\"":
      case "\"wy\"":
      case "week":
      case "w":
      case "wk":
      case "weekofyear":
      case "woy":
      case "wy":
        if (dateTimeDataType == DateTimeType.TIME)
          throw new BodoSQLCodegenException(
              "Unsupported unit for " + fnName + " with TIME input: " + inputTimeStr);
        unit = "week";
        break;

      case "\"quarter\"":
      case "\"q\"":
      case "\"qtr\"":
      case "\"qtrs\"":
      case "\"quarters\"":
      case "quarter":
      case "q":
      case "qtr":
      case "qtrs":
      case "quarters":
        if (dateTimeDataType == DateTimeType.TIME)
          throw new BodoSQLCodegenException(
              "Unsupported unit for " + fnName + " with TIME input: " + inputTimeStr);
        unit = "quarter";
        break;

      case "\"hour\"":
      case "\"h\"":
      case "\"hh\"":
      case "\"hr\"":
      case "\"hours\"":
      case "\"hrs\"":
      case "hour":
      case "h":
      case "hh":
      case "hr":
      case "hours":
      case "hrs":
        unit = "hour";
        break;

      case "\"minute\"":
      case "\"m\"":
      case "\"mi\"":
      case "\"min\"":
      case "\"minutes\"":
      case "\"mins\"":
      case "minute":
      case "m":
      case "mi":
      case "min":
      case "minutes":
      case "mins":
        unit = "minute";
        break;

      case "\"second\"":
      case "\"s\"":
      case "\"sec\"":
      case "\"seconds\"":
      case "\"secs\"":
      case "second":
      case "s":
      case "sec":
      case "seconds":
      case "secs":
        unit = "second";
        break;

      case "\"millisecond\"":
      case "\"ms\"":
      case "\"msec\"":
      case "\"milliseconds\"":
      case "millisecond":
      case "ms":
      case "msec":
      case "milliseconds":
        unit = "millisecond";
        break;

      case "\"microsecond\"":
      case "\"us\"":
      case "\"usec\"":
      case "\"microseconds\"":
      case "microsecond":
      case "us":
      case "usec":
      case "microseconds":
        unit = "microsecond";
        break;

      case "\"nanosecond\"":
      case "\"ns\"":
      case "\"nsec\"":
      case "\"nanosec\"":
      case "\"nsecond\"":
      case "\"nanoseconds\"":
      case "\"nanosecs\"":
      case "\"nseconds\"":
      case "nanosecond":
      case "ns":
      case "nsec":
      case "nanosec":
      case "nsecond":
      case "nanoseconds":
      case "nanosecs":
      case "nseconds":
        unit = "nanosecond";
        break;

      default:
        throw new BodoSQLCodegenException("Unsupported unit for " + fnName + ": " + inputTimeStr);
    }
    return unit;
  }
}
