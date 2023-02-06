package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.DateTimeHelpers.*;
import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.*;
import org.apache.calcite.sql.type.*;

public class DatetimeFnCodeGen {
  static List<String> fnList =
      Arrays.asList(
          "DAYNAME",
          "LAST_DAY",
          "MONTHNAME",
          "MONTH_NAME",
          "NEXT_DAY",
          "PREVIOUS_DAY",
          "WEEKDAY",
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
   * @param arg1Name The name of arg1
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgDatetimeFnInfo(
      String fnName, String arg1Expr, String arg1Name) {
    StringBuilder name = new StringBuilder();
    name.append(fnName).append("(").append(arg1Name).append(")");
    StringBuilder expr_code = new StringBuilder();

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMap.containsKey(fnName)) {
      expr_code.append(equivalentFnMap.get(fnName)).append("(").append(arg1Expr).append(")");
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
   * @param arg2Expr The string expression of arg2
   * @param arg2Name The name of arg2
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getDoubleArgDatetimeFnInfo(
      String fnName, String arg1Expr, String arg1Name, String arg2Expr, String arg2Name) {
    StringBuilder name = new StringBuilder();
    name.append(fnName).append("(").append(arg1Name).append(", ").append(arg2Name).append(")");
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
      return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for makedate
   *
   * @param arg1Info The VisitorInfo for the first argument
   * @param arg2Info The VisitorInfo for the second argument
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateMakeDateInfo(
      RexNodeVisitorInfo arg1Info, RexNodeVisitorInfo arg2Info) {
    String name = "MAKEDATE(" + arg1Info.getName() + ", " + arg2Info.getName() + ")";

    String outputExpr =
        "bodo.libs.bodosql_array_kernels.makedate("
            + arg1Info.getExprCode()
            + ", "
            + arg2Info.getExprCode()
            + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  /**
   * Generate code for computing a timestamp for the current time in the default timezone.
   *
   * @param opName The name of the function. Several functions map to the same operation.
   * @param tzInfo The Timezone information with which to create the Timestamp.
   * @return
   */
  public static RexNodeVisitorInfo generateCurrTimestampCode(String opName, BodoTZInfo tzInfo) {
    String fnName = opName + "()";
    String fnExpression = String.format("pd.Timestamp.now(%s)", tzInfo.getPyZone());
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
   * @param arg2Info The VisitorInfo for the second argument. = * @return the rexNodeVisitorInfo for
   *     the result.
   */
  public static RexNodeVisitorInfo generateDateTruncCode(
      RexNodeVisitorInfo arg1Info, RexNodeVisitorInfo arg2Info) {
    String name = "DATE_TRUNC(" + arg1Info.getName() + ", " + arg2Info.getName() + ")";
    String codeGen =
        String.format(
            "bodo.libs.bodosql_array_kernels.date_trunc(%s, %s)",
            arg1Info.getExprCode(), arg2Info.getExprCode());
    return new RexNodeVisitorInfo(name, codeGen);
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
          "bodo.hiframes.pd_series_ext.get_series_data(pd.Series("
              + arg1Info.getExprCode()
              + ").dt.strftime("
              + pythonFormatString
              + "))";
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

  /**
   * Helper function that handles codegen for YearWeek
   *
   * @param arg0Info The name and codegen for the argument.
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getYearWeekFnInfo(RexNodeVisitorInfo arg0Info) {
    String arg0Expr = arg0Info.getExprCode();

    // performs yearNum * 100 + week num
    // TODO: Add proper null checking on scalars by converting * and +
    // to an array kernel
    String outputExpr =
        String.format(
            "bodo.libs.bodosql_array_kernels.add_numeric(bodo.libs.bodosql_array_kernels.multiply_numeric(bodo.libs.bodosql_array_kernels.get_year(%s),"
                + " 100), bodo.libs.bodosql_array_kernels.get_weekofyear(%s))",
            arg0Expr, arg0Expr);

    String name = "YEARWEEK(" + arg0Info.getName() + ")";
    return new RexNodeVisitorInfo(name, outputExpr);
  }

  public static String intExprToIntervalDays(String expr) {
    return "bodo.libs.bodosql_array_kernels.int_to_days(" + expr + ")";
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
   * Helper function that handles the codegen for DATE_FROM_PARTS, TIME_FROM_PARTS,
   * TIMESTAMP_FROM_PARTS and all of their variants/aliases
   *
   * @return the rexNodeVisitorInfo for the function call
   */
  public static RexNodeVisitorInfo generateDateTimeTypeFromPartsCode(
      String fnName, List<RexNodeVisitorInfo> operandsInfo, String tzStr) {
    StringBuilder name = new StringBuilder();
    StringBuilder code = new StringBuilder();

    boolean time_mode = false;
    boolean date_mode = false;
    boolean timestamp_mode = false;

    int numArgs = operandsInfo.size();

    switch (fnName) {
      case "TIME_FROM_PARTS":
        time_mode = true;
        break;
      case "DATE_FROM_PARTS":
      case "DATEFROMPARTS":
        date_mode = true;
        break;
      default:
        timestamp_mode = true;
    }

    name.append(fnName).append("(");
    code.append("bodo.libs.bodosql_array_kernels.");

    if (time_mode) {
      code.append("time_from_parts");
    } else if (date_mode || timestamp_mode) {
      code.append("construct_timestamp");
    }

    code.append("(");

    for (int i = 0; i < numArgs; i++) {
      if (i != 0) {
        name.append(", ");
        code.append(", ");
      }
      name.append(operandsInfo.get(i).getName());
      code.append(operandsInfo.get(i).getExprCode());
    }

    // For time, add the nanosecond argument if necessary
    if (time_mode && numArgs == 3) {
      name.append(", 0");
      code.append(", 0");
    }
    // For date, fill in all the arguments only used for timestamp
    if (date_mode) {
      code.append(", 0, 0, 0, 0, None");
    }
    // For timestamp, fill in the nanosecond argument if necessary
    if (timestamp_mode && numArgs < 7) {
      name.append(", 0");
      code.append(", 0");
    }
    // For timestamp, fill in the time_zone argument if necessary
    if (timestamp_mode && numArgs < 8) {
      name.append(", ").append(tzStr);
      code.append(", ").append(tzStr);
    }

    name.append(")");
    code.append(")");

    return new RexNodeVisitorInfo(name.toString(), code.toString());
  }
}
