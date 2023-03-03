package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.List;

/**
 * Class that returns the generated code for a DateAdd expression after all inputs have been
 * visited.
 */
public class DateAddCodeGen {

  /**
   * Function that return the necessary generated code for a Snowflake DATEADD function call, which
   * adds an integer amount to a datetime of a certain unit.
   *
   * @param operandsInfo the list of arguments (UNIT, AMOUNT, START_DATETIME)
   * @return The code generated that matches the DATEADD expression.
   */
  public static RexNodeVisitorInfo generateSnowflakeDateAddCode(
      List<RexNodeVisitorInfo> operandsInfo, String unit) {
    // input check for time unit is moved to standardizeTimeUnit() function,
    // which is called in PandasCodeGenVisitor.java
    StringBuilder code = new StringBuilder();
    code.append("bodo.libs.bodosql_array_kernels.add_interval_")
        .append(unit)
        .append("s(")
        .append(operandsInfo.get(1).getExprCode())
        .append(", ")
        .append(operandsInfo.get(2).getExprCode())
        .append(")");

    return new RexNodeVisitorInfo(code.toString());
  }

  /**
   * Function that return the necessary generated code for a MySQL DATEADD function call, which
   * differs from Snowflake DATEADD as follows:
   *
   * <p>Both of the following add 42 days to column A: MySQL: DATEADD(A, 42) Snowflake:
   * DATEADD('day', 42, A)
   *
   * @param arg0 The first starting datetime (or string).
   * @param arg1 The amount of days to add to the starting datetime.
   * @param adding_delta Is the second argument a timedelta?
   * @param fnName The name of the function
   * @return The code generated that matches the DateAdd expression.
   */
  public static String generateMySQLDateAddCode(
      String arg0, String arg1, boolean adding_delta, String fnName) {
    StringBuilder addBuilder = new StringBuilder();
    if (fnName.equals("SUBDATE") || fnName.equals("DATE_SUB")) {
      arg1 = "bodo.libs.bodosql_array_kernels.negate(" + arg1 + ")";
    }
    if (adding_delta) {
      addBuilder
          .append("bodo.libs.bodosql_array_kernels.add_interval(")
          .append(arg0)
          .append(", ")
          .append(arg1)
          .append(")");
    } else {
      addBuilder
          .append("bodo.libs.bodosql_array_kernels.add_interval_days(")
          .append(arg1)
          .append(", ")
          .append(arg0)
          .append(")");
    }

    return addBuilder.toString();
  }

  /**
   * Function that returns the generated name for a DateAdd Function Call.
   *
   * @param arg0Name The first arg's name.
   * @param arg1Name The second arg's name.
   * @return The name generated that matches the DateAdd expression.
   */
  public static String generateDateAddName(String arg0Name, String arg1Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("DATE_ADD(").append(arg0Name).append(", ").append(arg1Name).append(")");
    return nameBuilder.toString();
  }
}
