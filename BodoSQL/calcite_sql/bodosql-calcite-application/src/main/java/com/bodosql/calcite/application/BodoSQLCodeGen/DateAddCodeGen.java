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
      List<RexNodeVisitorInfo> operandsInfo) {

    final String unit;
    final String unit_arg = operandsInfo.get(0).getName();
    StringBuilder name = new StringBuilder();
    StringBuilder code = new StringBuilder();

    name.append("DATEADD(")
        .append(unit_arg)
        .append(", ")
        .append(operandsInfo.get(1).getName())
        .append(", ")
        .append(operandsInfo.get(2).getName())
        .append(")");

    switch (unit_arg) {
      case "\"year\"":
      case "\"y\"":
      case "\"yy\"":
      case "\"yyy\"":
      case "\"yyyy\"":
      case "\"yr\"":
      case "\"years\"":
      case "\"yrs\"":
        unit = "years";
        break;

      case "\"quarter\"":
      case "\"q\"":
      case "\"qtr\"":
      case "\"qtrs\"":
      case "\"quarters\"":
        unit = "quarters";
        break;

      case "\"month\"":
      case "\"mm\"":
      case "\"mon\"":
      case "\"mons\"":
      case "\"months\"":
        unit = "months";
        break;

      case "\"week\"":
      case "\"w\"":
      case "\"wk\"":
      case "\"weekofyear\"":
      case "\"woy\"":
      case "\"wy\"":
        unit = "weeks";
        break;

      case "\"day\"":
      case "\"d\"":
      case "\"dd\"":
      case "\"days\"":
      case "\"dayofmonth\"":
        unit = "days";
        break;

      case "\"hour\"":
      case "\"h\"":
      case "\"hh\"":
      case "\"hr\"":
      case "\"hours\"":
      case "\"hrs\"":
        unit = "hours";
        break;

      case "\"minute\"":
      case "\"m\"":
      case "\"mi\"":
      case "\"min\"":
      case "\"minutes\"":
      case "\"mins\"":
        unit = "minutes";
        break;

      case "\"second\"":
      case "\"s\"":
      case "\"sec\"":
      case "\"seconds\"":
      case "\"secs\"":
        unit = "seconds";
        break;

      case "\"millisecond\"":
      case "\"ms\"":
      case "\"msec\"":
      case "\"milliseconds\"":
        unit = "milliseconds";
        break;

      case "\"microsecond\"":
      case "\"us\"":
      case "\"usec\"":
      case "\"microseconds\"":
        unit = "microseconds";
        break;

      case "\"nanosecond\"":
      case "\"ns\"":
      case "\"nsec\"":
      case "\"nanosec\"":
      case "\"nsecond\"":
      case "\"nanoseconds\"":
      case "\"nanonsecs\"":
      case "\"nsecs\"":
        unit = "nanoseconds";
        break;

      default:
        throw new BodoSQLCodegenException("Invalid DATEADD unit: " + unit_arg);
    }
    code.append("bodo.libs.bodosql_array_kernels.add_interval_")
        .append(unit)
        .append("(")
        .append(operandsInfo.get(1).getExprCode())
        .append(", ")
        .append(operandsInfo.get(2).getExprCode())
        .append(")");

    return new RexNodeVisitorInfo(name.toString(), code.toString());
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
   * @param manual_addition Is the second argument a timedelta?
   * @return The code generated that matches the DateAdd expression.
   */
  public static String generateMySQLDateAddCode(String arg0, String arg1, boolean manual_addition) {
    StringBuilder addBuilder = new StringBuilder();
    if (manual_addition) {
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
