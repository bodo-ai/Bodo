package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateCastCode;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.List;
import org.apache.calcite.sql.type.SqlTypeName;

/**
 * Class that returns the generated code for a DateAdd expression after all inputs have been
 * visited.
 */
public class DateAddCodeGen {

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
        throw new BodoSQLCodegenException("DATEADD unit quarters not supported yet");

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
   * Function that return the necessary generated code for a DateAdd Function Call.
   *
   * @param arg0 The first arg expr.
   * @param arg1 The second arg expr.
   * @param generateScalarCode Should scalar code be generated
   * @param strNeedsCast Is arg0 a string that needs casting.
   * @return The code generated that matches the DateAdd expression.
   */
  public static String generateDateAddCode(
      String arg0, String arg1, boolean generateScalarCode, boolean strNeedsCast) {
    // Note: Null handling is supported by Bodo/Pandas behavior
    // TODO: Only in the case that timestamp NULLS == NaN
    StringBuilder addBuilder = new StringBuilder();
    if (strNeedsCast) {
      arg0 = generateCastCode(arg0, SqlTypeName.TIMESTAMP, generateScalarCode);
    }
    if (generateScalarCode) {
      addBuilder
          .append("bodosql.libs.generated_lib.sql_null_checking_addition(")
          .append(arg0)
          .append(", ")
          .append(arg1)
          .append(")");
    } else {
      addBuilder.append("(pd.Series(").append(arg0).append(") + ").append(arg1).append(").values");
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
