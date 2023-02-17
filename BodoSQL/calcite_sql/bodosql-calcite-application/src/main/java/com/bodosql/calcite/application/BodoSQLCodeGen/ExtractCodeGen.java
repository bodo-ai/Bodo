package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlKind;

/** Class that returns the generated code for Extract after all inputs have been visited. */
public class ExtractCodeGen {

  // List of units that are unsupported for TIME inputs
  public static List<String> dayPlusUnits;

  static {
    dayPlusUnits = new ArrayList<String>();
    dayPlusUnits.add("YEAR");
    dayPlusUnits.add("QUARTER");
    dayPlusUnits.add("MONTH");
    dayPlusUnits.add("WEEK");
    dayPlusUnits.add("WEEKOFYEAR");
    dayPlusUnits.add("WEEKISO");
    dayPlusUnits.add("DAY");
    dayPlusUnits.add("DAYOFMONTH");
    dayPlusUnits.add("DOY");
    dayPlusUnits.add("DAYOFYEAR");
    dayPlusUnits.add("DOW");
    dayPlusUnits.add("DAYOFWEEK");
    dayPlusUnits.add("DAYOFWEEKISO");
  }

  // Used for doing null checking binary operations
  static SqlBinaryOperator addBinop =
      new SqlBinaryOperator("PLUS", SqlKind.PLUS, 0, true, null, null, null);
  static SqlBinaryOperator modBinop =
      new SqlBinaryOperator("MOD", SqlKind.MOD, 0, true, null, null, null);

  /**
   * Function that return the necessary generated code for an Extract call.
   *
   * @param datetimeVal The arg expr for selecting which datetime field to extract. This must be a
   *     constant string.
   * @param column The column arg expr.
   * @param isTime Is the input TIME data?
   * @return The code generated that matches the Extract expression.
   */
  public static String generateExtractCode(String datetimeVal, String column, boolean isTime) {
    if (isTime && dayPlusUnits.contains(datetimeVal)) {
      throw new BodoSQLCodegenException("Cannot extract unit " + datetimeVal + " from TIME values");
    }
    String extractCode;
    switch (datetimeVal) {
      case "NANOSECOND":
      case "MICROSECOND":
      case "MILLISECOND":
      case "SECOND":
      case "MINUTE":
      case "HOUR":
      case "MONTH":
      case "QUARTER":
      case "YEAR":
        extractCode =
            "bodo.libs.bodosql_array_kernels.get_" + datetimeVal.toLowerCase() + "(" + column + ")";
        break;
      case "DAY":
      case "DAYOFMONTH":
        extractCode = "bodo.libs.bodosql_array_kernels.dayofmonth(" + column + ")";
        break;
      case "DOW":
      case "DAYOFWEEK":
        extractCode = "bodo.libs.bodosql_array_kernels.dayofweek(" + column + ")";
        break;
      case "DAYOFWEEKISO":
        extractCode = "bodo.libs.bodosql_array_kernels.dayofweekiso(" + column + ")";
        break;
      case "DOY":
      case "DAYOFYEAR":
        extractCode = "bodo.libs.bodosql_array_kernels.dayofyear(" + column + ")";
        break;
      case "WEEK":
      case "WEEKOFYEAR":
      case "WEEKISO":
        extractCode = "bodo.libs.bodosql_array_kernels.get_weekofyear(" + column + ")";
        break;
      default:
        throw new BodoSQLCodegenException(
            "ERROR, datetime value: " + datetimeVal + " not supported inside of extract");
    }
    return extractCode;
  }

  /**
   * Function that returns the generated name for an Extract call.
   *
   * @param datetimeName The name for selecting which datetime field to extract.
   * @param columnName The name of the column arg.
   * @return The name generated that matches the Extract expression.
   */
  public static String generateExtractName(String datetimeName, String columnName) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("EXTRACT(").append(datetimeName).append(", ").append(columnName).append(")");
    return escapePythonQuotes(nameBuilder.toString());
  }

  /**
   * Returns the RexNodeVisitorInfo for DATE_PART by mapping the string literals to the same code
   * gen as EXTRACT
   *
   * @param operandsInfo The information about the arguments to the call
   * @param isTime Is the input TIME data?
   * @return The name generated that matches the Extract expression.
   */
  public static RexNodeVisitorInfo generateDatePart(
      List<RexNodeVisitorInfo> operandsInfo, boolean isTime) {
    String unit;
    switch (operandsInfo.get(0).getExprCode()) {
      case "\"year\"":
      case "\"y\"":
      case "\"yy\"":
      case "\"yyy\"":
      case "\"yyyy\"":
      case "\"yr\"":
      case "\"years\"":
      case "\"yrs\"":
        unit = "YEAR";
        break;

      case "\"month\"":
      case "\"mm\"":
      case "\"mon\"":
      case "\"mons\"":
      case "\"months\"":
        unit = "MONTH";
        break;

      case "\"day\"":
      case "\"d\"":
      case "\"dd\"":
      case "\"days\"":
      case "\"dayofmonth\"":
        unit = "DAY";
        break;

      case "\"dayofweek\"":
      case "\"weekday\"":
      case "\"dow\"":
      case "\"dw\"":
        unit = "DAYOFWEEK";
        break;

      case "\"dayofyear\"":
      case "\"yearday\"":
      case "\"doy\"":
      case "\"dy\"":
        unit = "DAYOFYEAR";
        break;

      case "\"week\"":
      case "\"w\"":
      case "\"wk\"":
      case "\"weekofyear\"":
      case "\"woy\"":
      case "\"wy\"":
        unit = "WEEK";
        break;

      case "\"weekiso\"":
      case "\"week_iso\"":
      case "\"weekofyeariso\"":
      case "\"weekofyear_iso\"":
        unit = "WEEKISO";
        break;

      case "\"quarter\"":
      case "\"q\"":
      case "\"qtr\"":
      case "\"qtrs\"":
      case "\"quarters\"":
        unit = "QUARTER";
        break;

      case "\"hour\"":
      case "\"h\"":
      case "\"hh\"":
      case "\"hr\"":
      case "\"hours\"":
      case "\"hrs\"":
        unit = "HOUR";
        break;

      case "\"minute\"":
      case "\"m\"":
      case "\"mi\"":
      case "\"min\"":
      case "\"minutes\"":
      case "\"mins\"":
        unit = "MINUTE";
        break;

      case "\"second\"":
      case "\"s\"":
      case "\"sec\"":
      case "\"seconds\"":
      case "\"secs\"":
        unit = "SECOND";
        break;

      case "\"nanosecond\"":
      case "\"ns\"":
      case "\"nsec\"":
      case "\"nanosec\"":
      case "\"nsecond\"":
      case "\"nanoseconds\"":
      case "\"nanosecs\"":
      case "\"nseconds\"":
        unit = "NANOSECOND";
        break;

      default:
        throw new BodoSQLCodegenException(
            "Unsupported DATE_PART unit: " + operandsInfo.get(0).getExprCode());
    }
    String code = generateExtractCode(unit, operandsInfo.get(1).getExprCode(), isTime);
    return new RexNodeVisitorInfo(code);
  }
}
