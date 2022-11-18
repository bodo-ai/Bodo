package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.List;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlKind;

/** Class that returns the generated code for Extract after all inputs have been visited. */
public class ExtractCodeGen {

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
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Extract expression.
   */
  public static String generateExtractCode(
      String datetimeVal, String column, boolean outputScalar) {
    String extractCode;
    switch (datetimeVal) {
      case "NANOSECOND":
      case "MICROSECOND":
      case "MILLISECOND":
      case "SECOND":
      case "MINUTE":
      case "HOUR":
      case "DAY":
      case "MONTH":
      case "QUARTER":
      case "YEAR":
        // For these cases, the equivalent pandas function is just
        // the lowercase of the extract flag
        if (outputScalar) {
          extractCode =
              "bodosql.libs.generated_lib.sql_null_checking_"
                  + datetimeVal.toLowerCase()
                  + "("
                  + column
                  + ")";
        } else {
          extractCode = "pd.Series(" + column + ").dt." + datetimeVal.toLowerCase() + ".values";
        }
        break;
      case "DAYOFMONTH":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_day" + "(" + column + ")";
        } else {
          extractCode = "pd.Series(" + column + ").dt.day.values";
        }
        break;
      case "DOW":
      case "DAYOFWEEK":
        // pandas has monday = 0, and counts up from there
        // MYSQL has sunday = 1, and counts up from there
        if (outputScalar) {
          // The scalar library fn handles the addition/modulo
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_dayofweek(" + column + ")";
        } else {
          extractCode = "((pd.Series(" + column + ").dt.dayofweek + 1) % 7 + 1).values";
        }
        break;
      case "DOY":
      case "DAYOFYEAR":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_dayofyear(" + column + ")";
        } else {
          extractCode = "pd.Series(" + column + ").dt.dayofyear.values";
        }
        break;
      case "WEEK":
      case "WEEKOFYEAR":
      case "WEEKISO":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_weekofyear(" + column + ")";
        } else {
          extractCode = "pd.Series(" + column + ").dt.isocalendar().week.values";
        }
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
   * @param outputScalar Whether the output is a scalar or not
   * @return The name generated that matches the Extract expression.
   */
  public static RexNodeVisitorInfo generateDatePart(
      List<RexNodeVisitorInfo> operandsInfo, boolean outputScalar) {
    StringBuilder name = new StringBuilder();
    name.append("DATE_PART(")
        .append(operandsInfo.get(0).getName())
        .append(", ")
        .append(operandsInfo.get(1).getName())
        .append(")");

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
            "Unsupported DATE_PART unit: " + operandsInfo.get(0).getName());
    }
    String code = generateExtractCode(unit, operandsInfo.get(1).getExprCode(), outputScalar);
    return new RexNodeVisitorInfo(name.toString(), code);
  }
}
