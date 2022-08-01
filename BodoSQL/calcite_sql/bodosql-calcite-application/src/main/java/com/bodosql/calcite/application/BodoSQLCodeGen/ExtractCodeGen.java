package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;

import com.bodosql.calcite.application.BodoSQLCodegenException;
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
   * @param dateVal The arg expr for selecting which date field to extract. This must be a constant
   *     string.
   * @param column The column arg expr.
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Extract expression.
   */
  public static String generateExtractCode(String dateVal, String column, boolean outputScalar) {
    String extractCode;
    switch (dateVal) {
      case "MICROSECOND":
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
                  + dateVal.toLowerCase()
                  + "("
                  + column
                  + ")";
        } else {
          extractCode = column + ".dt." + dateVal.toLowerCase();
        }
        break;
      case "DAYOFMONTH":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_day" + "(" + column + ")";
        } else {
          extractCode = column + ".dt.day";
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
          extractCode = "((" + column + ".dt.dayofweek + 1) % 7 + 1)";
        }
        break;
      case "DOY":
      case "DAYOFYEAR":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_dayofyear(" + column + ")";
        } else {
          extractCode = column + ".dt.dayofyear";
        }
        break;
      case "WEEK":
      case "WEEKOFYEAR":
      case "WEEKISO":
        if (outputScalar) {
          extractCode = "bodosql.libs.generated_lib.sql_null_checking_weekofyear(" + column + ")";
        } else {
          extractCode = column + ".dt.isocalendar().week";
        }
        break;
      default:
        throw new BodoSQLCodegenException(
            "ERROR, date value: " + dateVal + " not supported inside of extract");
    }
    return extractCode;
  }

  /**
   * Function that returns the generated name for an Extract call.
   *
   * @param dateName The name for selecting which date field to extract.
   * @param columnName The name of the column arg.
   * @return The name generated that matches the Extract expression.
   */
  public static String generateExtractName(String dateName, String columnName) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("EXTRACT(").append(dateName).append(", ").append(columnName).append(")");
    return escapePythonQuotes(nameBuilder.toString());
  }
}
