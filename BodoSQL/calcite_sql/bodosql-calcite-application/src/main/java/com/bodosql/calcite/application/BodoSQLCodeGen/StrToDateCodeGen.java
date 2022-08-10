package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.SQLToPython.FormatHelpers.SQLFormatToPandasToDatetimeFormat;

import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;

public class StrToDateCodeGen {
  /**
   * Function that return the necessary generated code for a StrToDate Function Call.
   *
   * @param strExpr The string Expression to be converted to a timestamp.
   * @param SQLFormatStr A string literal SQL format string. Required to be a string literal by the
   *     function signature
   * @return The code generated that matches the StrToDate expression.
   */
  public static String generateStrToDateCode(
      String strExpr, BodoSQLExprType.ExprType strExprType, String SQLFormatStr) {
    StringBuilder strBuilder = new StringBuilder();
    if (strExprType == BodoSQLExprType.ExprType.COLUMN) {
      strBuilder
          .append("pd.to_datetime(")
          .append(strExpr)
          .append(", format=")
          .append(SQLFormatToPandasToDatetimeFormat(SQLFormatStr))
          .append(")");
    } else {
      strBuilder
          .append("bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(")
          .append(strExpr)
          .append(", ")
          .append(SQLFormatToPandasToDatetimeFormat(SQLFormatStr))
          .append(")");
    }

    return strBuilder.toString();
  }

  /**
   * Function that returns the generated name for a StrToDate Function Call.
   *
   * @param argName The string Expression's name.
   * @param SQLFormatStr A string literal SQL format string.
   * @return The name generated that matches the StrToDate expression.
   */
  public static String generateStrToDateName(String argName, String SQLFormatStr) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder
        .append("STR_TO_DATE(")
        .append(argName)
        .append(", ")
        .append(SQLFormatStr)
        .append(")");
    return nameBuilder.toString();
  }

  /**
   * Does the codegen for MySQL DATE function
   *
   * @param datetimeStr the datetime string to convert to a date
   * @return RexVisitorInfo for the DATE function
   */
  public static RexNodeVisitorInfo generateDateFnCode(String datetimeStr) {
    return new RexNodeVisitorInfo(
        "DATE(" + datetimeStr + ")", "pd.Timestamp(" + datetimeStr + ").floor(freq=\"D\")");
  }

  /**
   * Does the codegen for MySQL TIMESTAMP function
   *
   * @param datetimeStr the datetime string to convert to a Timestamp
   * @return RexVisitorInfo for the TIMESTAMP function
   */
  public static RexNodeVisitorInfo generateTimestampFnCode(String datetimeStr) {
    return new RexNodeVisitorInfo(
        "TIMESTAMP(" + datetimeStr + ")", "pd.Timestamp(" + datetimeStr + ")");
  }
}
