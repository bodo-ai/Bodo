package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.SQLToPython.FormatHelpers.SQLFormatToPandasToDatetimeFormat;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.ir.Expr;
import java.util.List;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.type.SqlTypeName;

public class ConversionCodeGen {
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
          .append(").date");
    } else {
      strBuilder
          .append("pd.to_datetime(")
          .append(strExpr)
          .append(", format=")
          .append(SQLFormatToPandasToDatetimeFormat(SQLFormatStr))
          .append(").date()");
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
  public static Expr generateDateFnCode(String datetimeStr) {
    return new Expr.Raw("bodo.libs.bodosql_array_kernels.create_date(" + datetimeStr + ")");
  }

  /**
   * Handles codegen for Snowflake TO_DATE/TRY_TO_DATE function.
   *
   * @param operandsInfo List of operands
   * @param fnName The name of the function
   * @return RexVisitorInfo for the TO_DATE/TRY_TO_DATE function
   */
  public static Expr generateToDateFnCode(List<Expr> operandsInfo, String fnName) {
    if (operandsInfo.size() == 2) {
      throw new BodoSQLCodegenException(
          "Error, " + fnName + " with two arguments not yet supported");
    }
    StringBuilder exprCode = new StringBuilder();
    exprCode
        .append("bodo.libs.bodosql_array_kernels.")
        .append(fnName.toLowerCase())
        .append("(")
        .append(operandsInfo.get(0).emit())
        .append(", None)");
    return new Expr.Raw(exprCode.toString());
  }

  /**
   * Handles codegen for Snowflake TO_TIMESTAMP/TRY_TO_TIMESTAMP function.
   *
   * @param operandsInfo List of operands
   * @param operandsTypeInfo List of type information about the operands
   * @param tzStr String representing the timezone of the output data
   * @param fnName The name of the function being called
   * @return RexVisitorInfo for the TO_TIMESTAMP/TRY_TO_TIMESTAMP function
   */
  public static Expr generateToTimestampFnCode(
      List<Expr> operandsInfo, List<RexNode> operandsTypeInfo, String tzStr, String fnName) {
    String scaleStr = "0";
    if (operandsInfo.size() == 2) {
      if (SqlTypeName.INT_TYPES.contains(operandsTypeInfo.get(1).getType().getSqlTypeName())) {
        scaleStr = operandsInfo.get(1).emit();
      } else {
        throw new BodoSQLCodegenException(
            "Error, format string for " + fnName + " not yet supported");
      }
    }
    StringBuilder exprCode = new StringBuilder();

    String kernelName;
    if (fnName.startsWith("TRY_")) {
      kernelName = "try_to_timestamp";
    } else {
      kernelName = "to_timestamp";
    }

    exprCode
        .append("bodo.libs.bodosql_array_kernels.")
        .append(kernelName)
        .append("(")
        .append(operandsInfo.get(0).emit())
        .append(", None, ")
        .append(tzStr)
        .append(", ")
        .append(scaleStr)
        .append(")");
    return new Expr.Raw(exprCode.toString());
  }

  /**
   * Handles codegen for Snowflake TO_DOUBLE function.
   *
   * @param operandsInfo List of operands
   * @param fnName Name of the function (TO_DOUBLE or TRY_TO_DOUBLE)
   * @return RexVisitorInfo for the TO_DOUBLE function
   */
  public static Expr generateToDoubleFnCode(List<Expr> operandsInfo, String fnName) {
    if (operandsInfo.size() > 1) {
      throw new BodoSQLCodegenException(
          "Error, format string for " + fnName + " not yet supported");
    }
    String exprCode =
        "bodo.libs.bodosql_array_kernels."
            + fnName.toLowerCase()
            + "("
            + operandsInfo.get(0).emit()
            + ", None)";
    return new Expr.Raw(exprCode);
  }

  /**
   * Handles codegen for Snowflake TO_CHAR function.
   *
   * @param operandsInfo List of operands
   * @param fnName Name of the function (TO_CHAR or TRY_VARCHAR)
   * @param arg0IsDate If the argument to the function is of date type. This is used to ensure that
   *     a "date" formatted string is generated in the case that we are representing the SQL date in
   *     a pandas timestamp. IE we want: '2023-03-28 09:46:41.630549' (if the argument is timestamp)
   *     '2023-03-28' (if the argument is actually a date)
   *     <p>This argument will not be needed after we complete the transition to a dedicated date
   *     type.
   * @return RexVisitorInfo for the TO_CHAR function
   */
  public static Expr generateToCharFnCode(
      List<Expr> operandsInfo, String fnName, boolean arg0IsDate) {
    if (operandsInfo.size() > 1) {
      // TODO (BE-3742): Support format string for TO_CHAR
      throw new BodoSQLCodegenException(
          "Error, format string for " + fnName + " not yet supported");
    }
    String exprCode = "bodo.libs.bodosql_array_kernels.to_char(" + operandsInfo.get(0).emit();
    if (arg0IsDate) {
      exprCode += ", treat_timestamp_as_date = True";
    }
    exprCode += ")";
    return new Expr.Raw(exprCode);
  }

  /**
   * Handles codegen for Snowflake TO_BOOLEAN and TRY_TO_BOOLEAN function.
   *
   * @param operandsInfo List of operands
   * @param @param fnName Name of the function (TO_BOOLEAN or TRY_TO_BOOLEAN)
   * @return RexVisitorInfo for the TO_BOOLEAN or TRY_TO_BOOLEAN function
   */
  public static Expr generateToBooleanFnCode(List<Expr> operandsInfo, String fnName) {
    assert operandsInfo.size() == 1 : "Error, " + fnName + " function takes 1 argument";
    String exprCode =
        "bodo.libs.bodosql_array_kernels."
            + fnName.toLowerCase()
            + "("
            + operandsInfo.get(0).emit()
            + ")";
    return new Expr.Raw(exprCode);
  }

  /**
   * Handles codegen for Snowflake TO_BINARY and TRY_TO_BINARY function.
   *
   * @param operandsInfo List of operands
   * @param @param fnName Name of the function (TO_BINARY or TRY_TO_BINARY)
   * @return RexVisitorInfo for the TO_BINARY or TRY_TO_BINARY function
   */
  public static Expr generateToBinaryFnCode(List<Expr> operandsInfo, String fnName) {
    if (operandsInfo.size() > 1) {
      throw new BodoSQLCodegenException(
          fnName.toLowerCase() + ": format argument not yet supported");
    }
    String exprCode =
        "bodo.libs.bodosql_array_kernels."
            + fnName.toLowerCase()
            + "("
            + operandsInfo.get(0).emit()
            + ")";
    return new Expr.Raw(exprCode);
  }

  /**
   * Does the codegen for MySQL TIMESTAMP function
   *
   * @param datetimeStr the datetime string to convert to a Timestamp
   * @return RexVisitorInfo for the TIMESTAMP function
   */
  public static Expr generateTimestampFnCode(String datetimeStr) {
    return new Expr.Raw("bodo.libs.bodosql_array_kernels.create_timestamp(" + datetimeStr + ")");
  }
}
