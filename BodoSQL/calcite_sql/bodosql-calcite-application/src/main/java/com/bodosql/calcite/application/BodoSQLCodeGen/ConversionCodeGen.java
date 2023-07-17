package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.SQLToPython.FormatHelpers.SQLFormatToPandasToDatetimeFormat;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.List;

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
      String strExpr, boolean is_scalar, String SQLFormatStr) {
    StringBuilder strBuilder = new StringBuilder();
    if (!is_scalar) {
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
   * Handles codegen for Snowflake TO_DATE/TRY_TO_DATE function.
   *
   * @param operandsInfo List of operands
   * @param fnName The name of the function
   * @return RexVisitorInfo for the TO_DATE/TRY_TO_DATE function
   */
  public static Expr generateToDateFnCode(List<Expr> operandsInfo, String fnName) {
    StringBuilder exprCode = new StringBuilder();
    exprCode
        .append("bodo.libs.bodosql_array_kernels.")
        .append(fnName.toLowerCase())
        .append("(")
        .append(operandsInfo.get(0).emit())
        .append(", ");
    if (operandsInfo.size() == 2) {
      exprCode.append(operandsInfo.get(1).emit());
    } else {
      exprCode.append("None");
    }
    exprCode.append(")");
    return new Expr.Raw(exprCode.toString());
  }

  /**
   * Handles codegen for Snowflake TO_TIMESTAMP/TRY_TO_TIMESTAMP function.
   *
   * @param operandsInfo List of operands
   * @param tzStr String representing the timezone of the output data
   * @param fnName The name of the function being called
   * @return RexVisitorInfo for the TO_TIMESTAMP/TRY_TO_TIMESTAMP function
   */
  public static Expr generateToTimestampFnCode(
      List<Expr> operandsInfo, String tzStr, String fnName) {

    String kernelName;
    if (fnName.startsWith("TRY_")) {
      kernelName = "try_to_timestamp";
    } else {
      kernelName = "to_timestamp";
    }
    List<Expr> args;

    if (operandsInfo.size() == 2) {
      // 2nd argument is a format string
      if (operandsInfo.get(1) instanceof Expr.StringLiteral) {
        // kernel argument order: conversionVal, format_str, time_zone, scale
        args =
            List.of(
                operandsInfo.get(0),
                operandsInfo.get(1),
                new Expr.Raw(tzStr),
                new Expr.IntegerLiteral(0));
      } else {
        // 2nd argument is a scale (integer)
        args =
            List.of(
                operandsInfo.get(0), Expr.None.INSTANCE, new Expr.Raw(tzStr), operandsInfo.get(1));
      }
    } else {
      args =
          List.of(
              operandsInfo.get(0),
              Expr.None.INSTANCE,
              new Expr.Raw(tzStr),
              new Expr.IntegerLiteral(0));
    }
    return ExprKt.BodoSQLKernel(kernelName, args, List.of());
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
   * @return RexVisitorInfo for the TO_CHAR function
   */
  public static Expr generateToCharFnCode(List<Expr> operandsInfo) {
    List<Expr> args = new ArrayList<>();
    args.addAll(operandsInfo);
    if (operandsInfo.size() == 1) {
      args.add(Expr.None.INSTANCE);
    }
    return ExprKt.BodoSQLKernel("to_char", args, List.of());
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
