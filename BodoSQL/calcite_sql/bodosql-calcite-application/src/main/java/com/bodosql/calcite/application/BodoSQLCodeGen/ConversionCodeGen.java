package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.SQLToPython.FormatHelpers.SQLFormatToPandasToDatetimeFormat;
import static com.bodosql.calcite.ir.ExprKt.BodoSQLKernel;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.Call;
import com.bodosql.calcite.ir.Expr.None;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import kotlin.Pair;

public class ConversionCodeGen {
  /**
   * Function that return the necessary generated code for a StrToDate Function Call.
   *
   * @param strExpr The string Expression to be converted to a timestamp.
   * @param is_scalar Is the input/output a scalar or column?
   * @param SQLFormatStr A string literal SQL format string. Required to be a string literal by the
   *     function signature
   * @return The code generated that matches the StrToDate expression.
   */
  public static Expr generateStrToDateCode(Expr strExpr, boolean is_scalar, String SQLFormatStr) {
    List<Expr> args = List.of(strExpr);
    List<Pair<String, Expr>> namedArgs =
        List.of(new Pair<>("format", SQLFormatToPandasToDatetimeFormat(SQLFormatStr)));
    Expr.Call conversion = new Call("pd.to_datetime", args, namedArgs);
    if (!is_scalar) {
      return new Expr.Attribute(conversion, "date");
    } else {
      return new Expr.Method(conversion, "date", List.of(), List.of());
    }
  }

  /**
   * Handles codegen for Snowflake TO_DATE/TRY_TO_DATE function.
   *
   * @param operands List of operands
   * @param fnName The name of the function
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_DATE/TRY_TO_DATE function
   */
  public static Expr generateToDateFnCode(
      List<Expr> operands, String fnName, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    args.addAll(operands);
    if (operands.size() == 1) {
      args.add(None.INSTANCE);
    }
    assert args.size() == 2;
    return BodoSQLKernel(fnName.toLowerCase(Locale.ROOT), args, streamingNamedArgs);
  }

  /**
   * Handles codegen for Snowflake TO_TIMESTAMP/TRY_TO_TIMESTAMP function.
   *
   * @param operands List of operands
   * @param tzExpr Expr representing the timezone of the output data
   * @param fnName The name of the function being called
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_TIMESTAMP/TRY_TO_TIMESTAMP function
   */
  public static Expr generateToTimestampFnCode(
      List<Expr> operands,
      Expr tzExpr,
      String fnName,
      List<Pair<String, Expr>> streamingNamedArgs) {

    String kernelName;
    if (fnName.startsWith("TRY_")) {
      kernelName = "try_to_timestamp";
    } else {
      kernelName = "to_timestamp";
    }
    List<Expr> args;

    if (operands.size() == 2) {
      // 2nd argument is a format string
      if (operands.get(1) instanceof Expr.StringLiteral) {
        // kernel argument order: conversionVal, format_str, time_zone, scale
        args = List.of(operands.get(0), operands.get(1), tzExpr, new Expr.IntegerLiteral(0));
      } else {
        // 2nd argument is a scale (integer)
        args = List.of(operands.get(0), Expr.None.INSTANCE, tzExpr, operands.get(1));
      }
    } else {
      args = List.of(operands.get(0), Expr.None.INSTANCE, tzExpr, new Expr.IntegerLiteral(0));
    }
    return BodoSQLKernel(kernelName, args, streamingNamedArgs);
  }

  /**
   * Handles codegen for Snowflake TO_DOUBLE function.
   *
   * @param operands List of operands
   * @param fnName Name of the function (TO_DOUBLE or TRY_TO_DOUBLE)
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_DOUBLE function
   */
  public static Expr generateToDoubleFnCode(
      List<Expr> operands, String fnName, List<Pair<String, Expr>> streamingNamedArgs) {
    if (operands.size() > 1) {
      throw new BodoSQLCodegenException(
          "Error, format string for " + fnName + " not yet supported");
    }
    List<Expr> args = new ArrayList<>();
    args.addAll(operands);
    // Append none for the format string.
    // TODO(njriasan): Support format strings.
    args.add(None.INSTANCE);
    return BodoSQLKernel(fnName.toLowerCase(), args, streamingNamedArgs);
  }

  /**
   * Handles codegen for Snowflake TO_CHAR function.
   *
   * @param operandsInfo List of operands
   * @return Expr for the TO_CHAR function
   */
  public static Expr generateToCharFnCode(List<Expr> operandsInfo, List<Boolean> argScalars) {
    return BodoSQLKernel(
        "to_char",
        operandsInfo,
        List.of(new Pair<>("is_scalar", new Expr.BooleanLiteral(argScalars.get(0)))));
  }

  /**
   * Handles codegen for Snowflake TO_BOOLEAN and TRY_TO_BOOLEAN function.
   *
   * @param operands List of operands
   * @param fnName Name of the function (TO_BOOLEAN or TRY_TO_BOOLEAN)
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_BOOLEAN or TRY_TO_BOOLEAN function
   */
  public static Expr generateToBooleanFnCode(
      List<Expr> operands, String fnName, List<Pair<String, Expr>> streamingNamedArgs) {
    assert operands.size() == 1 : "Error, " + fnName + " function takes 1 argument";
    return BodoSQLKernel(fnName.toLowerCase(), operands, streamingNamedArgs);
  }

  /**
   * Handles codegen for Snowflake TO_BINARY and TRY_TO_BINARY function.
   *
   * @param operands List of operands
   * @param fnName Name of the function (TO_BINARY or TRY_TO_BINARY)
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_BINARY or TRY_TO_BINARY function
   */
  public static Expr generateToBinaryFnCode(
      List<Expr> operands, String fnName, List<Pair<String, Expr>> streamingNamedArgs) {
    if (operands.size() > 1) {
      throw new BodoSQLCodegenException(
          fnName.toLowerCase() + ": format argument not yet supported");
    }
    return BodoSQLKernel(fnName.toLowerCase(), operands, streamingNamedArgs);
  }

  /**
   * Handles codegen for Snowflake TO_ARRAY function.
   *
   * @param operands List of operands
   * @param argScalars Whether each argument is a scalar or a column
   * @return Expr for the TO_ARRAY function
   */
  public static Expr generateToArrayCode(List<Expr> operands, List<Boolean> argScalars) {
    ArrayList<Pair<String, Expr>> kwargs = new ArrayList<>();
    kwargs.add(new Pair<>("is_scalar", new Expr.BooleanLiteral(argScalars.get(0))));
    return BodoSQLKernel("to_array", operands, kwargs);
  }

  /**
   * Does the codegen for MySQL TIMESTAMP function
   *
   * @param args The input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TIMESTAMP function
   */
  public static Expr generateTimestampFnCode(
      List<Expr> args, List<Pair<String, Expr>> streamingNamedArgs) {
    return BodoSQLKernel("create_timestamp", args, streamingNamedArgs);
  }
}
