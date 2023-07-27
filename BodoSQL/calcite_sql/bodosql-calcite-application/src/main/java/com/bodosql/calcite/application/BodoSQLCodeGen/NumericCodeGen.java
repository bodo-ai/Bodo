package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.ir.ExprKt.BodoSQLKernel;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.None;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import kotlin.Pair;

// List defining all numeric functions which will be mapped to their corresponding array kernel in
// Python.
public class NumericCodeGen {
  static List<String> fnList =
      Arrays.asList(
          "ABS",
          "BITAND",
          "BITOR",
          "BITXOR",
          "BITSHIFTLEFT",
          "BITSHIFTRIGHT",
          "BITNOT",
          "CBRT",
          "CEIL",
          "EXP",
          "FACTORIAL",
          "FLOOR",
          "GETBIT",
          "LN",
          "LOG2",
          "LOG10",
          "MOD",
          "POW",
          "POWER",
          "ROUND",
          "SIGN",
          "SQRT",
          "SQUARE",
          "TRUNC",
          "TRUNCATE");

  // HashMap of all numeric functions which maps to array kernels
  // which handle all combinations of scalars/arrays/nulls.
  static HashMap<String, String> equivalentFnMap = new HashMap<>();

  static {
    for (String fn : fnList) {
      if (fn.equals("POW")) {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels.power");
      } else if (fn.equals("TRUNCATE")) {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels.trunc");
      } else {
        equivalentFnMap.put(fn, "bodo.libs.bodosql_array_kernels." + fn.toLowerCase());
      }
    }
  }

  /**
   * Helper function that handles codegen for FLOOR / CEIL.
   *
   * @param func Either "FLOOR" or "CEIL"
   * @param operands The inputs to the function
   * @return The Expr corresponding to the function call
   */
  public static Expr genFloorCeilCode(String func, List<Expr> operands) {
    if (operands.size() == 2) {
      return BodoSQLKernel(func.toLowerCase(), operands, List.of());
    } else if (operands.size() == 1) {
      return BodoSQLKernel(
          func.toLowerCase(), List.of(operands.get(0), new Expr.IntegerLiteral(0)), List.of());
    } else {
      throw new BodoSQLCodegenException(
          func + " expects 1 or 2 arguments, received " + operands.size() + " arguments");
    }
  }

  /**
   * Helper function that handles codegen for most numeric functions.
   *
   * @param fnName The name of the function
   * @param args The input expressions
   * @return The Expr corresponding to the function call
   */
  public static Expr getNumericFnCode(String fnName, List<Expr> args) {

    if (equivalentFnMap.containsKey(fnName)) {
      return new Expr.Call(equivalentFnMap.get(fnName), args);
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
  }

  /**
   * Function that returns the RexVisitorInfo for a LOG Function Call.
   *
   * @param args A list of the exprs for the arguments.
   * @return The Expr that matches the LOG expression.
   */
  public static Expr generateLogFnInfo(List<Expr> args) {
    String fnName;
    if (args.size() == 1) {
      // One operand, we default to log10 as that is the default behavior in mySQL
      fnName = "log10";
    } else {
      assert args.size() == 2;
      fnName = "log";
    }
    return BodoSQLKernel(fnName, args, List.of());
  }

  /**
   * Helper function that handles the codegen for TO_NUMBER/TO_NUMERIC/TO_DECIMAL function
   *
   * @param args The arguments to this call.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr for the function call.
   */
  public static Expr generateToNumberCode(
      List<Expr> args, List<Pair<String, Expr>> streamingNamedArgs) {
    return BodoSQLKernel("to_number", args, streamingNamedArgs);
  }

  /**
   * Helper function that handles the codegen for TRY_TO_NUMBER/TRY_TO_NUMERIC/TRY_TO_DECIMAL
   * function
   *
   * @param args The arguments to this call.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr for the function call.
   */
  public static Expr generateTryToNumberCode(
      List<Expr> args, List<Pair<String, Expr>> streamingNamedArgs) {
    return BodoSQLKernel("try_to_number", args, streamingNamedArgs);
  }

  /**
   * Generate the code for least/greatest.
   *
   * @param fnName Name of the operation. least or greatest.
   * @param operands Arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr for the function call.
   */
  public static Expr generateLeastGreatestCode(
      String fnName, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    // LEAST and GREATEST take in a variable number of arguments,
    // so we wrap these arguments in a tuple as input
    return BodoSQLKernel(
        fnName.toLowerCase(), List.of(new Expr.Tuple(operands)), streamingNamedArgs);
  }

  /**
   * Function that returns the RexVisitorInfo for a RANDOM Function Call.
   *
   * @param input The Table whose length the random output column must match (if not a single row)
   * @param isSingleRow true if the output should be a scalar
   * @return The RexVisitorInfo that matches the RANDOM expression.
   */
  public static Expr generateRandomFnInfo(BodoEngineTable input, boolean isSingleRow) {
    Expr arg;
    if (isSingleRow) {
      arg = None.INSTANCE;
    } else {
      arg = input;
    }
    return new Expr.Call("bodo.libs.bodosql_array_kernels.random_seedless", arg);
  }

  /**
   * Helper function that handles codegen for the UNIFORM function
   *
   * @param operands The list of expressions for each argument
   * @return The Expr corresponding to the function call
   */
  public static Expr generateUniformFnInfo(List<Expr> operands) {
    return BodoSQLKernel("uniform", operands, List.of());
  }
}
