package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.ir.ExprKt.bodoSQLKernel;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.None;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;

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
        equivalentFnMap.put(fn, "power");
      } else if (fn.equals("TRUNCATE")) {
        equivalentFnMap.put(fn, "trunc");
      } else if (fn.equals("MOD")) {
        equivalentFnMap.put(fn, "modulo_numeric");
      } else {
        equivalentFnMap.put(fn, fn.toLowerCase());
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
      return bodoSQLKernel(func.toLowerCase(), operands, List.of());
    } else if (operands.size() == 1) {
      return bodoSQLKernel(
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
      return bodoSQLKernel(equivalentFnMap.get(fnName), args, List.of());
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
  }

  /**
   * Helper function that handles codegen for the ROUND function. This will coerce the second
   * argument (round_scale) to an int literal if it is a call to np.int32.
   *
   * @param args The input expressions
   * @return The Expr corresponding to the function call (round)
   */
  public static Expr generateRoundCode(List<Expr> args) {
    assert args.size() == 2;
    String roundScaleStr = args.get(1).emit();
    // Convert the second argument to an int literal instead of a call with np.int32
    if (roundScaleStr.startsWith("np.int")) {
      Expr constRoundScale = new Expr.Raw(roundScaleStr.split("\\(|\\)")[1]);
      args.set(1, constRoundScale);
    }
    return bodoSQLKernel("round", args, List.of());
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
    return bodoSQLKernel(fnName, args, List.of());
  }

  /**
   * Helper function that handles the codegen for TO_NUMBER/TO_NUMERIC/TO_DECIMAL functions, or the
   * TRY_ versions.
   *
   * @param arg The input argument.
   * @param precision The target precision
   * @param scale The target scale
   * @param isTry Is the call a TRY_TO_XXX function.
   * @param outputDecimal Is the output type decimal. This code generation step is shared by both
   *     integer and decimal output types.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr for the function call.
   */
  public static Expr generateToNumberCode(
      Expr arg,
      int precision,
      int scale,
      boolean isTry,
      boolean outputDecimal,
      List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> exprs = new ArrayList<>();
    exprs.add(arg);
    // BodoSQL generally wraps all integer literals in calls to np.intx.
    // This breaks constant propagation in Bodo, so we need to do some
    // hacky stuff to get the literal values, so that Bodo can recognize
    // the prec/scale values as constants
    final int finalPrec;
    final int finalScale;
    if (precision == RelDataType.PRECISION_NOT_SPECIFIED) {
      finalPrec = 38;
    } else {
      finalPrec = precision;
    }
    if (scale == RelDataType.SCALE_NOT_SPECIFIED) {
      finalScale = 0;
    } else {
      finalScale = scale;
    }
    exprs.add(new Expr.IntegerLiteral(finalPrec));
    exprs.add(new Expr.IntegerLiteral(finalScale));
    exprs.add(new Expr.BooleanLiteral(outputDecimal));
    String name = isTry ? "try_to_number" : "to_number";
    return bodoSQLKernel(name, exprs, streamingNamedArgs);
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
    return bodoSQLKernel(
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
    return bodoSQLKernel("random_seedless", List.of(arg), List.of());
  }

  /**
   * Helper function that handles codegen for the UNIFORM function
   *
   * @param operands The list of expressions for each argument
   * @return The Expr corresponding to the function call
   */
  public static Expr generateUniformFnInfo(List<Expr> operands) {
    return bodoSQLKernel("uniform", operands, List.of());
  }
}
