package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

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

  // List defining all numeric functions to be mapped (from fnList) which will
  // have two arguments.
  static List<String> doubleArgFns =
      Arrays.asList(
          "BITAND",
          "BITOR",
          "BITXOR",
          "BITSHIFTLEFT",
          "BITSHIFTRIGHT",
          "GETBIT",
          "MOD",
          "POW",
          "POWER",
          "ROUND",
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
   * Helper function that handles codegen for Single argument numeric functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @return The Expr corresponding to the function call
   */
  public static String getSingleArgNumericFnInfo(String fnName, String arg1Expr) {

    if (equivalentFnMap.containsKey(fnName)) {
      return equivalentFnMap.get(fnName) + "(" + arg1Expr + ")";
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
  }

  /**
   * Helper function that handles codegen for double argument numeric functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg2Expr The string expression of arg2
   * @return The Expr corresponding to the function call
   */
  public static String getDoubleArgNumericFnInfo(String fnName, String arg1Expr, String arg2Expr) {

    if (!doubleArgFns.contains(fnName)) {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
    return equivalentFnMap.get(fnName) + "(" + arg1Expr + ", " + arg2Expr + ")";
  }

  /**
   * Function that return the necessary generated code for a CONV Function Call.
   *
   * @param inputExpr The first argument of the CONV call, either a scalar, or a column
   * @param curBaseExpr The second argument of the CONV call, the current base
   * @param newBaseExpr The second argument of the CONV call, the base to convert to.
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the CONV expression.
   */
  public static String generateConvCode(
      String inputExpr, String curBaseExpr, String newBaseExpr, boolean outputScalar) {
    StringBuilder strBuilder = new StringBuilder();
    strBuilder
        .append("bodo.libs.bodosql_array_kernels.conv(")
        .append(inputExpr)
        .append(", ")
        .append(curBaseExpr)
        .append(", ")
        .append(newBaseExpr)
        .append(")");
    return strBuilder.toString();
  }

  /**
   * Function that returns the generated name for a CONV Function Call.
   *
   * @param inputName The first argument of the CONV call, either a scalar, or a column
   * @param curBaseName The second argument of the CONV call, the current base
   * @param newBaseName The second argument of the CONV call, the base to convert to.
   * @return The name generated that matches the CONV expression.
   */
  public static String generateConvName(String inputName, String curBaseName, String newBaseName) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder
        .append("CONV(")
        .append(inputName)
        .append(", ")
        .append(curBaseName)
        .append(", ")
        .append(newBaseName)
        .append(")");
    return nameBuilder.toString();
  }

  /**
   * Function that returns the RexVisitorInfo for a LOG Function Call.
   *
   * @param operandsInfo A list of the visitor info, containing the information for each operand
   * @return The RexVisitorInfo that matches the LOG expression.
   */
  public static Expr generateLogFnInfo(List<Expr> operandsInfo) {
    StringBuilder exprStrBuilder = new StringBuilder();
    if (operandsInfo.size() == 1) {
      // One operand, we default to log10 as that is the default behavior in mySQL
      exprStrBuilder.append(
          "bodo.libs.bodosql_array_kernels.log10(" + operandsInfo.get(0).emit() + ")");
    } else {
      assert operandsInfo.size() == 2;

      exprStrBuilder
          .append("bodo.libs.bodosql_array_kernels.log(")
          .append(operandsInfo.get(0).emit())
          .append(",")
          .append(operandsInfo.get(1).emit())
          .append(")");
    }
    return new Expr.Raw(exprStrBuilder.toString());
  }

  /**
   * Helper function that handles the codegen for TO_NUMBER/TO_NUMERIC/TO_DECIMAL function
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateToNumberCode(Expr arg1Info, String fnName) {
    String outputExpr = "bodo.libs.bodosql_array_kernels.to_number(" + arg1Info.emit() + ")";
    return new Expr.Raw(outputExpr);
  }

  /**
   * Helper function that handles the codegen for TRY_TO_NUMBER/TRY_TO_NUMERIC/TRY_TO_DECIMAL
   * function
   *
   * @param arg1Info The VisitorInfo for the first argument.
   * @return the rexNodeVisitorInfo for the function call
   */
  public static Expr generateTryToNumberCode(Expr arg1Info, String fnName) {
    String outputExpr = "bodo.libs.bodosql_array_kernels.try_to_number(" + arg1Info.emit() + ")";
    return new Expr.Raw(outputExpr);
  }

  public static Expr generateLeastGreatestCode(String fnName, List<Expr> operands) {
    String kernelName = "bodo.libs.bodosql_array_kernels." + fnName.toLowerCase();

    // LEAST and GREATEST take in a variable number of arguments,
    // so we wrap these arguments in a tuple as input
    return new Expr.Call(kernelName, List.of(new Expr.Tuple(operands)));
  }

  /**
   * Function that returns the RexVisitorInfo for a RANDOM Function Call.
   *
   * @param inputName name of the DataFrame whose length the random output column must match (if not
   *     a single row)
   * @param isSingleRow true if the output should be a scalar
   * @return The RexVisitorInfo that matches the RANDOM expression.
   */
  public static Expr generateRandomFnInfo(String inputName, boolean isSingleRow) {
    StringBuilder exprStrBuilder = new StringBuilder();
    exprStrBuilder.append("bodo.libs.bodosql_array_kernels.random_seedless(");
    if (isSingleRow) {
      exprStrBuilder.append("None");
    } else {
      exprStrBuilder.append(inputName);
    }
    exprStrBuilder.append(")");
    return new Expr.Raw(exprStrBuilder.toString());
  }

  /**
   * Helper function that handles codegen for the UNIFORM function
   *
   * @param operands The list of expressions for each argument
   * @return The Expr corresponding to the function call
   */
  public static Expr generateUniformFnInfo(List<Expr> operands) {
    return ExprKt.BodoSQLKernel("uniform", operands, List.of());
  }
}
