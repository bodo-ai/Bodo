package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

// List definining all numeric functions which will be mapped to their corresponding array kernel in
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

  // List defining all numeric functions to be mapped (from fnList) which will have two arguments.
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
   * @param arg1Name The string name of arg1
   * @param outputScalar Should the output generate scalar code.
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgNumericFnInfo(
      String fnName, String arg1Expr, String arg1Name) {

    String new_fn_name = fnName + "(" + arg1Name + ")";
    if (equivalentFnMap.containsKey(fnName)) {
      return new RexNodeVisitorInfo(
          new_fn_name, equivalentFnMap.get(fnName) + "(" + arg1Expr + ")");
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
  }

  /**
   * Helper function that handles codegen for double argument numeric functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param arg2Expr The string expression of arg2
   * @param arg2Name The name of arg2
   * @param outputScalar Should the output generate scalar code.
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getDoubleArgNumericFnInfo(
      String fnName, String arg1Expr, String arg1Name, String arg2Expr, String arg2Name) {

    if (!doubleArgFns.contains(fnName)) {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + " not supported");
    }
    String new_fn_name = fnName + "(" + arg1Name + ", " + arg2Name + ")";
    return new RexNodeVisitorInfo(
        new_fn_name, equivalentFnMap.get(fnName) + "(" + arg1Expr + ", " + arg2Expr + ")");
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
   * @param exprTypes A list of the expression types of the operands
   * @return The RexVisitorInfo that matches the LOG expression.
   */
  public static RexNodeVisitorInfo generateLogFnInfo(
      List<RexNodeVisitorInfo> operandsInfo,
      List<BodoSQLExprType.ExprType> exprTypes,
      boolean isScalar) {
    StringBuilder exprStrBuilder = new StringBuilder();
    StringBuilder nameStrBuilder = new StringBuilder();
    if (operandsInfo.size() == 1) {
      // One operand, we default to log10 as that is the default behavior in mySQL
      exprStrBuilder.append(
          "bodo.libs.bodosql_array_kernels.log10(" + operandsInfo.get(0).getExprCode() + ")");
      nameStrBuilder.append("LOG(").append(operandsInfo.get(0).getName()).append(")");
    } else {
      assert operandsInfo.size() == 2;

      exprStrBuilder
          .append("bodo.libs.bodosql_array_kernels.log(")
          .append(operandsInfo.get(0).getExprCode())
          .append(",")
          .append(operandsInfo.get(1).getExprCode())
          .append(")");

      nameStrBuilder
          .append("LOG(")
          .append(operandsInfo.get(0).getName())
          .append(", ")
          .append(operandsInfo.get(0).getName())
          .append(")");
    }
    return new RexNodeVisitorInfo(nameStrBuilder.toString(), exprStrBuilder.toString());
  }
}
