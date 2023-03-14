package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class TrigCodeGen {
  static List<String> fnList =
      Arrays.asList(
          "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2", "cos", "cosh", "cot", "sin",
          "sinh", "tan", "tanh", "degrees", "radians");
  static List<String> doubleArgFns = Arrays.asList("atan2");

  // HashMap of all trig functions which maps to array kernels
  // which handle all combinations of scalars/arrays/nulls.
  static HashMap<String, String> equivalentFnMap = new HashMap<>();

  static {
    for (String fn : fnList) {
      equivalentFnMap.put(fn.toUpperCase(), "bodo.libs.bodosql_array_kernels." + fn);
    }
  }

  /**
   * Helper function that handles codegen for Single argument trig functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static Expr getSingleArgTrigFnInfo(String fnName, String arg1Expr) {
    if (equivalentFnMap.containsKey(fnName)) {
      return new Expr.Raw(equivalentFnMap.get(fnName) + "(" + arg1Expr + ")");
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
  }

  /**
   * Helper function that handles codegen for double argument trig functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg2Expr The string expression of arg2
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static Expr getDoubleArgTrigFnInfo(String fnName, String arg1Expr, String arg2Expr) {
    // Only ATAN2 is a double argument function
    assert doubleArgFns.contains(fnName.toLowerCase());
    return new Expr.Raw(equivalentFnMap.get(fnName) + "(" + arg1Expr + ", " + arg2Expr + ")");
  }
}
