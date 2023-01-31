package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
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
   * @param arg1Name The name of arg1
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgTrigFnInfo(
      String fnName, String arg1Expr, String arg1Name) {
    String new_fn_name = fnName + "(" + arg1Name + ")";
    if (equivalentFnMap.containsKey(fnName)) {
      return new RexNodeVisitorInfo(
          new_fn_name, equivalentFnMap.get(fnName) + "(" + arg1Expr + ")");
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
  }

  /**
   * Helper function that handles codegen for double argument trig functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param arg2Expr The string expression of arg2
   * @param arg2Name The name of arg2
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getDoubleArgTrigFnInfo(
      String fnName, String arg1Expr, String arg1Name, String arg2Expr, String arg2Name) {
    // Only ATAN2 is a double argument function
    assert doubleArgFns.contains(fnName.toLowerCase());
    String new_fn_name = fnName + "(" + arg1Name + "," + arg2Name + ")";
    return new RexNodeVisitorInfo(
        new_fn_name, equivalentFnMap.get(fnName) + "(" + arg1Expr + ", " + arg2Expr + ")");
  }
}
