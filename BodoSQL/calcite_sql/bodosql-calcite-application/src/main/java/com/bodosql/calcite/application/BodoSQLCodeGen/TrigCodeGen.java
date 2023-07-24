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
   * @param args The arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr getTrigFnCode(String fnName, List<Expr> args) {
    if (equivalentFnMap.containsKey(fnName)) {
      return new Expr.Call(equivalentFnMap.get(fnName), args);
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
  }
}
