package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.HashMap;

public class TrigCodeGen {
  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call for the scalar case in the form of FN(scalar_expr)
  // IE ABS(x) => np.abs(x)
  static HashMap<String, String> equivalentFnMapScalars;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call for the column case in the form of FN(Col_expr)
  // IE LN(C) => np.log(C)
  static HashMap<String, String> equivalentFnMapColumns;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a pandas method call for the column case
  // IE LN(C) => C.FNNAME()
  static HashMap<String, String> equivalentPandasMethodMapColumns;

  static {
    equivalentFnMapScalars = new HashMap<>();
    equivalentFnMapColumns = new HashMap<>();
    equivalentPandasMethodMapColumns = new HashMap<>();
    equivalentFnMapScalars.put("ACOS", "bodosql.libs.generated_lib.sql_null_checking_arccos");
    equivalentFnMapColumns.put("ACOS", "np.arccos");
    equivalentFnMapScalars.put("ASIN", "bodosql.libs.generated_lib.sql_null_checking_arcsin");
    equivalentFnMapColumns.put("ASIN", "np.arcsin");
    equivalentFnMapScalars.put("ATAN", "bodosql.libs.generated_lib.sql_null_checking_arctan");
    equivalentFnMapColumns.put("ATAN", "np.arctan");
    equivalentFnMapScalars.put("ATAN2", "bodosql.libs.generated_lib.sql_null_checking_arctan2");
    equivalentFnMapColumns.put("ATAN2", "np.arctan2");
    equivalentFnMapScalars.put("COS", "bodosql.libs.generated_lib.sql_null_checking_cos");
    equivalentFnMapColumns.put("COS", "np.cos");
    equivalentFnMapScalars.put("SIN", "bodosql.libs.generated_lib.sql_null_checking_sin");
    equivalentFnMapColumns.put("SIN", "np.sin");
    equivalentFnMapScalars.put("TAN", "bodosql.libs.generated_lib.sql_null_checking_tan");
    equivalentFnMapColumns.put("TAN", "np.tan");
    equivalentFnMapScalars.put("RADIANS", "bodosql.libs.generated_lib.sql_null_checking_radians");
    equivalentFnMapColumns.put("RADIANS", "np.radians");
    equivalentFnMapScalars.put("DEGREES", "bodosql.libs.generated_lib.sql_null_checking_degrees");
    equivalentFnMapColumns.put("DEGREES", "np.degrees");
  }

  /**
   * Helper function that handles codegen for Single argument trig functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param outputScalar Should the output generate scalar code.
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgTrigFnInfo(
      String fnName, String arg1Expr, String arg1Name, boolean outputScalar) {
    String new_fn_name = fnName + "(" + arg1Name + ")";
    // COT generates the same code for scalar and column
    if (fnName.equals("COT")) {
      return new RexNodeVisitorInfo(new_fn_name, generateCotCode(arg1Expr));
    } else if (outputScalar && equivalentFnMapScalars.containsKey(fnName)) {
      String scalar_fn_str = equivalentFnMapScalars.get(fnName);
      return new RexNodeVisitorInfo(new_fn_name, scalar_fn_str + "(" + arg1Expr + ")");
    } else {
      assert !outputScalar;
      if (equivalentFnMapColumns.containsKey(fnName)) {
        String fn_expr = equivalentFnMapColumns.get(fnName);
        return new RexNodeVisitorInfo(new_fn_name, fn_expr + "(" + arg1Expr + ")");
      } else if (equivalentPandasMethodMapColumns.containsKey(fnName)) {
        String pandas_method = equivalentPandasMethodMapColumns.get(fnName);
        return new RexNodeVisitorInfo(new_fn_name, arg1Expr + "." + pandas_method + "()");
      } else {
        throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
      }
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
   * @param outputScalar Should the output generate scalar code.
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getDoubleArgTrigFnInfo(
      String fnName,
      String arg1Expr,
      String arg1Name,
      String arg2Expr,
      String arg2Name,
      boolean outputScalar) {
    String new_fn_name = fnName + "(" + arg1Name + "," + arg2Name + ")";
    if (outputScalar && equivalentFnMapScalars.containsKey(fnName)) {
      String scalar_fn_str = equivalentFnMapScalars.get(fnName);
      return new RexNodeVisitorInfo(
          new_fn_name, scalar_fn_str + "(" + arg1Expr + ", " + arg2Expr + ")");
    } else {
      assert !outputScalar;
      if (equivalentFnMapColumns.containsKey(fnName)) {
        String fn_expr = equivalentFnMapColumns.get(fnName);
        return new RexNodeVisitorInfo(
            new_fn_name, fn_expr + "(" + arg1Expr + ", " + arg2Expr + ")");
      } else if (equivalentPandasMethodMapColumns.containsKey(fnName)) {
        String pandas_method = equivalentPandasMethodMapColumns.get(fnName);
        return new RexNodeVisitorInfo(
            new_fn_name, arg1Expr + "." + pandas_method + "(" + arg2Expr + ")");
      } else {
        throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
      }
    }
  }

  /**
   * Function that return the necessary generated code for a COT Function Call.
   *
   * @param strExpr The expression on which to perform the COT call.
   * @return The code generated that matches the Tan expression.
   */
  public static String generateCotCode(String strExpr) {
    StringBuilder strBuilder = new StringBuilder();
    strBuilder.append("np.divide(1,np.tan(").append(strExpr).append("))");
    return strBuilder.toString();
  }
}
