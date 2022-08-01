package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.HashMap;
import java.util.List;

public class NumericCodeGen {
  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call for the scalar case in the form of FN(scalar_expr)
  // IE ABS(x) => np.abs(x)
  static HashMap<String, String> equivalentFnMapScalars;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call in the form of FN(Col_expr) for the column case
  // IE LN(C) => np.log(C)
  static HashMap<String, String> equivalentFnMapColumns;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a pandas method for the column case
  // IE LN(C) => C.FNNAME()
  static HashMap<String, String> equivalentPandasMethodMapColumns;

  static {
    equivalentFnMapScalars = new HashMap<>();
    equivalentFnMapColumns = new HashMap<>();
    equivalentPandasMethodMapColumns = new HashMap<>();
    equivalentFnMapColumns.put("CEIL", "np.ceil");
    equivalentFnMapScalars.put("CEIL", "bodosql.libs.generated_lib.sql_null_checking_ceil");
    equivalentFnMapColumns.put("FLOOR", "np.floor");
    equivalentFnMapScalars.put("FLOOR", "bodosql.libs.generated_lib.sql_null_checking_floor");
    equivalentFnMapColumns.put("MOD", "np.mod");
    equivalentFnMapScalars.put("MOD", "bodosql.libs.generated_lib.sql_null_checking_mod");
    equivalentFnMapColumns.put("SIGN", "np.sign");
    equivalentFnMapScalars.put("SIGN", "bodosql.libs.generated_lib.sql_null_checking_sign");

    equivalentPandasMethodMapColumns.put("ROUND", "round");
    equivalentFnMapScalars.put("ROUND", "bodosql.libs.generated_lib.sql_null_checking_round");
    equivalentPandasMethodMapColumns.put("TRUNCATE", "round");
    equivalentFnMapScalars.put("TRUNCATE", "bodosql.libs.generated_lib.sql_null_checking_round");

    equivalentPandasMethodMapColumns.put("ABS", "abs");
    equivalentFnMapScalars.put("ABS", "bodosql.libs.generated_lib.sql_null_checking_abs");

    equivalentFnMapColumns.put("LOG2", "np.log2");
    equivalentFnMapScalars.put("LOG2", "bodosql.libs.generated_lib.sql_null_checking_log2");
    equivalentFnMapColumns.put("LOG10", "np.log10");
    equivalentFnMapScalars.put("LOG10", "bodosql.libs.generated_lib.sql_null_checking_log10");
    equivalentFnMapColumns.put("LN", "np.log");
    equivalentFnMapScalars.put("LN", "bodosql.libs.generated_lib.sql_null_checking_ln");
    equivalentFnMapColumns.put("EXP", "np.exp");
    equivalentFnMapScalars.put("EXP", "bodosql.libs.generated_lib.sql_null_checking_exp");

    equivalentPandasMethodMapColumns.put("POW", "pow");
    equivalentFnMapScalars.put("POW", "bodosql.libs.generated_lib.sql_null_checking_power");
    equivalentPandasMethodMapColumns.put("POWER", "pow");
    equivalentFnMapScalars.put("POWER", "bodosql.libs.generated_lib.sql_null_checking_power");
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
      String fnName, String arg1Expr, String arg1Name, boolean outputScalar) {

    String new_fn_name = fnName + "(" + arg1Name + ")";
    if (outputScalar && equivalentFnMapScalars.containsKey(fnName)) {
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
      // One operand, we default to log10 as that is the dfault behavior in mySQL
      if (exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR || isScalar) {
        exprStrBuilder.append("bodosql.libs.generated_lib.sql_null_checking_log10(");
      } else {
        exprStrBuilder.append("np.log10(");
      }
      exprStrBuilder.append(operandsInfo.get(0).getExprCode()).append(")");
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
