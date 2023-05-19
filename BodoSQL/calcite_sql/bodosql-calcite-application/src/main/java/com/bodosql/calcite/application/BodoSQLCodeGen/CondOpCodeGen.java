package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rex.RexCall;

public class CondOpCodeGen {

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call where any of the arguments can be scalars or vectors.
  // IE SQLFN(C1, s1, C2, s2) => FN(C1, s1, C2, s2)
  // EX REGR_VALY(A, 3.1) => bodo.libs.bodosql_array_kernels.regr_valy(table1['A'], 3.1)
  static HashMap<String, String> equivalentFnMap;

  static {
    equivalentFnMap = new HashMap<>();
    equivalentFnMap.put("REGR_VALX", "bodo.libs.bodosql_array_kernels.regr_valx");
    equivalentFnMap.put("REGR_VALY", "bodo.libs.bodosql_array_kernels.regr_valy");
    equivalentFnMap.put("BOOLAND", "bodo.libs.bodosql_array_kernels.booland");
    equivalentFnMap.put("BOOLOR", "bodo.libs.bodosql_array_kernels.boolor");
    equivalentFnMap.put("BOOLXOR", "bodo.libs.bodosql_array_kernels.boolxor");
    equivalentFnMap.put("BOOLNOT", "bodo.libs.bodosql_array_kernels.boolnot");
    equivalentFnMap.put("EQUAL_NULL", "bodo.libs.bodosql_array_kernels.equal_null");
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function call with one
   * argument
   *
   * @param fnName the name of the function being called
   * @param code1 the Python expression to calculate the argument
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static Expr getSingleArgCondFnInfo(String fnName, String code1) {

    String kernel_str;
    if (equivalentFnMap.containsKey(fnName)) {
      kernel_str = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    StringBuilder expr_code = new StringBuilder(kernel_str);
    expr_code.append("(");
    expr_code.append(code1);
    expr_code.append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function call with two
   * arguments
   *
   * @param fnName the name of the function being called
   * @param code1 the Python expression to calculate the first argument
   * @param code2 the Python expression to calculate the second argument
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static Expr getDoubleArgCondFnInfo(String fnName, String code1, String code2) {

    String kernel_str;
    if (equivalentFnMap.containsKey(fnName)) {
      kernel_str = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    StringBuilder expr_code = new StringBuilder(kernel_str);
    expr_code.append("(");
    expr_code.append(code1);
    expr_code.append(", ");
    expr_code.append(code2);
    expr_code.append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions COALESCE, DECODE, or any
   * of their variants.
   *
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static Expr visitVariadic(RexCall fnOperation, List<String> codeExprs) {
    String func_name = fnOperation.getOperator().toString();
    int n = fnOperation.operands.size();
    StringBuilder expr_code = new StringBuilder();
    if (func_name == "DECODE") {
      expr_code.append("bodo.libs.bodosql_array_kernels.decode((");
    } else {
      expr_code.append("bodo.libs.bodosql_array_kernels.coalesce((");
    }

    for (int i = 0; i < n; i++) {
      expr_code.append(codeExprs.get(i));

      if (i != (n - 1)) {
        expr_code.append(", ");
      } else {
        if (func_name == "ZEROIFNULL") {
          expr_code.append(", 0");
        }
        expr_code.append("))");
      }
    }

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a SQL IF function call. This function requires very
   * specific handling of the nullset
   *
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static Expr visitIf(RexCall fnOperation, List<String> codeExprs) {

    StringBuilder expr_code = new StringBuilder("bodo.libs.bodosql_array_kernels.cond(");
    expr_code
        .append(codeExprs.get(0))
        .append(", ")
        .append(codeExprs.get(1))
        .append(", ")
        .append(codeExprs.get(2))
        .append(")");

    return new Expr.Raw(expr_code.toString());
  }
}
