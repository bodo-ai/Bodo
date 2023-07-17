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
    equivalentFnMap.put("IF", "bodo.libs.bodosql_array_kernels.cond");
    equivalentFnMap.put("IFF", "bodo.libs.bodosql_array_kernels.cond");
    equivalentFnMap.put("DECODE", "bodo.libs.bodosql_array_kernels.decode");
    equivalentFnMap.put("HASH", "bodo.libs.bodosql_array_kernels.sql_hash");
    equivalentFnMap.put("COALESCE", "bodo.libs.bodosql_array_kernels.coalesce");
    equivalentFnMap.put("NVL", "bodo.libs.bodosql_array_kernels.coalesce");
    equivalentFnMap.put("NVL2", "bodo.libs.bodosql_array_kernels.coalesce");
    equivalentFnMap.put("IFNULL", "bodo.libs.bodosql_array_kernels.coalesce");
    equivalentFnMap.put("ZEROIFNULL", "bodo.libs.bodosql_array_kernels.coalesce");
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function in the
   * hashtable equivalentFnMap.
   *
   * @param fnName the name of the function being called
   * @param codeExprs the Python expressions to calculate the arguments
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr getCondFnInfo(String fnName, List<Expr> codeExprs) {

    String kernelName;
    if (equivalentFnMap.containsKey(fnName)) {
      kernelName = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    return new Expr.Call(kernelName, codeExprs);
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions COALESCE, HASH, DECODE,
   * or any of their variants.
   *
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr visitVariadic(RexCall fnOperation, List<Expr> codeExprs) {
    String fnName = fnOperation.getOperator().toString();
    String kernelName;
    if (equivalentFnMap.containsKey(fnName)) {
      kernelName = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    if (fnName == "ZEROIFNULL") {
      codeExprs.add(new Expr.Raw("0"));
    }
    return new Expr.Call(kernelName, new Expr.Tuple(codeExprs));
  }
}
