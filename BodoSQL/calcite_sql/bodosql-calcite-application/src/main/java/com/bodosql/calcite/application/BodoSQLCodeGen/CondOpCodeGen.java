package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import kotlin.Pair;

public class CondOpCodeGen {

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call where any of the arguments can be scalars or vectors.
  // IE SQLFN(C1, s1, C2, s2) => FN(C1, s1, C2, s2)
  // EX REGR_VALY(A, 3.1) => bodo.libs.bodosql_array_kernels.regr_valy(table1['A'], 3.1)
  static HashMap<String, String> equivalentFnMap;

  static {
    equivalentFnMap = new HashMap<>();
    equivalentFnMap.put("REGR_VALX", "regr_valx");
    equivalentFnMap.put("REGR_VALY", "regr_valy");
    equivalentFnMap.put("BOOLAND", "booland");
    equivalentFnMap.put("BOOLOR", "boolor");
    equivalentFnMap.put("BOOLXOR", "boolxor");
    equivalentFnMap.put("BOOLNOT", "boolnot");
    equivalentFnMap.put("EQUAL_NULL", "equal_null");
    equivalentFnMap.put("IFF", "cond");
    equivalentFnMap.put("DECODE", "decode");
    equivalentFnMap.put("HASH", "sql_hash");
    equivalentFnMap.put("COALESCE", "coalesce");
    equivalentFnMap.put("NVL2", "nvl2");
    equivalentFnMap.put("ZEROIFNULL", "coalesce");
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function in the
   * hashtable equivalentFnMap. These are functions that lack dictionary encoding optimizations.
   *
   * @param fnName the name of the function being called
   * @param codeExprs the Python expressions to calculate the arguments
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr getCondFuncCode(String fnName, List<Expr> codeExprs) {

    String kernelName;
    if (equivalentFnMap.containsKey(fnName)) {
      kernelName = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    return ExprKt.BodoSQLKernel(kernelName, codeExprs, List.of());
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function in the
   * hashtable equivalentFnMap. These are functions that contain optimizations for dictionary
   * encoding.
   *
   * @param fnName the name of the function being called
   * @param codeExprs the Python expressions to calculate the arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @param argScalars Whether each argument is a scalar or a column
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr getCondFuncCodeOptimized(
      String fnName,
      List<Expr> codeExprs,
      List<Pair<String, Expr>> streamingNamedArgs,
      List<Boolean> argScalars) {

    String kernelName;
    if (equivalentFnMap.containsKey(fnName)) {
      kernelName = equivalentFnMap.get(fnName);
    } else {
      // If we made it here, something has gone very wrong
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
    ArrayList<Pair<String, Expr>> kwargs = new ArrayList();
    kwargs.add(new Pair<String, Expr>("is_scalar_a", new Expr.BooleanLiteral(argScalars.get(0))));
    kwargs.add(new Pair<String, Expr>("is_scalar_b", new Expr.BooleanLiteral(argScalars.get(1))));
    kwargs.addAll(streamingNamedArgs);
    return ExprKt.BodoSQLKernel(kernelName, codeExprs, kwargs);
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions HASH.
   *
   * @param codeExprs the Python strings that calculate each of the arguments.
   * @param argScalars Whether each argument is a scalar or a column.
   * @param visitor A visitor used to lower globals.
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr visitHash(
      List<Expr> codeExprs, List<Boolean> argScalars, PandasCodeGenVisitor visitor) {
    ArrayList<Expr> scalarExprs = new ArrayList();
    for (Boolean isScalar : argScalars) {
      scalarExprs.add(new Expr.BooleanLiteral(isScalar));
    }
    Expr scalarGlobal = visitor.lowerAsMetaType(new Expr.Tuple(scalarExprs));
    return ExprKt.BodoSQLKernel(
        "sql_hash", List.of(new Expr.Tuple(codeExprs), scalarGlobal), List.of());
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions COALESCE, DECODE, or any
   * of their variants. These are functions may contain dictionary encoding optimizations.
   *
   * @param fnName the name of the function being called
   * @param codeExprs the Python expressions to calculate the arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr visitVariadic(
      String fnName, List<Expr> codeExprs, List<Pair<String, Expr>> streamingNamedArgs) {
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
    return ExprKt.BodoSQLKernel(kernelName, List.of(new Expr.Tuple(codeExprs)), streamingNamedArgs);
  }
}
