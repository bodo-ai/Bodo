package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.application.Utils.BodoCtx;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rel.type.*;
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
   * @param name1 the name of the argument
   * @param code1 the Python expression to calculate the argument
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo getSingleArgCondFnInfo(
      String fnName, String name1, String code1) {

    StringBuilder name = new StringBuilder(fnName);
    name.append("(");
    name.append(name1);
    name.append(")");

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

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a call to a SQL conditional function call with two
   * arguments
   *
   * @param fnName the name of the function being called
   * @param name1 the name of the first argument
   * @param code1 the Python expression to calculate the first argument
   * @param name2 the name of the second argument
   * @param code2 the Python expression to calculate the second argument
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo getDoubleArgCondFnInfo(
      String fnName, String name1, String code1, String name2, String code2) {

    StringBuilder name = new StringBuilder(fnName);
    name.append("(");
    name.append(name1);
    name.append(", ");
    name.append(name2);
    name.append(")");

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

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the name for Case.
   *
   * @param names The name of each argument.
   * @return The name generated for the Case call.
   */
  public static String generateCaseName(List<String> names) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("CASE(");
    for (int i = 0; i < names.size() - 1; i += 2) {
      nameBuilder
          .append("WHEN ")
          .append(names.get(i))
          .append(" THEN ")
          .append(names.get(i + 1))
          .append(" ");
    }
    nameBuilder.append("ELSE ").append(names.get(names.size() - 1)).append(" END)");
    return escapePythonQuotes(nameBuilder.toString());
  }

  /**
   * Function that returns the necessary generated code for Case.
   *
   * @param args The arguments to Case.
   * @param generateApply Is this function used to generate an apply.
   * @param inputVar The input variable.
   * @return The code generated for the Case call.
   */
  public static String generateCaseCode(
      List<String> args,
      boolean generateApply,
      BodoCtx ctx,
      String inputVar,
      RelDataType outputType,
      List<String> colNames,
      PandasCodeGenVisitor pdVisitorClass) {
    StringBuilder genCode = new StringBuilder();
    genCode.append("(");
    /*case statements are essentially an infinite number of Then/When clauses, followed by an
    else clause. So, we iterate through all the Then/When clauses, and then deal with the final
    else clause at the end*/
    for (int i = 0; i < args.size() - 1; i += 2) {
      String when = args.get(i);
      String then = args.get(i + 1);
      genCode.append(then);
      genCode.append(" if bodo.libs.bodosql_array_kernels.is_true(");
      genCode.append(when);
      genCode.append(") else (");
    }
    String else_ = args.get(args.size() - 1);
    genCode.append(else_);
    // append R parens equal to the number of Then/When clauses
    for (int j = 0; j < args.size() / 2; j++) {
      genCode.append(")");
    }

    genCode.append(")");
    if (generateApply) {
      return generateDfApply(inputVar, ctx, genCode.toString(), outputType, pdVisitorClass);
    }

    return genCode.toString();
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions COALESCE, DECODE, or any
   * of their variants.
   *
   * @param names the names of each of the arguments
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo visitVariadic(
      RexCall fnOperation, List<String> names, List<String> codeExprs) {
    String func_name = fnOperation.getOperator().toString();
    int n = fnOperation.operands.size();
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();
    if (func_name == "DECODE") {
      name.append("DECODE(");
      expr_code.append("bodo.libs.bodosql_array_kernels.decode((");
    } else {
      name.append("COALESCE(");
      expr_code.append("bodo.libs.bodosql_array_kernels.coalesce((");
    }

    for (int i = 0; i < n; i++) {
      name.append(names.get(i));
      if (i != n - 1) {
        name.append(", ");
      } else {
        name.append(")");
      }
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

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a SQL IF function call. This function requires very
   * specific handling of the nullset
   *
   * @param names the names of each of the arguments
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo visitIf(
      RexCall fnOperation, List<String> names, List<String> codeExprs) {

    StringBuilder name = new StringBuilder("IF(");
    StringBuilder expr_code = new StringBuilder("bodo.libs.bodosql_array_kernels.cond(");
    name.append(names.get(0))
        .append(", ")
        .append(names.get(1))
        .append(", ")
        .append(names.get(2))
        .append(")");
    expr_code
        .append(codeExprs.get(0))
        .append(", ")
        .append(codeExprs.get(1))
        .append(", ")
        .append(codeExprs.get(2))
        .append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }
}
