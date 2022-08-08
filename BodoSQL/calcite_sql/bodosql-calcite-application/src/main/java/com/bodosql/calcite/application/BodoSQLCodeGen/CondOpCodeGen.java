package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.application.Utils.BodoCtx;
import java.util.HashMap;
import java.util.HashSet;
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
   * @param nullSets The set of columns that need null checking for each argument.
   * @param inputVar The input variable.
   * @return The code generated for the Case call.
   */
  public static String generateCaseCode(
      List<String> args,
      boolean generateApply,
      List<HashSet<String>> nullSets,
      BodoCtx ctx,
      String inputVar) {
    StringBuilder genCode = new StringBuilder();
    genCode.append("(");
    /*case statements are essentially an infinite number of Then/When clauses, followed by an
    else clause. So, we iterate through all the Then/When clauses, and then deal with the final
    else clause at the end*/
    for (int i = 0; i < args.size() - 1; i += 2) {
      HashSet<String> whenCols = nullSets.get(i);
      String when = args.get(i);
      HashSet<String> thenCols = nullSets.get(i + 1);
      String then = args.get(i + 1);
      // If any column is NULL and unchecked, we return NULL because in general
      // op(NULL, val, ...) == NULL
      genCode.append(generateNullCheck(inputVar, thenCols, "None", then));
      genCode.append(" if ");
      String whenCheck = "";
      // If there are no columns to check, whenCheck will return an empty string
      whenCheck = checkNotNullColumns(inputVar, whenCols);
      if (!whenCheck.equals("")) {
        // If any column is not null checked we assume the condition evaluates to
        // NULL because in general op(NULL, val, ...) == NULL
        // NULL is treated as False in a boolean context
        genCode.append("(").append(whenCheck).append(" and ").append(when).append(")");
      } else {
        genCode.append(when);
      }
      genCode.append(" else (");
    }
    HashSet<String> elseCols = nullSets.get(args.size() - 1);
    String else_ = args.get(args.size() - 1);
    // If any column is NULL and unchecked, we return NULL because in general
    // op(NULL, val, ...) == NULL
    genCode.append(generateNullCheck(inputVar, elseCols, "None", else_));
    // append R parens equal to the number of Then/When clauses
    for (int j = 0; j < args.size() / 2; j++) {
      genCode.append(")");
    }

    genCode.append(")");
    if (generateApply) {
      // Generate the apply, but ignore the nullset, as it was already taken care of above
      return generateDfApply(
          inputVar,
          new BodoCtx(ctx.getColsToAddList(), new HashSet<>(), ctx.getNamedParams()),
          genCode.toString());
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
