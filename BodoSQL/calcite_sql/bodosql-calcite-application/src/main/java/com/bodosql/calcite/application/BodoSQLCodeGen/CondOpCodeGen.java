package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.application.Utils.BodoCtx;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import java.util.*;
import kotlin.Pair;
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
   * @param code1 the Python expression to calculate the argument
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo getSingleArgCondFnInfo(String fnName, String code1) {

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

    return new RexNodeVisitorInfo(expr_code.toString());
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
  public static RexNodeVisitorInfo getDoubleArgCondFnInfo(
      String fnName, String code1, String code2) {

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

    return new RexNodeVisitorInfo(expr_code.toString());
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
   * @param outputsArray Should this function output an array? If so we must generate a call to
   *     bodosql_case_placeholder.
   * @param inputVar The input variable.
   * @param outputType The type of the output scalar/column.
   * @param pdVisitorClass A reference to the PandasCodeGenVisitor used to create globals and
   *     variable names.
   * @param outerBuilder The module builder used for Code Generation.
   * @param innerBuilder The module builder used for generating global variables.
   * @return The code generated for the Case call.
   */
  public static String generateCaseCode(
      List<String> args,
      boolean outputsArray,
      BodoCtx ctx,
      String inputVar,
      RelDataType outputType,
      PandasCodeGenVisitor pdVisitorClass,
      Module.Builder outerBuilder,
      Module.Builder innerBuilder) {
    // We will unify the output of each block into a single variable.
    // TODO: Move variable generation to the Module.Builder
    if (outputsArray) {
      Variable outputVar = new Variable(pdVisitorClass.genGenericTempVar());
      // case statements are essentially an infinite number of When/Then clauses, followed by an
      // else clause. We first construct all pairs of condition + if body and finally add the else.

      List<Pair<Expr.Call, Op.Assign>> condAndBody = new ArrayList<>();

      for (int i = 0; i < args.size() - 1; i += 2) {
        // Create the condition
        Expr.Call when =
            new Expr.Call(
                "bodo.libs.bodosql_array_kernels.is_true", List.of(new Expr.Raw(args.get(i))));
        // Create the if body
        Op.Assign body = new Op.Assign(outputVar, new Expr.Raw(args.get(i + 1)));
        condAndBody.add(new Pair<>(when, body));
      }
      Op.Assign elseBlock = new Op.Assign(outputVar, new Expr.Raw(args.get(args.size() - 1)));
      // Add the case to the module
      innerBuilder.add(new Op.If(condAndBody, elseBlock));

      // Here we need to generate a call to bodosql_case_placeholder.
      // We assume, at this point, that the ctx.colsToAddList has been added to inputVar, and
      // the arguments have been renamed appropriately.

      // pass named parameters as kws to bodosql_case_placeholder()
      // sorting to make sure the same code is generated on each rank
      TreeSet<String> sortedParamSet = new TreeSet<>(ctx.getNamedParams());
      StringBuilder namedParamArgs = new StringBuilder();
      for (String param : sortedParamSet) {
        // TODO: Move these to use the Module.Builder
        namedParamArgs.append(param + "=" + param + ", ");
      }

      // generate bodosql_case_placeholder() call with inputs:
      // 1) a tuple of necessary input arrays
      // 2) number of output rows (same as input rows, needed for allocation)
      // 3) initialization code for unpacking the input array tuple with the right array names
      // (MetaType global)
      // 4) body of the CASE loop (global constant)
      // 5) loop variable name
      // 6) output array type
      // 7) Named parameters
      // For example:
      // S5 = bodo.utils.typing.bodosql_case_placeholder(
      //   (bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df3, 0), ),
      //   len(df3),
      //   MetaType(('  df3_0 = arrs[0]',)),
      //   '((None if (pd.isna(bodo.utils.indexing.scalar_optional_getitem(df3_0)) ) else
      // np.int64(1),
      //   IntegerArrayType(int64),
      //   '_temp4'
      // )
      Module.Builder initModule = new Module.Builder(outerBuilder.getSymbolTable());

      List<Expr.Call> inputDataArgs = new ArrayList<>();

      int i = 0;
      TreeSet<Integer> sortedUsedColumns = new TreeSet<>(ctx.getUsedColumns());
      for (int colNo : sortedUsedColumns) {
        inputDataArgs.add(
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
                List.of(new Expr.Raw(inputVar), new Expr.Raw(String.valueOf(colNo)))));
        initModule.add(
            new Op.Assign(
                new Variable(inputVar + "_" + colNo), new Expr.Raw(String.format("arrs[%d]", i))));
        i++;
      }
      Expr.Tuple inputData = new Expr.Tuple(inputDataArgs);
      // TODO: Move global generation to the Module.Builder + change the return types to Variables
      Expr.TripleQuotedString initBody =
          new Expr.TripleQuotedString(new Expr.Raw(initModule.build().emit(1)));
      Expr.Raw initGlobal = new Expr.Raw(pdVisitorClass.lowerAsMetaType(initBody.emit()));
      // have to use triple quotes here since the body can contain " or '. We use indent level 2
      // since
      // This is embedded inside a for loop.
      Expr.TripleQuotedString loopBody =
          new Expr.TripleQuotedString(new Expr.Raw(innerBuilder.build().emit(2)));
      Expr.Raw bodyGlobal = new Expr.Raw(pdVisitorClass.lowerAsGlobal(loopBody.emit()));

      Expr.Raw outputArrayTypeGlobal =
          new Expr.Raw(pdVisitorClass.lowerAsGlobal(sqlTypeToBodoArrayType(outputType, false)));
      // Add a new assignment for the case
      Variable arrVar = new Variable(pdVisitorClass.genGenericTempVar());
      Expr.Call functionCall =
          new Expr.Call(
              "bodo.utils.typing.bodosql_case_placeholder",
              List.of(
                  inputData,
                  new Expr.Call("len", List.of(new Expr.Raw(inputVar))),
                  initGlobal,
                  bodyGlobal,
                  new Expr.Raw(makeQuoted(outputVar.getName())),
                  outputArrayTypeGlobal,
                  new Expr.Raw(namedParamArgs.toString())));

      outerBuilder.add(new Op.Assign(arrVar, functionCall));
      // Return the name of the variable.
      return arrVar.getName();
    } else {
      // We can't handle generating code that should be nested inside the case statement yet.
      // stick to the ternary for now.
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
      return genCode.toString();
    }
  }

  /**
   * Return a pandas expression that replicates a call to the SQL functions COALESCE, DECODE, or any
   * of their variants.
   *
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo visitVariadic(RexCall fnOperation, List<String> codeExprs) {
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

    return new RexNodeVisitorInfo(expr_code.toString());
  }

  /**
   * Return a pandas expression that replicates a SQL IF function call. This function requires very
   * specific handling of the nullset
   *
   * @param codeExprs the Python strings that calculate each of the arguments
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public static RexNodeVisitorInfo visitIf(RexCall fnOperation, List<String> codeExprs) {

    StringBuilder expr_code = new StringBuilder("bodo.libs.bodosql_array_kernels.cond(");
    expr_code
        .append(codeExprs.get(0))
        .append(", ")
        .append(codeExprs.get(1))
        .append(", ")
        .append(codeExprs.get(2))
        .append(")");

    return new RexNodeVisitorInfo(expr_code.toString());
  }
}
