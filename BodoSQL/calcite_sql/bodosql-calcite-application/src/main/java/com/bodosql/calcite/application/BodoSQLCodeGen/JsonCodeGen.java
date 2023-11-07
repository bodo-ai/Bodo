package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.adapter.pandas.RexToPandasTranslator;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;

public class JsonCodeGen {
  static HashMap<String, String> jsonFnMap;

  static {
    jsonFnMap = new HashMap<>();
    jsonFnMap.put(
        "JSON_EXTRACT_PATH_TEXT", "bodo.libs.bodosql_array_kernels.json_extract_path_text");
    jsonFnMap.put("OBJECT_KEYS", "bodo.libs.bodosql_array_kernels.object_keys");
  }

  /**
   * Function that return the necessary generated code for the JSON function
   * OBJECT_CONSTRUCT_KEEP_NULL.
   *
   * @param operation The call to OBJECT_CONSTRUCT_KEEP_NULL
   * @param argScalars A list indicating which arguments are scalars
   * @param rexToPandas A visitor to transform each operand into Python code
   * @return The Expression to calculate the function call.
   */
  public static Expr getObjectConstructKeepNullCode(
      RexCall operation,
      List<Boolean> argScalars,
      RexToPandasTranslator rexToPandas,
      PandasCodeGenVisitor visitor) {
    ArrayList<Expr> valExprs = new ArrayList<>();
    ArrayList<Expr> keyExprs = new ArrayList<>();
    ArrayList<Expr> scalarExprs = new ArrayList<>();
    for (int i = 0; i < operation.operands.size(); i++) {
      if (i % 2 == 0) {
        String key = ((RexLiteral) (operation.operands.get(i))).getValueAs(String.class);
        keyExprs.add(new Expr.StringLiteral(key));
      } else {
        Expr operandInfo = operation.operands.get(i).accept(rexToPandas);
        // Need to unbox scalar timestamp values.
        if (argScalars.get(i)) {
          operandInfo =
              new Expr.Call(
                  "bodo.utils.conversion.unbox_if_tz_naive_timestamp", List.of(operandInfo));
        }
        valExprs.add(operandInfo);
        scalarExprs.add(new Expr.BooleanLiteral(argScalars.get(i)));
      }
    }
    Expr keyGlobal = visitor.lowerAsColNamesMetaType(new Expr.Tuple(keyExprs));
    Expr scalarGlobal = visitor.lowerAsMetaType(new Expr.Tuple(scalarExprs));
    return new Expr.Call(
        "bodo.libs.bodosql_array_kernels.object_construct_keep_null",
        List.of(new Expr.Tuple(valExprs), keyGlobal, scalarGlobal),
        List.of());
  }

  /**
   * Function that return the necessary generated code for a JSON function call in the hashmap
   * defined above.
   *
   * @param fnName The name of the function
   * @param operands The arguments to the function
   * @return The function call expression
   */
  public static Expr visitJsonFunc(String fnName, List<Expr> operands) {

    if (!(jsonFnMap.containsKey(fnName))) {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }

    return new Expr.Call(jsonFnMap.get(fnName), operands);
  }
}
