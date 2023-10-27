package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import java.util.HashMap;
import java.util.List;

public class JsonCodeGen {
  static HashMap<String, String> jsonFnMap;

  static {
    jsonFnMap = new HashMap<>();
    jsonFnMap.put(
        "JSON_EXTRACT_PATH_TEXT", "bodo.libs.bodosql_array_kernels.json_extract_path_text");
    jsonFnMap.put("OBJECT_KEYS", "bodo.libs.bodosql_array_kernels.object_keys");
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
