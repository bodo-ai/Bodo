package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.HashMap;

public class JsonCodeGen {
  static HashMap<String, String> jsonFnMap;

  static {
    jsonFnMap = new HashMap<>();
    jsonFnMap.put(
        "JSON_EXTRACT_PATH_TEXT", "bodo.libs.bodosql_array_kernels.json_extract_path_text");
  }
  /**
   * Function that return the necessary generated code for a two argument JSON function call.
   *
   * @param arg0 The first argument to the function
   * @param arg1 The second argument to the function
   * @return
   */
  public static RexNodeVisitorInfo generateJsonTwoArgsInfo(
      String fnName, RexNodeVisitorInfo arg0, RexNodeVisitorInfo arg1) {

    if (!(jsonFnMap.containsKey(fnName))) {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }

    StringBuilder output = new StringBuilder();
    output
        .append(jsonFnMap.get(fnName))
        .append("(")
        .append(arg0.getExprCode())
        .append(", ")
        .append(arg1.getExprCode())
        .append(")");

    return new RexNodeVisitorInfo(output.toString());
  }
}
