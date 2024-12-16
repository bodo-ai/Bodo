package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import kotlin.Pair;

public class JsonCodeGen {
  static HashMap<String, String> jsonFnMap;

  static {
    jsonFnMap = new HashMap<>();
    jsonFnMap.put("JSON_EXTRACT_PATH_TEXT", "json_extract_path_text");
    jsonFnMap.put("GET_PATH", "get_path");
    jsonFnMap.put("OBJECT_KEYS", "object_keys");
  }

  /**
   * Function that return the necessary generated code for the JSON function
   * OBJECT_CONSTRUCT_KEEP_NULL or OBJECT_CONSTRUCT.
   *
   * @return The Expression to calculate the function call.
   */
  public static Expr getObjectConstructKeepNullCode(
      String fnName,
      List<Expr.StringLiteral> keys,
      List<Expr> values,
      List<Expr.BooleanLiteral> scalars,
      BodoCodeGenVisitor visitor) {
    Expr keyGlobal = visitor.lowerAsColNamesMetaType(new Expr.Tuple(keys));
    Expr scalarGlobal = visitor.lowerAsMetaType(new Expr.Tuple(scalars));
    return ExprKt.bodoSQLKernel(
        fnName.toLowerCase(Locale.ROOT),
        List.of(new Expr.Tuple(values), keyGlobal, scalarGlobal),
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
  public static Expr visitJsonFunc(String fnName, List<Expr> operands, List<Boolean> arg_scalars) {

    if (!(jsonFnMap.containsKey(fnName))) {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }

    List<Pair<String, Expr>> kwargs = List.of();
    if (fnName.equals("GET_PATH")) {
      // TODO(aneesh) Ideally we'd do this for all methods, but that requires some discussion on how
      // to make this more generic
      kwargs = new ArrayList<>();
      kwargs.add(new Pair<>("is_scalar", new Expr.BooleanLiteral(arg_scalars.get(0))));
    }
    return ExprKt.bodoSQLKernel(jsonFnMap.get(fnName), operands, kwargs);
  }

  /**
   * Helper function that Handles the codegen for indexing operations into arrays/maps/variants.
   * This also handles the alternate GET(val, idx) syntax.
   *
   * @param arrayScalar Is the input array/map scalar
   * @param indexScalar Is the input index scalar
   * @param operands The inputs to the operand as Codegen Exprs
   * @return A codegen Expr for the indexing/GET operation
   */
  public static Expr visitGetOp(boolean arrayScalar, boolean indexScalar, List<Expr> operands) {
    assert operands.size() == 2;

    kotlin.Pair isScalarArrArg =
        new kotlin.Pair("is_scalar_arr", new Expr.BooleanLiteral(arrayScalar));

    kotlin.Pair isScalarIndArg =
        new kotlin.Pair("is_scalar_idx", new Expr.BooleanLiteral(indexScalar));
    List<Pair<String, Expr>> namedArgs = List.of(isScalarArrArg, isScalarIndArg);
    return ExprKt.bodoSQLKernel("arr_get", operands, namedArgs);
  }

  public static Expr visitGetIgnoreCaseOp(List<Expr> operands, List<Boolean> arg_scalars) {
    Boolean mapScalar = arg_scalars.get(0);
    Boolean indexScalar = arg_scalars.get(1);

    kotlin.Pair isScalarArrArg =
        new kotlin.Pair("is_scalar_arr", new Expr.BooleanLiteral(mapScalar));

    kotlin.Pair isScalarIndArg =
        new kotlin.Pair("is_scalar_idx", new Expr.BooleanLiteral(indexScalar));

    List<Pair<String, Expr>> namedArgs = List.of(isScalarArrArg, isScalarIndArg);

    return ExprKt.bodoSQLKernel("get_ignore_case", operands, namedArgs);
  }
}
