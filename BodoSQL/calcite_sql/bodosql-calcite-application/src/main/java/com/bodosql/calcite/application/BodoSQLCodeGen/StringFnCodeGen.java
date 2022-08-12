package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.generateBinOpCode;
import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.sql.SqlOperator;

public class StringFnCodeGen {

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call for the scalar case in the form of FN(scalar_expr). If the function has
  // more then one argument,
  // the mapping is only valid of all of the arguments are scalars
  // IE SQLFN(x1, x2, x3) => FN(x1, x2, x3)
  // EX CHAR(x) => chr(x)
  // OR CHAR(C) => C.apply(chr)
  static HashMap<String, String> equivalentFnMapScalars;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a pandas method call for the column case. If the function has more then one argument,
  // the mapping is valid if the first argument is a column, and all subsequent values are scalars
  // IE FN(C, x1, x2, x3) => C.FNAME(x1, x2, x3...)
  // EX UPPER(C) => C.str.upper()
  static HashMap<String, String> equivalentPandasMethodMapColumns;

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call where any of the arguments can be scalars or vectors.
  // IE SQLFN(C1, s1, C2, s2) => FN(C1, s1, C2, s2)
  // EX LPAD(A, 10, S) => bodo.libs.bodosql_array_kernels.lpad(A, 10, S)
  static HashMap<String, String> equivalentFnMapBroadcast;

  static {
    equivalentFnMapScalars = new HashMap<>();
    equivalentPandasMethodMapColumns = new HashMap<>();
    equivalentFnMapBroadcast = new HashMap<>();

    equivalentFnMapScalars.put("CHAR_LENGTH", "bodosql.libs.generated_lib.sql_null_checking_len");
    equivalentPandasMethodMapColumns.put("CHAR_LENGTH", "str.len");
    equivalentFnMapScalars.put("LENGTH", "bodosql.libs.generated_lib.sql_null_checking_len");
    equivalentPandasMethodMapColumns.put("LENGTH", "str.len");
    equivalentFnMapScalars.put(
        "CHARACTER_LENGTH", "bodosql.libs.generated_lib.sql_null_checking_len");
    equivalentPandasMethodMapColumns.put("CHARACTER_LENGTH", "str.len");
    equivalentFnMapScalars.put("LCASE", "bodosql.libs.generated_lib.sql_null_checking_lower");
    equivalentFnMapScalars.put("LOWER", "bodosql.libs.generated_lib.sql_null_checking_lower");
    equivalentPandasMethodMapColumns.put("LCASE", "str.lower");
    equivalentPandasMethodMapColumns.put("LOWER", "str.lower");
    equivalentFnMapScalars.put("UCASE", "bodosql.libs.generated_lib.sql_null_checking_upper");
    equivalentFnMapScalars.put("UPPER", "bodosql.libs.generated_lib.sql_null_checking_upper");
    equivalentPandasMethodMapColumns.put("UCASE", "str.upper");
    equivalentPandasMethodMapColumns.put("UPPER", "str.upper");
    equivalentFnMapBroadcast.put("LPAD", "bodo.libs.bodosql_array_kernels.lpad");
    equivalentFnMapBroadcast.put("RPAD", "bodo.libs.bodosql_array_kernels.rpad");
    equivalentFnMapBroadcast.put("LEFT", "bodo.libs.bodosql_array_kernels.left");
    equivalentFnMapBroadcast.put("RIGHT", "bodo.libs.bodosql_array_kernels.right");
    equivalentFnMapBroadcast.put("ORD", "bodo.libs.bodosql_array_kernels.ord_ascii");
    equivalentFnMapBroadcast.put("ASCII", "bodo.libs.bodosql_array_kernels.ord_ascii");
    equivalentFnMapBroadcast.put("CHAR", "bodo.libs.bodosql_array_kernels.char");
    equivalentFnMapBroadcast.put("CHR", "bodo.libs.bodosql_array_kernels.char");
    equivalentFnMapBroadcast.put("FORMAT", "bodo.libs.bodosql_array_kernels.format");
    equivalentFnMapBroadcast.put("REPEAT", "bodo.libs.bodosql_array_kernels.repeat");
    equivalentFnMapBroadcast.put("REVERSE", "bodo.libs.bodosql_array_kernels.reverse");
    equivalentFnMapBroadcast.put("REPLACE", "bodo.libs.bodosql_array_kernels.replace");
    equivalentFnMapBroadcast.put("SPACE", "bodo.libs.bodosql_array_kernels.space");
    equivalentFnMapBroadcast.put("STRCMP", "bodo.libs.bodosql_array_kernels.strcmp");
    equivalentFnMapBroadcast.put("INSTR", "bodo.libs.bodosql_array_kernels.instr");
    equivalentFnMapBroadcast.put("SUBSTRING", "bodo.libs.bodosql_array_kernels.substring");
    equivalentFnMapBroadcast.put("SUBSTR", "bodo.libs.bodosql_array_kernels.substring");
    equivalentFnMapBroadcast.put("MID", "bodo.libs.bodosql_array_kernels.substring");
    equivalentFnMapBroadcast.put(
        "SUBSTRING_INDEX", "bodo.libs.bodosql_array_kernels.substring_index");
    equivalentFnMapBroadcast.put("TRANSLATE3", "bodo.libs.bodosql_array_kernels.translate");
    equivalentFnMapBroadcast.put("SPLIT_PART", "bodo.libs.bodosql_array_kernels.split_part");
    equivalentFnMapBroadcast.put("STRTOK", "bodo.libs.bodosql_array_kernels.strtok");
  }

  /**
   * Helper function that handles codegen for Single argument string functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param arg1Name The name of arg1
   * @param isSingleRow boolean value that determines if this function call is taking place within
   *     an apply (or all the arguments are scalar)
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgStringFnInfo(
      String fnName, String arg1Expr, String arg1Name, boolean isSingleRow) {
    String new_fn_name = fnName + "(" + arg1Name + ")";

    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(new_fn_name, fn_expr + "(" + arg1Expr + ")");
      // Otherwise, either use the scalar implementation or the Series implementation
    } else if (isSingleRow) {
      if (equivalentFnMapScalars.containsKey(fnName)) {
        String scalar_fn_str = equivalentFnMapScalars.get(fnName);
        return new RexNodeVisitorInfo(new_fn_name, scalar_fn_str + "(" + arg1Expr + ")");
      }
    } else if (equivalentPandasMethodMapColumns.containsKey(fnName)) {
      String pandas_method = equivalentPandasMethodMapColumns.get(fnName);
      return new RexNodeVisitorInfo(new_fn_name, arg1Expr + "." + pandas_method + "()");
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for Two argument string functions
   *
   * @param fnName The name of the function
   * @param arg1Info The rexVisitorInfo of arg1
   * @param arg2Info The rexVisitorInfo of arg2
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getTwoArgStringFnInfo(
      String fnName, String inputVar, RexNodeVisitorInfo arg1Info, RexNodeVisitorInfo arg2Info) {

    String new_fn_name = fnName + "(" + arg1Info.getName() + "," + arg2Info.getName() + ")";

    String arg1Expr = arg1Info.getExprCode();
    String arg2Expr = arg2Info.getExprCode();

    // All of these functions should have a broadcasted BodoSQL array kernel
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(new_fn_name, fn_expr + "(" + arg1Expr + "," + arg2Expr + ")");
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Helper function that handles codegen for three argument string functions
   *
   * @param fnName The name of the function
   * @param arg1Info The rexVisitorInfo of arg1
   * @param arg2Info The rexVisitorInfo of arg2
   * @param arg3Info The rexVisitorInfo of arg3
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getThreeArgStringFnInfo(
      String fnName,
      RexNodeVisitorInfo arg1Info,
      RexNodeVisitorInfo arg2Info,
      RexNodeVisitorInfo arg3Info) {

    String new_fn_name =
        fnName
            + "("
            + arg1Info.getName()
            + ","
            + arg2Info.getName()
            + ", "
            + arg3Info.getName()
            + ")";

    String arg1Expr = arg1Info.getExprCode();
    String arg2Expr = arg2Info.getExprCode();
    String arg3Expr = arg3Info.getExprCode();

    // All of these functions should have a broadcasted BodoSQL array kernel
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(
          new_fn_name, fn_expr + "(" + arg1Expr + "," + arg2Expr + "," + arg3Expr + ")");
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Function that returns the rexInfo for the Concat Function Call.
   *
   * @param operandsInfo The rexInfo for all of the arguments to the Concat Call
   * @param exprTypes The exprType of each argument.
   * @param op the concat function SqlOperator. Must be passed in, so I can pass it to
   *     generateConcatWs
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @return The name generated that matches the Concat expression.
   */
  public static RexNodeVisitorInfo generateConcatFnInfo(
      List<RexNodeVisitorInfo> operandsInfo,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlOperator op,
      boolean isScalar) {
    assert op.getName() == "CONCAT";

    StringBuilder concatName = new StringBuilder();
    concatName.append("CONCAT(");
    // Iterate through the list of input operands, building the name
    for (RexNodeVisitorInfo curOpInfo : operandsInfo) {
      concatName.append(curOpInfo.getName()).append(", ");
    }
    concatName.append(")");

    RexNodeVisitorInfo separatorInfo = new RexNodeVisitorInfo("", "");

    RexNodeVisitorInfo concatWSInfo =
        generateConcatWSFnInfo(
            separatorInfo, BodoSQLExprType.ExprType.SCALAR, operandsInfo, exprTypes, op, isScalar);

    return new RexNodeVisitorInfo(concatName.toString(), concatWSInfo.getExprCode());
  }

  /**
   * Function that returns the rexInfo for the Concat_ws Function Call.
   *
   * @param separatorInfo the rexInfo for the string used for the separator
   * @param operandsInfo The rexInfo for the list of string arguments to be concatenated
   * @param op The concat or concat_ws function SqlOperator. Must be passed in, so I can pass it to
   *     generateBinopCode.
   * @return The name generated that matches the Concat_ws expression.
   */
  public static RexNodeVisitorInfo generateConcatWSFnInfo(
      RexNodeVisitorInfo separatorInfo,
      BodoSQLExprType.ExprType separatorType,
      List<RexNodeVisitorInfo> operandsInfo,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlOperator op,
      boolean isScalar) {

    assert op.getName() == "CONCAT" || op.getName() == "CONCAT_WS";
    StringBuilder concatWsName = new StringBuilder();
    concatWsName.append("CONCAT_WS(");
    List<String> concatWsArgs = new ArrayList<>();
    List<BodoSQLExprType.ExprType> concatWsExprTypes = new ArrayList<>();

    // Iterate through the list of input operands, building the name and the args/types list to pass
    // to generateBinOpCode
    for (int i = 0; i < operandsInfo.size() - 1; i++) {
      RexNodeVisitorInfo curOpInfo = operandsInfo.get(i);
      concatWsName.append(curOpInfo.getName()).append(", ");
      concatWsArgs.add(curOpInfo.getExprCode());
      concatWsExprTypes.add(exprTypes.get(0));
      // Add the separator in between each argument
      if (!separatorInfo.getExprCode().equals("")) {
        concatWsName.append(separatorInfo.getName()).append(", ");
        concatWsArgs.add(separatorInfo.getExprCode());
        concatWsExprTypes.add(separatorType);
      }
    }
    // append the final value
    RexNodeVisitorInfo finalOpInfo = operandsInfo.get(operandsInfo.size() - 1);
    concatWsName.append(finalOpInfo.getName()).append(", ");
    concatWsExprTypes.add(exprTypes.get(operandsInfo.size() - 1));
    concatWsArgs.add(finalOpInfo.getExprCode());

    concatWsName.append(")");

    String concatExpression = generateBinOpCode(concatWsArgs, concatWsExprTypes, op, isScalar);
    return new RexNodeVisitorInfo(concatWsName.toString(), concatExpression);
  }

  /**
   * Function that returns the rexInfo for a INITCAP Function call
   *
   * @param operandsInfo the information about the 1-2 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateInitcapInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    int argCount = operandsInfo.size();
    if (!(1 <= argCount && argCount <= 2)) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to INITCAP");
    }

    name.append("INITCAP(");
    name.append(operandsInfo.get(0));
    name.append(", ");

    expr_code.append("bodo.libs.bodosql_array_kernels.initcap(");
    expr_code.append(operandsInfo.get(0).getExprCode());
    expr_code.append(", ");

    // If 1 arguments was provided, provide a default delimeter string
    if (argCount == 1) {
      name.append("' \\t\\n\\r\\f\\u000b!?@\\\"^#$&~_,.:;+-*%/|\\[](){}<>'");
      expr_code.append("' \\t\\n\\r\\f\\u000b!?@\\\"^#$&~_,.:;+-*%/|\\[](){}<>'");
      // Otherwise, extract the delimeter string argument
    } else {
      name.append(operandsInfo.get(1).getName());
      expr_code.append(operandsInfo.get(1).getExprCode());
    }

    name.append(")");
    expr_code.append(")");
    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }
  /*
   * Function that returns the rexInfo for an STRTOK Function Call.
   *
   * @param operandsInfo the information about the 1-3 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateStrtok(List<RexNodeVisitorInfo> operandsInfo) {
    int argCount = operandsInfo.size();

    if (!(1 <= argCount && argCount <= 3)) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to STRTOK");
    }

    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    name.append("STRTOK(");
    name.append(operandsInfo.get(0).getName());
    expr_code.append("bodo.libs.bodosql_array_kernels.strtok(");
    expr_code.append(operandsInfo.get(0).getExprCode());

    // If 1 argument is provided, use space as the delimeter
    if (argCount == 1) {
      name.append(", ' '");
      expr_code.append(", ' '");
      // Otherwise, extract the delimeter argument
    } else {
      name.append(", ");
      name.append(operandsInfo.get(1).getName());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(1).getExprCode());
    }

    // If 1-2 arguments are provided, use 1 as the part
    if (argCount < 3) {
      name.append(", 1");
      expr_code.append(", 1");
      // Otherwise, extract the part argument
    } else {
      name.append(", ");
      name.append(operandsInfo.get(2).getName());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(2).getExprCode());
    }

    name.append(")");
    expr_code.append(")");
    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for an EDITDISTANCE Function Call.
   *
   * @param operandsInfo the information about the two/three arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateEditdistance(List<RexNodeVisitorInfo> operandsInfo) {

    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    name.append("EDITDISTANCE(");

    if (operandsInfo.size() == 2) {
      name.append(operandsInfo.get(0).getName());
      name.append(", ");
      name.append(operandsInfo.get(1).getName());
      expr_code.append("bodo.libs.bodosql_array_kernels.editdistance_no_max(");
      expr_code.append(operandsInfo.get(0).getExprCode());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(1).getExprCode());
    } else if (operandsInfo.size() == 3) {
      name.append(operandsInfo.get(0).getName());
      name.append(", ");
      name.append(operandsInfo.get(1).getName());
      name.append(", ");
      name.append(operandsInfo.get(2).getName());
      expr_code.append("bodo.libs.bodosql_array_kernels.editdistance_with_max(");
      expr_code.append(operandsInfo.get(0).getExprCode());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(1).getExprCode());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(2).getExprCode());
    } else {
      throw new BodoSQLCodegenException(
          "Error, invalid number of arguments passed to EDITDISTANCE");
    }
    name.append(")");
    expr_code.append(")");
    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo of a L/R/Trim function Call. Eventually, this may need to be
   * extended to handle trimming non whitespace characters
   *
   * @param flagInfo The rexInfo of the argument that determines from which sides we trim whitespace
   * @param stringToBeTrimmed The rexInfo of the string to be trimmed
   * @return The rexVisitorInfo for the trim call
   */
  public static RexNodeVisitorInfo generateTrimFnInfo(
      RexNodeVisitorInfo flagInfo,
      RexNodeVisitorInfo stringToBeTrimmed,
      BodoSQLExprType.ExprType exprType,
      boolean isSingleRow) {
    String trimFn;
    String trimName;

    switch (flagInfo.getExprCode()) {
      case "BOTH":
        trimFn = "strip";
        trimName = "TRIM";
        break;
      case "LEADING":
        trimFn = "lstrip";
        trimName = "LTRIM";
        break;
      case "TRAILING":
        trimFn = "rstrip";
        trimName = "RTRIM";
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Flag: " + flagInfo.getExprCode() + " not supported for Trim");
    }

    String outputName = trimName + "(" + stringToBeTrimmed.getName() + ")";
    String outputExpr;
    if (isSingleRow || exprType == exprType.SCALAR) {
      outputExpr =
          "bodosql.libs.generated_lib.sql_null_checking_"
              + trimFn
              + "("
              + stringToBeTrimmed.getExprCode()
              + ", "
              + "\" \")";
    } else {
      outputExpr = stringToBeTrimmed.getExprCode() + ".str." + trimFn + "(\" \")";
    }
    return new RexNodeVisitorInfo(outputName, outputExpr);
  }
}
