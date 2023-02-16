package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.HashMap;
import java.util.List;

public class StringFnCodeGen {

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call where any of the arguments can be scalars or vectors.
  // IE SQLFN(C1, s1, C2, s2) => FN(C1, s1, C2, s2)
  // EX LPAD(A, 10, S) => bodo.libs.bodosql_array_kernels.lpad(A, 10, S)
  static HashMap<String, String> equivalentFnMapBroadcast;

  static {
    equivalentFnMapBroadcast = new HashMap<>();
    equivalentFnMapBroadcast.put("CHAR_LENGTH", "bodo.libs.bodosql_array_kernels.length");
    equivalentFnMapBroadcast.put("LENGTH", "bodo.libs.bodosql_array_kernels.length");
    equivalentFnMapBroadcast.put("LEN", "bodo.libs.bodosql_array_kernels.length");
    equivalentFnMapBroadcast.put("CHARACTER_LENGTH", "bodo.libs.bodosql_array_kernels.length");
    equivalentFnMapBroadcast.put("LCASE", "bodo.libs.bodosql_array_kernels.lower");
    equivalentFnMapBroadcast.put("LOWER", "bodo.libs.bodosql_array_kernels.lower");
    equivalentFnMapBroadcast.put("UCASE", "bodo.libs.bodosql_array_kernels.upper");
    equivalentFnMapBroadcast.put("UPPER", "bodo.libs.bodosql_array_kernels.upper");
    equivalentFnMapBroadcast.put("CONTAINS", "bodo.libs.bodosql_array_kernels.contains");
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
    equivalentFnMapBroadcast.put(
        "RTRIMMED_LENGTH", "bodo.libs.bodosql_array_kernels.rtrimmed_length");
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
    equivalentFnMapBroadcast.put("STARTSWITH", "bodo.libs.bodosql_array_kernels.startswith");
    equivalentFnMapBroadcast.put("ENDSWITH", "bodo.libs.bodosql_array_kernels.endswith");
  }

  /**
   * Helper function that handles codegen for Single argument string functions
   *
   * @param fnName The name of the function
   * @param arg1Expr The string expression of arg1
   * @param fnName The name of arg1
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo getSingleArgStringFnInfo(String fnName, String arg1Expr) {
    // If the functions has a broadcasted array kernel, always use it
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(fn_expr + "(" + arg1Expr + ")");
      // Otherwise, either use the scalar implementation or the Series implementation
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

    String arg1Expr = arg1Info.getExprCode();
    String arg2Expr = arg2Info.getExprCode();

    // All of these functions should have a broadcasted BodoSQL array kernel
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(fn_expr + "(" + arg1Expr + "," + arg2Expr + ")");
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

    String arg1Expr = arg1Info.getExprCode();
    String arg2Expr = arg2Info.getExprCode();
    String arg3Expr = arg3Info.getExprCode();

    // All of these functions should have a broadcasted BodoSQL array kernel
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      String fn_expr = equivalentFnMapBroadcast.get(fnName);
      return new RexNodeVisitorInfo(
          fn_expr + "(" + arg1Expr + "," + arg2Expr + "," + arg3Expr + ")");
    }

    // If we made it here, something has gone very wrong
    throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
  }

  /**
   * Function that returns the rexInfo for the Concat Function Call.
   *
   * @param operandsInfo The rexInfo for all of the arguments to the Concat Call
   * @return The RexNodeVisitorInfo generated that matches the Concat expression.
   */
  public static RexNodeVisitorInfo generateConcatFnInfo(List<RexNodeVisitorInfo> operandsInfo) {
    RexNodeVisitorInfo separatorInfo = new RexNodeVisitorInfo(makeQuoted(""));

    RexNodeVisitorInfo concatWSInfo = generateConcatWSFnInfo(separatorInfo, operandsInfo);

    return new RexNodeVisitorInfo(concatWSInfo.getExprCode());
  }

  /**
   * Function that returns the rexInfo for the Concat_ws Function Call.
   *
   * @param separatorInfo the rexInfo for the string used for the separator
   * @param operandsInfo The rexInfo for the list of string arguments to be concatenated
   * @return The RexNodeVisitorInfo generated that matches the Concat_ws expression.
   */
  public static RexNodeVisitorInfo generateConcatWSFnInfo(
      RexNodeVisitorInfo separatorInfo, List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder concatCodeGen = new StringBuilder();
    concatCodeGen.append("bodo.libs.bodosql_array_kernels.concat_ws((");

    // Iterate through the list of input operands, building the name and the args/types list to pass
    // to generateBinOpCode
    for (int i = 0; i < operandsInfo.size(); i++) {
      RexNodeVisitorInfo curOpInfo = operandsInfo.get(i);
      concatCodeGen.append(curOpInfo.getExprCode()).append(", ");
    }
    concatCodeGen.append("), ").append(separatorInfo.getExprCode()).append(")");
    return new RexNodeVisitorInfo(concatCodeGen.toString());
  }

  /**
   * Function that returns the rexInfo for a INITCAP Function call
   *
   * @param operandsInfo the information about the 1-2 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateInitcapInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder expr_code = new StringBuilder();

    int argCount = operandsInfo.size();
    if (!(1 <= argCount && argCount <= 2)) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to INITCAP");
    }

    expr_code.append("bodo.libs.bodosql_array_kernels.initcap(");
    expr_code.append(operandsInfo.get(0).getExprCode());
    expr_code.append(", ");

    // If 1 arguments was provided, provide a default delimeter string
    if (argCount == 1) {
      expr_code.append("' \\t\\n\\r\\f\\u000b!?@\\\"^#$&~_,.:;+-*%/|\\[](){}<>'");
      // Otherwise, extract the delimeter string argument
    } else {
      expr_code.append(operandsInfo.get(1).getExprCode());
    }

    expr_code.append(")");
    return new RexNodeVisitorInfo(expr_code.toString());
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

    StringBuilder expr_code = new StringBuilder();

    expr_code.append("bodo.libs.bodosql_array_kernels.strtok(");
    expr_code.append(operandsInfo.get(0).getExprCode());

    // If 1 argument is provided, use space as the delimeter
    if (argCount == 1) {
      expr_code.append(", ' '");
      // Otherwise, extract the delimeter argument
    } else {
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(1).getExprCode());
    }

    // If 1-2 arguments are provided, use 1 as the part
    if (argCount < 3) {
      expr_code.append(", 1");
      // Otherwise, extract the part argument
    } else {
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(2).getExprCode());
    }

    expr_code.append(")");
    return new RexNodeVisitorInfo(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for an EDITDISTANCE Function Call.
   *
   * @param operandsInfo the information about the two/three arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateEditdistance(List<RexNodeVisitorInfo> operandsInfo) {

    StringBuilder expr_code = new StringBuilder();

    if (operandsInfo.size() == 2) {
      expr_code.append("bodo.libs.bodosql_array_kernels.editdistance_no_max(");
      expr_code.append(operandsInfo.get(0).getExprCode());
      expr_code.append(", ");
      expr_code.append(operandsInfo.get(1).getExprCode());
    } else if (operandsInfo.size() == 3) {
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
    expr_code.append(")");
    return new RexNodeVisitorInfo(expr_code.toString());
  }

  /*
   * Function that returns the rexInfo for an INSERT Function Call.
   *
   * @param operandsInfo the information about the 4 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateInsert(List<RexNodeVisitorInfo> operandsInfo) {
    int argCount = operandsInfo.size();

    if (argCount != 4) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to INSERT");
    }

    StringBuilder expr_code = new StringBuilder();

    expr_code.append("bodo.libs.bodosql_array_kernels.insert(");

    for (int i = 0; i < 4; i++) {
      expr_code.append(operandsInfo.get(i).getExprCode());
      if (i < 3) {
        expr_code.append(", ");
      }
    }

    expr_code.append(")");
    return new RexNodeVisitorInfo(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a POSITION/CHARINDEX Function Call.
   *
   * @param operandsInfo the information about the two/three arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generatePosition(List<RexNodeVisitorInfo> operandsInfo) {

    if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 3)) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to POSITION");
    }

    StringBuilder expr_code = new StringBuilder();

    expr_code.append("bodo.libs.bodosql_array_kernels.position(");
    expr_code.append(operandsInfo.get(0).getExprCode());
    expr_code.append(", ");
    expr_code.append(operandsInfo.get(1).getExprCode());
    expr_code.append(", ");

    if (operandsInfo.size() == 3) {
      expr_code.append(operandsInfo.get(2).getExprCode());

      // If 2 arguments are provided, the default value for the start position
      // is 1
    } else {
      expr_code.append("1");
    }

    expr_code.append(")");
    return new RexNodeVisitorInfo(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo of a L/R/Trim function Call.
   * Extended to handle trimming non whitespace characters
   *
   * @param trimName The argument that determines from which sides we trim characters
   * @param stringToBeTrimmed The rexInfo of the string to be trimmed
   * @param charactersToBeTrimmed The characters to trimmed from the string
   * @return The rexVisitorInfo for the trim call
   */
  public static RexNodeVisitorInfo generateTrimFnInfo(
      String trimName,
      RexNodeVisitorInfo stringToBeTrimmed,
      RexNodeVisitorInfo charactersToBeTrimmed) {

    String outputExpr =
        String.format(
                "bodo.libs.bodosql_array_kernels.%s(%s, %s)",
                trimName.toLowerCase(), stringToBeTrimmed.getExprCode(),
                charactersToBeTrimmed.getExprCode());
    return new RexNodeVisitorInfo(outputExpr);
  }
}
