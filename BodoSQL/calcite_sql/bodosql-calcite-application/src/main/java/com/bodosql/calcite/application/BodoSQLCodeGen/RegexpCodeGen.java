package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.ir.Expr;
import java.util.ArrayList;
import java.util.List;

public class RegexpCodeGen {

  /**
   * Extracts the code from a specific argument in a list of operands, or a default value if there
   * are not enough operands.
   *
   * @param operandsInfo the list of operands
   * @param i the operand whose code is to be extracted
   * @param defaultValue the value to return if there are not enough operands
   * @return either the code of operand i, or the default value
   */
  public static String getCodeWithDefault(List<Expr> operandsInfo, int i, String defaultValue) {
    if (i >= operandsInfo.size()) {
      return defaultValue;
    }
    return operandsInfo.get(i).emit();
  }

  /**
   * Function that returns the rexInfo for a REGEXP_LIKE Function call
   *
   * @param operandsInfo the information about the 2-3 arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpLikeInfo(List<Expr> operandsInfo) {
    StringBuilder expr_code = new StringBuilder();

    Expr source = operandsInfo.get(0);
    Expr pattern = operandsInfo.get(1);

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_like(");
    expr_code.append(source.emit()).append(", ");
    expr_code.append(pattern.emit()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, makeQuoted(""))).append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_COUNT Function call
   *
   * @param operandsInfo the information about the 2-4 arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpCountInfo(List<Expr> operandsInfo) {
    StringBuilder expr_code = new StringBuilder();

    Expr source = operandsInfo.get(0);
    Expr pattern = operandsInfo.get(1);

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_count(");
    expr_code.append(source.emit()).append(", ");
    expr_code.append(pattern.emit()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, makeQuoted(""))).append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_REPLACE Function call
   *
   * @param operandsInfo the information about the 2-6 arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpReplaceInfo(List<Expr> operandsInfo) {
    StringBuilder expr_code = new StringBuilder();

    Expr source = operandsInfo.get(0);
    Expr pattern = operandsInfo.get(1);

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_replace(");
    expr_code.append(source.emit()).append(", ");
    expr_code.append(pattern.emit()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, makeQuoted(""))).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 4, "0")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 5, makeQuoted(""))).append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_SUBSTR Function call
   *
   * @param operandsInfo the information about the 2-6 arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpSubstrInfo(List<Expr> operandsInfo) {
    List<String> funcArgs = new ArrayList<>();
    // funcArgs is the list of args to REGEXP_SUBSTR that will be
    // modified over the course of this function.
    // Below are the default values.
    funcArgs.add(operandsInfo.get(0).emit()); // source
    funcArgs.add(operandsInfo.get(1).emit()); // pattern
    funcArgs.add("1"); // position
    funcArgs.add("1"); // occurrence
    funcArgs.add(makeQuoted("c")); // Regexp parameters
    funcArgs.add("0"); // group number

    // Overwrite default arguments with user-defined ones
    for (int i = 2; i < operandsInfo.size(); i++) {
      Expr arg = operandsInfo.get(i);
      if (!arg.emit().equals("\"\"")) {
        funcArgs.set(i, arg.emit());
      }
    }

    int regexpParamsIndex = 4;
    int groupNumIndex = 5;

    String regexpParams = funcArgs.get(regexpParamsIndex);
    String groupNum = funcArgs.get(groupNumIndex);

    // Edge case: if regex parameters exist and the group
    // number is unspecified, <group_num> defaults to 1
    if (regexpParams.contains("e") && groupNum == "0") {
      funcArgs.set(groupNumIndex, "1");
    }

    // Edge case: if <group_num> is specified, Snowflake
    // allows extraction even if "e" is not present in the
    // regex parameters arg.
    if (groupNum != "0" && !regexpParams.contains("e")) {
      regexpParams = funcArgs.get(regexpParamsIndex);
      funcArgs.set(regexpParamsIndex, regexpParams + "+" + makeQuoted("e"));
    }

    StringBuilder expr_code = new StringBuilder();
    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_substr(");

    // Populate the output function call with the final arg values
    for (int i = 0; i < 5; i++) {
      expr_code.append(funcArgs.get(i)).append(", ");
    }
    expr_code.append(funcArgs.get(5));
    expr_code.append(")");

    return new Expr.Raw(expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_INSTR Function call
   *
   * @param operandsInfo the information about the 2-7 arguments
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpInstrInfo(List<Expr> operandsInfo) {
    List<String> funcArgs = new ArrayList<>();
    // funcArgs is the list of args to REGEXP_INSTR that will be
    // modified over the course of this function.
    // Below are the default values.
    funcArgs.add(operandsInfo.get(0).emit()); // source
    funcArgs.add(operandsInfo.get(1).emit()); // pattern
    funcArgs.add("1"); // position
    funcArgs.add("1"); // occurrence
    funcArgs.add("0"); // option
    funcArgs.add(makeQuoted("c")); // Regexp parameters
    funcArgs.add("0"); // group number

    // Overwrite default arguments with user-defined ones
    for (int i = 2; i < operandsInfo.size(); i++) {
      Expr arg = operandsInfo.get(i);
      if (!arg.emit().equals("\"\"")) {
        funcArgs.set(i, arg.emit());
      }
    }

    int regexpParamsIndex = 5;
    int groupNumIndex = 6;

    String regexpParams = funcArgs.get(regexpParamsIndex);
    String groupNum = funcArgs.get(groupNumIndex);

    // Edge case: if regex parameters exist and the group
    // number is unspecified, <group_num> defaults to 1
    if (regexpParams.contains("e") && groupNum == "0") {
      funcArgs.set(groupNumIndex, "1");
    }

    // Edge case: if <group_num> is specified, Snowflake
    // allows extraction even if "e" is not present in the
    // regex parameters arg.
    if (groupNum != "0" && !regexpParams.contains("e")) {
      regexpParams = funcArgs.get(regexpParamsIndex);
      funcArgs.set(regexpParamsIndex, regexpParams + "+" + makeQuoted("e"));
    }

    StringBuilder expr_code = new StringBuilder();
    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_instr(");

    for (int i = 0; i < 6; i++) {
      expr_code.append(funcArgs.get(i)).append(", ");
    }
    expr_code.append(funcArgs.get(6));
    expr_code.append(")");

    return new Expr.Raw(expr_code.toString());
  }
}
