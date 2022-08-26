package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import com.bodosql.calcite.application.RexNodeVisitorInfo;
import java.util.List;

public class RegexpCodeGen {

  /**
   * Extracts the name from a specific argument in a list of operands, or a default value if there
   * are not enough operands.
   *
   * @param operandsInfo the list of operands
   * @param i the operand whose name is to be extracted
   * @param defaultValue the value to return if there are not enough operands
   * @return either the name of operand i, or the default value
   */
  public static String getNameWithDefault(
      List<RexNodeVisitorInfo> operandsInfo, int i, String defaultValue) {
    if (i >= operandsInfo.size()) {
      return defaultValue;
    }
    return operandsInfo.get(i).getName();
  }

  /**
   * Extracts the code from a specific argument in a list of operands, or a default value if there
   * are not enough operands.
   *
   * @param operandsInfo the list of operands
   * @param i the operand whose code is to be extracted
   * @param defaultValue the value to return if there are not enough operands
   * @return either the code of operand i, or the default value
   */
  public static String getCodeWithDefault(
      List<RexNodeVisitorInfo> operandsInfo, int i, String defaultValue) {
    if (i >= operandsInfo.size()) {
      return defaultValue;
    }
    return operandsInfo.get(i).getExprCode();
  }

  /**
   * Function that returns the rexInfo for a REGEXP_LIKE Function call
   *
   * @param operandsInfo the information about the 2-3 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateRegexpLikeInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    RexNodeVisitorInfo source = operandsInfo.get(0);
    RexNodeVisitorInfo pattern = operandsInfo.get(1);

    name.append("REGEXP_LIKE(");
    name.append(source.getName()).append(", ");
    name.append(pattern.getName()).append(", ");
    name.append(getNameWithDefault(operandsInfo, 2, makeQuoted(""))).append(")");

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_like(");
    expr_code.append(source.getExprCode()).append(", ");
    expr_code.append(pattern.getExprCode()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, makeQuoted(""))).append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_COUNT Function call
   *
   * @param operandsInfo the information about the 2-4 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateRegexpCountInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    RexNodeVisitorInfo source = operandsInfo.get(0);
    RexNodeVisitorInfo pattern = operandsInfo.get(1);

    name.append("REGEXP_COUNT(");
    name.append(source.getName()).append(", ");
    name.append(pattern.getName()).append(", ");
    name.append(getNameWithDefault(operandsInfo, 2, "1")).append(",");
    name.append(getNameWithDefault(operandsInfo, 3, makeQuoted(""))).append(")");

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_count(");
    expr_code.append(source.getExprCode()).append(", ");
    expr_code.append(pattern.getExprCode()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, makeQuoted(""))).append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_REPLACE Function call
   *
   * @param operandsInfo the information about the 2-6 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateRegexpReplaceInfo(
      List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    RexNodeVisitorInfo source = operandsInfo.get(0);
    RexNodeVisitorInfo pattern = operandsInfo.get(1);

    name.append("REGEXP_REPLACE(");
    name.append(source.getName()).append(", ");
    name.append(pattern.getName()).append(", ");
    name.append(getNameWithDefault(operandsInfo, 2, makeQuoted(""))).append(",");
    name.append(getNameWithDefault(operandsInfo, 3, "1")).append(",");
    name.append(getNameWithDefault(operandsInfo, 4, "0")).append(",");
    name.append(getNameWithDefault(operandsInfo, 5, makeQuoted(""))).append(")");

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_replace(");
    expr_code.append(source.getExprCode()).append(", ");
    expr_code.append(pattern.getExprCode()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, makeQuoted(""))).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 4, "0")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 5, makeQuoted(""))).append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_SUBSTR Function call
   *
   * @param operandsInfo the information about the 2-6 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateRegexpSubstrInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    RexNodeVisitorInfo source = operandsInfo.get(0);
    RexNodeVisitorInfo pattern = operandsInfo.get(1);

    name.append("REGEXP_SUBSTR(");
    name.append(source.getName()).append(", ");
    name.append(pattern.getName()).append(", ");
    name.append(getNameWithDefault(operandsInfo, 2, "1")).append(",");
    name.append(getNameWithDefault(operandsInfo, 3, "1")).append(",");
    // If a group was provided, ensure that the flags contain 'e' since providing
    // a group number implies that extraction should be done
    if (operandsInfo.size() == 6) {
      name.append(operandsInfo.get(4).getName() + "+" + makeQuoted("e")).append(",");
    } else {
      name.append(getNameWithDefault(operandsInfo, 4, makeQuoted(""))).append(",");
    }
    name.append(getNameWithDefault(operandsInfo, 5, "1")).append(")");

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_substr(");
    expr_code.append(source.getExprCode()).append(", ");
    expr_code.append(pattern.getExprCode()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, "1")).append(",");
    // If a group was provided, ensure that the flags contain 'e'
    if (operandsInfo.size() == 6) {
      expr_code.append(operandsInfo.get(4).getExprCode() + "+" + makeQuoted("e")).append(",");
    } else {
      expr_code.append(getNameWithDefault(operandsInfo, 4, makeQuoted(""))).append(",");
    }
    expr_code.append(getCodeWithDefault(operandsInfo, 5, "1")).append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }

  /**
   * Function that returns the rexInfo for a REGEXP_INSTR Function call
   *
   * @param operandsInfo the information about the 2-7 arguments
   * @return The RexNodeVisitorInfo corresponding to the function call
   */
  public static RexNodeVisitorInfo generateRegexpInstrInfo(List<RexNodeVisitorInfo> operandsInfo) {
    StringBuilder name = new StringBuilder();
    StringBuilder expr_code = new StringBuilder();

    RexNodeVisitorInfo source = operandsInfo.get(0);
    RexNodeVisitorInfo pattern = operandsInfo.get(1);

    name.append("REGEXP_INSTR(");
    name.append(source.getName()).append(", ");
    name.append(pattern.getName()).append(", ");
    name.append(getNameWithDefault(operandsInfo, 2, "1")).append(",");
    name.append(getNameWithDefault(operandsInfo, 3, "1")).append(",");
    name.append(getNameWithDefault(operandsInfo, 4, "0")).append(",");
    // If a group was provided, ensure that the flags contain 'e' since providing
    // a group number implies that extraction should be done
    if (operandsInfo.size() == 7) {
      name.append(operandsInfo.get(5).getName() + "+" + makeQuoted("e")).append(",");
    } else {
      name.append(getNameWithDefault(operandsInfo, 5, makeQuoted(""))).append(",");
    }
    name.append(getNameWithDefault(operandsInfo, 6, "1")).append(")");

    expr_code.append("bodo.libs.bodosql_array_kernels.regexp_instr(");
    expr_code.append(source.getExprCode()).append(", ");
    expr_code.append(pattern.getExprCode()).append(", ");
    expr_code.append(getCodeWithDefault(operandsInfo, 2, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 3, "1")).append(",");
    expr_code.append(getCodeWithDefault(operandsInfo, 4, "0")).append(",");
    // If a group was provided, ensure that the flags contain 'e'
    if (operandsInfo.size() == 7) {
      expr_code.append(operandsInfo.get(5).getExprCode() + "+" + makeQuoted("e")).append(",");
    } else {
      expr_code.append(getNameWithDefault(operandsInfo, 5, makeQuoted(""))).append(",");
    }
    expr_code.append(getCodeWithDefault(operandsInfo, 6, "1")).append(")");

    return new RexNodeVisitorInfo(name.toString(), expr_code.toString());
  }
}
