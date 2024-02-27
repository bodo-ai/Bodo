package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.List;
import kotlin.Pair;

public class RegexpCodeGen {

  /**
   * Extracts the code from a specific argument in a list of operands, or a default value if there
   * are not enough operands.
   *
   * @param operands the list of operands
   * @param i the operand whose code is to be extracted
   * @param defaultValue the value to return if there are not enough operands
   * @return either the code of operand i, or the default value
   */
  public static Expr getCodeWithDefault(List<Expr> operands, int i, Expr defaultValue) {
    if (i >= operands.size()) {
      return defaultValue;
    }
    return operands.get(i);
  }

  /**
   * Function that returns the rexInfo for a REGEXP_LIKE Function call
   *
   * @param operands the information about the 2-3 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpLikeInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    args.add(operands.get(0));
    args.add(operands.get(1));
    args.add(getCodeWithDefault(operands, 2, new Expr.StringLiteral("")));
    return ExprKt.bodoSQLKernel("regexp_like", args, streamingNamedArgs);
  }

  /**
   * Function that returns the Code generated for a REGEXP_COUNT Function call
   *
   * @param operands the information about the 2-4 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpCountInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    args.add(operands.get(0));
    args.add(operands.get(1));
    args.add(getCodeWithDefault(operands, 2, Expr.Companion.getOne()));
    args.add(getCodeWithDefault(operands, 3, new Expr.StringLiteral("")));
    return ExprKt.bodoSQLKernel("regexp_count", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for a REGEXP_REPLACE Function call
   *
   * @param operands the information about the 2-6 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpReplaceInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    args.add(operands.get(0));
    args.add(operands.get(1));
    args.add(getCodeWithDefault(operands, 2, new Expr.StringLiteral("")));
    args.add(getCodeWithDefault(operands, 3, Expr.Companion.getOne()));
    args.add(getCodeWithDefault(operands, 4, Expr.Companion.getZero()));
    args.add(getCodeWithDefault(operands, 5, new Expr.StringLiteral("")));
    return ExprKt.bodoSQLKernel("regexp_replace", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for a REGEXP_SUBSTR Function call
   *
   * @param operands the information about the 2-6 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpSubstrInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    args.add(operands.get(0));
    args.add(operands.get(1));
    args.add(getCodeWithDefault(operands, 2, Expr.Companion.getOne()));
    args.add(getCodeWithDefault(operands, 3, Expr.Companion.getOne()));
    args.add(getCodeWithDefault(operands, 4, new Expr.StringLiteral("c")));
    args.add(getCodeWithDefault(operands, 5, Expr.Companion.getZero()));

    int regexpParamsIndex = 4;
    int groupNumIndex = 5;

    String regexpParams = args.get(regexpParamsIndex).emit();
    String groupNum = args.get(groupNumIndex).emit();

    // Edge case: if regex parameters exist and the group
    // number is unspecified, <group_num> defaults to 1
    if (regexpParams.contains("e") && groupNum.equals("0")) {
      args.set(groupNumIndex, Expr.Companion.getOne());
    }

    // Edge case: if <group_num> is specified, Snowflake
    // allows extraction even if "e" is not present in the
    // regex parameters arg.
    if (!groupNum.equals("0") && !regexpParams.contains("e")) {
      args.set(
          regexpParamsIndex,
          new Expr.Binary("+", args.get(regexpParamsIndex), new Expr.StringLiteral("e")));
    }
    return ExprKt.bodoSQLKernel("regexp_substr", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for a REGEXP_INSTR Function call
   *
   * @param operands the information about the 2-7 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateRegexpInstrInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {

    List<Expr> args = new ArrayList<>();
    args.add(operands.get(0)); // source
    args.add(operands.get(1)); // pattern
    args.add(getCodeWithDefault(operands, 2, Expr.Companion.getOne())); // position
    args.add(getCodeWithDefault(operands, 3, Expr.Companion.getOne())); // occurrence
    args.add(getCodeWithDefault(operands, 4, Expr.Companion.getZero())); // option
    args.add(getCodeWithDefault(operands, 5, new Expr.StringLiteral("c"))); // Regex parameters
    args.add(getCodeWithDefault(operands, 6, Expr.Companion.getZero())); // group number

    int regexpParamsIndex = 5;
    int groupNumIndex = 6;

    String regexpParams = args.get(regexpParamsIndex).emit();
    String groupNum = args.get(groupNumIndex).emit();

    // Edge case: if regex parameters exist and the group
    // number is unspecified, <group_num> defaults to 1
    if (regexpParams.contains("e") && groupNum.equals("0")) {
      args.set(groupNumIndex, Expr.Companion.getOne());
    }

    // Edge case: if <group_num> is specified, Snowflake
    // allows extraction even if "e" is not present in the
    // regex parameters arg.
    if (!groupNum.equals("0") && !regexpParams.contains("e")) {
      args.set(
          regexpParamsIndex,
          new Expr.Binary("+", args.get(regexpParamsIndex), new Expr.StringLiteral("e")));
    }
    return ExprKt.bodoSQLKernel("regexp_instr", args, streamingNamedArgs);
  }
}
