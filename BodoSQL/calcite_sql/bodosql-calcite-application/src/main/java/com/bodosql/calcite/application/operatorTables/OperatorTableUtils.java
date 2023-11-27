package com.bodosql.calcite.application.operatorTables;

import java.util.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.type.BodoOperandTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.util.Pair;
import org.apache.commons.lang3.StringUtils;

public class OperatorTableUtils {

  /**
   * Helper function to determine output type nullability for functions whose output can only be
   * null if there is a null input.
   *
   * @param operandTypes List of input types. The output is nullable if any input is nullable.
   * @return Is the output nullable.
   */
  public static boolean isOutputNullableCompile(List<RelDataType> operandTypes) {
    for (RelDataType operandType : operandTypes) {
      if (operandType.isNullable()) {
        return true;
      }
    }
    return false;
  }

  /**
   * Creates an OperandTypeChecker for a function that has several arguments with consistent types,
   * but some of them are optional. Calling the function with arguments (3, A, B, C, D, E) is the
   * same as creating an OR of the families (A, B, C), (A, B, C, D), (A, B, C, D, E)
   *
   * @param min the minimum number of arguments required for the function call
   * @param families the types for each argument when they are provided
   * @return an OperandTypeChecker with the specs mentioned above
   */
  public static SqlOperandTypeChecker argumentRange(int min, SqlTypeFamily... families) {
    assert min <= families.length;
    List<SqlTypeFamily> familyList = new ArrayList<SqlTypeFamily>();
    for (int i = 0; i < min; i++) {
      familyList.add(families[i]);
    }
    SqlOperandTypeChecker rule = OperandTypes.family(familyList);
    for (int i = min; i < families.length; i++) {
      familyList.add(families[i]);
      rule = OperandTypes.or(rule, OperandTypes.family(familyList));
    }
    return rule;
  }

  public static String genSignature(String fnName, List<String> argTypes) {
    return fnName + "(" + StringUtils.join(argTypes, ", ") + ")";
  }

  /**
   * Creates an OperandTypeChecker for a function that has several arguments but some of them are
   * optional. Calling the function with arguments (3, A, B, C, D, E) is the same as creating an OR
   * of Operand.sequence(A, B, C), Operand.sequence(A, B, C, D), Operand.sequence(A, B, C, D, E)
   *
   * @param min The minimum number of arguments required for the function call
   * @param fnName The name of the function
   * @param arguments A list of pairs containing the rule to check if a given argument is valid, and
   *     the representation of that argument for purposes of generating the function signature. IE:
   *     [ (SqlOperandTypes.POSITIVE_INTEGER_LITERAL, "POSITIVE INTEGER LITERAL") ]
   * @return an OperandTypeChecker with the specs mentioned above
   */
  public static SqlOperandTypeChecker argumentRangeExplicit(
      int min, String fnName, List<Pair<SqlSingleOperandTypeChecker, String>> arguments) {
    assert min <= arguments.size();

    // First, generate the
    ArrayList<String> argNames = new ArrayList<>();
    ArrayList<SqlSingleOperandTypeChecker> argRules = new ArrayList<>();

    for (int i = 0; i < min; i++) {
      argRules.add(arguments.get(i).left);
      argNames.add(arguments.get(i).right);
    }

    SqlOperandTypeChecker rule;
    // Since sequence requires more than one rule,
    // we need to have some special handling for the 0
    // non-optional argument case, and the 1 non-optional argument case
    switch (argRules.size()) {
      case 0:
        rule = OperandTypes.NILADIC;
        break;
      case 1:
        rule = argRules.get(0);
        break;
      default:
        rule = BodoOperandTypes.sequence(genSignature(fnName, argNames), argRules);
    }

    // Next, add all the optionals
    for (int i = min; i < arguments.size(); i++) {
      argRules.add(arguments.get(i).left);
      argNames.add(arguments.get(i).right);
      rule = rule.or(BodoOperandTypes.sequence(genSignature(fnName, argNames), argRules));
    }
    return rule;
  }
}
