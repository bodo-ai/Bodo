package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;

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
}
