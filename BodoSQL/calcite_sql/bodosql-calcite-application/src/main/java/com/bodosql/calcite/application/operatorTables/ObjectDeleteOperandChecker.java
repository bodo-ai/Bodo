package com.bodosql.calcite.application.operatorTables;

import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;

public class ObjectDeleteOperandChecker extends SameOperandTypeChecker {
  /**
   * Parameter type-checking strategy for the OBJECT_DELETE function, which has the following rules:
   *
   * <p>- There must be at least 2 arguments - The first argument must be a JSON type - All
   * subsequent arguments must be strings
   */
  public ObjectDeleteOperandChecker() {
    super(-1);
  }

  static ObjectDeleteOperandChecker INSTANCE;

  static {
    INSTANCE = new ObjectDeleteOperandChecker();
  }

  @Override
  protected boolean checkOperandTypesImpl(
      SqlOperatorBinding operatorBinding,
      boolean throwOnFailure,
      @Nullable SqlCallBinding callBinding) {
    if (throwOnFailure && callBinding == null) {
      throw new IllegalArgumentException(
          "callBinding must be non-null in case throwOnFailure=true");
    }
    // Extract and verify the number of operands
    int nOperandsActual = operatorBinding.getOperandCount();
    if (nOperandsActual < 2) {
      throw new IllegalArgumentException(
          "OBJECT_DELETE functions must be called on at least 2 arguments");
    }
    // Extract each operand type and verify that it is valid
    final List<Integer> operandList = getOperandList(operatorBinding.getOperandCount());
    for (int i : operandList) {
      if (i == 0) {
        // Throw an error if the first argument is not a JSON type
        if ((operatorBinding.getOperandType(0).getSqlTypeName().getFamily() != SqlTypeFamily.MAP)) {
          if (throwOnFailure) {
            throw new IllegalArgumentException("OBJECT_DELETE first argument must be MAP");
          } else {
            return false;
          }
        }
      } else {
        // Throw an error if any of the keys to delete are not strings
        if ((operatorBinding.getOperandType(i).getSqlTypeName().getFamily()
            != SqlTypeFamily.CHARACTER)) {
          if (throwOnFailure) {
            throw new IllegalArgumentException(
                "OBJECT_DELETE subsequent arguments must be strings");
          } else {
            return false;
          }
        }
      }
    }
    return true;
  }
}
