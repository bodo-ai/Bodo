package com.bodosql.calcite.application.operatorTables;

import static org.apache.calcite.util.Static.RESOURCE;

import java.util.Collections;
import java.util.List;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.checkerframework.checker.nullness.qual.Nullable;

public class ObjectConstructOperandChecker extends SameOperandTypeChecker {
  /**
   * Parameter type-checking strategy for the OBJECT_CONSTRUCT function(s), which have the following
   * rules:
   *
   * <ul>
   *   <li>There must be an even number of arguments in the form (key1, value1, key2, val2, ...)
   *   <li>The keys must all be strings - The values can be any type
   * </ul>
   *
   * Note: the -1 indicates that this function is variadic. <br>
   * [BSE-1940] TODO: investigate using implicit casting to aid type harmonization for this and
   * other semi-structured functions.
   */
  public ObjectConstructOperandChecker() {
    super(-1);
  }

  static ObjectConstructOperandChecker INSTANCE;

  static {
    INSTANCE = new ObjectConstructOperandChecker();
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
    if ((nOperandsActual % 2) != 0) {
      throw new IllegalArgumentException(
          "OBJECT_CONSTRUCT_KEEP_NULL functions must be called on an even number of arguments");
    }
    // Extract each operand type and verify that it is valid
    final List<Integer> operandList = getOperandList(operatorBinding.getOperandCount());
    for (int i : operandList) {
      if ((i % 2) == 0) {
        // Throw an error if any of the keys are not string-literals
        if ((operatorBinding.getOperandType(i).getSqlTypeName().getFamily()
                != SqlTypeFamily.CHARACTER)
            || !callBinding.getCall().operand(0).isA(Collections.singleton(SqlKind.LITERAL))) {
          if (throwOnFailure) {
            throw callBinding
                .getValidator()
                .newValidationError(
                    callBinding.operand(i),
                    RESOURCE.argumentMustBeLiteral("OBJECT_CONSTRUCT_KEEP_NULL"));
          } else {
            return false;
          }
        }
      }
    }
    return true;
  }
}
