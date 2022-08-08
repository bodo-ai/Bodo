package com.bodosql.calcite.application.BodoSQLOperatorTables;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.util.Static.RESOURCE;

import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;

public class DecodeOperandChecker extends SameOperandTypeChecker {
  /**
   * Parameter type-checking strategy for the DECODE function, which has the following rules:
   *
   * <p>There must be 3+ arguments - Argument 0 must have the same type as arguments 1, 3, 5, etc.
   * (if there are an even number of arguments, then the final argument is exempt) - Argument 2 must
   * have the same type as arguments 4, 6, 8, etc. (if there are an even number of arguments, then
   * the final argument must also match). If there is no default argument, then it acts as if the
   * default fallback value is NULL.
   *
   * <p>Influenced by SameOperandTypeExceptLastOperandChecker
   * https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/type/SameOperandTypeExceptLastOperandChecker.java#L39
   */
  public DecodeOperandChecker() {
    super(-1);
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
    if (nOperandsActual < 3) {
      throw new IllegalArgumentException("DECODE must be called on at least 3 arguments");
    }
    // Extract each operand type and verify that it is valid
    RelDataType[] types = new RelDataType[nOperandsActual];
    final List<Integer> operandList = getOperandList(operatorBinding.getOperandCount());
    for (int i : operandList) {
      if (operatorBinding.isOperandNull(i, false)) {
        if (requireNonNull(callBinding, "callBinding").isTypeCoercionEnabled()) {
          types[i] = operatorBinding.getTypeFactory().createSqlType(SqlTypeName.NULL);
        } else if (throwOnFailure) {
          throw callBinding
              .getValidator()
              .newValidationError(callBinding.operand(i), RESOURCE.nullIllegal());
        } else {
          return false;
        }
      } else {
        types[i] = operatorBinding.getOperandType(i);
      }
    }

    // Loop over each argument and verify that it either matches the type of
    // argument 0 or 2, depending on whether it corresponds to an input or
    // an output (this is the part that is different from SameOperandTypeExceptLastOperandChecker)
    RelDataType inputType = types[0];
    RelDataType outputType = types[2];
    Boolean faliure;
    for (int i = 1; i < nOperandsActual; i += 1) {
      if (i == 2) {
        continue;
      }
      if ((i % 2 == 1) && (i != (nOperandsActual - 1))) {
        faliure = !SqlTypeUtil.isComparable(types[i], inputType);
      } else {
        faliure = !SqlTypeUtil.isComparable(types[i], outputType);
      }
      if (faliure) {
        if (throwOnFailure) {
          throw requireNonNull(callBinding, "callBinding")
              .newValidationError(RESOURCE.needSameTypeParameter());
        } else {
          return false;
        }
      }
    }
    return true;
  }
}
