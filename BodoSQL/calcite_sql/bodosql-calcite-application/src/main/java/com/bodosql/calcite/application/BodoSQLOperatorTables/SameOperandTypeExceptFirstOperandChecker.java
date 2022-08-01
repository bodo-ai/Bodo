package com.bodosql.calcite.application.BodoSQLOperatorTables;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.util.Static.RESOURCE;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;

public class SameOperandTypeExceptFirstOperandChecker extends SameOperandTypeChecker {
  // ~ Instance fields --------------------------------------------------------

  protected final String firstOperandTypeName;

  // ~ Constructors -----------------------------------------------------------

  /**
   * Parameter type-checking strategy where all operand types except the first one must be the same.
   *
   * <p>Influenced by SameOperandTypeExceptLastOperandChecker
   * https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/type/SameOperandTypeExceptLastOperandChecker.java#L39
   */
  public SameOperandTypeExceptFirstOperandChecker(int nOperands, String firstOperandTypeName) {
    super(nOperands);
    this.firstOperandTypeName = firstOperandTypeName;
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  protected boolean checkOperandTypesImpl(
      SqlOperatorBinding operatorBinding,
      boolean throwOnFailure,
      @Nullable SqlCallBinding callBinding) {
    if (throwOnFailure && callBinding == null) {
      throw new IllegalArgumentException(
          "callBinding must be non-null in case throwOnFailure=true");
    }
    int nOperandsActual = nOperands;
    if (nOperandsActual == -1) {
      nOperandsActual = operatorBinding.getOperandCount();
    }
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
    int prev = -1;
    for (int i : operandList) {
      if (prev > 0 && i < operandList.get(operandList.size())) {
        if (!SqlTypeUtil.isComparable(types[i], types[prev])) {
          if (!throwOnFailure) {
            return false;
          }

          // REVIEW jvs 5-June-2005: Why don't we use
          // newValidationSignatureError() here?  It gives more
          // specific diagnostics.
          throw requireNonNull(callBinding, "callBinding")
              .newValidationError(RESOURCE.needSameTypeParameter());
        }
      }
      prev = i;
    }
    return true;
  }

  @Override
  public String getAllowedSignatures(SqlOperator op, String opName) {
    final String typeName = getTypeName();
    if (nOperands == -1) {
      return SqlUtil.getAliasedSignature(op, opName, ImmutableList.of(typeName, typeName, "..."));
    } else {
      List<String> types = new ArrayList<>();
      types.add(firstOperandTypeName);
      for (int i = 0; i < nOperands - 1; i++) {
        types.add(typeName);
      }
      return SqlUtil.getAliasedSignature(op, opName, types);
    }
  }
}
