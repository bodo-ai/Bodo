package com.bodosql.calcite.application.operatorTables;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;

public class VariantOperandChecker implements SqlSingleOperandTypeChecker {

  public static VariantOperandChecker INSTANCE;

  static {
    INSTANCE = new VariantOperandChecker();
  }

  /** Checks a single operand to see if it is a VARIANT type. */
  @Override
  public boolean checkSingleOperandType(
      SqlCallBinding callBinding, SqlNode operand, int iFormalOperand, boolean throwOnFailure) {
    RelDataType type = SqlTypeUtil.deriveType(callBinding, operand);
    if (!(type instanceof VariantSqlType)) {
      if (throwOnFailure) {
        throw new IllegalArgumentException("Argument must be a variant; received " + type);
      }
      return false;
    }
    return true;
  }

  @Override
  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    return callBinding.getOperandCount() == 1
        && checkSingleOperandType(callBinding, callBinding.operands().get(0), 0, throwOnFailure);
  }

  @Override
  public SqlOperandCountRange getOperandCountRange() {
    return SqlOperandCountRanges.of(1);
  }

  @Override
  public String getAllowedSignatures(SqlOperator op, String opName) {
    return "'" + opName + "(<VARIANT>)'";
  }
}
