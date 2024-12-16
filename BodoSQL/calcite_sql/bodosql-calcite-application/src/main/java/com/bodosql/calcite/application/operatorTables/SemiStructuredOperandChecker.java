package com.bodosql.calcite.application.operatorTables;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.type.ArraySqlType;
import org.apache.calcite.sql.type.MapSqlType;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;

public class SemiStructuredOperandChecker implements SqlSingleOperandTypeChecker {

  static SemiStructuredOperandChecker INSTANCE;

  static {
    INSTANCE = new SemiStructuredOperandChecker();
  }

  /** Checks a single operand to see if it is a VARIANT, ARRAY or JSON type. */
  @Override
  public boolean checkSingleOperandType(
      SqlCallBinding callBinding, SqlNode operand, int iFormalOperand, boolean throwOnFailure) {
    RelDataType type = SqlTypeUtil.deriveType(callBinding, operand);
    if (!((type instanceof VariantSqlType)
        || (type instanceof ArraySqlType)
        || (type instanceof MapSqlType))) {
      if (throwOnFailure) {
        throw new IllegalArgumentException(
            "Argument must be a variant, array or map; received " + type);
      }
      return false;
    }
    return true;
  }

  // This operand checker is not designed to be called on the entire call binding,
  // only on a single operand.
  @Override
  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    throw new UnsupportedOperationException();
  }

  @Override
  public SqlOperandCountRange getOperandCountRange() {
    return SqlOperandCountRanges.of(1);
  }

  @Override
  public String getAllowedSignatures(SqlOperator op, String opName) {
    return "<SEMI_STRUCTURED>";
  }
}
