package org.apache.calcite.sql.type;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlOperatorBinding;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;

import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoReturnTypes {
  /**
   * Type-inference strategy whereby the result type of a call is an integer
   * if both operands ares.
   */
  public static final SqlReturnTypeInference DATE_SUB = opBinding -> {
    RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    RelDataType type1 = opBinding.getOperandType(0);
    RelDataType type2 = opBinding.getOperandType(1);
    if (type1.getSqlTypeName() == SqlTypeName.DATE
        && type2.getSqlTypeName() == SqlTypeName.DATE) {
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
    // This rule doesn't apply
    return null;
  };

  /**
   * Same as {@link #DATE_SUB} but returns with nullability if any
   * of the operands is nullable by using
   * {@link SqlTypeTransforms#TO_NULLABLE}.
   */
  public static final SqlReturnTypeInference DATE_SUB_NULLABLE =
      DATE_SUB.andThen(SqlTypeTransforms.TO_NULLABLE);

  /**
   * Type-inference strategy whereby the result type of a call is
   * {@link ReturnTypes#DECIMAL_SUM_NULLABLE} with a fallback to date - date, and
   * finally with a fallback to {@link ReturnTypes#LEAST_RESTRICTIVE}
   * These rules are used for subtraction.
   */
  public static final SqlReturnTypeInference NULLABLE_SUB =
      new SqlReturnTypeInferenceChain(
          ReturnTypes.DECIMAL_SUM_NULLABLE, DATE_SUB_NULLABLE, ReturnTypes.LEAST_RESTRICTIVE);

  public static final SqlReturnTypeInference TZAWARE_TIMESTAMP =
      new SqlReturnTypeInference() {
        @Override public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
          return BodoRelDataTypeFactory.createTZAwareSqlType(
              opBinding.getTypeFactory(), null);
        }
      };

  public static final SqlReturnTypeInference TZAWARE_TIMESTAMP_NULLABLE =
      TZAWARE_TIMESTAMP.andThen(SqlTypeTransforms.TO_NULLABLE);

  public static final SqlReturnTypeInference UTC_TIMESTAMP =
      new SqlReturnTypeInference() {
        @Override public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
          return BodoRelDataTypeFactory.createTZAwareSqlType(
              opBinding.getTypeFactory(), BodoTZInfo.UTC);
        }
      };
}
