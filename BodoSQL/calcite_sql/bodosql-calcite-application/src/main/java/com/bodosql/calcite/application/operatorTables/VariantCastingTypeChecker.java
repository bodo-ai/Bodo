package com.bodosql.calcite.application.operatorTables;

import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;
import org.apache.calcite.sql.validate.implicit.BodoTypeCoercionImpl;
import org.checkerframework.checker.nullness.qual.Nullable;

public class VariantCastingTypeChecker extends SameOperandTypeChecker {
  /**
   * Parameter type-checking strategy for functions which must have coercible types. This checker
   * will insert any necessary casts before delegating to another type checker which may enforce its
   * own, more function-specific rules.
   */
  public VariantCastingTypeChecker(
      SqlOperandTypeChecker innerChecker, List<Integer> variantArgs, boolean castToVariant) {
    super(-1);
    this.innerChecker = innerChecker;
    this.variantArgs = variantArgs;
    this.castToVariant = castToVariant;
  }

  SqlOperandTypeChecker innerChecker;
  List<Integer> variantArgs;
  boolean castToVariant;

  @Override
  protected boolean checkOperandTypesImpl(
      SqlOperatorBinding operatorBinding,
      boolean throwOnFailure,
      @Nullable SqlCallBinding callBinding) {
    // If the types are natively accepted by the function to begin with, then use those types.
    if (innerChecker.checkOperandTypes(callBinding, false)) {
      return true;
    }

    if (callBinding == null) {
      throw new RuntimeException("could not get call binding");
    }

    // If all non-variant types with indices in variantArgs have the same type, then
    // we cast the all variant args to that type.
    if (callBinding.isTypeCoercionEnabled()) {
      List<RelDataType> types = callBinding.collectOperandTypes();
      RelDataType commonType = null;
      for (int i = 0; i < types.size(); i++) {
        if (variantArgs.contains(i)) {
          if (types.get(i) instanceof VariantSqlType) {
            if (castToVariant) {
              commonType = types.get(i);
            }
            continue;
          }

          if (!this.castToVariant) {
            if (commonType == null) {
              commonType = types.get(i);
            } else {
              if (!SqlTypeUtil.isComparable(types.get(i), commonType)) {
                if (throwOnFailure) {
                  throw new RuntimeException("Found incompatible types");
                }
              }
            }
          }
        }
      }

      if (commonType != null) {
        SqlCall call = callBinding.getCall();
        for (int i : variantArgs) {
          boolean argIsVariant = types.get(i) instanceof VariantSqlType;
          if (argIsVariant == !castToVariant) {
            RelDataType targetType =
                callBinding
                    .getTypeFactory()
                    .createTypeWithNullability(commonType, types.get(i).isNullable());
            BodoTypeCoercionImpl typeCoercion =
                (BodoTypeCoercionImpl) callBinding.getValidator().getTypeCoercion();
            typeCoercion.coerceOperandType(callBinding.getScope(), call, i, targetType);
          }
        }
      }
    }

    // Ensure that the coalesced types are accepted by the function
    return innerChecker.checkOperandTypes(callBinding, throwOnFailure);
  }
}
