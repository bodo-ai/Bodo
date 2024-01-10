package com.bodosql.calcite.application.utils;

import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.implicit.BodoTypeCoercionImpl;
import org.checkerframework.checker.nullness.qual.Nullable;

public class CoalesableTypeChecker extends SameOperandTypeChecker {
  /**
   * Parameter type-checking strategy for functions which must have coercible types. This checker
   * will insert any necessary casts before delegating to another type checker which may enforce its
   * own, more function-specific rules.
   */
  public CoalesableTypeChecker(SqlOperandTypeChecker innerChecker) {
    super(-1);
    this.innerChecker = innerChecker;
  }

  SqlOperandTypeChecker innerChecker;

  @Override
  protected boolean checkOperandTypesImpl(
      SqlOperatorBinding operatorBinding,
      boolean throwOnFailure,
      @Nullable SqlCallBinding callBinding) {
    // If the types are natively accepted by the function to begin with, then use those types.
    if (innerChecker.checkOperandTypes(callBinding, false)) {
      return true;
    }

    // Attempt to coalesce argument types
    if (callBinding == null) {
      throw new RuntimeException("could not get call binding");
    }

    if (callBinding.isTypeCoercionEnabled()) {
      SqlValidator validator = callBinding.getScope().getValidator();
      BodoTypeCoercionImpl typeCoercion =
          new BodoTypeCoercionImpl(operatorBinding.getTypeFactory(), validator);

      if (!typeCoercion.coalesceCoercionImpl(callBinding)) {
        if (throwOnFailure) {
          throw new RuntimeException("coalesce of types failed");
        } else {
          return false;
        }
      }
    }

    // Ensure that the coalesced types are accepted by the function
    return innerChecker.checkOperandTypes(callBinding, throwOnFailure);
  }
}
