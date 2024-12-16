package com.bodosql.calcite.sql.func;

import com.bodosql.calcite.sql.validate.BodoCoercionUtil;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeName;

public class CoercingOperandChecker implements SqlOperandTypeChecker {

  private final List<List<SqlTypeName>> acceptedSignatures;
  private final SqlOperandCountRange argRanges;

  /**
   * Takes a list of accepted signatures (list of argument types) for a possibly overloaded
   * function. For example, Suppose we define a function MyFunc that supports the following
   * signatures: MyFunc(<INTEGER>, <VARCHAR>) MyFunc(<VARCHAR>, <BOOLEAN>)
   *
   * <p>The operand checker for this function could be constructed as: new
   * CoercingOperandChecker(listOf( listOf(SqlTypeName.INTEGER, SqlTypeName.VARCHAR),
   * listOf(SqlTypeName.VARCHAR, SqlTypeName.BOOLEAN) )) This checker will also accept being called
   * with MyFunc(<BOOLEAN>, <VARCHAR>), but will insert a cast (TO_NUMBER) for the first argument.
   *
   * @param acceptedSignatures list of accepted signatures
   */
  CoercingOperandChecker(List<List<SqlTypeName>> acceptedSignatures) {
    assert !acceptedSignatures.isEmpty();

    this.acceptedSignatures = acceptedSignatures;
    int minArgs = Integer.MAX_VALUE;
    int maxArgs = 0;
    for (List<SqlTypeName> args : acceptedSignatures) {
      int n = args.size();
      minArgs = Math.min(n, minArgs);
      maxArgs = Math.max(n, maxArgs);
    }
    argRanges = SqlOperandCountRanges.between(minArgs, maxArgs);
  }

  private static final int INIT_OVERLOAD_INDEX = -1;
  private static final int INVALID_OVERLOAD_INDEX = -2;

  protected boolean typesAreEqual(SqlTypeName a, SqlTypeName b) {
    // For now simple equality will suffice, but we may want to extend this to consider the type
    // family instead (or otherwise broaden the check)
    return a == b;
  }

  @Override
  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    List<RelDataType> operandTypes = callBinding.collectOperandTypes();

    // Index of argument set that this call could select if casts were inserted
    int castableOverloadIndex = INIT_OVERLOAD_INDEX;
    for (int overloadIndex = 0; overloadIndex < acceptedSignatures.size(); overloadIndex++) {
      boolean typesMatch = true;
      boolean requiresCast = false;

      List<SqlTypeName> signatureArgs = acceptedSignatures.get(overloadIndex);
      if (signatureArgs.size() == operandTypes.size()) {
        for (int i = 0; i < signatureArgs.size(); i++) {
          if (!typesAreEqual(operandTypes.get(i).getSqlTypeName(), signatureArgs.get(i))) {
            typesMatch = false;
            if (!callBinding.isTypeCoercionEnabled()) {
              // If coercion isn't enabled, then we can skip checking the rest of the arguments for
              // this overload and move on.
              break;
            }

            // If this argument could be coerced to the type required by this overload, then save
            // the index of this overload, so we can insert the cast if no other overload matches
            // first.
            RelDataType targetType =
                callBinding.getTypeFactory().createSqlType(signatureArgs.get(i));
            if (BodoCoercionUtil.Companion.canCastFromUDF(operandTypes.get(i), targetType, true)) {
              // We don't want to do the cast yet, because some other argument may not match, and a
              // different set of arguments might need to be selected instead.
              requiresCast = true;
            }
          }
        }

        // All arguments match this overload's signature
        if (typesMatch) {
          return true;
        }

        // Some arguments do not match this overload's signature, but can be coerced.
        if (requiresCast) {
          // Coercion is ambiguous - but don't throw an error immediately in case an alternative
          // overload can be selected.
          if (castableOverloadIndex != INIT_OVERLOAD_INDEX) {
            castableOverloadIndex = INVALID_OVERLOAD_INDEX;
          } else {
            castableOverloadIndex = overloadIndex;
          }
        }
      }
    }

    // If coercion makes this call ambiguous, reject the call
    if (castableOverloadIndex == INVALID_OVERLOAD_INDEX) {
      // TODO(Nick) explore just selecting the first match instead
      if (throwOnFailure) {
        throw new RuntimeException("Ambiguous function call");
      }
      return false;
    }

    // If we have identified an overload that we can coerce to, insert the casts required
    if (castableOverloadIndex != INIT_OVERLOAD_INDEX) {
      List<SqlTypeName> signatureArgs = acceptedSignatures.get(castableOverloadIndex);
      assert callBinding.isTypeCoercionEnabled();

      // This signature is valid, we just need to insert casts to select it.
      for (int i = 0; i < signatureArgs.size(); i++) {
        if (operandTypes.get(i).getSqlTypeName() != signatureArgs.get(i)) {
          RelDataType targetType = callBinding.getTypeFactory().createSqlType(signatureArgs.get(i));
          SqlOperator castFn = BodoCoercionUtil.Companion.getCastFunction(targetType);
          SqlNode castedArg = castFn.createCall(SqlParserPos.ZERO, callBinding.operand(i));
          callBinding.getCall().setOperand(i, castedArg);
        }
      }

      return true;
    }

    // No overload matched - reject the call
    if (throwOnFailure) {
      // TODO(aneesh): it would be nice to track the "closest" match, or otherwise provide a better
      // reason (e.g. wrong arguments, or arg0 isn't castable, etc)
      throw new RuntimeException("Invalid call");
    }
    return false;
  }

  @Override
  public SqlOperandCountRange getOperandCountRange() {
    return argRanges;
  }

  @Override
  public String getAllowedSignatures(SqlOperator op, String opName) {
    String signatures = "";
    for (List<SqlTypeName> signature : acceptedSignatures) {
      signatures += SqlUtil.getAliasedSignature(op, opName, signature) + "\n";
    }
    return signatures;
  }
}
