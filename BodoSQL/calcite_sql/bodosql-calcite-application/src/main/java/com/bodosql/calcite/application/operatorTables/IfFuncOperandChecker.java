package com.bodosql.calcite.application.operatorTables;

import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.VariantSqlType;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * IfFuncOperandChecker will first check if one of the then/else branches is a STRING while the
 * other is VARIANT, and will cast the VARIANT to STRING, before applying more generic rules, which
 * check that the first argument must be boolean, and that the next two args are either comparable,
 * or if exactly one of the last two args is VARIANT, then cast the other to VARIANT as well.
 */
public class IfFuncOperandChecker extends VariantCastingTypeChecker {
  static IfFuncOperandChecker INSTANCE;

  static {
    INSTANCE = new IfFuncOperandChecker();
  }

  public IfFuncOperandChecker() {
    super(new SameOperandTypeExceptFirstOperandChecker(3, "BOOLEAN"), List.of(1, 2), true);
  }

  @Override
  protected boolean checkOperandTypesImpl(
      SqlOperatorBinding operatorBinding,
      boolean throwOnFailure,
      @Nullable SqlCallBinding callBinding) {

    int nOperandsActual = operatorBinding.getOperandCount();
    if (nOperandsActual != 3) {
      throw new IllegalArgumentException("IF/IFF expects exactly 3 arguments");
    }

    List<RelDataType> types = operatorBinding.collectOperandTypes();
    RelDataType arg1 = types.get(1);
    RelDataType arg2 = types.get(2);
    boolean arg1IsVariant = arg1 instanceof VariantSqlType;
    boolean arg2IsVariant = arg2 instanceof VariantSqlType;
    boolean arg1IsString = SqlTypeFamily.STRING.contains(arg1);
    boolean arg2IsString = SqlTypeFamily.STRING.contains(arg2);

    if (arg1IsString || arg2IsString) {
      if (arg1IsVariant || arg2IsVariant) {
        int targetIdx = arg1IsVariant ? 1 : 2;
        SqlCall call = callBinding.getCall();
        SqlNode castedOperand =
            CastingOperatorTable.TO_CHAR.createCall(
                SqlParserPos.ZERO, callBinding.operand(targetIdx));
        call.setOperand(targetIdx, castedOperand);
      }
    }

    return super.checkOperandTypesImpl(operatorBinding, throwOnFailure, callBinding);
  }
}
