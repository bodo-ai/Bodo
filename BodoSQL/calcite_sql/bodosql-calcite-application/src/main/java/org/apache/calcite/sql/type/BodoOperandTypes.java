package org.apache.calcite.sql.type;



import com.bodosql.calcite.application.operatorTables.DatetimeFnUtils;
import com.bodosql.calcite.application.operatorTables.VariantOperandChecker;

import com.bodosql.calcite.application.operatorTables.OperatorTableUtils;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.util.Pair;

import java.util.List;

public class BodoOperandTypes {


  public static final SqlSingleOperandTypeChecker VARIANT = VariantOperandChecker.INSTANCE;

  static class DateTimePartOperandTypeChecker implements SqlSingleOperandTypeChecker {

    /**
     * Implementation of checkSingleOperandType that always throws an error if there's a failure.
     * @param callBinding The call whose operandTypes are being validated.
     * @param operand The operand being validated
     * @param iFormalOperand The index of the operand in the call's list of operand values.
     */
    private boolean checkSingleOperandTypeWrapper(SqlCallBinding callBinding, SqlNode operand, int iFormalOperand) {
      if (callBinding.getOperandCount() <= iFormalOperand) {
        throw new IllegalArgumentException("Error: expected at least " + iFormalOperand + " arguments. Found " + callBinding.getOperandCount() + " .");
      }
      if (!(operand instanceof SqlLiteral)){
       throw new IllegalArgumentException("Error: operand is not a literal.");
      }
      if (!(((SqlLiteral) operand).getValue() instanceof DatetimeFnUtils.DateTimePart)){
        throw new IllegalArgumentException("Error: operand is not a valid DateTimePart Symbol.");
      }

     return true;
    }

    @Override
    public boolean checkSingleOperandType(SqlCallBinding callBinding, SqlNode operand, int iFormalOperand, boolean throwOnFailure) {
      try {
        return this.checkSingleOperandTypeWrapper(callBinding, operand, iFormalOperand);
      } catch (IllegalArgumentException e) {
        if (throwOnFailure){
          throw e;
        } else {
          return false;
        }
      }
    }

    @Override
    public String getAllowedSignatures(SqlOperator op, String opName) {
      return "<DATETIME_INTERVAL_SYMBOL>";
    }
  }
  public static final SqlSingleOperandTypeChecker DATETIME_INTERVAL_SYMBOL = new DateTimePartOperandTypeChecker();
  public static final SqlSingleOperandTypeChecker DATE_NUMERIC =
      OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.NUMERIC);
  public static final SqlSingleOperandTypeChecker INTEGER_DATE =
      OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.DATE);
  public static final SqlSingleOperandTypeChecker DATE_DATE =
      OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.DATE);

  public static final SqlSingleOperandTypeChecker PLUS_OPERATOR =
      OperandTypes.or(OperandTypes.NUMERIC_NUMERIC, OperandTypes.INTERVAL_SAME_SAME, OperandTypes.DATETIME_INTERVAL,
          OperandTypes.INTERVAL_DATETIME, DATE_NUMERIC, INTEGER_DATE);

  public static final SqlSingleOperandTypeChecker MINUS_OPERATOR =
      OperandTypes.or(
          OperandTypes.NUMERIC_NUMERIC, OperandTypes.INTERVAL_SAME_SAME, OperandTypes.DATETIME_INTERVAL, DATE_NUMERIC,
          DATE_DATE
      );

  public static final SqlSingleOperandTypeChecker CHARACTER_CHARACTER = OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER);


  public static final SqlSingleOperandTypeChecker ARRAY_OR_MAP =
          OperandTypes.family(SqlTypeFamily.ARRAY)
                  .or(OperandTypes.family(SqlTypeFamily.MAP))
                  .or(OperandTypes.family(SqlTypeFamily.ANY));

  /**
   * Creates an operand checker from a sequence of single-operand checkers.
   * This is an copy of OperandTypes.sequence, that takes a list of rules instead of variable number of arguments.
   */
  public static SqlOperandTypeChecker sequence(String allowedSignatures,
                                               List<SqlSingleOperandTypeChecker> rules) {
    return new CompositeOperandTypeChecker(
            CompositeOperandTypeChecker.Composition.SEQUENCE,
            ImmutableList.copyOf(rules), allowedSignatures, null, null);
  }

  public static SqlOperandTypeChecker TO_NUMBER_OPERAND_TYPE_CHECKER = OperatorTableUtils.argumentRangeExplicit(1, "TO_NUMBER",
          List.of(Pair.of(OperandTypes.BOOLEAN.or(OperandTypes.NUMERIC).or(OperandTypes.CHARACTER).or(BodoOperandTypes.VARIANT), "BOOLEAN, NUMERIC, CHAR, or VARIANT"),
          Pair.of(OperandTypes.POSITIVE_INTEGER_LITERAL, "POSITIVE INTEGER LITERAL"),
          Pair.of(OperandTypes.POSITIVE_INTEGER_LITERAL, "POSITIVE INTEGER LITERAL")));

  public static SqlOperandTypeChecker TRY_TO_NUMBER_OPERAND_TYPE_CHECKER = OperatorTableUtils.argumentRangeExplicit(1, "TRY_TO_NUMBER",
          List.of(Pair.of(OperandTypes.BOOLEAN.or(OperandTypes.NUMERIC).or(OperandTypes.CHARACTER), "BOOLEAN, NUMERIC, CHAR, or VARIANT"),
                  Pair.of(OperandTypes.POSITIVE_INTEGER_LITERAL, "POSITIVE INTEGER LITERAL"),
                  Pair.of(OperandTypes.POSITIVE_INTEGER_LITERAL, "POSITIVE INTEGER LITERAL")));



}
