package org.apache.calcite.sql.type;



import com.bodosql.calcite.application.operatorTables.VariantOperandChecker;

import com.bodosql.calcite.application.operatorTables.OperatorTableUtils;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.util.Pair;

import java.util.List;

public class BodoOperandTypes {


  public static final SqlSingleOperandTypeChecker VARIANT = VariantOperandChecker.INSTANCE;

  public static final SqlSingleOperandTypeChecker DATE_INTEGER =
      OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.INTEGER);
  public static final SqlSingleOperandTypeChecker INTEGER_DATE =
      OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.DATE);
  public static final SqlSingleOperandTypeChecker DATE_DATE =
      OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.DATE);

  public static final SqlSingleOperandTypeChecker PLUS_OPERATOR =
      OperandTypes.or(OperandTypes.NUMERIC_NUMERIC, OperandTypes.INTERVAL_SAME_SAME, OperandTypes.DATETIME_INTERVAL,
          OperandTypes.INTERVAL_DATETIME, DATE_INTEGER, INTEGER_DATE);

  public static final SqlSingleOperandTypeChecker MINUS_OPERATOR =
      OperandTypes.or(
          OperandTypes.NUMERIC_NUMERIC, OperandTypes.INTERVAL_SAME_SAME, OperandTypes.DATETIME_INTERVAL, DATE_INTEGER,
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
