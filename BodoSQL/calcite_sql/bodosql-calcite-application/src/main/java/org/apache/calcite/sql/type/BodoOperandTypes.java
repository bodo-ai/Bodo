package org.apache.calcite.sql.type;

public class BodoOperandTypes {
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
}
