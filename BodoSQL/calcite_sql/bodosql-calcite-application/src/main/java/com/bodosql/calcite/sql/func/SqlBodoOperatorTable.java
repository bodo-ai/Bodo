package com.bodosql.calcite.sql.func;

import com.bodosql.calcite.application.operatorTables.SqlNullPolicyFunction;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlQuantifyOperator;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.BodoOperandTypes;
import org.apache.calcite.sql.type.InferTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.util.ReflectiveSqlOperatorTable;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;

public class SqlBodoOperatorTable extends ReflectiveSqlOperatorTable {
  private static @MonotonicNonNull SqlBodoOperatorTable instance;

  public static final SqlQuantifyOperator SOME_NULLEQ =
      new SqlQuantifyOperator(SqlKind.SOME, SqlKind.NULL_EQUALS);

  public static final SqlQuantifyOperator ALL_NULLEQ =
      new SqlQuantifyOperator(SqlKind.ALL, SqlKind.NULL_EQUALS);

  public static final SqlLikeQuantifyOperator ANY_LIKE =
      new SqlLikeQuantifyOperator("LIKE ANY", SqlKind.LIKE, SqlKind.SOME, true);

  public static final SqlLikeQuantifyOperator ALL_LIKE =
      new SqlLikeQuantifyOperator("LIKE ALL", SqlKind.LIKE, SqlKind.ALL, true);

  public static final SqlLikeQuantifyOperator ANY_ILIKE =
      new SqlLikeQuantifyOperator("ILIKE ANY", SqlKind.LIKE, SqlKind.SOME, false);

  public static final SqlLikeQuantifyOperator ALL_ILIKE =
      new SqlLikeQuantifyOperator("ILIKE ALL", SqlKind.LIKE, SqlKind.ALL, false);

  public static final SqlBinaryOperator NULL_EQUALS =
      new SqlBinaryOperator(
          "<=>",
          SqlKind.NULL_EQUALS,
          30,
          true,
          // Note: The return value is never null, only True/False.
          ReturnTypes.BOOLEAN,
          InferTypes.FIRST_KNOWN,
          OperandTypes.COMPARABLE_UNORDERED_COMPARABLE_UNORDERED);

  public static SqlQuantifyOperator some(SqlKind comparisonKind) {
    switch (comparisonKind) {
      case EQUALS:
        return SqlStdOperatorTable.SOME_EQ;
      case NOT_EQUALS:
        return SqlStdOperatorTable.SOME_NE;
      case LESS_THAN:
        return SqlStdOperatorTable.SOME_LT;
      case LESS_THAN_OR_EQUAL:
        return SqlStdOperatorTable.SOME_LE;
      case GREATER_THAN:
        return SqlStdOperatorTable.SOME_GT;
      case GREATER_THAN_OR_EQUAL:
        return SqlStdOperatorTable.SOME_GE;
      case NULL_EQUALS:
        return SOME_NULLEQ;
      default:
        throw new AssertionError(comparisonKind);
    }
  }

  public static SqlQuantifyOperator all(SqlKind comparisonKind) {
    switch (comparisonKind) {
      case EQUALS:
        return SqlStdOperatorTable.ALL_EQ;
      case NOT_EQUALS:
        return SqlStdOperatorTable.ALL_NE;
      case LESS_THAN:
        return SqlStdOperatorTable.ALL_LT;
      case LESS_THAN_OR_EQUAL:
        return SqlStdOperatorTable.ALL_LE;
      case GREATER_THAN:
        return SqlStdOperatorTable.ALL_GT;
      case GREATER_THAN_OR_EQUAL:
        return SqlStdOperatorTable.ALL_GE;
      case NULL_EQUALS:
        return ALL_NULLEQ;
      default:
        throw new AssertionError(comparisonKind);
    }
  }

  /** The <code>LOCALTIMESTAMP [(<i>precision</i>)]</code> function. */
  public static final SqlFunction LOCALTIMESTAMP =
      new SqlAbstractTZSessionFunction("LOCALTIMESTAMP");

  /** The <code>CURRENT_TIMESTAMP [(<i>precision</i>)]</code> function. */
  public static final SqlFunction CURRENT_TIMESTAMP =
      new SqlAbstractTZSessionFunction("CURRENT_TIMESTAMP");

  /** The <code>TIMESTAMPADD</code> function. */
  public static final SqlFunction TIMESTAMP_ADD = new BodoSqlTimestampAddFunction();

  /** The <code>TIMESTAMPDIFF</code> function. */
  public static final SqlFunction TIMESTAMP_DIFF = new BodoSqlTimestampDiffFunction();

  public static final SqlNullPolicyFunction LAST_DAY =
      SqlNullPolicyFunction.createAnyPolicy(
          "LAST_DAY",
          ReturnTypes.DATE_NULLABLE,
          OperandTypes.or(
              OperandTypes.DATETIME,
              OperandTypes.sequence(
                  "LAST_DAY(DATE/TIMESTAMP, STRING/DATETIME_INTERVAL)",
                  OperandTypes.DATETIME,
                  BodoOperandTypes.DATETIME_INTERVAL_SYMBOL)),
          SqlFunctionCategory.TIMEDATE);

  /**
   * A dummy function that allows adding intervals that are illegal to add using the + operator.
   * This is an internal-only operator for supporting interval literals with commas. For example, we
   * want to parse "INTERVAL '1 year, 1 second'" as "INTERVAL '1 year' + INTERVAL '1 second'", but
   * that addition would not type check. Instead, we use COMBINE_INTERVALS to represent the
   * addition, and let bodo handle the actual addition implementation.
   */
  public static final SqlBasicFunction COMBINE_INTERVALS =
      SqlBasicFunction.create(
          "COMBINE_INTERVALS",
          ReturnTypes.ARG0_INTERVAL,
          OperandTypes.family(SqlTypeFamily.DATETIME_INTERVAL, SqlTypeFamily.DATETIME_INTERVAL),
          SqlFunctionCategory.TIMEDATE);

  public static synchronized SqlBodoOperatorTable instance() {
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new SqlBodoOperatorTable();
      instance.init();
    }
    return instance;
  }
}
