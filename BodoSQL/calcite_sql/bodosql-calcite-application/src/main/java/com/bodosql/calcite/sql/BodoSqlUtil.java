package com.bodosql.calcite.sql;

import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;

import java.math.BigDecimal;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoSqlUtil {
  /** Creates a table sample spec from the parser information. */
  public static @Nullable SqlSampleSpec createTableSample(
      @NonNull SqlParserPos pos,
      boolean isBernoulli,
      @NonNull SqlNumericLiteral sampleRate,
      boolean sampleByRows,
      boolean isRepeatable,
      int repeatableSeed) {
    BigDecimal rate = sampleRate.bigDecimalValue();
    if (rate == null) {
      return null;
    }

    if (sampleByRows) {
      final BigDecimal maxRows = BigDecimal.valueOf(1000000L);
      if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(maxRows) > 0) {
        throw SqlUtil.newContextException(
            pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, maxRows));
      }

      return isRepeatable
          ? SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate, repeatableSeed)
          : SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate);
    }

    final BigDecimal oneHundred = BigDecimal.valueOf(100L);
    if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(oneHundred) > 0) {
      throw SqlUtil.newContextException(
          pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, oneHundred));
    }

    // Treat TABLESAMPLE(0) and TABLESAMPLE(100) as no table
    // sampling at all.  Not strictly correct: TABLESAMPLE(0)
    // should produce no output, but it simplifies implementation
    // to know that some amount of sampling will occur.
    // In practice values less than ~1E-43% are treated as 0.0 and
    // values greater than ~99.999997% are treated as 1.0
    float fRate = rate.divide(oneHundred).floatValue();
    if (fRate <= 0.0f || fRate >= 1.0f) {
      return null;
    }
    return isRepeatable
        ? SqlSampleSpec.createTableSample(isBernoulli, fRate, repeatableSeed)
        : SqlSampleSpec.createTableSample(isBernoulli, fRate);
  }

  /**
   * Return if this OP represents a SQL default value.
   *
   * @param op The Sql operator.
   * @return Is this SqlStdOperatorTable.DEFAULT
   */
  public static boolean isDefaultCall(SqlOperator op) {
    return op == SqlStdOperatorTable.DEFAULT;
  }

  /**
   * Return if this SqlNode represents a SQL default value.
   *
   * @param node The SqlNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(SqlNode node) {
    if (node instanceof SqlCall) {
      return isDefaultCall(((SqlCall) node).getOperator());
    }
    return false;
  }

  /**
   * Return if this RexNode represents a SQL default value.
   *
   * @param node The RexNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(RexNode node) {
    if (node instanceof RexCall) {
      return isDefaultCall(((RexCall) node).getOperator());
    }
    return false;
  }
}
